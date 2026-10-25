import os
import signal
import re
import stat
import struct
import subprocess
import threading
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()

FP = "fp1"


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.slot_store = str(tmp_path / "store")
    server.slot_fingerprint = FP
    # exit at once on SIGTERM unless a test needs the drain handshake
    server.router_lost = 0
    server.drain_idle_exit = 0


def store_path(*parts: str) -> str:
    assert server.slot_store is not None
    return os.path.join(server.slot_store, *parts)


def wait_exit(timeout: float) -> int | None:
    assert server.process is not None
    try:
        return server.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def fill_slot(id_slot: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def save(id_slot: int, filename: str):
    return server.make_request("POST", f"/slots/{id_slot}?action=save", data={"filename": filename})


def write_file(path: str, age_s: float = 0, data: bytes = b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if age_s > 0:
        t = time.time() - age_s
        os.utime(path, (t, t))


def test_save_is_atomic_and_listed():
    global server
    write_file(store_path("fp0", "old.slot"), data=b"12345")
    write_file(store_path("fp0", ".old.abc.tmp"))
    write_file(store_path("fp0", "notes.txt"))
    server.slot_instance = "pod-a"
    server.start()

    # the fingerprint directory is created with mode 0770
    assert stat.S_IMODE(os.stat(store_path(FP)).st_mode) == 0o770

    res = server.make_request("GET", "/slot-store/info")
    assert res.status_code == 200
    boot_id = res.body.pop("boot_id")
    assert res.body == {"fingerprint": FP, "instance": "pod-a"}
    assert re.fullmatch(r"[0-9a-f]{16}", boot_id)
    assert server.make_request("GET", "/slot-store/info").body["boot_id"] == boot_id

    # a restarted process reports a different boot id
    server.stop()
    server.start()
    res = server.make_request("GET", "/slot-store/info")
    assert res.body["boot_id"] != boot_id

    fill_slot(1)
    t_save = int(time.time())
    res = save(1, "k1.slot")
    assert res.status_code == 200
    assert res.body["n_saved"] == 84
    assert res.body["n_written"] == os.path.getsize(store_path(FP, "k1.slot"))
    # saved under the final name, no temporary file left behind
    assert os.listdir(store_path(FP)) == ["k1.slot"]

    # a second save replaces the file
    res = save(1, "k1.slot")
    assert res.status_code == 200
    assert os.listdir(store_path(FP)) == ["k1.slot"]

    res = server.make_request("GET", "/slot-store")
    assert res.status_code == 200
    assert res.body["fingerprint"] == FP
    files = sorted(res.body["files"], key=lambda f: f["key"])
    assert [(f["fingerprint"], f["key"]) for f in files] == [(FP, "k1"), ("fp0", "old")]
    assert files[0]["size"] == os.path.getsize(store_path(FP, "k1.slot"))
    assert abs(files[0]["mtime"] - t_save) <= 2
    assert files[1]["size"] == 5


def test_store_needs_api_key():
    global server
    server.api_key = "test-key"
    server.start()
    for path in ["/slot-store", "/slot-store/info", "/slot-store/drained"]:
        assert server.make_request("GET", path).status_code == 401
    res = server.make_request("POST", "/slot-store/evict", data={"files": []})
    assert res.status_code == 401
    assert server.make_request("POST", "/slot-store/drained").status_code == 401
    assert server.make_request("GET", "/slot-store/drained", headers={"Authorization": "Bearer test-key"}).body["drained"] is False
    res = server.make_request("GET", "/slot-store/info", headers={"Authorization": "Bearer test-key"})
    assert res.status_code == 200


def test_no_store_no_endpoints():
    global server
    server.slot_store = None
    server.slot_fingerprint = None
    server.start()
    assert server.make_request("GET", "/slot-store").status_code == 404


@pytest.mark.parametrize("case", ["no_fingerprint", "no_store", "with_save_path", "bad_fingerprint"])
def test_bad_flags_refuse_start(tmp_path, case):
    global server
    if case == "no_fingerprint":
        server.slot_fingerprint = None
    elif case == "no_store":
        server.slot_store = None
    elif case == "with_save_path":
        server.slot_save_path = str(tmp_path)
    else:
        server.slot_fingerprint = "../fp"
    with pytest.raises(RuntimeError):
        server.start()


def test_restore_from_store():
    global server
    server.start()
    # both directories are created by the server, with mode 0770
    assert stat.S_IMODE(os.stat(store_path()).st_mode) == 0o770
    assert stat.S_IMODE(os.stat(store_path(FP)).st_mode) == 0o770
    fill_slot(1)
    assert save(1, "k1.slot").status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "k1.slot"})
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == os.path.getsize(store_path(FP, "k1.slot"))

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "missing.slot"})
    assert res.status_code != 200


def test_evict():
    global server
    write_file(store_path("fp0", "old.slot"))
    server.start()
    fill_slot(1)
    assert save(1, "k1.slot").status_code == 200
    assert save(1, "k2.slot").status_code == 200

    res = server.make_request("POST", "/slot-store/evict", data={"files": [
        {"fingerprint": FP, "key": "k1"},
        {"fingerprint": "fp0", "key": "old"},
        {"fingerprint": FP, "key": "missing"},
    ]})
    assert res.status_code == 200
    assert res.body == {"removed": 2}
    assert os.listdir(store_path(FP)) == ["k2.slot"]
    assert os.listdir(store_path("fp0")) == []

    res = server.make_request("GET", "/slot-store")
    assert [(f["fingerprint"], f["key"]) for f in res.body["files"]] == [(FP, "k2")]


def test_name_validation():
    global server
    write_file(store_path("fp0", "old.slot"))
    server.start()
    fill_slot(1)
    for filename in ["../k1.slot", ".k1.slot", "k1.bin", "k1", "a/b.slot", ".slot", "k.1.slot", "k1.slot.tmp"]:
        res = save(1, filename)
        assert res.status_code == 400, filename
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": filename})
        assert res.status_code == 400, filename
    assert os.listdir(store_path(FP)) == []

    bad_bodies = [
        {},
        {"files": "x"},
        {"files": [{"fingerprint": "fp0"}]},
        {"files": [{"fingerprint": "fp0", "key": 1}]},
        {"files": [{"fingerprint": "..", "key": "old"}]},
        {"files": [{"fingerprint": "fp0", "key": "../fp0/old"}]},
        {"files": [{"fingerprint": "fp0", "key": ""}]},
        {"files": [{"fingerprint": "", "key": "old"}]},
        # one bad entry rejects the whole request, nothing is removed
        {"files": [{"fingerprint": "fp0", "key": "old"}, {"fingerprint": "fp0", "key": "a.b"}]},
    ]
    for body in bad_bodies:
        res = server.make_request("POST", "/slot-store/evict", data=body)
        assert res.status_code == 400, body
    assert os.path.exists(store_path("fp0", "old.slot"))


def test_drained_flag_and_router_seen():
    global server
    server.start()
    res = server.make_request("GET", "/slot-store/drained")
    assert res.status_code == 200
    assert res.body == {"drained": False, "router_seen_seconds": -1}
    # reading the flag is not router contact
    res = server.make_request("GET", "/slot-store/drained")
    assert res.body["router_seen_seconds"] == -1

    assert server.make_request("GET", "/slot-store/info").status_code == 200
    time.sleep(1.2)
    res = server.make_request("GET", "/slot-store/drained")
    assert res.body["router_seen_seconds"] in (1, 2)

    assert server.make_request("GET", "/slot-store").status_code == 200
    res = server.make_request("GET", "/slot-store/drained")
    assert res.body["router_seen_seconds"] == 0

    assert server.make_request("POST", "/slot-store/drained").status_code == 200
    res = server.make_request("GET", "/slot-store/drained")
    assert res.body == {"drained": True, "router_seen_seconds": 0}


def test_drain_exits_after_drained():
    global server
    server.router_lost = 300
    server.start()
    fill_slot(1)
    assert server.make_request("GET", "/slot-store/info").status_code == 200
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)

    # idle, but the router is alive and has not set the drained flag
    assert wait_exit(2) is None
    assert server.make_request("GET", "/health").status_code == 503
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 2})
    assert res.status_code == 503
    # the router can still save slots and use the store
    assert save(1, "k1.slot").status_code == 200
    assert server.make_request("GET", "/slot-store").status_code == 200
    assert wait_exit(1) is None

    assert server.make_request("POST", "/slot-store/drained").status_code == 200
    assert wait_exit(3) == 0


def test_drain_router_lost_fallback():
    global server
    server.router_lost = 1
    server.drain_idle_exit = 1
    server.start()
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)

    # router heartbeats keep the server alive
    t_end = time.time() + 3
    while time.time() < t_end:
        assert server.make_request("GET", "/slot-store/info").status_code == 200
        time.sleep(0.3)
    assert server.process.poll() is None

    # router lost and idle: exit
    t_lost = time.time()
    assert wait_exit(5) == 0
    assert time.time() - t_lost >= 0.5


def test_drain_router_never_seen():
    global server
    server.router_lost = 300
    server.drain_idle_exit = 1
    server.start()
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)
    assert wait_exit(4) == 0


def test_existing_dir_without_group_access_warns(tmp_path):
    global server
    os.makedirs(store_path(FP))
    os.chmod(store_path(FP), 0o700)
    server.log_path = str(tmp_path / "server.log")
    server.start()
    server.stop()
    with open(server.log_path) as f:
        log = f.read()
    assert f"'{store_path(FP)}' has mode 700" in log
    # an existing directory is left as is
    assert stat.S_IMODE(os.stat(store_path(FP)).st_mode) == 0o700


def test_drained_reads_do_not_delay_exit():
    global server
    server.router_lost = 300
    server.drain_idle_exit = 1
    server.start()
    stop = threading.Event()

    def poll():
        with requests.Session() as session:
            while not stop.is_set():
                try:
                    session.get(server.make_url("/slot-store/drained"), timeout=1)
                except Exception:
                    pass

    threads = [threading.Thread(target=poll) for _ in range(16)]
    for t in threads:
        t.start()
    try:
        assert server.process is not None
        server.process.send_signal(signal.SIGTERM)
        # never seen router and idle: exit, even while the flag is polled
        assert wait_exit(5) == 0
    finally:
        stop.set()
        for t in threads:
            t.join()


def test_janitor_removes_old_tmp():
    global server
    write_file(store_path("fpx", ".a.1.tmp"), age_s=7200)
    write_file(store_path("fpx", ".b.2.tmp"))
    write_file(store_path("fpx", "c.slot"), age_s=7200)
    server.start()
    t_end = time.time() + 5
    while os.path.exists(store_path("fpx", ".a.1.tmp")) and time.time() < t_end:
        time.sleep(0.1)
    assert sorted(os.listdir(store_path("fpx"))) == [".b.2.tmp", "c.slot"]


@pytest.mark.parametrize("mode,ok", [("after", True), ("before", False)])
def test_rename_error_counts_as_success(monkeypatch, mode, ok):
    # "after": the rename is done but reports an error, like a retried rename over NFS
    # "before": the rename fails and nothing is renamed
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_RENAME_ERROR", mode)
    server.start()
    fill_slot(1)
    res = save(1, "k1.slot")
    if ok:
        assert res.status_code == 200
        assert os.listdir(store_path(FP)) == ["k1.slot"]
    else:
        assert res.status_code == 500
        assert os.listdir(store_path(FP)) == []


def test_short_file_keeps_previous_slot(monkeypatch):
    # the tail of the file is lost, like a buffered write that failed at close: the old slot file stays
    global server
    write_file(store_path(FP, "k1.slot"), data=b"previous")
    monkeypatch.setenv("LLAMA_SERVER_TEST_SAVE_TRUNCATE", "1")
    server.start()
    fill_slot(1)
    res = save(1, "k1.slot")
    assert res.status_code == 500
    assert "bytes" in res.body["error"]["message"]
    assert os.listdir(store_path(FP)) == ["k1.slot"]
    assert open(store_path(FP, "k1.slot"), "rb").read() == b"previous"


TRAILER = 40


def restore(id_slot: int, filename: str, s: ServerProcess | None = None):
    s = s or server
    return s.make_request("POST", f"/slots/{id_slot}?action=restore", data={"filename": filename})


def prompt_n_germany(id_slot: int) -> int:
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": id_slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def test_save_writes_trailer():
    global server
    server.start()
    fill_slot(1)
    res = save(1, "k1.slot")
    assert res.status_code == 200
    path = store_path(FP, "k1.slot")
    data = open(path, "rb").read()
    assert data[-TRAILER:-32] == b"B3SLOT01"
    assert os.listdir(store_path(FP)) == ["k1.slot"]
    # the sizes include the trailer
    assert res.body["n_written"] == len(data)
    res = server.make_request("GET", "/slot-store")
    assert [(f["key"], f["size"]) for f in res.body["files"]] == [("k1", len(data))]


@pytest.mark.parametrize("damage", ["flip", "flip_trailer", "no_trailer"])
def test_restore_damaged_fails(damage):
    global server
    server.start()
    fill_slot(1)
    assert save(1, "k1.slot").status_code == 200
    path = store_path(FP, "k1.slot")
    data = bytearray(open(path, "rb").read())
    if damage == "flip":
        data[len(data) // 2] ^= 0xff
    elif damage == "flip_trailer":
        data[-1] ^= 0xff
    else:
        data = data[:-TRAILER]
    with open(path, "wb") as f:
        f.write(data)

    # slot 0 holds the shared prefix before the restore
    fill_slot(0)
    res = restore(0, "k1.slot")
    assert res.status_code == 400
    msg = res.body["error"]["message"]
    assert ("no checksum trailer" if damage == "no_trailer" else "checksum mismatch") in msg
    # the failed restore leaves the slot empty: the whole prompt is processed
    assert prompt_n_germany(0) > 6

    # a good file restores again
    assert save(1, "k1.slot").status_code == 200
    assert restore(0, "k1.slot").status_code == 200
    assert prompt_n_germany(0) == 6


def test_restore_uses_the_bytes_it_checked(monkeypatch):
    # the file is overwritten right after it was read: the restore must not open it again
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_RESTORE_REPLACE", "1")
    server.start()
    fill_slot(1)
    res = save(1, "k1.slot")
    assert res.status_code == 200
    n_written = res.body["n_written"]
    res = restore(0, "k1.slot")
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == n_written
    assert open(store_path(FP, "k1.slot"), "rb").read() == b"replaced"
    assert prompt_n_germany(0) == 6


def test_concurrent_saves_of_one_key():
    global server
    other = ServerPreset.tinyllama2()
    other.temperature = 0.0
    other.slot_store = server.slot_store
    other.slot_fingerprint = FP
    other.router_lost = 0
    other.drain_idle_exit = 0
    other.server_port = server.server_port + 1
    server.start()
    other.start()
    errors = []

    def run(s: ServerProcess):
        try:
            res = s.make_request("POST", "/completion", data={
                "prompt": "What is the capital of France?",
                "id_slot": 1,
                "cache_prompt": True,
            })
            assert res.status_code == 200
            for _ in range(10):
                res = s.make_request("POST", "/slots/1?action=save", data={"filename": "k1.slot"})
                assert res.status_code == 200
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s,)) for s in (server, other)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert os.listdir(store_path(FP)) == ["k1.slot"]
    for s in (server, other):
        res = restore(0, "k1.slot", s)
        assert res.status_code == 200
        assert res.body["n_restored"] == 84


def test_restore_larger_than_max_fails():
    global server
    server.start()
    fill_slot(1)
    assert save(1, "k1.slot").status_code == 200
    server.stop()
    server.slot_max_bytes = 1000
    server.start()
    res = restore(0, "k1.slot")
    assert res.status_code == 400
    assert "more than --slot-max-bytes 1000" in res.body["error"]["message"]


def test_restore_huge_file_fails_without_allocating():
    # a sparse 1 TiB file is rejected from its size, before any buffer is allocated
    global server
    server.start()
    path = store_path(FP, "huge.slot")
    with open(path, "wb") as f:
        f.truncate(1 << 40)
    res = restore(0, "huge.slot")
    assert res.status_code == 400
    assert "more than --slot-max-bytes" in res.body["error"]["message"]
    assert server.make_request("GET", "/health").status_code == 200


@pytest.mark.parametrize("sync", [False, True])
def test_save_larger_than_max_fails(monkeypatch, sync):
    # a file larger than --slot-max-bytes could not be restored: the save fails, async or not
    global server
    if sync:
        monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_SAVE_SYNC", "1")
    write_file(store_path(FP, "k1.slot"), data=b"previous")
    server.slot_max_bytes = 1000
    server.start()
    fill_slot(1)
    res = save(1, "k1.slot")
    assert res.status_code == 500
    assert "more than --slot-max-bytes 1000" in res.body["error"]["message"]
    assert os.listdir(store_path(FP)) == ["k1.slot"]
    assert open(store_path(FP, "k1.slot"), "rb").read() == b"previous"


def save_in_thread(id_slot: int, filename: str, results: list, timeout: float | None = None):
    def run():
        t0 = time.time()
        try:
            res = server.make_request("POST", f"/slots/{id_slot}?action=save", data={"filename": filename}, timeout=timeout)
            results.append((filename, res.status_code, res.body, time.time() - t0))
        except Exception as e:
            results.append((filename, None, e, time.time() - t0))
    t = threading.Thread(target=run)
    t.start()
    return t


SWA_BASE = "The quick brown fox jumps over the lazy dog. " * 20


def use_swa_draft_server():
    # a SWA model with a draft model: the file also holds the context checkpoint and the draft state appendices
    global server
    store = server.slot_store
    server = ServerPreset.tinygemma3()
    server.temperature = 0.0
    server.cache_ram = 0
    server.n_ubatch = 32
    server.model_draft_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"  # same file as the target, already in the HF cache
    server.spec_type = "draft-simple"
    server.slot_store = store
    server.slot_fingerprint = FP
    server.router_lost = 0
    server.drain_idle_exit = 0


@pytest.mark.parametrize("appendix", [False, True])
def test_async_save_matches_sync_save(monkeypatch, tmp_path, appendix):
    # two slow saves fill the writer, the third one is written synchronously from the same slot content
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_WRITE_DELAY_MS", "2000")
    if appendix:
        use_swa_draft_server()
        prompt = SWA_BASE + "The first ending of this story is a happy one."
    else:
        prompt = "What is the capital of France?"
    server.log_path = str(tmp_path / "server.log")
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "id_slot": 1, "cache_prompt": True})
    assert res.status_code == 200

    results = []
    threads = [save_in_thread(1, "a1.slot", results)]
    time.sleep(0.3)
    threads.append(save_in_thread(1, "a2.slot", results))
    time.sleep(0.3)
    t_sync = time.time()
    res = save(1, "s1.slot")
    assert res.status_code == 200
    # the synchronous save is committed after the pending ones
    assert time.time() - t_sync >= 3
    assert sorted(os.listdir(store_path(FP))) == ["a1.slot", "a2.slot", "s1.slot"]
    for t in threads:
        t.join()
    assert sorted((r[0], r[1]) for r in results) == [("a1.slot", 200), ("a2.slot", 200)]
    assert all(r[3] >= 2 for r in results)

    files = {name: open(store_path(FP, name), "rb").read() for name in ["a1.slot", "a2.slot", "s1.slot"]}
    assert files["a1.slot"] == files["s1.slot"]
    assert files["a2.slot"] == files["s1.slot"]
    if appendix:
        assert struct.pack("<II", 0x504b4353, 1) in files["s1.slot"]  # "SCKP" v1
        assert struct.pack("<II", 0x54464453, 1) in files["s1.slot"]  # "SDFT" v1
    n_saved = res.body["n_saved"]
    assert n_saved > 0
    assert res.body["n_written"] == len(files["s1.slot"])
    for r in results:
        assert r[2]["n_saved"] == n_saved
        assert r[2]["n_written"] == len(files["s1.slot"])
    assert sorted(os.listdir(store_path(FP))) == ["a1.slot", "a2.slot", "s1.slot"]

    server.stop()
    with open(server.log_path) as f:
        log = f.read()
    assert "2 slot saves pending, saving synchronously" in log
    assert "waiting for 2 pending slot save(s) before a synchronous save" in log
    assert len(re.findall(r"saved to '.*a[12]\.slot': snapshot [0-9.]+ s, write [0-9.]+ s, [0-9.]+ MiB", log)) == 2

    # the async file restores
    server.log_path = None
    server.start()
    res = restore(0, "a1.slot")
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == len(files["a1.slot"])
    if not appendix:
        assert prompt_n_germany(0) == 6


def test_sync_save_is_committed_after_pending_saves(monkeypatch):
    # two slow saves of k1 fill the writer, then the slot changes and a third save of k1 goes synchronously:
    # the file left is the last save, not an older snapshot renamed over it
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_WRITE_DELAY_MS", "1500")
    server.start()
    fill_slot(1)
    results = []
    threads = [save_in_thread(1, "k1.slot", results)]
    time.sleep(0.3)
    threads.append(save_in_thread(1, "k1.slot", results))
    time.sleep(0.3)

    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time",
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 8,
    })
    assert res.status_code == 200
    res = save(1, "k1.slot")
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    for t in threads:
        t.join()
    assert [r[1] for r in results] == [200, 200]
    assert all(r[2]["n_saved"] == 84 for r in results)
    assert n_saved != 84

    assert os.listdir(store_path(FP)) == ["k1.slot"]
    assert os.path.getsize(store_path(FP, "k1.slot")) == res.body["n_written"]
    res = restore(0, "k1.slot")
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved


def test_snapshot_appendix_failure_fails_the_save(monkeypatch):
    # the appendix cannot be written to the snapshot: the save fails, no partial file is published
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_SNAPSHOT_FAIL", "1")
    use_swa_draft_server()
    write_file(store_path(FP, "k1.slot"), data=b"previous")
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": SWA_BASE + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = save(1, "k1.slot")
    assert res.status_code == 500
    assert "incomplete context checkpoints or draft state" in res.body["error"]["message"]
    assert os.listdir(store_path(FP)) == ["k1.slot"]
    assert open(store_path(FP, "k1.slot"), "rb").read() == b"previous"
    # the server keeps working
    assert server.make_request("GET", "/health").status_code == 200


def test_slow_save_does_not_block_other_slots(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_WRITE_DELAY_MS", "3000")
    server.start()
    fill_slot(0)
    results = []
    t_save = time.time()
    t = save_in_thread(0, "k0.slot", results)
    time.sleep(0.3)

    # a completion on slot 1, and a restore into it, while the save of slot 0 is pending
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert t.is_alive()
    assert results == []
    assert time.time() - t_save < 2.5
    # the file appears only once it is complete
    assert os.listdir(store_path(FP)) == []

    t.join()
    assert results[0][1] == 200
    assert results[0][3] >= 3
    assert results[0][2]["timings"]["save_ms"] >= 3000
    assert os.listdir(store_path(FP)) == ["k0.slot"]
    assert restore(1, "k0.slot").status_code == 200
    assert prompt_n_germany(1) == 6


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_exit_waits_for_pending_save(monkeypatch, sig):
    # SIGTERM drains, SIGINT ends the loop at once: either way the pending file is written before the exit
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_SLOT_WRITE_DELAY_MS", "2000")
    server.start()
    fill_slot(1)
    results = []
    t = save_in_thread(1, "k1.slot", results)
    time.sleep(0.3)
    assert server.process is not None
    server.process.send_signal(sig)
    assert wait_exit(10) == 0
    t.join()
    if sig == signal.SIGTERM:
        # the save request is in flight, the drain waits for it
        assert results[0][1] == 200
    data = open(store_path(FP, "k1.slot"), "rb").read()
    assert data[-TRAILER:-32] == b"B3SLOT01"
    assert os.listdir(store_path(FP)) == ["k1.slot"]

    server.start()
    res = restore(0, "k1.slot")
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == len(data)
    assert prompt_n_germany(0) == 6
