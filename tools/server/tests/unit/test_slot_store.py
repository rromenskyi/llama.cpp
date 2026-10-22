import os
import signal
import stat
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
    assert res.body == {"fingerprint": FP, "instance": "pod-a"}

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
