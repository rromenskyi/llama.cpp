import glob
import hashlib
import http.server
import os
import re
import threading
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


# the files served over HTTP, taken from the test model cache that ServerPreset.load_all() fills
FILES: dict[str, str] = {}
SHA: dict[str, str] = {}


def find_model(repo: str, name: str) -> str:
    found = glob.glob(f"tmp/models--ggml-org--{repo}/snapshots/*/{name}")
    assert found, f"{name} is not in the test model cache"
    return found[0]


def load_files():
    if FILES:
        return
    FILES["/model"] = find_model("test-model-stories260K", "stories260K-f32.gguf")
    FILES["/gemma"] = find_model("tinygemma3-GGUF", "tinygemma3-Q8_0.gguf")
    FILES["/mmproj"] = find_model("tinygemma3-GGUF", "mmproj-tinygemma3.gguf")
    for path, f in FILES.items():
        SHA[path] = hashlib.sha256(open(f, "rb").read()).hexdigest()


class FileServer:
    # serves FILES over HTTP in 64 KiB chunks with a short pause, so concurrent downloads overlap,
    # counts the GETs of each path and runs on_get(path) before the body is sent
    def __init__(self):
        self.gets: dict[str, int] = {}
        self.on_get = None
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                outer.gets[path] = outer.gets.get(path, 0) + 1
                if path not in FILES:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if outer.on_get:
                    outer.on_get(path)
                data = open(FILES[path], "rb").read()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                step = 65536 if len(data) < 4 << 20 else 4 << 20
                for i in range(0, len(data), step):
                    self.wfile.write(data[i:i + step])
                    time.sleep(0.05)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}?token=secret"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


files: FileServer
cache: str


def make_server(port_offset: int = 0) -> ServerProcess:
    s = ServerPreset.tinyllama2()
    s.offline = False
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_url = files.url("/model")
    s.model_sha256 = SHA["/model"]
    s.model_cache = cache
    s.server_port += port_offset
    return s


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server, files, cache
    load_files()
    files = FileServer()
    cache = str(tmp_path / "models")
    server = make_server()
    yield
    files.stop()


def final(path: str = "/model") -> str:
    return os.path.join(cache, SHA[path] + ".gguf")


def listing() -> list[str]:
    return sorted(os.listdir(cache))


def cached(path: str = "/model") -> list[str]:
    # a cached model and its blake3 sidecar
    return [SHA[path] + ".gguf", SHA[path] + ".gguf.blake3"]


def sidecar(path: str = "/model") -> str:
    return final(path) + ".blake3"


def sha_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_file(path: str, data: bytes = b"garbage", age_s: float = 0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if age_s > 0:
        t = time.time() - age_s
        os.utime(path, (t, t))


def start_fails(s: ServerProcess) -> int:
    with pytest.raises(RuntimeError):
        s.start()
    assert s.process is not None
    return s.process.wait(timeout=10)


def test_download_and_hit(tmp_path):
    global server
    server.log_path = str(tmp_path / "first.log")
    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 4})
    assert res.status_code == 200
    server.stop()
    # the query string of the URL is not logged
    log = open(tmp_path / "first.log").read()
    assert f"127.0.0.1:{files.port}/model" in log
    assert "secret" not in log

    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()


def test_two_servers_one_download(tmp_path):
    servers = [make_server(i) for i in range(3)]
    errors = []

    def run(s: ServerProcess, i: int):
        s.log_path = str(tmp_path / f"server{i}.log")
        try:
            s.start()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s, i)) for i, s in enumerate(servers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    logs = "".join(open(tmp_path / f"server{i}.log").read() for i in range(3))
    assert "waiting for" in logs
    for s in servers:
        assert s.make_request("GET", "/health").status_code == 200


def test_bad_cached_file_replaced(tmp_path):
    global server
    write_file(final())
    server.log_path = str(tmp_path / "server.log")
    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    assert "mismatch" in open(tmp_path / "server.log").read()


def test_servers_start_on_one_bad_file(tmp_path):
    # the servers hash the same bad file at once and race to move it aside
    write_file(final(), os.urandom(64 << 20))
    servers = [make_server(i) for i in range(3)]
    errors = []

    def run(s: ServerProcess, i: int):
        s.log_path = str(tmp_path / f"server{i}.log")
        try:
            s.start()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=run, args=(s, i)) for i, s in enumerate(servers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    for s in servers:
        res = s.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 4})
        assert res.status_code == 200


def test_bad_file_published_during_download(tmp_path):
    global server
    # another process publishes a bad file while this one downloads
    files.on_get = lambda path: write_file(final())
    server.log_path = str(tmp_path / "server.log")
    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    assert "mismatch" in open(tmp_path / "server.log").read()


def test_wrong_sha_fails():
    global server
    server.model_sha256 = "0" * 64
    assert start_fails(server) != 0
    assert files.gets == {"/model": 1}
    assert listing() == []


def test_http_error_fails():
    global server
    server.model_url = files.url("/missing")
    assert start_fails(server) != 0
    assert listing() == []


def test_bad_args():
    global server
    server.model_sha256 = "abc"
    assert start_fails(server) != 0
    server.model_sha256 = None
    assert start_fails(server) != 0
    server.model_sha256 = SHA["/model"]
    server.model_cache_ttl = -1
    assert start_fails(server) != 0
    assert files.gets == {}


def test_ttl_too_short(tmp_path):
    # a ttl shorter than the touch interval of the other servers would evict files in use
    global server
    server.log_path = str(tmp_path / "server.log")
    server.model_cache_ttl = 3599
    assert start_fails(server) != 0
    assert files.gets == {}
    assert "--model-cache-ttl must be at least 3600 seconds" in open(server.log_path).read()


def test_read_only_cache_fails(tmp_path):
    # an error other than an existing lock fails the start at once instead of waiting forever
    global server
    if os.geteuid() == 0:
        pytest.skip("root ignores the directory mode")
    os.makedirs(cache)
    os.chmod(cache, 0o555)
    server.log_path = str(tmp_path / "server.log")
    t_start = time.time()
    try:
        assert start_fails(server) != 0
    finally:
        os.chmod(cache, 0o755)
    assert time.time() - t_start < 10
    assert files.gets == {}
    log = open(server.log_path).read()
    assert f"cannot create the lock {cache}/.{SHA['/model']}.lock/owner in {cache}: Permission denied" in log


@pytest.mark.slow
def test_stale_lock_taken_over(tmp_path):
    global server
    # a dead owner, and an owner that died before it wrote its token
    servers = [make_server(i) for i in range(2)]
    for i, s in enumerate(servers):
        s.model_cache = str(tmp_path / f"cache{i}")
        s.log_path = str(tmp_path / f"server{i}.log")
        lock = os.path.join(s.model_cache, f".{SHA['/model']}.lock")
        write_file(os.path.join(lock, "owner"), b"dead-pod\n" if i == 0 else b"")
        write_file(os.path.join(lock, "beat"), b"7\n")
    errors = []

    def run(s: ServerProcess):
        try:
            s.start(timeout_seconds=240)
        except Exception as e:
            errors.append(e)

    t_start = time.time()
    threads = [threading.Thread(target=run, args=(s,)) for s in servers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert time.time() - t_start >= 120
    assert files.gets == {"/model": 2}
    for i, s in enumerate(servers):
        assert s.model_cache is not None
        assert sorted(os.listdir(s.model_cache)) == cached()
        log = open(tmp_path / f"server{i}.log").read()
        assert "took over the stale lock" in log


def test_ttl_eviction():
    global server
    day = 24 * 3600
    other = "1" * 64
    young = "2" * 64
    orphan = "3" * 64
    write_file(os.path.join(cache, f"{other}.gguf"), age_s=8 * day)
    write_file(os.path.join(cache, f"{other}.gguf.blake3"), b"a" * 64 + b"\n", age_s=8 * day)
    write_file(os.path.join(cache, f"{young}.gguf"), age_s=6 * day)
    write_file(os.path.join(cache, f"{young}.gguf.blake3"), b"b" * 64 + b"\n", age_s=6 * day)
    write_file(os.path.join(cache, f"{orphan}.gguf.blake3"), b"c" * 64 + b"\n")
    write_file(os.path.join(cache, f".{other}.pod-a.1.blake3.tmp"), age_s=2 * 3600)
    write_file(os.path.join(cache, "notes.gguf"), age_s=8 * day)
    write_file(os.path.join(cache, f".{other}.pod-a.1.tmp"), age_s=2 * 3600)
    write_file(os.path.join(cache, f".{young}.pod-a.1.tmp"), age_s=60)
    write_file(os.path.join(cache, f".{other}.lock", "owner"), b"dead-pod\n", age_s=2 * 3600)
    os.utime(os.path.join(cache, f".{other}.lock"), (time.time() - 2 * 3600,) * 2)
    write_file(os.path.join(cache, f".{young}.lock", "owner"), b"live-pod\n", age_s=2 * 3600)
    write_file(os.path.join(cache, f".{young}.lock", "beat"), b"7\n")
    # the model this server uses is kept even when it was not touched for longer than the ttl
    write_file(final(), open(FILES["/model"], "rb").read(), age_s=30 * day)
    server.start()
    assert files.gets == {}
    assert listing() == sorted([
        f".{young}.lock",
        f".{young}.pod-a.1.tmp",
        f"{young}.gguf",
        f"{young}.gguf.blake3",
        "notes.gguf",
    ] + cached())
    # the used model is touched at start
    assert time.time() - os.stat(final()).st_mtime < 60


def test_ttl_option():
    global server
    other = "1" * 64
    write_file(os.path.join(cache, f"{other}.gguf"), age_s=7200)
    server.model_cache_ttl = 3600
    server.start()
    assert listing() == cached()


def test_mmproj():
    global server
    server.model_url = files.url("/gemma")
    server.model_sha256 = SHA["/gemma"]
    server.mmproj_url = files.url("/mmproj")
    server.mmproj_sha256 = SHA["/mmproj"]
    server.start()
    assert files.gets == {"/gemma": 1, "/mmproj": 1}
    assert listing() == sorted(cached("/gemma") + cached("/mmproj"))
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["modalities"]["vision"]


def b3_of_sidecar(path: str = "/model") -> str:
    text = open(sidecar(path), "rb").read()
    assert len(text) == 65 and text.endswith(b"\n")
    return text[:64].decode()


def test_sidecar_written_and_used(tmp_path):
    global server
    server.start()
    server.stop()
    b3 = b3_of_sidecar()
    assert all(c in "0123456789abcdef" for c in b3)

    server.log_path = str(tmp_path / "second.log")
    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    assert b3_of_sidecar() == b3
    log = open(tmp_path / "second.log").read()
    assert f"verified {final()} with blake3 in " in log
    assert "with sha256" not in log


def test_legacy_file_gets_sidecar(tmp_path):
    # a file cached before sidecars existed is checked with sha256 once, then with blake3
    global server
    write_file(final(), open(FILES["/model"], "rb").read())
    server.log_path = str(tmp_path / "first.log")
    server.start()
    server.stop()
    assert files.gets == {}
    assert listing() == cached()
    assert f"verified {final()} with sha256 in " in open(tmp_path / "first.log").read()

    server.log_path = str(tmp_path / "second.log")
    server.start()
    assert f"verified {final()} with blake3 in " in open(tmp_path / "second.log").read()


def test_corrupt_model_with_valid_sidecar(tmp_path):
    global server
    server.start()
    server.stop()
    b3 = b3_of_sidecar()
    # flip one byte in place, the size stays the same
    with open(final(), "r+b") as f:
        f.seek(os.path.getsize(final()) // 2)
        c = f.read(1)
        f.seek(-1, os.SEEK_CUR)
        f.write(bytes([c[0] ^ 0xff]))
    assert sha_of(final()) != SHA["/model"]

    server.log_path = str(tmp_path / "server.log")
    server.start()
    assert files.gets == {"/model": 2}
    assert listing() == cached()
    assert sha_of(final()) == SHA["/model"]
    assert b3_of_sidecar() == b3
    log = open(tmp_path / "server.log").read()
    assert "checking sha256" in log
    assert "mismatch" in log


@pytest.mark.parametrize("content", [b"0" * 64 + b"\n", b"garbage\n", b""])
def test_bad_sidecar_falls_back_to_sha256(tmp_path, content: bytes):
    global server
    server.start()
    server.stop()
    b3 = b3_of_sidecar()
    write_file(sidecar(), content)

    server.log_path = str(tmp_path / "server.log")
    server.start()
    assert files.gets == {"/model": 1}
    assert listing() == cached()
    # the sidecar is written again from the file that hashed right with sha256
    assert b3_of_sidecar() == b3
    log = open(tmp_path / "server.log").read()
    assert "checking sha256" in log
    assert f"verified {final()} with sha256 in " in log


def test_loads_from_private_link(tmp_path):
    # the verified file is loaded through a hard link private to the process, removed once the model is loaded
    global server
    server.log_path = str(tmp_path / "server.log")
    server.start()
    log = open(server.log_path).read()
    link = re.search(r"loading the verified model from '\S*/\." + SHA["/model"] + r"\.\S+\.load'", log)
    assert link is not None, "the model was not loaded from a private link"
    assert listing() == cached()
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 4})
    assert res.status_code == 200


def test_unreadable_sidecar_left_alone(tmp_path):
    global server
    if os.geteuid() == 0:
        pytest.skip("root ignores the file mode")
    write_file(final(), open(FILES["/model"], "rb").read())
    write_file(sidecar(), b"0" * 64 + b"\n")
    os.chmod(sidecar(), 0)
    server.log_path = str(tmp_path / "server.log")
    try:
        server.start()
    finally:
        os.chmod(sidecar(), 0o644)
    assert files.gets == {}
    assert open(sidecar(), "rb").read() == b"0" * 64 + b"\n"
    log = open(server.log_path).read()
    assert f"cannot read {sidecar()}: Permission denied, checking sha256" in log
    assert f"verified {final()} with sha256 in " in log
