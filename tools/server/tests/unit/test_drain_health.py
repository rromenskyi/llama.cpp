import signal
import subprocess
import threading
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def start_stream(n_predict: int):
    # run a streamed completion in a thread, return once the first chunk arrived
    result = {"chunks": [], "error": None, "started": threading.Event()}

    def run():
        try:
            for chunk in server.make_stream_request("POST", "/completion", data={
                "prompt": "Once upon a time",
                "n_predict": n_predict,
                "ignore_eos": True,
                "stream": True,
            }):
                result["chunks"].append(chunk)
                result["started"].set()
        except Exception as e:
            result["error"] = e
        finally:
            result["started"].set()

    thread = threading.Thread(target=run)
    thread.start()
    assert result["started"].wait(timeout=30)
    assert result["error"] is None
    return thread, result


def wait_exit(timeout: float) -> int | None:
    assert server.process is not None
    try:
        return server.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def get_health(path: str):
    headers = {"Authorization": f"Bearer {server.api_key}"} if server.api_key else None
    return server.make_request("GET", path, headers=headers)


def test_no_flags_unchanged():
    global server
    server.start()
    for path in ["/health", "/health/ready", "/health/live"]:
        res = get_health(path)
        assert res.status_code == 200
        assert res.body["status"] == "ok"
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)
    assert wait_exit(5) == 0


def test_health_ready_live_need_api_key():
    global server
    server.api_key = "test-key"
    server.start()
    assert server.make_request("GET", "/health").status_code == 200
    for path in ["/health/ready", "/health/live"]:
        assert server.make_request("GET", path).status_code == 401
        assert get_health(path).status_code == 200


def test_sigterm_waits_for_stream(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_DECODE_SLEEP_MS", "50")
    server.drain_propagation = 1
    server.start()
    thread, result = start_stream(40)  # about 2 s of decode
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)
    thread.join(timeout=30)
    assert result["error"] is None
    last = result["chunks"][-1]
    assert last["stop"] is True
    assert last["tokens_predicted"] == 40
    t_end = time.time()
    assert wait_exit(5) == 0
    assert time.time() - t_end < 3


def test_second_signal_exits(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_DECODE_SLEEP_MS", "100")
    server.drain_propagation = 30
    server.start()
    thread, _ = start_stream(200)  # about 20 s of decode
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    assert server.process.poll() is None  # draining
    server.process.send_signal(signal.SIGTERM)
    assert wait_exit(3) is not None
    thread.join(timeout=10)


def test_ready_503_while_draining(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_DECODE_SLEEP_MS", "50")
    server.drain_propagation = 2
    server.start()
    assert get_health("/health/ready").status_code == 200

    thread, result = start_stream(100)  # about 5 s of decode
    assert server.process is not None
    server.process.send_signal(signal.SIGTERM)
    t_sigterm = time.time()
    time.sleep(0.2)

    res = get_health("/health")
    assert res.status_code == 503
    res = get_health("/health/ready")
    assert res.status_code == 503
    assert res.body["reason"] == "draining"
    assert get_health("/health/live").status_code == 200

    # new requests are still accepted while the endpoint removal propagates
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 2})
    assert res.status_code == 200

    time.sleep(max(0, t_sigterm + 2.5 - time.time()))
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 2})
    assert res.status_code == 503
    assert res.body["error"]["type"] == "unavailable_error"
    assert res.body["error"]["message"] == "draining"

    thread.join(timeout=30)
    assert result["error"] is None
    assert result["chunks"][-1]["stop"] is True
    assert wait_exit(5) == 0


def test_stall_ready_and_live(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_DECODE_SLEEP_MS", "4000")
    server.stall_ready = 1
    server.stall_live = 2
    server.start()
    assert get_health("/health/ready").status_code == 200
    assert get_health("/health/live").status_code == 200

    def run():
        server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 1})

    thread = threading.Thread(target=run)
    t_start = time.time()
    thread.start()

    time.sleep(1.6)
    res = get_health("/health/ready")
    assert res.status_code == 503
    assert res.body["reason"] == "stalled"
    assert get_health("/health/live").status_code == 200

    time.sleep(max(0, t_start + 2.8 - time.time()))
    res = get_health("/health/live")
    assert res.status_code == 503
    assert res.body["reason"] == "stalled"

    thread.join(timeout=30)
    assert get_health("/health/ready").status_code == 200
    assert get_health("/health/live").status_code == 200


def test_ready_no_flap_during_slow_prefill(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_SERVER_TEST_DECODE_SLEEP_MS", "300")
    server.stall_ready = 1
    server.start()

    tok = server.make_request("POST", "/tokenize", data={"content": "Hello"}).body["tokens"][-1]
    result = {}

    def run():
        # 200 tokens in batches of 32: 7 decodes of 300 ms, longer than --stall-ready
        result["res"] = server.make_request("POST", "/completion", data={"prompt": [tok] * 200, "n_predict": 1})

    thread = threading.Thread(target=run)
    thread.start()
    codes = []
    while thread.is_alive():
        codes.append(get_health("/health/ready").status_code)
        time.sleep(0.05)
    thread.join()
    assert result["res"].status_code == 200
    assert result["res"].body["timings"]["prompt_n"] >= 192
    assert len(codes) > 20
    assert all(code == 200 for code in codes)
