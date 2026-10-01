import asyncio
import json

import httpx
import pytest

from app.safegpt_client import (
    BREAKER_BASE_COOLDOWN_SECONDS,
    BREAKER_MAX_COOLDOWN_SECONDS,
    SafeGPTClient,
    SafeGPTStuckExecuteError,
)

STUCK_BODY = json.dumps({
    "success": False,
    "statusCode": 500,
    "message": (
        "HTTP 400 (invalid_request_error: string_above_max_length)\n"
        "Parameter: input[0].content[1].text\n\n"
        "Invalid 'input[0].content[1].text': string too long. Expected a string with "
        "maximum length 10485760, but got a string with length 12912575 instead."
    ),
})


def _response(status_code: int, body: str) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test/v1/Message/Execute")
    return httpx.Response(status_code, content=body.encode(), request=request)


def stuck_response() -> httpx.Response:
    return _response(500, STUCK_BODY)


def ok_response(content: str = "pong") -> httpx.Response:
    return _response(200, json.dumps({"content": content}))


class FakeHTTPXClient:
    """Stand-in for httpx.AsyncClient: pops canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.post_calls = 0

    async def post(self, url, headers=None, json=None):
        self.post_calls += 1
        return self._responses.pop(0)

    async def aclose(self):
        pass


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def wire_client(client: SafeGPTClient, monkeypatch, batches):
    """`batches`: list of response-lists. SafeGPTClient no longer holds a persistent
    client — every individual HTTP attempt calls `_new_client()` fresh (see
    app/safegpt_client.py), so each entry here maps 1:1 to one attempt in call order."""
    remaining = list(batches)
    fakes = []

    def _new_client():
        fake = FakeHTTPXClient(remaining.pop(0))
        fakes.append(fake)
        return fake

    client._new_client = _new_client
    return fakes


@pytest.fixture
def clock(monkeypatch):
    fake_clock = FakeClock()
    monkeypatch.setattr("app.safegpt_client.time.monotonic", fake_clock)
    return fake_clock


def run(coro):
    return asyncio.run(coro)


def test_stuck_signature_self_heals_on_retry_success(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    fakes = wire_client(client, monkeypatch, [
        [stuck_response()],       # attempt 1: fails
        [ok_response("healed")],  # attempt 2, fresh client: succeeds
    ])
    content = run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert content == "healed"
    assert client.execute_breaker_status() == "closed"
    assert fakes[0].post_calls == 1
    assert fakes[1].post_calls == 1


def test_stuck_signature_opens_breaker_after_exhausting_self_heal_attempts(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    # SELF_HEAL_MAX_ATTEMPTS = 2: both attempts fail identically, exhausting self-heal.
    wire_client(client, monkeypatch, [
        [stuck_response()],
        [stuck_response()],
    ])
    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert client.execute_breaker_status().startswith("open")


def test_open_breaker_rejects_without_new_http_call(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    fakes = wire_client(client, monkeypatch, [
        [stuck_response()],
        [stuck_response()],
    ])
    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    total_calls_before = sum(f.post_calls for f in fakes)

    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))

    total_calls_after = sum(f.post_calls for f in fakes)
    assert total_calls_after == total_calls_before  # no HTTP call made while open


def test_probe_after_cooldown_succeeds_and_closes_breaker(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    wire_client(client, monkeypatch, [
        [stuck_response()],          # 1st execute() call, attempt 1: fails
        [stuck_response()],          # 1st execute() call, attempt 2: fails -> breaker opens
        [stuck_response()],          # 2nd execute() call (the probe), attempt 1: fails
        [ok_response("recovered")],  # probe, attempt 2: succeeds -> breaker closes
    ])
    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert client.execute_breaker_status().startswith("open")

    clock.advance(BREAKER_BASE_COOLDOWN_SECONDS + 1)
    content = run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert content == "recovered"
    assert client.execute_breaker_status() == "closed"


def test_probe_failure_doubles_cooldown_up_to_cap(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    wire_client(client, monkeypatch, [
        [stuck_response()],  # 1st execute() call, attempt 1: fails
        [stuck_response()],  # 1st execute() call, attempt 2: fails -> breaker opens (base cooldown)
        [stuck_response()],  # 2nd execute() call (the probe), attempt 1: fails
        [stuck_response()],  # probe, attempt 2: fails -> breaker reopens, cooldown doubles
    ])
    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert client._breaker_cooldown == BREAKER_BASE_COOLDOWN_SECONDS

    clock.advance(BREAKER_BASE_COOLDOWN_SECONDS + 1)
    with pytest.raises(SafeGPTStuckExecuteError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert client._breaker_cooldown == min(BREAKER_BASE_COOLDOWN_SECONDS * 2, BREAKER_MAX_COOLDOWN_SECONDS)


def test_unrelated_error_does_not_open_breaker_when_closed(clock, monkeypatch):
    client = SafeGPTClient("https://example.test", "tok")
    wire_client(client, monkeypatch, [
        [_response(401, json.dumps({"message": "unauthorized"}))],
    ])
    with pytest.raises(httpx.HTTPStatusError):
        run(client.execute(system_message="sys", prompt="hi", model="m"))
    assert client.execute_breaker_status() == "closed"
