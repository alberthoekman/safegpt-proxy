import argparse
import asyncio
import json

import httpx
import pytest

from app import safegpt_client
from app.safegpt_client import SafeGPTClient
from scripts import diagnose_safegpt


def make_client(handler) -> SafeGPTClient:
    client = SafeGPTClient("https://safegpt.test", "tok")
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(safegpt_client, "FAILED_REQUEST_DIR", tmp_path)
    return tmp_path


def test_failed_request_is_saved_exactly_as_sent(dump_dir):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(500, text="boom")

    client = make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.execute("sys", "see https://pypi.org/simple", "m"))

    files = list(dump_dir.glob("*-Message-Execute.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["url"] == "https://safegpt.test/v1/Message/Execute"
    assert record["status"] == 500 and record["response"] == "boom"
    assert record["payload"] == sent[0]  # the defanged payload that actually went out
    assert "://" not in record["payload"]["prompt"]


def test_successful_request_is_not_saved(dump_dir):
    client = make_client(lambda request: httpx.Response(200, json={"content": "pong"}))
    assert asyncio.run(client.execute("sys", "ping", "m")) == "pong"
    assert not list(dump_dir.iterdir())


def test_message_create_dump_drops_conversation_id(dump_dir):
    client = make_client(lambda request: httpx.Response(400, text="bad"))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.create_message_stream("conv-123", "hi", "m"))
    assert [p.name.split("-", 1)[1] for p in dump_dir.iterdir()] == ["Message-Create.json"]


def test_diagnose_replays_saved_payload_verbatim(dump_dir, tmp_path):
    payload = {"systemMessage": "sys", "prompt": "exact bytes"}
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"url": "https://safegpt.test/v1/Message/Execute", "status": 500, "payload": payload, "response": ""}), encoding="utf-8")
    received = []

    def handler(request):
        received.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"content": "ok"})

    client = make_client(handler)
    args = argparse.Namespace(payload=str(saved), system="", prompt="")
    asyncio.run(diagnose_safegpt.send(client, args, "m"))
    assert received == [("https://safegpt.test/v1/Message/Execute", payload)]
    assert list(dump_dir.iterdir()) == [saved]  # replay does not create another dump
