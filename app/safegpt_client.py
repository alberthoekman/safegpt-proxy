import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger("proxy.safegpt")

# SafeGPT fetches every scheme URL in a prompt and inlines the page: "https://pypi.org/simple"
# alone turned a 110 KB request into 12.8 MB and a 500 (string_above_max_length). A word
# joiner after the colon stops the fetch; the model still reads (and usually writes) the URL
# normally, and restore_urls() removes any joiner it copies into its reply.
WORD_JOINER = "⁠"
_URL_SCHEME_RE = re.compile(r"\b([a-z][a-z0-9+.-]*):(?=//)", re.IGNORECASE)


def defang_urls(text: str) -> str:
    return _URL_SCHEME_RE.sub(lambda m: f"{m.group(1)}:{WORD_JOINER}", text or "")


def restore_urls(text: str) -> str:
    return (text or "").replace(WORD_JOINER, "")


# Every failed SafeGPT call is saved here exactly as sent, so it can be replayed with
# `python -m scripts.diagnose_safegpt --payload <file>` instead of reconstructed by hand.
FAILED_REQUEST_DIR = Path("logs/failed")


def dump_failed_request(url: str, payload: Dict[str, Any], resp: httpx.Response) -> Optional[Path]:
    # ".../v1/Message/Create/{conversationId}" -> "Message-Create"
    endpoint = "-".join(url.split("/v1/", 1)[-1].split("/")[:2])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    path = FAILED_REQUEST_DIR / f"{stamp}-{endpoint}.json"
    try:
        FAILED_REQUEST_DIR.mkdir(parents=True, exist_ok=True)
        record = {"url": url, "status": resp.status_code, "payload": payload, "response": resp.text}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save failed SafeGPT request: %s", exc)
        return None
    logger.warning("SafeGPT returned %d; request saved to %s", resp.status_code, path)
    return path


class SafeGPTClient:
    """Thin async HTTP client for SafeGPT API calls."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(timeout=None)

    async def close(self):
        await self.client.aclose()

    def headers(self, accept: str = "application/json") -> Dict[str, str]:
        # Only Message/Create streams; Execute and Conversation/Create return plain JSON.
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": accept,
        }

    async def models(self):
        return None

    async def _post(self, url: str, payload: Dict[str, Any], accept: str = "application/json") -> httpx.Response:
        resp = await self.client.post(url, headers=self.headers(accept), json=payload)
        if resp.is_error:
            dump_failed_request(url, payload, resp)
        resp.raise_for_status()
        return resp

    async def execute(self, system_message: str, prompt: str, model: str) -> str:
        url = f"{self.base_url}/v1/Message/Execute"
        payload = {"systemMessage": defang_urls(system_message), "prompt": defang_urls(prompt)}
        # SafeGPT's string_above_max_length error reports the size *after* it inlines fetched
        # URLs; logging what actually leaves the proxy makes such a mismatch obvious.
        logger.info("SafeGPT Execute outgoing payload: %d chars (system=%d, prompt=%d)",
                    len(json.dumps(payload, ensure_ascii=False)), len(system_message), len(prompt))
        resp = await self._post(url, payload)
        try:
            data = resp.json()
        except ValueError:
            # SafeGPT occasionally answers 200 with a body that is not JSON; treat as empty.
            logger.warning("SafeGPT Execute returned a non-JSON body: %r", resp.text[:300])
            return ""
        return restore_urls(data.get("content") or "")

    async def create_conversation(
        self,
        model: str,
        prompt: str,
        auto_tools: bool = False,
        chat_app_ids: Optional[List[str]] = None,
        conversation_type: int = 0,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/Conversation/Create"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": defang_urls(prompt),
            "autoTools": auto_tools,
            "conversationType": conversation_type,
        }
        if chat_app_ids is not None:
            payload["chatAppIds"] = chat_app_ids
        resp = await self._post(url, payload)
        return resp.json()

    async def create_message_stream(self, conversation_id: str, prompt: str, model: str) -> httpx.Response:
        url = f"{self.base_url}/v1/Message/Create/{conversation_id}"
        payload = {"prompt": defang_urls(prompt)}
        return await self._post(url, payload, accept="application/json, text/event-stream")

async def iter_safegpt_sse_lines(resp: httpx.Response):
    buffer = ""
    async for chunk in resp.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event_name = ""
            data_value = ""
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_name = line[len("event: "):].strip()
                elif line.startswith("data: "):
                    data_value = line[len("data: "):].strip()
            if event_name or data_value:
                yield event_name, data_value
    if buffer.strip():
        event_name = ""
        data_value = ""
        for line in buffer.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_value = line[len("data: "):].strip()
        if event_name or data_value:
            yield event_name, data_value

def parse_safegpt_data(data_value: str) -> str:
    raw = data_value.strip()
    if not raw:
        return ""
    try:
        val = json.loads(raw)
        return restore_urls(val if isinstance(val, str) else json.dumps(val, ensure_ascii=False))
    except Exception:
        return restore_urls(raw.strip('"'))
