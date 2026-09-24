import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger("proxy.safegpt")

class SafeGPTClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(timeout=None)

    async def close(self):
        await self.client.aclose()

    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    async def models(self):
        return None

    async def execute(self, system_message: str, prompt: str, model: str) -> str:
        url = f"{self.base_url}/v1/Message/Execute"
        payload = {"systemMessage": system_message, "prompt": prompt}
        resp = await self.client.post(url, headers=self.headers(), json=payload)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            # SafeGPT occasionally answers 200 with a body that is not JSON; treat as empty.
            logger.warning("SafeGPT Execute returned a non-JSON body: %r", resp.text[:300])
            return ""
        return data.get("content") or ""

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
            "prompt": prompt,
            "autoTools": auto_tools,
            "conversationType": conversation_type,
        }
        if chat_app_ids is not None:
            payload["chatAppIds"] = chat_app_ids
        resp = await self.client.post(url, headers=self.headers(), json=payload)
        resp.raise_for_status()
        return resp.json()

    async def create_message_stream(self, conversation_id: str, prompt: str, model: str) -> httpx.Response:
        url = f"{self.base_url}/v1/Message/Create/{conversation_id}"
        payload = {"prompt": prompt}
        resp = await self.client.post(url, headers=self.headers(), json=payload)
        resp.raise_for_status()
        return resp

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
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    except Exception:
        return raw.strip('"')
