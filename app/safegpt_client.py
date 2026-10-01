import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger("proxy.safegpt")

# SafeGPT's own error text for its "string_above_max_length" rejection, e.g.:
# "...Expected a string with maximum length 10485760, but got a string with length 12912575..."
_MAX_LENGTH_ERROR_RE = re.compile(r"maximum length (\d+), but got a string with length (\d+)")

# Total attempts (including the first) before declaring the endpoint stuck. A later,
# more thorough live test (see claude.md's "Message/Execute" note, 2026-09-29 update)
# showed three sequential, non-identical, fresh-client attempts all fail identically at
# a size well below anything that previously succeeded — evidence this is a real,
# currently-active upstream constraint rather than a random per-attempt hiccup. Extra
# retries against that don't change the outcome, they just spend more calls, so this is
# kept low: one retry absorbs a genuinely transient blip without hammering a wall.
SELF_HEAL_MAX_ATTEMPTS = 2

# Cooldown after the breaker opens, and the cap on exponential backoff after repeated
# probe failures. Kept short (rather than the original 60s/600s) for the same reason as
# SELF_HEAL_MAX_ATTEMPTS above: without evidence of a sustained outage, blocking every
# Cline request for a long stretch after a short unlucky streak does more harm than good.
BREAKER_BASE_COOLDOWN_SECONDS = 10
BREAKER_MAX_COOLDOWN_SECONDS = 120


class SafeGPTStuckExecuteError(RuntimeError):
    """Message/Execute is in the known stuck-connection state (see claude.md's
    "Message/Execute" known-issue note): SafeGPT rejects the request with a fixed,
    fabricated oversized-request error regardless of what we actually send. This is not
    caused by our request size — retrying with a smaller prompt will not help.
    """

    def __init__(self, message: str, reported_size: int, actual_size: int, raw_message: str):
        super().__init__(message)
        self.reported_size = reported_size
        self.actual_size = actual_size
        self.raw_message = raw_message


def _match_stuck_signature(resp_text: str, payload_size: int) -> Optional[re.Match]:
    match = _MAX_LENGTH_ERROR_RE.search(resp_text)
    if not match:
        return None
    reported_size = int(match.group(2))
    # A mismatch this large means SafeGPT isn't measuring the request we just sent.
    return match if reported_size > payload_size * 2 else None


class SafeGPTClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        # No shared/persistent httpx client: every call opens one via _new_client() and
        # closes it when done (see _new_client()'s docstring for why). A prior version
        # kept one long-lived client on self and swapped it during self-heal retries with
        # no lock guarding it — a real race if two proxy requests called into this client
        # concurrently. Since keep-alive is already disabled below, there is no
        # performance reason to share a client instance at all.
        # Circuit breaker state, scoped to Message/Execute only — create_conversation()
        # and create_message_stream() (the normal-chat path) are unaffected by this bug.
        self._breaker_lock = asyncio.Lock()
        self._breaker_open_until: Optional[float] = None
        self._breaker_cooldown = BREAKER_BASE_COOLDOWN_SECONDS
        self._breaker_probing = False

    def _new_client(self) -> httpx.AsyncClient:
        # Message/Execute has been observed failing with a fixed, request-independent
        # "oversized" error when a connection is reused (keep-alive) over the life of a
        # long-running process, while a brand-new connection (e.g. from Postman) always
        # succeeds. Disable connection reuse so every call gets an isolated one.
        return httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_keepalive_connections=0))

    async def close(self):
        # No persistent client to close; kept so main.py's shutdown hook stays valid.
        pass

    def headers(self, accept: str = "application/json") -> Dict[str, str]:
        # Only request text/event-stream where a response can actually stream
        # (create_message_stream). Sending it to Message/Execute — which always
        # returns a plain {"content": "..."} body and never streams — appears to route
        # SafeGPT into a different, broken response-framing path: that's the real cause
        # of the fixed, fabricated "oversized request" error (see claude.md's
        # "Message/Execute" known-issue note), not connection reuse or request size.
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": accept,
            "Connection": "close",
        }

    async def models(self):
        return None

    def execute_breaker_status(self) -> str:
        if self._breaker_open_until is None:
            return "closed"
        remaining = max(self._breaker_open_until - time.monotonic(), 0)
        return f"open (probe in {remaining:.0f}s)" if remaining > 0 else "open (probing)"

    async def _breaker_gate(self) -> None:
        """Fail fast if the breaker is open and not yet eligible for a probe; otherwise
        mark this call as the probe if it's the first one eligible after cooldown."""
        if self._breaker_open_until is None:
            return
        async with self._breaker_lock:
            if self._breaker_open_until is None:
                return
            now = time.monotonic()
            if now < self._breaker_open_until or self._breaker_probing:
                remaining = max(self._breaker_open_until - now, 0)
                raise SafeGPTStuckExecuteError(
                    "SafeGPT Message/Execute breaker is open (known stuck-connection state "
                    f"seen recently); next probe in {remaining:.0f}s. This is not a local "
                    "sizing issue — retrying now will not help.",
                    reported_size=0, actual_size=0, raw_message="breaker_open",
                )
            self._breaker_probing = True

    def _breaker_close(self) -> None:
        self._breaker_open_until = None
        self._breaker_cooldown = BREAKER_BASE_COOLDOWN_SECONDS
        self._breaker_probing = False

    def _breaker_open(self) -> None:
        # Exponential backoff only when a probe itself failed; a fresh (non-probe)
        # detection always starts from the base cooldown.
        if self._breaker_probing:
            self._breaker_cooldown = min(self._breaker_cooldown * 2, BREAKER_MAX_COOLDOWN_SECONDS)
        else:
            self._breaker_cooldown = BREAKER_BASE_COOLDOWN_SECONDS
        self._breaker_open_until = time.monotonic() + self._breaker_cooldown
        self._breaker_probing = False
        logger.error(
            "SafeGPT Message/Execute breaker OPEN for %ds (stuck-connection signature confirmed "
            "after a fresh-connection retry also failed)", self._breaker_cooldown,
        )

    async def _execute_once(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        client = self._new_client()
        try:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                # SafeGPT occasionally answers 200 with a body that is not JSON; treat as empty.
                logger.warning("SafeGPT Execute returned a non-JSON body: %r", resp.text[:300])
                return ""
            return data.get("content") or ""
        finally:
            await client.aclose()

    async def _execute_with_self_heal(self, url: str, payload: Dict[str, Any], payload_size: int) -> str:
        last_match: Optional[re.Match] = None
        last_exc: Optional[httpx.HTTPStatusError] = None
        for attempt in range(1, SELF_HEAL_MAX_ATTEMPTS + 1):
            try:
                content = await self._execute_once(url, payload, self.headers())
            except httpx.HTTPStatusError as exc:
                match = _match_stuck_signature(exc.response.text, payload_size)
                if match is None:
                    raise
                last_match, last_exc = match, exc
                if attempt < SELF_HEAL_MAX_ATTEMPTS:
                    logger.warning(
                        "SafeGPT Execute hit the known stuck-connection signature (reported %d "
                        "chars vs our %d bytes) on attempt %d/%d; retrying on a fresh connection.",
                        int(match.group(2)), payload_size, attempt, SELF_HEAL_MAX_ATTEMPTS,
                    )
                continue
            if attempt > 1:
                logger.info("SafeGPT Execute recovered on attempt %d/%d.", attempt, SELF_HEAL_MAX_ATTEMPTS)
            return content
        raise SafeGPTStuckExecuteError(
            f"SafeGPT Message/Execute hit the same fabricated oversized-request error on all "
            f"{SELF_HEAL_MAX_ATTEMPTS} attempts (each on a fresh connection). Opening the "
            "breaker briefly to avoid hammering this endpoint further; it will self-probe soon.",
            reported_size=int(last_match.group(2)), actual_size=payload_size,
            raw_message=last_exc.response.text[:2000],
        ) from last_exc

    async def execute(self, system_message: str, prompt: str, model: str) -> str:
        await self._breaker_gate()
        is_probe = self._breaker_probing
        url = f"{self.base_url}/v1/Message/Execute"
        payload = {"systemMessage": system_message, "prompt": prompt}
        # Logged here (not just before this call in router.py) so the actual outgoing byte
        # size is on record even if something between prompt-building and this call inflates
        # it beyond what router.py's own character counts show.
        payload_size = len(json.dumps(payload, ensure_ascii=False))
        logger.info(
            "SafeGPT Execute outgoing payload: %d bytes (system=%d chars, prompt=%d chars)%s",
            payload_size, len(system_message), len(prompt), " [breaker probe]" if is_probe else "",
        )
        try:
            content = await self._execute_with_self_heal(url, payload, payload_size)
        except SafeGPTStuckExecuteError:
            # Confirmed twice (first failure + fresh-connection retry): always open the breaker.
            self._breaker_open()
            raise
        except Exception:
            # Unrelated failure while probing: reopen (exponential backoff) without claiming
            # it's the known stuck-signature bug.
            if is_probe:
                self._breaker_open()
            raise
        if is_probe:
            self._breaker_close()
            logger.warning("SafeGPT Execute breaker closed — upstream recovered.")
        return content

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
        client = self._new_client()
        try:
            resp = await client.post(url, headers=self.headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
        finally:
            await client.aclose()

    async def create_message_stream(self, conversation_id: str, prompt: str, model: str) -> httpx.Response:
        # `client.post(...)` (as opposed to `client.stream(...)`) always reads the whole
        # response body into memory before returning, so closing the client right after is
        # safe — callers' later `resp.aiter_text()`/`resp.aread()` just replay that buffer.
        url = f"{self.base_url}/v1/Message/Create/{conversation_id}"
        payload = {"prompt": prompt}
        client = self._new_client()
        try:
            resp = await client.post(
                url, headers=self.headers(accept="application/json, text/event-stream"), json=payload
            )
            resp.raise_for_status()
            return resp
        finally:
            await client.aclose()

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
