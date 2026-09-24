import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import tools as toolemu
from app.models import (
    MODEL_ID,
    chat_completion_chunk,
    chat_completion_response,
    models_list,
    responses_object,
)
from app.safegpt_client import SafeGPTClient, iter_safegpt_sse_lines, parse_safegpt_data
from app.session import cleanup_sessions, get_or_create_session, load_response, store_response

router = APIRouter()
logger = logging.getLogger("proxy")

# These are injected from main.py
safegpt_client: Optional[SafeGPTClient] = None
default_model_id: str = MODEL_ID
web_search_chat_app_id: str = ""
code_interpreter_chat_app_id: str = ""

def set_dependencies(client: SafeGPTClient, model_id: str, web_id: str, code_id: str):
    global safegpt_client, default_model_id, web_search_chat_app_id, code_interpreter_chat_app_id
    safegpt_client = client
    default_model_id = model_id
    web_search_chat_app_id = web_id
    code_interpreter_chat_app_id = code_id

def normalize_path(path: str) -> str:
    if path.startswith("/v1/"):
        return path[3:]
    return path

def message_text(m: Dict[str, Any]) -> str:
    content = m.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)

def extract_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = body.get("messages") or []
    if not msgs:
        raw = body.get("input") or []
        msgs = [item for item in raw if isinstance(item, dict) and item.get("type") == "message"]
    return msgs

def classify_jetbrains_request(body: Dict[str, Any]) -> str:
    messages = extract_messages(body)
    if not messages:
        return "normal_chat"
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = message_text(m)
            break
    low = last_user.lower()
    if "suggest a most specific title" in low:
        return "orchestration_title"
    if "give subqueries that would be useful to search for in project" in low:
        return "orchestration_subqueries"
    if "determine if the following context is required" in low:
        return "orchestration_relevance"
    return "normal_chat"

def should_use_execute(body: Dict[str, Any]) -> bool:
    return classify_jetbrains_request(body) in {
        "orchestration_title",
        "orchestration_subqueries",
        "orchestration_relevance",
    }

def build_execute_prompt(body: Dict[str, Any]) -> str:
    messages = extract_messages(body)
    for m in reversed(messages):
        if m.get("role") == "user":
            return message_text(m).strip()
    return ""

def build_execute_system_message(body: Dict[str, Any]) -> str:
    messages = extract_messages(body)
    system_parts = [message_text(m) for m in messages if m.get("role") == "system"]
    return "\n\n".join(system_parts).strip() or "You are a helpful assistant."

def estimated_session_key(request: Request, body: Dict[str, Any]) -> str:
    auth = request.headers.get("authorization", "")
    model = body.get("model", default_model_id)
    return f"{auth}:{model}"

def select_chat_app_ids_for_request(body: Dict[str, Any]) -> List[str]:
    # tools enabled by default; SafeGPT decides what to use
    ids = []
    if web_search_chat_app_id:
        ids.append(web_search_chat_app_id)
    if code_interpreter_chat_app_id:
        ids.append(code_interpreter_chat_app_id)
    return ids

def sse_data(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

def sse_event(event_name: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

def sse_done() -> str:
    return "data: [DONE]\n\n"

def responses_envelope(model: str, response_id: str, status: str, output: List[Dict[str, Any]] = None, usage: Dict[str, Any] = None) -> Dict[str, Any]:
    obj = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model or MODEL_ID,
        "output": output or [],
        "parallel_tool_calls": False,
        "text": {"format": {"type": "text"}},
    }
    if usage is not None:
        obj["usage"] = usage
    return obj

async def stream_responses_events(text_chunks, model: str, include_usage: bool = True):
    """Emit OpenAI Responses-API SSE events. text_chunks is an async iterator of delta strings."""
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    seq = [0]

    def event(event_type: str, payload: Dict[str, Any]) -> str:
        payload = {"type": event_type, "sequence_number": seq[0], **payload}
        seq[0] += 1
        return sse_event(event_type, payload)

    yield event("response.created", {"response": responses_envelope(model, response_id, "in_progress")})
    yield event("response.in_progress", {"response": responses_envelope(model, response_id, "in_progress")})
    yield event("response.output_item.added", {
        "output_index": 0,
        "item": {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
    })
    yield event("response.content_part.added", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })

    text_parts = []
    async for delta_text in text_chunks:
        if not delta_text:
            continue
        text_parts.append(delta_text)
        yield event("response.output_text.delta", {
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": delta_text,
        })

    final_text = "".join(text_parts)
    yield event("response.output_text.done", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "text": final_text,
    })
    yield event("response.content_part.done", {
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": final_text, "annotations": []},
    })
    final_item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": final_text, "annotations": []}]}
    yield event("response.output_item.done", {
        "output_index": 0,
        "item": final_item,
    })
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0} if include_usage else None
    yield event("response.completed", {
        "response": responses_envelope(model, response_id, "completed", output=[final_item], usage=usage),
    })

async def deltas_from_text(text: str):
    if text:
        yield text

async def deltas_from_safegpt(safegpt_resp):
    async for event_name, data_value in iter_safegpt_sse_lines(safegpt_resp):
        if event_name.startswith("safegpt.message.content.web_search_note.") or event_name.startswith("safegpt.message.content.code_interpreter_note."):
            logger.info("Tool note: %s %s", event_name, data_value)
            continue
        if event_name == "safegpt.message.content.text.delta":
            delta_text = parse_safegpt_data(data_value)
            if delta_text:
                yield delta_text
            continue
        if event_name == "safegpt.message.done":
            return

async def stream_openai_from_safegpt(safegpt_resp, model: str, include_usage: bool = True):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    started = False
    async for event_name, data_value in iter_safegpt_sse_lines(safegpt_resp):
        if event_name.startswith("safegpt.message.content.web_search_note.") or event_name.startswith("safegpt.message.content.code_interpreter_note."):
            logger.info("Tool note: %s %s", event_name, data_value)
            continue
        if event_name == "safegpt.message.content.text.delta":
            delta_text = parse_safegpt_data(data_value)
            if delta_text:
                if not started:
                    started = True
                    yield sse_data(chat_completion_chunk(model, chunk_id, created, role="assistant"))
                yield sse_data(chat_completion_chunk(model, chunk_id, created, content_delta=delta_text))
            continue
        if event_name == "safegpt.message.done":
            if not started:
                yield sse_data(chat_completion_chunk(model, chunk_id, created, role="assistant"))
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} if include_usage else None
            yield sse_data(chat_completion_chunk(model, chunk_id, created, finish_reason="stop", usage=usage))
            yield sse_done()
            return
    if not started:
        yield sse_data(chat_completion_chunk(model, chunk_id, created, role="assistant"))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} if include_usage else None
    yield sse_data(chat_completion_chunk(model, chunk_id, created, finish_reason="stop", usage=usage))
    yield sse_done()

# --- emulated tool calling (Codex and other agentic clients) -----------------------

AGENT_ITEM_TYPES = {
    "function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output",
    "local_shell_call", "local_shell_call_output", "shell_call", "shell_call_output",
    "apply_patch_call", "apply_patch_call_output",
}

def uses_tool_path(body: Dict[str, Any]) -> bool:
    """Agentic requests are handled statelessly via Message/Execute with the full transcript.

    Codex always sends `instructions` and resends the whole history each turn, so the
    per-session SafeGPT conversation used for JetBrains chat would lose that context.
    """
    if toolemu.has_client_tools(body) or body.get("instructions") or body.get("previous_response_id"):
        return True
    raw = body.get("input")
    if isinstance(raw, list) and any(isinstance(i, dict) and i.get("type") in AGENT_ITEM_TYPES for i in raw):
        return True
    return any(isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls")) for m in body.get("messages") or [])

# SafeGPT's Execute call is not streamed; resend `response.in_progress` while waiting so
# clients with an SSE idle timeout (Codex) keep the stream open.
KEEPALIVE_SECONDS = 10
# Extra SafeGPT calls when a tool-enabled reply has no tool call (see tools.needs_followup).
MAX_NUDGES = 2
# Resends of the identical SafeGPT request when it comes back empty.
EMPTY_RETRIES = 2

def describe_upstream_error(exc: Exception) -> str:
    detail = str(exc) or type(exc).__name__
    if isinstance(exc, httpx.HTTPStatusError):
        detail = f"SafeGPT returned {exc.response.status_code}: {exc.response.text[:2000]}"
    logger.error("SafeGPT call failed: %s", detail)
    return detail

def upstream_error(exc: Exception) -> JSONResponse:
    detail = describe_upstream_error(exc)
    return JSONResponse(status_code=502, content={"error": {"message": detail, "type": "upstream_error", "code": "safegpt_error"}})

async def stream_response_items(produce, model: str, response_id: str, on_complete=None):
    """Emit Responses-API SSE events; `produce()` returns the complete output items."""
    seq = [0]

    def event(event_type: str, payload: Dict[str, Any]) -> str:
        payload = {"type": event_type, "sequence_number": seq[0], **payload}
        seq[0] += 1
        return sse_event(event_type, payload)

    yield event("response.created", {"response": responses_envelope(model, response_id, "in_progress")})
    yield event("response.in_progress", {"response": responses_envelope(model, response_id, "in_progress")})

    task = asyncio.ensure_future(produce())
    try:
        while not (await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS))[0]:
            yield event("response.in_progress", {"response": responses_envelope(model, response_id, "in_progress")})
    finally:
        # Client disconnected: stop waiting on SafeGPT.
        if not task.done():
            task.cancel()
    try:
        items = task.result()
    except Exception as exc:
        failed = responses_envelope(model, response_id, "failed")
        failed["error"] = {"code": "server_error", "message": describe_upstream_error(exc)}
        yield event("response.failed", {"response": failed})
        return
    if on_complete:
        on_complete(items)

    for idx, item in enumerate(items):
        item_id = item["id"]
        itype = item["type"]
        if itype == "message":
            text = item["content"][0]["text"]
            yield event("response.output_item.added", {"output_index": idx, "item": {**item, "status": "in_progress", "content": []}})
            yield event("response.content_part.added", {
                "item_id": item_id, "output_index": idx, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            })
            yield event("response.output_text.delta", {"item_id": item_id, "output_index": idx, "content_index": 0, "delta": text})
            yield event("response.output_text.done", {"item_id": item_id, "output_index": idx, "content_index": 0, "text": text})
            yield event("response.content_part.done", {
                "item_id": item_id, "output_index": idx, "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
        elif itype == "function_call":
            yield event("response.output_item.added", {"output_index": idx, "item": {**item, "status": "in_progress", "arguments": ""}})
            yield event("response.function_call_arguments.delta", {"item_id": item_id, "output_index": idx, "delta": item["arguments"]})
            yield event("response.function_call_arguments.done", {"item_id": item_id, "output_index": idx, "name": item["name"], "arguments": item["arguments"]})
        elif itype == "custom_tool_call":
            yield event("response.output_item.added", {"output_index": idx, "item": {**item, "status": "in_progress", "input": ""}})
            yield event("response.custom_tool_call_input.delta", {"item_id": item_id, "output_index": idx, "delta": item["input"]})
            yield event("response.custom_tool_call_input.done", {"item_id": item_id, "output_index": idx, "input": item["input"]})
        else:
            yield event("response.output_item.added", {"output_index": idx, "item": {**item, "status": "in_progress"}})
        yield event("response.output_item.done", {"output_index": idx, "item": item})
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    yield event("response.completed", {"response": responses_envelope(model, response_id, "completed", output=items, usage=usage)})

def chat_message_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = "\n".join(i["content"][0]["text"] for i in items if i["type"] == "message")
    tool_calls = [
        {"id": i["call_id"], "type": "function", "function": {"name": i["name"], "arguments": i["arguments"]}}
        for i in items if i["type"] == "function_call"
    ]
    message: Dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message

async def stream_chat_from_items(items: List[Dict[str, Any]], model: str, include_usage: bool):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    message = chat_message_from_items(items)
    yield sse_data(chat_completion_chunk(model, chunk_id, created, role="assistant"))
    if message["content"]:
        yield sse_data(chat_completion_chunk(model, chunk_id, created, content_delta=message["content"]))
    for index, call in enumerate(message.get("tool_calls") or []):
        chunk = chat_completion_chunk(model, chunk_id, created)
        chunk["choices"][0]["delta"]["tool_calls"] = [{"index": index, **call}]
        yield sse_data(chunk)
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    yield sse_data(chat_completion_chunk(model, chunk_id, created, finish_reason=finish_reason))
    if include_usage:
        chunk = chat_completion_chunk(model, chunk_id, created)
        chunk["choices"] = []
        chunk["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        yield sse_data(chunk)
    yield sse_done()

async def handle_tool_request(body: Dict[str, Any], is_responses: bool):
    model = body.get("model", default_model_id)
    stream = bool(body.get("stream", False))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
    parallel = body.get("parallel_tool_calls", True) is not False

    if is_responses:
        items = toolemu.normalize_input(body)
        prev_id = body.get("previous_response_id")
        if prev_id:
            previous = load_response(prev_id)
            if previous is None:
                return JSONResponse(status_code=404, content={"error": {
                    "message": f"Previous response with id '{prev_id}' not found.",
                    "type": "invalid_request_error", "param": "previous_response_id", "code": "previous_response_not_found",
                }})
            items = previous + items
    else:
        items = toolemu.chat_messages_to_items(body.get("messages") or [])

    choice = toolemu.parse_tool_choice(body.get("tool_choice"))
    specs = toolemu.select_tools(toolemu.normalize_tools(toolemu.collect_tools(body)), choice)
    text_cfg = body.get("text")
    if not is_responses and isinstance(body.get("response_format"), dict):
        text_cfg = {"format": {**(body["response_format"].get("json_schema") or {}), "type": body["response_format"].get("type")}}
    system_message, prompt = toolemu.build_prompts(items, body.get("instructions"), specs, choice, parallel, text_cfg)
    logger.info(
        "Tool-emulation request: %d input items, %d tools (%s), tool_choice=%s, system=%d chars, prompt=%d chars",
        len(items), len(specs), ", ".join(s.name for s in specs), choice.mode, len(system_message), len(prompt),
    )

    async def execute(prompt_text: str, label: str) -> str:
        # SafeGPT intermittently returns an empty reply, much more often for long outputs
        # such as whole-file rewrites. It fails fast, so first resend the identical request;
        # on the last retry ask for a shorter reply instead.
        request = prompt_text
        for attempt in range(EMPTY_RETRIES + 1):
            reply_text = await safegpt_client.execute(system_message=system_message, prompt=request, model=model)
            logger.info("SafeGPT %s: %s", label, reply_text)
            if reply_text.strip() or attempt == EMPTY_RETRIES:
                break
            shorten = attempt + 1 == EMPTY_RETRIES
            logger.info("Empty SafeGPT reply; retry %d%s", attempt + 1, " asking for a shorter reply" if shorten else " with the same request")
            request = toolemu.shorter_reply_prompt(prompt_text) if shorten else prompt_text
        return reply_text

    user_request = toolemu.last_user_request(items)

    async def produce() -> List[Dict[str, Any]]:
        raw_reply = await execute(prompt, "raw reply")
        reply = toolemu.parse_reply(raw_reply)
        for attempt in range(1, MAX_NUDGES + 1):
            if not toolemu.needs_followup(reply, specs):
                break
            logger.info("Reply has no tool call; asking the model to act or confirm FINAL (attempt %d)", attempt)
            nudged = await execute(toolemu.followup_prompt(prompt, reply.text, user_request), "follow-up reply")
            if toolemu.is_final_marker(nudged):
                break
            retry = toolemu.parse_reply(nudged)
            if retry.calls:
                reply = toolemu.ParsedReply(text=reply.text, calls=retry.calls)
            elif not reply.text and retry.text:
                reply = retry
        output = toolemu.build_output_items(reply, specs, parallel)
        if not output:
            output = toolemu.build_output_items(toolemu.ParsedReply(text=raw_reply.strip() or "(empty response)"), specs, parallel)
        logger.info("Emitting output items: %s", json.dumps([{k: v for k, v in i.items() if k != "content"} for i in output], ensure_ascii=False))
        return output

    if is_responses and stream:
        response_id = f"resp_{uuid.uuid4().hex}"
        return StreamingResponse(
            stream_response_items(produce, model, response_id, on_complete=lambda out: store_response(response_id, items + out)),
            media_type="text/event-stream",
        )

    try:
        output = await produce()
    except httpx.HTTPError as exc:
        return upstream_error(exc)

    if not is_responses:
        if stream:
            return StreamingResponse(stream_chat_from_items(output, model, include_usage), media_type="text/event-stream")
        response = chat_completion_response(model, "")
        message = chat_message_from_items(output)
        response["choices"][0]["message"] = message
        response["choices"][0]["finish_reason"] = "tool_calls" if message.get("tool_calls") else "stop"
        return JSONResponse(response)

    response_id = f"resp_{uuid.uuid4().hex}"
    store_response(response_id, items + output)
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response = responses_envelope(model, response_id, "completed", output=output, usage=usage)
    response["output_text"] = "\n".join(i["content"][0]["text"] for i in output if i["type"] == "message")
    return JSONResponse(response)

@router.get("/v1/models")
@router.get("/models")
def get_models():
    cleanup_sessions()
    return models_list()

@router.get("/healthz")
@router.get("/v1/healthz")
def healthz_root():
    return {"ok": True}

@router.post("/v1/responses")
@router.post("/responses")
@router.post("/v1/chat/completions")
@router.post("/chat/completions")
async def chat_completions(request: Request):
    assert safegpt_client is not None
    cleanup_sessions()
    body = await request.json()
    headers = dict(request.headers)
    is_responses = normalize_path(request.url.path) == "/responses"
    logger.info("=== %s request ===", request.url.path)
    logger.info("Headers: %s", json.dumps(headers, indent=2, ensure_ascii=False))
    logger.info("Body: %s", json.dumps(body, indent=2, ensure_ascii=False))

    if uses_tool_path(body):
        return await handle_tool_request(body, is_responses)

    model = body.get("model", default_model_id)
    stream = bool(body.get("stream", False))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

    session_key = estimated_session_key(request, body)
    state = get_or_create_session(session_key, model, select_chat_app_ids_for_request(body))
    state.touch()

    if should_use_execute(body):
        content = await safegpt_client.execute(
            system_message=build_execute_system_message(body),
            prompt=build_execute_prompt(body),
            model=model,
        )
        if stream:
            if is_responses:
                return StreamingResponse(
                    stream_responses_events(deltas_from_text(content), model=model, include_usage=include_usage),
                    media_type="text/event-stream",
                )
            async def execute_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                created = int(time.time())
                yield sse_data(chat_completion_chunk(model, chunk_id, created, role="assistant"))
                if content:
                    yield sse_data(chat_completion_chunk(model, chunk_id, created, content_delta=content))
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} if include_usage else None
                yield sse_data(chat_completion_chunk(model, chunk_id, created, finish_reason="stop", usage=usage))
                yield sse_done()
            return StreamingResponse(execute_stream(), media_type="text/event-stream")
        if is_responses:
            return JSONResponse(responses_object(model, f"resp_{uuid.uuid4().hex}", content))
        return JSONResponse(chat_completion_response(model, content))

    messages = extract_messages(body)
    last_user_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_prompt = message_text(m).strip()
            break
    if not last_user_prompt:
        last_user_prompt = " ".join(message_text(m) for m in messages).strip()

    if not state.conversation_id:
        conv = await safegpt_client.create_conversation(
            model=model,
            prompt=last_user_prompt,
            auto_tools=False,
            chat_app_ids=state.chat_app_ids,
            conversation_type=0,
        )
        state.conversation_id = conv.get("conversationId")
        state.touch()

    if stream:
        safegpt_resp = await safegpt_client.create_message_stream(state.conversation_id, last_user_prompt, model)
        if is_responses:
            return StreamingResponse(
                stream_responses_events(deltas_from_safegpt(safegpt_resp), model=model, include_usage=include_usage),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            stream_openai_from_safegpt(safegpt_resp, model=model, include_usage=include_usage),
            media_type="text/event-stream",
        )

    safegpt_resp = await safegpt_client.create_message_stream(state.conversation_id, last_user_prompt, model)
    raw = await safegpt_resp.aread()
    text = raw.decode("utf-8", errors="replace")

    final_text_parts = []
    for block in text.split("\n\n"):
        event_name = ""
        data_value = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_value = line[len("data: "):].strip()
        if event_name == "safegpt.message.content.text.delta":
            final_text_parts.append(parse_safegpt_data(data_value))
        elif event_name == "safegpt.message.content.text.done":
            done_text = parse_safegpt_data(data_value)
            if done_text:
                final_text_parts = [done_text]

    final_text = "".join(final_text_parts).strip()
    if is_responses:
        return JSONResponse(responses_object(model, f"resp_{uuid.uuid4().hex}", final_text))
    return JSONResponse(chat_completion_response(model, final_text))

# @router.post("/chat/completions")
# @router.post("/responses")
# @router.post("/v1/chat/completions")
# async def chat_completions(request: Request):
#     headers = dict(request.headers)
#
#     logger.info("=== /chat/completions request ===")
#     logger.info("Headers: %s", json.dumps(headers, indent=2, ensure_ascii=False))
#
#     raw_body = await request.body()
#     body_text = raw_body.decode("utf-8", errors="replace").strip()
#
#     logger.info("Raw body: %r", body_text)
#
#     body = {}
#     if body_text:
#         try:
#             body = json.loads(body_text)
#         except json.JSONDecodeError:
#             logger.warning("Request body was not valid JSON; using empty body.")
#             body = {}
#
#     stream = bool(body.get("stream", False))
#     model = body.get("model", MODEL_ID)
#     messages = body.get("messages", [])
#
#     system_parts = []
#     user_parts = []
#     assistant_parts = []
#
#     for m in messages:
#         role = m.get("role")
#         content = m.get("content", "")
#         if role == "system":
#             system_parts.append(content)
#         elif role == "user":
#             user_parts.append(content)
#         elif role == "assistant":
#             assistant_parts.append(content)
#
#     reply_text = (
#         "Proxy received your request.\n\n"
#         f"System:\n{''.join(system_parts).strip()}\n\n"
#         f"User:\n{''.join(user_parts).strip()}\n\n"
#         f"Assistant history messages: {len(assistant_parts)}\n"
#         f"Raw message count: {len(messages)}"
#     )
#
#     if not stream:
#         return chat_completion_response(model, reply_text)
#
#     async def event_stream():
#         chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
#         created = int(time.time())
#
#         first = chat_completion_chunk(model, chunk_id, created, role="assistant")
#         yield f"data: {json.dumps(first)}\n\n"
#
#         second = chat_completion_chunk(model, chunk_id, created, content_delta=reply_text)
#         yield f"data: {json.dumps(second)}\n\n"
#
#         done = chat_completion_chunk(model, chunk_id, created, finish_reason="stop")
#         yield f"data: {json.dumps(done)}\n\n"
#         yield "data: [DONE]\n\n"
#
#     return StreamingResponse(event_stream(), media_type="text/event-stream")
