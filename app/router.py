import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from app.models import models_list, chat_completion_response, chat_completion_chunk, MODEL_ID
from app.session import get_or_create_session, cleanup_sessions
from app.safegpt_client import SafeGPTClient, iter_safegpt_sse_lines, parse_safegpt_data

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
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)

def extract_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    return body.get("messages", []) or []

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

def sse_done() -> str:
    return "data: [DONE]\n\n"

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

@router.get("/v1/models")
@router.get("/models")
def get_models():
    cleanup_sessions()
    return models_list()

@router.get("/healthz")
@router.get("/v1/healthz")
def healthz_root():
    return {"ok": True}

@router.post("/v1/chat/completionss")
@router.post("/chat/completionss")
async def chat_completions(request: Request):
    assert safegpt_client is not None
    cleanup_sessions()
    body = await request.json()
    headers = dict(request.headers)
    logger.info("=== /chat/completions request ===")
    logger.info("Headers: %s", json.dumps(headers, indent=2, ensure_ascii=False))
    logger.info("Body: %s", json.dumps(body, indent=2, ensure_ascii=False))

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
    return JSONResponse(chat_completion_response(model, final_text))

@router.post("/chat/completions")
@router.post("/responses")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    headers = dict(request.headers)

    logger.info("=== /chat/completions request ===")
    logger.info("Headers: %s", json.dumps(headers, indent=2, ensure_ascii=False))

    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="replace").strip()

    logger.info("Raw body: %r", body_text)

    body = {}
    if body_text:
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            logger.warning("Request body was not valid JSON; using empty body.")
            body = {}

    stream = bool(body.get("stream", False))
    model = body.get("model", MODEL_ID)
    messages = body.get("messages", [])

    system_parts = []
    user_parts = []
    assistant_parts = []

    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    reply_text = (
        "Proxy received your request.\n\n"
        f"System:\n{''.join(system_parts).strip()}\n\n"
        f"User:\n{''.join(user_parts).strip()}\n\n"
        f"Assistant history messages: {len(assistant_parts)}\n"
        f"Raw message count: {len(messages)}"
    )

    if not stream:
        return chat_completion_response(model, reply_text)

    async def event_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        first = chat_completion_chunk(model, chunk_id, created, role="assistant")
        yield f"data: {json.dumps(first)}\n\n"

        second = chat_completion_chunk(model, chunk_id, created, content_delta=reply_text)
        yield f"data: {json.dumps(second)}\n\n"

        done = chat_completion_chunk(model, chunk_id, created, finish_reason="stop")
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
