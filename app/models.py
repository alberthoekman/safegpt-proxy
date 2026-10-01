import time
import uuid

MODEL_ID = "gpt-5.6-sol"

def models_list():
    """Build the OpenAI-style model list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "safegpt-proxy",
            }
        ],
    }

def chat_completion_response(model: str, content: str):
    """Build a non-streaming chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

def chat_completion_chunk(model: str, chunk_id: str, created: int, content_delta=None, role=None, finish_reason=None, usage=None):
    """Build a single OpenAI-compatible streaming chunk."""
    delta = {}
    if role is not None:
        delta["role"] = role
    if content_delta is not None:
        delta["content"] = content_delta
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk

def responses_usage(input_tokens: int = 0, output_tokens: int = 0):
    """Build a Responses API usage object, including the *_details fields.

    Some Responses API clients (JetBrains AI Assistant's koog library) deserialize this
    strictly and reject a usage object without input_tokens_details/output_tokens_details.
    """
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }

def responses_output_item(item_id: str, text: str, status: str = "completed"):
    """Build one Responses API assistant message item."""
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text, "annotations": []}
        ],
    }

def responses_object(model: str, response_id: str, text: str, item_id: str = None, status: str = "completed", usage=None):
    """Build a complete Responses API response object."""
    item_id = item_id or f"msg_{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model or MODEL_ID,
        "output": [responses_output_item(item_id, text, status="completed")],
        "output_text": text,
        "parallel_tool_calls": False,
        "text": {"format": {"type": "text"}},
        "usage": usage or responses_usage(),
    }
