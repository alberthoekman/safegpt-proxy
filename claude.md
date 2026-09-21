# Claude.md — SafeGPT proxy for JetBrains AI Assistant

## Purpose

Build and maintain a local FastAPI proxy that makes SafeGPT usable from JetBrains AI Assistant as an OpenAI-compatible provider.

The proxy must primarily support the **OpenAI Responses API** (`/responses` and `/v1/responses`) because JetBrains is now sending Responses-style requests when the `terra` model is selected.

The proxy should also keep backward compatibility with:
- `GET /models`
- `GET /v1/models`
- `POST /chat/completions`
- `POST /v1/chat/completions`

## Core behavior

### 1) Model listing
Expose a minimal OpenAI-style model list.

Return at least one model id, typically:
- `gpt-5.6-terra`

You may map this internally to the SafeGPT model id required by the SafeGPT API.

### 2) Responses API handling
The main entrypoint is now:
- `POST /responses`
- `POST /v1/responses`

JetBrains sends payloads shaped like:
```json
{
  "input": [
    {
      "role": "user",
      "type": "message",
      "content": [
        {
          "type": "input_text",
          "text": "hello"
        }
      ]
    }
  ],
  "model": "gpt-5.6-terra",
  "stream": true
}
```

Your handler should:
- parse `input[]`
- extract text from `content[]`
- classify the request
- route to the correct SafeGPT backend call
- stream back an OpenAI-compatible response when `stream=true`

### 3) Request routing
Use these routing rules:

#### Orchestration prompts
For hidden JetBrains prompts like:
- chat title generation
- retrieval subquery generation
- context relevance checks

Route to:
- `POST /v1/Message/Execute`

#### Normal chat
For actual user chat:
- create one SafeGPT conversation per JetBrains session
- reuse that conversation for later turns
- send new messages to:
  - `POST /v1/Message/Create/{conversationId}`

#### Tools
SafeGPT tools are already enabled by default.

SafeGPT supports at least:
- web search
- code interpreter

The proxy should:
- keep JetBrains tool calling enabled
- use hardcoded SafeGPT `chatAppIds`
- set `autoTools: false`
- allow SafeGPT to decide when to use web search or code interpreter

## Important SafeGPT API facts

### Conversation creation
SafeGPT conversation creation requires:
- `model`
- `prompt`
- `autoTools`

Example:
```json
{
  "model": "gpt-5.6-terra",
  "prompt": "Hello",
  "autoTools": false,
  "chatAppIds": ["..."]
}
```

SafeGPT returns:
- `conversationId`
- metadata such as `conversationName`, timestamps, `messages`, and token counters

### Message creation
SafeGPT message creation can be called with:
```json
{
  "prompt": "hello"
}
```

This may return a streaming SSE response.

### Streaming event patterns from SafeGPT
SafeGPT streaming emits events such as:

#### Normal assistant output
- `safegpt.message.user.id`
- `safegpt.message.id`
- `safegpt.message.content.id`
- repeated `safegpt.message.content.text.delta`
- `safegpt.message.content.text.done`
- `safegpt.message.done`

#### Web search tool usage
- `safegpt.message.content.web_search_note.start`
- `safegpt.message.content.web_search_note.delta`
- `safegpt.message.content.web_search_note.done`

#### Code interpreter tool usage
- `safegpt.message.content.code_interpreter_note.start`
- `safegpt.message.content.code_interpreter_note.delta`
- `safegpt.message.content.code_interpreter_note.done`

The proxy should:
- suppress these internal note events from JetBrains
- log them for observability
- forward only OpenAI-compatible output externally

## OpenAI-compatible stream translation

Translate SafeGPT SSE to OpenAI SSE:

### For assistant chunks
Emit:
- first chunk with `delta.role = assistant`
- subsequent chunks with `delta.content`
- final chunk with `finish_reason = stop`
- terminal `[DONE]`

### Usage
If JetBrains requests:
```json
"stream_options": { "include_usage": true }
```

emit a final chunk containing usage fields, even if token counts are zero for now.

## Session handling

Keep **one SafeGPT conversation per JetBrains session**.

### Suggested session key
Use a composite key based on:
- authorization header
- model id
- and later, if available, a stable JetBrains session id

### Session state should store
- JetBrains session key
- SafeGPT conversation id
- selected model
- hardcoded tool app ids
- timestamps

Use in-memory storage initially. Redis or disk persistence can be added later.

## Response shape differences

### chat/completions
Old chat format:
- `messages[]`

### responses
New format:
- `input[]`

The proxy must support both, but `responses` is now the primary target.

## Implementation guidance

### Recommended file layout
- `main.py`
- `app/__init__.py`
- `app/settings.py`
- `app/models.py`
- `app/session.py`
- `app/safegpt_client.py`
- `app/router.py`

### Current behavior that must be fixed or preserved
- support both `/models` and `/v1/models`
- support both `/responses` and `/v1/responses`
- keep `/chat/completions` support for compatibility
- log actual request path
- continue supporting HTTP/1.1 inspection mode during debugging

## Practical build steps

1. Implement `/responses` first.
2. Parse `input[]`.
3. Convert it to an internal text transcript.
4. Decide whether the request is orchestration or normal chat.
5. For orchestration, use SafeGPT `Message/Execute`.
6. For normal chat, create/reuse a SafeGPT conversation and call `Message/Create/{conversationId}`.
7. Translate SafeGPT SSE back into OpenAI SSE.
8. Hide SafeGPT tool-note events from JetBrains.
9. Return OpenAI-compatible JSON/SSE responses.

## Observed JetBrains behavior

JetBrains may send:
- repeated user messages
- hidden orchestration prompts
- Responses API payloads with `input[]`
- model `gpt-5.6-terra`
- streaming requests

Do not assume every request is the visible user message. Some requests are internal sub-steps.

## Notes on tools

You do **not** need to manually choose web search vs code interpreter in most cases.

SafeGPT should decide when to use them, as long as:
- the correct SafeGPT chat apps are enabled
- `autoTools: false`
- the conversation is created with the intended `chatAppIds`

## Code quality expectations

- Keep the proxy modular
- Keep request parsing strict but tolerant
- Make stream translation reliable
- Preserve enough logging to debug JetBrains behavior
- Support both `/responses` and `/chat/completions` during the transition
- Prefer clear separation between:
  - request classification
  - SafeGPT API client
  - session storage
  - OpenAI-compatible response building

## Minimal acceptance criteria

The proxy is considered working when:
- JetBrains can connect to it as an OpenAI-compatible provider
- `/responses` requests are accepted
- SafeGPT conversations are created and reused
- SafeGPT message streams are translated back into OpenAI-compatible SSE
- JetBrains hidden orchestration prompts work
- tool-note events remain hidden from JetBrains but visible in logs
