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

See "Message/Execute (orchestration calls)" below for a known connection-reuse bug that
affected this endpoint (2026-09-29).

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

### Message/Execute (orchestration calls)
`POST /v1/Message/Execute` is stateless by contract:
```json
{
  "systemMessage": "...",
  "prompt": "..."
}
```
SafeGPT returns `{"content": "..."}`. No `model`, session/conversation id, or
`chatAppIds` are sent or expected.

**Known issue (found 2026-09-29, mitigated):** SafeGPT started rejecting every Execute
call with a fixed, fabricated error:
```
HTTP 500: {"success":false,"statusCode":500,"message":"HTTP 400 (invalid_request_error:
string_above_max_length)\nParameter: input[0].content[1].text\n\nInvalid
'input[0].content[1].text': string too long. Expected a string with maximum length
10485760, but got a string with length 12912575 instead."}
```
Evidence gathered, in order: our real payloads measured 57,770–~64,000 bytes while
SafeGPT reported the identical `12,912,575`-char figure regardless of actual request
size; a different SafeGPT API key reproduced the identical error; a *fresh*
`httpx.AsyncClient` with connection reuse disabled still failed on its very first
request (ruling out connection-reuse/keep-alive desync); a plain request via Postman
succeeded immediately. That pointed at `SafeGPTClient.headers()` sending
`Accept: application/json, text/event-stream` on **every** call including
`Message/Execute` (which never streams) as a plausible cause — fixed by narrowing the
default `Accept` to `application/json` (see "Fix" below). **However**, a live bisection
after that fix — reconstructing a real failing request with near-exact fidelity (system
message within 60 chars of the real 14,455; a realistic multi-turn prompt) — succeeded
every time across 9 separate live calls. So the signature does **not** reproduce
deterministically from request content, size, or (as far as tested) the `Accept` header
either. Current best understanding: this is an **intermittent/probabilistic** failure on
SafeGPT's side (e.g. one bad backend instance behind a load balancer that occasionally
serves a request), not a sustained outage and not something our request content
controls. This is not fixable by trimming the outgoing prompt — the existing
`SAFEGPT_MAX_PROMPT_CHARS` / tool-output-size guards in `app/router.py` / `app/tools.py`
stay (they address a real, separate sizing concern), but they never touched this bug.

**Update (2026-09-29, later the same day):** the "intermittent/probabilistic" theory
above was itself disproven by a further live test. Three sequential, non-identical,
fresh-client requests were replayed directly against `SafeGPTClient` (bypassing Cline
entirely): a ~19.4KB reconstruction failed identically twice in a row (including with a
random nonce appended, on a brand-new client with no shared breaker state — ruling out a
content-hash/cache theory), while the *same system message* with an ~8.5KB and ~12.5KB
prompt both succeeded. This is a real, size-correlated, currently-active threshold
somewhere between ~12.5KB and ~19.4KB — far below the ~31–32KB payloads that succeeded
9/9 times earlier that same day, meaning the threshold appears to shrink over time. That
is much more consistent with a **rolling usage quota** (characters/tokens per hour or day
on this API key/account) than a random per-request hiccup, especially since every retry
is itself a real call that could be consuming more of the same budget.

A full codebase dive (see `HANDOFF.md`'s "Known SafeGPT quirks" for the file-by-file
findings) found **no bug in this proxy that explains the mismatch**: `httpx`'s actual
wire encoding (checked directly against the installed 0.28.1 source) matches our own
size logging almost exactly (our log is if anything slightly larger, never smaller);
`app/tools.py`'s transcript/prompt construction has no unbounded-growth or duplication
path; `app/session.py`'s response store only affects the unused (for Cline)
`previous_response_id` chaining path. One real bug *was* found and fixed regardless: a
shared, unsynchronized `httpx.AsyncClient` (`SafeGPTClient.client`) that self-heal
retries closed and replaced without a lock — a latent race under concurrent proxy
requests. Since keep-alive was already disabled, there was no reason to share a client
instance at all; every call now opens and closes its own via `_new_client()`.

One more clue worth keeping on record: the error's exact shape — `invalid_request_error`,
path `input[0].content[1].text`, limit `10485760` (exactly 10 MiB) — matches OpenAI's own
documented Responses API per-field text-length limit and error format, not anything
SafeGPT documents as its own. This is consistent with `Message/Execute` being a thin
relay in front of real OpenAI's Responses API, where `input[0]` is assembled by SafeGPT
itself — meaning the bloat most likely happens after our request leaves this process.
Not proof, but the only theory consistent with every piece of evidence gathered so far.

Mitigations in `app/safegpt_client.py` (`SafeGPTClient`):
1. `_execute_with_self_heal()` retries up to `SELF_HEAL_MAX_ATTEMPTS` (2 — lowered from 3
   once evidence showed extra retries don't change the outcome against what looks like a
   real, currently-active upstream constraint; one retry still absorbs a genuinely
   transient blip without spending extra calls against a wall).
2. If all attempts fail identically, `execute()` raises `SafeGPTStuckExecuteError` and
   opens a small circuit breaker with a short cooldown (`BREAKER_BASE_COOLDOWN_SECONDS =
   10`, capped at `BREAKER_MAX_COOLDOWN_SECONDS = 120`) so a failure streak doesn't lock
   out every subsequent Cline request for long. It self-probes and recovers automatically.
3. No persistent/shared `httpx.AsyncClient` — every call (`execute`, `create_conversation`,
   `create_message_stream`) opens and closes its own via `_new_client()`, which also keeps
   connection reuse disabled (`httpx.Limits(max_keepalive_connections=0)` + `Connection:
   close`).
4. `headers()` narrowing (`Accept: application/json` by default; `create_message_stream()`
   explicitly opts into `text/event-stream`) is kept as a plausible, harmless contributing
   fix, even though it was not confirmed as the sole root cause.

Run `python scripts/diagnose_safegpt.py` for a minimal, reproducible check of this
endpoint independent of the Cline/proxy pipeline. Given the evidence above, the
recommended next step if this recurs is a SafeGPT support ticket (citing the OpenAI-limit
number and the shrinking-threshold observation), not another proxy-side size guess.

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
