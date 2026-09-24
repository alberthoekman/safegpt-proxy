# SafeGPT Proxy for JetBrains AI Assistant

This project is a local FastAPI proxy that makes SafeGPT available to JetBrains AI Assistant as an OpenAI-compatible provider.

It translates JetBrains requests like:

- `GET /models`
- `GET /v1/models`
- `POST /chat/completions`
- `POST /v1/chat/completions`

into SafeGPT API calls such as:

- `POST /v1/Message/Execute`
- `POST /v1/Conversation/Create`
- `POST /v1/Message/Create/{conversationId}`

It also translates SafeGPT streaming responses back into OpenAI-compatible SSE chunks.

## What it does

- Exposes an OpenAI-compatible model list
- Routes JetBrains orchestration prompts to SafeGPT `Message/Execute`
- Creates and reuses one SafeGPT conversation per JetBrains session
- Streams SafeGPT responses back as OpenAI chat-completion chunks
- Hides SafeGPT internal tool-note events from JetBrains while preserving them in logs
- Supports SafeGPT tools via hardcoded `chatAppIds`
- Uses `autoTools: false` by default, as requested

## Project layout

```text
safegpt-proxy/
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── models.py
    ├── router.py
    ├── safegpt_client.py
    ├── session.py
    └── settings.py
```

## Requirements

- Python 3.11+
- Access to a SafeGPT instance
- A SafeGPT API token
- JetBrains AI Assistant configured to use this proxy as an OpenAI-compatible provider

## Installation

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Using uv

```bash
uv sync
```

## Configuration

Set these environment variables:

- `HOST`  
  Host to bind to. Default: `0.0.0.0`

- `PORT`  
  Port to bind to. Default: `8000`

- `SAFEGPT_BASE_URL`  
  Base URL of your SafeGPT API. Example: `http://localhost:8080`

- `SAFEGPT_TOKEN`  
  SafeGPT bearer token.

- `DEFAULT_MODEL_ID`  
  OpenAI-compatible model ID exposed to JetBrains. Default: `gpt-5.6-terra`

- `SAFEGPT_WEB_SEARCH_CHAT_APP_ID`  
  Hardcoded SafeGPT app id for web search.

- `SAFEGPT_CODE_INTERPRETER_CHAT_APP_ID`  
  Hardcoded SafeGPT app id for code interpreter.

## Run the proxy

### With Python

```bash
python main.py
```

### With Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## JetBrains setup

In JetBrains AI Assistant:

1. Add a custom OpenAI-compatible provider
2. Set the proxy base URL
3. Use the exposed model id, for example `gpt-5.6-terra`
4. Keep tool calling enabled if you want SafeGPT tools to be available
5. Point JetBrains to this proxy instead of calling SafeGPT directly

## Codex setup

Add a provider to `~/.codex/config.toml`:

```toml
model = "gpt-5.6-sol"
model_provider = "safegpt"

[model_providers.safegpt]
name = "SafeGPT proxy"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
```

## Cline setup (recommended agent)

Cline uses plain Chat Completions function tools, which the proxy emulates with far fewer workarounds than Codex needs.

1. In Cline, choose the **OpenAI Compatible** provider.
2. Base URL: `http://localhost:8000/v1`, API key: any value, Model ID: `gpt-5.6-sol`.
3. Leave tool/function-calling support enabled for the model; without it Cline sends no tools at all.
4. Use **Act mode** to let it edit files. In Plan mode Cline only offers read-only tools.
5. Consider disabling Cline skills you don't need: the `skills` tool tells the model to invoke a matching skill before anything else, and SafeGPT's model tends to pick one even when none fits.
6. Keep file edits on manual approval, at least at first: SafeGPT's model sometimes rewrites a file by deleting and re-adding it.

SafeGPT ignores `reasoning_effort` and the requested model; it always uses its own configured model.

### How tool calls work

SafeGPT has no native function calling (`Message/Execute` only takes `systemMessage` and `prompt`), so the proxy emulates it:

1. Requests that carry client-side tools (`function`, `custom`, `namespace`, `local_shell`, `shell`, `apply_patch`), `instructions`, `previous_response_id` or tool-call history are handled statelessly via `Message/Execute`.
2. The tool definitions and a strict call protocol are put at the start of the system message; the full transcript, including earlier tool calls and their results, becomes the prompt.
3. The model answers with `<tool_call name="...">ARGS</tool_call>` blocks, which the proxy turns into real `function_call`, `custom_tool_call`, `local_shell_call`, `shell_call` or `apply_patch_call` output items (streamed as the matching Responses SSE events).
4. `/chat/completions` gets the same treatment and returns `tool_calls` with `finish_reason: "tool_calls"`.
5. Function tools whose only parameter is a single string (such as Cline's `apply_patch` with `input`) are offered to the model as freeform tools; the proxy does the JSON encoding, so patches never need escaping.

Hosted OpenAI tools (`web_search`, `file_search`, `code_interpreter`, `mcp`, ...) are ignored. The reply is buffered until SafeGPT finishes, because the proxy has to parse it before sending anything, so tool-path responses arrive all at once rather than token by token.

Agent specifics:

- Newer Codex versions ("responses lite") send no top-level `instructions`/`tools`; tools arrive as an `additional_tools` input item. Both shapes are supported.
- In Codex "code mode" the only real tool is `exec`, which runs JavaScript that calls the actual tools (`exec_command`, `apply_patch`, ...). SafeGPT's model reliably calls tools directly but writes unusable scripts, so the proxy offers the nested tools directly, wraps each call into a one-line `exec` script, and unwraps those scripts again when replaying history.
- The model often stops after announcing or planning its next step ("I'll create the files now.", "Plan: 1) ..."), and SafeGPT sometimes returns an empty reply. Whenever a tool-enabled reply has no tool call, the proxy asks SafeGPT once more (up to twice) to either emit the tool call for its next step or answer `FINAL` if it is really done or needs the user. This applies to Codex and Cline alike, and costs one extra call on genuine final answers.
- Malformed JSON arguments are repaired where possible (unescaped newlines inside strings, trailing text after the object).
- While SafeGPT is working on a streamed Responses request, `response.in_progress` events are sent every 10 s so Codex's stream idle timeout does not fire.

## How routing works

### 1. Model listing
`GET /models` and `GET /v1/models` return a static OpenAI-style model list.

### 2. Orchestration prompts
JetBrains sends hidden prompts for things like:

- chat title generation
- retrieval subqueries
- context relevance checks

These are routed to:

- `POST /v1/Message/Execute`

### 3. Normal chat
The first real chat request creates a SafeGPT conversation using:

- `POST /v1/Conversation/Create`

Then follow-up messages go to:

- `POST /v1/Message/Create/{conversationId}`

### 4. Streaming
SafeGPT SSE events are translated into OpenAI-compatible SSE chunks.

SafeGPT tool-note events such as:

- `web_search_note.*`
- `code_interpreter_note.*`

are suppressed from JetBrains but logged by the proxy.

## Tool behavior

Tools are enabled by default in the proxy configuration.

SafeGPT decides when to use:

- web search
- code interpreter

The proxy only needs to ensure the correct SafeGPT `chatAppIds` are attached to the conversation.

## Notes on session handling

The proxy uses an in-memory session map to keep one SafeGPT conversation per JetBrains session.

Current session key logic is simple and based on request headers and model selection. If JetBrains exposes a better stable session id later, use that instead.

## Limitations

- Session state is in-memory only
- Non-streaming SafeGPT message creation is handled conservatively
- Tool selection is left to SafeGPT, not the proxy
- The conversation replay logic is minimal and can be improved if you need richer multi-turn history handling

## Suggested next improvements

- Persist session state to disk or Redis
- Add explicit mapping for more JetBrains request types
- Improve history replay into SafeGPT conversations
- Add structured tool-call handling if JetBrains requires it
- Add better token usage accounting

## Development

You can format and lint with:

```bash
ruff check .
ruff format .
```

## License

Add your preferred license here.
