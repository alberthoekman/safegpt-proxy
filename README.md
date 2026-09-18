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
