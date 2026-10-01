# Handoff: tool calling through the SafeGPT proxy

Last updated: 2026-09-24. Written so a new session can continue without the old chat.

## Goal

The user must use SafeGPT (https://api.safegpt.nl) and wants coding agents (first Codex, now **Cline**) to use it with real tool calling. SafeGPT has **no native function calling**: `POST /v1/Message/Execute` takes only `{systemMessage, prompt}` and returns `{"content": "..."}`. It ignores any model name and `reasoning_effort`. So this proxy **emulates** OpenAI tool calling on top of plain text.

## Current state

- Tool emulation is committed in `fd8314e` ("Built to Cline"). Follow-up changes (FINAL challenge, chat keep-alive) are uncommitted. Branch: `main`.
- 35 tests pass: `uv run --extra dev pytest -q tests`. Lint the changed files with `uv run --extra dev ruff check app/tools.py app/router.py tests` (clean). The remaining ruff findings in `app/settings.py` and `app/safegpt_client.py` are old imports; leave them.
- The user runs the proxy with `python main.py` (port 8000) and must **restart it** after code changes.
- **Recommended agent: Cline** (OpenAI Compatible provider, base URL `http://localhost:8000/v1`, model `gpt-5.6-sol`, tool support on, **Act mode** to edit). Codex also works but needs more workarounds. See the setup sections in `README.md`.

## How the emulation works (`app/tools.py`, `app/router.py`)

1. `uses_tool_path()` in `router.py` sends a request to the stateless tool path if it has client tools (top-level `tools` **or** a Codex `additional_tools` input item), `instructions`, `previous_response_id`, or tool-call history. Other requests (plain JetBrains chat) keep the old per-session SafeGPT conversation path.
2. `build_prompts()` puts the **tool protocol first** in the system message (client instructions after it; Codex's ~60 KB would otherwise bury it). The whole transcript, including earlier tool calls and results, becomes the prompt, and a reminder is added at the end.
3. The model answers with `<tool_call name="X">ARGS</tool_call>`. `parse_reply()` also accepts a missing closing tag and a trailing `FINAL`. `build_output_items()` / `_call_item()` turn calls into `function_call`, `custom_tool_call`, `local_shell_call`, `shell_call` or `apply_patch_call` items. Chat Completions gets `tool_calls` with `finish_reason: "tool_calls"`.
4. **Single-string-parameter function tools** (Cline's `apply_patch(input)`) are shown to the model as freeform: it writes the raw patch and the proxy JSON-encodes it (`wrap_param`, `_wrapped_args`). Past calls are shown back in raw form.
5. JSON arguments are repaired where possible (`_load_json_object`: unescaped newlines inside strings, trailing text).
6. **Codex specifics:** tools come in `additional_tools`, and `functions` is the default namespace (emit calls without a namespace). In "code mode" the only real tool is `exec` (JavaScript). The proxy parses the nested tools out of `exec`'s description, offers them directly, hides `exec`, and wraps each call as `const result = await tools.X(args); text(...)` (`nested_call_js` / `unwrap_nested_call`).
7. **Streaming Responses:** the SSE stream opens immediately and sends `response.in_progress` every 10 s while SafeGPT works (Codex's idle timeout only resets on real events). Upstream failures become a `response.failed` event.
8. **Reliability loop** in `handle_tool_request()` → `produce()` in `router.py`:
   - **Empty replies:** SafeGPT intermittently returns `{"content":""}`, fast, more often for long outputs. It is not a content filter; this was verified by probing. `execute()` resends the identical request once, then asks for a shorter reply (`shorter_reply_prompt`). Up to `EMPTY_RETRIES = 2`.
   - **No tool call:** the model often stops after announcing or planning ("I'll…", "Plan: 1)…"). If a tool-enabled reply has no call, `followup_prompt()` asks it to emit the call for its next step, or to reply exactly `FINAL` only if every requested change is done or it needs the user. It restates the real user request (`last_user_request`, which skips Cline's "[TASK RESUMPTION]" notices). Up to `MAX_NUDGES = 2`.
9. The tool protocol also says: never delete a file in order to rewrite it, and keep edits small. This was added after Cline's model deleted `README.md` to rewrite it; the README was restored from git and my sections re-applied.

## Live results so far (real SafeGPT)

- Codex, "make the files": 7 of 7 runs produced real `exec` / `exec_command` calls.
- Cline plan/act loop works end to end: tools run and results come back.
- Latest replay of "update the readme: add 'moi' at the top", after reading the file: **4 of 5 runs produced a valid in-place `apply_patch` Update**. In 1 run the model wrote "I'll prepend `moi`… then verify" and still answered `FINAL` to the follow-up.

## Known SafeGPT quirks

- **`string_above_max_length` with a size far above what was sent (fixed).** SafeGPT fetches every `scheme://` URL in a prompt and inlines the page, then measures the result. Symptoms: a fixed reported size (12,868,788 / 12,912,575 chars) regardless of the real request size (57 KB-110 KB), and a "threshold" that seemed to move depending on whether the prompt happened to contain a URL. The `laptop` branch (commit `9ab55cc`) chased this with other theories (the `Accept` header, connection reuse, a rolling usage quota) and added a retry/circuit breaker plus a 6,000-char tool-output cap; none of that addressed the cause and it was not merged. Fix: `defang_urls()` / `restore_urls()` in `app/safegpt_client.py` (see open issue 6). Kept from `laptop`: `Accept: application/json` except on `Message/Create`, and a log line with the real outgoing Execute payload size, so a reported-vs-sent mismatch is visible at once.
- **Intermittent empty replies:** see item 8 in "How the emulation works": SafeGPT sometimes returns `{"content":""}` fast, more often for long outputs. Handled by resend-then-shorten retries.
- **Standalone check and replay:** `python -m scripts.diagnose_safegpt` sends one Execute call through `SafeGPTClient`, bypassing the proxy and Cline. Every SafeGPT error response makes the proxy save the exact request to `logs/failed/<UTC time>-<Endpoint>.json` (log line: "request saved to ..."); replay it byte for byte with `--payload <file>` instead of reconstructing it.

## Open issues / next steps

1. **Wrong `FINAL` answers (~1 in 5): fix implemented, not yet tested live.** `tools.announces_next_step()` detects drafts that announce a step ("I'll", "I need to", "let me", "Plan:", "then verify"). If the follow-up answers FINAL to such a draft, `produce()` asks once more with `followup_prompt(..., insist=True)`, which says FINAL is not valid now. Replay the "add 'moi' at the top" case live to measure. Live log 2026-09-25 showed it missing drafts with typographic apostrophes ("I’ll", "I’m") and progressive forms ("I’m locating…"); the check now normalizes `’` and matches "I'm/I am <verb>ing". "Let me know" no longer counts as a step.
2. **Chat Completions streaming now opens immediately (not yet tested live with Cline).** `stream_chat_items()` sends the role chunk at once, then `: keep-alive` SSE comments every 10 s while SafeGPT works, then content/tool calls. Upstream failures become a `data: {"error": ...}` chunk followed by `[DONE]`. This targets the unanswered first request in `pasted-text-4` (Cline sent "[TASK RESUMPTION]" after no `200 OK`); still unknown whether that was a timeout or a user cancel.
3. **Finished answers restated instead of FINAL** (live log: "Done - I added moi" repeated, 2 wasted calls). Now, if a follow-up again has no tool call and the draft does not announce a next step, the draft is accepted as final after one follow-up.
4. **Test all tools:** Cline sends 57 tools (6 own: read_files, search_codebase, fetch_web_content, apply_patch, ask_question, run_commands; 51 from the jetbrains-pycharm MCP server, including a second `jetbrains-pycharm__apply_patch`), making a ~75 KB system message. Phase 1 (Cline own tools) findings: the model called short MCP names (`search_file`, `search_text`), which Cline rejected -> `tools.resolve_tool()` now maps a name to the single tool matching case-insensitively or by `__`/`.` suffix, and the protocol asks for full prefixed names. The delete rule was over-applied (refused a requested delete) -> protocol now allows requested deletes. "Refactor this" guessed instead of asking -> protocol now says ask on vague requests. `run_commands` worked (the `uvicorn` error was Cline using system python, not `.venv`). Not yet re-tested live. Next: re-run Phase 1, then the PyCharm MCP tools, then compare with that MCP server disabled.
5. **Latency:** retries and follow-ups can mean several SafeGPT calls (~2 s each, up to ~10–20 s per step) with nothing shown in Cline meanwhile.
6. **Prompt size:** SafeGPT returned 500 `string_above_max_length` (12,868,788 chars, limit 10,485,760) although the proxy sent only ~110 KB. **Cause, verified live:** SafeGPT fetches every `scheme://` URL in the prompt and inlines the page; `https://pypi.org/simple` from a `pyproject.toml` read did it. Bare domains are not fetched. Fix in `safegpt_client.py`: `defang_urls()` puts a word joiner (U+2060) after the scheme colon in every outgoing prompt/system message, and `restore_urls()` strips it from replies (verified live: the model writes the URL back clean). As a second safeguard each tool result is clipped to `MAX_TOOL_OUTPUT_CHARS` (200 K, head plus tail, with a note asking for a smaller range) and `fit_turns()` drops the oldest turns after the task when the transcript exceeds `MAX_TRANSCRIPT_CHARS` (8 M). Both log a warning, so the next log shows which tool produced it.
7. Commit `fd8314e` holds the tool-emulation work. Items 1-6 are uncommitted, as is the merge of the useful `laptop` branch parts (strict Responses `usage` with `*_details` for JetBrains/koog, FastAPI `lifespan` instead of `on_event`, 502 on Execute failures in the orchestration path, `scripts/diagnose_safegpt.py`). Ask the user before committing.

## How to debug with the user

- The user pastes proxy logs (they appear under `C:\Users\ahoek\.t3\userdata\attachments\*.txt`). Useful log lines: `Tool-emulation request`, `SafeGPT raw reply`, `Empty SafeGPT reply`, `Reply has no tool call`, `SafeGPT follow-up reply`, `Emitting output items`.
- To replay a logged request live: extract the `INFO:proxy:Body: {...}` JSON blocks from the log with a regex, then POST them through `TestClient` with a real `SafeGPTClient` (use `with TestClient(app)` / `__enter__()`; otherwise each request gets a new event loop and the shared httpx client fails with "Event loop is closed"). Scratch scripts doing this are in `%TEMP%` (`live_cline.py`, `raw_followup.py`, `probe_len.py`, `probe_filter.py`). Replays only call SafeGPT; nothing is executed on disk.
- `%TEMP%\cline-src` is a sparse clone of the Cline source (used to confirm it uses native tool calls via `@ai-sdk/openai-compatible` → `/chat/completions`). It and the scratch scripts can be deleted.

## User notes

- The SafeGPT token appeared in full in an early pasted log before masking was added (`main.py` now logs `***` plus the last 4 characters); the user was advised to rotate it.
- The user prefers explanations with a recommendation and wants tool calling kept (Aider was rejected for lacking it).
