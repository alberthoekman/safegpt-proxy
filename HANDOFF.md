# Handoff: tool calling through the SafeGPT proxy

Last updated: 2026-09-24. Written so a new session can continue without the old chat.

## Goal

The user must use SafeGPT (https://api.safegpt.nl) and wants coding agents (first Codex, now **Cline**) to use it with real tool calling. SafeGPT has **no native function calling**: `POST /v1/Message/Execute` takes only `{systemMessage, prompt}` and returns `{"content": "..."}`. It ignores any model name and `reasoning_effort`. So this proxy **emulates** OpenAI tool calling on top of plain text.

## Current state

- **Nothing is committed.** Changed or new: `app/tools.py` (new), `app/router.py`, `app/session.py`, `app/safegpt_client.py`, `main.py`, `README.md`, `tests/test_tool_emulation.py` (new), and this file. Branch: `main`.
- 25 tests pass: `uv run --extra dev pytest -q tests`. Lint the changed files with `uv run --extra dev ruff check app/tools.py app/router.py tests` (clean). The remaining ruff findings in `app/settings.py` and `app/safegpt_client.py` are old imports; leave them.
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

## Open issues / next steps

1. **Wrong `FINAL` answers remain (~1 in 5).** Idea: if the draft reads as future intent ("I'll", "I need to", "Plan:", "next") and the follow-up says FINAL, ask once more, or ignore FINAL in that case. Keep it simple and test live.
2. **Unanswered first request** in the user's last log (`pasted-text-4`): the first `/v1/chat/completions` logged "Emitting output items" but no `200 OK` access line, and Cline then sent "[TASK RESUMPTION]". This was not yet asked: did the user cancel, or did Cline time out or error? For Chat Completions streaming, the proxy computes the whole reply **before** starting the stream (no keep-alive, unlike the Responses path). If Cline has a first-byte timeout, move chat streaming to the same pattern as `stream_response_items` (send the role chunk immediately, then keep-alives, then the content).
3. **Latency:** retries and follow-ups can mean several SafeGPT calls (~2 s each, up to ~10–20 s per step) with nothing shown in Cline meanwhile.
4. Nothing is committed yet; ask the user before committing.

## How to debug with the user

- The user pastes proxy logs (they appear under `C:\Users\ahoek\.t3\userdata\attachments\*.txt`). Useful log lines: `Tool-emulation request`, `SafeGPT raw reply`, `Empty SafeGPT reply`, `Reply has no tool call`, `SafeGPT follow-up reply`, `Emitting output items`.
- To replay a logged request live: extract the `INFO:proxy:Body: {...}` JSON blocks from the log with a regex, then POST them through `TestClient` with a real `SafeGPTClient` (use `with TestClient(app)` / `__enter__()`; otherwise each request gets a new event loop and the shared httpx client fails with "Event loop is closed"). Scratch scripts doing this are in `%TEMP%` (`live_cline.py`, `raw_followup.py`, `probe_len.py`, `probe_filter.py`). Replays only call SafeGPT; nothing is executed on disk.
- `%TEMP%\cline-src` is a sparse clone of the Cline source (used to confirm it uses native tool calls via `@ai-sdk/openai-compatible` → `/chat/completions`). It and the scratch scripts can be deleted.

## User notes

- The SafeGPT token appeared in full in an early pasted log before masking was added (`main.py` now logs `***` plus the last 4 characters); the user was advised to rotate it.
- The user prefers explanations with a recommendation and wants tool calling kept (Aider was rejected for lacking it).
