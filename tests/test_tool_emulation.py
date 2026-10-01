import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import router as router_module
from app import tools as toolemu

SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Runs a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]},
}
APPLY_PATCH_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a patch",
    "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "*** Begin Patch" LF'},
}


class FakeSafeGPT:
    """Returns `reply` for normal calls; answers the proxy's no-tool-call follow-up with FINAL."""

    def __init__(self, reply: str, followup_reply: str = "FINAL"):
        self.reply = reply
        self.followup_reply = followup_reply
        self.calls = []
        self.followups = []

    async def execute(self, system_message: str, prompt: str, model: str) -> str:
        if "That reply contains no tool call" in prompt or "Your previous reply was empty" in prompt:
            self.followups.append(prompt)
            return self.followup_reply
        self.calls.append({"system": system_message, "prompt": prompt})
        return self.reply


@pytest.fixture
def make_client():
    def _make(reply: str, followup_reply: str = "FINAL"):
        fake = FakeSafeGPT(reply, followup_reply)
        router_module.set_dependencies(fake, "gpt-5.6-sol", "", "")
        app = FastAPI()
        app.include_router(router_module.router)
        return TestClient(app), fake

    return _make


def sse_events(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[len("data: "):] for line in block.splitlines() if line.startswith("data: ")]
        if data and data[0] != "[DONE]":
            events.append(json.loads(data[0]))
    return events


def test_parse_reply_with_text_and_calls():
    reply = toolemu.parse_reply(
        'Let me look.\n<tool_call name="shell">\n```json\n{"command": ["ls"]}\n```\n</tool_call>\n'
        '<tool_call name="apply_patch">\n*** Begin Patch\n*** End Patch\n</tool_call>'
    )
    assert reply.text == "Let me look."
    assert reply.calls == [("shell", '```json\n{"command": ["ls"]}\n```'), ("apply_patch", "*** Begin Patch\n*** End Patch")]


def test_parse_reply_tolerates_json_style_block():
    reply = toolemu.parse_reply('<tool_call>{"name": "shell", "arguments": {"command": ["pwd"]}}</tool_call>')
    assert reply.calls == [("shell", '{"command": ["pwd"]}')]


def test_transcript_renders_previous_calls_and_results():
    system_parts, transcript = toolemu.render_transcript([
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "be terse"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        {"type": "reasoning", "encrypted_content": "xyz"},
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": '{"command":["ls"]}'},
        {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
    ])
    assert system_parts == ["be terse"]
    assert '<tool_call name="shell">\n{"command":["ls"]}\n</tool_call>' in transcript
    assert "[tool result: shell (call_id=c1)]\na.txt" in transcript
    assert "xyz" not in transcript


def test_responses_function_and_custom_calls_streamed(make_client):
    client, fake = make_client(
        'Checking.\n<tool_call name="shell">\n{"command": ["ls", "-la"]}\n</tool_call>\n'
        '<tool_call name="apply_patch">\n*** Begin Patch\n*** End Patch\n</tool_call>'
    )
    resp = client.post("/v1/responses", json={
        "model": "gpt-5.6-sol",
        "instructions": "You are Codex.",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "tools": [SHELL_TOOL, APPLY_PATCH_TOOL, {"type": "web_search"}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "stream": True,
    })
    assert resp.status_code == 200
    events = sse_events(resp.text)
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    done_items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert [i["type"] for i in done_items] == ["message", "function_call", "custom_tool_call"]
    assert json.loads(done_items[1]["arguments"]) == {"command": ["ls", "-la"]}
    assert done_items[2]["input"] == "*** Begin Patch\n*** End Patch"
    assert done_items[1]["call_id"].startswith("call_")
    assert [e["sequence_number"] for e in events] == list(range(len(events)))

    system = fake.calls[0]["system"]
    assert system.startswith("# Tool use") and "You are Codex." in system
    assert "### shell (JSON)" in system and "### apply_patch (freeform)" in system
    assert "web_search" not in system


def test_responses_plain_text_and_previous_response_id(make_client):
    client, fake = make_client("All done.")
    first = client.post("/responses", json={
        "instructions": "x",
        "input": "hello",
        "tools": [SHELL_TOOL],
    }).json()
    assert first["output"][0]["type"] == "message"
    assert first["output_text"] == "All done."

    client.post("/responses", json={"previous_response_id": first["id"], "input": "again", "tools": [SHELL_TOOL]})
    assert "[user]\nhello" in fake.calls[1]["prompt"]
    assert "[assistant]\nAll done." in fake.calls[1]["prompt"]

    missing = client.post("/responses", json={"previous_response_id": "resp_nope", "input": "x"})
    assert missing.status_code == 404


def test_parallel_disabled_keeps_first_call(make_client):
    client, _ = make_client(
        '<tool_call name="shell">\n{"command": ["a"]}\n</tool_call>\n<tool_call name="shell">\n{"command": ["b"]}\n</tool_call>'
    )
    out = client.post("/responses", json={"input": "x", "tools": [SHELL_TOOL], "parallel_tool_calls": False}).json()
    assert len(out["output"]) == 1


def test_namespace_tools_round_trip(make_client):
    client, fake = make_client('<tool_call name="github.search">\n{"q": "x"}\n</tool_call>')
    out = client.post("/responses", json={
        "input": "x",
        "tools": [{"type": "namespace", "name": "github", "description": "", "tools": [
            {"type": "function", "name": "search", "parameters": {"type": "object"}},
        ]}],
    }).json()
    item = out["output"][0]
    assert (item["type"], item["name"], item["namespace"]) == ("function_call", "search", "github")
    assert "### github.search (JSON)" in fake.calls[0]["system"]


def test_chat_completions_tool_calls(make_client):
    client, fake = make_client('<tool_call name="shell">\n{"command": ["ls"]}\n</tool_call>')
    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "list"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "shell", "arguments": '{"command":["pwd"]}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "/home"},
        ],
        "tools": [{"type": "function", "function": {k: v for k, v in SHELL_TOOL.items() if k != "type"}}],
    }
    out = client.post("/v1/chat/completions", json=body).json()
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "shell"
    assert "[tool result: shell (call_id=t1)]\n/home" in fake.calls[0]["prompt"]

    streamed = client.post("/v1/chat/completions", json={**body, "stream": True})
    chunks = sse_events(streamed.text)
    assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks if c["choices"])
    assert streamed.text.strip().endswith("data: [DONE]")


def test_cline_streamed_tool_call_shape(make_client):
    """Shape of a real Cline (AI SDK openai-compatible) round trip that Cline accepted."""
    client, fake = make_client('<tool_call name="read_files">\n{"files": [{"path": "C:\\\\repo\\\\README.md"}]}\n</tool_call>')
    read_files = {"type": "function", "function": {
        "name": "read_files",
        "description": "Read the content of text or image files at the provided absolute paths.",
        "parameters": {"type": "object", "properties": {"files": {"type": "array"}}, "required": ["files"], "additionalProperties": False},
    }}
    body = {
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "You are Cline, an AI coding agent."},
            {"role": "user", "content": "update the readme"},
        ],
        "tools": [read_files],
        "tool_choice": "auto",
        "reasoning_effort": "high",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunks = sse_events(client.post("/v1/chat/completions", json=body).text)

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    call = next(c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"])
    assert call["index"] == 0 and call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "read_files"
    assert json.loads(call["function"]["arguments"]) == {"files": [{"path": "C:\\repo\\README.md"}]}
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 0
    assert fake.calls[0]["system"].startswith("# Tool use") and "You are Cline" in fake.calls[0]["system"]


def test_codex_responses_lite_additional_tools(make_client):
    """Codex 'responses lite' sends tools as an input item and no top-level instructions/tools."""
    js = 'const r = await tools.exec_command({ cmd: "dir" });\ntext(r.output);'
    client, fake = make_client(f'<tool_call name="exec">\n{js}\n</tool_call>')
    body = {
        "model": "gpt-6-astra",
        "input": [
            {"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": [
                    {"type": "custom", "name": "exec", "description": "Run JavaScript", "format": {"type": "grammar", "syntax": "lark", "definition": "start: SOURCE"}},
                    {"type": "function", "name": "wait", "parameters": {"type": "object"}},
                ]},
                {"type": "namespace", "name": "clock", "description": "", "tools": [
                    {"type": "function", "name": "sleep", "parameters": {"type": "object"}},
                ]},
            ]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "make the files"}]},
            {"type": "custom_tool_call", "call_id": "c0", "namespace": "functions", "name": "exec", "input": "text(1)"},
            {"type": "custom_tool_call_output", "call_id": "c0", "output": [{"type": "input_text", "text": "1"}]},
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "stream": True,
    }
    events = sse_events(client.post("/responses", json=body).text)
    item = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
    assert item["type"] == "custom_tool_call"
    assert item["name"] == "exec" and "namespace" not in item
    assert item["input"] == js

    system, prompt = fake.calls[0]["system"], fake.calls[0]["prompt"]
    assert system.startswith("# Tool use") and "You are Codex." in system
    assert "### exec (freeform)" in system and "### clock.sleep (JSON)" in system
    assert "functions.exec" not in system
    assert '<tool_call name="exec">\ntext(1)\n</tool_call>' in prompt
    assert "[tool result: exec (call_id=c0)]\n1" in prompt


CODE_MODE_EXEC = {
    "type": "custom",
    "name": "exec",
    "description": (
        "Run JavaScript code to orchestrate/compose tool calls\n- Runs raw JavaScript.\n\n"
        "### `apply_patch`\nEdit files. This is a FREEFORM tool.\n\nexec tool declaration:\n```ts\n"
        "declare const tools: { apply_patch(input: string): Promise<unknown>; };\n```\n\n"
        "### `exec_command`\nRuns a command.\n\nexec tool declaration:\n```ts\n"
        "declare const tools: { exec_command(args: {\n  // Shell command to execute.\n  cmd: string;\n}): Promise<{ output: string }>; };\n```\n"
    ),
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: SOURCE"},
}


def code_mode_body(extra_input=()):
    return {
        "input": [
            {"type": "additional_tools", "role": "developer", "tools": [
                {"type": "namespace", "name": "functions", "description": "", "tools": [CODE_MODE_EXEC]},
            ]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "make the files"}]},
            *extra_input,
        ],
        "parallel_tool_calls": False,
    }


def test_code_mode_nested_tools_are_exposed_and_wrapped(make_client):
    client, fake = make_client('<tool_call name="exec_command">\n{"cmd": "dir"}\n</tool_call>')
    item = client.post("/responses", json=code_mode_body()).json()["output"][0]
    assert (item["type"], item["name"]) == ("custom_tool_call", "exec")
    assert item["input"].startswith('const result = await tools.exec_command({"cmd": "dir"});')

    system = fake.calls[0]["system"]
    assert "### exec_command (JSON)" in system and "cmd: string;" in system
    assert "### apply_patch (freeform)" in system
    assert "### exec (" not in system  # hidden: models write unsupported Node scripts for it


def test_code_mode_freeform_nested_tool_and_history_unwrapping(make_client):
    patch = '*** Begin Patch\n*** Add File: a.py\n+print("hi")\n*** End Patch'
    client, fake = make_client(f'<tool_call name="apply_patch">\n{patch}\n</tool_call>')
    item = client.post("/responses", json=code_mode_body()).json()["output"][0]
    assert item["input"].startswith("const result = await tools.apply_patch(" + json.dumps(patch) + ");")

    history = [
        {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": item["input"]},
        {"type": "custom_tool_call_output", "call_id": "c1", "output": "Done!"},
    ]
    client.post("/responses", json=code_mode_body(history))
    prompt = fake.calls[1]["prompt"]
    assert f'<tool_call name="apply_patch">\n{patch}\n</tool_call>' in prompt
    assert "[tool result: apply_patch (call_id=c1)]\nDone!" in prompt


def test_parse_reply_accepts_unclosed_final_block():
    reply = toolemu.parse_reply('Creating it.<tool_call name="exec_command">\n{"cmd": "dir"}')
    assert reply.text == "Creating it."
    assert reply.calls == [("exec_command", '{"cmd": "dir"}')]


def test_plan_without_tool_call_gets_followed_up(make_client):
    plan = "Plan:\n1) Re-scan the repo.\n2) Read the README.\n3) Edit it."
    client, fake = make_client(plan, followup_reply='<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>')
    out = client.post("/responses", json={"input": "edit the readme", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["message", "function_call"]
    assert out[0]["content"][0]["text"] == plan
    assert f"[assistant]\n{plan}" in fake.followups[0]


def test_followup_restates_real_user_request(make_client):
    client, fake = make_client("I'm ready to continue, but I need to make the edit now.")
    client.post("/v1/chat/completions", json={"tools": [CLINE_RUN_COMMANDS], "messages": [
        {"role": "user", "content": "update the readme: add 'moi' at the top"},
        {"role": "user", "content": "[TASK RESUMPTION] Please continue where you left off."},
    ]})
    assert 'The user asked: "update the readme: add \'moi\' at the top"' in fake.followups[0]
    assert "that step must be a tool call now" in fake.followups[0]


def test_final_answer_confirmed_by_model_is_kept(make_client):
    answer = "The API is in app.py; run it with `python app.py`."
    client, fake = make_client(answer)  # follow-up answers FINAL
    out = client.post("/responses", json={"input": "thanks", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["message"] and out[0]["content"][0]["text"] == answer
    assert len(fake.calls) == 1 and len(fake.followups) == 1


def test_final_after_announced_step_is_challenged(make_client):
    draft = "I'll prepend `moi` to the README, then verify."
    call = '<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>'
    client, fake = make_client(draft)
    replies = iter(["FINAL", call])

    async def execute(system_message, prompt, model):
        if "That reply contains no tool call" in prompt:
            fake.followups.append(prompt)
            return next(replies)
        return draft

    fake.execute = execute
    out = client.post("/responses", json={"input": "add moi to the readme", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["message", "function_call"]
    assert "FINAL is not a valid answer now" not in fake.followups[0]
    assert "FINAL is not a valid answer now" in fake.followups[1]


def test_repeated_done_answer_is_accepted_after_one_followup(make_client):
    # Live log: the finished task was restated instead of FINAL, costing two extra calls.
    done = "Done — I added `moi` to the top of `README.md`."
    client, fake = make_client(done, followup_reply="Done — `moi` was added to the top of `README.md`.")
    msg = cline_call(client, [{"role": "user", "content": "add 'moi' at the top of the readme"}], [CLINE_APPLY_PATCH])
    assert msg["content"] == done and "tool_calls" not in msg
    assert len(fake.followups) == 1


def test_claimed_done_without_edit_still_gets_the_call(make_client):
    # Live log: the model claimed the edit before making it; the follow-up produced the patch.
    patch = "*** Begin Patch\n*** Update File: README.md\n@@\n+# moi\n+\n # Title\n*** End Patch"
    client, fake = make_client("I’ve added “moi” at the top of the README.", followup_reply=f'<tool_call name="apply_patch">\n{patch}\n</tool_call>')
    msg = cline_call(client, [{"role": "user", "content": "add 'moi' at the top of the readme"}], [CLINE_APPLY_PATCH])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"input": patch}
    assert len(fake.followups) == 1


def test_announced_step_without_call_is_nudged_twice(make_client):
    client, fake = make_client("I'll edit the README next.", followup_reply="Sure, editing it now.")
    cline_call(client, [{"role": "user", "content": "edit the readme"}], [CLINE_APPLY_PATCH])
    assert len(fake.followups) == 2


def test_short_mcp_tool_name_is_resolved_to_prefixed_tool(make_client):
    # Live Cline run: the model called "search_text" for "jetbrains-pycharm__search_text".
    search_text = {"type": "function", "function": {"name": "jetbrains-pycharm__search_text", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
    client, _ = make_client('<tool_call name="search_text">\n{"q": "hello"}\n</tool_call>')
    msg = cline_call(client, [{"role": "user", "content": "where is hello used?"}], [search_text, CLINE_RUN_COMMANDS])
    assert msg["tool_calls"][0]["function"]["name"] == "jetbrains-pycharm__search_text"


def test_ambiguous_short_tool_name_is_not_guessed():
    specs = toolemu.normalize_tools([
        {"type": "function", "name": "a__search", "parameters": {"type": "object"}},
        {"type": "function", "name": "b__search", "parameters": {"type": "object"}},
    ])
    assert toolemu.resolve_tool("search", specs) is None
    assert toolemu.resolve_tool("B__SEARCH", specs).name == "b__search"


def test_protocol_allows_requested_deletes_and_asks_on_vague_requests():
    protocol = toolemu.build_tool_protocol(toolemu.normalize_tools([SHELL_TOOL]), toolemu.parse_tool_choice(None), True)
    assert "Deleting a file is fine when the user asks for it" in protocol
    assert "instead of guessing" in protocol and "including any prefix" in protocol


def test_announces_next_step():
    assert toolemu.announces_next_step("I'll prepend it now.")
    assert toolemu.announces_next_step("Done reading.\nPlan:\n1) edit")
    assert toolemu.announces_next_step("Next, I update the file.")
    assert not toolemu.announces_next_step("The API is in app.py; run it with `python app.py`.")
    # Typographic apostrophes and progressive forms from live logs.
    assert toolemu.announces_next_step("Got it — I’ll limit the inspection to this folder and continue.")
    assert toolemu.announces_next_step("I’m locating the existing helpers so I can generalize this.")
    assert not toolemu.announces_next_step("Added the test. Let me know if you want more cases.")


def test_huge_tool_result_is_clipped():
    huge = "A" * 300_000 + "MIDDLE" + "Z" * 300_000
    items = [
        {"type": "message", "role": "user", "content": "read uv.lock"},
        {"type": "function_call", "call_id": "c1", "name": "read_files", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": huge},
    ]
    _, transcript = toolemu.render_transcript(items)
    assert len(transcript) < toolemu.MAX_TOOL_OUTPUT_CHARS + 1_000
    assert "characters omitted by the proxy" in transcript
    assert "MIDDLE" not in transcript
    assert transcript.rstrip().endswith("Z")


def test_small_tool_result_is_not_clipped():
    _, transcript = toolemu.render_transcript([
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
    ])
    assert "characters omitted" not in transcript
    assert "[tool result: shell (call_id=c1)]\na.txt" in transcript


def test_responses_usage_includes_token_details(make_client):
    # JetBrains' koog client rejects a usage object without the *_details fields.
    client, _ = make_client('<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>')
    usage = client.post("/responses", json={"input": "list", "tools": [SHELL_TOOL]}).json()["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}


def test_orchestration_upstream_error_returns_502(make_client):
    import httpx

    client, fake = make_client("")

    async def boom(**_):
        raise httpx.ConnectError("down")

    fake.execute = boom
    resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Suggest a most specific title for this chat"}]})
    assert resp.status_code == 502
    assert "down" in resp.json()["error"]["message"]


def test_urls_are_defanged_for_safegpt_and_restored_in_replies():
    from app.safegpt_client import defang_urls, restore_urls

    text = 'url = "https://pypi.org/simple"\nsee http://x.io and file:///C:/a; ratio 1:2, C:/Projects'
    out = defang_urls(text)
    assert "://" not in out
    assert "ratio 1:2, C:/Projects" in out
    assert restore_urls(out) == text


def test_transcript_over_limit_drops_oldest_turns_but_keeps_task():
    turns = ["[user]\ntask"] + [f"[tool result: t{i}]\n" + "x" * 100 for i in range(10)]
    fitted = toolemu.fit_turns(turns, limit=500)
    assert fitted[0] == "[user]\ntask"
    assert "earlier turns omitted" in fitted[1]
    assert fitted[-1] == turns[-1]
    assert sum(len(t) + 2 for t in fitted) < 700
    assert toolemu.fit_turns(turns[:2], limit=500) == turns[:2]


def test_chat_stream_sends_role_chunk_and_keepalives_before_reply(make_client, monkeypatch):
    import asyncio

    monkeypatch.setattr(router_module, "KEEPALIVE_SECONDS", 0.01)
    client, fake = make_client("")

    async def slow(system_message, prompt, model):
        await asyncio.sleep(0.05)
        return '<tool_call name="run_commands">\n{"commands": ["dir"]}\n</tool_call>'

    fake.execute = slow
    text = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "list"}], "tools": [CLINE_RUN_COMMANDS], "stream": True,
    }).text
    assert text.startswith("data: ") and ": keep-alive\n\n" in text
    chunks = sse_events(text)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert sum(1 for c in chunks if c["choices"][0]["delta"].get("role")) == 1
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert text.strip().endswith("data: [DONE]")


def test_chat_stream_reports_upstream_error(make_client):
    import httpx

    client, fake = make_client("")

    async def boom(**_):
        raise httpx.ConnectError("down")

    fake.execute = boom
    text = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "tools": [CLINE_RUN_COMMANDS], "stream": True}).text
    events = sse_events(text)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "down" in events[-1]["error"]["message"]
    assert text.strip().endswith("data: [DONE]")


def test_empty_reply_is_retried(make_client):
    client, fake = make_client("", followup_reply='<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>')
    out = client.post("/responses", json={"input": "edit the readme", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["function_call"]
    assert "Your previous reply was empty" in fake.followups[0]


def test_empty_reply_resends_identical_request_first(make_client):
    client, fake = make_client("")
    replies = iter(["", '<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>'])
    prompts = []

    async def execute(system_message, prompt, model):
        prompts.append(prompt)
        return next(replies)

    fake.execute = execute
    out = client.post("/responses", json={"input": "edit the readme", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["function_call"]
    assert len(prompts) == 2 and prompts[0] == prompts[1]


def test_repeated_empty_reply_asks_for_shorter_reply(make_client):
    client, fake = make_client("")
    replies = iter(["", "", '<tool_call name="shell">\n{"command": ["dir"]}\n</tool_call>'])
    prompts = []

    async def execute(system_message, prompt, model):
        prompts.append(prompt)
        return next(replies)

    fake.execute = execute
    out = client.post("/responses", json={"input": "rewrite the readme", "tools": [SHELL_TOOL]}).json()["output"]
    assert [i["type"] for i in out] == ["function_call"]
    assert prompts[0] == prompts[1]
    assert prompts[2].startswith(prompts[0]) and "produced no output, most likely because it was too long" in prompts[2]


def test_no_followup_without_tools(make_client):
    client, fake = make_client("Hello!")
    client.post("/responses", json={"instructions": "x", "input": "hi"})
    assert fake.followups == []


CLINE_APPLY_PATCH = {"type": "function", "function": {
    "name": "apply_patch",
    "description": "Use `apply_patch` to edit files. Pass the patch text directly as the `input` string.",
    "parameters": {"type": "object", "properties": {"input": {"type": "string", "minLength": 1, "description": "The freeform apply_patch payload."}}, "required": ["input"], "additionalProperties": False},
}}
CLINE_RUN_COMMANDS = {"type": "function", "function": {
    "name": "run_commands",
    "parameters": {"type": "object", "properties": {"commands": {"type": "array", "items": {"type": "string"}}}, "required": ["commands"]},
}}


def cline_call(client, messages, tools):
    out = client.post("/v1/chat/completions", json={"messages": messages, "tools": tools}).json()
    return out["choices"][0]["message"]


def test_single_string_tool_is_freeform_and_json_encoded_by_proxy(make_client):
    patch = '*** Begin Patch\n*** Add File: README.md\n+# Title\n+Say "hi"\n*** End Patch'
    client, fake = make_client(f'<tool_call name="apply_patch">\n{patch}\n</tool_call>')
    msg = cline_call(client, [{"role": "user", "content": "edit the readme"}], [CLINE_APPLY_PATCH])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"input": patch}
    system = fake.calls[0]["system"]
    assert "### apply_patch (freeform)" in system and "raw value of the `input` string" in system

    history = [
        {"role": "user", "content": "edit the readme"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "apply_patch", "arguments": json.dumps({"input": patch})}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "Successfully applied patch"},
    ]
    cline_call(client, history, [CLINE_APPLY_PATCH])
    assert f'<tool_call name="apply_patch">\n{patch}\n</tool_call>' in fake.calls[1]["prompt"]


def test_single_string_tool_accepts_json_anyway(make_client):
    client, _ = make_client('<tool_call name="apply_patch">\n{"input": "*** Begin Patch\\n*** End Patch"}\n</tool_call>')
    msg = cline_call(client, [{"role": "user", "content": "x"}], [CLINE_APPLY_PATCH])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"input": "*** Begin Patch\n*** End Patch"}


def test_json_args_repaired_from_logged_failures(make_client):
    # Unclosed call followed by FINAL (seen live), and a raw newline inside a JSON string.
    client, _ = make_client('<tool_call name="run_commands">\n{"commands":["dir"]}FINAL')
    msg = cline_call(client, [{"role": "user", "content": "x"}], [CLINE_RUN_COMMANDS])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"commands": ["dir"]}

    client, _ = make_client('<tool_call name="run_commands">\n{"commands":["echo a\necho b"]}\n</tool_call>')
    msg = cline_call(client, [{"role": "user", "content": "x"}], [CLINE_RUN_COMMANDS])
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"commands": ["echo a\necho b"]}


def test_stream_emits_response_failed_on_upstream_error(make_client):
    import httpx

    client, fake = make_client("")

    async def boom(**_):
        raise httpx.ConnectError("down")

    fake.execute = boom
    events = sse_events(client.post("/responses", json={"input": "x", "tools": [SHELL_TOOL], "stream": True}).text)
    assert events[-1]["type"] == "response.failed"
    assert "down" in events[-1]["response"]["error"]["message"]


def test_upstream_error_returns_502(make_client):
    import httpx

    client, fake = make_client("")

    async def boom(**_):
        raise httpx.ConnectError("down")

    fake.execute = boom
    resp = client.post("/responses", json={"input": "x", "tools": [SHELL_TOOL]})
    assert resp.status_code == 502
