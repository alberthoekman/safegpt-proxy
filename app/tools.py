"""Prompt-based emulation of OpenAI tool calling on top of SafeGPT.

SafeGPT only accepts plain prompts (`Message/Execute` takes a system message and a
prompt), so client-side tools (Codex `shell`, `apply_patch`, `update_plan`, MCP tools...)
are described in the system message and the model is asked to answer with
`<tool_call name="...">` blocks. Those blocks are parsed back into Responses API output
items (`function_call`, `custom_tool_call`, `local_shell_call`, ...).
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("proxy.tools")

# Tool types the client executes itself. Everything else (web_search, file_search,
# code_interpreter, mcp, image_generation, ...) is hosted by OpenAI and cannot be emulated.
CLIENT_TOOL_TYPES = {"function", "custom", "namespace", "local_shell", "shell", "apply_patch"}

# Codex wraps its top-level tools in this namespace and expects calls without it.
DEFAULT_NAMESPACE = "functions"

TOOL_CALL_RE = re.compile(
    r"<tool_call(?P<attrs>[^>]*)>\r?\n?(?P<body>.*?)\r?\n?</tool_call>",
    re.DOTALL,
)
OPEN_TAG_RE = re.compile(r'<tool_call\s+name="(?P<name>[^"]+)"[^>]*>')
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
FENCE_RE = re.compile(r"^\s*```[\w+-]*\r?\n(?P<inner>.*?)\r?\n```\s*$", re.DOTALL)

SHELL_PARAMETERS = {
    "type": "object",
    "properties": {
        "command": {"type": "array", "items": {"type": "string"}, "description": "argv of the command to run"},
        "workdir": {"type": "string"},
        "timeout_ms": {"type": "integer"},
    },
    "required": ["command"],
}
SHELL_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "commands": {"type": "array", "items": {"type": "string"}, "description": "shell commands to run in order"},
        "timeout_ms": {"type": "integer"},
    },
    "required": ["commands"],
}
APPLY_PATCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["create_file", "update_file", "delete_file"]},
        "path": {"type": "string"},
        "diff": {"type": "string", "description": "V4A diff; omit for delete_file"},
    },
    "required": ["type", "path"],
}


@dataclass
class ToolSpec:
    name: str  # name the model uses in <tool_call name="...">
    kind: str  # function | custom | local_shell | shell | apply_patch | nested
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None
    format: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    api_name: str = ""  # name reported back to the client
    declaration: str = ""  # TypeScript declaration of a nested code-mode tool
    string_input: bool = False  # nested tool taking a raw string (e.g. apply_patch)
    hidden: bool = False  # not advertised to the model, but calls to it are still accepted
    wrap_param: Optional[str] = None  # function tool whose only parameter is this string

    @property
    def freeform(self) -> bool:
        return self.kind == "custom" or (self.kind == "nested" and self.string_input) or self.wrap_param is not None


@dataclass
class ToolChoice:
    mode: str = "auto"  # auto | none | required
    forced: Optional[str] = None
    allowed: Optional[List[str]] = None


@dataclass
class ParsedReply:
    text: str
    calls: List[Tuple[str, str]] = field(default_factory=list)  # (tool name, raw body)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# --- tool definitions -------------------------------------------------------------


def normalize_tools(tools: Any) -> List[ToolSpec]:
    specs: List[ToolSpec] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        ttype = tool.get("type")
        if ttype == "namespace":
            ns = tool.get("name", "")
            if ns == DEFAULT_NAMESPACE:
                ns = None
            for inner in tool.get("tools") or []:
                spec = _tool_spec(inner, namespace=ns)
                if spec:
                    specs.append(spec)
            continue
        spec = _tool_spec(tool)
        if spec:
            specs.append(spec)
        elif ttype:
            logger.info("Ignoring hosted tool type %r (not available through SafeGPT)", ttype)
    return _expand_code_mode(specs)


# --- Codex "code mode" ------------------------------------------------------------
#
# Codex can expose a single freeform `exec` tool that runs JavaScript, with the real tools
# (exec_command, apply_patch, ...) only callable from that script as `await tools.x(...)`.
# Models behind SafeGPT reliably call tools directly but not via generated JavaScript, so
# the nested tools are offered as first-class tools and their calls are wrapped into `exec`.

CODE_MODE_TOOL = "exec"
NESTED_SECTION_RE = re.compile(r"^### `(?P<name>[A-Za-z_$][\w$]*)`\s*\n(?P<body>.*?)(?=^### `|\Z)", re.DOTALL | re.MULTILINE)
NESTED_DECL_RE = re.compile(r"exec tool declaration:\s*```ts\s*\n(?P<decl>.*?)\n```", re.DOTALL)
NESTED_JS_RE = re.compile(
    r"^const result = await tools\.(?P<name>[A-Za-z_$][\w$]*)\((?P<args>.*)\);\n"
    r"text\(typeof result === \"string\" \? result : JSON\.stringify\(result\)\);$",
    re.DOTALL,
)


def _expand_code_mode(specs: List[ToolSpec]) -> List[ToolSpec]:
    exec_spec = next((s for s in specs if s.name == CODE_MODE_TOOL and s.kind == "custom"), None)
    if exec_spec is None or "exec tool declaration:" not in exec_spec.description:
        return specs
    taken = {s.name for s in specs}
    nested: List[ToolSpec] = []
    for match in NESTED_SECTION_RE.finditer(exec_spec.description):
        name, body = match.group("name"), match.group("body")
        decl = NESTED_DECL_RE.search(body)
        if not decl or name in taken:
            continue
        declaration = decl.group("decl").strip()
        nested.append(ToolSpec(
            name=name,
            kind="nested",
            description=body[: decl.start()].strip(),
            namespace=exec_spec.namespace,
            api_name=exec_spec.api_name,
            declaration=declaration,
            string_input=bool(re.search(rf"{re.escape(name)}\(input: string\)", declaration)),
        ))
    if not nested:
        return specs
    # Offered alongside the nested tools, models write Node-style scripts that `exec` rejects.
    exec_spec.hidden = True
    return specs + nested


def nested_call_js(name: str, args: str) -> str:
    return f"const result = await tools.{name}({args});\ntext(typeof result === \"string\" ? result : JSON.stringify(result));"


def unwrap_nested_call(js: str) -> Optional[Tuple[str, str]]:
    """Reverse of nested_call_js: (tool name, tool_call body) for rendering history."""
    match = NESTED_JS_RE.match(js or "")
    if not match:
        return None
    args = match.group("args")
    try:
        value = json.loads(args)
    except ValueError:
        return match.group("name"), args
    return match.group("name"), value if isinstance(value, str) else args


def _single_string_param(params: Any) -> Optional[str]:
    """Name of the only (string) parameter, e.g. Cline's `apply_patch(input)`.

    Such tools are offered to the model as freeform so it writes the raw text and the
    proxy does the JSON encoding: models escape long patches inside JSON unreliably.
    """
    if not isinstance(params, dict) or params.get("type") != "object":
        return None
    props = params.get("properties") or {}
    if len(props) != 1:
        return None
    (name, prop), = props.items()
    return name if isinstance(prop, dict) and prop.get("type") == "string" else None


def _tool_spec(tool: Dict[str, Any], namespace: Optional[str] = None) -> Optional[ToolSpec]:
    ttype = tool.get("type")
    if ttype == "function":
        # Chat Completions nests the definition under "function".
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        api_name = fn.get("name", "")
        name = f"{namespace}.{api_name}" if namespace else api_name
        params = fn.get("parameters")
        return ToolSpec(name, "function", fn.get("description", ""), params, None, namespace, api_name, wrap_param=_single_string_param(params))
    if ttype == "custom":
        api_name = tool.get("name", "")
        name = f"{namespace}.{api_name}" if namespace else api_name
        return ToolSpec(name, "custom", tool.get("description", ""), None, tool.get("format"), namespace, api_name)
    if ttype == "local_shell":
        return ToolSpec("local_shell", "local_shell", "Run a command on the user's machine.", SHELL_PARAMETERS, api_name="local_shell")
    if ttype == "shell":
        return ToolSpec("shell", "shell", "Run shell commands on the user's machine.", SHELL_TOOL_PARAMETERS, api_name="shell")
    if ttype == "apply_patch":
        return ToolSpec(
            "apply_patch", "apply_patch", "Create, update or delete one file using a V4A diff.", APPLY_PATCH_PARAMETERS, api_name="apply_patch"
        )
    return None


def collect_tools(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-level `tools` plus tools sent as `additional_tools` input items.

    Codex in "responses lite" mode omits the top-level `tools` and `instructions` and
    instead starts `input` with an `additional_tools` item.
    """
    tools = [t for t in body.get("tools") or [] if isinstance(t, dict)]
    raw = body.get("input")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "additional_tools":
                tools.extend(t for t in item.get("tools") or [] if isinstance(t, dict))
    return tools


def has_client_tools(body: Dict[str, Any]) -> bool:
    return any(t.get("type") in CLIENT_TOOL_TYPES for t in collect_tools(body))


def parse_tool_choice(raw: Any) -> ToolChoice:
    if raw in (None, "auto"):
        return ToolChoice()
    if raw in ("none", "required"):
        return ToolChoice(mode=raw)
    if isinstance(raw, dict):
        ctype = raw.get("type")
        if ctype == "allowed_tools":
            names = [t.get("name") or (t.get("function") or {}).get("name") or t.get("type") for t in raw.get("tools") or []]
            return ToolChoice(mode=raw.get("mode", "auto"), allowed=[n for n in names if n])
        name = raw.get("name") or (raw.get("function") or {}).get("name")
        if ctype in ("function", "custom") and name:
            return ToolChoice(mode="required", forced=name)
        if ctype in ("local_shell", "shell", "apply_patch"):
            return ToolChoice(mode="required", forced=ctype)
    return ToolChoice()


def select_tools(specs: List[ToolSpec], choice: ToolChoice) -> List[ToolSpec]:
    if choice.mode == "none":
        return []
    if choice.forced:
        return [s for s in specs if choice.forced in (s.name, s.api_name)]
    if choice.allowed is not None:
        return [s for s in specs if s.name in choice.allowed or s.api_name in choice.allowed]
    return specs


# --- prompt rendering -------------------------------------------------------------


def build_tool_protocol(specs: List[ToolSpec], choice: ToolChoice, parallel: bool) -> str:
    if not specs:
        return ""
    lines = [
        "# Tool use",
        "",
        "You are running inside a client that executes tools for you. To call a tool, output a block of exactly this form:",
        "",
        '<tool_call name="TOOL_NAME">',
        "ARGUMENTS",
        "</tool_call>",
        "",
        "Rules:",
        "- For JSON tools, ARGUMENTS is one JSON object that matches the tool's parameter schema.",
        "- For freeform tools, ARGUMENTS is the raw input text, verbatim. Do not wrap it in JSON, quotes or code fences.",
        "- You may write a short message before your tool calls. Write nothing after the last </tool_call>; stop and wait.",
        "- A reply without a tool_call block ends your turn and hands control back to the user. Never announce an action "
        "(\"I'll create the files now\") without including the tool_call that performs it in the same reply.",
        "- Instructions below that mention calling tools, sending commentary or preambles, or tool-call channels all refer to this <tool_call> format.",
        "- Tool results are returned to you in the next turn as [tool result] entries. Never invent or predict tool results.",
        "- When the task is complete and no further tool is needed, answer with plain text and no tool_call blocks.",
        "- Only use the tools listed below, with the exact names shown.",
        "- Never delete a file in order to rewrite it; change existing files in place.",
        "- Keep each tool call small: split large file edits into several smaller edits, one section at a time.",
    ]
    if parallel:
        lines.append("- You may emit several tool_call blocks in one reply when the calls are independent.")
    else:
        lines.append("- Emit at most one tool_call block per reply.")
    if choice.forced:
        lines.append(f'- You MUST call the tool "{choice.forced}" in this reply.')
    elif choice.mode == "required":
        lines.append("- You MUST call at least one tool in this reply.")
    lines += ["", "## Available tools"]
    for spec in specs:
        if spec.hidden:
            continue
        lines.append("")
        if spec.freeform:
            lines.append(f"### {spec.name} (freeform)")
        else:
            lines.append(f"### {spec.name} (JSON)")
        if spec.description:
            lines.append(spec.description.strip())
        if spec.kind == "nested":
            if spec.string_input:
                lines.append("ARGUMENTS is the raw input string. TypeScript signature:")
            else:
                lines.append("ARGUMENTS is the JSON object `args`. TypeScript signature:")
            lines.append(spec.declaration)
        elif spec.wrap_param:
            prop = (spec.parameters or {}).get("properties", {}).get(spec.wrap_param, {})
            lines.append(f"ARGUMENTS is the raw value of the `{spec.wrap_param}` string, written verbatim (not JSON).")
            if prop.get("description"):
                lines.append(f"`{spec.wrap_param}`: {prop['description']}")
        elif spec.freeform:
            fmt = spec.format or {}
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                lines.append(f"The input must conform to this {fmt.get('syntax', 'lark')} grammar:")
                lines.append(fmt["definition"].strip())
        else:
            params = spec.parameters or {"type": "object", "properties": {}}
            lines.append("Parameters JSON schema: " + json.dumps(params, ensure_ascii=False))
    return "\n".join(lines)


def build_output_format_instructions(text_cfg: Any) -> str:
    fmt = (text_cfg or {}).get("format") if isinstance(text_cfg, dict) else None
    if not isinstance(fmt, dict):
        return ""
    if fmt.get("type") == "json_schema":
        schema = json.dumps(fmt.get("schema") or {}, ensure_ascii=False)
        return f"# Output format\n\nYour final answer must be only a JSON object (no prose, no code fences) that matches this schema:\n{schema}"
    if fmt.get("type") == "json_object":
        return "# Output format\n\nYour final answer must be only a valid JSON object (no prose, no code fences)."
    return ""


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("input_text", "output_text", "text", "refusal"):
                    parts.append(part.get("text") or part.get("refusal") or "")
                elif ptype in ("input_image", "image_url"):
                    parts.append("[image omitted]")
                elif ptype in ("input_file", "file"):
                    parts.append(f"[file omitted: {part.get('filename', '')}]")
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


def _render_call(name: str, body: str) -> str:
    return f'<tool_call name="{name}">\n{body}\n</tool_call>'


def _qualified(item: Dict[str, Any]) -> str:
    ns = item.get("namespace")
    return f"{ns}.{item.get('name', '')}" if ns and ns != DEFAULT_NAMESPACE else item.get("name", "")


def render_transcript(items: List[Dict[str, Any]], wrap_params: Optional[Dict[str, str]] = None) -> Tuple[List[str], str]:
    """Split Responses input items into (system/developer texts, conversation transcript).

    `wrap_params` maps single-string-parameter tools to that parameter, so their past
    calls are shown in the same raw form the model is asked to write.
    """
    wrap_params = wrap_params or {}
    system_parts: List[str] = []
    turns: List[str] = []
    call_names: Dict[str, str] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type") or ("message" if "role" in item else None)
        if itype == "message":
            role = item.get("role", "user")
            text = content_to_text(item.get("content"))
            if role in ("system", "developer"):
                if text.strip():
                    system_parts.append(text)
            elif text.strip():
                turns.append(f"[{role}]\n{text}")
        elif itype == "function_call":
            name, body = _qualified(item), item.get("arguments") or "{}"
            if name in wrap_params:
                args = _load_json_object(body)
                if args and isinstance(args.get(wrap_params[name]), str):
                    body = args[wrap_params[name]]
            call_names[item.get("call_id", "")] = name
            turns.append(f"[assistant]\n{_render_call(name, body)}")
        elif itype == "custom_tool_call":
            name, body = _qualified(item), item.get("input") or ""
            if name == CODE_MODE_TOOL:
                name, body = unwrap_nested_call(body) or (name, body)
            call_names[item.get("call_id", "")] = name
            turns.append(f"[assistant]\n{_render_call(name, body)}")
        elif itype == "local_shell_call":
            action = item.get("action") or {}
            args = {"command": action.get("command") or [], "workdir": action.get("working_directory"), "timeout_ms": action.get("timeout_ms")}
            args = {k: v for k, v in args.items() if v is not None}
            call_names[item.get("call_id", "")] = "local_shell"
            turns.append(f"[assistant]\n{_render_call('local_shell', json.dumps(args, ensure_ascii=False))}")
        elif itype == "shell_call":
            action = item.get("action") or {}
            args = {"commands": action.get("commands") or [], "timeout_ms": action.get("timeout_ms")}
            args = {k: v for k, v in args.items() if v is not None}
            call_names[item.get("call_id", "")] = "shell"
            turns.append(f"[assistant]\n{_render_call('shell', json.dumps(args, ensure_ascii=False))}")
        elif itype == "apply_patch_call":
            call_names[item.get("call_id", "")] = "apply_patch"
            turns.append(f"[assistant]\n{_render_call('apply_patch', json.dumps(item.get('operation') or {}, ensure_ascii=False))}")
        elif itype in ("function_call_output", "custom_tool_call_output", "shell_call_output", "apply_patch_call_output"):
            call_id = item.get("call_id", "")
            output = item.get("output")
            if itype == "shell_call_output" and isinstance(output, list):
                output = json.dumps(output, ensure_ascii=False)
            elif itype == "apply_patch_call_output":
                output = f"status: {item.get('status', '')}\n{item.get('output') or ''}".strip()
            else:
                output = content_to_text(output)
            turns.append(f"[tool result: {call_names.get(call_id, 'tool')} (call_id={call_id})]\n{output}")
        elif itype == "local_shell_call_output":
            call_id = item.get("call_id") or item.get("id", "")
            turns.append(f"[tool result: local_shell (call_id={call_id})]\n{content_to_text(item.get('output'))}")
        elif itype in ("reasoning", "additional_tools"):
            # Encrypted reasoning is meaningless to SafeGPT; tools go into the system message.
            continue
        else:
            logger.info("Skipping unsupported input item type %r", itype)
    return system_parts, "\n\n".join(turns)


def normalize_input(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = body.get("input")
    if isinstance(raw, str):
        return [{"type": "message", "role": "user", "content": raw}]
    if isinstance(raw, list):
        return [i for i in raw if isinstance(i, dict)]
    return []


def chat_messages_to_items(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Chat Completions `messages` into Responses-style input items."""
    items: List[Dict[str, Any]] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id", ""), "output": content_to_text(m.get("content"))})
            continue
        if m.get("content"):
            items.append({"type": "message", "role": role, "content": m.get("content")})
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            items.append({"type": "function_call", "call_id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"})
    return items


def build_prompts(
    items: List[Dict[str, Any]],
    instructions: Optional[str],
    specs: List[ToolSpec],
    choice: ToolChoice,
    parallel: bool,
    text_cfg: Any = None,
) -> Tuple[str, str]:
    """Return (system_message, prompt) for SafeGPT Message/Execute."""
    system_parts, transcript = render_transcript(items, {s.name: s.wrap_param for s in specs if s.wrap_param})
    # The tool protocol goes first: client instructions (Codex sends ~60 KB) would otherwise bury it.
    protocol = build_tool_protocol(specs, choice, parallel)
    sections = [s for s in [protocol, instructions or "", *system_parts] if s and s.strip()]
    fmt = build_output_format_instructions(text_cfg)
    if fmt:
        sections.append(fmt)
    system_message = "\n\n".join(sections).strip() or "You are a helpful assistant."
    prompt = (
        "Below is the conversation so far. Continue it by writing the next [assistant] reply "
        "(without the [assistant] label).\n\n" + transcript
    )
    if protocol:
        prompt += (
            "\n\n---\nReminder: if the task needs any action (reading or writing files, running commands, ...), "
            'your reply must contain the <tool_call name="..."> block that performs it. Do not just describe what you will do. '
            "Reply with plain text only when the task is complete or you need input from the user."
        )
    return system_message, prompt


# --- reply parsing ----------------------------------------------------------------


def parse_reply(text: str) -> ParsedReply:
    calls: List[Tuple[str, str]] = []
    first_start: Optional[int] = None
    for match in TOOL_CALL_RE.finditer(text or ""):
        attrs = dict(ATTR_RE.findall(match.group("attrs")))
        body = match.group("body")
        name = attrs.get("name")
        if not name:
            # Tolerate <tool_call>{"name": ..., "arguments": ...}</tool_call>
            try:
                obj = json.loads(_strip_fence(body))
            except ValueError:
                continue
            if not isinstance(obj, dict) or not obj.get("name"):
                continue
            name = obj["name"]
            args = obj.get("arguments", obj.get("input", {}))
            body = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        if first_start is None:
            first_start = match.start()
        calls.append((name, body))
    # Models sometimes stop without the closing tag; take the rest of the reply as the body.
    tail_start = max((m.end() for m in TOOL_CALL_RE.finditer(text or "")), default=0)
    unclosed = OPEN_TAG_RE.search(text or "", tail_start)
    if unclosed:
        body = text[unclosed.end():].strip("\r\n")
        body = re.sub(rf"\s*(?:</tool_call|{FINAL_MARKER})\s*$", "", body.rstrip())
        if first_start is None:
            first_start = unclosed.start()
        calls.append((unclosed.group("name"), body))
    if not calls:
        return ParsedReply(text=_strip_label(text or "").strip())
    return ParsedReply(text=_strip_label(text[:first_start]).strip(), calls=calls)


FINAL_MARKER = "FINAL"


def needs_followup(reply: ParsedReply, specs: List[ToolSpec]) -> bool:
    """A tool-enabled reply without a tool call, which would end the agent's turn.

    Models behind SafeGPT often stop after announcing or planning the next step ("I'll
    create the files now.", "Plan: 1) ..."), and sometimes return nothing at all, so
    such replies are double-checked with the model before being handed back.
    """
    return bool(specs) and not reply.calls


def followup_prompt(prompt: str, draft: str, user_request: str = "") -> str:
    request = f'The user asked: "{user_request.strip()[:500]}"\n' if user_request.strip() else ""
    if not draft:
        return (
            f"{prompt}\n\n[system]\n{request}Your previous reply was empty. Reply now with either the "
            '<tool_call name="..."> block(s) for your next step, or your answer to the user.'
        )
    return (
        f"{prompt}\n\n[assistant]\n{draft}\n\n"
        f"[system]\n{request}That reply contains no tool call, so it would end your turn and nothing more would happen. "
        "Reply now with only the <tool_call name=\"...\"> block(s) that perform your next step, no other text. "
        "If your reply says you still need to do something (read, edit, run, verify), that step must be a tool call now. "
        f"Reply with exactly {FINAL_MARKER} only if every change the user asked for has already been made in the "
        "conversation above, or you cannot continue without the user's answer to a question."
    )


def last_user_request(items: List[Dict[str, Any]]) -> str:
    """Latest real user message, skipping client notices such as Cline's "[TASK RESUMPTION] ..."."""
    texts = [content_to_text(i.get("content")).strip() for i in items if i.get("type", "message") == "message" and i.get("role") == "user"]
    texts = [t for t in texts if t]
    real = [t for t in texts if not t.startswith(("[", "<"))]
    return (real or texts or [""])[-1]


def shorter_reply_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n[system]\nYour previous attempt at this reply produced no output, most likely because it was "
        "too long. Write a much shorter reply: if you are editing a file, change only one section per tool call "
        "and continue with the rest in later turns."
    )


def is_final_marker(text: str) -> bool:
    return text.strip().strip(".").upper() == FINAL_MARKER


def _strip_label(text: str) -> str:
    return re.sub(r"^\s*\[assistant\]\s*\n?", "", text)


def _strip_fence(body: str) -> str:
    m = FENCE_RE.match(body)
    return m.group("inner") if m else body.strip()


def _load_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse the leading JSON object, tolerating raw control characters (e.g. unescaped
    newlines) inside strings and trailing text after the object."""
    for strict in (True, False):
        try:
            obj, _ = json.JSONDecoder(strict=strict).raw_decode(text)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _json_args(body: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return (arguments string, parsed object or None)."""
    cleaned = _strip_fence(body)
    obj = _load_json_object(cleaned)
    if obj is None:
        # Leave it to the client to report the malformed arguments back to the model.
        return cleaned, None
    return json.dumps(obj, ensure_ascii=False), obj


def _wrapped_args(spec: ToolSpec, body: str) -> str:
    raw = _strip_fence(body)
    obj = _load_json_object(raw) if raw.startswith("{") else None
    if obj is not None and set(obj) == {spec.wrap_param} and isinstance(obj[spec.wrap_param], str):
        return json.dumps(obj, ensure_ascii=False)  # the model sent JSON after all
    return json.dumps({spec.wrap_param: raw}, ensure_ascii=False)


def build_output_items(reply: ParsedReply, specs: List[ToolSpec], parallel: bool) -> List[Dict[str, Any]]:
    by_name = {s.name: s for s in specs}
    by_name.update({s.api_name: s for s in specs if s.api_name not in by_name})
    items: List[Dict[str, Any]] = []
    if reply.text:
        items.append({
            "id": new_id("msg"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": reply.text, "annotations": []}],
        })
    calls = reply.calls if parallel else reply.calls[:1]
    for name, body in calls:
        spec = by_name.get(name)
        if spec is None:
            logger.warning("Model called unknown tool %r; forwarding as function call", name)
            spec = ToolSpec(name, "function", api_name=name)
        items.append(_call_item(spec, body))
    return items


def _call_item(spec: ToolSpec, body: str) -> Dict[str, Any]:
    call_id = new_id("call")
    if spec.kind == "nested":
        if spec.string_input:
            args = json.dumps(_strip_fence(body), ensure_ascii=False)
        else:
            args = _json_args(body)[0] or "{}"
        js = nested_call_js(spec.name, args)
        item = {"id": new_id("ctc"), "type": "custom_tool_call", "status": "completed", "call_id": call_id, "name": spec.api_name, "input": js}
    elif spec.kind == "custom":
        item = {"id": new_id("ctc"), "type": "custom_tool_call", "status": "completed", "call_id": call_id, "name": spec.api_name, "input": _strip_fence(body)}
    elif spec.kind == "local_shell":
        _, args = _json_args(body)
        args = args or {}
        command = args.get("command") or []
        if isinstance(command, str):
            command = ["bash", "-lc", command]
        action = {"type": "exec", "command": command, "env": args.get("env") or {}}
        if args.get("workdir"):
            action["working_directory"] = args["workdir"]
        if args.get("timeout_ms"):
            action["timeout_ms"] = args["timeout_ms"]
        return {"id": new_id("lsh"), "type": "local_shell_call", "status": "completed", "call_id": call_id, "action": action}
    elif spec.kind == "shell":
        _, args = _json_args(body)
        args = args or {}
        commands = args.get("commands") or []
        if isinstance(commands, str):
            commands = [commands]
        action = {"commands": commands}
        if args.get("timeout_ms"):
            action["timeout_ms"] = args["timeout_ms"]
        return {"id": new_id("sh"), "type": "shell_call", "status": "completed", "call_id": call_id, "action": action}
    elif spec.kind == "apply_patch":
        _, args = _json_args(body)
        operation = {k: v for k, v in (args or {}).items() if k in ("type", "path", "diff")}
        return {"id": new_id("apc"), "type": "apply_patch_call", "status": "completed", "call_id": call_id, "operation": operation}
    else:
        arguments = _wrapped_args(spec, body) if spec.wrap_param else _json_args(body)[0]
        item = {"id": new_id("fc"), "type": "function_call", "status": "completed", "call_id": call_id, "name": spec.api_name, "arguments": arguments}
    if spec.namespace:
        item["namespace"] = spec.namespace
    return item
