"""Layouts: how a call is written as messages and how the reply is read back.

An adapter is an lmcc adapter: a chat template, a reader, transports by
purpose (reasoning, tools) and formats by type. functai ships four, by name:

- ``None`` / ``"xml"``: tagged sections (``<answer>…</answer>``), the default
- ``"chat"``: DSPy's ``[[ ## name ## ]]`` sections, for prompts migrated from DSPy
- ``"json"``: one JSON object the provider enforces (models with native structured output)

Or write your own chat template right in the decorator::

    @ai(template=[
        system("You are a pirate. {instruction}"),
        turns(),                                   # examples and conversation go here
        user("Text: {text}"),
    ])
    def summarize(text: str) -> str: ...

A template without an output pattern is fine for one output: the whole reply
is the value. With several outputs, spell the pattern, e.g.
``{% for f in outputs %}<{f.name}>\\n{f.value}\\n</{f.name}>\\n{% endfor %}``;
lmcc reads the reply back through the same pattern.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence

import lmcc
import lmcc_std
from lmcc import core as lmcc_core
from lmcc.adapter import assistant, developer, system, turns, user  # noqa: F401 — re-exported
from lmcc.errors import refuse
from lmcc.reader import Reader

__all__ = ["system", "user", "assistant", "developer", "turns", "xml_adapter", "chat_adapter",
           "json_adapter", "template_adapter", "Layout", "bind", "REGISTRY", "DEFAULT_FORMATS"]

REGISTRY = lmcc.default_registry
"""lmcc's default registry, so a type bound with ``lmcc.format(T, ...)`` reaches
functai too; importing functai adds lmcc's standard vocabulary to it."""
lmcc_std.install(REGISTRY)

# Structured values (lists, dicts, dataclasses, pydantic models) are JSON unless an
# adapter says otherwise; a wildcard never re-spells a scalar.
DEFAULT_FORMATS = {"*": lmcc.use("json")}


# ------------------------------------------------------------------ the reply reader


class ReplyReader(Reader):
    """The whole reply is the one output's value: for chat templates with no
    output pattern (``user("Text: {text}")``) and a single output."""

    def __init__(self, spec: dict):
        if set(spec) != {"kind"}:
            refuse("entry-malformed", "reader: functai_reply takes only 'kind'",
                   fix={"action": "edit-entry", "path": "reader"})
        self.spec = dict(spec)

    def split(self, text: str, field_names: List[str]) -> Dict[str, str]:
        if len(field_names) != 1:
            refuse("parse-ambiguous", "the template has no output pattern, so the reply can hold "
                                      f"one output, not {len(field_names)}: {field_names}")
        return {field_names[0]: lmcc_core.strip(text)}

    def join(self, spelled) -> str:
        return "\n".join(text for _name, text in spelled)


REGISTRY.register_reader("functai_reply", ReplyReader, version="1.0.0", exist_ok=True)


# ------------------------------------------------------------------ transports


def _transports() -> Dict[str, lmcc.Transport]:
    """Reasoning and tools travel by what the model can do: its own thinking
    channel and native tool calls where it has them, text otherwise."""
    return {
        "reasoning": lmcc.Transport(choose=[
            {"when": {"capability": "native_reasoning"}, "use": REGISTRY.transport("native_reasoning", {})},
            {"else": REGISTRY.transport("prefix_cot", {})}]),
        "tools": lmcc.Transport(choose=[
            {"when": {"capability": "native_function_calling"}, "use": REGISTRY.transport("native_tools", {})},
            {"else": REGISTRY.transport("fenced_tools", {})}]),
    }


# ------------------------------------------------------------------ the shipped layouts

_INPUT_TAGS = "{% for f in inputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"


def xml_adapter() -> lmcc.Adapter:
    """Tagged sections: instructions and the reply pattern in the system message,
    examples and past turns as messages, the inputs in tags."""
    return lmcc.adapter(
        name="functai_xml",
        formats=DEFAULT_FORMATS,
        messages=[
            lmcc.system("{instruction}\n\nReply in exactly this form:\n"
                        "{% for f in outputs %}<{f.name}>\n{f.value}\n</{f.name}>\n{% endfor %}"),
            lmcc.turns(),
            lmcc.user(_INPUT_TAGS),
        ],
        transports=_transports())


def chat_adapter() -> lmcc.Adapter:
    """DSPy's ChatAdapter layout: ``[[ ## name ## ]]`` sections ending with
    ``[[ ## completed ## ]]`` (the prompt bytes differ from DSPy's; the shape is the same)."""
    return lmcc.adapter(
        name="functai_chat",
        formats=DEFAULT_FORMATS,
        messages=[
            lmcc.system("{instruction}\n\nRespond with the corresponding output fields, each under its "
                        "header, then end with [[ ## completed ## ]]:\n\n"
                        "{% for f in outputs %}[[ ## {f.name} ## ]]\n{f.value}\n\n{% endfor %}"
                        "[[ ## completed ## ]]"),
            lmcc.turns(),
            lmcc.user("{% for f in inputs %}[[ ## {f.name} ## ]]\n{f.value}\n\n{% endfor %}"),
        ],
        transports=_transports())


def json_adapter(*, probabilities: Optional[str] = None) -> lmcc.Adapter:
    """The reply is one JSON object the provider enforces (lmcc ``json_object``),
    for models that declare ``native_structured_output``. With ``probabilities``
    (``"if_available"`` or ``"required"``) the provider is asked for the
    distribution over each choice's answers."""
    reader = {"kind": "json_object", **({"probabilities": probabilities} if probabilities else {})}
    transports = _transports()
    del transports["reasoning"]            # reasoning is an ordinary member of the object here
    return lmcc.adapter(
        name="functai_json",
        messages=[lmcc.system("{instruction}"), lmcc.turns(), lmcc.user(_INPUT_TAGS)],
        reader=reader, formats=DEFAULT_FORMATS, transports=transports)


def judgment_adapter(signature: lmcc.SignatureCore) -> lmcc.Adapter:
    """For providers that answer only judgments (TypeSafe's Jev): no system prompt,
    the input alone as the state, the reply a JSON object."""
    inputs = [f for f in signature.inputs if f.purpose == "plain"]
    body = "{" + inputs[0].name + "}" if len(inputs) == 1 else _INPUT_TAGS
    return lmcc.adapter(name="functai_judgment", messages=[lmcc.turns(), lmcc.user(body)],
                        reader={"kind": "json_object"}, formats=DEFAULT_FORMATS)


def judgment_signature(signature: lmcc.SignatureCore) -> lmcc.SignatureCore:
    """Each output's question is its description, else the docstring."""
    outputs = signature.outputs
    doc = signature.instructions

    def question(f):
        if f.desc or not doc:
            return f.desc
        return doc if len(outputs) == 1 else f"{doc} ({f.name})"
    return lmcc.SignatureCore(doc, [dataclasses.replace(f, desc=question(f)) if f.direction == "output" else f
                                    for f in signature.fields])


NAMED = {"xml": xml_adapter, "default": xml_adapter, "tags": xml_adapter,
         "chat": chat_adapter, "chatadapter": chat_adapter,
         "json": json_adapter, "jsonadapter": json_adapter}


# ------------------------------------------------------------------ templates


def _message(m: Any, i: int) -> dict:
    """One template entry: an lmcc message or directive, or an OpenAI-style
    ``{"role", "content"}`` dict."""
    if isinstance(m, dict):
        if "directive" in m or ("role" in m and "text" in m):
            return dict(m)
        if "role" in m and "content" in m and isinstance(m["content"], str):
            return {"role": m["role"], "text": m["content"]}
    raise TypeError(f"template[{i}]: expected system(...), user(...), assistant(...), developer(...), "
                    f"turns(), or a {{'role', 'content'}} dict; got {m!r}")


def template_adapter(messages: Sequence[Any], *, name: str = "functai_template",
                     reader: str = "derived") -> lmcc.Adapter:
    """An adapter from a chat template written in ``@ai(template=[...])``, with
    functai's formats and transports. ``turns()`` marks where examples and the
    conversation so far go; without it, they go right before the last user message."""
    msgs = [_message(m, i) for i, m in enumerate(messages)]
    if not any("directive" in m for m in msgs):
        last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=len(msgs))
        msgs.insert(last_user, lmcc.turns())
    return lmcc.adapter(name=name, messages=msgs, formats=DEFAULT_FORMATS, transports=_transports(),
                        reader={"kind": reader})


@dataclasses.dataclass(frozen=True)
class Layout:
    """How a function lays out its calls: a named/given adapter or a template."""
    adapter: Any = None                 # None | str | lmcc.Adapter | artifact dict
    template: Optional[tuple] = None    # messages, when written in the decorator

    def key(self) -> Any:
        if self.template is not None:
            return ("template", id(self.template))
        return ("adapter", self.adapter if isinstance(self.adapter, (str, type(None))) else id(self.adapter))


def _named(adapter: Any) -> lmcc.Adapter:
    if isinstance(adapter, lmcc.Adapter):
        return adapter
    if isinstance(adapter, dict):
        return lmcc.load(adapter, registry=REGISTRY)
    if isinstance(adapter, str):
        key = adapter.lower().replace("-", "_").replace(" ", "")
        if key in NAMED:
            return NAMED[key]()
        raise ValueError(f"unknown adapter {adapter!r}; use one of {sorted(set(NAMED) - {'chatadapter', 'jsonadapter'})}, "
                         f"an lmcc.Adapter, or template=[...]")
    mod = type(adapter).__module__ or ""
    if mod.startswith("dspy") or (isinstance(adapter, type) and (adapter.__module__ or "").startswith("dspy")):
        raise TypeError("DSPy adapters no longer run in functai 1.0: use adapter='chat' (DSPy's layout), "
                        "adapter='json', or write the layout with template=[system(...), turns(), user(...)]")
    raise TypeError(f"adapter must be None, 'xml', 'chat', 'json', an lmcc.Adapter or an adapter "
                    f"artifact dict, not {type(adapter).__name__}")


def bind(layout: Layout, signature: lmcc.SignatureCore, capabilities: Dict[str, Any], provider: str) -> lmcc.Plan:
    """Bind the layout to the signature for a model's capabilities. Every
    refusal fires here, before any request is sent."""
    if layout.template is not None:
        adapter = template_adapter(layout.template)
        try:
            return adapter.bind(signature, capabilities, registry=REGISTRY)
        except lmcc.Refusal as err:
            # no output pattern anywhere in the template: the whole reply is the value
            if err.code != "not-readable" or (err.fix or {}).get("path") != "template":
                raise
        plan = template_adapter(layout.template, reader="functai_reply").bind(signature, capabilities,
                                                                               registry=REGISTRY)
        visible = [f.name for f in plan.visible_outputs]
        if len(visible) != 1:
            raise lmcc.Refusal(
                "not-readable",
                f"the template has no output pattern, so the reply can only be one output, but this "
                f"function has {len(visible)}: {visible}. Add the pattern to the template, e.g. "
                "'{% for f in outputs %}<{f.name}>\\n{f.value}\\n</{f.name}>\\n{% endfor %}'",
                fix={"action": "edit-template", "path": "template"})
        return plan
    if layout.adapter is None and provider in _judgment_providers():
        return judgment_adapter(signature).bind(judgment_signature(signature), capabilities, registry=REGISTRY)
    adapter = _named(layout.adapter if layout.adapter is not None else "xml")
    return adapter.bind(signature, capabilities, registry=REGISTRY)


def _judgment_providers():
    from .models import JUDGMENT_ONLY
    return JUDGMENT_ONLY
