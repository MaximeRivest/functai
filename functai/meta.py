"""AI functions that work on AI functions: write or refine an instruction,
propose instruction candidates for optimizers, synthesize training examples.

They are ordinary functai functions, so ``phistory()`` shows their prompts too.
Their own settings are fixed (tags layout, no memory) so a global
``configure(stateful=True, adapter=...)`` does not change how they work.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .core import _ai, ai
from .docments import get_source

_FIXED = dict(adapter="xml", stateful=False, module="predict", include_fn_name_in_instructions=False,
              autocompile=False, autoinstruct=False, instruction_autorefine_calls=0)


@ai(**_FIXED)
def _write_instruction(function_name: str, source: str, inputs: List[str], outputs: List[str],
                       current_instruction: str) -> str:
    """Write one clear, complete instruction (a system prompt) for an AI that performs this Python
    function. The source code, its docstring, type hints and comments define the task. Name the exact
    inputs and outputs, state every constraint the code states, and explain the approach briefly.
    Reply with the instruction text only."""
    return _ai


@ai(**_FIXED)
def _refine_instruction(signature: str, current_instruction: str, observations: str) -> str:
    """Refine the system instruction of an AI function. The observed outputs are NOT ground truth:
    treat them as noisy hints to improve clarity, constraints and failure handling. Keep the exact
    input and output names. Reply with the revised instruction only."""
    return _ai


@ai(**_FIXED)
def _propose_instruction(function_name: str, source: str, signature: str, current_instruction: str,
                         examples: str, previous_proposals: List[str], tip: str) -> str:
    """Propose a new instruction (a system prompt) for an AI function that will make it score higher
    on its task. Use the source code, the signature and the examples to understand the task. Make it
    different from the previous proposals, and follow the tip. Keep the exact input and output names.
    Reply with the instruction text only."""
    return _ai


@ai(**_FIXED)
def _synthesize_examples(task: str, input_names: List[str], output_names: List[str], n: int) -> List[Dict[str, Any]]:
    """Write n diverse, realistic examples of this task. Each example is a JSON object whose keys are
    exactly the input names and the output names, with correct outputs."""
    return _ai


@ai(**_FIXED)
def _synthesize_inputs(task: str, input_names: List[str], n: int) -> List[Dict[str, Any]]:
    """Write n diverse, realistic inputs for this task. Each is a JSON object whose keys are exactly
    the input names."""
    return _ai


def _io(fn) -> tuple:
    spec = fn._spec(instructions=None)
    ins = [f.name for f in spec.signature.inputs if f.purpose == "plain"]
    outs = [f.name for f in spec.signature.outputs if f.purpose not in ("tools.calls",)]
    return spec, ins, outs


def write_instruction(fn, *, lm: Any = None) -> str:
    spec, ins, outs = _io(fn)
    text = _write_instruction.using(lm=lm)(
        function_name=fn.__name__, source=get_source(fn._fn) or "", inputs=ins, outputs=outs,
        current_instruction=spec.signature.instructions)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("the instruction writer returned no text")
    return text.strip()


def refine_instruction(fn, observed: List[Dict[str, Any]], *, lm: Any = None) -> str:
    from .core import signature_text
    lines: List[str] = []
    for i, ex in enumerate(observed, 1):
        lines.append(f"Example {i}:")
        lines += [f"  input {k}: {v}" for k, v in (ex.get("inputs") or {}).items()]
        lines += [f"  output {k} (noisy): {v}" for k, v in (ex.get("outputs") or {}).items()]
    text = _refine_instruction.using(lm=lm)(signature=signature_text(fn), current_instruction=fn.instructions,
                                            observations="\n".join(lines) or "(none)")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("the instruction refiner returned no text")
    return text.strip()


def propose_instruction(fn, *, examples: str, previous: List[str], tip: str, lm: Any = None,
                        base_instruction: Optional[str] = None) -> str:
    from .core import signature_text
    text = _propose_instruction.using(lm=lm, temperature=1.0)(
        function_name=fn.__name__, source=get_source(fn._fn) or "", signature=signature_text(fn),
        current_instruction=base_instruction or fn.instructions, examples=examples or "(none)",
        previous_proposals=previous, tip=tip or "(no tip)")
    return (text or "").strip()


def task_text(fn) -> str:
    spec, ins, outs = _io(fn)
    src = get_source(fn._fn) or ""
    return spec.signature.instructions + (f"\n\nSource:\n{src}" if src else "")


def synthesize(fn, n: int, *, lm: Any = None, labeler: Any = None) -> List[Dict[str, Any]]:
    """``n`` examples as ``{"inputs", "outputs"}`` dicts: written whole by the
    model ``lm``, or (with ``labeler``, an AI function) inputs written by ``lm``
    and labeled by running ``labeler``."""
    spec, ins, outs = _io(fn)
    task = task_text(fn)
    out: List[Dict[str, Any]] = []
    if labeler is None:
        items = _synthesize_examples.using(lm=lm, temperature=1.0)(task=task, input_names=ins,
                                                                   output_names=list(spec.outputs), n=n)
        for item in items or []:
            if isinstance(item, dict) and all(k in item for k in ins) and all(k in item for k in spec.outputs):
                out.append({"inputs": {k: item[k] for k in ins}, "outputs": {k: item[k] for k in spec.outputs}})
        return out
    items = _synthesize_inputs.using(lm=lm, temperature=1.0)(task=task, input_names=ins, n=n)
    for item in items or []:
        if not (isinstance(item, dict) and all(k in item for k in ins)):
            continue
        inputs = {k: item[k] for k in ins}
        try:
            pred = labeler(**inputs, all=True)
        except Exception:  # noqa: BLE001 — a failed label drops the example
            continue
        out.append({"inputs": inputs, "outputs": {k: pred.get(k) for k in spec.outputs if k in pred}})
    return out


def examples_text(rows: List[Dict[str, Any]], input_names: List[str], limit: int = 5) -> str:
    lines: List[str] = []
    for row in rows[:limit]:
        ins = {k: v for k, v in row.items() if k in input_names}
        outs = {k: v for k, v in row.items() if k not in input_names}
        lines.append(json.dumps({"inputs": ins, "outputs": outs}, ensure_ascii=False, default=str))
    return "\n".join(lines)
