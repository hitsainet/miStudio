"""No load path may name a precision: every model load takes the resolver's dtype.

⚠ WHY. Every miStudio load path cast 16-bit rows to `torch.float16` while the checkpoints are
bfloat16 and miLLM serves bfloat16, so every probe and SAE was fitted on activations nothing
serves. It surfaced as a probe failing parity on import (combined Δ 0.251 against 0.10) and was
confirmed on hardware 2026-10-03. The fix is one resolver (`src/ml/native_dtype.py`); this guard
keeps a literal from creeping back in at a site the resolver never sees — the shape this estate
keeps shipping (a fix applied at one representative site and never generalised).

Two rules, both read off the AST rather than the text (a text search matches the comments that
describe the old defect, which this module is full of):

  1. No dtype LITERAL (`torch.float16`, `torch.half`, `torch.bfloat16`, `torch.float32`, or the
     strings "float16"/"bfloat16"/"float32"/"auto") is passed as `torch_dtype=`, `dtype=` or
     `bnb_4bit_compute_dtype=` to a model load, a split plan, a bnb config, a load-kwargs dict,
     or the quantization-config helper.
  2. Every function that calls a MODEL's `from_pretrained` also calls a resolver.

`EXEMPT` is empty, and should stay so: the CPU branch that loads float32 does so through the
rule's own FP32 case, not a literal.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: Calls whose dtype keywords decide what a model loads or computes at.
DTYPE_SINKS = {"from_pretrained", "plan_split_load", "BitsAndBytesConfig", "get_quantization_config",
               "preflight_split", "dict"}
DTYPE_KEYWORDS = {"torch_dtype", "dtype", "bnb_4bit_compute_dtype"}
LITERAL_ATTRS = {"float16", "half", "bfloat16", "float32", "float"}
LITERAL_STRINGS = {"float16", "bfloat16", "float32", "auto", "half", "fp16", "bf16"}

#: Functions that resolve a load dtype. A model load must sit in a function calling one.
RESOLVERS = {"resolve_for_config", "resolve_for_snapshot", "resolve_load_dtype", "_steering_dtype"}

#: (file relative to src, function name) — deliberately empty.
EXEMPT: set = set()


def _name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_literal_dtype(node: ast.AST) -> bool:
    # ANY base, not just the name `torch`: a mutation spelled `__import__("torch").float16`
    # walked past a `torch.`-only check in this repo once (test_jlens_honours_quantization.py's
    # history), and did so again against this guard's first draft.
    if isinstance(node, ast.Attribute):
        return node.attr in LITERAL_ATTRS
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.lower() in LITERAL_STRINGS
    if isinstance(node, ast.IfExp):
        return _is_literal_dtype(node.body) or _is_literal_dtype(node.orelse)
    return False


def _is_model_load(call: ast.Call) -> bool:
    """`<Model class>.from_pretrained(...)` — not a tokenizer's or a config's."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "from_pretrained"):
        return False
    owner = func.value
    owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
    return "ForCausalLM" in owner_name or owner_name.startswith("AutoModel")


def _functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _scan():
    literals, unresolved, loads = [], [], []
    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text(), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in DTYPE_SINKS:
                for kw in node.keywords:
                    if kw.arg in DTYPE_KEYWORDS and _is_literal_dtype(kw.value):
                        literals.append(f"{rel}:{node.lineno} {_name(node.func)}({kw.arg}=...)")
        for fn in _functions(tree):
            # Only the calls in THIS function's own body, not in nested functions.
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            model_loads = [c for c in calls if _is_model_load(c)]
            if not model_loads:
                continue
            loads.extend(f"{rel}:{c.lineno}" for c in model_loads)
            if (rel, fn.name) in EXEMPT:
                continue
            if not any(_name(c.func) in RESOLVERS for c in calls):
                unresolved.append(f"{rel}:{fn.lineno} {fn.name}")
    return literals, unresolved, loads


def test_no_dtype_literal_reaches_a_load():
    literals, _, _ = _scan()
    assert literals == [], (
        "A precision is hardcoded where a model is loaded or planned. Take the dtype from "
        "src/ml/native_dtype.py instead:\n  " + "\n  ".join(literals)
    )


def test_every_model_load_sits_beside_a_resolver_call():
    _, unresolved, _ = _scan()
    assert unresolved == [], (
        "These functions load a model without resolving its dtype:\n  " + "\n  ".join(unresolved)
    )


def test_the_guard_sees_the_estates_model_loads():
    """⚠ A guard that finds nothing proves nothing. Seven loads exist; if the scan stops seeing
    them (a renamed class, a moved call), both tests above pass vacuously."""
    _, _, loads = _scan()
    files = {entry.split(":")[0] for entry in loads}
    for expected in ("ml/model_loader.py", "services/activation_service.py",
                     "services/steering_service.py", "services/logit_lens_service.py",
                     "services/jlens_model_registry.py", "services/local_labeling_service.py"):
        assert expected in files, f"the guard no longer sees the model load in {expected}"
    assert len(loads) >= 7


def test_the_literal_detector_bites(tmp_path):
    """The detector itself, on code that names a precision three ways."""
    sample = (
        "import torch\n"
        "def f(A, d):\n"
        "    A.from_pretrained(p, torch_dtype=torch.float16)\n"
        "    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)\n"
        "    kw = dict(torch_dtype=torch.half if d else torch.float32)\n"
        "    A.from_pretrained(p, dtype='auto')\n"
        "    A.from_pretrained(p, torch_dtype=__import__('torch').float16)\n"
    )
    tree = ast.parse(sample)
    hits = [kw for node in ast.walk(tree) if isinstance(node, ast.Call) and _name(node.func) in DTYPE_SINKS
            for kw in node.keywords if kw.arg in DTYPE_KEYWORDS and _is_literal_dtype(kw.value)]
    assert len(hits) == 5
