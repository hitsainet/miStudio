"""No GPU is hard-coded anywhere in backend/src — only gpu_placement chooses a card.

The node gained a second GPU on 2026-09-13 and the new card took index 0. Every
job written against "cuda", mem_get_info(0) or CUDA_VISIBLE_DEVICES="0" moved to
it without anyone deciding so. This guard walks the AST (never the source text:
a text scrape matches the comments describing a fix and passes for the wrong
reason) and records each hard-coded choice by module, enclosing function and
kind.

A "cuda" string literal counts wherever it can end up choosing a device —
passed, assigned, returned, or used as a parameter default. The first version of
this guard only looked at call keywords and missed `self._device = "cuda"`,
`device = "cuda" if ... else "cpu"` and `capture_device: str = "cuda"`, which is
most of the real code: a guard with a blind spot passes by construction. Only
the uses that READ a device rather than choose one are exempt, each named in
`_reads_a_device`.

RATCHET. `hard_coded_gpu_ratchet.json` lists the sites that existed when the
guard landed. The suite fails on any site NOT in the list (new hard-coding), and
on any listed site that no longer exists (delete its entry — the list only
shrinks). Phase 1 of 0xcc/plans/Multi-GPU-Plan.md ends with the list empty.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
RATCHET = Path(__file__).with_name("hard_coded_gpu_ratchet.json")

#: The one module allowed to name devices: it is where the choice is made.
EXEMPT_MODULES = {"services/gpu_placement.py"}

#: torch.cuda calls that act on the CURRENT device when given no device.
DEVICE_DEFAULTING_CUDA_CALLS = {
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "max_memory_allocated",
    "max_memory_reserved",
    "reset_peak_memory_stats",
    "get_device_properties",
    "get_device_name",
    "synchronize",
}

#: String methods that inspect a device name rather than choose one.
INSPECTING_STR_METHODS = {"startswith", "endswith", "split", "partition", "replace"}


def _is_cuda_string(node: ast.AST) -> bool:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return False
    value = node.value.strip()
    return value == "cuda" or (value.startswith("cuda:") and value[5:].isdigit())


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_constant_int(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def _reads_a_device(node: ast.Constant) -> bool:
    """True when the literal only inspects a device, so it chooses nothing."""
    parent = getattr(node, "_parent", None)
    # device.type == "cuda", "cuda" in str(device)
    if isinstance(parent, ast.Compare):
        return True
    # device.type in ("cuda", "mps")
    if isinstance(parent, (ast.Tuple, ast.List, ast.Set)) and isinstance(getattr(parent, "_parent", None), ast.Compare):
        return True
    # str(device).startswith("cuda")
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) \
            and parent.func.attr in INSPECTING_STR_METHODS and node in parent.args:
        return True
    # a docstring or a bare string statement
    if isinstance(parent, ast.Expr):
        return True
    # f"cuda:{index}" — built from an index someone already chose
    if isinstance(parent, ast.JoinedStr):
        return True
    # {"cuda": ...} — a key naming a backend, not a device choice
    if isinstance(parent, ast.Dict) and node in parent.keys:
        return True
    return False


class _Scanner(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.sites: list[str] = []

    def _record(self, kind: str) -> None:
        self.sites.append(f"{self.module}::{'.'.join(self.scope) or '<module>'}::{kind}")

    def _enter(self, node) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Constant(self, node: ast.Constant) -> None:
        if _is_cuda_string(node) and not _reads_a_device(node):
            self._record("'cuda' literal")

    def _check_gpu_id_default(self, arg: ast.arg, default: ast.AST | None) -> None:
        if default is None or arg.arg != "gpu_id":
            return
        if _is_constant_int(default):
            self._record("gpu_id default")
        # FastAPI: `gpu_id: int = Query(0, ...)`
        elif isinstance(default, ast.Call) and _dotted(default.func).rsplit(".", 1)[-1] == "Query" \
                and default.args and _is_constant_int(default.args[0]):
            self._record("gpu_id Query default")

    def visit_arguments(self, node: ast.arguments) -> None:
        positional = node.posonlyargs + node.args
        for arg, default in zip(positional[len(positional) - len(node.defaults):], node.defaults):
            self._check_gpu_id_default(arg, default)
        for arg, default in zip(node.kwonlyargs, node.kw_defaults):
            self._check_gpu_id_default(arg, default)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        tail = name.rsplit(".", 1)[-1]

        if tail == "cuda" and not node.args and not node.keywords and isinstance(node.func, ast.Attribute) \
                and _dotted(node.func.value) != "torch":
            self._record(".cuda()")
        if name.startswith("torch.cuda.") and tail in DEVICE_DEFAULTING_CUDA_CALLS:
            if not node.args and not any(k.arg == "device" for k in node.keywords):
                self._record(f"torch.cuda.{tail}() on the current device")
            elif node.args and _is_constant_int(node.args[0]):
                self._record(f"torch.cuda.{tail}({node.args[0].value})")
        if tail == "nvmlDeviceGetHandleByIndex" and node.args and _is_constant_int(node.args[0]):
            self._record(f"nvmlDeviceGetHandleByIndex({node.args[0].value})")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) \
                    and target.slice.value == "CUDA_VISIBLE_DEVICES":
                self._record("CUDA_VISIBLE_DEVICES assignment")
        self.generic_visit(node)


def _scan_source(module: str, source: str) -> list[str]:
    tree = ast.parse(source)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]
    scanner = _Scanner(module)
    scanner.visit(tree)
    return scanner.sites


def scan() -> Counter:
    sites: Counter = Counter()
    for path in sorted(SRC.rglob("*.py")):
        module = path.relative_to(SRC).as_posix()
        if module not in EXEMPT_MODULES:
            sites.update(_scan_source(module, path.read_text()))
    return sites


def _ratchet() -> Counter:
    return Counter(json.loads(RATCHET.read_text()))


def new_sites(found: Counter, allowed: Counter) -> dict[str, int]:
    """Sites found more often than the ratchet allows: new hard-coding."""
    return {site: count - allowed.get(site, 0) for site, count in found.items() if count > allowed.get(site, 0)}


def stale_sites(found: Counter, allowed: Counter) -> dict[str, int]:
    """Ratchet entries the code no longer has: the list must shrink with it."""
    return {site: count - found.get(site, 0) for site, count in allowed.items() if count > found.get(site, 0)}


def test_no_new_hard_coded_gpu():
    new = new_sites(scan(), _ratchet())
    assert not new, (
        "New hard-coded GPU choice(s). Take the device from services/gpu_placement "
        "instead:\n" + "\n".join(f"  {site} (+{extra})" for site, extra in sorted(new.items()))
    )


def test_the_ratchet_only_shrinks():
    stale = stale_sites(scan(), _ratchet())
    assert not stale, (
        "These hard-coded GPU sites are gone — remove them from hard_coded_gpu_ratchet.json:\n"
        + "\n".join(f"  {site} (-{fewer})" for site, fewer in sorted(stale.items()))
    )


def test_the_ratchet_comparison_catches_added_and_removed_sites():
    """The two ratchet tests pass on today's tree whether or not their comparison
    works, because today's tree IS the ratchet. Prove the comparison on counts
    that differ."""
    allowed = Counter({"a.py::f::'cuda' literal": 2, "b.py::g::gpu_id default": 1})

    assert new_sites(Counter({"a.py::f::'cuda' literal": 3, "b.py::g::gpu_id default": 1}), allowed) == {
        "a.py::f::'cuda' literal": 1
    }
    assert new_sites(Counter({"c.py::h::.cuda()": 1}), Counter()) == {"c.py::h::.cuda()": 1}
    assert stale_sites(Counter({"a.py::f::'cuda' literal": 2}), allowed) == {"b.py::g::gpu_id default": 1}
    assert new_sites(allowed.copy(), allowed) == {} and stale_sites(allowed.copy(), allowed) == {}


PROBE = '''
import os, torch, pynvml
from fastapi import Query

def endpoint(gpu_id: int = Query(0, ge=0)):
    pass

def defaults(gpu_id: int = 0, capture_device: str = "cuda"):
    pass

def job():
    torch.device("cuda")
    model.to("cuda:0")
    load(device_map="cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self._device = "cuda"
    torch.cuda.mem_get_info()
    torch.cuda.memory_allocated(0)
    pynvml.nvmlDeviceGetHandleByIndex(0)
    env = {}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    tensor.cuda()
    return "cuda"

def reads_only(device, index):
    """Moves tensors to "cuda" devices."""
    if device.type == "cuda":
        pass
    if str(device).startswith("cuda"):
        pass
    if device.type in ("cuda", "mps"):
        pass
    name = f"cuda:{index}"
    backends = {"cuda": 1}
    torch.device("cpu")
    model.to(device)
    torch.cuda.mem_get_info(device)
    torch.cuda.is_available()
'''


def _kinds(sites: list[str], function: str) -> Counter:
    return Counter(site.rsplit("::", 1)[-1] for site in sites if f"::{function}::" in site)


def test_the_scanner_sees_each_kind_it_claims_to_detect():
    sites = _scan_source("probe.py", PROBE)

    assert _kinds(sites, "endpoint") == Counter({"gpu_id Query default": 1})
    assert _kinds(sites, "defaults") == Counter({"gpu_id default": 1, "'cuda' literal": 1})
    assert _kinds(sites, "job") == Counter({
        # torch.device, .to, device_map=, the conditional, self._device, return
        "'cuda' literal": 6,
        "torch.cuda.mem_get_info() on the current device": 1,
        "torch.cuda.memory_allocated(0)": 1,
        "nvmlDeviceGetHandleByIndex(0)": 1,
        "CUDA_VISIBLE_DEVICES assignment": 1,
        ".cuda()": 1,
    })
    assert _kinds(sites, "reads_only") == Counter()
