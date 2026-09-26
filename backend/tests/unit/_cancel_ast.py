"""AST helpers for cancellation tests. NOT a test module.

WHY THIS EXISTS. Review round 1 found six assertions in the cancellation work
that were satisfied by something other than the behaviour: a substring matched
inside the comment explaining it, `cancel_check=` matched by `cancel_check=None`,
`str.index` on a name matched its own assignment, a glob that matched the
asserting file itself. Every one of them was written by someone who had just
recorded that lesson.

The fix is not more careful reading. It is to stop asserting about text. These
helpers read structure, so "the call exists", "the argument is not None" and
"these two handlers hang off the same try" mean what they say.
"""

from __future__ import annotations

import ast
import inspect
from typing import Iterable, List, Optional


def source_of(obj) -> str:
    """Source of a function, unwrapping every decorator layer."""
    return inspect.getsource(inspect.unwrap(obj))


def _tree(obj) -> ast.AST:
    import textwrap

    return ast.parse(textwrap.dedent(source_of(obj)))


def calls_named(obj, name: str) -> List[ast.Call]:
    """Every `ast.Call` in `obj` whose callee is `name` (bare or attribute)."""
    out = []
    for node in ast.walk(_tree(obj)):
        if not isinstance(node, ast.Call):
            continue
        callee = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if callee == name:
            out.append(node)
    return out


def keyword_of(call: ast.Call, name: str) -> Optional[ast.AST]:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def passes_real_value(obj, callee: str, kwarg: str) -> bool:
    """Does some call to `callee` pass `kwarg` as something other than None?

    THE POINT: `assert "cancel_check=" in source` is satisfied by
    `cancel_check=None`, which is the exact Phase-3 faithfulness defect the
    whole arc exists to fix.
    """
    for call in calls_named(obj, callee):
        value = keyword_of(call, kwarg)
        if value is None:
            continue
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        return True
    return False


def first_string_arg(call: ast.Call) -> Optional[str]:
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def scopes_passed_to(obj, callee: str) -> set:
    """Scope names passed positionally to `callee` inside `obj`."""
    found = set()
    for call in calls_named(obj, callee):
        value = first_string_arg(call)
        if value:
            found.add(value)
    return found


def handler_order(obj) -> List[List[str]]:
    """Exception names per `try`, in order, GROUPED BY THE TRY THEY BELONG TO.

    `src.index("except A") < src.index("except B")` compares text positions
    across the whole function and says nothing about whether the two handlers
    are even siblings — moving one into an inner `try` preserves the order and
    changes the semantics.
    """
    groups = []
    for node in ast.walk(_tree(obj)):
        if not isinstance(node, ast.Try):
            continue
        names = []
        for handler in node.handlers:
            if handler.type is None:
                names.append("*")
            else:
                names.append(ast.unparse(handler.type))
        groups.append(names)
    return groups


def catches_before(obj, first: str, second: str) -> bool:
    """Is there a single `try` whose handlers list `first` before `second`?"""
    for names in handler_order(obj):
        has_first = [i for i, n in enumerate(names) if first in n]
        has_second = [i for i, n in enumerate(names) if second in n]
        if has_first and has_second and min(has_first) < min(has_second):
            return True
    return False


def handler_body(obj, exc_name: str) -> str:
    """Source of ONLY the named handler's body.

    `src[src.index("except X"):]` is the rest of the whole function, so a call
    that moved OUT of the handler into a later one still matches.
    """
    for node in ast.walk(_tree(obj)):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if handler.type is not None and exc_name in ast.unparse(handler.type):
                return "\n".join(ast.unparse(stmt) for stmt in handler.body)
    return ""


def guard_counts(obj, name: str, call_name: str):
    """(guarded, total) calls to `call_name` sitting under an `if` on `name`."""
    guarded, total = 0, 0
    for node in ast.walk(_tree(obj)):
        if not isinstance(node, ast.If):
            continue
        if name not in ast.unparse(node.test):
            continue
        for branch in (node.body, node.orelse):
            for stmt in branch:
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        callee = getattr(inner.func, "id", None) or getattr(
                            inner.func, "attr", None
                        )
                        if callee == call_name:
                            guarded += 1
    total = len(calls_named(obj, call_name))
    return guarded, total


def guarded_by(obj, name: str, call_name: str) -> bool:
    """Is every call to `call_name` inside an `if` whose test mentions `name`?

    `src.index("job_had_started") < src.index("rmtree")` matches the ASSIGNMENT,
    which is trivially before the call — so deleting the guard survives.
    """
    tree = _tree(obj)
    guarded, total = 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if name not in test:
            continue
        for branch in (node.body, node.orelse):
            for stmt in branch:
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        callee = getattr(inner.func, "id", None) or getattr(
                            inner.func, "attr", None
                        )
                        if callee == call_name:
                            guarded += 1
    for call in calls_named(obj, call_name):
        total += 1
    return total > 0 and guarded == total
