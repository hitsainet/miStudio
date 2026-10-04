"""Every writer of ``training_metrics`` in the application puts the SAE's hook on per-SAE rows (review R2-B).

Migration d5a1f3c7e9b2 keys the table on (training, step, layer, COALESCE(hook_type, '')). A
per-SAE or held-out row written WITHOUT its hook is accepted on a single-hook run and collides
on a multi-hook one, and a row written with ANOTHER SAE's hook is misattributed silently. The
runtime tests (test_multi_hook_training_run, test_training_metrics_hook_key) prove the three
call sites A5 fixed; they cannot see a fourth writer added later. This reads the AST of every
module under ``src/`` for the CALLS, never the text:

* every ``log_metric(...)`` call names ``layer_idx`` and ``hook_type`` explicitly, with no
  ``**`` pass-through; a NULL layer (the aggregate) goes with a NULL hook, any other layer with
  a hook that is not the constant None, and the two come from the SAME SAE: either one ``for``
  target tuple ``(layer, hook)`` or subscripts [0] and [1] of one key;
* ``log_metric`` puts its ``hook_type`` parameter on the row it builds;
* the only other ``TrainingMetric(...)`` construction is ``TrainingService.add_metric``'s
  ``**kwargs`` pass-through, and nothing in ``src/`` calls ``add_metric``;
* there is no bulk or raw-SQL insert into the table.

MUTATION CONTROLS (R2-B; one line of src/workers/training_tasks.py broken at a time, this file
run, bytes restored, sha256 and `git diff` verified; the table is in the R2-B record):
  W1 per-SAE call: hook_type=hook_type -> hook_type=None          -> the call-site test
  W2 held-out call: hook_type=sae_key[1] -> hook_type=sae_key[0]  -> the call-site test
  W3 aggregate call: hook_type=None -> hook_type="residual"       -> the call-site test
  W4 log_metric builds the row without hook_type (K3)             -> the log_metric test
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def _trees():
    for path in sorted(SRC.rglob("*.py")):
        yield path.relative_to(SRC).as_posix(), ast.parse(path.read_text(), filename=str(path))


def _name(call: ast.Call):
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _keyword(call: ast.Call, arg: str):
    return next((k.value for k in call.keywords if k.arg == arg), None)


def _is_none(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _calls(name: str):
    """(module, call, enclosing For nodes innermost first, enclosing function name)."""
    found = []
    for module, tree in _trees():
        def visit(node, fors, function):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function = node.name
            if isinstance(node, ast.For):
                fors = [node] + fors
            if isinstance(node, ast.Call) and _name(node) == name:
                found.append((module, node, fors, function))
            for child in ast.iter_child_nodes(node):
                visit(child, fors, function)

        visit(tree, [], None)
    return found


def _same_sae(layer, hook, fors) -> bool:
    """layer and hook are the two halves of ONE SAE key."""
    # key[0] / key[1], possibly wrapped (e.g. -1 - int(sae_key[0])).
    layer_subs = [n for n in ast.walk(layer) if isinstance(n, ast.Subscript)]
    if isinstance(hook, ast.Subscript) and isinstance(hook.value, ast.Name) and isinstance(hook.slice, ast.Constant):
        return hook.slice.value == 1 and any(
            isinstance(s.value, ast.Name) and s.value.id == hook.value.id
            and isinstance(s.slice, ast.Constant) and s.slice.value == 0
            for s in layer_subs
        )
    # for layer, hook in combinations: the names are that tuple's first and second targets.
    if isinstance(layer, ast.Name) and isinstance(hook, ast.Name):
        for loop in fors:
            if isinstance(loop.target, ast.Tuple) and len(loop.target.elts) == 2:
                first, second = loop.target.elts
                if isinstance(first, ast.Name) and isinstance(second, ast.Name):
                    if (first.id, second.id) == (layer.id, hook.id):
                        return True
    return False


def test_every_log_metric_call_names_its_layer_and_the_same_saes_hook():
    calls = [(m, c, f) for m, c, f, fn in _calls("log_metric")]
    sites = []
    for module, call, fors in calls:
        where = module + ":" + str(call.lineno)
        assert not any(k.arg is None for k in call.keywords), where + " passes **kwargs: the hook cannot be checked"
        layer, hook = _keyword(call, "layer_idx"), _keyword(call, "hook_type")
        assert layer is not None, where + " does not name layer_idx"
        assert hook is not None, where + " does not name hook_type: a per-SAE row would lose its hook"
        if _is_none(layer):
            assert _is_none(hook), where + ": an aggregate row (layer NULL) must carry no hook"
            sites.append("aggregate")
        else:
            assert not _is_none(hook), where + ": a per-SAE row must carry its SAE's hook"
            assert _same_sae(layer, hook, fors), (
                where + ": layer_idx " + ast.unparse(layer) + " and hook_type " + ast.unparse(hook)
                + " are not the two halves of one SAE key"
            )
            sites.append("held-out" if "-" in ast.unparse(layer) else "per-SAE")
    # Pinned, so a new writer is reviewed here rather than slipping past.
    assert sorted(sites) == ["aggregate", "held-out", "per-SAE"], [
        (m + ":" + str(c.lineno), s) for (m, c, _), s in zip(calls, sites)
    ]


def test_log_metric_puts_its_hook_type_parameter_on_the_row():
    builds = [(m, c, fn) for m, c, _, fn in _calls("TrainingMetric") if fn == "log_metric"]
    assert len(builds) == 1, builds
    _, call, _ = builds[0]
    hook = _keyword(call, "hook_type")
    assert isinstance(hook, ast.Name) and hook.id == "hook_type", ast.dump(call)


def test_the_only_other_construction_is_add_metric_and_nothing_calls_it():
    others = sorted((m, fn) for m, _, _, fn in _calls("TrainingMetric") if fn != "log_metric")
    assert others == [("services/training_service.py", "add_metric")], others
    assert _calls("add_metric") == [], [(m, c.lineno) for m, c, _, _ in _calls("add_metric")]


def test_there_is_no_bulk_or_raw_insert_into_training_metrics():
    offenders = []
    for module, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node) in {"insert", "bulk_insert_mappings", "bulk_save_objects"}:
                if any(isinstance(a, ast.Name) and a.id == "TrainingMetric" for a in node.args):
                    offenders.append((module, node.lineno, _name(node)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = " ".join(node.value.upper().split())
                if "INSERT INTO TRAINING_METRICS" in text or "COPY TRAINING_METRICS" in text:
                    offenders.append((module, node.lineno, "raw SQL"))
    assert offenders == [], offenders
