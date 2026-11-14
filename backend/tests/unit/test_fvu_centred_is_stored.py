"""Where each FVU is stored: the centred value beside the legacy one, never in its place.

SAE TRAINING REMEDIATION, ITEM 5 (2026-09-15). ``training_metrics.fvu`` and
``trainings.current_fvu`` keep their legacy global-mean meaning in every row; the
per-dimension-centred value goes to ``fvu_centred`` / ``current_fvu_centred``.
A swap anywhere between the SAE's losses dict and the database would put
new-scale numbers under an old-scale name (or the reverse) with every column
still populated — nothing would look wrong.

Two layers of guard:

* the WRITERS (``update_training_progress``, ``log_metric``) are exercised with
  distinct values and their payloads asserted;
* the STEP LOOP is read by its syntax tree: the centred value comes from the
  ``'fvu_centred'`` key and reaches the ``fvu_centred`` keyword of every in-sample
  writer and the progress event. A behavioural run of the loop needs extraction
  files and the rolling buffer, both being rebuilt in parallel workstreams.

NOT COVERED HERE: the held-out rows. That call site belongs to the held-out
evaluation (WS-DATA), which reports ``fvu_centred`` from its aggregator; the
integrator wires ``fvu_centred=`` there.

MUTATION CONTROLS (2026-09-15; each applied alone, this file run, source restored and
verified by sha256). All went red:
  S1 progress writes fvu_centred into current_fvu               -> test_progress_writes_each_value_to_its_own_column
  S2 metric row stores the legacy value in fvu_centred          -> test_a_metric_row_holds_each_value_in_its_own_column
  S3 step loop reads losses['fvu'] as the centred value         -> test_the_centred_value_is_read_from_its_own_key
  S4 aggregated row passes avg_fvu as fvu_centred               -> test_every_in_sample_writer_passes_the_centred_value_...
  S5 progress event sends avg_fvu under "fvu_centred"           -> test_the_progress_event_carries_both_under_their_own_names
"""

import ast
import inspect
from contextlib import contextmanager

import pytest

from src.workers import training_tasks
from src.workers.training_tasks import TrainingTask


@contextmanager
def _db(captured=None):
    class _Session:
        def add(self, obj):
            captured.append(obj)

        def commit(self):
            pass

    yield _Session()


class TestTheWritersKeepTheTwoApart:
    def test_progress_writes_each_value_to_its_own_column(self, monkeypatch):
        calls = []
        monkeypatch.setattr(training_tasks, "record_progress", lambda kind, target, **kw: calls.append(kw) or True)
        monkeypatch.setattr(TrainingTask, "get_db", lambda self: _db([]))

        TrainingTask().update_training_progress("t1", step=10, total_steps=100, loss=1.0,
                                                fvu=0.26, fvu_centred=0.32)

        assert len(calls) == 1
        assert calls[0]["current_fvu"] == 0.26
        assert calls[0]["current_fvu_centred"] == 0.32

    def test_a_missing_centred_value_does_not_erase_a_recorded_one(self, monkeypatch):
        calls = []
        monkeypatch.setattr(training_tasks, "record_progress", lambda kind, target, **kw: calls.append(kw) or True)
        monkeypatch.setattr(TrainingTask, "get_db", lambda self: _db([]))

        TrainingTask().update_training_progress("t1", step=10, total_steps=100, loss=1.0, fvu=0.26)

        assert "current_fvu_centred" not in calls[0]

    def test_a_metric_row_holds_each_value_in_its_own_column(self, monkeypatch):
        added = []
        monkeypatch.setattr(TrainingTask, "get_db", lambda self: _db(added))

        TrainingTask().log_metric("t1", step=10, loss=1.0, fvu=0.26, fvu_centred=0.32, layer_idx=11)

        [metric] = added
        assert (metric.fvu, metric.fvu_centred, metric.layer_idx) == (0.26, 0.32, 11)


def _train_body():
    tree = ast.parse(inspect.getsource(training_tasks))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")


def _calls(node, *names):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and (getattr(n.func, "attr", None) in names or getattr(n.func, "id", None) in names)
    ]


class TestTheStepLoopFeedsEachColumnItsOwnValue:
    def test_the_centred_value_is_read_from_its_own_key(self):
        body = _train_body()
        assigns = [
            n for n in ast.walk(body)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fvu_centred_val" for t in n.targets)
        ]
        assert len(assigns) == 1
        value = assigns[0].value
        assert isinstance(value, ast.Call) and getattr(value.func, "attr", None) == "get"
        assert isinstance(value.args[0], ast.Constant) and value.args[0].value == "fvu_centred"

    def test_every_in_sample_writer_passes_the_centred_value_to_the_centred_keyword(self):
        body = _train_body()
        in_sample = []
        for call in _calls(body, "log_metric", "update_training_progress"):
            keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
            if "fvu" in keywords and ("avg_fvu" in keywords["fvu"] or "layer_fvu" in keywords["fvu"]):
                in_sample.append(keywords)
        assert len(in_sample) == 3, "expected the aggregated row, the per-SAE rows and the progress update"
        for keywords in in_sample:
            assert "centred" not in keywords["fvu"], keywords
            assert "fvu_centred" in keywords and "centred" in keywords["fvu_centred"], keywords

    def test_the_progress_event_carries_both_under_their_own_names(self):
        body = _train_body()
        events = [
            n for n in ast.walk(body)
            if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "fvu" for k in n.keys)
        ]
        assert events
        for event in events:
            fields = {k.value: ast.unparse(v) for k, v in zip(event.keys, event.values) if isinstance(k, ast.Constant)}
            assert "centred" not in fields["fvu"]
            assert "centred" in fields.get("fvu_centred", "")
