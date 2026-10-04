"""Calibration runs ONE forward per layer, and every bar it places is bit-identical to the old passes.

The calibrating stage used to run, for every probe, a base pass plus one pass per contract window:
each re-read the corpus, re-rendered the same rows and ran a full forward. `pmr_416ce6b1ee63`
(9 probes over 3 layers) paid 36 forwards at ~482 s — 17,346 s of a run whose answer needed 3.

⚠ THE REFACTOR IS ONLY SAFE IF NOTHING MOVES, AND bf16 MAKES THAT A REAL RISK. bf16 is not
batch-invariant: the same row scored in a different batch differs by up to 0.18. So the claim here
is not "close" — every persisted negative and every threshold must be EQUAL to what the old
one-probe, one-scope passes produced. The reference below is that old shape, built from the
public single-probe scorer the old stage called.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.ml.probe_monitor_model import ProbeHead
from src.services import probe_monitor_capture, probe_monitor_run
from src.services.probe_monitor_capture import (
    ScoreSpec,
    forward_scores,
    forward_scores_by_scope,
    plan_batches,
)
from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES, RenderedExample

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

D_MODEL = 32
LAYERS = (1, 2)
TARGET_FPR = 0.05


@pytest.fixture(scope="module")
def tiny_model():
    """A real `LlamaForCausalLM` with random weights: 4 layers, d=32."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _conversation(rng: np.random.Generator) -> RenderedExample:
    """system, user, assistant[, user, assistant] — so `input` and `last_assistant` really differ."""
    ids, roles, messages = [1], [""], [-1]  # a BOS no message claims
    turns = ["system", "user", "assistant"] + (["user", "assistant"] if rng.random() < 0.5 else [])
    for index, role in enumerate(turns):
        n = int(rng.integers(2, 14))
        ids += [int(v) for v in rng.integers(2, 64, size=n)]
        roles += [role] * n
        messages += [index] * n
    return RenderedExample(input_ids=ids, token_roles=roles, token_message=messages, text="")


@pytest.fixture(scope="module")
def rows():
    rng = np.random.default_rng(7)
    rendered = [_conversation(rng) for _ in range(60)]
    # Mostly negatives, with a few excluded rows that must NOT be counted as negatives.
    labels = [0 if i % 7 else 2 for i in range(len(rendered))]
    return rendered, labels


def _head(layer: int, seed: int) -> ProbeHead:
    generator = torch.Generator().manual_seed(seed)
    return ProbeHead(
        weight=torch.randn(D_MODEL, generator=generator),
        bias=0.1,
        mean=torch.zeros(D_MODEL),
        std=torch.ones(D_MODEL),
        attention_query=torch.randn(D_MODEL, generator=generator),
        layer=layer,
    )


RULES = [("mean", {}), ("rolling_mean_max", {"window": 3}), ("max", {})]


def _probes():
    """Three rules on each of two layers: the shape of a real multi-layer, multi-rule run."""
    out = []
    for layer in LAYERS:
        for offset, (rule, params) in enumerate(RULES):
            probe = SimpleNamespace(
                id=f"pm_l{layer}_{rule}", variant="dense", threshold=None, definition_path=None,
            )
            trained = SimpleNamespace(rule=rule, rule_params=dict(params))
            out.append((probe, _head(layer, seed=10 * layer + offset), trained, [], []))
    return out


def _context(tmp_path: Path, scope: str = "all") -> probe_monitor_run.RunContext:
    return probe_monitor_run.RunContext(
        run_id="pmr_test", config={}, artifact_dir=tmp_path, seed=0, scope=scope,
        max_length=256, target_fpr=TARGET_FPR, rules=[r for r, _ in RULES], rolling_windows=[3],
        top_n_layers=2, val_fraction=0.2,
    )


class TestOneForwardScoresEveryScope:
    """The scorer: several scopes off one forward equal several one-scope forwards, exactly."""

    @pytest.mark.parametrize("budget", [16_384, 60])
    def test_each_scope_equals_its_own_forward_bit_for_bit(self, tiny_model, rows, budget):
        rendered, _ = rows
        assert len(plan_batches(rendered, token_budget=60)) > 5, "the small budget must split"
        specs = [ScoreSpec(head=_head(1, s), rule=r, rule_params=p) for s, (r, p) in enumerate(RULES)]
        scopes = list(CONTRACT_WINDOW_SCOPES.values())
        together = forward_scores_by_scope(
            tiny_model, rendered, specs, scopes=scopes, token_budget=budget
        )
        for scope in scopes:
            for index, spec in enumerate(specs):
                alone = forward_scores(
                    tiny_model, rendered, spec.head, rule=spec.rule, rule_params=spec.rule_params,
                    scope=scope, token_budget=budget,
                )
                assert [r.aggregate for r in together[scope][index]] == [r.aggregate for r in alone]
                assert [r.n_scored for r in together[scope][index]] == [r.n_scored for r in alone]
                assert [r.token_scores for r in together[scope][index]] == [r.token_scores for r in alone]

    def test_the_scopes_really_differ(self, tiny_model, rows):
        """Specificity: if every scope selected the same tokens, the test above would prove nothing."""
        rendered, _ = rows
        spec = ScoreSpec(head=_head(1, 0), rule="mean")
        out = forward_scores_by_scope(
            tiny_model, rendered, [spec], scopes=list(CONTRACT_WINDOW_SCOPES.values())
        )
        counts = {scope: [r.n_scored for r in rows_[0]] for scope, rows_ in out.items()}
        assert counts["all"] != counts["input"]
        assert counts["all"] != counts["last_assistant"]
        assert counts["input"] != counts["last_assistant"]

    def test_dropping_token_traces_keeps_aggregates_and_counts(self, tiny_model, rows):
        rendered, _ = rows
        spec = ScoreSpec(head=_head(2, 3), rule="rolling_mean_max", rule_params={"window": 3})
        kept = forward_scores_by_scope(tiny_model, rendered, [spec], scopes=["input"])["input"][0]
        dropped = forward_scores_by_scope(
            tiny_model, rendered, [spec], scopes=["input"], keep_token_scores=False
        )["input"][0]
        assert [r.aggregate for r in dropped] == [r.aggregate for r in kept]
        assert [r.n_scored for r in dropped] == [r.n_scored for r in kept]
        assert all(r.token_scores == [] for r in dropped)

    def test_the_model_runs_once_per_batch_not_once_per_scope(self, tiny_model, rows):
        rendered, _ = rows
        calls = []
        handle = tiny_model.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            forward_scores_by_scope(
                tiny_model, rendered, [ScoreSpec(head=_head(1, 0), rule="mean")],
                scopes=list(CONTRACT_WINDOW_SCOPES.values()), token_budget=60,
            )
        finally:
            handle.remove()
        assert len(calls) == len(plan_batches(rendered, token_budget=60))

    def test_the_batches_are_exactly_the_planned_ones(self, tiny_model, rows):
        """⚠ THE ONE PROPERTY THE COMPARISONS ABOVE CANNOT SEE. They compare this scorer with the
        single-probe form, which now runs through the same loop — so a change to the batch plan
        inside it would move both sides together and stay green, while moving every bf16
        threshold against what the old code persisted. So the batches the MODEL receives are
        pinned to `plan_batches` directly: same rows, same order, same padded width."""
        rendered, _ = rows
        seen = []

        def record(module, args, kwargs):
            seen.append(kwargs["input_ids"].tolist())

        handle = tiny_model.register_forward_pre_hook(record, with_kwargs=True)
        try:
            forward_scores_by_scope(
                tiny_model, rendered, [ScoreSpec(head=_head(1, 0), rule="mean")],
                scopes=list(CONTRACT_WINDOW_SCOPES.values()), token_budget=60,
            )
        finally:
            handle.remove()
        # The exact ids, row by row and in order — a permutation inside a batch, or two batches of
        # equal width swapped, keeps every shape and still moves a bf16 score.
        planned = []
        for batch in plan_batches(rendered, token_budget=60):
            width = max(len(rendered[i].input_ids) for i in batch)
            planned.append([
                rendered[i].input_ids + [0] * (width - len(rendered[i].input_ids)) for i in batch
            ])
        assert seen == planned

    def test_repeated_or_empty_scopes_are_refused(self):
        spec = ScoreSpec(head=_head(1, 0), rule="mean")
        with pytest.raises(ValueError, match="distinct"):
            forward_scores_by_scope(object(), [], [spec], scopes=["all", "all"])
        with pytest.raises(ValueError, match="no scopes"):
            forward_scores_by_scope(object(), [], [spec], scopes=[])


def _reference(model, rows, probes, context, windows=tuple(CONTRACT_WINDOW_SCOPES)):
    """THE OLD STAGE'S ARITHMETIC: one single-probe, single-scope forward per pass."""
    from src.services.probe_monitor_metrics import length_band_decisions
    from src.services.probe_monitor_trainer import calibrate

    rendered, labels = rows
    out = {}
    for probe, head, trained, _v, _l in probes:
        def scored(scope):
            result = forward_scores(
                model, rendered, head, rule=trained.rule, rule_params=trained.rule_params,
                scope=scope,
            )
            negatives = [r.aggregate for r, label in zip(result, labels) if label == 0]
            lengths = [r.n_scored for r, label in zip(result, labels) if label == 0]
            return negatives, lengths

        base, lengths = scored(context.scope)
        cal = calibrate(base, target_fpr=context.target_fpr, source="calibration_set")
        windows = {}
        for window in windows:
            scope = CONTRACT_WINDOW_SCOPES[window]
            negatives, _ = scored(scope)
            windows[window] = (
                negatives,
                calibrate(negatives, target_fpr=context.target_fpr, source="calibration_set"),
            )
        out[probe.id] = dict(
            base=base, lengths=lengths, calibration=cal, windows=windows,
            bands=length_band_decisions(
                base, lengths, target_fpr=context.target_fpr, global_threshold=cal.threshold
            ),
        )
    return out


class TestTheStagePlacesTheSameBars:
    """The whole calibrating stage, run for real, against the old shape's arithmetic."""

    def _run(self, monkeypatch, tiny_model, rows, tmp_path, scope="all"):
        monkeypatch.setattr(
            probe_monitor_run, "_prepare_calibration_rows", lambda db, ds, ctx, tok: rows
        )
        probes = _probes()
        context = _context(tmp_path, scope)
        beats = []
        probe_monitor_run._calibrate_probes(
            None, probes, "pmd_cal", context, tiny_model, SimpleNamespace(pad_token_id=0),
            "llama", beat=lambda done, total: beats.append((done, total)),
        )
        return probes, context, beats

    @pytest.mark.parametrize("scope", ["all", "user"])
    def test_every_bar_and_array_is_identical(self, monkeypatch, tiny_model, rows, tmp_path, scope):
        probes, context, _ = self._run(monkeypatch, tiny_model, rows, tmp_path, scope)
        expected = _reference(tiny_model, rows, probes, context)
        for probe, *_ in probes:
            want = expected[probe.id]
            assert probe.threshold == want["calibration"].threshold
            assert probe.realised_fpr == want["calibration"].realised_fpr
            assert probe.threshold_source == "calibration_set"
            assert np.array_equal(np.load(probe.calibration_scores_path), np.asarray(want["base"], dtype=np.float32))
            assert probe.length_bands == want["bands"]
            for window, (negatives, cal) in want["windows"].items():
                got = probe.window_decisions[window]
                assert got["threshold"] == cal.threshold
                assert got["realised_fpr"] == cal.realised_fpr
                assert got["n_negatives"] == len(negatives)
                assert got["scope"] == CONTRACT_WINDOW_SCOPES[window]
                assert np.array_equal(
                    np.load(got["scores_path"]), np.asarray(negatives, dtype=np.float32)
                )

    def test_the_lengths_come_from_the_run_scope_and_nowhere_else(self, monkeypatch, tiny_model, rows, tmp_path):
        """A correctness guard: lengths from another window describe a different span, and a band
        table built over them would be plausible and wrong."""
        probes, context, _ = self._run(monkeypatch, tiny_model, rows, tmp_path)
        expected = _reference(tiny_model, rows, probes, context)
        for probe, *_ in probes:
            stored = np.load(probe.calibration_lengths_path).tolist()
            assert stored == [float(v) for v in expected[probe.id]["lengths"]]

    def test_excluded_rows_are_not_negatives(self, monkeypatch, tiny_model, rows, tmp_path):
        probes, _, _ = self._run(monkeypatch, tiny_model, rows, tmp_path)
        n_negatives = sum(1 for label in rows[1] if label == 0)
        assert n_negatives < len(rows[1])
        for probe, *_ in probes:
            assert len(np.load(probe.calibration_scores_path)) == n_negatives

    def test_one_forward_per_layer_per_batch(self, monkeypatch, tiny_model, rows, tmp_path):
        rendered, _ = rows
        calls = []
        handle = tiny_model.register_forward_pre_hook(lambda module, args: calls.append(1))
        try:
            self._run(monkeypatch, tiny_model, rows, tmp_path)
        finally:
            handle.remove()
        assert len(calls) == len(LAYERS) * len(plan_batches(rendered))

    def test_the_rows_are_prepared_once(self, monkeypatch, tiny_model, rows, tmp_path):
        prepared = []

        def counting(db, ds, ctx, tok):
            prepared.append(ds)
            return rows

        monkeypatch.setattr(probe_monitor_run, "_prepare_calibration_rows", counting)
        probe_monitor_run._calibrate_probes(
            None, _probes(), "pmd_cal", _context(tmp_path), tiny_model,
            SimpleNamespace(pad_token_id=0), "llama",
        )
        assert prepared == ["pmd_cal"]

    def test_progress_beats_reach_the_end_and_never_go_backwards(self, monkeypatch, tiny_model, rows, tmp_path):
        _, _, beats = self._run(monkeypatch, tiny_model, rows, tmp_path)
        assert beats, "the stage wrote nothing while it worked"
        done = [d for d, _ in beats]
        assert done == sorted(done)
        assert beats[-1][0] == beats[-1][1]
        assert {total for _, total in beats} == {len(LAYERS) * 1000}

    def test_without_a_calibration_set_nothing_is_scored(self, monkeypatch, tiny_model, tmp_path):
        monkeypatch.setattr(
            probe_monitor_run, "_prepare_calibration_rows",
            lambda *a: pytest.fail("no calibration set, so nothing may be prepared"),
        )
        monkeypatch.setattr(probe_monitor_run, "_aggregate_rows", lambda *a: [])
        probes = _probes()
        probe_monitor_run._calibrate_probes(
            None, probes, None, _context(tmp_path), tiny_model, SimpleNamespace(pad_token_id=0),
            "llama",
        )
        assert all(p.threshold is None for p, *_ in probes)


def _no_reply(rng: np.random.Generator) -> RenderedExample:
    """A row with no assistant turn: its `last_assistant` mask is empty."""
    ids, roles, messages = [1], [""], [-1]
    for index, role in enumerate(["system", "user"]):
        n = int(rng.integers(2, 10))
        ids += [int(v) for v in rng.integers(2, 64, size=n)]
        roles += [role] * n
        messages += [index] * n
    return RenderedExample(input_ids=ids, token_roles=roles, token_message=messages, text="")


class TestAWindowThatCannotBeScoredIsAbsentNotFatal:
    """⚠ REVIEW ROUND 1 (HIGH). One forward per window used to fail ALONE: the old window loop
    caught it, left that window out and placed every other bar. Sharing the forward first let
    `combine`'s empty-row refusal escape and fail the whole stage — every probe on every layer
    losing its bar, after hours of training — on any calibration row with no reply."""

    @pytest.fixture
    def mixed_rows(self, rows):
        rendered, labels = rows
        rng = np.random.default_rng(11)
        return rendered + [_no_reply(rng) for _ in range(5)], labels + [0] * 5

    def test_the_scorer_drops_only_the_scope_it_cannot_score(self, tiny_model, mixed_rows):
        rendered, _ = mixed_rows
        out = forward_scores_by_scope(
            tiny_model, rendered, [ScoreSpec(head=_head(1, 0), rule="mean")],
            scopes=["all", "input", "last_assistant"], required_scopes=["all"],
        )
        assert set(out) == {"all", "input"}
        assert len(out["all"][0]) == len(rendered)

    def test_a_required_scope_still_raises(self, tiny_model, mixed_rows):
        rendered, _ = mixed_rows
        with pytest.raises(ValueError, match="no unmasked tokens"):
            forward_scores_by_scope(
                tiny_model, rendered, [ScoreSpec(head=_head(1, 0), rule="mean")],
                scopes=["all", "last_assistant"], required_scopes=["last_assistant"],
            )

    def test_by_default_every_scope_is_required(self, tiny_model, mixed_rows):
        """Evaluation and offline scoring keep their old behaviour: a refusal is a refusal."""
        rendered, _ = mixed_rows
        with pytest.raises(ValueError):
            forward_scores_by_scope(
                tiny_model, rendered, [ScoreSpec(head=_head(1, 0), rule="mean")],
                scopes=["last_assistant"],
            )

    def test_the_stage_places_every_other_bar(self, monkeypatch, tiny_model, mixed_rows, tmp_path):
        monkeypatch.setattr(
            probe_monitor_run, "_prepare_calibration_rows", lambda db, ds, ctx, tok: mixed_rows
        )
        probes = _probes()
        context = _context(tmp_path)
        probe_monitor_run._calibrate_probes(
            None, probes, "pmd_cal", context, tiny_model, SimpleNamespace(pad_token_id=0), "llama",
        )
        expected = _reference(tiny_model, mixed_rows, [p for p in probes], context, windows=("all", "prompt"))
        for probe, *_ in probes:
            assert probe.threshold == expected[probe.id]["calibration"].threshold
            assert set(probe.window_decisions) == {"all", "prompt"}, (
                "the response window could not be scored and must be ABSENT, with the others kept"
            )
            assert probe.length_bands == expected[probe.id]["bands"]


class TestTheStageIsWired:
    """`execute_probe_run` must call the stage, or none of the above runs in production."""

    def test_the_gpu_recut_arm_scores_its_windows_in_one_call(self):
        """The arm re-scores one probe's windows. It used to run a forward per window."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(probe_monitor_run.recut_probe_windows_on_gpu))
        names = [
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert names.count("_score_calibration_group") == 1
        assert names.count("_prepare_calibration_rows") == 1
        assert "_calibrate_windows" in names

    def test_execute_probe_run_calls_calibrate_probes(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(probe_monitor_run.execute_probe_run))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_calibrate_probes"
        ]
        assert len(calls) == 1
        assert any(k.arg == "beat" for k in calls[0].keywords), "the stage must report progress"

    def test_the_old_per_probe_scorer_is_gone(self):
        """Leaving it would let a future caller reintroduce four forwards per probe silently."""
        assert not hasattr(probe_monitor_run, "_calibration_negative_scores")

    def test_forward_scores_many_is_a_view_of_the_scoped_scorer(self):
        """One scoring path: the single-scope form must not grow its own forward."""
        import inspect

        source = inspect.getsource(probe_monitor_capture.forward_scores_many)
        assert "forward_scores_by_scope" in source and "register_hooks" not in source
