"""ENH-006 G15: a probe training run's TIME forecast, made before the operator approves the run.

The defect: a Llama run listed a 1,000,000-row everyday view among its evaluation sets; scoring every set
whole meant an ETA of 347 hours, found only after hours of GPU time. The memory forecast said nothing about
time, and the approval card the operator decided from said nothing either.

The pure model is tested on REAL run records (`tests/fixtures/probe_run_time_history.json`, read over the
read-only MCP tools on 2026-11-09), so a fixture whose fields agree by construction cannot hide a wrong unit.
The wiring tests (route, MCP tool, approval, card) are in this file and `test_mcp_approval_forecast.py`.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from src.services import probe_run_time_forecast as tf

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "probe_run_time_history.json").read_text()
)
RUNS = FIXTURE["runs"]
VIEWS = {view_id: tf.view_rows(counts) for view_id, counts in FIXTURE["view_counts"].items()}
LLAMA = "m_40e78d80"
CARD_3090 = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


def _run(run_id):
    return next(run for run in RUNS if run["id"] == run_id)


def _before(run):
    return [r for r in RUNS if (r.get("completed_at") or "9") < run["created_at"]]


def _forecast_of(run, history=None, **overrides):
    """What the preflight would have said for `run`, from the runs completed before it was submitted."""
    batch = (run["environment"].get("memory_forecast") or {}).get("batches_bytes")
    kwargs = dict(
        units=tf.units_of_run(run, VIEWS), history=_before(run) if history is None else history,
        rows_by_view=VIEWS, model_id=run["model_id"], scope=run["config"]["scope"], batch_bytes=batch,
        eval_max_rows=run["config"].get("eval_max_rows"), eval_dataset_ids=run["eval_dataset_ids"],
        calibration_dataset_id=run["calibration_dataset_id"],
    )
    kwargs.update(overrides)
    return tf.forecast_time(**kwargs)


class TestTheWorkUnits:
    """Each stage's unit, counted from what a run RECORDED — the run the brief names, `pmr_2da5b9c6af8f`."""

    def test_the_named_run_counts_as_its_record_says(self):
        units = tf.units_of_run(_run("pmr_2da5b9c6af8f"), VIEWS)
        assert units.rows == 33550
        assert units.scored_tokens == 4257928
        assert units.tokens == 4257928 + 1006500, "the template-prefix positions are forwarded too"
        assert units.swept_layers == 9 and units.chosen_layers == 1
        # mean + max + rolling_mean_max at windows 16 and 32: one probe each, the windowed rule one per window.
        assert units.probes == 4
        assert units.calibration_rows == 7500
        # Six sets; the 1,000,000-row everyday view is sampled to eval_max_rows 5,000.
        assert units.eval_rows_by_set["pmd_b1f459c4693d"] == 5000
        assert units.eval_rows == 1443 + 2645 + 2325 + 2482 + 5000 + 2000

    def test_each_stage_multiplies_its_own_factors(self):
        per = tf.units_of_run(_run("pmr_2da5b9c6af8f"), VIEWS).per_stage()
        assert per == {
            "rendering": 33550,
            "pooled_capture": 5264428,
            "layer_selection": 9 * 33550,
            "token_capture": 1 * 5264428,
            "training": 4 * 4257928,
            "calibrating": 1 * 7500,
            "evaluating": 1 * 15895,
        }

    @pytest.mark.parametrize("config,chosen,expected", [
        ({"rules": ["mean", "max", "attention"], "rolling_windows": [16, 32]}, 1, 3),
        ({"rules": ["mean", "rolling_mean_max"], "rolling_windows": [32, 64]}, 3, 9),
        ({"rules": ["rolling_mean_max"]}, 1, 1),
        ({"rules": ["mean"], "sae_variant": True, "sae_k": [64, 128]}, 2, 6),
    ])
    def test_probes_follow_rule_windows(self, config, chosen, expected):
        assert tf.probes_trained(config, chosen) == expected

    def test_no_eval_max_rows_scores_every_set_whole(self):
        total, by_set = tf.eval_rows_of(["pmd_b1f459c4693d", "pmd_03e7d9c82e6b"], VIEWS, None)
        assert by_set == {"pmd_b1f459c4693d": 1_000_000, "pmd_03e7d9c82e6b": 2000} and total == 1_002_000

    def test_an_unknown_set_size_is_unknown_not_zero(self):
        total, by_set = tf.eval_rows_of(["pmd_b1f459c4693d", "pmd_gone"], VIEWS, 5000)
        assert total is None and by_set["pmd_gone"] is None

    def test_no_calibration_view_runs_no_forward(self):
        assert tf.calibration_rows_of(None, VIEWS, {}) == 0
        assert tf.calibration_rows_of("pmd_ca6b3814c34a", VIEWS, {}) == tf.LEGACY_CALIBRATION_MAX_ROWS

    def test_a_request_counts_like_a_run(self):
        """The preflight's memory answer and a run's render summary carry the same figures; counted alike."""
        run = _run("pmr_2da5b9c6af8f")
        memory = {"checked": True, "rows": 33550, "real_tokens": 4257928, "template_prefix_skipped": 1006500,
                  "swept_layers": 9}
        request = {"config": run["config"], "eval_dataset_ids": run["eval_dataset_ids"],
                   "calibration_dataset_id": run["calibration_dataset_id"]}
        assert tf.units_of_request(request, memory, VIEWS) == tf.units_of_run(run, VIEWS)

    def test_an_unchecked_memory_answer_leaves_the_token_work_unknown(self):
        units = tf.units_of_request({"config": {}}, {"checked": False, "reason": "x"}, VIEWS)
        assert units.rows is None and units.tokens is None and units.scored_tokens is None


class TestTheForecast:
    def test_the_named_run_from_the_runs_before_it(self):
        """The backtest row for `pmr_2da5b9c6af8f` (7 h 51 min measured): basis, device and range."""
        run = _run("pmr_2da5b9c6af8f")
        answer = _forecast_of(run)
        assert answer["forecast"] is True and answer["complete"] is True
        assert [r["run_id"] for r in answer["basis"]["runs"]] == [
            "pmr_55715c1c87e2", "pmr_64ea4248247a", "pmr_a9568b05afce", "pmr_dae885c03d6c", "pmr_192c5725e9c0"]
        assert answer["training_device"]["expected"] == "cpu"
        actual = sum(v for k, v in run["environment"]["stage_seconds"].items() if k in tf.FORECAST_STAGES)
        # Measured on the real records: typical 5.63 h, range 3.80-7.69 h, against 7.84 h actual. The total
        # range misses by 2% because CPU training (15,615 s) was slower per unit than any run before it.
        assert answer["seconds"]["typical"] == pytest.approx(20253, rel=0.01)
        assert answer["seconds"]["low"] < answer["seconds"]["typical"] < answer["seconds"]["high"] < actual
        # The token-driven stages are within 10% of what the run then measured.
        for stage in ("pooled_capture", "token_capture", "calibrating", "evaluating"):
            measured = run["environment"]["stage_seconds"][stage]
            assert answer["stages"][stage]["seconds"]["typical"] == pytest.approx(measured, rel=0.10), stage

    def test_the_range_is_the_min_median_and_max_rate(self):
        run = _run("pmr_2da5b9c6af8f")
        stage = _forecast_of(run)["stages"]["pooled_capture"]
        rates = [r["seconds"] / r["units"] for r in stage["rates"]]
        work = stage["units"]
        assert stage["seconds"] == {"low": round(work * min(rates), 1),
                                    "typical": round(work * statistics.median(rates), 1),
                                    "high": round(work * max(rates), 1)}
        assert stage["unit"] == tf.STAGE_UNITS["pooled_capture"]

    def test_the_total_is_the_sum_of_the_stages(self):
        answer = _forecast_of(_run("pmr_2da5b9c6af8f"))
        for key in ("low", "typical", "high"):
            assert answer["seconds"][key] == pytest.approx(
                sum(s["seconds"][key] for s in answer["stages"].values()), abs=0.5)

    def test_the_g15_run_shape_is_flagged_in_hours_before_it_starts(self):
        """The defect: the 1,000,000-row view scored whole. Without `eval_max_rows` the evaluating stage alone
        forecasts days — the number the operator never saw before approving."""
        run = dict(_run("pmr_2da5b9c6af8f"))
        run["config"] = {**run["config"], "eval_max_rows": None}
        answer = _forecast_of(run, history=_before(_run("pmr_2da5b9c6af8f")), eval_max_rows=None)
        assert answer["eval_rows_by_set"]["pmd_b1f459c4693d"] == 1_000_000
        assert answer["stages"]["evaluating"]["seconds"]["typical"] > 72 * 3600
        assert "every set whole" in answer["summary"]

    def test_no_comparable_run_gives_no_number(self):
        answer = _forecast_of(_run("pmr_2da5b9c6af8f"), history=[])
        assert answer["forecast"] is False
        assert "seconds" not in answer and "hours" not in answer
        assert answer["summary"].startswith("No time forecast:")
        assert answer["eval_rows"] == 15895, "the rows it would score are still said"

    def test_another_model_is_not_comparable(self):
        run = _run("pmr_2da5b9c6af8f")
        others = [{**r, "model_id": "m_other"} for r in _before(run)]
        assert _forecast_of(run, history=others)["forecast"] is False

    def test_another_scope_is_not_comparable(self):
        run = _run("pmr_2da5b9c6af8f")
        others = [{**r, "config": {**r["config"], "scope": "last_user"}} for r in _before(run)]
        assert _forecast_of(run, history=others)["forecast"] is False

    def test_a_run_without_stage_timings_is_not_a_measurement(self):
        run = _run("pmr_2da5b9c6af8f")
        bare = [{**r, "environment": {**r["environment"], "stage_seconds": {}}} for r in _before(run)]
        assert _forecast_of(run, history=bare)["forecast"] is False

    def test_only_the_most_recent_runs_per_stage(self):
        run = _run("pmr_2da5b9c6af8f")
        answer = _forecast_of(run, runs_per_stage=2)
        assert [r["run_id"] for r in answer["stages"]["pooled_capture"]["rates"]] == [
            "pmr_55715c1c87e2", "pmr_64ea4248247a"]

    def test_evaluating_takes_its_rate_from_runs_that_scored_the_same_sets_first(self):
        """`pmr_8a54cf9c1112` is older and scored other sets; the newer runs scored the same six."""
        run = _run("pmr_2da5b9c6af8f")
        rates = _forecast_of(run, runs_per_stage=1)["stages"]["evaluating"]["rates"]
        assert rates[0]["run_id"] == "pmr_64ea4248247a", "the one earlier run with the identical six sets"

    def test_unseen_evaluation_sets_are_said(self):
        run = _run("pmr_192c5725e9c0")
        older = [_run("pmr_8a54cf9c1112")]
        answer = _forecast_of(run, history=older)
        assert answer["caveats"] and "never scored" in answer["caveats"][0]
        assert "may be off" in answer["summary"]


class TestTrainingDevicesAreKeptApart:
    def test_a_run_without_the_record_trained_on_the_cpu(self):
        assert tf.training_device_of_run({"environment": {}}) == "cpu"

    def test_mixed_layers_are_no_measurement_of_either(self):
        mixed = {"environment": {"training_device": {"1": {"device": "cuda:0"}, "2": {"device": "cpu"}}}}
        assert tf.training_device_of_run(mixed) is None

    def test_the_expected_device_is_the_runs_own_rule_on_measured_free_memory(self):
        run = _run("pmr_2da5b9c6af8f")
        small = tf.expected_training_device(1 << 30, _before(run))
        large = tf.expected_training_device(35815931904, _before(run))
        assert small["expected"] == "gpu" and large["expected"] == "cpu"

    def test_a_gpu_forecast_uses_no_cpu_rate(self):
        """Every earlier Llama run trained on the CPU: a run whose batches fit the card has NO training rate."""
        run = _run("pmr_2da5b9c6af8f")
        answer = _forecast_of(run, batch_bytes=1 << 30)
        assert answer["training_device"]["expected"] == "gpu"
        assert "training" in answer["missing_stages"] and answer["complete"] is False
        assert answer["summary"].startswith("At least ")
        assert "Not measured, so not counted: training" in answer["summary"]


class TestTheCard:
    def test_a_named_card_narrows_the_runs_to_it(self):
        run = _run("pmr_2da5b9c6af8f")
        history = _before(run) + [{**_run("pmr_55715c1c87e2"), "id": "pmr_other_card", "gpu_uuid": "GPU-other",
                                   "completed_at": "2026-11-08T00:00:00Z"}]
        pinned = _forecast_of(run, history=history, gpu=CARD_3090)
        assert pinned["basis"]["card"] == {"pinned": True, "cards": [CARD_3090]}
        assert "pmr_other_card" not in [r["run_id"] for r in pinned["basis"]["runs"]]
        auto = _forecast_of(run, history=history, gpu="auto")
        assert auto["basis"]["card"]["pinned"] is False
        assert set(auto["basis"]["card"]["cards"]) == {CARD_3090, "GPU-other"}

    def test_a_named_card_with_no_runs_says_so(self):
        run = _run("pmr_2da5b9c6af8f")
        answer = _forecast_of(run, gpu="GPU-unseen")
        assert answer["basis"]["card"]["pinned"] is False and "GPU-unseen" in answer["basis"]["card"]["note"]


class TestTheSummary:
    def test_one_line_with_the_range_the_basis_and_the_rows(self):
        summary = _forecast_of(_run("pmr_2da5b9c6af8f"))["summary"]
        assert "\n" not in summary
        assert summary.startswith("About 5.6 h (3.8 h to 7.7 h)")
        assert "5 completed run(s) of this model at scope all" in summary
        assert "Evaluation scores 15,895 rows over 6 set(s), each capped at eval_max_rows 5,000" in summary
        assert "Training expected on the CPU" in summary
        assert "Least certain: training" in summary
