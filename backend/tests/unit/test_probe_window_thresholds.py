"""A threshold per window, because a threshold is a quantile of ONE window's negatives.

miLLM can read one probe's weights over `all`, `prompt` and `response` separately — the prompt
says something about the user, the response about the model, and a mean over both answers
neither. But the operating point does not travel with them: `threshold` is the
`(1 - target_fpr)` quantile of negatives **aggregated under one window**, so over a different
window the same number is a quantile of a different distribution and no longer names the same
false-positive rate.

This estate has already paid for the general form of that mistake. A high-stakes probe fired on
*"What is the capital of France?"* because its 1% budget had been spent on plain-prose negatives
and then applied to chat, and the recorded lesson was that *"a false-positive rate is a property
of the negative distribution the monitor will actually see"*. A narrower window is a different
negative distribution by exactly that argument.

⚠ AND THE CONTRACT'S `prompt` IS NOT miSTUDIO'S `user`. `prompt` is positional — `[0, n_prompt)` —
which holds the system preamble and every earlier assistant turn. `user` excludes both. The
internal scope `input` exists to name the split a served request actually has, and
`CONTRACT_WINDOW_SCOPES` is the one place that mapping lives.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.schemas.probe_definition import Decision, WindowDecision
from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES


class _Probe:
    def __init__(self, window_decisions=None):
        self.window_decisions = window_decisions


def _decisions(probe):
    from src.services.probe_definition_builder import _window_decisions

    return _window_decisions(probe)


class TestTheWindowMapping:
    def test_prompt_means_input_not_user(self):
        """⚠ THE CROSS-REPO DISAGREEMENT, PINNED IN THE ONE PLACE IT IS DECIDED. Calibrating a
        `prompt` bar under `user` cuts it for a narrower window than miLLM reads."""
        assert CONTRACT_WINDOW_SCOPES["prompt"] == "input"
        assert CONTRACT_WINDOW_SCOPES["prompt"] != "user"

    def test_response_is_the_LAST_assistant_turn(self):
        """Earlier assistant turns are context the model was conditioned on, not what it produced
        this time — and they sit inside the consumer's `prompt` span, not its `response`."""
        assert CONTRACT_WINDOW_SCOPES["response"] == "last_assistant"

    def test_every_contract_window_is_mapped(self):
        assert set(CONTRACT_WINDOW_SCOPES) == {"all", "prompt", "response"}


class TestTheContractRefusesAnEmptyClaim:
    def test_a_window_entry_without_a_threshold_is_refused(self):
        """An entry claiming a window was calibrated while placing no bar is indistinguishable
        from "not calibrated" to a consumer, while looking like more evidence."""
        with pytest.raises(ValidationError, match="carry no threshold"):
            Decision(threshold=1.0, windows={"prompt": WindowDecision(threshold=None)})

    def test_a_populated_entry_is_accepted(self):
        d = Decision(threshold=1.0, windows={"prompt": WindowDecision(threshold=4.5)})
        assert d.windows["prompt"].threshold == 4.5

    def test_None_and_empty_are_different(self):
        """`None` means never attempted, `{}` means attempted and nothing placed. Only the first
        is a reason for a consumer to assume the single threshold transfers."""
        assert Decision(threshold=1.0).windows is None
        assert Decision(threshold=1.0, windows={}).windows == {}

    def test_an_unknown_window_name_is_refused(self):
        with pytest.raises(ValidationError):
            Decision(threshold=1.0, windows={"middle": WindowDecision(threshold=1.0)})


class TestTheExporterDropsEmptyEntries:
    def test_it_exports_what_was_calibrated(self):
        got = _decisions(_Probe({
            "prompt": {"threshold": 4.5, "target_fpr": 0.01, "realised_fpr": 0.008,
                       "n_negatives": 620, "scope": "input"},
        }))
        assert set(got) == {"prompt"}
        assert got["prompt"].threshold == 4.5
        assert got["prompt"].scope == "input", (
            "the internal scope is not recorded, so a reader cannot tell what `prompt` meant here"
        )

    def test_an_entry_with_no_threshold_is_DROPPED_not_exported_as_null(self):
        """⚠ Exporting it would fail the contract's own validator at the last moment instead of
        here — and a null threshold reaching a consumer that defaulted it to 0.0 would turn a
        probe that said nothing into one firing on half its input."""
        got = _decisions(_Probe({
            "prompt": {"threshold": 4.5},
            "response": {"threshold": None, "n_negatives": 0},
        }))
        assert set(got) == {"prompt"}

    @pytest.mark.parametrize("stored", [None, {}, "nonsense", {"prompt": None},
                                        {"prompt": {"threshold": None}}])
    def test_nothing_calibrated_exports_None_not_an_empty_dict(self, stored):
        assert _decisions(_Probe(stored)) is None

    def test_a_probe_row_without_the_column_is_tolerated(self):
        """A row from before the column existed. `getattr` with no attribute must not raise on
        the export path."""
        class _Old:
            pass

        assert _decisions(_Old()) is None


class TestTheCalibrationStageReportsWhileItWorks:
    """⚠ WRITTEN FROM A LIVE RUN, NOT FROM READING. On `pmr_15c9e85a1d04` the `calibrating` stage
    sat at a fixed **85%** for over half an hour, writing nothing to the row, because per-window
    thresholds turned one scoring pass into `probes x (1 + windows)` passes over the whole
    calibration corpus — about seven minutes each.

    Two failures in that sentence:

    * **Silence.** A stage that writes nothing for half an hour is indistinguishable from a dead
      worker, and this estate has already reaped a LIVE 5.8-hour packing job on exactly that.
    * **A useless number.** The only way to answer "which pass is it on?" was to grep a worker log
      for a line that happens to be emitted once per pass. That is what an operator actually had
      to do, mid-run, to answer the question.

    The scorer has always taken a progress callback; the calibration path simply never passed one.

    ⚠ AND SINCE 2026-10-03 THE STAGE RUNS ONE FORWARD PER LAYER, not one per probe per window
    (`test_calibration_one_forward_per_layer.py`), so "a pass" is now a layer's forward. The
    guarantees below are the same ones, re-pinned on the new shape — by running the code rather
    than by reading its source.
    """

    @staticmethod
    def _context(artifact_dir=None, scope="all"):
        # ⚠ A stand-in must never be more forgiving than the thing it stands in for: the helper
        # reads `target_fpr`, `scope` and `artifact_dir`, and each was once discovered by a crash.
        return SimpleNamespace(
            target_fpr=0.01,
            scope=scope,
            artifact_dir=Path(artifact_dir) if artifact_dir is not None else Path(mkdtemp()),
        )

    @staticmethod
    def _scores(scopes=("all", "input", "last_assistant")):
        from src.services.probe_monitor_run import CalibrationScores

        return {
            scope: CalibrationScores(negatives=[1.0, 2.0, 3.0], n_rows=3, lengths=[10, 20, 30])
            for scope in scopes
        }

    @staticmethod
    def _probe(probe_id="pm_x"):
        probe = _Probe()
        probe.id = probe_id
        probe.threshold = None
        probe.definition_path = None
        return probe

    def test_the_base_equivalent_pass_LEAVES_ITS_LENGTHS_BEHIND(self):
        """⚠ The window loop persists the per-row lengths from the scope that equals the run's,
        which is what makes a per-length bar re-cuttable without the GPU. Before that, the re-cut
        arm re-scored three windows, saved them, and was then refused by `propose` over the
        lengths it had not kept."""
        from src.services import probe_monitor_run as run_mod

        probe = self._probe()
        run_mod._calibrate_windows(probe, self._context(), "calibration_set", self._scores())
        assert probe.calibration_lengths_path, (
            "the base-equivalent window scored its lengths and discarded them, so a "
            "per-length bar still cannot move without the model"
        )

    def test_every_window_gets_a_decision(self):
        from src.services import probe_monitor_run as run_mod

        decisions = run_mod._calibrate_windows(
            self._probe(), self._context(), "calibration_set", self._scores()
        )
        assert set(decisions) == set(CONTRACT_WINDOW_SCOPES)

    def test_it_reports_WITHIN_a_forward_not_only_between(self, monkeypatch):
        """⚠ The load-bearing half. Ticking only between forwards still leaves a long gap, and
        the reaper reads the GAP, not the total."""
        from src.services import probe_monitor_run as run_mod

        def fake_group(db, rows, group, scopes, model, tokenizer, architecture, progress=None,
                       required_scopes=()):
            # Two batches per forward, as a real forward over a corpus would report.
            progress(1, 2)
            progress(2, 2)
            return [self._scores(scopes) for _ in group]

        monkeypatch.setattr(run_mod, "_prepare_calibration_rows", lambda *a: ([], []))
        monkeypatch.setattr(run_mod, "_score_calibration_group", fake_group)
        probes = [
            (self._probe(f"pm_{layer}"), SimpleNamespace(layer=layer),
             SimpleNamespace(rule="mean", rule_params={}), [], [])
            for layer in (3, 7)
        ]
        beats = []
        run_mod._calibrate_probes(
            None, probes, "pmd_cal", self._context(), None, None, "llama",
            beat=lambda done, total: beats.append((done, total)),
        )
        within = [done for done, total in beats if done % 1000]
        assert within, (
            f"no beat landed inside a forward ({beats}), so a long forward writes nothing to the "
            f"row while it runs"
        )

    def test_the_SCORER_is_actually_given_the_callback(self, monkeypatch):
        """⚠ WRITTEN BECAUSE A CONTROL SURVIVED: a stub upstream reports progress because the stub
        was written to, not because the production path does. So this runs the real
        `_score_calibration_group` and catches what it hands the real scorer."""
        from src.services import probe_monitor_capture
        from src.services import probe_monitor_run as run_mod

        seen = {}

        def fake_scorer(model, rendered, specs, **kwargs):
            seen.update(kwargs)
            return {scope: [[] for _ in specs] for scope in kwargs["scopes"]}

        monkeypatch.setattr(probe_monitor_capture, "forward_scores_by_scope", fake_scorer)

        def callback(done, total):
            return None

        run_mod._score_calibration_group(
            None, ([], []),
            [(SimpleNamespace(variant="dense"), SimpleNamespace(layer=1),
              SimpleNamespace(rule="mean", rule_params={}))],
            ["all"], None, SimpleNamespace(pad_token_id=0), "llama", progress=callback,
        )
        assert seen.get("progress") is callback
        assert seen.get("keep_token_scores") is False
