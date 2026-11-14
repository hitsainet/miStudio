"""A capture's duration estimate times the capture's own work, finalisation included.

016 FPRD §8 criterion 1 asks for the estimate within ±30% of the wall clock. On hardware
(2026-10-17, ``cap_207942e2ce97``: LFM2.5-1.2B, SAEs L12+L13, 2,000 x 512 tokens) it read
**1.9 min against 317 s** (created 12:36:39 → completed 12:41:56 UTC). The store's file
times split that run: load + probe ~10 s (store directory at 12:36:49), batches ~77 s (L12's
finalise ends 12:40:01.6, L13's runs 12:40:01.6 → 12:41:56.3 = 115 s, so L12's began
~12:38:06), and **~230 s finalising** — sorting 95.7M events at 99% progress, which the
old ``probe_seconds / PROBE_SAMPLES x sample_cap`` did not count at all.

The estimate now rehearses the real capture batch and finalise on the probe documents and
scales them: batches linearly, the finalise sort by a measured power law. These tests pin the
model on the measured run, the scaling law, the wiring into ``run_capture``, and that the
faster sort orders events exactly as the structured ``argsort`` did.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.services import circuit_capture_service as cap
from src.services import circuit_capture_store as store
from tests.unit import test_circuits_run_split as S

MEASURED_WALL_SECONDS = 317.5          # 12:36:39.30 → 12:41:56.84 UTC
MEASURED_EVENTS = 95_651_640
PROBE_EVENTS = 1_503_820                # the run's own probe (32 docs, both layers)
OLD_ESTIMATE_MINUTES = 1.9


def _measured_case_inputs():
    """The run's inputs to the new model. Two are derived, and say how.

    - per document: the batches' 77 s over 2,000 documents (store file times);
    - rehearsal finalise: ``EventWriter.finalize`` (structured argsort, as the run used) took
      0.95 s for 0.75M events on the development host, which finalised 47.8M in 112.3 s
      against the node's 115 s — so x 115 / 112.3 on the node, both layers. (Re-measure on
      hardware: the GPU was owned by another run when this was written.)
    """
    return dict(setup_seconds=14.0, per_doc_seconds=77.0 / 2000, sample_cap=2000,
                finalize_seconds=2 * 0.95 * 115 / 112.3, finalize_events=PROBE_EVENTS,
                events_est=93_988_750, n_layers=2)


class TestTheMeasuredRun:
    def test_the_old_estimate_missed_by_more_than_the_criterion(self):
        """The record, not a test of code: 1.9 min against 317 s is -64%."""
        assert abs(OLD_ESTIMATE_MINUTES * 60 / MEASURED_WALL_SECONDS - 1) > 0.30

    def test_the_model_lands_within_thirty_percent(self):
        est = cap.estimate_capture_seconds(**_measured_case_inputs())
        assert abs(est["total_seconds"] / MEASURED_WALL_SECONDS - 1) <= 0.30, est

    def test_finalisation_is_most_of_it(self):
        """The term that was missing is the largest one on this run."""
        est = cap.estimate_capture_seconds(**_measured_case_inputs())
        assert est["finalize"] > est["capture"] > est["setup"]
        assert 180 < est["finalize"] < 260

    def test_the_breakdown_adds_up(self):
        est = cap.estimate_capture_seconds(**_measured_case_inputs())
        assert est["total_seconds"] == pytest.approx(
            est["setup"] + est["capture"] + est["finalize"], abs=0.2)


class TestScaling:
    BASE = dict(setup_seconds=0.0, per_doc_seconds=0.0, sample_cap=0,
                finalize_seconds=1.0, finalize_events=1_000_000, n_layers=1)

    def test_finalise_grows_faster_than_linearly(self):
        """A sort: ten times the events costs more than ten times the seconds."""
        one = cap.estimate_capture_seconds(**self.BASE, events_est=1_000_000)["finalize"]
        ten = cap.estimate_capture_seconds(**self.BASE, events_est=10_000_000)["finalize"]
        assert one == pytest.approx(1.0, abs=0.05)
        assert ten == pytest.approx(10 ** cap.FINALIZE_SCALING_EXPONENT, rel=0.01)
        assert ten > 10 * math.log2(1e7) / math.log2(1e6), "n log n under-read the 3090 run"

    def test_events_are_scaled_per_layer(self):
        """Each layer sorts its own events, so two layers of N cost 2 x f(N), not f(2N)."""
        one = cap.estimate_capture_seconds(**{**self.BASE, "n_layers": 1,
                                              "finalize_events": 1_000_000},
                                           events_est=8_000_000)["finalize"]
        two = cap.estimate_capture_seconds(**{**self.BASE, "n_layers": 2,
                                              "finalize_events": 2_000_000},
                                           events_est=16_000_000)["finalize"]
        assert two == pytest.approx(one, rel=0.01)

    def test_capture_is_linear_in_documents(self):
        est = cap.estimate_capture_seconds(
            setup_seconds=3.0, per_doc_seconds=0.05, sample_cap=2000, finalize_seconds=0.0,
            finalize_events=0, events_est=0, n_layers=1)
        assert est["capture"] == pytest.approx(100.0)
        assert est["total_seconds"] == pytest.approx(103.0)

    def test_no_events_means_no_finalise_term(self):
        est = cap.estimate_capture_seconds(
            setup_seconds=0.0, per_doc_seconds=0.0, sample_cap=10, finalize_seconds=0.5,
            finalize_events=100, events_est=0, n_layers=1)
        assert est["finalize"] == 0.0


def _capture_estimate(monkeypatch, rehearsal=None):
    """Drive the real ``run_capture`` (confirmed=False) on a tiny CPU Llama."""
    model = S._llama()
    rows = S._rows(layers=(1, 2))
    db = S._Db(rows)
    saes = {1: S._sae(1), 2: S._sae(2)}
    patches = [patch("datasets.load_from_disk", return_value=S._dataset()),
               patch("src.ml.model_loader.load_model_from_hf",
                     return_value=(model, S.TOKENIZER, None, {})),
               patch.object(cap, "_load_sae_sync",
                            side_effect=lambda record, device: saes[record.layer]),
               patch.object(cap, "_sae_fingerprint", return_value="fp"),
               patch("src.services.extraction_service.cleanup_gpu_memory")]
    if rehearsal is not None:
        patches.append(patch.object(cap, "_rehearse_capture", return_value=rehearsal))
    for p in patches:
        p.start()
    try:
        out = cap.CircuitCaptureService.run_capture(db, "cap_1", confirmed=False,
                                                    device=S.CPU)
    finally:
        for p in patches:
            p.stop()
    return out


class TestTheEstimateIsWired:
    def test_the_rehearsal_reaches_the_minutes_the_user_confirms_against(self, monkeypatch):
        """A 10-minute rehearsed finalise must show in ``minutes``, or it is decoration."""
        out = _capture_estimate(monkeypatch, rehearsal={
            "per_doc_seconds": 0.0, "docs": 3, "finalize_seconds": 600.0,
            "events": 10})
        est = out["estimate"]
        assert out["status"] == "estimated"
        assert est["duration_seconds"]["finalize"] > 0
        assert est["minutes"] == pytest.approx(
            est["duration_seconds"]["total_seconds"] / 60.0, abs=0.06)
        assert est["minutes"] >= 10.0

    def test_the_real_rehearsal_runs_the_capture_batch(self, monkeypatch):
        """Unpatched: the rehearsal really captures and finalises events on the probe docs."""
        seen = {}
        real = cap._rehearse_capture

        def spy(*a, **k):
            seen.update(real(*a, **k))
            return seen

        with patch.object(cap, "_rehearse_capture", side_effect=spy):
            out = _capture_estimate(monkeypatch)
        assert seen["docs"] == 3
        assert seen["events"] > 0, "the rehearsal wrote nothing; it timed an empty loop"
        assert seen["per_doc_seconds"] > 0
        assert out["estimate"]["duration_seconds"]["total_seconds"] > 0


class TestTheFasterSortIsTheSameSort:
    @pytest.mark.parametrize("writer", [store.EventWriter, store.ErrNormWriter,
                                        store.AttnTopKWriter])
    def test_identical_to_the_structured_argsort_including_ties(self, writer):
        rng = np.random.default_rng(7)
        n = 20_000
        data = np.empty(n, dtype=writer.dtype)
        for name in writer.dtype.names:
            kind = writer.dtype[name]
            if kind.kind == "f":
                data[name] = rng.integers(0, 4, n).astype(kind)   # many exact ties
            else:
                data[name] = rng.integers(0, 6, n).astype(kind)
        expected = data[np.argsort(data, order=list(writer.sort_keys), kind="stable")]
        got = data[store._sort_order(data, writer.sort_keys)]
        assert np.array_equal(got, expected)

    def test_the_writer_finalises_through_it(self, tmp_path):
        w = store.EventWriter(tmp_path / "layer_0.events")
        w.append(np.array([2, 1, 1]), np.array([0, 5, 1]), np.array([9, 9, 3]),
                 np.array([0.5, 0.25, 1.0]))
        with patch.object(store, "_sort_order", wraps=store._sort_order) as spy:
            w.finalize()
        assert spy.call_count == 1
        out = np.load(tmp_path / "layer_0.events")
        assert out["feature_idx"].tolist() == [3, 9, 9]
        assert out["doc_id"].tolist() == [1, 1, 2]
        assert out["token_pos"].tolist() == [1, 5, 0]
