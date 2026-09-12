"""Detection scoring against REAL activation-row shapes.

The wiring added earlier today was covered by tests that mocked assemble_items
to return [{"text": "t"}]. That fixture invented the very key the production
rows lack, so the suite was green while every real scoring run died on
KeyError('text') and the blanket except reported it as {"scored": false} — a
BROKEN measurement indistinguishable from an absent one.

Every fixture here uses the row shape the database actually produces:
prefix_tokens / prime_token / suffix_tokens / max_activation / sample_index
(labeling_service.py:428-437 and the negative-sampling SQL). None of them
carries "text". If the production code stops rendering, these go red.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services import labeling_detection_scorer as scorer
from src.services.labeling_trial_service import LabelingTrialService


def _row(prime, i, prefix=("the", "quick"), suffix=("jumped", "over")):
    """A row exactly as the DB yields it — deliberately WITHOUT 'text'."""
    return {
        "sample_index": i,
        "max_activation": 5.0 - i * 0.1,
        "prefix_tokens": ["▁" + t for t in prefix],
        "prime_token": "▁" + prime,
        "suffix_tokens": ["▁" + t for t in suffix],
        "prime_activation_index": len(prefix),
    }


def _svc():
    with patch.object(LabelingTrialService, "__init__", lambda self: None):
        s = LabelingTrialService()
    s.db = MagicMock()
    return s


def _feature(fid, ext="ext1"):
    f = MagicMock()
    f.id = fid
    f.extraction_job_id = ext
    return f


class TestRowsAreRenderedBeforeScoring:
    def test_items_reaching_the_scorer_all_carry_text(self):
        """score_feature reads item["text"]; nothing else supplies it."""
        seen = {}

        def _capture(features, judge, **kw):
            seen["items"] = [it for f in features for it in f["items"]]
            return {"panel_ba": 0.5}

        svc = _svc()
        with patch.object(scorer, "sample_negatives",
                          return_value=([_row("dose", 90)], [_row("cat", 91)])), \
             patch.object(scorer, "run_gate", return_value={"passed": True}), \
             patch.object(scorer, "score_panel", _capture), \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            svc._score_detection(
                run=MagicMock(id="r", panel_id="p"),
                features=[_feature("f1")],
                results=[{"feature_id": "f1", "status": "ok",
                          "specific": "fda_mentions", "description": "d",
                          "neuron_index": 1}],
                examples_by_feature={"f1": [_row("FDA", i) for i in range(6)]},
                labeler=MagicMock(),
            )

        assert seen.get("items"), "no items reached the scorer"
        missing = [it for it in seen["items"] if "text" not in it]
        assert not missing, (
            f"{len(missing)} items reached score_feature without a 'text' key — "
            "score_feature would raise KeyError and the panel would report "
            "scored:false"
        )
        assert all(isinstance(it["text"], str) and it["text"]
                   for it in seen["items"])

    def test_negatives_are_rendered_from_their_own_rows(self):
        """Checks the PIPELINE's output, not render_passage in isolation.

        The first version of this test called render_passage directly, so a
        mutation that rendered negatives as the constant string "NEGATIVE"
        SURVIVED it. That is a total answer leak: if the two classes differ by
        formatting, the judge separates them without reading the label and
        scores 1.0 against anything.
        """
        seen = {}

        def _capture(features, judge, **kw):
            seen["items"] = [it for f in features for it in f["items"]]
            return {"panel_ba": 0.5}

        svc = _svc()
        with patch.object(scorer, "sample_negatives",
                          return_value=([_row("dose", 90)], [_row("cat", 91)])), \
             patch.object(scorer, "run_gate", return_value={"passed": True}), \
             patch.object(scorer, "score_panel", _capture), \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            svc._score_detection(
                run=MagicMock(id="r", panel_id="p"),
                features=[_feature("f1")],
                results=[{"feature_id": "f1", "status": "ok",
                          "specific": "fda_mentions", "description": "d",
                          "neuron_index": 1}],
                examples_by_feature={"f1": [_row("FDA", i) for i in range(6)]},
                labeler=MagicMock(),
            )

        texts = [it["text"] for it in seen["items"]]
        # Each negative's own prime token must survive into its passage.
        assert any("dose" in t for t in texts), (
            "the hard negative's own content is absent from its rendered text; "
            "negatives are not being rendered from their rows"
        )
        assert any("cat" in t for t in texts), (
            "the easy negative's own content is absent from its rendered text"
        )
        # And no class may be a constant.
        assert len(set(texts)) > 1, "all passages rendered identically"
        for t in texts:
            for marker in ("<<", ">>", "max_activation"):
                assert marker not in t, (
                    f"{marker!r} leaked into a passage; the judge could score "
                    "1.0 against any label at all"
                )

    def test_the_real_row_shape_has_no_text_key(self):
        """Guards the fixture itself.

        If someone 'helpfully' adds text to the DB row shape, the test above
        stops proving anything — it would pass without any rendering.
        """
        assert "text" not in _row("FDA", 0), (
            "the fixture now carries 'text', so it can no longer detect a "
            "missing render_passage call"
        )


class TestShapeErrorsAreNotReportedAsJudgeFailures:
    def test_a_missing_key_is_flagged_as_a_wiring_error(self):
        svc = _svc()
        with patch.object(scorer, "sample_negatives",
                          side_effect=KeyError("prime_token")):
            out = svc._score_detection(
                run=MagicMock(id="r", panel_id="p"),
                features=[_feature("f1")],
                results=[{"feature_id": "f1", "status": "ok",
                          "specific": "x", "description": "d",
                          "neuron_index": 1}],
                examples_by_feature={"f1": [_row("FDA", 0)]},
                labeler=MagicMock(),
            )
        assert out["scored"] is False
        assert out.get("wiring_error") is True, (
            "a data-shape bug was reported as an ordinary scoring failure; "
            "that is how KeyError('text') went unnoticed"
        )
        assert "wiring error" in out["reason"]

    def test_a_genuine_judge_failure_is_not_flagged_as_wiring(self):
        svc = _svc()
        with patch.object(scorer, "sample_negatives",
                          return_value=([_row("a", 9)], [_row("b", 8)])), \
             patch.object(scorer, "run_gate",
                          side_effect=RuntimeError("judge exploded")), \
             patch.object(LabelingTrialService, "_build_judge",
                          return_value=lambda p: "[1]"):
            out = svc._score_detection(
                run=MagicMock(id="r", panel_id="p"),
                features=[_feature("f1")],
                results=[{"feature_id": "f1", "status": "ok",
                          "specific": "x", "description": "d",
                          "neuron_index": 1}],
                examples_by_feature={"f1": [_row("FDA", i) for i in range(6)]},
                labeler=MagicMock(),
            )
        assert out["scored"] is False
        assert not out.get("wiring_error")
        assert "judge exploded" in out["reason"]
