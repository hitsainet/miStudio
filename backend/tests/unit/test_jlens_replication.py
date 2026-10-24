"""
The replication report (BR-001).

The requirement is "published whether favourable or not", so the property under
test is that the writer HAS NO OPINION about the contents — there is no
favourable branch to take, no draft state, and no way to record a partial run as
a complete one.

MUTATION CONTROLS (each must turn this file red):
  * skip writing when the result is unfavourable   -> "unfavourable is written" fails
  * report a partial run as complete               -> "partial is visible" fails
  * coerce an unmeasured figure to 0.0             -> "not measured" fails
  * default the reference commit                   -> "commit required" fails
  * score next-token agreement                     -> "BR-004" fails
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from src.services.jlens_replication import (
    EVALUATION_SETS,
    LensResult,
    ReplicationReport,
    load_replication_report,
    save_replication_report,
)


def complete_result(lens: str, auc: float = 0.8) -> LensResult:
    return LensResult(
        lens=lens,
        pass_at_k_auc={k: auc for k in EVALUATION_SETS},
        ablation_kl=0.4,
        swap_success_rate=0.6,
    )


def test_an_unfavourable_result_is_written_exactly_like_a_favourable_one(tmp_path: Path):
    """The requirement, made structural: the writer has no branch to take."""
    bad = ReplicationReport(
        model_id="m1",
        reference_commit="abc1234",
        results=[complete_result("JACOBIAN_LENS", auc=0.05)],
        notes="the workspace claim set did not replicate at this scale",
    )
    save_replication_report(tmp_path, bad)

    loaded = load_replication_report(tmp_path)
    assert loaded is not None
    assert loaded["complete"] is True
    assert loaded["results"][0]["pass_at_k_auc"]["multihop"] == 0.05
    assert "did not replicate" in loaded["notes"]


def test_a_partial_run_is_visible_as_partial(tmp_path: Path):
    """A partial replication is a real result; presenting it as complete is not.

    Reporting only the sets that ran would show a clean table over whatever
    happened to finish.
    """
    partial = LensResult(
        lens="JACOBIAN_LENS",
        pass_at_k_auc={"multihop": 0.7, "arithmetic": 0.6},
        ablation_kl=0.3,
    )
    report = ReplicationReport("m1", "abc1234", [partial])

    assert report.is_complete is False
    assert set(report.missing["JACOBIAN_LENS"]) == set(EVALUATION_SETS) - {
        "multihop",
        "arithmetic",
    }

    save_replication_report(tmp_path, report)
    loaded = load_replication_report(tmp_path)
    assert loaded["complete"] is False
    assert loaded["missing"]["JACOBIAN_LENS"]


def test_an_unmeasured_figure_stays_null_rather_than_zero(tmp_path: Path):
    """0.0 is a score; null is "we did not measure".

    Coercing the second into the first reports the lens as having failed an
    evaluation that never ran.
    """
    report = ReplicationReport(
        "m1",
        "abc1234",
        [LensResult(lens="LOGIT_LENS", pass_at_k_auc={"multihop": 0.9})],
    )
    save_replication_report(tmp_path, report)

    raw = json.loads((tmp_path / "replication-report.json").read_text())
    auc = raw["results"][0]["pass_at_k_auc"]
    assert auc["multihop"] == 0.9
    assert auc["arithmetic"] is None
    assert raw["results"][0]["swap_success_rate"] is None


def test_every_evaluation_set_appears_even_when_unmeasured(tmp_path: Path):
    """Absence has to be visible; a missing key just looks like a shorter table."""
    report = ReplicationReport(
        "m1", "abc1234", [LensResult(lens="LOGIT_LENS", pass_at_k_auc={})]
    )
    save_replication_report(tmp_path, report)

    raw = json.loads((tmp_path / "replication-report.json").read_text())
    assert set(raw["results"][0]["pass_at_k_auc"]) == set(EVALUATION_SETS)


def test_the_reference_commit_is_required(tmp_path: Path):
    """Upstream is unmaintained, so nobody will fix a discrepancy later.

    A figure without its commit cannot be compared to anything.
    """
    with pytest.raises(ValueError, match="commit"):
        ReplicationReport("m1", "   ", [complete_result("LOGIT_LENS")])


def test_a_missing_report_is_None(tmp_path: Path):
    assert load_replication_report(tmp_path) is None


def test_a_corrupt_report_is_None_rather_than_partial(tmp_path: Path):
    (tmp_path / "replication-report.json").write_text("{not json")
    assert load_replication_report(tmp_path) is None


def test_the_report_never_scores_next_token_agreement():
    """BR-004 again, at the surface most likely to reach for it.

    A replication that scored agreement would report the J-lens as a failure,
    because it is deliberately worse on that measure than the logit lens
    through most of the network.
    """
    from src.services import jlens_replication

    tree = ast.parse(inspect.getsource(jlens_replication))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            label = node.id if isinstance(node, ast.Name) else node.attr
            assert "agreement" not in label.lower(), (
                f"{label!r} in the replication report's executable path (BR-004)"
            )
