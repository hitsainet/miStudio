"""`val_auroc` is labelled as selection-maximised, because it is not a held-out estimate.

⚠ ONE VALIDATION SPLIT DOES THREE JOBS. It ranks the layers (7 layers x 2 poolings on the
reference recipe), it picks the epoch — the MAXIMUM over up to 400 — and it is then reported as
the probe's validation AUROC. That is not leakage: the split is honest and `select_layers` fits
its standardisation on train. It is optimistic bias from repeated selection, and nothing
distinguished the result from a genuinely held-out number.

It matters because the two appear together. Stage 1 recorded **0.9982** in-distribution against
**0.8841** out of distribution. Read as the same kind of number that says a probe collapsing off
its training distribution; what it actually shows is one upper bound beside one measurement.

The honest in-distribution number is an evaluation VIEW marked `in_distribution` — scored like
any other set, never touched by training — which the evidence ladder already requires before it
will claim rung 1.

⚠ The exported `mistudio.probe-definition/v1` document does NOT carry `val_auroc` at all, and a
test here keeps it that way: the document's `evidence` block is what a consumer trusts, and a
selection-maximised number has no place in it.
"""

from __future__ import annotations

import pytest

from src.services.probe_monitor_metrics import validation_caveat


class TestTheCaveatDescribesTheNumber:
    def test_it_reports_the_value_and_flags_the_selection(self):
        out = validation_caveat({"val_auroc": 0.9982, "best_epoch": 307})
        assert out["val_auroc"] == 0.9982
        assert out["selection_maximised"] is True
        assert set(out["selected_over"]) == {"epoch", "layer"}

    def test_it_names_BOTH_selections(self):
        """⚠ Naming only the epoch would understate it. The layer sweep ranks on the same split,
        so the number survived two selections, not one."""
        out = validation_caveat({"val_auroc": 0.99})
        assert "epoch" in out["selected_over"] and "layer" in out["selected_over"]

    def test_it_points_at_the_held_out_alternative(self):
        """A caveat that says only "this is optimistic" leaves a reader with nothing to do."""
        out = validation_caveat({"val_auroc": 0.99})
        assert "in_distribution" in out["held_out_alternative"]

    def test_a_stored_note_is_preferred_over_the_default(self):
        """The wording travels with the probe, so a probe trained under different wording keeps
        the description it was actually given."""
        out = validation_caveat({"val_auroc": 0.9, "val_auroc_note": "bespoke wording"})
        assert out["note"] == "bespoke wording"

    @pytest.mark.parametrize("metrics", [None, {}, {"val_auroc": None}, {"best_epoch": 3}])
    def test_no_validation_auroc_gives_no_caveat(self, metrics):
        """A probe that never had one must not be handed a caveat about a number it lacks —
        `None` reads as "there is no such figure", which is the truth."""
        assert validation_caveat(metrics) is None


class TestTheProbeRecordsIt:
    def test_the_run_stores_the_flag_and_the_note(self):
        """The label lives on the probe, not only in the report, so anything reading the row
        directly sees it too."""
        import inspect

        from src.services import probe_monitor_run

        source = inspect.getsource(probe_monitor_run)
        assert '"val_auroc_is_selection_maximised": True' in source
        assert '"val_auroc_note"' in source

    def test_the_default_is_TRUE_when_the_probe_predates_the_flag(self):
        """⚠ Defaulting to False would silently describe every historical probe as held-out —
        a column added to stop a misattribution introducing one, which this estate has shipped
        before (`chat_format` NOT NULL DEFAULT 'auto')."""
        out = validation_caveat({"val_auroc": 0.9})     # no flag stored
        assert out["selection_maximised"] is True


class TestTheReportSurfacesIt:
    def test_the_endpoint_assembles_it(self):
        import ast
        import inspect

        from src.api.v1.endpoints import probe_monitors

        tree = ast.parse(inspect.getsource(probe_monitors.get_probe_report))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "validation_caveat" in called, (
            f"get_probe_report does not call validation_caveat; calls: {sorted(called)}"
        )

    def test_the_schema_carries_the_field(self):
        from src.schemas.probe_monitor import ProbeReport

        assert "validation_caveat" in ProbeReport.model_fields


class TestTheExportedDocumentDoesNotCarryIt:
    """⚠ THE DOCUMENT IS WHAT A CONSUMER TRUSTS. A selection-maximised number in the evidence
    block would be read as measured evidence by a reader with no way to know otherwise — and
    unlike the report, the document travels to another repository and another team."""

    def test_the_contract_has_no_val_auroc_field(self):
        from src.schemas import probe_definition

        import inspect

        source = inspect.getsource(probe_definition)
        assert "val_auroc" not in source, (
            "the probe-definition contract has gained a val_auroc field; it is "
            "selection-maximised and does not belong in evidence a consumer trusts"
        )

    def test_the_builder_does_not_export_it(self):
        import inspect

        from src.services import probe_definition_builder

        source = inspect.getsource(probe_definition_builder)
        assert "val_auroc" not in source, (
            "the export builder now reads val_auroc; the evidence block must carry measured "
            "evaluations, not a maximum over selections"
        )
