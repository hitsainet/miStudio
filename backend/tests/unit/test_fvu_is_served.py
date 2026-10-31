"""FVU must reach the API response, not merely the database.

The migration added trainings.current_fvu and the WebSocket carried it, but the
first attempt inserted the schema field into TrainingUpdate — an INPUT model —
because a regex matched the first `current_l0_sparsity` in the file. The column
existed, the WS worked, and the REST payload silently lacked the field, so a
page load showed no FVU until the next live event arrived.

Verified against the deployed API, which returned:
    current_fvu in payload: False

These tests assert the RESPONSE contract, which is the thing the UI reads.
"""

import pytest

from src.schemas.training import TrainingResponse, TrainingUpdate


class TestTheResponseModelCarriesFvu:
    def test_training_response_declares_current_fvu(self):
        assert "current_fvu" in TrainingResponse.model_fields, (
            "TrainingResponse has no current_fvu; the UI reads this payload on "
            "page load and would show nothing until a WebSocket event arrived"
        )

    def test_it_sits_beside_the_other_live_metrics(self):
        """If l0/dead are served, fvu must be too — they populate one card."""
        f = TrainingResponse.model_fields
        for peer in ("current_loss", "current_l0_sparsity", "current_dead_neurons"):
            assert peer in f
        assert "current_fvu" in f

    def test_it_is_optional_so_non_reporting_runs_still_serialise(self):
        """Architectures that do not compute FVU send NULL, not 0."""
        assert TrainingResponse.model_fields["current_fvu"].default is None

    def test_a_payload_without_fvu_still_validates(self):
        """Historical rows predate the column and must not 500 the endpoint."""
        obj = TrainingResponse.model_construct(current_fvu=None)
        assert obj.current_fvu is None


class TestRoundTrip:
    def test_a_reported_value_survives_serialisation(self):
        obj = TrainingResponse.model_construct(current_fvu=0.0972)
        dumped = obj.model_dump()
        assert dumped["current_fvu"] == pytest.approx(0.0972)

    def test_none_is_preserved_rather_than_becoming_zero(self):
        """0.0 means perfect reconstruction; None means not reported."""
        dumped = TrainingResponse.model_construct(current_fvu=None).model_dump()
        assert dumped["current_fvu"] is None
        assert dumped["current_fvu"] != 0.0
