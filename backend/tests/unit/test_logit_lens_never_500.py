"""The logit-lens endpoint must never return 500. Ever.

It is one optional tab in a feature modal. When it cannot be computed the
honest outcome is "here is why" — not a stack trace rendered as
"HTTP error! status: 500" across the whole panel, which is what every gemma-4
SAE feature showed because the unembedding lookup knew only two flat key names.

The reason travels in `interpretation`, a field the view already renders. Empty
`top_tokens` makes the token list disappear rather than break.

These tests drive the REAL handler with real exceptions. Importing the module is
not enough: the first version of this handler referenced `datetime`, which this
module only imports inside a different function, so the code written to prevent
a 500 would have raised NameError and caused one. That is invisible to an import
check and obvious the moment the path is executed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.v1.endpoints import features as ep


def _call(side_effect=None, return_value=None):
    svc = MagicMock()
    svc.calculate_logit_lens = AsyncMock(
        side_effect=side_effect, return_value=return_value)
    with patch.object(ep, "AnalysisService", lambda db: svc):
        return asyncio.run(ep.get_logit_lens("feat_x", db=MagicMock()))


class TestNoExceptionEscapes:
    @pytest.mark.parametrize("exc", [
        ValueError("No unembedding tensor in /data/.../model.safetensors"),
        RuntimeError("CUDA out of memory"),
        KeyError("model.embed_tokens.weight"),
        FileNotFoundError("/data/models/raw/missing"),
        OSError("disk gone"),
        Exception("something nobody anticipated"),
    ])
    def test_every_failure_becomes_a_clean_response(self, exc):
        r = _call(side_effect=exc)
        assert r.top_tokens == []
        assert r.probabilities == []
        assert "unavailable" in r.interpretation.lower()

    def test_the_reason_reaches_the_user(self):
        """A bare 'unavailable' is not actionable; the cause must survive."""
        r = _call(side_effect=ValueError("No unembedding tensor in foo.safetensors"))
        assert "unembedding" in r.interpretation

    def test_computed_at_is_populated(self):
        """It is a required field; an unset one would fail response validation
        and turn the handler that prevents a 500 into one."""
        r = _call(side_effect=ValueError("x"))
        assert r.computed_at is not None


class TestTheSuccessPathIsUnchanged:
    def test_a_real_result_passes_straight_through(self):
        from datetime import datetime, timezone

        from src.schemas.feature import LogitLensResponse

        good = LogitLensResponse(
            top_tokens=["▁the", "▁a"], probabilities=[0.6, 0.4],
            interpretation="promotes determiners",
            computed_at=datetime.now(timezone.utc),
        )
        r = _call(return_value=good)
        assert r.top_tokens == ["▁the", "▁a"]
        assert r.interpretation == "promotes determiners"

    def test_a_missing_feature_is_still_404_not_a_soft_message(self):
        """404 is a real answer. Only UNEXPECTED failures degrade to a message."""
        with pytest.raises(HTTPException) as ei:
            _call(return_value=None)
        assert ei.value.status_code == 404

    def test_an_explicit_http_exception_is_not_swallowed(self):
        with pytest.raises(HTTPException) as ei:
            _call(side_effect=HTTPException(status_code=403, detail="nope"))
        assert ei.value.status_code == 403
