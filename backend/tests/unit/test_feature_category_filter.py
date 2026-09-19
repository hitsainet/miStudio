"""The category filter must actually filter — and must not fail open.

The MCP tool `search_features` advertised a `category` parameter and passed it
to /extractions/{id}/features. That endpoint never declared the parameter, so
FastAPI dropped it as unknown and the query ran unfiltered. The call returned
the ENTIRE extraction — 30,719 rows for category=uninterpretable — with a 200
and a `total` that looked authoritative.

That is the worst shape a filter bug can take: no error, no warning, and a
plausible answer. An agent asking "how many features are uninterpretable" got
"all of them" and nothing said otherwise.
"""

import pytest
from sqlalchemy import func

from src.schemas.feature import FeatureSearchRequest


class TestTheSchemaCarriesCategory:
    def test_category_is_accepted_and_optional(self):
        assert FeatureSearchRequest(category="semantic").category == "semantic"
        assert FeatureSearchRequest().category is None

    def test_it_survives_the_round_trip_the_endpoint_uses(self):
        """The endpoint builds this object; a dropped field there is the bug."""
        p = FeatureSearchRequest(search="x", category="uninterpretable",
                                 sort_by="name", sort_order="asc")
        assert p.category == "uninterpretable"
        assert p.search == "x"


class TestTheEndpointDeclaresIt:
    """FastAPI drops query params a signature does not name — silently."""

    def _sig_params(self, fn):
        import inspect
        return set(inspect.signature(fn).parameters)

    def test_extraction_features_endpoint_accepts_category(self):
        from src.api.v1.endpoints.features import list_extraction_features
        assert "category" in self._sig_params(list_extraction_features), (
            "the endpoint does not declare `category`, so FastAPI discards it "
            "and the response is unfiltered while looking filtered"
        )

    def test_the_training_features_endpoint_accepts_category(self):
        """The sibling /trainings/{id}/features had the same gap."""
        from src.api.v1.endpoints.features import list_features
        assert "category" in self._sig_params(list_features)

    def test_the_mcp_tool_and_the_api_agree(self):
        """The tool has always sent this; the API must accept what it sends."""
        import inspect
        from src.api.v1.endpoints.features import list_extraction_features

        src = open("src/mcp_server/tools/features.py").read()
        assert "category=category" in src, "MCP no longer forwards category"
        assert "category" in inspect.signature(list_extraction_features).parameters


class TestTheServiceAppliesIt:
    """Both the data query AND the count query must be filtered.

    Filtering one and not the other returns a page of N rows over a total of
    everything — the same lie, harder to spot.
    """

    def _service_source(self):
        return open("src/services/feature_service.py").read()

    def test_the_filter_is_applied_to_data_queries(self):
        src = self._service_source()
        assert src.count("query = query.where(\n                func.lower(Feature.category)") == 2, (
            "expected the category filter on both data queries"
        )

    def test_the_filter_is_applied_to_count_queries(self):
        src = self._service_source()
        assert src.count("count_query = count_query.where(\n                func.lower(Feature.category)") == 2, (
            "a filtered page over an unfiltered total misreports how many "
            "features match"
        )

    def test_matching_is_case_insensitive(self):
        src = self._service_source()
        assert "func.lower(Feature.category)" in src
        assert ".strip().lower()" in src


class TestEmptyValuesDoNotFilter:
    """None and "" must mean 'no filter', not 'match empty category'."""

    @pytest.mark.parametrize("value", [None, ""])
    def test_falsy_category_is_not_a_filter(self, value):
        p = FeatureSearchRequest(category=value)
        assert not p.category


class TestTheEndpointActuallyForwardsIt:
    """Declaring the parameter is not the same as using it.

    A control removing `category=category` from the FeatureSearchRequest
    construction SURVIVED the signature tests above: the endpoint accepted the
    value and dropped it on the floor. That is the original bug moved one layer
    down, so it needs a test that watches the value arrive at the service.
    """

    @pytest.mark.asyncio
    async def test_category_reaches_the_service_search_params(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.v1.endpoints import features as ep

        captured = {}

        class _Svc:
            def __init__(self, db):
                pass

            async def list_features_by_extraction(self, extraction_id, search_params):
                captured["params"] = search_params
                return MagicMock(features=[], total=0)

        with patch.object(ep, "FeatureService", _Svc):
            await ep.list_extraction_features(
                extraction_id="extr_x", search=None, sort_by="activation_freq",
                sort_order="desc", category="uninterpretable", is_favorite=None,
                limit=1, offset=0, min_activation_freq=None,
                max_activation_freq=None, min_max_activation=None,
                max_max_activation=None, db=AsyncMock(),
            )

        assert captured.get("params") is not None, "the service was never called"
        assert captured["params"].category == "uninterpretable", (
            "the endpoint accepted `category` and did not pass it to the "
            "service — the filter is dropped one layer below the signature"
        )

    @pytest.mark.asyncio
    async def test_category_reaches_the_service_from_the_training_endpoint(self):
        """The sibling handler forwards it too.

        There are TWO forwarding sites. A control aimed at the first one
        survived until this test existed — not because the guard was weak, but
        because the mutation landed in the handler the other test does not
        exercise. Both sites need their own coverage.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api.v1.endpoints import features as ep

        captured = {}

        class _Svc:
            def __init__(self, db):
                pass

            async def list_features(self, training_id, search_params):
                captured["params"] = search_params
                return MagicMock(features=[], total=0)

        # NOT a bare except. The first version of this test passed arguments
        # this handler does not take, raised TypeError, and swallowed it — so it
        # reported "the service was never called" and would have reported the
        # same for a genuinely broken forward. Only failures AFTER the service
        # call are tolerated.
        with patch.object(ep, "FeatureService", _Svc):
            await ep.list_features(
                training_id="train_x", search=None, sort_by="activation_freq",
                sort_order="desc", category="semantic", is_favorite=None,
                limit=1, offset=0, db=AsyncMock(),
            )

        assert captured.get("params") is not None, "the service was never called"
        assert captured["params"].category == "semantic"
