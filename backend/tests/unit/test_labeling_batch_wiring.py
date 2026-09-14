"""The bulk labeling path must actually REACH the batched client.

A capability is not shipped until a test fails when its wiring is removed. The
batched client had full unit coverage while nothing called it — which is this
repo's signature failure, not a hypothetical one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.services.labeling_service import LabelingService


def _feature(i):
    return SimpleNamespace(
        id=f"feat{i}", neuron_index=i, nlp_analysis=None, star_color=None
    )


class _Loop:
    """One persistent loop, set as current — like the real shared loop.

    A fresh loop per call is not equivalent: asyncio.gather() binds its futures
    to whatever loop is current WHEN IT IS CALLED, which happens inside
    _label_batch before run_until_complete ever sees them.
    """

    def __init__(self):
        import asyncio

        self.ran = []
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def run_until_complete(self, coro):
        self.ran.append(coro)
        return self._loop.run_until_complete(coro)


@pytest.fixture
def svc():
    with patch.object(LabelingService, "__init__", lambda self, db=None: None):
        return LabelingService()


def _call(svc, labeler, n=4):
    feats = [_feature(i) for i in range(n)]
    return svc._label_batch(
        labeling_service=labeler,
        loop=_Loop(),
        batch_features=feats,
        batch_examples=[[{"prime_token": "x"}] for _ in feats],
        batch_all_examples=[[] for _ in feats],
        feature_logit_effects={},
        template_config={},
        user_prompt_template="{examples_block}",
        system_message="sys",
    )


class TestBatchedClientIsReached:
    def test_bulk_labeling_calls_the_batched_client(self, svc):
        labeler = MagicMock()

        async def _batched(requests, batch_size=None):
            return [
                {"category": "c", "specific": f"l{i}", "description": ""}
                for i in range(len(requests))
            ]

        labeler.generate_labels_from_examples_batched = MagicMock(
            side_effect=_batched
        )
        with patch("src.services.labeling_service.settings") as st:
            st.labeling_batch_size = 8
            out = _call(svc, labeler, 4)

        assert labeler.generate_labels_from_examples_batched.call_count == 1, (
            "bulk labeling did not reach the batched client; the 5.6x speedup "
            "is unreachable in production"
        )
        assert len(out) == 4

    def test_the_batch_size_setting_is_passed_through(self, svc):
        """Not just 'was called' — a call with the wrong argument passes that."""
        labeler = MagicMock()

        async def _batched(requests, batch_size=None):
            return [{"category": "c", "specific": "l", "description": ""}
                    for _ in requests]

        labeler.generate_labels_from_examples_batched = MagicMock(
            side_effect=_batched
        )
        with patch("src.services.labeling_service.settings") as st:
            st.labeling_batch_size = 5
            _call(svc, labeler, 3)

        kwargs = labeler.generate_labels_from_examples_batched.call_args.kwargs
        assert kwargs["batch_size"] == 5

    def test_every_feature_appears_in_the_payload(self, svc):
        """Payload, not just call count — order and content carry the mapping."""
        seen = {}
        labeler = MagicMock()

        async def _batched(requests, batch_size=None):
            seen["ids"] = [r["feature_id"] for r in requests]
            return [{"category": "c", "specific": "l", "description": ""}
                    for _ in requests]

        labeler.generate_labels_from_examples_batched = MagicMock(
            side_effect=_batched
        )
        with patch("src.services.labeling_service.settings") as st:
            st.labeling_batch_size = 8
            _call(svc, labeler, 4)
        assert seen["ids"] == ["feat0", "feat1", "feat2", "feat3"]


class TestSerialRemainsAvailable:
    def test_batch_size_one_uses_the_per_feature_path(self, svc):
        labeler = MagicMock()
        labeler.generate_labels_from_examples_batched = MagicMock()

        async def _single(**kw):
            return {"category": "c", "specific": kw["feature_id"],
                    "description": ""}

        labeler.generate_label_from_examples = MagicMock(side_effect=_single)
        with patch("src.services.labeling_service.settings") as st:
            st.labeling_batch_size = 1
            out = _call(svc, labeler, 3)

        labeler.generate_labels_from_examples_batched.assert_not_called()
        assert labeler.generate_label_from_examples.call_count == 3
        assert [o["specific"] for o in out] == ["feat0", "feat1", "feat2"]

    def test_a_client_without_the_batched_method_still_works(self, svc):
        """An older client object must not break bulk labeling."""
        labeler = MagicMock(spec=["generate_label_from_examples"])

        async def _single(**kw):
            return {"category": "c", "specific": kw["feature_id"],
                    "description": ""}

        labeler.generate_label_from_examples = MagicMock(side_effect=_single)
        with patch("src.services.labeling_service.settings") as st:
            st.labeling_batch_size = 8
            out = _call(svc, labeler, 2)
        assert len(out) == 2


class TestTrialsStaySerial:
    def test_the_trial_service_does_not_use_the_batched_client(self):
        """Batch composition changes greedy output under int8.

        A trial exists to vary the template and nothing else, so it must not
        acquire a second variable. This is structural — the trial service calls
        generate_label_from_examples directly — and this test pins that.
        """
        src = open("src/services/labeling_trial_service.py").read()
        assert "generate_labels_from_examples_batched" not in src, (
            "the trial path reached the batched client; batch composition "
            "would become an uncontrolled variable alongside the template"
        )
        assert "_label_batch" not in src
