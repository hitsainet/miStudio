"""Template lens (BR-023, 027 task 4.3): directions for multi-token concepts.

The day-one contract fields (`TemplateLensSpec`) had no compute path behind them.
"""

from types import SimpleNamespace

import pytest
import torch

from src.services import jlens_template_lens as tl


def _states(n, d, shift=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    if shift is not None:
        x = x + shift
    return x


class TestTheMath:
    def test_the_direction_finds_the_planted_shift_after_whitening(self):
        d = 12
        g = torch.Generator().manual_seed(1)
        scale = torch.ones(d)
        scale[0] = 10.0  # a high-variance axis the raw mean difference would favour
        baseline = torch.randn(400, d, generator=g) * scale
        shift = torch.zeros(d)
        shift[0] = 2.0   # in the noisy axis: small in std units
        shift[3] = 1.0   # in a quiet axis: large in std units
        phrase = torch.randn(200, d, generator=g) * scale + shift
        (t,) = tl.build_templates({"New York": {4: phrase}}, {4: baseline}, ridge=0.01)
        assert t.direction.norm().item() == pytest.approx(1.0, abs=1e-5)
        assert abs(t.direction[3]) > abs(t.direction[0]), (
            "whitening must favour the axis where the shift is large in std units"
        )
        assert t.separation > 1.0 and t.n_contexts == 200 and t.layer == 4

    def test_too_few_baseline_contexts_are_refused(self):
        with pytest.raises(ValueError, match="at least"):
            tl.whitening_operator(_states(tl.MIN_BASELINE - 1, 6))

    def test_filtered_words_are_refused_not_built(self):
        with pytest.raises(ValueError, match="filter list"):
            tl.build_templates({"the": {0: _states(5, 4)}}, {0: _states(20, 4)},
                               filter_words=["The"])

    def test_identical_contexts_are_refused(self):
        base = _states(20, 4)
        with pytest.raises(ValueError, match="indistinguishable"):
            mu, inv, cov, _ = tl.whitening_operator(base)
            tl.template_direction("x", 0, mu.unsqueeze(0), mu, inv, cov)


class TestTheTaskBody:
    def test_final_positions_whitened_and_the_limitations_carried(self, monkeypatch):
        from src.workers import jlens_probe_tasks as tasks

        d = 6

        class _Service:
            def __init__(self, **kw):
                self.capture_device = "cpu"
                self.tokenizer = lambda text, return_tensors=None: {
                    "input_ids": torch.tensor([[len(text), 1, 2]])}

            def _capture_residuals(self, ids, layers):
                n = int(ids[0, 0])
                g = torch.Generator().manual_seed(n)
                h = torch.randn(3, d, generator=g)
                if n >= 100:      # phrase contexts: shifted at the LAST position only
                    h[-1, 2] += 5.0
                return SimpleNamespace(by_layer={L: h for L in layers})

        monkeypatch.setattr("src.services.jlens_readout_service.ReadoutService", _Service)
        monkeypatch.setattr("src.workers.jlens_progress.update_row", lambda *a, **k: True)
        me = SimpleNamespace(update_state=lambda **k: None, request=SimpleNamespace(id="t"))
        loaded = SimpleNamespace(model=None, tokenizer=None, unembedding=None, name="tiny",
                                 structure=SimpleNamespace(num_layers=4))
        out = tasks._template_lens(
            me, loaded=loaded,
            vocabulary={"New York": ["x" * (100 + i) for i in range(10)]},
            baseline_contexts=["y" * (10 + i) for i in range(20)],
            layers=[1], ridge=0.1, filter_words=None,
        )
        (entry,) = out["templates"]
        assert entry["direction_available"] is True and entry["n_contexts"] == 10
        direction = torch.tensor(entry["layers"]["1"]["direction"])
        assert int(direction.abs().argmax()) == 2, "the shift lives in dimension 2"
        assert out["limitations"] == list(tl.LIMITATIONS) and len(out["limitations"]) == 3
        with pytest.raises(ValueError, match="layers"):
            tasks._template_lens(me, loaded=loaded, vocabulary={"a": ["b"]},
                                 baseline_contexts=["c"] * 8, layers=[9], ridge=0.1,
                                 filter_words=None)


def test_the_route_is_served_and_bounded():
    from pydantic import ValidationError

    from src.api.v1.endpoints.jlens import TemplateLensRequest
    from src.main import app

    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/jlens/template-lens"]
    assert "get" in paths["/api/v1/jlens/template-lens/{task_id}"]
    with pytest.raises(ValidationError):
        TemplateLensRequest(model_id="m", vocabulary={f"p{i}": ["c"] for i in range(9)},
                            baseline_contexts=["b"] * 8, layers=list(range(8)))


def test_the_route_queues_exactly_what_was_asked():
    import asyncio
    from unittest.mock import MagicMock, patch

    from src.api.v1.endpoints.jlens import TemplateLensRequest, template_lens
    from tests.unit.test_jlens_gpu_placement import _db_with_model

    queued = MagicMock(return_value=MagicMock(id="t-tl"))
    with patch("src.api.v1.endpoints.jlens.gpu_delay", return_value=queued), \
            patch("src.workers.jlens_progress.open_row") as open_row, \
            patch("src.api.v1.endpoints.jlens.resolve_gpu_request", return_value="auto"):
        out = asyncio.run(template_lens(
            TemplateLensRequest(model_id="m_1", vocabulary={"New York": ["I flew to"]},
                                baseline_contexts=["b"] * 8, layers=[3], ridge=0.2,
                                filter_words=["the"]),
            db=_db_with_model(),
        ))
    assert out.task_id == "t-tl" and queued.call_count == 1
    sent = queued.call_args.kwargs
    assert sent["vocabulary"] == {"New York": ["I flew to"]} and sent["layers"] == [3]
    assert sent["ridge"] == 0.2 and sent["filter_words"] == ["the"]
    assert open_row.call_args.args[0] == "jlens_template_lens"
