"""Sparse decomposition by gradient pursuit (022 tasks 5.3, 5.6, 6.3; BR-008 mode c, BR-009).

`DecompositionResult` was declared, validated and tested at the SCHEMA level while nothing
produced one — no solver, no service, no `/decompose` route (confirmed 2026-09-26 against
the served schema). These tests cover the solver's math, the one property FPRD §3.3 makes
normative (pursuit is NOT top-k by inner product), the provenance contract, and the route
in the app's served schema.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.schemas.jlens import DecompositionResult
from src.services import jlens_decomposition as jd
from src.services.jlens_readout_service import IdentityTransport, JacobianTransport


def _unit(v):
    return v / v.norm()


@pytest.fixture
def frame():
    torch.manual_seed(0)
    return torch.randn(300, 24)


class TestPursuitRecoversPlantedComponents:
    def test_logit_lens(self, frame):
        d = jd.LensDictionary(frame, IdentityTransport(), layer=2)
        h = 2.0 * _unit(frame[5]) + 1.0 * _unit(frame[17])
        r = jd.gradient_pursuit(h, d, k=2)
        assert r.active == [5, 17]
        # Gradient pursuit is an APPROXIMATE solver (one line-searched gradient step
        # per iteration, not a least-squares refit): the support is exact, the
        # coefficients converge to within the tolerance it records.
        assert r.coefficients == pytest.approx([2.0, 1.0], abs=1e-2)
        assert r.fve > 0.9999

    def test_jacobian_lens_atoms_are_pulled_back_through_J_transpose(self, frame):
        torch.manual_seed(1)
        J = torch.randn(24, 24)
        d = jd.LensDictionary(frame, JacobianTransport({4: J}), layer=4)
        atom = lambda t: _unit(J.T @ frame[t])  # noqa: E731
        h = 3.0 * atom(9) + 1.0 * atom(40)
        r = jd.gradient_pursuit(h, d, k=2, max_iterations=60)
        assert r.active == [9, 40]
        assert r.coefficients == pytest.approx([3.0, 1.0], abs=2e-2)
        assert r.fve > 0.999

    def test_coefficients_are_nonnegative_and_the_support_never_exceeds_k(self, frame):
        d = jd.LensDictionary(frame, IdentityTransport(), layer=0)
        torch.manual_seed(3)
        r = jd.gradient_pursuit(torch.randn(24), d, k=5)
        assert len(r.active) <= 5
        assert all(c > 0 for c in r.coefficients)
        assert len(set(r.active)) == len(r.active)

    def test_a_zero_residual_is_an_empty_exact_decomposition(self, frame):
        d = jd.LensDictionary(frame, IdentityTransport(), layer=0)
        r = jd.gradient_pursuit(torch.zeros(24), d, k=3)
        assert r.active == [] and r.residual_norm == 0.0


class TestPursuitIsNotTopK:
    """022 task 5.6 — FPRD §3.3 forbids substituting top-k by inner product."""

    def test_they_return_different_active_sets_on_a_correlated_frame(self):
        torch.manual_seed(7)
        d_model = 32
        frame = torch.randn(50, d_model)
        a = _unit(torch.randn(d_model))
        b = _unit(torch.randn(d_model))
        b = _unit(b - (b @ a) * a)  # orthogonal second component
        frame[0] = a
        frame[1] = _unit(a + 0.08 * _unit(torch.randn(d_model)))  # near-duplicate of a
        frame[2] = b
        h = 3.0 * a + 1.0 * b
        dictionary = jd.LensDictionary(frame, IdentityTransport(), layer=0)

        top = set(jd.top_k_by_inner_product(h, dictionary, 2))
        pursued = set(jd.gradient_pursuit(h, dictionary, k=2, max_iterations=80).active)

        assert top == {0, 1}, "precondition: the near-duplicate outranks the second component"
        assert pursued == {0, 2}, (
            "pursuit must re-select against the RESIDUAL, so the duplicate of an atom "
            "already in the support stops looking attractive"
        )


class TestTheDictionaryIsNeverFormed:
    def test_atom_norms_are_computed_in_bounded_chunks(self, frame, monkeypatch):
        J = torch.randn(24, 24)
        transport = JacobianTransport({1: J})
        widths = []
        real = transport.pullback

        def spy(v, layer):
            widths.append(v.shape[-1] if v.ndim == 2 else 1)
            return real(v, layer)

        monkeypatch.setattr(transport, "pullback", spy)
        monkeypatch.setattr(jd, "NORM_CHUNK_ROWS", 64)
        d = jd.LensDictionary(frame, transport, layer=1)
        assert max(widths) <= 64
        assert d.norms == pytest.approx((frame @ J).norm(dim=1), rel=1e-4)

    def test_synthesis_touches_only_the_active_rows(self, frame):
        d = jd.LensDictionary(frame, IdentityTransport(), layer=0)
        out = d.synthesize([3], torch.tensor([2.0]))
        assert torch.allclose(out, 2.0 * _unit(frame[3]), atol=1e-6)


class TestProvenanceAndTheControl:
    def _row(self, frame, seed=11):
        d = jd.LensDictionary(frame, IdentityTransport(), layer=6)
        h = 2.0 * _unit(frame[5]) + 0.5 * _unit(frame[8])
        return jd.decompose(h, d, k=4, control_seed=seed, decode=lambda i: f"t{i}", position=3)

    def test_the_row_validates_against_the_contract_with_its_seed(self, frame):
        row = self._row(frame)
        parsed = DecompositionResult.model_validate(row)
        assert parsed.provenance.control_seed == 11
        assert parsed.provenance.solver == jd.SOLVER_ID
        assert parsed.provenance.k == 4
        assert parsed.layer == 6 and parsed.position == 3

    def test_the_control_is_reproducible_from_the_seed_alone(self, frame):
        h = torch.randn(24)
        assert torch.equal(jd.size_matched_control(h, 5), jd.size_matched_control(h, 5))
        assert not torch.equal(jd.size_matched_control(h, 5), jd.size_matched_control(h, 6))
        assert jd.size_matched_control(h, 5).norm() == pytest.approx(float(h.norm()), rel=1e-5)

    def test_excess_fve_is_measured_against_the_control(self, frame):
        row = self._row(frame)
        assert row["excess_fve"] == pytest.approx(row["fve"] - row["control_fve"])
        assert row["fve"] > row["control_fve"], "a planted signal must beat a random direction"

    def test_same_seed_same_result(self, frame):
        assert self._row(frame, 9) == self._row(frame, 9)


class TestTheRouteIsServed:
    def test_decompose_and_its_poll_are_in_the_served_schema(self):
        from src.main import app

        paths = app.openapi()["paths"]
        assert "post" in paths["/api/v1/jlens/decompose"]
        assert "get" in paths["/api/v1/jlens/decompose/{task_id}"]

    def test_the_route_queues_the_task_with_the_seed_it_returns(self):
        from fastapi.testclient import TestClient

        from src.core.deps import get_db
        from src.main import app

        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: object()))

        async def fake_db():
            yield db

        queued = MagicMock(return_value=MagicMock(id="t-dec"))
        app.dependency_overrides[get_db] = fake_db
        try:
            with patch("src.api.v1.endpoints.jlens.gpu_delay", return_value=queued), patch(
                "src.workers.jlens_progress.open_row"
            ) as open_row:
                resp = TestClient(app).post(
                    "/api/v1/jlens/decompose",
                    json={"model_id": "m_1", "prompt": "hi", "layers": [2], "k": 3},
                )
        finally:
            app.dependency_overrides.pop(get_db, None)
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert queued.call_count == 1
        sent = queued.call_args.kwargs
        assert sent["control_seed"] == body["control_seed"], (
            "the seed returned to the caller must be the seed the task runs with"
        )
        assert sent["k"] == 3 and sent["layers"] == [2]
        assert open_row.call_args.args[0] == "jlens_decompose"

    def test_the_cell_cap_is_enforced(self):
        from pydantic import ValidationError

        from src.schemas.jlens import MAX_DECOMPOSE_CELLS, DecomposeRequest

        with pytest.raises(ValidationError, match="cells"):
            DecomposeRequest(
                model_id="m", prompt="x", layers=list(range(64)), positions=list(range(64)),
            )
        assert MAX_DECOMPOSE_CELLS < 64 * 64


class TestTheTaskBody:
    """`_decompose` as the worker runs it: capture → pursuit per cell → contract check."""

    def _run(self, monkeypatch, *, layers=(1,), positions=None, artifact_id=None, n_layers=3):
        from types import SimpleNamespace

        from src.workers import jlens_probe_tasks as tasks

        torch.manual_seed(5)
        W_U = torch.randn(40, 8)
        planted = 2.0 * _unit(W_U[7]) + 1.0 * _unit(W_U[30])
        residual = torch.stack([torch.randn(8), torch.randn(8), planted])  # 3 positions

        class _Service:
            def __init__(self, **kw):
                self.tokenizer = lambda text, return_tensors=None: {
                    "input_ids": torch.tensor([[1, 2, 3]])
                }
                self.capture_device = "cpu"
                self.d_model = 8
                self.W_U = W_U

            def _capture_residuals(self, input_ids, layers):
                return SimpleNamespace(by_layer={l: residual for l in layers},
                                       hook_target="layers_module[L]")

            def _decode_batch(self, ids):
                return [f"t{i}" for i in ids]

        monkeypatch.setattr("src.services.jlens_readout_service.ReadoutService", _Service)
        monkeypatch.setattr("src.workers.jlens_progress.update_row", lambda *a, **k: True)
        loaded = SimpleNamespace(model=None, tokenizer=None, unembedding=W_U, name="tiny",
                                 structure=SimpleNamespace(num_layers=n_layers))
        me = SimpleNamespace(update_state=lambda **k: None, request=SimpleNamespace(id="t-d"))
        return tasks._decompose(me, loaded=loaded, prompt="abc", layers=list(layers),
                                positions=positions, k=2, max_iterations=40,
                                control_seed=77, artifact_id=artifact_id)

    def test_the_last_position_by_default_with_its_provenance(self, monkeypatch):
        out = self._run(monkeypatch)
        assert out["mode"] == "decomposition" and out["lens_type"] == "LOGIT_LENS"
        (cell,) = out["decompositions"]
        assert cell["position"] == 2 and cell["layer"] == 1
        assert set(cell["active_token_ids"]) == {7, 30}
        assert cell["active_tokens"] == [f"t{i}" for i in cell["active_token_ids"]]
        assert cell["provenance"]["control_seed"] == 77
        DecompositionResult.model_validate(cell)

    def test_negative_positions_count_from_the_end_and_out_of_range_is_refused(self, monkeypatch):
        out = self._run(monkeypatch, positions=[0, -1])
        assert [c["position"] for c in out["decompositions"]] == [0, 2]
        with pytest.raises(ValueError, match="positions"):
            self._run(monkeypatch, positions=[5])
        with pytest.raises(ValueError, match="layers"):
            self._run(monkeypatch, layers=(9,))


class TestTheSupportNeverRepeatsAnAtom:
    """⚠ A SURVIVING MUTATION WROTE THIS TEST (2026-10-16 review). Deleting the line that
    masks already-active atoms left every test green: the planted-component fixtures are
    solved before an active atom can out-correlate a new one. On a correlated frame with
    slack in k, a gradient step is NOT an exact least-squares fit, the residual still
    correlates with the active atoms, and the unmasked argmax re-admits one — a duplicate
    in `active_tokens` reported as two components."""

    def test_correlated_frame_with_slack_in_k(self):
        g = torch.Generator().manual_seed(0)
        base = torch.randn(10, 16, generator=g)
        frame = torch.cat([base, base + 0.3 * torch.randn(10, 16, generator=g)])
        d = jd.LensDictionary(frame, IdentityTransport(), layer=0)
        torch.randn(16, generator=g)  # keep the generator stream the probe used
        h = frame[:4].T @ torch.rand(4, generator=g) * 3
        r = jd.gradient_pursuit(h, d, k=6, max_iterations=40)
        assert len(set(r.active)) == len(r.active), f"duplicate atoms: {r.active}"


class TestTheTaskRefusesARowOffContract:
    """⚠ ALSO WRITTEN BY A SURVIVOR: removing the task's `DecompositionResult` check left
    the suite green, because the solver never emits an invalid row. The check exists for
    the day it does — so a row without its control seed must FAIL the task, not ship."""

    def test_a_row_without_its_seed_fails_the_task(self, monkeypatch):
        import src.services.jlens_decomposition as real

        original = real.decompose

        def seedless(*a, **k):
            row = original(*a, **k)
            row["provenance"].pop("control_seed")
            return row

        monkeypatch.setattr(real, "decompose", seedless)
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="control_seed"):
            TestTheTaskBody()._run(monkeypatch)
