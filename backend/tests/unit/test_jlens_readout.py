"""Wire-format conformance, readout modes, and mandatory provenance.

The wire format is NOT ours to design (BR-029, PADR IDL-45) — it mirrors
Neuronpedia's lens stream so one viewer serves both and disagreement between
implementations is diagnostic rather than mysterious. Two details are
load-bearing and easy to get subtly wrong:

  * `top_tokens` are DECODED STRINGS. Emitting ids type-checks against a looser
    schema and renders as unreadable cells.
  * `layers_by_type` drives the client's layer axis. The reference panel
    hardcodes 21 layers at 0,5,...,100; the reference model has 16. Nothing may
    assume a count or spacing.

And one that is a correctness rule rather than a format rule:

  * a sparse-decomposition figure without its control seed is INVALID, not
    merely undocumented — occupancy and excess-FVE are defined as an excess
    over a random-direction control, so the number is unreproducible without it.

MUTATION CONTROLS:
  * emit ids instead of decoded strings              -> string test fails
  * hardcode a layer count/spacing in the stream     -> layer-axis test fails
  * make control_seed optional                       -> provenance test fails
  * let JACOBIAN_LENS fall back to identity          -> no-silent-fallback fails
  * drop the terminal done message                   -> terminator test fails
"""

import pytest
import torch

from src.schemas.jlens import (
    DecompositionProvenance,
    DecompositionResult,
    LensTypeSlice,
    ReadoutRequest,
)
from src.services.jlens_readout_service import (
    IdentityTransport,
    JacobianTransport,
    envelope_bound_bytes,
)

from tests.unit.test_jlens_model_agnostic import make_service


class TestWireFormat:
    def test_stream_is_meta_then_tokens_then_done(self):
        svc = make_service()
        msgs = list(svc.stream("abcd", [IdentityTransport()], top_n=3))

        assert msgs[0].kind == "meta"
        assert msgs[-1].kind == "done", "stream has no terminator"
        assert all(m.kind == "token" for m in msgs[1:-1])

    def test_top_tokens_are_decoded_strings_not_ids(self):
        svc = make_service()
        msgs = list(svc.stream("abc", [IdentityTransport()], top_n=3))
        token_msgs = [m for m in msgs if m.kind == "token"]

        for m in token_msgs:
            for sl in m.results:
                for layer_row in sl.top_tokens:
                    for tok in layer_row:
                        assert isinstance(tok, str), (
                            f"{tok!r} is {type(tok).__name__}; ids render as "
                            "unreadable cells in the client"
                        )

    def test_schema_rejects_ids_with_an_explanation(self):
        """The validator must actually fire, not be shadowed by type coercion."""
        with pytest.raises(ValueError, match="DECODED STRING"):
            LensTypeSlice(
                type="LOGIT_LENS", top_tokens=[[1, 2]], top_probs=[[0.5, 0.5]]
            )

    def test_layer_axis_comes_from_the_model_not_a_constant(self):
        """16 layers, 4 layers, 3 layers — the stream must follow the model."""
        for n_layers in (3, 4, 7):
            svc = make_service(n_layers=n_layers)
            msgs = list(svc.stream("ab", [IdentityTransport()], top_n=2))
            meta = msgs[0]
            assert meta.layers_by_type["LOGIT_LENS"] == list(range(n_layers))
            for m in msgs[1:-1]:
                for sl in m.results:
                    assert len(sl.top_tokens) == n_layers

    def test_selected_layers_are_honoured(self):
        svc = make_service(n_layers=6)
        msgs = list(svc.stream("ab", [IdentityTransport()], layers=[1, 4], top_n=2))
        assert msgs[0].layers_by_type["LOGIT_LENS"] == [1, 4]
        assert all(len(sl.top_tokens) == 2 for m in msgs[1:-1] for sl in m.results)

    def test_out_of_range_layer_is_refused(self):
        svc = make_service(n_layers=4)
        with pytest.raises(ValueError, match="outside range"):
            list(svc.stream("ab", [IdentityTransport()], layers=[9]))

    def test_probs_are_parallel_to_tokens(self):
        svc = make_service()
        msgs = list(svc.stream("abc", [IdentityTransport()], top_n=4))
        for m in msgs[1:-1]:
            for sl in m.results:
                for toks, probs in zip(sl.top_tokens, sl.top_probs):
                    assert len(toks) == len(probs)


class TestNoSilentLensFallback:
    def test_jacobian_without_artifact_is_refused_at_the_request(self):
        with pytest.raises(ValueError, match="without artifact_id"):
            ReadoutRequest(model_id="m", prompt="hi", types=["JACOBIAN_LENS"])

    def test_logit_needs_no_artifact(self):
        r = ReadoutRequest(model_id="m", prompt="hi")
        assert r.artifact_id is None
        assert IdentityTransport().requires_artifact() is False

    def test_missing_layer_raises_rather_than_serving_logit_data(self):
        """Falling back to identity would serve logit data under a Jacobian
        label — prohibited by rung discipline (BR-019)."""
        t = JacobianTransport({0: torch.eye(8)})
        t.apply(torch.randn(8), 0)  # present layer is fine
        with pytest.raises(KeyError, match="Refusing to fall back"):
            t.apply(torch.randn(8), 3)

    def test_non_square_jacobian_is_refused(self):
        with pytest.raises(ValueError, match="expected a square"):
            JacobianTransport({0: torch.randn(8, 5)})


class TestDecompositionProvenance:
    def test_control_seed_is_mandatory(self):
        """Occupancy is defined as an excess over a random control; without the
        seed the figure cannot be reproduced or believed."""
        with pytest.raises(ValueError):
            DecompositionProvenance(
                k=25,
                solver="gradient_pursuit",
                iterations=40,
                convergence_criterion="delta<1e-3",
                control_construction="gaussian",
            )

    def test_complete_provenance_validates(self):
        p = DecompositionProvenance(
            k=25,
            solver="gradient_pursuit",
            iterations=40,
            convergence_criterion="delta<1e-3",
            control_seed=1234,
            control_construction="gaussian",
        )
        assert p.control_seed == 1234

    def test_active_set_cannot_exceed_k(self):
        prov = DecompositionProvenance(
            k=2,
            solver="gp",
            iterations=1,
            convergence_criterion="c",
            control_seed=1,
            control_construction="g",
        )
        with pytest.raises(ValueError, match="exceeds sparsity"):
            DecompositionResult(
                layer=0,
                position=0,
                active_tokens=["a", "b", "c"],
                coefficients=[1.0, 0.5, 0.2],
                residual_norm=0.1,
                provenance=prov,
            )


class TestEnvelopeIsModelDerived:
    def test_bound_scales_with_the_model(self):
        """A constant bound passes on one model and misses a real
        materialisation on another."""
        lfm2 = envelope_bound_bytes(d_model=2048, n_layers=16)
        gemma = envelope_bound_bytes(d_model=2304, n_layers=26)
        assert lfm2 != gemma
        assert lfm2 == int(2048 * 2048 * 2 * 16 * 1.5)

    def test_materialised_dictionary_blows_the_bound(self):
        """The BR-006 guard: catching a materialised W_U J."""
        d_model, n_layers, n_vocab = 2048, 16, 65536
        bound = envelope_bound_bytes(d_model, n_layers)
        materialised = n_vocab * d_model * 2 * n_layers
        assert materialised > bound * 10

    def test_bound_moves_with_layer_count(self):
        assert envelope_bound_bytes(2048, 16) < envelope_bound_bytes(2048, 32)


class TestProbe:
    def test_probe_scores_named_directions_without_ranking(self):
        svc = make_service()
        ids = torch.tensor([[1, 2, 3]])
        residuals = svc._capture_residuals(ids, [0])
        h = residuals.by_layer[0][0]

        scores = svc.probe(h, 0, ["a", "b"], IdentityTransport())
        assert set(scores) == {"a", "b"}
        assert all(isinstance(v, float) for v in scores.values())


class TestNoNextTokenAgreementMetric:
    def test_agreement_is_not_computed_anywhere(self):
        """BR-004: the lens is deliberately worse than the logit lens on
        next-token agreement. A check that rewards it is a defect."""
        from pathlib import Path

        from src.services import jlens_readout_service as mod

        src = Path(mod.__file__).read_text().lower()
        for banned in ("next_token_agreement", "top_k_agreement", "agreement_score"):
            assert banned not in src, f"{banned} present; BR-004 forbids it"


class TestReviewRound1Fixes:
    """Findings from feature 022 review round 1, each pinned.

    MUTATION CONTROLS:
      * revert to L2 normalisation            -> normalisation test fails
      * remove prompt/tokens bounds           -> input-bound tests fail
      * cast J inside apply()                 -> per-call-cast test fails
      * remove the decode cache               -> decode-cache test fails
    """

    def test_normalisation_uses_the_models_own_final_norm(self):
        """F1. A plain L2 normalisation is not what a transformer does.

        RMSNorm divides by sqrt(mean(x^2)) AND applies a learned per-dimension
        weight. Substituting L2 drops the learned weighting, which shifts token
        rankings wherever that weight is non-uniform — while still producing a
        plausible-looking readout.
        """
        import inspect

        from src.services.jlens_readout_service import ReadoutService

        src = inspect.getsource(ReadoutService._normalize)
        assert "_final_norm" in src, "does not consult the model's own norm"

        rank_src = inspect.getsource(ReadoutService._rank_at)
        assert "self._normalize(" in rank_src
        assert "transported.norm()" not in rank_src, (
            "still L2-normalising inline; the learned norm weight is dropped"
        )

    def test_final_norm_is_resolved_by_name_list_not_architecture(self):
        from src.services.jlens_readout_service import _FINAL_NORM_NAMES

        assert "ln_f" in _FINAL_NORM_NAMES and "norm" in _FINAL_NORM_NAMES, (
            "final-norm lookup must cover multiple families as DATA, so a new "
            "family is a list entry rather than an architecture branch"
        )

    def test_models_own_norm_is_actually_applied(self):
        """Behavioural, not just structural: a learned norm weight must change
        the result relative to the fallback."""
        import torch

        from tests.unit.test_jlens_model_agnostic import make_service

        svc = make_service()
        x = torch.randn(8)

        svc._final_norm = None
        fallback = svc._normalize(x)

        ln = torch.nn.LayerNorm(8)
        with torch.no_grad():
            ln.weight.fill_(3.0)   # non-uniform vs the fallback's implicit 1.0
        svc._final_norm = ln
        applied = svc._normalize(x)

        assert not torch.allclose(fallback, applied, atol=1e-4), (
            "the model's norm made no difference — _normalize is ignoring it"
        )

    def test_prompt_length_is_bounded(self):
        """F2. Readout cost is O(positions x layers x top_n) and each position
        holds a d_model residual — an unbounded prompt amplifies one request."""
        from src.schemas.jlens import ReadoutRequest

        with pytest.raises(ValueError):
            ReadoutRequest(model_id="m", prompt="x" * 50_000)

    def test_probe_token_list_is_bounded(self):
        """F3. Same amplification through a different field."""
        from src.schemas.jlens import ProbeRequest

        with pytest.raises(ValueError):
            ProbeRequest(model_id="m", prompt="hi", tokens=["a"] * 5_000)

    def test_layer_list_is_bounded(self):
        from src.schemas.jlens import ReadoutRequest

        with pytest.raises(ValueError):
            ReadoutRequest(model_id="m", prompt="hi", layers=list(range(5_000)))

    def test_jacobian_is_cast_once_not_per_call(self):
        """F4. Casting inside apply() copies a d_model^2 matrix on every
        (layer, position) — 8 MB per call at d_model 2048, thousands of times
        per readout."""
        import inspect

        from src.services.jlens_readout_service import JacobianTransport

        apply_src = inspect.getsource(JacobianTransport.apply)
        assert "j.to(" not in apply_src, (
            "J is cast inside apply(); cast once at construction instead"
        )

        j16 = {0: torch.eye(8, dtype=torch.float16)}
        t = JacobianTransport(j16)
        assert t._j[0].dtype == torch.float32, "not cast at construction"

    def test_decode_is_cached(self):
        """F5. Without a cache the tokenizer is called once per
        (layer, position, k) and dominates runtime."""
        from tests.unit.test_jlens_model_agnostic import make_service

        svc = make_service()
        calls = {"n": 0}
        inner = svc.tokenizer.decode

        def counting(ids):
            calls["n"] += 1
            return inner(ids)

        svc.tokenizer.decode = counting
        list(svc.stream("aaa", [IdentityTransport()], top_n=4))
        cached_calls = calls["n"]

        assert cached_calls <= svc.n_vocab, (
            f"{cached_calls} decode calls for a {svc.n_vocab}-token vocabulary "
            "— the cache is not deduplicating"
        )


class TestReviewRound2Fixes:
    """Round 2 finding: per-field bounds do not bound the PRODUCT.

    Round 1 bounded prompt (8000 chars), layers (512) and top_n (100)
    individually and treated resource exhaustion as handled. Each limit is
    reasonable; together they permit roughly 102 MILLION ranked readouts
    holding about 8.4 GB of residuals from one request that passes every field
    validator.

    BR-028 requires an operation that cannot fit the envelope to fail with a
    STATED REASON rather than degrade silently.

    MUTATION CONTROLS:
      * remove the budget check from stream()    -> enforcement test fails
      * raise the cell ceiling to unlimited      -> cell test fails
      * check the budget AFTER capture           -> ordering test fails
    """

    def test_cell_count_ceiling_is_enforced(self):
        """Absolute inputs, not inputs derived from the constant.

        Deriving the input from MAX_READOUT_CELLS makes the test
        self-referential: raising the ceiling to 10^12 also raises the input,
        so it still exceeds and the test passes against an effectively
        unlimited budget. Verified — that mutation survived until this was
        rewritten.
        """
        from src.services.jlens_readout_service import (
            MAX_READOUT_CELLS,
            ReadoutTooLarge,
            check_readout_budget,
        )

        assert MAX_READOUT_CELLS <= 1_000_000, (
            f"cell ceiling is {MAX_READOUT_CELLS:,}, high enough to permit the "
            "multi-GB readout this limit exists to prevent"
        )

        with pytest.raises(ReadoutTooLarge, match="cells"):
            check_readout_budget(n_positions=2_000_000, n_layers=4, d_model=8)

    def test_residual_memory_ceiling_is_enforced(self):
        from src.services.jlens_readout_service import (
            ReadoutTooLarge,
            check_readout_budget,
        )

        from src.services.jlens_readout_service import MAX_RESIDUAL_BYTES

        assert MAX_RESIDUAL_BYTES <= 2 * 1024**3, (
            f"residual ceiling is {MAX_RESIDUAL_BYTES/1e9:.1f} GB, high enough "
            "to permit the exhaustion this limit exists to prevent"
        )

        # Few cells, enormous width — passes the cell check, must fail on bytes.
        with pytest.raises(ReadoutTooLarge, match="GB of residuals"):
            check_readout_budget(
                n_positions=100, n_layers=10, d_model=1_000_000
            )

    def test_a_reasonable_request_is_allowed(self):
        from src.services.jlens_readout_service import check_readout_budget

        # The reference model on a normal prompt.
        check_readout_budget(n_positions=200, n_layers=16, d_model=2048)

    def test_the_error_states_the_numbers_and_a_remedy(self):
        """A limit without a remedy is a dead end for the caller."""
        from src.services.jlens_readout_service import (
            ReadoutTooLarge,
            check_readout_budget,
        )

        try:
            check_readout_budget(500_000, 4, 2048)
        except ReadoutTooLarge as e:
            msg = str(e)
            assert "limit" in msg
            assert "Narrow the layer selection" in msg, (
                "the error does not tell the caller what to change"
            )
        else:
            pytest.fail("no error raised")

    def test_stream_enforces_the_budget_before_capturing(self):
        """Ordering matters: checking after capture means the OOM already
        happened."""
        import inspect

        from src.services.jlens_readout_service import ReadoutService

        src = inspect.getsource(ReadoutService.stream)

        # Compare EXECUTABLE lines only. A comment above the call mentions
        # check_readout_budget by name, so a raw str.index() finds the comment
        # and reports the right order no matter where the call actually sits —
        # the same trap as searching a docstring for residual_norm_module.
        code = "\n".join(
            l for l in src.splitlines() if not l.strip().startswith("#")
        )
        assert "check_readout_budget(" in code, (
            "stream() does not enforce the budget; per-field bounds alone "
            "permit ~8.4 GB of residuals from one valid request"
        )
        assert code.index("check_readout_budget(") < code.index(
            "self._capture_residuals("
        ), "budget is checked AFTER capture — the memory is already allocated"

    def test_oversized_request_raises_rather_than_running(self):
        from tests.unit.test_jlens_model_agnostic import make_service
        from src.services.jlens_readout_service import (
            ReadoutTooLarge,
            jlens_budget_override,
        )

        svc = make_service(n_layers=4)
        with jlens_budget_override(max_cells=2):
            with pytest.raises(ReadoutTooLarge):
                list(svc.stream("abcdef", [IdentityTransport()], top_n=2))


class TestReviewRound3Fixes:
    """Round 3 findings, both silent-failure shapes.

    MUTATION CONTROLS:
      * accept any callable as the final norm  -> nn.Module test fails
      * make ReadoutResponse inherit meta      -> envelope test fails
    """

    def test_final_norm_must_be_a_module_not_merely_callable(self):
        """`callable()` is far too loose for model-agnostic resolution.

        A BOUND METHOD named `norm` passes callable(), and so does any other
        callable attribute sharing the name. Normalising with the wrong object
        yields a readout that looks fine and ranks tokens wrongly — the same
        silent class as hooking the wrong module.
        """
        import inspect

        from src.services.jlens_readout_service import ReadoutService

        src = inspect.getsource(ReadoutService._resolve_final_norm)
        assert "isinstance(candidate, torch.nn.Module)" in src, (
            "final-norm resolution accepts any callable; a bound method named "
            "'norm' would be selected and silently used"
        )

    def test_a_callable_that_is_not_a_module_is_rejected(self):
        """Behavioural: a decoy method named `norm` must not be picked up."""
        from types import SimpleNamespace

        from src.services.jlens_readout_service import ReadoutService

        class Decoy:
            def norm(self, x):        # callable, not an nn.Module
                return x * 0.0

        resolved = ReadoutService._resolve_final_norm(Decoy())
        assert resolved is None, (
            f"picked up {resolved!r} — a bound method, not a normalisation "
            "module"
        )

    def test_a_real_module_is_still_found(self):
        import torch

        from src.services.jlens_readout_service import ReadoutService

        class WithNorm:
            def __init__(self):
                self.norm = torch.nn.LayerNorm(8)

        resolved = ReadoutService._resolve_final_norm(WithNorm())
        assert isinstance(resolved, torch.nn.Module)

    def test_readout_response_does_not_impersonate_a_meta_message(self):
        """The envelope must CONTAIN a meta message, not inherit one.

        Inheriting carried `kind: "meta"` onto the envelope, so the response
        announced itself as a meta message while also carrying a tokens array.
        A client dispatching on `kind` mis-handles that — in the one format
        this feature exists to conform to.
        """
        from src.api.v1.endpoints.jlens import ReadoutResponse
        from src.schemas.jlens import LensMetaMessage

        assert not issubclass(ReadoutResponse, LensMetaMessage), (
            "ReadoutResponse inherits LensMetaMessage and therefore its "
            "kind='meta' discriminator"
        )
        assert "meta" in ReadoutResponse.model_fields
        assert "kind" not in ReadoutResponse.model_fields


# ── mixed precision (found on the cluster, not in any fixture) ──────────────


def test_a_readout_survives_a_MIXED_DTYPE_checkpoint():
    """gemma-2-2b-it died here with "expected BFloat16 but found Half".

    A real checkpoint is not one dtype: the final norm can be bfloat16 while
    the residual and the unembedding are fp16, and torch RAISES rather than
    promoting. Every fixture in this file used a single dtype, so nothing
    caught it until the model ran on the cluster.
    """
    import torch

    from src.services.jlens_readout_service import IdentityTransport, ReadoutService
    from src.schemas.jlens import LensTokenMessage

    d_model, n_vocab = 8, 32

    class Block(torch.nn.Module):
        def forward(self, hidden, **_):
            return (hidden * 1.01,)

    class BF16Norm(torch.nn.Module):
        """A final norm stored in bfloat16, as gemma's is.

        Uses a MATMUL rather than an elementwise multiply, because that is what
        actually raises. Torch PROMOTES fp16 * bf16 silently, so a fixture built
        on `*` reproduces the dtypes without reproducing the failure — which is
        how the first version of this test passed against the broken code.
        """

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.eye(d_model, dtype=torch.bfloat16)
            )

        def forward(self, x):
            return x @ self.weight

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block() for _ in range(2)])
            # Residuals come out fp16 — a DIFFERENT family from the norm.
            self.embed = torch.nn.Embedding(16, d_model, dtype=torch.float16)
            self.norm = BF16Norm()
            self.config = None

        def forward(self, input_ids=None, **_):
            hidden = self.embed(input_ids)
            for b in self.blocks:
                hidden = b(hidden)[0]
            return hidden

    class Structure:
        def __init__(self, m):
            self.layers_module = m.blocks
            self.num_layers = 2
            self.attention_module = None
            self.residual_norm_module = None

    class Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids] if isinstance(ids, list) else f"t{ids}"

        def decode(self, ids, **_):
            return "t"

    model = Model()
    service = ReadoutService(
        model=model,
        tokenizer=Tok(),
        structure=Structure(model),
        # fp16 unembedding against a bfloat16 norm: the exact cluster shape.
        unembedding=torch.randn(n_vocab, d_model, dtype=torch.float16),
        model_name="mixed",
    )

    messages = list(service.stream("abc", [IdentityTransport()], top_n=3))
    tokens = [m for m in messages if isinstance(m, LensTokenMessage)]
    assert tokens, "the readout produced nothing on a mixed-dtype model"
    for t in tokens:
        for row in t.results[0].top_tokens:
            assert len(row) == 3

    # SECOND FAMILY MISMATCH, at the matvec rather than the norm. After the
    # norm is cast correctly the residual and a fp16 unembedding agree by
    # accident, so this is the only shape that exercises the W_U cast: a
    # bfloat16 unembedding against an fp16 residual.
    bf16_service = ReadoutService(
        model=model,
        tokenizer=Tok(),
        structure=Structure(model),
        unembedding=torch.randn(n_vocab, d_model, dtype=torch.bfloat16),
        model_name="mixed-unembedding",
    )
    bf16_tokens = [
        m
        for m in bf16_service.stream("abc", [IdentityTransport()], top_n=3)
        if isinstance(m, LensTokenMessage)
    ]
    assert bf16_tokens, "the readout died on a bfloat16 unembedding"


# ---------------------------------------------------------------------------
# Device agreement between the model and its inputs
#
# Reported from the cluster UI as "Expected all tensors to be on the same
# device, but got index is on cpu, different from other tensors on cuda:0".
# Raised inside the embedding lookup, before any readout maths ran.
#
# CAUSE: the worker's model cache was keyed by MODEL ID ALONE. A fit loads the
# model with capture_device="cuda"; the next readout asked for "cpu", hit the
# same key, and received the CUDA-resident model while still placing input_ids
# on CPU. Two independent claims about one model's device, silently disagreeing.
#
# No existing test caught it because every fixture builds its model on CPU and
# asks for CPU — the two agreed by construction, so the mismatch could not arise.
#
# MUTATION CONTROLS (each must turn this section red):
#   * capture_device defaults back to "cpu"        -> "follows the model" fails
#   * cache key drops the "@device" suffix         -> "device is part of identity" fails
# ---------------------------------------------------------------------------


class _DeviceStampedParam:
    """Minimal stand-in for a parameter that reports a device."""

    def __init__(self, device: str):
        self.device = device


class _ModelOnDevice:
    def __init__(self, device: str):
        self._device = device

    def parameters(self):
        yield _DeviceStampedParam(self._device)


def test_the_readout_follows_the_model_rather_than_assuming_cpu():
    """capture_device must be READ OFF the model, not defaulted alongside it."""
    from src.services.jlens_readout_service import _model_device

    assert _model_device(_ModelOnDevice("cuda:0")) == "cuda:0"
    assert _model_device(_ModelOnDevice("cpu")) == "cpu"
    # A model with no parameters at all yields None, and the caller falls back —
    # it must not raise, or a malformed model turns a readout into a 500.
    assert _model_device(object()) is None


def test_the_device_is_part_of_a_cached_models_identity():
    """A caller asking for CPU must not be handed a CUDA model.

    That is not a preference being overridden; it is a different object than
    the one requested, and it crashed the readout after a fit.
    """
    from src.services import jlens_model_registry as reg

    keys = []

    class _Rec:
        id = "m_x"
        repo_id = "org/model"
        file_path = None

    original = reg._CACHE.get_or_load
    reg._CACHE.get_or_load = lambda key, loader: (keys.append(key), None)[1]
    try:
        for device in ("cpu", "cuda"):
            try:
                reg.load_for_readout(_Rec(), capture_device=device)
            except Exception:
                pass
    finally:
        reg._CACHE.get_or_load = original

    assert len(keys) == 2, "load_for_readout did not reach the cache"
    assert keys[0] != keys[1], (
        f"cpu and cuda produced the SAME cache key {keys[0]!r}; a caller asking "
        "for one device would be handed the other"
    )
    assert keys[0].endswith("@cpu") and keys[1].endswith("@cuda")


# ---------------------------------------------------------------------------
# The final norm belongs to the MODEL, so it lives where the model lives
#
# Second face of the same fault, found by the fit's own SEMANTIC check on the
# cluster: "found at least two devices, cuda:0 and cpu". The readout runs on
# READOUT_DEVICE by design while the model may be resident on an accelerator,
# and `_normalize` calls a live module belonging to that model. Fixing the
# input_ids placement moved the collision here rather than removing it.
#
# MUTATION CONTROLS:
#   * drop the .to(device=norm_device)  -> "borrows the norm" fails
#   * return without .to(x.device)      -> "comes straight back" fails
# ---------------------------------------------------------------------------


class _NormClaimingDevice(torch.nn.Module):
    """A norm whose parameters report a device, without needing that device.

    Only `parameters()` is faked — the module still computes normally — so the
    test can assert WHERE `_normalize` decided to send the tensor without
    requiring the hardware. The genuine cross-device execution is covered by
    the CUDA-gated test below.
    """

    def __init__(self, d_model: int, claimed: str):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self._claimed = torch.device(claimed)

    def parameters(self, recurse: bool = True):
        param = torch.nn.Parameter(self.weight.detach())
        param._claimed_device = self._claimed
        yield _DeviceLyingParam(self._claimed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


class _DeviceLyingParam:
    """Reports a device. `_module_device` reads only `.device`."""

    def __init__(self, device):
        self.device = device


def _service_with_norm(norm):
    from src.services.jlens_readout_service import ReadoutService

    svc = ReadoutService.__new__(ReadoutService)
    svc._final_norm = norm
    return svc


def test_normalize_sends_the_tensor_to_the_norms_device_and_brings_it_back():
    """Portable form: assert the DEVICE DECISION, not the hardware.

    Records every `.to(...)` `_normalize` performs. Two things must be true and
    each is a separate mutation: it must move the input to where the norm
    actually is, and it must return on the READOUT's device — a result left on
    the model's device just relocates the collision to the unembedding matvec.
    """
    from src.services.jlens_readout_service import ReadoutService

    d = 8
    norm = _NormClaimingDevice(d, "cuda:0")
    svc = _service_with_norm(norm)

    moves = []
    original_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        if "device" in kwargs:
            moves.append(str(kwargs["device"]))
        elif args and isinstance(args[0], (str, torch.device)):
            moves.append(str(args[0]))
        # Drop the device so the call succeeds on a machine without one; the
        # DECISION is what is under test.
        kwargs.pop("device", None)
        if args and isinstance(args[0], (str, torch.device)):
            args = args[1:]
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = recording_to
    try:
        out = ReadoutService._normalize(svc, torch.randn(d))
    finally:
        torch.Tensor.to = original_to

    assert "cuda:0" in moves, (
        f"_normalize never moved the tensor to the norm's device; moves={moves}. "
        "Calling a module on another device raises 'found at least two devices'"
    )
    assert "cpu" in moves, (
        f"_normalize never brought the result back to the readout's device; "
        f"moves={moves}. The collision moves to the unembedding matvec instead"
    )
    assert out.dtype == torch.float32


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a real second device"
)
def test_normalize_survives_a_genuinely_gpu_resident_norm():
    """The unfaked version, where the hardware exists."""
    from src.services.jlens_readout_service import ReadoutService

    d = 8
    norm = torch.nn.LayerNorm(d).to("cuda:0")
    svc = _service_with_norm(norm)

    x = torch.randn(d)  # cpu
    out = ReadoutService._normalize(svc, x)
    assert out.device == x.device


# ---------------------------------------------------------------------------
# Artifacts are portable documents
#
# `torch.save` records each tensor's device. A fit runs on the GPU, so an
# artifact written straight from one is tagged cuda:0 and raises "Attempting to
# deserialize object on a CUDA device" for any consumer without one. Our own
# loader passes map_location="cpu" and would never have noticed — the failure
# belongs entirely to somebody else's machine.
#
# MUTATION CONTROL: drop the .to("cpu") in write_staged -> this fails.
# ---------------------------------------------------------------------------


def test_a_written_artifact_is_moved_to_cpu_before_it_is_saved(tmp_path):
    """Assert the MOVE, not the resulting tag.

    Checking the saved tensor's device is vacuous on a CPU-only machine — the
    tensors are already there, so deleting the `.to("cpu")` changes nothing
    locally and the mutation survives. Verified: that exact test passed against
    the unfixed code. What must hold everywhere is that write_staged ASKS for
    cpu, which is observable without a GPU.
    """
    from src.services.jlens_artifact_service import JLensArtifactService

    service = JLensArtifactService(tmp_path)
    requested = []
    original_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        target = kwargs.get("device") or (args[0] if args else None)
        if isinstance(target, (str, torch.device)):
            requested.append(str(target))
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = recording_to
    try:
        ref = service.write_staged(
            "org/model",
            {24: torch.randn(4, 4), 25: torch.randn(4, 4)},
            "corpus: test\n",
        )
    finally:
        torch.Tensor.to = original_to

    assert requested.count("cpu") >= 2, (
        f"write_staged moved {requested.count('cpu')} of 2 tensors to cpu "
        f"(calls: {requested}). torch.save records each tensor's device, so a "
        "GPU-fitted artifact is tagged cuda:0 and raises 'Attempting to "
        "deserialize object on a CUDA device' for any consumer without one"
    )

    # And the file must still load the way an EXTERNAL consumer would: with no
    # map_location, because they have no reason to pass one.
    payload = torch.load(ref.lens_path, weights_only=True)
    # AND IN THE CONFORMANT WRAPPER, which is what makes it loadable by anyone
    # other than us: the reference consumer reads `payload["J"]` and raises
    # without it. This test is about DEVICE placement, so it goes through the
    # normaliser for the layer set rather than pinning the container shape.
    from src.services.jlens_artifact_service import normalise_payload

    assert "J" in payload, sorted(payload)
    assert set(normalise_payload(payload)) == {24, 25}


# ---------------------------------------------------------------------------
# EACH LENS TYPE CARRIES ITS OWN LAYER LIST (F1, F9)
#
# Reproduced on the cluster before this was written:
#   POST /jlens/readout  m_88d55564  types=[JACOBIAN_LENS, LOGIT_LENS]
#   FAILURE: 'No Jacobian for layer 0. Refusing to fall back to identity...'
#
# The panel sends no explicit layers, the server defaulted to every layer, and
# the 9-of-16-layer LFM2 artifact refused the first one it lacked. The refusal
# was correct; making the request was not. `layers_by_type` is per lens type in
# the wire format precisely so the two can differ, and both were given the same
# list — which ALSO made the client's Diff compare row i of one axis against
# row i of the other.
#
# MUTATION CONTROLS (each must turn this section red):
#   * layers_by_type back to {type: selected} for all  -> "own list" fails
#   * covers() returns all layers on JacobianTransport -> "partial" fails
#   * capture the full `selected` instead of `needed`  -> "captures only" fails
# ---------------------------------------------------------------------------


class _StubTok:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, ids):
        return f"t{ids[0]}"


def _service_over(n_layers: int, d_model: int = 4):
    """A ReadoutService whose capture is stubbed — the axis logic is the subject."""
    from src.services.jlens_readout_service import ReadoutService

    svc = ReadoutService.__new__(ReadoutService)
    svc.tokenizer = _StubTok()
    svc.model_name = "test/model"
    svc.model = None
    svc.capture_device = "cpu"
    svc.structure = type("S", (), {"num_layers": n_layers})()
    svc.W_U = torch.randn(7, d_model)
    svc.n_vocab, svc.d_model = svc.W_U.shape
    svc._final_norm = None
    svc._decode_cache = {}
    svc._capture_residuals = lambda ids, layers: type(
        "C", (), {"by_layer": {l: torch.randn(3, d_model) for l in layers},
                  "hook_target": "layers_module[L]"}
    )()
    return svc


def test_a_partial_jacobian_artifact_reports_only_the_layers_it_has():
    from src.schemas.jlens import LensMetaMessage
    from src.services.jlens_readout_service import (
        IdentityTransport,
        JacobianTransport,
    )

    fitted = [1, 2, 3, 10, 11]
    jac = JacobianTransport({l: torch.eye(4) for l in fitted})
    svc = _service_over(16)

    messages = list(svc.stream("hi", [jac, IdentityTransport()], layers=None, top_n=2))
    meta = next(m for m in messages if isinstance(m, LensMetaMessage))

    # "own list": the Jacobian reports its fitted layers, the logit lens all 16.
    assert meta.layers_by_type["JACOBIAN_LENS"] == fitted, (
        f"Jacobian axis is {meta.layers_by_type['JACOBIAN_LENS']}; a partial "
        "artifact must report the layers it actually holds"
    )
    assert meta.layers_by_type["LOGIT_LENS"] == list(range(16))

    # "partial": and the stream completes rather than raising at layer 0.
    from src.schemas.jlens import LensTokenMessage

    tokens = [m for m in messages if isinstance(m, LensTokenMessage)]
    assert tokens, "the stream produced no tokens"
    jac_slice = next(s for s in tokens[0].results if s.type == "JACOBIAN_LENS")
    assert len(jac_slice.top_tokens) == len(fitted)
    logit_slice = next(s for s in tokens[0].results if s.type == "LOGIT_LENS")
    assert len(logit_slice.top_tokens) == 16


def test_the_readout_captures_only_layers_some_lens_will_read():
    """Capture is the expensive half; capturing 16 to read 5 is wasted work."""
    from src.services.jlens_readout_service import JacobianTransport

    fitted = [2, 7]
    svc = _service_over(16)
    seen = {}

    def spy(ids, layers):
        seen["layers"] = list(layers)
        return type(
            "C", (), {"by_layer": {l: torch.randn(3, 4) for l in layers},
                      "hook_target": "layers_module[L]"}
        )()

    svc._capture_residuals = spy
    list(svc.stream("hi", [JacobianTransport({l: torch.eye(4) for l in fitted})],
                    layers=None, top_n=2))

    assert seen["layers"] == fitted, (
        f"captured {seen['layers']} to read {fitted}"
    )


def test_a_lens_covering_nothing_is_refused_rather_than_emitted_empty():
    """An empty slice renders as a lens that found nothing."""
    import pytest

    from src.services.jlens_readout_service import JacobianTransport

    svc = _service_over(16)
    jac = JacobianTransport({20: torch.eye(4)})  # outside the requested range

    with pytest.raises(ValueError, match="covers any of layers"):
        list(svc.stream("hi", [jac], layers=[0, 1, 2], top_n=2))
