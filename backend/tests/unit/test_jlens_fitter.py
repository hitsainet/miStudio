"""
Fitter correctness against a KNOWN Jacobian.

A fitted lens cannot be checked by looking at it. Every failure mode in this
area produces a well-shaped tensor of plausible magnitude: the wrong hook point,
an unfrozen norm, a transposed accumulation, an unweighted shard merge. So the
fixture here is a stack whose exact Jacobian is known analytically — a
composition of linear blocks, where `J = W_n ... W_{l+1}` — and the assertion is
equality with that product, not a smoke test.

MUTATION CONTROLS (each must turn this file red):
  * hook the norm module instead of layers_module[L]  -> "hook target" fails
  * transpose the assembled Jacobian                  -> "known Jacobian" fails
  * drop weighting from merge_shards                  -> "shard merge" fails
  * accept a corpus below the floor                   -> "corpus floor" fails
  * converge on a readout proxy instead of J          -> "convergence signal" fails
"""

from __future__ import annotations

import pytest
import torch

from src.ml.jlens_fitter import (
    MIN_PROMPTS,
    JacobianFitter,
    linearisation_residual,
    jacobian_batched,
    jacobian_by_jvp,
    merge_shards,
    relative_change,
)


class ScaleNorm(torch.nn.Module):
    """A normalisation stand-in: class name carries "norm", applies a scale.

    DELIBERATELY A DISTINCT, NON-UNIT SCALE. A block built as
    `Linear . Norm` is the shape that separates the two hook points: capturing
    at the decoder layer's output includes the NEXT block's norm in the
    downstream map, and capturing at that norm excludes it. With a unit scale
    the two coincide and a negative control on the hook target proves nothing —
    the "fixtures agree by construction" trap.
    """

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.clone())

    def forward(self, hidden):
        return hidden * self.scale


class NormedBlock(torch.nn.Module):
    """Decoder-layer stand-in: normalise, then a linear map. Returns a tuple."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.input_norm = ScaleNorm(scale)
        self.weight = torch.nn.Parameter(weight.clone())

    def forward(self, hidden):
        return (self.input_norm(hidden) @ self.weight.T,)


class TinyStack(torch.nn.Module):
    def __init__(self, weights, scales):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [NormedBlock(w, s) for w, s in zip(weights, scales)]
        )
        self.embed = torch.nn.Embedding(16, weights[0].shape[1])

    def forward(self, input_ids=None, **_):
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)[0]
        return hidden


class Structure:
    """Minimal TransformerStructure stand-in."""

    def __init__(self, stack: TinyStack):
        self.layers_module = stack.blocks
        self.num_layers = len(stack.blocks)
        self.attention_module = None
        # Present, plausible, and the WRONG thing to hook — the trap this module
        # exists to avoid, reproduced faithfully.
        self.residual_norm_module = stack.blocks[0].input_norm


class Tokenizer:
    def __call__(self, text, return_tensors=None):
        ids = [(ord(c) % 15) + 1 for c in text[:6]] or [1]
        return {"input_ids": torch.tensor([ids])}


D = 6


def make_stack(seed: int = 0):
    torch.manual_seed(seed)
    weights = [torch.randn(D, D) * 0.3 for _ in range(4)]
    # Scales well away from 1 so the two hook points cannot coincide.
    scales = [torch.rand(D) * 2.0 + 0.5 for _ in range(4)]
    return TinyStack(weights, scales), weights, scales


#: Index of the block the Jacobian is taken TO, for a 4-block stack under the
#: default `penultimate` target. Derived, not hardcoded to 2, so the fixture
#: cannot silently agree with a changed default.
def target_index(n_blocks: int, target: str = "penultimate") -> int:
    return n_blocks - 1 if target == "final" else max(0, n_blocks - 2)


def analytic_jacobian(weights, scales, layer: int, target: str = "penultimate"):
    """d h_target / d h_layer = product of (W_i . diag(s_i)) for layer < i <= target.

    The stack is position-independent, so no cross-position term exists: the
    sum over target positions t' collapses to the t' = t term, and the mean over
    source positions t is that same product. The new expectation therefore has
    the SAME form as the old one — only the upper limit moved, from the final
    block to the target block.
    """
    stop = target_index(len(weights), target)
    j = torch.eye(D)
    for w, s in zip(weights[layer + 1 : stop + 1], scales[layer + 1 : stop + 1]):
        j = (w @ torch.diag(s)) @ j
    return j


def batchable(w, bias=None):
    """A map accepting [d] or [n, d], like the real sub-network."""

    def fn(h):
        out = h @ w.T if h.dim() > 1 else w @ h
        return out if bias is None else out + bias

    return fn


def test_batched_extraction_matches_a_known_linear_map():
    torch.manual_seed(1)
    w = torch.randn(D, D)
    j = jacobian_batched(batchable(w), torch.randn(D), chunk=2)
    assert torch.allclose(j, w, atol=1e-4)


def test_batched_extraction_recovers_J_from_an_AFFINE_map():
    """Blocks have biases, so the map is affine, not linear.

    Subtracting fn(0) is what makes the extraction exact. Without it every
    column carries the bias and the lens is wrong by a constant that looks like
    signal.
    """
    torch.manual_seed(9)
    w = torch.randn(D, D)
    bias = torch.randn(D) * 3.0
    j = jacobian_batched(batchable(w, bias), torch.randn(D), chunk=3)
    assert torch.allclose(j, w, atol=1e-4)


def test_batched_extraction_agrees_with_the_jvp_reference():
    """The fast path assumes affineness; the reference assumes nothing.

    d_model jvp calls per layer per prompt is millions of forward passes on a
    real model, so the batched path is what runs. Its assumption is therefore
    verified against the general method rather than asserted.
    """
    torch.manual_seed(10)
    w = torch.randn(D, D)
    bias = torch.randn(D)
    point = torch.randn(D)
    fast = jacobian_batched(batchable(w, bias), point, chunk=4)
    reference = jacobian_by_jvp(batchable(w, bias), point)
    assert torch.allclose(fast, reference, atol=1e-4)


def test_linearisation_residual_is_zero_for_a_map_that_really_is_linear():
    """A DIAGNOSTIC, not a gate — and the distinction was a real correction.

    The earlier `affine_residual` compared a GLOBAL affine prediction against
    the map, on the premise that freezing attention and norms makes the
    residual-to-residual map affine. It does not: the MLP activation stays
    non-linear. On the first real fit that check measured 40.3 against a 1e-3
    limit and refused a perfectly good fit.

    A Jacobian IS a local linearisation. What is worth recording is how far it
    holds LOCALLY, which is what this measures.
    """
    torch.manual_seed(11)
    w = torch.randn(D, D)
    point = torch.randn(D)

    linear = batchable(w)
    assert linearisation_residual(linear, point, jacobian_batched(linear, point)) < 1e-4


def test_linearisation_residual_is_LARGER_for_a_curved_map():
    """Informative, not disqualifying: it says how local the lens is."""
    torch.manual_seed(12)
    w = torch.randn(D, D)
    point = torch.randn(D)

    def curved(h):
        base = h @ w.T if h.dim() > 1 else w @ h
        return base + torch.tanh(base * 3.0) * 5.0

    j = jacobian_batched(curved, point)
    straight = jacobian_batched(batchable(w), point)

    assert linearisation_residual(curved, point, j) > linearisation_residual(
        batchable(w), point, straight
    )


def test_the_jacobian_is_the_DERIVATIVE_not_a_secant():
    """The correction a hardware run forced.

    For a CURVED map the secant `fn(e_i) - fn(0)` and the derivative at the
    point are different matrices. The secant is what the first implementation
    computed, on a premise about freezing that was wrong — and it is
    well-shaped, plausible and not a Jacobian.
    """
    torch.manual_seed(13)
    w = torch.randn(D, D)
    point = torch.randn(D) * 2.0

    def curved(h):
        base = h @ w.T if h.dim() > 1 else w @ h
        return base + torch.tanh(base) * 3.0

    derivative = jacobian_batched(curved, point)
    reference = jacobian_by_jvp(curved, point)

    # The production path agrees with forward-mode AD at the point...
    assert torch.allclose(derivative, reference, atol=1e-4)

    # ...and does NOT agree with the secant, which is the point of the fix.
    zero = curved(torch.zeros_like(point))
    secant = torch.stack(
        [
            curved(torch.eye(D)[i]) - zero
            for i in range(D)
        ],
        dim=1,
    )
    assert not torch.allclose(derivative, secant, atol=1e-2), (
        "the fixture is not curved enough to tell a secant from a derivative"
    )


def test_extraction_is_not_the_transpose():
    """A transposed assembly is symmetric-looking and passes a norm check."""
    torch.manual_seed(2)
    w = torch.randn(D, D)
    j = jacobian_batched(batchable(w), torch.randn(D), chunk=3)
    assert not torch.allclose(j, w.T, atol=1e-3), "fixture is accidentally symmetric"
    assert torch.allclose(j, w, atol=1e-4)


def test_fit_recovers_the_known_jacobian_at_every_layer():
    stack, weights, scales = make_stack(3)
    fitter = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=2, chunk=3
    )
    result = fitter.fit([f"prompt {i}" for i in range(4)])

    assert result.d_model == D
    # Only layers AT OR BELOW the target have a defined Jacobian to it; above
    # it the gradient is zero by causality and `fit` refuses them outright.
    for layer in range(target_index(len(weights)) + 1):
        expected = analytic_jacobian(weights, scales, layer)
        got = result.jacobians[layer].to(torch.float32)
        assert torch.allclose(got, expected, atol=2e-2), f"layer {layer}"


class NormHookedFitter(JacobianFitter):
    """The WRONG fitter: captures the residual at the next block's NORM.

    This is what hooking `residual_norm_module` actually does — the norm's
    rescaling ends up outside the fitted map instead of inside it. Reproduced
    rather than described, because in production the difference is plausible
    numbers and no error at all.

    Overrides `_capture_module`, the seam the real fitter exposes for exactly
    this control. An earlier version overrode `_sub_network`; when the fitter
    moved to reverse mode that method vanished and the override became INERT —
    the control silently stopped biting while still reporting green.
    """

    def _capture_module(self, layer: int):
        if layer + 1 >= self.structure.num_layers:
            # No next block to mis-hook; the top of the stack is unaffected
            # either way, which is itself worth knowing — the defect hides in
            # depth rather than at the edges.
            return super()._capture_module(layer)
        return self.structure.layers_module[layer + 1].input_norm


def test_hook_target_is_the_decoder_layer_not_a_norm():
    """A wrong capture point must never ship silently.

    Hooking `residual_norm_module` puts the norm's rescaling outside the fitted
    map — plausible numbers, signal scaled away, no error (PADR IDL-38).

    DETECTION NOW COMES IN TWO FORMS and either is a pass: a norm downstream of
    the target has NO gradient path to it and is refused outright, and one
    upstream yields a measurably different lens. The old assertion accepted only
    the second, so tightening the fitter would have looked like a regression.
    """
    stack, _, _ = make_stack(11)
    structure = Structure(stack)
    right = JacobianFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)
    wrong = NormHookedFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)

    layer = 0  # well below the target, so both fitters can run
    good = right.fit(["abc"], layers=[layer]).jacobians[layer].to(torch.float32)
    try:
        bad = wrong.fit(["abc"], layers=[layer]).jacobians[layer].to(torch.float32)
    except ValueError:
        return  # refused outright — the strongest possible detection

    assert not torch.allclose(good, bad, atol=1e-3), (
        "hooking the norm produced the same lens as hooking resid_post — the "
        "control does not bite, so a wrong hook target would ship undetected"
    )

def test_corpus_floor_is_refused_not_warned():
    stack, _, _ = make_stack(5)
    fitter = JacobianFitter(stack, Tokenizer(), Structure(stack), freeze_qk=False)
    with pytest.raises(ValueError, match=str(MIN_PROMPTS)):
        fitter.fit(["one", "two"])


def test_shard_merge_is_weighted_by_prompt_count():
    """An unweighted mean over-weights a short shard, silently."""
    a = {0: torch.full((2, 2), 1.0)}
    b = {0: torch.full((2, 2), 3.0)}

    merged = merge_shards([a, b], [300, 100])
    assert torch.allclose(merged[0], torch.full((2, 2), 1.5))

    unweighted = merge_shards([a, b], [1, 1])
    assert torch.allclose(unweighted[0], torch.full((2, 2), 2.0))
    assert not torch.allclose(merged[0], unweighted[0])


def test_shard_merge_rejects_mismatched_layer_sets():
    with pytest.raises(ValueError, match="different layer sets"):
        merge_shards([{0: torch.zeros(2, 2)}, {1: torch.zeros(2, 2)}], [1, 1])


def test_convergence_signal_is_a_property_of_j_alone():
    """The stopping rule must not consult a readout.

    BR-004 forbids next-token agreement as a quality metric, and every
    readout-quality proxy drifts toward it. `relative_change` takes only the
    accumulated Jacobians, which is enforced here by its signature having
    nowhere to put a model.
    """
    prev = {0: torch.ones(3, 3)}
    same = {0: torch.ones(3, 3)}
    moved = {0: torch.ones(3, 3) * 2}

    assert relative_change(prev, same) == pytest.approx(0.0)
    assert relative_change(prev, moved) > 0.1
    assert relative_change({}, moved) == float("inf")

    import inspect

    params = set(inspect.signature(relative_change).parameters)
    assert params == {"previous", "current"}, (
        "relative_change gained a parameter; if that parameter is a model or a "
        "readout, the fitter is converging on output agreement (BR-004)"
    )


def test_fitter_module_names_no_architecture():
    """The old SUPPORTED_ARCHITECTURES whitelist is not coming back (BR-032)."""
    import ast
    import inspect

    from src.ml import jlens_fitter

    source = inspect.getsource(jlens_fitter)
    tree = ast.parse(source)
    # Docstrings may name models when explaining WHY; executable code may not.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Name):
            assert not any(
                arch in node.id.lower()
                for arch in ("lfm2", "gemma", "llama", "granite", "qwen", "mistral")
            ), f"architecture name in executable path: {node.id}"


def test_norm_discovery_does_not_capture_a_block_merely_named_for_a_norm():
    """`endswith("norm")`, not `contains`.

    A substring match captures anything NAMED for a norm — `NormedBlock` here,
    a `NormalizedAttention` elsewhere — and freezing a decoder block is not a
    no-op: it replaces the whole block with an elementwise rescaling and yields
    a lens with no error anywhere. This fixture exists because that is exactly
    what happened when the rule was `contains`.
    """
    from src.ml.jlens_fitter import _norm_modules

    stack, _, _ = make_stack(20)
    found = _norm_modules(stack)

    assert found, "no norm modules discovered at all"
    assert all(type(m).__name__ == "ScaleNorm" for m in found), (
        f"captured a non-norm module: {[type(m).__name__ for m in found]}"
    )
    assert not any(isinstance(m, NormedBlock) for m in found)


class LeakyFitter(JacobianFitter):
    """A fitter whose sub-network is NOT affine — an incomplete freeze.

    Reproduces the failure the affine guard exists for: if a norm or an
    attention pattern escapes the freeze, the extracted matrix is a local
    linearisation of nothing in particular, and it is a well-shaped tensor of
    plausible magnitude.
    """

    def _sub_network(self, input_ids, layer):
        point, forward = super()._sub_network(input_ids, layer)

        def leaky(h):
            out = forward(h)
            return out + torch.tanh(out) * 5.0

        return point, leaky


def test_position_spread_is_recorded_and_means_what_it_says():
    """The per-layer number is SPREAD ACROSS SOURCE POSITIONS, not a residual.

    It replaced a "linearisation residual" that had stopped describing
    anything: it compared the MEAN J against a SINGLE position's activation and
    the SUMMED target — three different objects.

    Spread is zero on a position-independent stack because every source
    position genuinely gives the same Jacobian there, and non-zero the moment
    positions mix. For intervention work it is the number that matters: a lens
    with large spread transports differently depending on where in the sequence
    it is applied.
    """
    stack, _, _ = make_stack(21)
    flat = JacobianFitter(stack, Tokenizer(), stack_structure := Structure(stack),
                          min_prompts=1, chunk=3)
    result = flat.fit(["abc"], layers=[0])
    assert result.position_spread_mean, "no per-layer spread was recorded at all"
    assert result.position_spread_mean[0] == pytest.approx(0.0, abs=1e-5), (
        "a position-independent stack must show ZERO spread; a non-zero value "
        "means the number is measuring something else"
    )

    model, structure, tok = _mixing_model(3, 4, 6)
    mixed = JacobianFitter(model, tok, structure, min_prompts=1, chunk=2)
    mixed_result = mixed.fit(["x"], layers=[0])
    assert mixed_result.position_spread_mean[0] > 1e-4, (
        "a position-MIXING stack must show non-zero spread, or the measure is "
        "not sensitive to the thing it claims to measure"
    )

def test_a_properly_frozen_fit_is_accepted():
    """Negative control for the guard: it must not refuse a good fit."""
    stack, _, _ = make_stack(22)
    fitter = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=1, chunk=3
    )
    assert fitter.fit(["abc"]).jacobians


# ---------------------------------------------------------------------------
# Grouped-query attention under the freeze
#
# The first real fit against a GQA model died in `weights @ value` with an 8-vs-4
# head mismatch. Every test above it was green: the analytic stack has no
# attention at all, and the hardware acceptance run used GPT-2, which is plain
# multi-head attention where n_kv_heads == n_heads. The GQA branch had never
# once executed, so the fixtures agreed by construction.
#
# The shape error is the GOOD failure. The dangerous one is silent: expanding V
# with `repeat` instead of `repeat_interleave` produces a correctly shaped
# result that pairs each query head with the WRONG KV head, and nothing raises.
#
# MUTATION CONTROLS (each must turn this section red):
#   * delete the n_rep expansion entirely  -> "gqa is handled" fails (RuntimeError)
#   * repeat_interleave -> repeat          -> "kv head pairing" fails
#   * n_rep > 1 -> n_rep > 0 (or >= 1)     -> "mha is untouched" fails
# ---------------------------------------------------------------------------


def _reference_kv_repeat(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """transformers' `repeat_kv`, written out independently of the code under test."""
    b, h, s, d = t.shape
    return t[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@pytest.mark.parametrize(
    "n_heads,n_kv_heads,label",
    [(8, 4, "gqa"), (8, 1, "mqa"), (4, 4, "mha")],
)
def test_frozen_sdpa_handles_every_kv_head_arity(n_heads, n_kv_heads, label):
    """The freeze must survive GQA, MQA and MHA, and match a reference by VALUE.

    Head counts come off the tensors, never off a config or an architecture
    name — a model this repo has never seen must work (BR-032).
    """
    from src.ml.jlens_fitter import frozen_attention_and_norms

    torch.manual_seed(0)
    b, s, d = 1, 5, 6
    n_rep = n_heads // n_kv_heads
    q = torch.randn(b, n_heads, s, d, dtype=torch.float64)
    k = torch.randn(b, n_kv_heads, s, d, dtype=torch.float64)
    v = torch.randn(b, n_kv_heads, s, d, dtype=torch.float64)

    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    # What the unfrozen model computes, with V repeated the way transformers does.
    expected = real_sdpa(q, _reference_kv_repeat(k, n_rep), _reference_kv_repeat(v, n_rep))

    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        # "gqa is handled" / "mha is untouched": this raised RuntimeError
        # ("size of tensor a (8) must match ... b (4)") before the fix.
        got = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, enable_gqa=(n_rep > 1)
        )

    assert got.shape == (b, n_heads, s, d), f"{label}: wrong output shape"
    # "kv head pairing": `repeat` instead of `repeat_interleave` has the right
    # shape here and fails only on these values.
    assert torch.allclose(got, expected, atol=1e-9), f"{label}: wrong attention output"


def test_the_freeze_stops_gradient_at_qk_but_not_at_v_under_gqa():
    """The POINT of the patch, asserted on the arity that broke it.

    A patch that merely stops crashing could equally have stopped freezing.
    """
    from src.ml.jlens_fitter import frozen_attention_and_norms

    torch.manual_seed(1)
    b, n_heads, n_kv, s, d = 1, 8, 4, 4, 6
    q = torch.randn(b, n_heads, s, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, n_kv, s, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, n_kv, s, d, dtype=torch.float64, requires_grad=True)

    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    out.sum().backward()

    assert v.grad is not None and v.grad.abs().sum() > 0, "V must carry gradient"
    assert q.grad is None or q.grad.abs().sum() == 0, "Q must be frozen"
    assert k.grad is None or k.grad.abs().sum() == 0, "K must be frozen"


def test_the_sdpa_patch_is_removed_on_exit():
    """A leaked global patch would silently freeze attention for every later caller."""
    from src.ml.jlens_fitter import frozen_attention_and_norms

    before = torch.nn.functional.scaled_dot_product_attention
    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        assert torch.nn.functional.scaled_dot_product_attention is not before
    assert torch.nn.functional.scaled_dot_product_attention is before


# ---------------------------------------------------------------------------
# The recipe contract and the written config must AGREE (F3-F7)
#
# `JLensArtifactRecipe` declared target_layer, target_position_scope,
# aggregation, seq_len and library_versions. `_config_yaml` wrote none of those
# names, and two of the schema's defaults actively contradicted the fitter:
#
#   target_layer          "penultimate"      -> the fit runs to the FINAL block
#   target_position_scope "all_subsequent"   -> the extraction is SELF_ONLY
#
# So the schema read as a description of the artifact and described something
# else, while the provenance BR-007 requires said nothing about how the lens
# was built.
#
# MUTATION CONTROLS (each must turn this section red):
#   * drop target_position_scope from the config   -> "vocabulary" fails
#   * schema default back to "all_subsequent"      -> "agrees with the fitter" fails
#   * config claims frozen_qk wholesale            -> "per layer" fails
#   * residuals back to the last prompt only       -> "over the corpus" fails
# ---------------------------------------------------------------------------


class _RecipeLoaded:
    name = "org/model"
    d_model = 4
    n_layers = 6
    n_vocab = 32
    model = None
    structure = type("S", (), {"num_layers": 6, "attention_module": None})()


class _RecipeResult:
    # `_config_yaml` records the fitted layers from this.
    jacobians = {0: None, 1: None}
    scales = {0: 1.0, 1: 2.5}
    prompts_seen = 120
    converged = True
    convergence_delta = 1e-3
    position_spread_mean = {0: 0.011, 1: 0.022}
    position_spread_max = {0: 0.031, 1: 0.044}
    degenerate_layers = [1]


def _written_config(freeze_qk=True):
    from src.workers.jlens_fit_tasks import _config_yaml

    return _config_yaml(
        _RecipeLoaded(), _RecipeResult(), freeze_qk=freeze_qk, corpus_name="c"
    )


def test_the_config_uses_the_recipe_schema_vocabulary():
    """Every recipe field the fitter can know must appear in the artifact."""
    text = _written_config()
    for key in (
        "target_layer:",
        "target_position_scope:",
        "aggregation:",
        "seq_len:",
        "corpus:",
        "n_prompts:",
        "dtype:",
    ):
        assert key in text, (
            f"{key!r} is declared by JLensArtifactRecipe and absent from the "
            "written config — the schema is a contract nothing honours"
        )


def test_the_declared_scope_agrees_with_what_the_fitter_does():
    """The schema must describe the recipe the code implements.

    THIS TEST PREVIOUSLY PINNED THE WRONG RECIPE. It asserted
    `self_only_isolated`, faithfully describing a fitter that took one source
    position on a length-1 sub-network — and the source paper takes an
    expectation over ALL source positions of the summed effect on ALL
    subsequent target positions. A test can hold an implementation and its
    schema in perfect agreement while both disagree with the thing they model.
    """
    from src.schemas.jlens import JLensArtifactRecipe

    fields = JLensArtifactRecipe.model_fields
    assert fields["target_position_scope"].default == "all_subsequent", (
        "the schema must default to the paper's scope: an expectation over "
        "source positions of the effect on all subsequent target positions"
    )
    assert fields["target_layer"].default == "penultimate", (
        "BRD A.2 choice 1 defaults to penultimate — the last block is "
        "specialised for next-token calibration and adds readout noise"
    )
    assert fields["attention_gradients"].default == "full", (
        "the paper's standard recipe is FULL backward; freezing Q/K is an "
        "ablation, not the default"
    )

    text = _written_config()
    assert "target_position_scope: all_subsequent" in text
    assert "source_position_aggregation: mean_over_all_positions" in text
    assert "differentiation_mode: reverse" in text


def test_the_freeze_is_recorded_per_layer_not_wholesale():
    """A hybrid model's lens is not 'frozen_qk' when 6 of 16 layers attend."""
    text = _written_config(freeze_qk=True)
    assert "attention_gradients_requested: frozen_qk" in text
    assert "attention_gradients_applied_to_layers:" in text
    # The bare wholesale claim must be gone.
    assert "\nattention_gradients: frozen_qk" not in text, (
        "the config still describes the artifact as frozen_qk wholesale, which "
        "overstates the treatment on any model where some layers do not attend"
    )


def test_the_position_spread_describes_the_CORPUS():
    """MIS-E2E-081 renamed these keys: they hold source-position spread, not a
    linearisation residual, and the artifact travels to HuggingFace."""
    """It used to be whichever prompt happened to be last."""
    text = _written_config()
    assert "source_position_spread_mean:" in text
    assert "source_position_spread_max:" in text
    # Both figures, because the mean says what is typical and the max says how
    # bad it gets — a lens is judged on the second.
    assert "0.031" in text and "0.011" in text


def test_the_fitter_accumulates_position_spread_over_every_prompt():
    """Not the last one. Exercised through a real fit."""
    stack, _, _ = make_stack(31)
    fitter = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=3, chunk=3
    )
    result = fitter.fit(["abc", "abcd", "abcde"])

    assert result.position_spread_mean, "no corpus residual was recorded"
    for layer, worst in result.position_spread_max.items():
        assert worst >= result.position_spread_mean[layer] - 1e-9, (
            f"layer {layer}: max {worst} is below the mean "
            f"{result.position_spread_mean[layer]} — the accumulation is wrong"
        )


def test_the_sdpa_patch_is_serialised_and_always_restored():
    """F11: the patch is process-wide, so overlapping freezes must not nest.

    Two concurrent fits would otherwise restore each other's originals in the
    wrong order and leave attention frozen for every model in the process
    afterwards — silently, permanently, presenting only as "readouts went
    strange".

    THE SLEEP IS LOad-BEARING. Without it the threads enter and leave the
    window faster than they can interleave, and removing the lock entirely
    still passes — verified: that mutation survived the first version of this
    test. Holding the window open forces the race the lock exists to prevent.

    MUTATION CONTROL: delete the _FREEZE_LOCK acquire/release -> this fails.
    """
    import threading
    import time

    from src.ml.jlens_fitter import frozen_attention_and_norms

    pristine = torch.nn.functional.scaled_dot_product_attention
    order = []
    lock = threading.Lock()

    def worker(tag):
        with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
            with lock:
                order.append(f"{tag}-in")
            assert torch.nn.functional.scaled_dot_product_attention is not pristine
            time.sleep(0.03)
            with lock:
                order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert torch.nn.functional.scaled_dot_product_attention is pristine, (
        "the SDPA patch leaked: every later model in this process is now "
        "running with frozen attention"
    )
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0], (
            f"freeze windows interleaved, so two fits shared one patched "
            f"global and restored it out of order: {order}"
        )


# ---------------------------------------------------------------------------
# The final layer's lens IS the logit lens (degenerate layers)
#
# The last decoder layer has no blocks after it, so its sub-network is the
# identity map and J = I exactly. That is correct — and it means a Diff at that
# layer is empty because the two lenses ARE the same lens, not because they
# happen to agree. Observed on the cluster: gemma L25 read identically through
# both lenses while L24 differed, and nothing in the product said why.
#
# MUTATION CONTROLS:
#   * identity_distance returns 0 for everything -> "only the identity" fails
#   * degenerate_layers is never populated       -> "records" fails
# ---------------------------------------------------------------------------


def test_identity_distance_is_zero_only_for_the_identity():
    from src.ml.jlens_fitter import IDENTITY_TOLERANCE, identity_distance

    assert identity_distance(torch.eye(8)) == pytest.approx(0.0, abs=1e-9)
    # A scaled identity is NOT the identity: it is the same direction with a
    # different magnitude, and probe scores through it differ.
    assert identity_distance(torch.eye(8) * 2.0) > IDENTITY_TOLERANCE
    assert identity_distance(torch.randn(8, 8)) > IDENTITY_TOLERANCE
    # Non-square cannot be compared and must not read as "close to identity".
    assert identity_distance(torch.randn(4, 8)) == float("inf")


def test_the_fit_records_which_layers_are_degenerate():
    """The TARGET layer's Jacobian to itself is the identity.

    Under the default penultimate target that is block N-2, not N-1. Layer N-1
    is above the target and is REFUSED outright — its gradient is zero by
    causality, and a zero lens reads out as uniform noise wearing a confident
    face.
    """
    stack, _, _ = make_stack(37)
    structure = Structure(stack)
    fitter = JacobianFitter(
        stack, Tokenizer(), structure, min_prompts=1, chunk=3
    )
    tgt = target_index(structure.num_layers)
    result = fitter.fit(["abc"], layers=[tgt])

    assert tgt in result.degenerate_layers, (
        f"layer {tgt} is the target block, so J = I there — degenerate_layers "
        f"was {result.degenerate_layers}"
    )


def test_a_layer_above_the_target_is_refused_not_silently_dropped():
    """A zero Jacobian is worse than a missing one: it reads out confidently."""
    import pytest as _pytest

    stack, _, _ = make_stack(38)
    structure = Structure(stack)
    fitter = JacobianFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)

    with _pytest.raises(ValueError, match="above the penultimate target"):
        fitter.fit(["abc"], layers=[structure.num_layers - 1])

    # And "every layer" means every layer that CAN be fitted, rather than
    # raising on the ordinary call.
    assert max(fitter.fit(["abc"]).jacobians) == target_index(structure.num_layers)



# ---------------------------------------------------------------------------
# THE PAPER'S DEFINITION (D1 + D2)
#
#     J_l = E_t [ sum_{t' >= t} d h_target,t' / d h_l,t ]
#
# The old forward-mode fitter took ONE source position per prompt and ran the
# remaining blocks on a length-1 sequence. That is neither the expectation over
# source positions the paper takes, nor the sum over subsequent target
# positions — and the length-1 sub-network also gave the perturbed position full
# attention weight instead of its real share.
#
# MUTATION CONTROLS (each must turn this section red):
#   * mean over source positions -> take only the last  -> "every source" fails
#   * cotangent set at one target position only         -> "all subsequent" fails
# ---------------------------------------------------------------------------


class _MixingBlock(torch.nn.Module):
    """Mixes positions, so source-position averaging is observable.

    A position-independent block makes every source position identical, and a
    fitter that used only the last one would agree by construction — the exact
    trap that let the old single-position path look correct.
    """

    def __init__(self, d_model: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(d_model, d_model, generator=g) * 0.3)

    def forward(self, hidden, **_kw):
        pooled = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        return (pooled @ self.w,)


def _mixing_model(n_blocks: int, d_model: int, seq: int):
    blocks = torch.nn.ModuleList(
        [_MixingBlock(d_model, seed=i) for i in range(n_blocks)]
    )

    class _S:
        layers_module = blocks
        num_layers = n_blocks
        attention_module = None

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.zeros(1, seq, dtype=torch.long)}

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.embed = torch.nn.Embedding(2, d_model)

        def forward(self, input_ids=None, **_kw):
            h = self.embed(input_ids)
            # Positions differ, or averaging over them proves nothing.
            h = h + torch.arange(
                h.shape[1], dtype=h.dtype, device=h.device
            ).view(1, -1, 1) * 0.1
            for b in self.blocks:
                h = b(h)[0]
            return h

    return _Model(), _S(), _Tok()


def _reference_J(model, structure, tok, layer: int, seq: int, d_model: int):
    """The paper's quantity, computed independently of the fitter."""
    captured = {}

    def cap(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    tgt = target_index(structure.num_layers)

    def cap_t(_m, _i, out):
        captured["t"] = out[0] if isinstance(out, tuple) else out

    h1 = structure.layers_module[layer].register_forward_hook(cap)
    h2 = structure.layers_module[tgt].register_forward_hook(cap_t)
    try:
        model(input_ids=tok("x")["input_ids"])
    finally:
        h1.remove(); h2.remove()

    src, target = captured["h"], captured["t"]
    rows = []
    for j in range(d_model):
        cot = torch.zeros_like(target)
        cot[..., j] = 1.0            # every target position t'
        (g,) = torch.autograd.grad(target, src, grad_outputs=cot, retain_graph=True)
        rows.append(g[0].mean(dim=0))  # mean over source positions t
    return torch.stack(rows)


def test_J_matches_the_papers_definition_on_a_position_mixing_stack():
    """Independent reference, not a restatement of the implementation."""
    d_model, n_blocks, seq = 4, 3, 5
    model, structure, tok = _mixing_model(n_blocks, d_model, seq)
    fitter = JacobianFitter(model, tok, structure, min_prompts=1, chunk=2)

    got = fitter.fit(["x"], layers=[0]).jacobians[0].to(torch.float32)
    want = _reference_J(model, structure, tok, 0, seq, d_model)

    assert torch.allclose(got, want, atol=1e-3), (
        f"fitted J does not match E_t[sum_t' dh/dh]:\n{got}\nvs\n{want}"
    )


def test_every_source_position_contributes():
    """Averaging over t must differ from taking the last t alone."""
    d_model, n_blocks, seq = 4, 3, 6
    model, structure, tok = _mixing_model(n_blocks, d_model, seq)
    fitter = JacobianFitter(model, tok, structure, min_prompts=1, chunk=2)
    got = fitter.fit(["x"], layers=[0]).jacobians[0].to(torch.float32)

    # The same computation restricted to the FINAL source position — what the
    # old fitter produced.
    captured = {}
    tgt = target_index(structure.num_layers)
    h1 = structure.layers_module[0].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("h", o[0] if isinstance(o, tuple) else o)
    )
    h2 = structure.layers_module[tgt].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("t", o[0] if isinstance(o, tuple) else o)
    )
    try:
        model(input_ids=tok("x")["input_ids"])
    finally:
        h1.remove(); h2.remove()
    rows = []
    for j in range(d_model):
        cot = torch.zeros_like(captured["t"]); cot[..., j] = 1.0
        (g,) = torch.autograd.grad(
            captured["t"], captured["h"], grad_outputs=cot, retain_graph=True
        )
        rows.append(g[0, -1])           # LAST source position only
    last_only = torch.stack(rows)

    assert not torch.allclose(got, last_only, atol=1e-4), (
        "averaging over source positions gave the same answer as using only "
        "the last one, so this fixture does not mix positions and the test "
        "proves nothing"
    )


def test_the_sdpa_patch_is_serialised_and_always_restored():
    """F11: the patch is process-wide, so overlapping freezes must not nest.

    Two concurrent fits would otherwise restore each other's originals in the
    wrong order and leave attention frozen for every model in the process
    afterwards — silently, permanently, presenting only as "readouts went
    strange".

    THE SLEEP IS LOad-BEARING. Without it the threads enter and leave the
    window faster than they can interleave, and removing the lock entirely
    still passes — verified: that mutation survived the first version of this
    test. Holding the window open forces the race the lock exists to prevent.

    MUTATION CONTROL: delete the _FREEZE_LOCK acquire/release -> this fails.
    """
    import threading
    import time

    from src.ml.jlens_fitter import frozen_attention_and_norms

    pristine = torch.nn.functional.scaled_dot_product_attention
    order = []
    lock = threading.Lock()

    def worker(tag):
        with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
            with lock:
                order.append(f"{tag}-in")
            assert torch.nn.functional.scaled_dot_product_attention is not pristine
            time.sleep(0.03)
            with lock:
                order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert torch.nn.functional.scaled_dot_product_attention is pristine, (
        "the SDPA patch leaked: every later model in this process is now "
        "running with frozen attention"
    )
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0], (
            f"freeze windows interleaved, so two fits shared one patched "
            f"global and restored it out of order: {order}"
        )


# ---------------------------------------------------------------------------
# The final layer's lens IS the logit lens (degenerate layers)
#
# The last decoder layer has no blocks after it, so its sub-network is the
# identity map and J = I exactly. That is correct — and it means a Diff at that
# layer is empty because the two lenses ARE the same lens, not because they
# happen to agree. Observed on the cluster: gemma L25 read identically through
# both lenses while L24 differed, and nothing in the product said why.
#
# MUTATION CONTROLS:
#   * identity_distance returns 0 for everything -> "only the identity" fails
#   * degenerate_layers is never populated       -> "records" fails
# ---------------------------------------------------------------------------


def test_identity_distance_is_zero_only_for_the_identity():
    from src.ml.jlens_fitter import IDENTITY_TOLERANCE, identity_distance

    assert identity_distance(torch.eye(8)) == pytest.approx(0.0, abs=1e-9)
    # A scaled identity is NOT the identity: it is the same direction with a
    # different magnitude, and probe scores through it differ.
    assert identity_distance(torch.eye(8) * 2.0) > IDENTITY_TOLERANCE
    assert identity_distance(torch.randn(8, 8)) > IDENTITY_TOLERANCE
    # Non-square cannot be compared and must not read as "close to identity".
    assert identity_distance(torch.randn(4, 8)) == float("inf")


def test_the_fit_records_which_layers_are_degenerate():
    """The TARGET layer's Jacobian to itself is the identity.

    Under the default penultimate target that is block N-2, not N-1. Layer N-1
    is above the target and is REFUSED outright — its gradient is zero by
    causality, and a zero lens reads out as uniform noise wearing a confident
    face.
    """
    stack, _, _ = make_stack(37)
    structure = Structure(stack)
    fitter = JacobianFitter(
        stack, Tokenizer(), structure, min_prompts=1, chunk=3
    )
    tgt = target_index(structure.num_layers)
    result = fitter.fit(["abc"], layers=[tgt])

    assert tgt in result.degenerate_layers, (
        f"layer {tgt} is the target block, so J = I there — degenerate_layers "
        f"was {result.degenerate_layers}"
    )


def test_a_layer_above_the_target_is_refused_not_silently_dropped():
    """A zero Jacobian is worse than a missing one: it reads out confidently."""
    import pytest as _pytest

    stack, _, _ = make_stack(38)
    structure = Structure(stack)
    fitter = JacobianFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)

    with _pytest.raises(ValueError, match="above the penultimate target"):
        fitter.fit(["abc"], layers=[structure.num_layers - 1])

    # And "every layer" means every layer that CAN be fitted, rather than
    # raising on the ordinary call.
    assert max(fitter.fit(["abc"]).jacobians) == target_index(structure.num_layers)



# ---------------------------------------------------------------------------
# The sub-network's SEQUENCE LENGTH changes the answer (F4, properly)
#
# The cheap path runs the remaining blocks on a LENGTH-1 sequence. A softmax
# over a single key is 1.0, so downstream attention hands the perturbed
# position its entire attention weight — where the real forward pass might give
# it 0.05. The value path comes out scaled up by that ratio.
#
# That is not a scope choice, it is a different computation, which is why the
# recipe now records `self_only_isolated` for it rather than `self_only`.
#
# MUTATION CONTROLS:
#   * full_sequence path truncates kwargs like the cheap one -> "real length" fails
#   * forward_full perturbs every position, not just the last -> "one position" fails
# ---------------------------------------------------------------------------


class _AttendingBlock(torch.nn.Module):
    """A block that MIXES POSITIONS, so sequence length is observable.

    Every fixture above is position-independent, which is precisely why a
    length-1 sub-network looked equivalent for so long: with no cross-position
    mixing the two paths agree by construction.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d_model) * 0.5)

    def forward(self, hidden, position_bias=None, **_kwargs):
        """Mixes positions AND consumes a per-position kwarg.

        `position_bias` is what makes kwarg TRUNCATION observable: a block that
        ignores its kwargs cannot tell a full-length mask from a mask sliced to
        one position, which is why an earlier version of this fixture let the
        truncation mutation survive.
        """
        if position_bias is not None:
            if position_bias.shape[1] != hidden.shape[1]:
                raise AssertionError(
                    f"position_bias has length {position_bias.shape[1]} for a "
                    f"sequence of {hidden.shape[1]} — the kwargs were sliced to "
                    "a different length than the hidden states"
                )
            hidden = hidden + position_bias.unsqueeze(-1)
        # Uniform attention over the (causal) prefix, then a linear map. With S
        # positions the last row averages S values; with S = 1 it sees only
        # itself, and its own contribution is S times heavier.
        pooled = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        return (pooled @ self.w,)


def _attending_structure(n_layers: int, d_model: int):
    blocks = torch.nn.ModuleList([_AttendingBlock(d_model) for _ in range(n_layers)])

    class _S:
        layers_module = blocks
        num_layers = n_layers
        attention_module = None

    return _S()


# The two forward-mode sub-network tests that lived here are GONE with the code
# they tested. They compared an isolated length-1 sub-network against a
# full-length one; the reverse-mode fitter has neither, and keeping tests for a
# deleted path would report coverage of a recipe the product no longer offers.
# Their subject — that sequence context changes the answer — is now covered by
# `test_J_matches_the_papers_definition_on_a_position_mixing_stack` and
# `test_every_source_position_contributes`, against the paper's definition
# rather than against one implementation of it.


def test_the_sdpa_patch_is_serialised_and_always_restored():
    """F11: the patch is process-wide, so overlapping freezes must not nest.

    Two concurrent fits would otherwise restore each other's originals in the
    wrong order and leave attention frozen for every model in the process
    afterwards — silently, permanently, presenting only as "readouts went
    strange".

    THE SLEEP IS LOad-BEARING. Without it the threads enter and leave the
    window faster than they can interleave, and removing the lock entirely
    still passes — verified: that mutation survived the first version of this
    test. Holding the window open forces the race the lock exists to prevent.

    MUTATION CONTROL: delete the _FREEZE_LOCK acquire/release -> this fails.
    """
    import threading
    import time

    from src.ml.jlens_fitter import frozen_attention_and_norms

    pristine = torch.nn.functional.scaled_dot_product_attention
    order = []
    lock = threading.Lock()

    def worker(tag):
        with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
            with lock:
                order.append(f"{tag}-in")
            assert torch.nn.functional.scaled_dot_product_attention is not pristine
            time.sleep(0.03)
            with lock:
                order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert torch.nn.functional.scaled_dot_product_attention is pristine, (
        "the SDPA patch leaked: every later model in this process is now "
        "running with frozen attention"
    )
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0], (
            f"freeze windows interleaved, so two fits shared one patched "
            f"global and restored it out of order: {order}"
        )


# ---------------------------------------------------------------------------
# The final layer's lens IS the logit lens (degenerate layers)
#
# The last decoder layer has no blocks after it, so its sub-network is the
# identity map and J = I exactly. That is correct — and it means a Diff at that
# layer is empty because the two lenses ARE the same lens, not because they
# happen to agree. Observed on the cluster: gemma L25 read identically through
# both lenses while L24 differed, and nothing in the product said why.
#
# MUTATION CONTROLS:
#   * identity_distance returns 0 for everything -> "only the identity" fails
#   * degenerate_layers is never populated       -> "records" fails
# ---------------------------------------------------------------------------


def test_identity_distance_is_zero_only_for_the_identity():
    from src.ml.jlens_fitter import IDENTITY_TOLERANCE, identity_distance

    assert identity_distance(torch.eye(8)) == pytest.approx(0.0, abs=1e-9)
    # A scaled identity is NOT the identity: it is the same direction with a
    # different magnitude, and probe scores through it differ.
    assert identity_distance(torch.eye(8) * 2.0) > IDENTITY_TOLERANCE
    assert identity_distance(torch.randn(8, 8)) > IDENTITY_TOLERANCE
    # Non-square cannot be compared and must not read as "close to identity".
    assert identity_distance(torch.randn(4, 8)) == float("inf")


def test_the_fit_records_which_layers_are_degenerate():
    """The TARGET layer's Jacobian to itself is the identity.

    Under the default penultimate target that is block N-2, not N-1. Layer N-1
    is above the target and is REFUSED outright — its gradient is zero by
    causality, and a zero lens reads out as uniform noise wearing a confident
    face.
    """
    stack, _, _ = make_stack(37)
    structure = Structure(stack)
    fitter = JacobianFitter(
        stack, Tokenizer(), structure, min_prompts=1, chunk=3
    )
    tgt = target_index(structure.num_layers)
    result = fitter.fit(["abc"], layers=[tgt])

    assert tgt in result.degenerate_layers, (
        f"layer {tgt} is the target block, so J = I there — degenerate_layers "
        f"was {result.degenerate_layers}"
    )


def test_a_layer_above_the_target_is_refused_not_silently_dropped():
    """A zero Jacobian is worse than a missing one: it reads out confidently."""
    import pytest as _pytest

    stack, _, _ = make_stack(38)
    structure = Structure(stack)
    fitter = JacobianFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)

    with _pytest.raises(ValueError, match="above the penultimate target"):
        fitter.fit(["abc"], layers=[structure.num_layers - 1])

    # And "every layer" means every layer that CAN be fitted, rather than
    # raising on the ordinary call.
    assert max(fitter.fit(["abc"]).jacobians) == target_index(structure.num_layers)



# ---------------------------------------------------------------------------
# THE PAPER'S DEFINITION (D1 + D2)
#
#     J_l = E_t [ sum_{t' >= t} d h_target,t' / d h_l,t ]
#
# The old forward-mode fitter took ONE source position per prompt and ran the
# remaining blocks on a length-1 sequence. That is neither the expectation over
# source positions the paper takes, nor the sum over subsequent target
# positions — and the length-1 sub-network also gave the perturbed position full
# attention weight instead of its real share.
#
# MUTATION CONTROLS (each must turn this section red):
#   * mean over source positions -> take only the last  -> "every source" fails
#   * cotangent set at one target position only         -> "all subsequent" fails
# ---------------------------------------------------------------------------


class _MixingBlock(torch.nn.Module):
    """Mixes positions, so source-position averaging is observable.

    A position-independent block makes every source position identical, and a
    fitter that used only the last one would agree by construction — the exact
    trap that let the old single-position path look correct.
    """

    def __init__(self, d_model: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.nn.Parameter(torch.randn(d_model, d_model, generator=g) * 0.3)

    def forward(self, hidden, **_kw):
        pooled = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        return (pooled @ self.w,)


def _mixing_model(n_blocks: int, d_model: int, seq: int):
    blocks = torch.nn.ModuleList(
        [_MixingBlock(d_model, seed=i) for i in range(n_blocks)]
    )

    class _S:
        layers_module = blocks
        num_layers = n_blocks
        attention_module = None

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.zeros(1, seq, dtype=torch.long)}

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.embed = torch.nn.Embedding(2, d_model)

        def forward(self, input_ids=None, **_kw):
            h = self.embed(input_ids)
            # Positions differ, or averaging over them proves nothing.
            h = h + torch.arange(
                h.shape[1], dtype=h.dtype, device=h.device
            ).view(1, -1, 1) * 0.1
            for b in self.blocks:
                h = b(h)[0]
            return h

    return _Model(), _S(), _Tok()


def _reference_J(model, structure, tok, layer: int, seq: int, d_model: int):
    """The paper's quantity, computed independently of the fitter."""
    captured = {}

    def cap(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    tgt = target_index(structure.num_layers)

    def cap_t(_m, _i, out):
        captured["t"] = out[0] if isinstance(out, tuple) else out

    h1 = structure.layers_module[layer].register_forward_hook(cap)
    h2 = structure.layers_module[tgt].register_forward_hook(cap_t)
    try:
        model(input_ids=tok("x")["input_ids"])
    finally:
        h1.remove(); h2.remove()

    src, target = captured["h"], captured["t"]
    rows = []
    for j in range(d_model):
        cot = torch.zeros_like(target)
        cot[..., j] = 1.0            # every target position t'
        (g,) = torch.autograd.grad(target, src, grad_outputs=cot, retain_graph=True)
        rows.append(g[0].mean(dim=0))  # mean over source positions t
    return torch.stack(rows)


def test_J_matches_the_papers_definition_on_a_position_mixing_stack():
    """Independent reference, not a restatement of the implementation."""
    d_model, n_blocks, seq = 4, 3, 5
    model, structure, tok = _mixing_model(n_blocks, d_model, seq)
    fitter = JacobianFitter(model, tok, structure, min_prompts=1, chunk=2)

    got = fitter.fit(["x"], layers=[0]).jacobians[0].to(torch.float32)
    want = _reference_J(model, structure, tok, 0, seq, d_model)

    assert torch.allclose(got, want, atol=1e-3), (
        f"fitted J does not match E_t[sum_t' dh/dh]:\n{got}\nvs\n{want}"
    )


def test_every_source_position_contributes():
    """Averaging over t must differ from taking the last t alone."""
    d_model, n_blocks, seq = 4, 3, 6
    model, structure, tok = _mixing_model(n_blocks, d_model, seq)
    fitter = JacobianFitter(model, tok, structure, min_prompts=1, chunk=2)
    got = fitter.fit(["x"], layers=[0]).jacobians[0].to(torch.float32)

    # The same computation restricted to the FINAL source position — what the
    # old fitter produced.
    captured = {}
    tgt = target_index(structure.num_layers)
    h1 = structure.layers_module[0].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("h", o[0] if isinstance(o, tuple) else o)
    )
    h2 = structure.layers_module[tgt].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("t", o[0] if isinstance(o, tuple) else o)
    )
    try:
        model(input_ids=tok("x")["input_ids"])
    finally:
        h1.remove(); h2.remove()
    rows = []
    for j in range(d_model):
        cot = torch.zeros_like(captured["t"]); cot[..., j] = 1.0
        (g,) = torch.autograd.grad(
            captured["t"], captured["h"], grad_outputs=cot, retain_graph=True
        )
        rows.append(g[0, -1])           # LAST source position only
    last_only = torch.stack(rows)

    assert not torch.allclose(got, last_only, atol=1e-4), (
        "averaging over source positions gave the same answer as using only "
        "the last one, so this fixture does not mix positions and the test "
        "proves nothing"
    )


def test_the_sdpa_patch_is_serialised_and_always_restored():
    """F11: the patch is process-wide, so overlapping freezes must not nest.

    Two concurrent fits would otherwise restore each other's originals in the
    wrong order and leave attention frozen for every model in the process
    afterwards — silently, permanently, presenting only as "readouts went
    strange".

    THE SLEEP IS LOad-BEARING. Without it the threads enter and leave the
    window faster than they can interleave, and removing the lock entirely
    still passes — verified: that mutation survived the first version of this
    test. Holding the window open forces the race the lock exists to prevent.

    MUTATION CONTROL: delete the _FREEZE_LOCK acquire/release -> this fails.
    """
    import threading
    import time

    from src.ml.jlens_fitter import frozen_attention_and_norms

    pristine = torch.nn.functional.scaled_dot_product_attention
    order = []
    lock = threading.Lock()

    def worker(tag):
        with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
            with lock:
                order.append(f"{tag}-in")
            assert torch.nn.functional.scaled_dot_product_attention is not pristine
            time.sleep(0.03)
            with lock:
                order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert torch.nn.functional.scaled_dot_product_attention is pristine, (
        "the SDPA patch leaked: every later model in this process is now "
        "running with frozen attention"
    )
    for i in range(0, len(order), 2):
        assert order[i].split("-")[0] == order[i + 1].split("-")[0], (
            f"freeze windows interleaved, so two fits shared one patched "
            f"global and restored it out of order: {order}"
        )


# ---------------------------------------------------------------------------
# The final layer's lens IS the logit lens (degenerate layers)
#
# The last decoder layer has no blocks after it, so its sub-network is the
# identity map and J = I exactly. That is correct — and it means a Diff at that
# layer is empty because the two lenses ARE the same lens, not because they
# happen to agree. Observed on the cluster: gemma L25 read identically through
# both lenses while L24 differed, and nothing in the product said why.
#
# MUTATION CONTROLS:
#   * identity_distance returns 0 for everything -> "only the identity" fails
#   * degenerate_layers is never populated       -> "records" fails
# ---------------------------------------------------------------------------


def test_identity_distance_is_zero_only_for_the_identity():
    from src.ml.jlens_fitter import IDENTITY_TOLERANCE, identity_distance

    assert identity_distance(torch.eye(8)) == pytest.approx(0.0, abs=1e-9)
    # A scaled identity is NOT the identity: it is the same direction with a
    # different magnitude, and probe scores through it differ.
    assert identity_distance(torch.eye(8) * 2.0) > IDENTITY_TOLERANCE
    assert identity_distance(torch.randn(8, 8)) > IDENTITY_TOLERANCE
    # Non-square cannot be compared and must not read as "close to identity".
    assert identity_distance(torch.randn(4, 8)) == float("inf")


def test_the_fit_records_which_layers_are_degenerate():
    """The TARGET layer's Jacobian to itself is the identity.

    Under the default penultimate target that is block N-2, not N-1. Layer N-1
    is above the target and is REFUSED outright — its gradient is zero by
    causality, and a zero lens reads out as uniform noise wearing a confident
    face.
    """
    stack, _, _ = make_stack(37)
    structure = Structure(stack)
    fitter = JacobianFitter(
        stack, Tokenizer(), structure, min_prompts=1, chunk=3
    )
    tgt = target_index(structure.num_layers)
    result = fitter.fit(["abc"], layers=[tgt])

    assert tgt in result.degenerate_layers, (
        f"layer {tgt} is the target block, so J = I there — degenerate_layers "
        f"was {result.degenerate_layers}"
    )


def test_a_layer_above_the_target_is_refused_not_silently_dropped():
    """A zero Jacobian is worse than a missing one: it reads out confidently."""
    import pytest as _pytest

    stack, _, _ = make_stack(38)
    structure = Structure(stack)
    fitter = JacobianFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)

    with _pytest.raises(ValueError, match="above the penultimate target"):
        fitter.fit(["abc"], layers=[structure.num_layers - 1])

    # And "every layer" means every layer that CAN be fitted, rather than
    # raising on the ordinary call.
    assert max(fitter.fit(["abc"]).jacobians) == target_index(structure.num_layers)



# ---------------------------------------------------------------------------
# The sub-network's SEQUENCE LENGTH changes the answer (F4, properly)
#
# The cheap path runs the remaining blocks on a LENGTH-1 sequence. A softmax
# over a single key is 1.0, so downstream attention hands the perturbed
# position its entire attention weight — where the real forward pass might give
# it 0.05. The value path comes out scaled up by that ratio.
#
# That is not a scope choice, it is a different computation, which is why the
# recipe now records `self_only_isolated` for it rather than `self_only`.
#
# MUTATION CONTROLS:
#   * full_sequence path truncates kwargs like the cheap one -> "real length" fails
#   * forward_full perturbs every position, not just the last -> "one position" fails
# ---------------------------------------------------------------------------


class _AttendingBlock(torch.nn.Module):
    """A block that MIXES POSITIONS, so sequence length is observable.

    Every fixture above is position-independent, which is precisely why a
    length-1 sub-network looked equivalent for so long: with no cross-position
    mixing the two paths agree by construction.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d_model) * 0.5)

    def forward(self, hidden, position_bias=None, **_kwargs):
        """Mixes positions AND consumes a per-position kwarg.

        `position_bias` is what makes kwarg TRUNCATION observable: a block that
        ignores its kwargs cannot tell a full-length mask from a mask sliced to
        one position, which is why an earlier version of this fixture let the
        truncation mutation survive.
        """
        if position_bias is not None:
            if position_bias.shape[1] != hidden.shape[1]:
                raise AssertionError(
                    f"position_bias has length {position_bias.shape[1]} for a "
                    f"sequence of {hidden.shape[1]} — the kwargs were sliced to "
                    "a different length than the hidden states"
                )
            hidden = hidden + position_bias.unsqueeze(-1)
        # Uniform attention over the (causal) prefix, then a linear map. With S
        # positions the last row averages S values; with S = 1 it sees only
        # itself, and its own contribution is S times heavier.
        pooled = hidden.cumsum(dim=1) / torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        return (pooled @ self.w,)


def _attending_structure(n_layers: int, d_model: int):
    blocks = torch.nn.ModuleList([_AttendingBlock(d_model) for _ in range(n_layers)])

    class _S:
        layers_module = blocks
        num_layers = n_layers
        attention_module = None

    return _S()


# ---------------------------------------------------------------------------
# The convergence threshold is the CALLER'S to set, and is recorded as used
#
# Both 400-prompt fits reported converged=false at the built-in 1e-3, and the
# only lever available was more corpus — an hour of GPU to answer a question
# about a threshold. Exposing it means a looser criterion can be ASKED for.
#
# What must never happen is a recipe naming a threshold the fit did not use:
# two artifacts fitted at different deltas would then be compared as though
# they had met the same criterion.
#
# MUTATION CONTROLS:
#   * task ignores the argument and takes the default -> "honours" fails
#   * config writes the default instead of the used   -> "records" fails
# ---------------------------------------------------------------------------


def test_the_fitter_honours_a_caller_supplied_convergence_delta():
    from src.ml.jlens_fitter import DEFAULT_CONVERGENCE_DELTA

    stack, _, _ = make_stack(41)
    loose = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False,
        min_prompts=1, chunk=3, convergence_delta=5e-2,
    )
    assert loose.convergence_delta == 5e-2
    assert loose.convergence_delta != DEFAULT_CONVERGENCE_DELTA

    result = loose.fit(["abc", "abcd"])
    # The RESULT carries what was used, which is what the config writer reads.
    assert result.convergence_delta == 5e-2


def test_the_config_records_the_convergence_delta_that_was_actually_used():
    """A recipe naming an unused threshold is worse than one naming none."""

    class _R(_RecipeResult):
        convergence_delta = 5e-2

    from src.workers.jlens_fit_tasks import _config_yaml

    text = _config_yaml(_RecipeLoaded(), _R(), freeze_qk=True, corpus_name="c")
    assert "convergence_delta: 0.05" in text, (
        f"config did not record the delta the fit ran at:\n{text[:400]}"
    )


def test_the_TASK_threads_the_caller_s_convergence_delta_to_the_fitter():
    """Exercises `fit_jlens_artifact` itself, not JacobianFitter directly.

    An earlier version of this section tested the fitter and the config writer
    in isolation. Both mutations survived: constructing a JacobianFitter by
    hand never runs the task, so a task that ignored its own argument passed.
    """
    from unittest.mock import MagicMock, patch

    import src.workers.jlens_fit_tasks as task_mod

    seen = {}

    class _Fitter:
        def __init__(self, *_a, convergence_delta=None, **_kw):
            seen["delta"] = convergence_delta

        def fit(self, *_a, **_kw):
            raise RuntimeError("stop here — construction is what is under test")

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        repo_id="org/model"
    )

    from contextlib import contextmanager

    @contextmanager
    def fake_db():
        yield db

    with patch("src.ml.jlens_fitter.JacobianFitter", _Fitter), patch(
        "src.core.database.get_sync_db", fake_db
    ), patch(
        "src.services.jlens_model_registry.load_for_readout",
        return_value=MagicMock(),
    ), patch.object(
        task_mod.fit_jlens_artifact, "update_state", MagicMock()
    ):
        try:
            task_mod.fit_jlens_artifact.run(
                model_id="m_1", prompts=["a"], convergence_delta=0.05
            )
        except RuntimeError:
            pass

    assert seen.get("delta") == 0.05, (
        f"the task built its fitter with delta={seen.get('delta')}, not the "
        "0.05 the caller asked for — the argument is being dropped"
    )


# ---------------------------------------------------------------------------
# Two gaps found by mutation, not by reading (review round 2)
#
# Both of these mutations SURVIVED a full green suite:
#   * the None-gradient branch silently `continue`d, leaving a zero block in a
#     matrix that was then accumulated as if measured
#   * `freeze_qk` flipped back to True, reinstating an ablation as the default
#     recipe with nothing objecting
# ---------------------------------------------------------------------------


def test_a_layer_off_the_path_to_the_target_is_refused_not_zeroed():
    """A zero block is worse than a missing one — it reads out confidently.

    `sums[layer]` starts at zeros, so a skipped gradient leaves that band of
    rows at zero and averages it in as though it had been measured.

    Reached by capturing at a module DOWNSTREAM of the target: the layer index
    passes the `beyond` guard, but no gradient path exists.
    """
    stack, _, _ = make_stack(51)
    structure = Structure(stack)
    wrong = NormHookedFitter(stack, Tokenizer(), structure, min_prompts=1, chunk=3)
    tgt = target_index(structure.num_layers)

    with pytest.raises(ValueError, match="received no gradient"):
        wrong.fit(["abc"], layers=[tgt])


def test_the_default_recipe_is_the_papers_full_backward():
    """Freezing is an ABLATION in the paper, not the standard recipe.

    Defaulting to frozen would silently make every artifact this product
    produces a variant, while the config recorded it as though it were the
    baseline everyone else's numbers came from.
    """
    stack, _, _ = make_stack(52)
    fitter = JacobianFitter(stack, Tokenizer(), Structure(stack))

    assert fitter.freeze_qk is False, (
        "attention-pattern freezing must be opt-in; the paper's standard recipe "
        "is full backward"
    )
    assert fitter.freeze_norms is False, (
        "norm freezing must be opt-in: freezing makes the map exactly affine, "
        "which is not what the paper computes"
    )
    assert fitter.target_layer == "penultimate", (
        "BRD A.2 choice 1 defaults to penultimate — the last block is "
        "specialised for next-token calibration and adds readout noise"
    )


# ---------------------------------------------------------------------------
# GAPS FOUND BY MUTATION, NOT BY READING (review round 2)
#
# Two mutations survived a full green suite:
#   * a None gradient silently `continue`d, leaving that block of rows at ZERO
#     and accumulating it as though measured — a lens with a dead band that
#     reads out as confident uniform noise
#   * `freeze_qk` flipped back to True, silently restoring the ablation as the
#     default recipe when the paper's standard is full backward
#
# Both had fixes in place. Neither had a test, so neither fix was defended.
# ---------------------------------------------------------------------------


class _DetachedBlock(torch.nn.Module):
    """A block that BREAKS the gradient path without changing any shape.

    This is what an un-differentiable step looks like from the outside: the
    forward pass succeeds, the tensors are the right size, and the Jacobian to
    anything upstream is simply undefined.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.eye(d_model))

    def forward(self, hidden, **_kw):
        return (hidden.detach() @ self.w,)


def test_a_layer_with_no_gradient_path_is_refused_not_zeroed():
    """A zero row-block is worse than a missing layer: it reads out confidently.

    MUTATION CONTROL: replace the raise with `continue` and this fails.
    """
    d_model, seq = 4, 3
    blocks = torch.nn.ModuleList(
        [_MixingBlock(d_model, seed=0), _DetachedBlock(d_model), _MixingBlock(d_model, seed=1)]
    )

    class _S:
        layers_module = blocks
        num_layers = 3
        attention_module = None

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.zeros(1, seq, dtype=torch.long)}

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.embed = torch.nn.Embedding(2, d_model)

        def forward(self, input_ids=None, **_kw):
            h = self.embed(input_ids)
            for b in self.blocks:
                h = b(h)[0]
            return h

    fitter = JacobianFitter(_M(), _Tok(), _S(), min_prompts=1, chunk=2)

    # Layer 0 sits below a detached block, so nothing reaches the target from it.
    with pytest.raises(ValueError, match="no gradient from the target"):
        fitter.fit(["x"], layers=[0])


def test_the_default_recipe_is_FULL_BACKWARD_not_the_ablation():
    """The paper's standard recipe, asserted as a default rather than assumed.

    Freezing attention patterns is an ABLATION in the source. Shipping it as
    the default silently changes what every artifact means, and nothing in the
    suite noticed when it was flipped.

    MUTATION CONTROL: set `freeze_qk: bool = True` (or `freeze_norms`) and this
    fails.
    """
    import inspect

    sig = inspect.signature(JacobianFitter.__init__)
    assert sig.parameters["freeze_qk"].default is False, (
        "freeze_qk defaults to True — that is the paper's ablation, not its "
        "standard recipe, and every artifact fitted here would silently be the "
        "variant rather than the baseline"
    )
    assert sig.parameters["freeze_norms"].default is False, (
        "freeze_norms defaults to True — freezing makes the map exactly affine, "
        "which is convenient and is not what the paper computes"
    )
    assert sig.parameters["target_layer"].default == "penultimate"

    # And the constructed object agrees with its own signature.
    stack, _, _ = make_stack(51)
    fitter = JacobianFitter(stack, Tokenizer(), Structure(stack), min_prompts=1)
    assert fitter.freeze_qk is False
    assert fitter.freeze_norms is False
    assert fitter.target_layer == "penultimate"


def test_the_backward_chunk_is_narrowed_to_stay_within_a_memory_bound(monkeypatch):
    """Observes the ACTUAL backward, not a restatement of the formula.

    Peak memory now scales with SEQUENCE LENGTH: reverse mode allocates
    `chunk x seq_len x d_model` per captured layer, so a constant retuned for
    one prompt length is not a bound. The corpus holds both short and long
    prompts, and the long ones are where a fit that passed every test OOMs on
    the first real card.

    An earlier version of this test recomputed the narrowing arithmetic and
    asserted against its own answer. Removing the narrowing from the fitter left
    it GREEN — it was testing a formula, not a code path.

    MUTATION CONTROL: replace `chunk = max(1, min(...))` with `chunk = self.chunk`
    and this fails.
    """
    import src.ml.jlens_fitter as fitter_mod

    d_model, n_blocks, seq = 4, 2, 8
    model, structure, tok = _mixing_model(n_blocks, d_model, seq)

    # A bound small enough that the chunk MUST narrow below DEFAULT_CHUNK.
    per_dim = seq * d_model * 4 * 1          # one captured layer
    monkeypatch.setattr(fitter_mod, "MAX_BACKWARD_BYTES", per_dim * 2)

    widths = []
    real_grad = torch.autograd.grad

    def spy(outputs, inputs, grad_outputs=None, **kw):
        if grad_outputs is not None and kw.get("is_grads_batched"):
            widths.append(int(grad_outputs.shape[0]))
        return real_grad(outputs, inputs, grad_outputs=grad_outputs, **kw)

    monkeypatch.setattr(torch.autograd, "grad", spy)

    fitter = JacobianFitter(
        model, tok, structure, min_prompts=1, chunk=fitter_mod.DEFAULT_CHUNK
    )
    fitter.fit(["x"], layers=[0])

    assert widths, "no batched backward was observed at all"
    assert max(widths) <= 2, (
        f"the backward ran with a cotangent batch of {max(widths)} under a bound "
        f"that permits 2 — the chunk was not narrowed, so a long prompt would "
        "allocate without limit"
    )
    assert max(widths) >= 1, "narrowing must never reach zero"


def test_a_layer_that_never_ran_is_refused_not_silently_dropped():
    """Coverage loss must be loud.

    A layer present in `layers_module` but never executed captures nothing. The
    old code excluded it from `present`, so it never accumulated, never appeared
    in the result, and the artifact shipped with fewer layers than requested and
    nothing saying which.

    MUTATION CONTROL: turn the `missing` check into `if False` and this fails.
    """
    d_model = 4

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.eye(d_model))

        def forward(self, hidden, **_kw):
            return (hidden @ self.w,)

    blocks = torch.nn.ModuleList([_Block() for _ in range(4)])

    class _S:
        layers_module = blocks
        num_layers = 4
        attention_module = None

    class _Tok:
        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

    class _SkippingModel(torch.nn.Module):
        """Runs blocks 0 and 2 only — block 1 is declared and never executed."""

        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.embed = torch.nn.Embedding(2, d_model)

        def forward(self, input_ids=None, **_kw):
            h = self.embed(input_ids)
            for i in (0, 2):
                h = self.blocks[i](h)[0]
            return h

    fitter = JacobianFitter(_SkippingModel(), _Tok(), _S(), min_prompts=1, chunk=2)

    with pytest.raises(ValueError, match=r"never produced an activation"):
        fitter.fit(["abc"], layers=[0, 1])


def test_the_semantic_probe_scans_every_fitted_layer_by_default():
    """The check may not assume WHERE an unspoken intermediate lives.

    OBSERVED ON HARDWARE, TWICE. Defaulting to the last fitted layer (L15 of
    16) tested next-token content, not an intermediate. Defaulting to "two
    thirds up" then discarded a converged 15-layer LFM2 artifact whose L9
    readout was ' tourism'/' located'/' geography' — the correct concept field,
    with the specific token elsewhere in the stack.

    Both defaults were the same mistake: asserting a depth. Which depth carries
    a bridge entity is a property of the model, and BR-002 forbids this project
    assuming a band it has not measured for the model in front of it.

    MUTATION CONTROL: default `scan` back to a single layer — `[fitted_layers[-1]]`
    or the old two-thirds expression — and this fails.
    """
    from unittest.mock import MagicMock, patch

    import src.workers.jlens_fit_tasks as task_mod

    seen = {}

    def fake_check_semantic(
        readout, prompt, layers, expected_intermediate, top_k=8, control_prompt=None
    ):
        seen["layers"] = layers
        seen["control_prompt"] = control_prompt
        from src.services.jlens_validation import CheckClass, CheckResult, CheckStatus

        return CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "stub")

    loaded = MagicMock()
    loaded.n_layers = 16
    service = MagicMock()
    service._load_payload.return_value = {i: None for i in range(15)}
    service.layer_scales.return_value = {}

    from src.services import jlens_readout_service, jlens_validation

    with patch.object(jlens_validation, "check_semantic", fake_check_semantic), patch.object(
        jlens_readout_service, "ReadoutService", lambda **kw: MagicMock()
    ), patch.object(
        jlens_readout_service, "JacobianTransport", lambda j, **kw: object()
    ):
        task_mod._run_semantic_check(
            service=service,
            ref=MagicMock(),
            loaded=loaded,
            probe={
                "prompt": "p",
                "expected_intermediate": " France",
                "control_prompt": "unrelated",
            },
            fitted_layers=list(range(15)),
        )

    assert seen["layers"] == list(range(15)), (
        f"probe scanned {seen['layers']}; a default that names a depth asserts "
        "a band this project has not measured"
    )
    assert seen["control_prompt"] == "unrelated", (
        "the fixture's control prompt was dropped, so the scan runs without the "
        "false-positive control that makes it more than a rubber stamp"
    )


def test_an_explicit_probe_layer_is_honoured_exactly():
    """A caller naming a layer is making a claim ABOUT that layer.

    Scanning around it would answer a question they did not ask, and would
    silently turn a failing single-layer assertion into a pass.

    MUTATION CONTROL: make the explicit branch fall through to `fitted_layers`
    and this fails.
    """
    from unittest.mock import MagicMock, patch

    import src.workers.jlens_fit_tasks as task_mod

    seen = {}

    def fake_check_semantic(
        readout, prompt, layers, expected_intermediate, top_k=8, control_prompt=None
    ):
        seen["layers"] = layers
        from src.services.jlens_validation import CheckClass, CheckResult, CheckStatus

        return CheckResult(CheckClass.SEMANTIC, CheckStatus.PASS, "stub")

    loaded = MagicMock()
    loaded.n_layers = 16
    service = MagicMock()
    service._load_payload.return_value = {i: None for i in range(15)}
    service.layer_scales.return_value = {}

    from src.services import jlens_readout_service, jlens_validation

    with patch.object(jlens_validation, "check_semantic", fake_check_semantic), patch.object(
        jlens_readout_service, "ReadoutService", lambda **kw: MagicMock()
    ), patch.object(
        jlens_readout_service, "JacobianTransport", lambda j, **kw: object()
    ):
        task_mod._run_semantic_check(
            service=service,
            ref=MagicMock(),
            loaded=loaded,
            probe={"prompt": "p", "expected_intermediate": " France", "layer": 6},
            fitted_layers=list(range(15)),
        )

    assert seen["layers"] == [6]


def test_a_failed_check_reports_what_it_actually_saw():
    """A refusal that will not say what it found cannot be acted on.

    OBSERVED ON HARDWARE: two fits failed SEMANTIC and reported only that the
    intermediate was "absent from the top-8". Learning what the lens DID
    surface required publishing the artifact the check had just refused —
    circular, and the reason the fixture could not be diagnosed.

    MUTATION CONTROL: drop `evidence` from the result dict and this fails.
    """
    from src.services.jlens_validation import (
        CheckClass,
        CheckResult,
        CheckStatus,
        ValidationReport,
    )

    report = ValidationReport(
        [
            CheckResult(
                CheckClass.SEMANTIC,
                CheckStatus.FAIL,
                "' France' absent from the top-8 at layer 9",
                {"top": [" Paris", " the", " a"]},
            )
        ]
    )
    payload = [
        {
            "check": r.check.value,
            "status": r.status.value,
            "detail": r.detail,
            "evidence": r.evidence,
        }
        for r in report.results
    ]
    assert payload[0]["evidence"]["top"] == [" Paris", " the", " a"], (
        "the check's own evidence was dropped, so a failure says only that "
        "something was absent and never what was present"
    )


def test_a_fit_that_fails_validation_KEEPS_its_staged_artifact(tmp_path):
    """The cheap half of the work must not destroy the expensive half.

    OBSERVED ON HARDWARE. A converged 15-layer LFM2 fit — 754 seconds of GPU
    time — was deleted the instant one fixture token failed to appear, leaving
    nothing to re-validate. Proving the lens was fine then required paying for
    the entire fit a second time.

    Staging is excluded from discovery, so keeping it serves nothing, and
    `write_staged` clears it before the next fit, so it cannot accumulate.

    MUTATION CONTROL: re-add `shutil.rmtree(service.staging_dir(repo_id))` to
    the non-serviceable branch of the task and this fails.
    """
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    import torch

    import src.workers.jlens_fit_tasks as task_mod
    from src.services.jlens_artifact_service import JLensArtifactService

    d_model = 8
    layers = [0, 1]

    class _Result:
        jacobians = {i: torch.eye(d_model, dtype=torch.float16) for i in layers}
        scales = {i: 1.0 for i in layers}
        prompts_seen = 42
        converged = True
        last_delta = 1e-4
        mean_seq_len = 12.0
        degenerate_layers: list = []
        position_spread_mean = 0.0
        position_spread_max = 0.0
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2

        @staticmethod
        def size_bytes():
            return 2 * 8 * 8 * 2

    class _Fitter:
        def __init__(self, *_a, **_kw):
            pass

        def fit(self, *_a, **_kw):
            return _Result()

    loaded = MagicMock()
    loaded.d_model = d_model
    loaded.n_vocab = 64
    loaded.n_layers = 4
    loaded.name = "org/model"

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        repo_id="org/model"
    )

    @contextmanager
    def fake_db():
        yield db

    root = tmp_path / "artifacts"

    with patch("src.ml.jlens_fitter.JacobianFitter", _Fitter), patch(
        "src.core.database.get_sync_db", fake_db
    ), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch.object(
        type(task_mod.settings),
        "jlens_artifacts_dir",
        property(lambda _self: root),
    ), patch.object(
        task_mod.fit_jlens_artifact, "update_state", MagicMock()
    ):
        # NO semantic probe -> SEMANTIC is NOT_RUN -> not serviceable -> the
        # branch under test.
        out = task_mod.fit_jlens_artifact.run(model_id="m_1", prompts=["a"])

    assert out["published"] is False, "precondition: this fit must not publish"

    staging = JLensArtifactService(root).staging_dir("org/model")
    assert staging.is_dir(), (
        "the staged fit was deleted after a failed validation; re-validating it "
        "now costs a full refit"
    )
    assert list(staging.glob("*_jacobian_lens.pt")), (
        f"staging survived but is empty: {list(staging.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Two defects a fully green suite published to hardware.
#
# The LFM2 artifact fitted on 2026-08-03 converged over 888 prompts and wrote a
# recipe claiming `target_layer: final` for a fit that targeted the PENULTIMATE
# block, and `attention_gradients_requested: frozen_qk` for a request that
# believed it was asking for the paper-standard full backward. Neither was
# caught by reading, by 2338 passing tests, or by the review round that
# introduced the parameter.
# ---------------------------------------------------------------------------


def test_the_recipe_records_the_target_layer_that_WAS_USED():
    """A recipe naming a target the fit did not use is worse than none (BR-007).

    `target_layer` was added to `_config_yaml`'s signature, threaded from the
    API through the task and into the call — and then dropped on the last line,
    where the value was a hardcoded literal "final". Every penultimate fit
    published provenance for a recipe it did not run.

    MUTATION CONTROL: put back `"target_layer: final",` and this fails.
    """
    from src.workers.jlens_fit_tasks import _config_yaml

    from unittest.mock import MagicMock

    class _Loaded:
        name = "org/model"
        d_model = 8
        n_vocab = 64
        n_layers = 16
        structure = MagicMock(num_layers=16)
        model = MagicMock(config=None)

    class _R:
        jacobians = {i: None for i in range(15)}
        prompts_seen = 888
        converged = True
        mean_seq_len = 68.2
        degenerate_layers: list = []
        position_spread_mean = 0.0
        position_spread_max = 0.0
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2
        scales = {i: 1.0 for i in range(15)}

    for target in ("penultimate", "final"):
        text = _config_yaml(
            _Loaded(), _R(), freeze_qk=False, corpus_name="c", target_layer=target
        )
        assert f"target_layer: {target}" in text, (
            f"asked for target_layer={target!r}; the recipe says otherwise:\n"
            + "\n".join(l for l in text.splitlines() if "target_layer" in l)
        )


def test_full_backward_is_the_DEFAULT_recipe_everywhere_a_fit_starts():
    """Frozen Q/K is an ablation. It must not be what a caller gets by default.

    The fitter class already defaulted `freeze_qk=False`, which made this look
    settled — but every real fit is started through the task, the REST schema or
    the MCP tool, and all three defaulted to True. The class default was
    unreachable, so the aligned recipe could only be obtained by naming the flag,
    and the MCP tool's own description said "OFF by default" while defaulting it
    on.

    Asserts the ENTRY POINTS, not the class: a test of the class passes against
    exactly the state that shipped the ablation to hardware.

    MUTATION CONTROL: flip any one of the three back to True and this fails.
    """
    import inspect

    from src.api.v1.endpoints.jlens import FitRequest
    from src.workers.jlens_fit_tasks import fit_jlens_artifact

    task_default = inspect.signature(
        fit_jlens_artifact.run if hasattr(fit_jlens_artifact, "run") else fit_jlens_artifact
    ).parameters["freeze_qk"].default
    assert task_default is False, f"Celery task defaults freeze_qk={task_default}"

    assert FitRequest.model_fields["freeze_qk"].default is False, (
        "the REST schema defaults to the ablation"
    )

    from src.mcp_server.tools import jlens as mcp_jlens

    src = inspect.getsource(mcp_jlens)
    marker = "recorded per layer rather than claimed wholesale\")] = "
    assert marker in src, "the MCP freeze_qk annotation moved; update this test"
    default_text = src.split(marker, 1)[1].split(",", 1)[0].strip()
    assert default_text == "False", (
        f"the MCP tool defaults freeze_qk={default_text}, so an agent asking for "
        "a standard fit silently gets the ablation"
    )


# ---------------------------------------------------------------------------
# THE FIT MUST GIVE THE CARD BACK.
#
# `load_for_readout` caches the model so a fit does not reload it per prompt.
# `clear_cache` existed to drop it and had ZERO callers — the same
# declared-but-never-wired shape this repo has shipped before. Observed: an
# LFM2 fit finished at 22:09 and 4.0 GB was still resident on the shared 3090
# at 0% utilisation half an hour later, on the card miLLM serves from.
# ---------------------------------------------------------------------------


def _fit_task_harness(tmp_path, cuda: bool, blow_up: bool = False):
    """Run `fit_jlens_artifact` with everything mocked, recording cache clears."""
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    import torch

    import src.workers.jlens_fit_tasks as task_mod
    from src.services import jlens_model_registry

    cleared = []

    class _Result:
        jacobians = {i: torch.eye(8, dtype=torch.float16) for i in (0, 1)}
        scales = {0: 1.0, 1: 1.0}
        prompts_seen = 4
        converged = True
        mean_seq_len = 10.0
        degenerate_layers: list = []
        position_spread_mean = 0.0
        position_spread_max = 0.0
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2

        @staticmethod
        def size_bytes():
            return 8 * 8 * 2 * 2

    class _Fitter:
        def __init__(self, *_a, **_kw):
            pass

        def fit(self, *_a, **_kw):
            if blow_up:
                raise RuntimeError("CUDA out of memory")
            return _Result()

    loaded = MagicMock()
    loaded.d_model, loaded.n_vocab, loaded.n_layers = 8, 64, 4
    loaded.name = "org/model"

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        repo_id="org/model"
    )

    @contextmanager
    def fake_db():
        yield db

    with patch("src.ml.jlens_fitter.JacobianFitter", _Fitter), patch(
        "src.core.database.get_sync_db", fake_db
    ), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch.object(
        jlens_model_registry, "clear_cache", lambda: cleared.append(True)
    ), patch(
        "torch.cuda.is_available", lambda: cuda
    ), patch.object(
        type(task_mod.settings),
        "jlens_artifacts_dir",
        property(lambda _s: tmp_path / "artifacts"),
    ), patch.object(
        task_mod.fit_jlens_artifact, "update_state", MagicMock()
    ):
        try:
            task_mod.fit_jlens_artifact.run(model_id="m_1", prompts=["a"])
            raised = False
        except RuntimeError:
            raised = True
    return cleared, raised


def test_a_gpu_fit_RELEASES_the_model_when_it_finishes(tmp_path):
    """Otherwise the card stays occupied until the pod restarts.

    MUTATION CONTROL: drop the `finally: clear_cache()` block and this fails.

    NOT SUFFICIENT ON ITS OWN — see the weakref test below. This asserts the
    release was CALLED, and a release that is called while the caller still
    holds the model frees nothing.
    """
    cleared, _ = _fit_task_harness(tmp_path, cuda=True)
    assert cleared, (
        "the fit finished without releasing the model; the GPU stays occupied "
        "at 0% utilisation on a card that is shared with serving"
    )


def test_the_task_has_DROPPED_the_model_by_the_time_it_releases(tmp_path):
    """Being called is not being effective.

    `clear_cache` nulls the cache entry and then runs `gc.collect()` +
    `torch.cuda.empty_cache()`. If the task's own frame still references the
    model at that moment, gc collects nothing and `empty_cache` has no free
    blocks to return — so the weights stay on the card.

    OBSERVED ON HARDWARE, through the first version of this release and its
    three passing tests: 7706 MiB fell to 2608 MiB, and 2608 MiB is LFM2's fp16
    weights. The tests could not see it because they asked whether the release
    RAN, never whether anything was freed.

    This asserts the property that actually matters: at the moment of release,
    no strong reference to the loaded model survives anywhere. A weakref is the
    only way to ask that question without a GPU.

    MUTATION CONTROL: delete the `loaded = None` line and this fails while the
    test above still passes — which is exactly the state that shipped.
    """
    import gc
    import weakref
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    import torch

    import src.workers.jlens_fit_tasks as task_mod
    from src.services import jlens_model_registry

    class _Loaded:
        """A real object, not a Mock: MagicMock children keep parents alive."""

        def __init__(self):
            self.d_model, self.n_vocab, self.n_layers = 8, 64, 4
            self.name = "org/model"
            self.model = None
            self.tokenizer = None
            self.structure = None
            self.unembedding = None

    holder = {}

    def make_loaded(*_a, **_kw):
        # Built HERE and weakly referenced, so the test itself never holds a
        # strong reference — `return_value=obj` would pin it and the assertion
        # could never fail.
        obj = _Loaded()
        holder["ref"] = weakref.ref(obj)
        return obj

    alive_at_release = {}

    def spy_clear_cache():
        gc.collect()
        ref = holder.get("ref")
        alive_at_release["alive"] = ref is not None and ref() is not None

    class _Result:
        jacobians = {i: torch.eye(8, dtype=torch.float16) for i in (0, 1)}
        scales = {0: 1.0, 1: 1.0}
        prompts_seen = 4
        converged = True
        mean_seq_len = 10.0
        degenerate_layers: list = []
        position_spread_mean = 0.0
        position_spread_max = 0.0
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2

        @staticmethod
        def size_bytes():
            return 8 * 8 * 2 * 2

    class _Fitter:
        def __init__(self, *_a, **_kw):
            pass

        def fit(self, *_a, **_kw):
            return _Result()

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        repo_id="org/model"
    )

    @contextmanager
    def fake_db():
        yield db

    with patch("src.ml.jlens_fitter.JacobianFitter", _Fitter), patch(
        "src.core.database.get_sync_db", fake_db
    ), patch(
        "src.services.jlens_model_registry.load_for_readout", side_effect=make_loaded
    ), patch.object(
        jlens_model_registry, "clear_cache", spy_clear_cache
    ), patch(
        "torch.cuda.is_available", lambda: True
    ), patch.object(
        type(task_mod.settings),
        "jlens_artifacts_dir",
        property(lambda _s: tmp_path / "artifacts"),
    ), patch.object(
        task_mod.fit_jlens_artifact, "update_state", MagicMock()
    ):
        task_mod.fit_jlens_artifact.run(model_id="m_1", prompts=["a"])

    assert "alive" in alive_at_release, "the release never ran"
    assert alive_at_release["alive"] is False, (
        "the task still held the loaded model when it released the GPU, so "
        "gc collected nothing and empty_cache returned no blocks — the weights "
        "stay resident on a card that is shared with serving"
    )


def test_a_gpu_fit_RELEASES_the_model_even_when_it_FAILS(tmp_path):
    """An OOM is exactly when the card most needs to come back.

    MUTATION CONTROL: move the release out of `finally` into the success path
    and this fails while the test above still passes — which is why both exist.
    """
    cleared, raised = _fit_task_harness(tmp_path, cuda=True, blow_up=True)
    assert raised, "precondition: this fit must fail"
    assert cleared, "a failed fit kept the GPU"


def test_a_CPU_fit_does_not_evict_the_cache(tmp_path):
    """There is no card to give back, and evicting costs the next readout a reload.

    MUTATION CONTROL: clear unconditionally and this fails.
    """
    cleared, _ = _fit_task_harness(tmp_path, cuda=False)
    assert not cleared, "a CPU fit evicted the cache for no benefit"


def test_the_progress_meta_carries_the_DENOMINATOR_and_the_THRESHOLD(tmp_path):
    """The tile's numbers come from here, and nothing else produced them.

    FOUND BY MUTATION, not by reading: deleting `total_prompts` from the meta
    left the whole suite green. Every test that touched it asserted how a
    CONSUMER renders the field — the listing, the subtitle, the banner — and
    none asserted that the PRODUCER emits it. A consumer test passes just as
    happily against a producer that sends nothing.

    `total_prompts` existed as a local used to compute the percentage and was
    then dropped, so a reader could show "53%" but not "634 / 1200".
    `convergence_delta` is the threshold the delta is racing; a delta with no
    target is a number nobody can judge.

    MUTATION CONTROL: remove either key from the `beat({...})` payload.
    """
    from contextlib import contextmanager
    from unittest.mock import MagicMock, patch

    import torch

    import src.workers.jlens_fit_tasks as task_mod

    reported = []

    class _Progress:
        prompts_seen = 634
        last_delta = 0.00103
        converged = False

    class _Result:
        jacobians = {i: torch.eye(8, dtype=torch.float16) for i in (0, 1)}
        scales = {0: 1.0, 1: 1.0}
        prompts_seen = 634
        converged = True
        mean_seq_len = 10.0
        degenerate_layers: list = []
        position_spread_mean = 0.0
        position_spread_max = 0.0
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2

        @staticmethod
        def size_bytes():
            return 8 * 8 * 2 * 2

    class _Fitter:
        convergence_delta = 1e-3
        convergence_criterion = "split_half_agreement"
        # Kept faithful to `FitResult` — see
        # test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads.
        d_model = 8
        n_layers = 2

        def __init__(self, *_a, **_kw):
            pass

        def fit(self, _prompts, layers=None, on_progress=None):
            if on_progress:
                on_progress(_Progress())
            return _Result()

    loaded = MagicMock()
    loaded.d_model, loaded.n_vocab, loaded.n_layers = 8, 64, 4
    loaded.name = "org/model"

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        repo_id="org/model"
    )

    @contextmanager
    def fake_db():
        yield db

    def capture(*_a, **kw):
        if kw.get("meta"):
            reported.append(kw["meta"])

    with patch("src.ml.jlens_fitter.JacobianFitter", _Fitter), patch(
        "src.core.database.get_sync_db", fake_db
    ), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch(
        "torch.cuda.is_available", lambda: False
    ), patch.object(
        type(task_mod.settings),
        "jlens_artifacts_dir",
        property(lambda _s: tmp_path / "artifacts"),
    ), patch.object(
        task_mod.fit_jlens_artifact, "update_state", capture
    ):
        task_mod.fit_jlens_artifact.run(
            model_id="m_1", prompts=["a"] * 1200, convergence_delta=1e-3
        )

    fitting = [m for m in reported if m.get("stage") == "fitting" and "last_delta" in m]
    assert fitting, f"no fitting progress was reported: {reported}"
    meta = fitting[-1]

    assert meta.get("total_prompts") == 1200, (
        "the progress meta carries no denominator, so a reader can show a "
        f"percentage but not '634 / 1200': {meta}"
    )
    assert meta.get("convergence_delta") == 1e-3, (
        f"the delta is reported with no threshold to judge it against: {meta}"
    )
    assert meta.get("prompts_seen") == 634


def test_the_hand_rolled_result_stubs_carry_every_field_the_writer_reads():
    """The stubs above duplicate `FitResult`'s shape by hand.

    Adding `convergence_criterion` to the real dataclass broke five tests with
    `AttributeError` at run time, because each stub is a separate hand-written
    copy — exactly the "fixture agrees by construction until it doesn't" shape.
    This makes the next field addition fail HERE, with a message naming the
    missing attribute, instead of five tests away with an AttributeError.
    """
    import ast
    import inspect as _inspect

    from src.ml.jlens_fitter import FitResult

    required = set(FitResult.__dataclass_fields__) - {"jacobians", "scales", "deltas"}

    src = _inspect.getsource(__import__(__name__, fromlist=["x"]))
    tree = ast.parse(src)
    stubs = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "_Result"
    ]
    assert stubs, "the _Result stubs moved — this guard is now vacuous"

    for stub in stubs:
        present = {
            t.id
            for node in stub.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(t, ast.Name)
        } | {
            n.name for n in stub.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = required - present
        assert not missing, (
            f"_Result stub at line {stub.lineno} is missing {sorted(missing)} — "
            f"add them, or the tests using it will fail with AttributeError"
        )
