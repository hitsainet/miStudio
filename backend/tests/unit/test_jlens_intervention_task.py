"""
The intervention task, measured the way the source paper measures.

WHAT CHANGED AND WHY. The first version applied a primitive to a captured
activation, pushed it through the Jacobian transport, and reported the mean
absolute displacement in lens space. The paper instead perturbs and "allow[s]
the forward pass to continue", scoring "the fraction of trials in which the swap
places the target-appropriate answer at the top of the model's output
distribution", with "Wilson 95% CIs". No activation-space norm appears as an
effect size anywhere in it.

The deviation was not cosmetic. The transport is linear and `apply_additive` is
`h + s*v`, so `J(h + s*v) - J(h) = s*J(v)` and the activation cancels — the
reported number could not depend on the prompt, the position, or the forward
pass that produced the activation. Confirmed on hardware: "My favorite pet is a"
and "The capital of France is" both returned 0.01739214 to seven significant
figures, while the result advertised `positions: [5]` as though it mattered.

MUTATION CONTROLS:
  * read the logits without running the layers -> "the hook fires" fails
  * hook a norm module instead of the layer    -> "the hook fires" fails
  * score a fixed prompt                       -> "depends on the prompt" fails
  * drop the control arm                       -> "control is run" fails
  * claim rung 3                               -> "rung is 2" fails
"""

import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

D_MODEL = 6
VOCAB = 16
#: MUST BE < D_MODEL. `W_U` here is `torch.eye(VOCAB, D_MODEL)`, whose rows past
#: D_MODEL-1 are all ZERO — a target id of 7 gave a zero direction vector, so no
#: perturbation was possible and every arm scored identically. The first version
#: of these tests could not see that, because none of them checked the
#: intervention had any effect at all.
TARGET_ID = 3


class _Tok:
    """Different prompts tokenise DIFFERENTLY.

    A stub that ignores its input makes every prompt identical, which would let
    the prompt-dependence test pass against a prompt-independent measurement —
    the exact defect this file exists to pin.
    """

    def __call__(self, text, return_tensors=None):
        ids = [(ord(c) % VOCAB) for c in text[:4]] or [0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def encode(self, s, add_special_tokens=False):
        # DISTINCT IDS. Mapping two token strings to the same id made a
        # coordinate swap a swap of a direction with itself, which the primitive
        # correctly refuses — a fixture that agreed by construction and hid the
        # behaviour under test. Both are < D_MODEL so `torch.eye(VOCAB, D_MODEL)`
        # gives them non-zero rows.
        return {"Paris": [TARGET_ID], "dog": [2]}.get(s.strip(), [7, 8])


class _Layer(torch.nn.Module):
    """A real Module: `register_forward_hook` is what is under test."""

    def forward(self, x):
        return x


class _Model:
    """Embeds ids, RUNS the layers, and unembeds. Hooks fire because the layers
    are called, which is the property the paper's method depends on."""

    device = "cpu"

    def __init__(self, layers):
        self.layers = layers
        self.calls = 0
        torch.manual_seed(0)
        self.embed = torch.randn(VOCAB, D_MODEL)
        self.w_u = torch.randn(VOCAB, D_MODEL)

    def __call__(self, input_ids=None):
        self.calls += 1
        h = self.embed[input_ids]
        for layer in self.layers:
            h = layer(h)
        return types.SimpleNamespace(logits=h @ self.w_u.T)


def _service_stub(tok):
    svc = MagicMock()
    svc.tokenizer = tok
    svc.capture_device = "cpu"
    svc.d_model = D_MODEL
    svc.W_U = torch.eye(VOCAB, D_MODEL)
    return svc


@contextmanager
def _patched(service, layers, model_box=None):
    model = _Model(layers)
    if model_box is not None:
        model_box["model"] = model
    loaded = types.SimpleNamespace(
        name="org/model",
        model=model,
        tokenizer=service.tokenizer,
        structure=types.SimpleNamespace(num_layers=2, layers_module=layers),
        unembedding=service.W_U,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = object()

    @contextmanager
    def fake_db():
        yield db

    from src.workers.jlens_intervention_tasks import run_intervention_task

    with patch("src.core.database.get_sync_db", fake_db), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch(
        "src.services.jlens_readout_service.ReadoutService", return_value=service
    ), patch(
        "torch.cuda.is_available", lambda: False
    ), patch.object(
        run_intervention_task, "update_state", MagicMock()
    ):
        yield


def _run(**overrides):
    from src.workers.jlens_intervention_tasks import run_intervention_task

    tok = _Tok()
    service = _service_stub(tok)
    layers = [_Layer(), _Layer()]
    kwargs = dict(
        model_id="m_1",
        prompt="hello",
        primitive="additive",
        layers=[0, 1],
        direction_token="Paris",
        strength=1.0,
        k=4,
        control_seed=11,
    )
    kwargs.update(overrides)
    model_box = {}
    with _patched(service, layers, model_box):
        return run_intervention_task.run(**kwargs), model_box["model"]


class TestTheForwardPassIsContinued:
    def test_the_perturbation_REACHES_the_running_model(self):
        """Not "a hook was registered" — the OUTPUT has to change.

        An earlier version of this test registered its own hooks on the layers
        and asserted they fired. They fire whether or not the TASK hooks
        anything, because the model runs its layers regardless, so the test
        passed against a task that never registered a hook at all. Deleting the
        registration left it green.

        The observable that cannot be faked: with the perturbation applied, the
        intervened ranks must differ from the baseline ranks. If the hook never
        reaches the model, the two arms are the same forward pass.

        MUTATION CONTROL: remove the `register_forward_hook` loop and this
        fails.
        """
        out, _model = _run(
            prompts=["abcd", "efgh", "ijkl", "mnop"],
            strength=250.0,  # large enough that a rank MUST move
        )
        assert out["baseline_top1"]["hits"] != out["intervened_top1"]["hits"] or (
            out["baseline_top5"]["hits"] != out["intervened_top5"]["hits"]
        ), (
            "the intervened arm scored identically to the baseline: the "
            "perturbation never reached the model's forward pass"
        )

    def test_the_outcome_DEPENDS_ON_THE_PROMPT(self):
        """The defect that motivated the rewrite, pinned.

        The lens-space measurement returned 0.01739214 for both "My favorite pet
        is a" and "The capital of France is" — identical to seven significant
        figures, because `h` cancels out of `J(h + s*v) - J(h)`.

        Asserted on the BASELINE arm, which depends on nothing but the prompt,
        so the test cannot be satisfied by control randomness.

        MUTATION CONTROL: score `prompt` instead of each trial's text and this
        fails — both runs then measure the same string.
        """
        a, _ = _run(prompts=["aaaa", "aaab", "aaac", "aaad"])
        b, _ = _run(prompts=["wxyz", "zyxw", "qrst", "tsrq"])
        assert (a["baseline_top1"]["hits"], a["baseline_top5"]["hits"]) != (
            b["baseline_top1"]["hits"],
            b["baseline_top5"]["hits"],
        ), (
            "two disjoint prompt sets produced identical baselines; the "
            "measurement is not reading the prompt"
        )


class TestTheControlIsRealWork:
    def test_every_trial_runs_THREE_forward_passes(self):
        """Baseline, intervened and control, all on the SAME prompt.

        Asserting `n == n_trials` is not enough: `n` counts trials, so it stays
        correct when an arm is never actually measured. Setting `control_rank`
        to None left that assertion green. The count of forward passes cannot
        be faked the same way.

        MUTATION CONTROL: drop the control arm and this fails at 2 passes/trial.
        """
        out, model = _run(prompts=["abcd", "efgh", "ijkl"])
        assert out["n_trials"] == 3
        assert model.calls == 9, (
            f"{model.calls} forward passes for 3 trials; expected 9 "
            "(baseline + intervened + control each)"
        )
        for arm in ("baseline", "intervened", "control"):
            assert out[f"{arm}_top1"]["n"] == 3

    def test_the_control_construction_is_reconstructible(self):
        """"A random direction" is not a control; "k directions from seed s" is."""
        out, _ = _run(k=4, control_seed=11)
        assert out["control"]["k"] == 4
        assert out["control"]["seed"] == 11
        assert out["control"]["construction"] == "gaussian_unit_norm"

    def test_the_finding_is_reported_as_a_SEPARATION_not_a_bare_rate(self):
        out, _ = _run(prompts=["abcd", "efgh"])
        assert "excess_top1_over_control" in out
        assert "separated_from_control" in out
        assert out["excess_top1_over_control"] == pytest.approx(
            out["intervened_top1"]["rate"] - out["control_top1"]["rate"], abs=1e-9
        )

    def test_a_multi_token_direction_is_REFUSED_not_truncated(self):
        with pytest.raises(ValueError, match="tokens"):
            _run(direction_token="two words")

    def test_a_target_that_is_not_one_token_is_refused(self):
        """A rank in a next-token distribution is defined for a single token."""
        with pytest.raises(ValueError, match="tokens"):
            _run(direction_token="Paris", target_token="two words")


class TestTheClaimIsNotOverstated:
    def test_it_reports_rung_TWO_and_says_what_that_means(self):
        """The perturbation reaches the model and the model's output is read.

        That is a real intervention — rung 2 — where the lens-space version was
        honestly rung 1. It is still one model, one direction and one prompt
        set, and the caveat says so rather than implying generality.

        MUTATION CONTROL: claim rung 3 and this fails.
        """
        out, _ = _run()
        assert out["evidence_rung"] == 2
        assert "forward pass" in out["method"]
        assert "separation" in out["caveat"].lower()
        assert "never that none exists" in out["caveat"]

    def test_the_primitive_and_its_parameters_travel_with_the_result(self):
        """A number whose recipe is unrecorded cannot be reproduced or compared."""
        out, _ = _run(strength=2.5, layers=[0])
        assert out["primitive"] == "additive"
        assert out["parameters"]["strength"] == 2.5
        assert out["layers"] == [0]
        assert out["target_token"] == "Paris"

    def test_an_unknown_primitive_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown primitive"):
            _run(primitive="wishful_thinking")


class TestTheEvidenceIsFiledWithTheLens:
    """Writing a recorder is not calling it.

    This repo shipped 16 MCP tools that were implemented, unit-tested and never
    registered. A `record_intervention_result` with no caller is the same defect:
    the sidecar exists in tests and never appears next to a real artifact.

    MUTATION CONTROLS:
      * drop the `service.record_intervention_result(...)` call -> "it is filed" fails
      * file it when artifact_id is absent                  -> "no artifact" fails
      * omit steering_recipe from the record                -> "recipe" fails
    """

    @contextmanager
    def _recorder(self):
        recorded = []

        class _Svc:
            def __init__(self, _root):
                pass

            def record_intervention_result(self, repo_id, record):
                recorded.append((repo_id, record))

        with patch(
            "src.services.jlens_artifact_service.JLensArtifactService", _Svc
        ), patch(
            "src.api.v1.endpoints.jlens._jacobian_transport",
            return_value=types.SimpleNamespace(lens_type="JACOBIAN_LENS"),
        ):
            yield recorded

    def test_a_run_against_an_artifact_FILES_its_evidence(self):
        with self._recorder() as recorded:
            _run(artifact_id="model", prompts=["abcd", "efgh"])

        assert len(recorded) == 1, (
            f"{len(recorded)} evidence records filed; the measurement ran but "
            "nothing was written beside the lens"
        )
        repo_id, record = recorded[0]
        assert repo_id == "org/model"

        # THE RECIPE, asserted by content. A record that says an effect exists
        # without saying what to apply cannot be used by a consumer.
        recipe = record["steering_recipe"]
        assert recipe["primitive"] == "additive"
        assert recipe["direction_token"] == "Paris"
        assert recipe["layers"] == [0, 1]
        assert "resid_post" in recipe["hook_target"]
        assert record["evidence_rung"] == 2
        assert record["n_trials"] == 2
        assert "separated_from_control" in record["evidence"]

    def test_a_run_WITHOUT_an_artifact_files_nothing(self):
        """A raw unembedding direction has nothing to do with any lens.

        Crediting one for a finding it played no part in would put evidence
        beside a dictionary that was never used to produce it.
        """
        with self._recorder() as recorded:
            _run(prompts=["abcd"])
        assert recorded == []

    def test_a_failure_to_FILE_does_not_lose_the_MEASUREMENT(self):
        """The expensive half must survive the cheap half failing.

        A read-only directory should cost a warning, not the forward passes.
        """

        class _Broken:
            def __init__(self, _root):
                pass

            def record_intervention_result(self, *_a, **_kw):
                raise OSError("read-only file system")

        with patch(
            "src.services.jlens_artifact_service.JLensArtifactService", _Broken
        ), patch(
            "src.api.v1.endpoints.jlens._jacobian_transport",
            return_value=types.SimpleNamespace(lens_type="JACOBIAN_LENS"),
        ):
            out, _ = _run(artifact_id="model", prompts=["abcd"])
        assert out["evidence_rung"] == 2 and out["n_trials"] == 1


class TestPositionsAreResolvedPerTrial:
    """"The last position" is a property of a PROMPT, not of the experiment.

    OBSERVED ON HARDWARE. `chosen_positions` was computed once from `prompt` and
    applied as an absolute index to every trial. A 24-trial sweep at strengths
    2, 10 and 40 returned byte-identical rates — 5/24 at all three — because only
    the prompts long enough to contain absolute position 8 were ever perturbed,
    and those saturated at the lowest strength. The other trials were skipped by
    a bounds guard in the hook and scored as though the intervention had been
    applied and had done nothing.

    A 20x change in strength moving nothing was the tell, exactly as two
    unrelated prompts returning 0.01739214 had been the tell before it.

    MUTATION CONTROLS:
      * resolve positions once from `prompt`  -> "per trial" fails
      * accept an impossible explicit position -> "refused" fails
      * stop counting skips                    -> "skips are counted" fails
    """

    def test_prompts_of_DIFFERENT_lengths_are_each_perturbed_at_their_own_end(self):
        """Short and long prompts must both be intervened on.

        With a single absolute index the short ones are silently untouched, and
        their baseline and intervened arms become identical by construction.
        """
        # "abcdefgh" is 4 tokens under _Tok (text[:4]); "ab" is 2.
        out, model = _run(prompts=["abcdefgh", "ab"], strength=250.0)
        assert out["positions_skipped"] == 0, (
            f"{out['positions_skipped']} perturbations were skipped: a trial "
            "shorter than the resolved index was scored without being perturbed"
        )
        assert model.calls == 6  # 2 trials x 3 arms

    def test_an_explicit_position_that_some_prompts_LACK_is_REFUSED(self):
        """Not silently mixed.

        Running perturbed and unperturbed trials under one label produces a rate
        that describes neither, which is what 5/24 was.
        """
        with pytest.raises(ValueError, match="do not exist in"):
            _run(prompts=["abcdefgh", "ab"], positions=[3])

    def test_an_explicit_position_valid_EVERYWHERE_is_honoured(self):
        """A caller naming a reachable position gets exactly that."""
        out, _ = _run(prompts=["abcd", "efgh"], positions=[1])
        assert out["parameters"]["positions"] == [1]
        assert out["positions_skipped"] == 0

    def test_the_default_is_recorded_as_per_prompt_not_as_a_number(self):
        """A recorded `positions: [8]` reads as "position 8 everywhere".

        That is what the artifact record said while every prompt was being
        perturbed somewhere different — or not at all.
        """
        out, _ = _run(prompts=["abcd", "efghij"])
        assert out["parameters"]["positions"] == "last-per-prompt"

    def test_skipped_perturbations_are_COUNTED_not_passed_over(self):
        """Zero is the only acceptable value, and it has to be visible.

        MUTATION CONTROL: drop `positions_skipped` from the result and this
        fails — the number that would have exposed the hardware bug in one look.
        """
        out, _ = _run(prompts=["abcd"])
        assert out["positions_skipped"] == 0


class TestTheCardComesBack:
    """`loaded = None` was not enough, and the hardware said so.

    The fit task's release works because it hands `loaded` to a helper and keeps
    nothing else. This task builds a `ReadoutService` with `model=loaded.model`,
    which holds its own strong reference entirely independent of the NAME
    `loaded` — so nulling that name freed nothing. Measured after a 3-strength
    sweep: 2570 MiB of LFM2 weights still resident, on the card serving shares.

    Asserted with a weakref, because "clear_cache was called" is what let the
    first version of this mistake ship twice.

    MUTATION CONTROLS:
      * drop `service = None`      -> "nothing holds the model" fails
      * drop `hook_layers = None`  -> "nothing holds the model" fails
      * release only on success    -> "released after a failure" fails
    """

    class _FaithfulService:
        """A real class, NOT a MagicMock — and it keeps the model like the real
        `ReadoutService` does.

        Both halves matter. `patch(..., return_value=mock)` creates a MagicMock
        that records its call arguments, and the task calls it with
        `model=loaded.model`; `_mock_call_args` then pins the model for the life
        of the test, so the weakref stays alive no matter how perfect the
        release is. That is a property of the harness, not the code, and it made
        this test fail against a correct implementation — measuring the mock
        rather than the task.

        Storing `self.model` is equally deliberate: the real ReadoutService
        does, and a stub that did not could never show that nulling `service` in
        the release is what actually frees the card.
        """

        def __init__(self, model=None, tokenizer=None, structure=None,
                     unembedding=None, model_name=None):
            self.model = model
            self.tokenizer = tokenizer
            self.capture_device = "cpu"
            self.d_model = D_MODEL
            self.W_U = torch.eye(VOCAB, D_MODEL)

    def _run_and_weigh(self, blow_up=False):
        import gc
        import weakref

        import src.workers.jlens_intervention_tasks as task_mod
        from src.services import jlens_model_registry

        tok = _Tok()
        service = _service_stub(tok)
        layers = [_Layer(), _Layer()]
        holder = {}
        alive = {}

        def make_loaded(*_a, **_kw):
            # Built HERE and only weakly referenced, so the TEST never pins it.
            model = _Model(layers)
            obj = types.SimpleNamespace(
                name="org/model",
                model=model,
                tokenizer=tok,
                structure=types.SimpleNamespace(num_layers=2, layers_module=layers),
                unembedding=service.W_U,
            )
            holder["ref"] = weakref.ref(model)
            return obj

        def spy_clear_cache():
            gc.collect()
            ref = holder.get("ref")
            alive["at_release"] = ref is not None and ref() is not None

        class _Boom(dict):
            def __getitem__(self, _k):
                raise RuntimeError("CUDA out of memory")

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = object()

        @contextmanager
        def fake_db():
            yield db

        with patch("src.core.database.get_sync_db", fake_db), patch(
            "src.services.jlens_model_registry.load_for_readout", side_effect=make_loaded
        ), patch(
            "src.services.jlens_readout_service.ReadoutService",
            self._FaithfulService,
        ), patch.object(
            jlens_model_registry, "clear_cache", spy_clear_cache
        ), patch(
            "torch.cuda.is_available", lambda: True
        ), patch.object(
            task_mod.run_intervention_task, "update_state", MagicMock()
        ):
            kwargs = dict(
                model_id="m_1",
                prompt="abcd",
                primitive="wishful" if blow_up else "additive",
                layers=[0, 1],
                direction_token="Paris",
                k=2,
                control_seed=3,
            )
            try:
                task_mod.run_intervention_task.run(**kwargs)
                raised = False
            except ValueError:
                raised = True
        return alive, raised

    def test_NOTHING_holds_the_model_when_the_card_is_released(self):
        alive, _ = self._run_and_weigh()
        assert "at_release" in alive, "the release never ran"
        assert alive["at_release"] is False, (
            "something still referenced the model at release time — most likely "
            "the ReadoutService, which is constructed with model=loaded.model "
            "and is not freed by nulling the name `loaded`"
        )

    def test_it_is_released_after_a_FAILURE_too(self):
        """An OOM is when the card most needs to come back."""
        alive, raised = self._run_and_weigh(blow_up=True)
        assert raised, "precondition: this run must fail"
        # The failure happens before the model loads, so the release may not run
        # at all — what must NOT happen is a release that leaves it held.
        assert alive.get("at_release", False) is False


class TestThePrimitiveThatRUNSIsThePrimitiveREQUESTED:
    """The hook dispatched on two branches and defaulted the rest to additive.

    So `coordinate_swap` ran an additive steer and the result carried
    `"primitive": "coordinate_swap"` in its `steering_recipe` — which since
    d88dc73 is written into `interventions.json`, the file built to travel with
    the lens to a serving runtime. A label that survives the journey while the
    experiment behind it did not is worse than no record.

    MUTATION CONTROLS:
      * restore the `else: apply_additive` fall-through -> "swap DIFFERS" fails
      * accept a swap with one token                    -> "two tokens" fails
      * let dynamic_topk_ablation fall through          -> "REFUSED" fails
    """

    def test_the_task_dispatches_to_the_SWAP_and_not_to_additive(self):
        """Which primitive actually ran, asserted by call count on both.

        An earlier version of this compared the SCORES of a swap run and a
        steer run and asserted they differed. That was not the comparison it
        claimed: a swap scores its partner token and a steer scores its own, so
        the two arms measured different targets, and with a small fake model
        both could sit at zero hits and agree for reasons having nothing to do
        with the dispatch.

        Both functions are spied and both counts are asserted. "coordinate_swap
        was called" alone would still pass if additive ran as well, which is
        precisely the fall-through being ruled out. What the swap DOES is
        pinned separately by the unit tests in test_jlens_intervention.py.

        MUTATION CONTROL: restore `else: apply_additive` and this fails on the
        additive count.
        """
        from src.services import jlens_intervention as prim

        real_swap = prim.apply_coordinate_swap
        real_add = prim.apply_additive
        calls = {"swap": 0, "additive": 0}

        def spy_swap(*a, **kw):
            calls["swap"] += 1
            return real_swap(*a, **kw)

        def spy_add(*a, **kw):
            calls["additive"] += 1
            return real_add(*a, **kw)

        with patch.object(prim, "apply_coordinate_swap", spy_swap), patch.object(
            prim, "apply_additive", spy_add
        ):
            out, _ = _run(
                primitive="coordinate_swap",
                direction_token="Paris",
                target_token="dog",
                prompts=["abcd", "efgh"],
                layers=[0, 1],
            )

        assert out["primitive"] == "coordinate_swap"
        # 2 trials x 2 layers x 1 position x TWO perturbed arms = 8.
        #
        # THE CONTROL RUNS THE SAME PRIMITIVE. A control that performs a
        # different operation is not size-matched to anything — for a swap it
        # must also be a swap, exchanging a RANDOM direction's coordinate with
        # the same partner, so the only difference between the arms is which
        # direction was chosen. The baseline arm passes no vector and registers
        # no hook, which is why this is 8 and not 12.
        assert calls["swap"] == 8, (
            f"the swap ran {calls['swap']} times; expected one per "
            "(trial, layer) on the intervened AND control arms"
        )
        assert calls["additive"] == 0, (
            f"apply_additive ran {calls['additive']} times during a swap: the "
            "hook is still falling through"
        )

    def test_a_STEER_dispatches_to_additive_and_not_to_the_swap(self):
        """The mirror image, so neither branch can absorb the other."""
        from src.services import jlens_intervention as prim

        real_swap = prim.apply_coordinate_swap
        real_add = prim.apply_additive
        calls = {"swap": 0, "additive": 0}

        def spy_swap(*a, **kw):
            calls["swap"] += 1
            return real_swap(*a, **kw)

        def spy_add(*a, **kw):
            calls["additive"] += 1
            return real_add(*a, **kw)

        with patch.object(prim, "apply_coordinate_swap", spy_swap), patch.object(
            prim, "apply_additive", spy_add
        ):
            _run(primitive="additive", direction_token="Paris",
                 prompts=["abcd", "efgh"], layers=[0, 1])

        assert calls["additive"] == 8 and calls["swap"] == 0

    def test_a_swap_with_ONE_token_is_refused(self):
        """Two coordinates or it is not an exchange."""
        with pytest.raises(ValueError, match="TWO different tokens"):
            _run(primitive="coordinate_swap", direction_token="Paris")

    def test_a_swap_with_the_SAME_token_twice_is_refused(self):
        with pytest.raises(ValueError, match="TWO different tokens"):
            _run(
                primitive="coordinate_swap",
                direction_token="Paris",
                target_token="Paris",
            )

    def test_an_UNIMPLEMENTED_primitive_is_REFUSED_not_substituted(self):
        """Being one enum member from a working primitive is not a licence.

        MUTATION CONTROL: delete the refusal and this fails — the run would
        succeed, reporting an additive result labelled dynamic_topk_ablation.
        """
        with pytest.raises(ValueError, match="not implemented"):
            _run(primitive="dynamic_topk_ablation", direction_token="Paris")

class TestTheDirectionIsScaledSoTheControlIsActuallyMatched:
    """BR-018 says matched-norm. It was not, and no test could see it.

    `build_control` returns UNIT-norm random directions. The intervened arm used
    a raw unembedding row, whose norm varies several-fold across tokens on a
    real model. So an additive run pushed `strength * ||W_U[t]||` against a
    control pushing `strength * 1`, and the two were compared as though the only
    difference between them were semantic — under a report reading "against a
    matched-norm random control".

    THE EXISTING FIXTURE COULD NOT DETECT IT. `W_U` is `torch.eye(VOCAB,
    D_MODEL)`, every row of which ALREADY has unit norm, so normalising and not
    normalising are the same operation on it. Deleting the scaling left all 63
    tests green. This file therefore builds its own W_U with rows of DIFFERENT,
    non-unit norms — the only shape on which the two behaviours differ.

    MUTATION CONTROLS:
      * `named = named` instead of `named / direction_norm` -> "displacement is
        the strength" fails, and "records what it divided by" fails
      * normalise but do not record the original norm        -> "records" fails
      * scale the CONTROL up to match instead                -> "both arms move
        the same distance" fails
    """

    #: The norm of the row for `direction_token="Paris"`, which `_Tok.encode`
    #: maps to TARGET_ID. Not 1, and not shared with any other row, so nothing
    #: can pass by coincidence. Asserted in `_service` rather than trusted.
    NAMED_NORM = float(TARGET_ID + 1)

    @classmethod
    def _service(cls):
        tok = _Tok()
        svc = MagicMock()
        svc.tokenizer = tok
        svc.capture_device = "cpu"
        svc.d_model = D_MODEL
        # ROWS OF DIFFERENT NORMS. Row i has norm (i + 1), so the target row's
        # is NAMED_NORM and no two rows agree.
        rows = torch.eye(VOCAB, D_MODEL)
        for i in range(VOCAB):
            rows[i] = rows[i] * (i + 1)
        assert abs(float(torch.linalg.norm(rows[TARGET_ID])) - cls.NAMED_NORM) < 1e-6
        svc.W_U = rows
        return svc

    @classmethod
    def _run(cls, **overrides):
        from src.workers.jlens_intervention_tasks import run_intervention_task

        service = cls._service()
        layers = [_Layer(), _Layer()]
        kwargs = dict(
            model_id="m_1",
            prompt="hello",
            primitive="additive",
            layers=[0],
            direction_token="Paris",
            strength=1.0,
            k=4,
            control_seed=11,
        )
        kwargs.update(overrides)
        with _patched(service, layers):
            return run_intervention_task.run(**kwargs)

    @staticmethod
    def _displacements(monkeyable):
        """Every vector actually handed to `apply_additive`, with its strength."""
        seen = []
        import src.workers.jlens_intervention_tasks as mod
        from src.services import jlens_intervention as prim

        real = prim.apply_additive

        def spy(activation, direction, strength):
            seen.append(float(torch.linalg.norm(direction * strength)))
            return real(activation, direction, strength)

        return seen, spy

    def test_the_DISPLACEMENT_is_the_strength_not_the_row_norm(self):
        """At strength 1 the residual moves by 1, whatever the token's row norm.

        Without the scaling it moves by 5 for this token and by something else
        for the next one, so a strength sweep on two tokens is two different
        experiments wearing the same numbers.
        """
        from src.services import jlens_intervention as prim

        seen, spy = self._displacements(None)
        with patch.object(prim, "apply_additive", spy):
            self._run(strength=1.0)

        assert seen, "apply_additive was never called; the fixture is not exercising it"
        # EVERY call, not the first: the intervened arm and the control arm both
        # go through this, and the point is that they agree.
        for magnitude in seen:
            assert abs(magnitude - 1.0) < 1e-5, seen

    def test_BOTH_ARMS_move_the_same_distance(self):
        """The definition of a matched-norm control, asserted directly."""
        from src.services import jlens_intervention as prim

        seen, spy = self._displacements(None)
        with patch.object(prim, "apply_additive", spy):
            self._run(strength=2.5)

        assert len(seen) >= 2, seen
        assert max(seen) - min(seen) < 1e-5, seen
        assert abs(seen[0] - 2.5) < 1e-5, seen

    def test_it_RECORDS_what_it_divided_by(self):
        """So an older record is identifiable, and a run is reproducible.

        Scaling silently changes what `strength` means. A record carrying no
        `direction_scaling` was produced under the old convention, where the
        number was multiplied by a row norm nobody wrote down.
        """
        service = self._service()
        layers = [_Layer(), _Layer()]
        recorded = {}

        from src.workers.jlens_intervention_tasks import run_intervention_task

        class _Svc:
            def __init__(self, *a, **k):
                pass

            def record_intervention_result(self, name, record):
                recorded.update(record)

        with _patched(service, layers), patch(
            "src.services.jlens_artifact_service.JLensArtifactService", _Svc
        ), patch(
            "src.api.v1.endpoints.jlens._jacobian_transport",
            return_value=types.SimpleNamespace(lens_type="JACOBIAN_LENS"),
        ):
            run_intervention_task.run(
                model_id="m_1",
                prompt="hello",
                primitive="additive",
                layers=[0],
                direction_token="Paris",
                strength=1.0,
                k=4,
                control_seed=11,
                # The slug for "org/model", which is what `_patched` loads.
                artifact_id="model",
            )

        recipe = recorded.get("steering_recipe", {})
        assert recipe.get("direction_scaling") == "unit", recipe
        assert (
            abs(recipe.get("direction_norm_before_scaling", 0) - self.NAMED_NORM) < 1e-5
        ), recipe
        # AND THE CLAIM BESIDE THE CONTROL IS EARNED, not decorative.
        assert recorded.get("control", {}).get("norm_matched_to_intervention") is True

    def test_a_ZERO_direction_is_refused_rather_than_dividing_by_zero(self):
        """Row 0 of the fixture is scaled by 1 and is fine; a genuinely zero
        row would otherwise produce NaN and score three arms of noise."""
        service = self._service()
        service.W_U = torch.zeros(VOCAB, D_MODEL)
        layers = [_Layer(), _Layer()]

        from src.workers.jlens_intervention_tasks import run_intervention_task

        with _patched(service, layers):
            with pytest.raises(ValueError, match="zero vector"):
                run_intervention_task.run(
                    model_id="m_1",
                    prompt="hello",
                    primitive="additive",
                    layers=[0],
                    direction_token="Paris",
                    strength=1.0,
                    k=4,
                    control_seed=11,
                )

class TestTheReturnedPayloadDoesNotCONTRADICTItself:
    """Three sites reported the same facts and two of them disagreed.

    MUTATION CONTROLS:
      * put the literal caveat back after `**summary` -> "carries the sample-size
        caveat" fails
      * report the nominal strength in `parameters`   -> "agrees about strength"
        fails
    """

    def test_it_carries_the_SAMPLE_SIZE_caveat_not_only_the_generic_one(self):
        """The literal sat AFTER `**summary` in the same dict.

        So `summary()`'s derived text — "the verdict describes the sample and
        not the intervention. Add prompts." — was silently overwritten, and a
        one-trial run returned the generic "overlapping intervals mean no effect
        was demonstrated here": exactly the reading the sample-size caveat was
        added to prevent. It survived only in the on-disk record, and only when
        an artifact_id was supplied.
        """
        out, _model = _run(prompt="hello")
        assert out["n_trials"] == 1, "this test needs the single-trial case"
        assert out["separation_attainable"] is False
        assert "not attainable" in out["caveat"], out["caveat"]
        # AND THE GENERAL ONE SURVIVES TOO — they are both true and the
        # specific one does not replace the standing warning.
        assert "never that none exists" in out["caveat"]

    def test_an_ATTAINABLE_run_carries_only_the_general_caveat(self):
        out, _model = _run(prompts=["abcd", "efgh", "ijkl", "mnop"])
        assert out["n_trials"] == 4
        assert out["separation_attainable"] is True
        assert "not attainable" not in out["caveat"]
        assert "never that none exists" in out["caveat"]

    def test_the_payload_AGREES_WITH_ITSELF_about_strength(self):
        """`parameters.strength` was the third site and was left behind.

        The evidence block and the recipe were both nulled for primitives that
        ignore strength, under a comment saying the two "must not disagree about
        what was applied". An agent reading `parameters.strength == 40.0` beside
        `strength: null` in the same payload either reports 40 as applied or
        treats the response as corrupt.
        """
        out, _model = _run(
            primitive="coordinate_swap",
            direction_token="Paris",
            target_token="dog",
            strength=40.0,
        )
        assert out["primitive"] == "coordinate_swap"
        assert out["strength"] is None
        assert out["parameters"]["strength"] is None, out["parameters"]
        # WHAT WAS ASKED FOR IS STILL VISIBLE, so the caller can see their own
        # request was ignored rather than silently dropped.
        assert out["parameters"]["requested_strength"] == 40.0

    def test_an_ADDITIVE_run_still_reports_the_strength_it_used(self):
        """The nulling must not swallow the case where strength is real."""
        out, _model = _run(primitive="additive", strength=2.5)
        assert out["strength"] == 2.5
        assert out["parameters"]["strength"] == 2.5


class TestTheLayerBudgetIsEnforcedWhereTheMODELIs:
    """The browser capped clicks; nothing capped an MCP agent.

    `MAX_INTERVENED_LAYERS` is a flat 64 and never binds on any model this
    project runs, and the quarter-of-the-stack rule lived only in TypeScript —
    so `default_swap_layers` had no production caller in either language while
    two places re-derived it.

    MUTATION CONTROL: drop the `default_swap_layers` check and "flags" fails.
    """

    def test_a_WHOLE_STACK_intervention_is_flagged(self):
        """Both layers of a 2-layer stack; the budget is max(1, 2 // 4) = 1."""
        out, _model = _run(layers=[0, 1])
        flag = out["parameters"]["over_layer_budget"]
        assert flag == {"requested": 2, "budget": 1}, flag

    def test_a_WITHIN_BUDGET_intervention_is_not(self):
        """Otherwise the flag would be on every run and mean nothing."""
        out, _model = _run(layers=[1])
        assert out["parameters"]["over_layer_budget"] is None

    def test_it_WARNS_rather_than_refusing(self):
        """A deliberate whole-stack intervention is a legitimate experiment.

        Only the caller knows which it is, so the run completes and says so.
        """
        out, _model = _run(layers=[0, 1])
        assert out["evidence_rung"] == 2
        assert out["n_trials"] >= 1

