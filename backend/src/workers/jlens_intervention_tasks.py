"""
J-space interventions as a background task (BR-016..018).

WHY THIS FILE EXISTS. `jlens_intervention.py` had NO route, NO MCP tool and NO
UI — the four primitives, the control construction and the clamp were fully
implemented and unit-tested while nothing in the product could run one. The
suite was green because every test imported the module directly.

THE CONTROL IS RUN HERE, NOT OPTIONALLY (BR-018). Both arms run on the same
prompt, the same layers and the same positions, and the finding is the DIFFERENCE
— `excess_top1_over_control`, with both sides visible. The raw intervened rate is
not a finding and is never returned without the control it is meaningless
without. The control's directions are unit-norm, so the intervened direction is
scaled to unit norm too; otherwise the arms differ in magnitude and the
comparison measures that instead.

RUNG 2, AND THIS DOCSTRING ONCE SAID OTHERWISE. The first implementation applied
a primitive to a captured activation, pushed the result through the Jacobian
transport, and reported the mean displacement in lens space — rung 1, and a
number that `jlens_causal.py` shows was independent of the prompt entirely. What
runs now perturbs inside the model's own forward pass, lets it continue, and
scores the target token's RANK in the model's real output over many trials with
Wilson intervals, which is what the source paper measures. `InterventionResult`,
`control_outcome` and `excess_over_control` are all from the old shape and none
of them exist any more.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app
from .task_heartbeat import beat
from . import jlens_progress

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_intervention_tasks.run_intervention",
    bind=True,
    max_retries=0,
)
@jlens_progress.owns_its_failure
def run_intervention_task(
    self,
    model_id: str,
    prompt: str,
    primitive: str,
    layers: List[int],
    #: EXTRA PROMPTS FOR THE SAME EXPERIMENT. The paper reports a FRACTION of
    #: trials — 50 two-hop prompts, 192 swap trials — never one number from one
    #: prompt. A single trial has no interval and cannot be separated from its
    #: control; `prompt` alone is accepted and reported as n=1, which the Wilson
    #: interval will correctly render as almost no information.
    prompts: Optional[List[str]] = None,
    #: The token whose RANK is scored. Defaults to `direction_token`: steering
    #: along a direction and asking whether that token surfaces is the common
    #: case. A coordinate swap wants them different — push direction A, ask
    #: whether answer B arrives.
    target_token: Optional[str] = None,
    direction: Optional[List[float]] = None,
    direction_token: Optional[str] = None,
    strength: float = 1.0,
    k: int = 1,
    control_seed: int = 0,
    positions: Optional[List[int]] = None,
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one intervention AND its size-matched control, and report the excess.

    `control_seed` is required in practice for the same reason it is on the band
    report: "a random direction" is not a control, "k random directions from
    seed s" is, and a control nobody can reconstruct is not one.
    """
    import torch

    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..core.config import settings
    from ..services.jlens_artifact_service import JLensArtifactService
    from ..services.jlens_causal import CausalReport, Trial
    from ..services.jlens_intervention import (
        Primitive,
        apply_additive,
        apply_coordinate_swap,
        apply_projective_ablation,
        build_control,
        default_swap_layers,
    )
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_readout_service import (
        READOUT_DEVICE,
        IdentityTransport,
        JacobianTransport,
        ReadoutService,
        check_readout_budget,
    )

    try:
        chosen = Primitive(primitive)
    except ValueError as exc:
        raise ValueError(
            f"unknown primitive {primitive!r}; one of "
            f"{[p.value for p in Primitive]}"
        ) from exc

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")

        self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
        # ON THE GPU WHEN THERE IS ONE. A readout is a single forward pass and
        # stays on CPU deliberately; this runs THREE per trial across N trials,
        # which is a different order of work. Released in the `finally` below —
        # and the release drops this frame's reference FIRST, because
        # `clear_cache` runs gc and `empty_cache` immediately, and a live
        # reference here means neither frees anything.
        device = "cuda" if torch.cuda.is_available() else None
        try:
            loaded = load_for_readout(record, capture_device=device)
        except ModelNotAvailable as exc:
            raise ValueError(str(exc)) from exc

    try:
        # RESOLVED FOR PROVENANCE AND VALIDATION, not for measurement. Building it
        # runs the artifact's publish gate, so an unvalidated lens cannot be used to
        # justify an intervention; `lens_type` is then recorded with the result. The
        # measurement itself happens inside the model, not through the transport.
        if artifact_id:
            from ..api.v1.endpoints.jlens import _jacobian_transport

            transport = _jacobian_transport(loaded, artifact_id)
        else:
            transport = IdentityTransport()

        service = ReadoutService(
            model=loaded.model,
            tokenizer=loaded.tokenizer,
            structure=loaded.structure,
            unembedding=loaded.unembedding,
            model_name=loaded.name,
        )

        n_layers = int(loaded.structure.num_layers)
        out_of_range = [l for l in layers if l < 0 or l >= n_layers]
        if out_of_range:
            raise ValueError(f"layers {out_of_range} outside range 0..{n_layers - 1}")

        # THE SCALE-DERIVED BUDGET, ENFORCED WHERE THE MODEL IS. `MAX_INTERVENED_LAYERS`
        # is a flat 64 and never binds on anything this project runs, and the
        # browser's quarter-of-the-stack cap protects only clicks — an MCP agent
        # calling with every layer of a 26-layer model hooked the whole stack at
        # strength 1, which is the oversteer BR-017 v0.2 exists to prevent.
        #
        # `default_swap_layers` is the derivation, and it had no production
        # caller in EITHER language until this: the browser re-derived the same
        # quarter in TypeScript, so the rule lived in two places that could
        # disagree and in neither place that a non-browser caller reaches.
        #
        # A WARNING, NOT A REFUSAL. A deliberate whole-stack intervention is a
        # legitimate experiment; an accidental one is the common case, and the
        # difference is only knowable to the caller.
        budget = default_swap_layers(n_layers)
        if len(layers) > budget:
            logger.warning(
                "Intervening at %d of %d layers, above the scale-derived budget "
                "of %d (a quarter of the stack). Swaps and steers oversteer "
                "easily at this width on small models.",
                len(layers),
                n_layers,
                budget,
            )
            over_budget = {"requested": len(layers), "budget": budget}
        else:
            over_budget = None

        # BUDGETED AGAINST THE PROMPT THAT WILL ACTUALLY RUN. `prompt` is only
        # the trial set when `prompts` is absent — otherwise it is discarded a
        # few lines below, and checking it was checking a string this run never
        # touches. The LONGEST is the bound, because every trial pays its own
        # forward passes and one long entry costs what a long `prompt` would.
        budgeted = max(list(prompts) if prompts else [prompt], key=len)
        encoded = service.tokenizer(budgeted, return_tensors="pt")
        input_ids = encoded["input_ids"].to(service.capture_device)
        n_positions = int(input_ids.shape[-1])
        check_readout_budget(n_positions, len(layers), service.d_model)

        # NO CAPTURE PASS. The lens-space version needed the residuals to transport
        # them; this one perturbs inside the model's own forward pass and never sees
        # them, so capturing would be a full extra pass whose result is discarded.

        if direction is None and direction_token:
            # A TOKEN'S DIRECTION IS ITS UNEMBEDDING ROW. Resolved here rather than
            # in the browser: the client has neither W_U nor any way to produce a
            # d_model vector, which is why this surface had no UI.
            ids = service.tokenizer.encode(direction_token, add_special_tokens=False)
            if not ids:
                raise ValueError(
                    f"{direction_token!r} does not tokenise to anything; there is "
                    "no direction to intervene along"
                )
            if len(ids) > 1:
                # STATED, not silently truncated. A multi-token string has no single
                # direction, and taking the first piece would intervene along
                # something the caller did not name.
                raise ValueError(
                    f"{direction_token!r} is {len(ids)} tokens. A lens direction is "
                    "defined for a SINGLE token; pick one, or pass an explicit "
                    "direction vector."
                )
            named = service.W_U[ids[0]].to(READOUT_DEVICE).to(torch.float32)
        elif direction is not None:
            named = torch.tensor(direction, dtype=torch.float32, device=READOUT_DEVICE)
            if named.shape[-1] != service.d_model:
                raise ValueError(
                    f"direction has {named.shape[-1]} dimensions, model has "
                    f"{service.d_model}"
                )
        elif chosen in (Primitive.ADDITIVE, Primitive.PROJECTIVE_ABLATION):
            raise ValueError(f"{chosen.value} needs a direction to act along")
        else:
            named = None

        # UNIT NORM, SO THE CONTROL IS ACTUALLY MATCHED (BR-018).
        #
        # `build_control` returns unit-norm random directions. An unembedding row
        # does not have unit norm and the norms vary several-fold across tokens,
        # so an additive run pushed `strength * ||W_U[t]||` while its control
        # pushed `strength * 1`. The arms were then compared as though the only
        # difference between them were semantic. On a token with a large row that
        # separates the intervals on magnitude alone, and the report says
        # "against a matched-norm random control" over the top of it.
        #
        # It also makes `strength` mean one thing. Before this, a sweep at
        # 2/10/40 on ' Paris' and the same sweep on ' dog' were different
        # experiments wearing the same numbers, and neither the recipe nor the
        # UI said so.
        #
        # `projective_ablation` and `coordinate_swap` normalise internally, so
        # this changes nothing for them; it is recorded for all of them anyway
        # because the recipe has to describe what ran (BR-007).
        direction_norm: Optional[float] = None
        if named is not None:
            direction_norm = float(torch.linalg.norm(named))
            if direction_norm == 0.0:
                raise ValueError(
                    "the direction is the zero vector; it has no orientation to "
                    "intervene along"
                )
            named = named / direction_norm

        # THE TARGET IS SCORED BY ID, resolved once. A multi-token target has no
        # single rank in a next-token distribution, so it is refused rather than
        # truncated to its first piece — which would score a different token than
        # the caller named.
        wanted = target_token or direction_token
        if not wanted:
            raise ValueError(
                "a target token is required: the finding is the rank of a NAMED "
                "token in the model's output, so there is nothing to score without "
                "one. Pass target_token, or direction_token to use the same token "
                "for both."
            )
        target_ids = service.tokenizer.encode(wanted, add_special_tokens=False)
        if len(target_ids) != 1:
            raise ValueError(
                f"target {wanted!r} is {len(target_ids)} tokens; a rank in a "
                "next-token distribution is defined for a single token"
            )
        target_id = int(target_ids[0])

        trial_prompts = list(prompts) if prompts else [prompt]

        # AN UNIMPLEMENTED PRIMITIVE IS REFUSED, NOT SUBSTITUTED. The hook has
        # no branch for this one, and the old `else` quietly ran an additive
        # steer under its name. Nothing about being one enum member away from a
        # working primitive makes silently running a different experiment
        # acceptable.
        if chosen is Primitive.DYNAMIC_TOPK_ABLATION:
            raise ValueError(
                "dynamic_topk_ablation is not implemented for the forward-pass "
                "path: it needs the lens coordinates at the intervened site, "
                "which this measurement does not compute. Use additive, "
                "projective_ablation or coordinate_swap."
            )

        # A SWAP NEEDS TWO DIRECTIONS. `direction_token` is the coordinate to
        # move and `target_token` the one to exchange it with; a swap with one
        # token would be an additive push wearing the wrong name, which is the
        # defect being fixed here.
        swap_partner = None
        if chosen is Primitive.COORDINATE_SWAP:
            if not target_token or target_token == direction_token:
                raise ValueError(
                    "coordinate_swap needs TWO different tokens: "
                    "`direction_token` is the coordinate to move and "
                    "`target_token` the one to exchange it with. Supplying one "
                    "token would run an additive steer under a swap's name."
                )
            partner_ids = service.tokenizer.encode(
                target_token, add_special_tokens=False
            )
            if len(partner_ids) != 1:
                raise ValueError(
                    f"swap partner {target_token!r} is {len(partner_ids)} "
                    "tokens; a lens coordinate is defined for a single token"
                )
            swap_partner = service.W_U[partner_ids[0]].to(READOUT_DEVICE).to(
                torch.float32
            )

        # POSITIONS ARE RESOLVED PER TRIAL. They used to be computed ONCE from
        # `prompt` and applied to every trial, so "the last position" of the
        # first prompt became an absolute index into all the others. Trials
        # shorter than that index were silently skipped by a bounds guard in the
        # hook and then scored as though the intervention HAD been applied and
        # had done nothing.
        #
        # Observed on hardware: a 24-trial sweep at strengths 2, 10 and 40
        # returned byte-identical rates — 5/24 every time. Only the prompts long
        # enough to contain absolute position 8 were ever perturbed, and those
        # saturated at the lowest strength, so a 20x change in strength moved
        # nothing. The identical numbers were the tell.
        def _sites_for(n_tokens: int):
            if positions is None:
                # THIS prompt's last token, not the first prompt's.
                return [n_tokens - 1]
            return [q if q >= 0 else n_tokens + q for q in positions]

        if positions is not None:
            # VALIDATED AGAINST EVERY TRIAL, up front. An explicit position that
            # does not exist in some prompts cannot be the experiment the caller
            # asked for, and running a mixture of perturbed and unperturbed
            # trials under one label is worse than refusing.
            impossible = []
            for text in trial_prompts:
                n_t = int(
                    service.tokenizer(text, return_tensors="pt")["input_ids"].shape[-1]
                )
                if any(q < 0 or q >= n_t for q in _sites_for(n_t)):
                    impossible.append((text, n_t))
            if impossible:
                shown = "; ".join(f"{t!r} has {n} tokens" for t, n in impossible[:3])
                raise ValueError(
                    f"positions {list(positions)} do not exist in "
                    f"{len(impossible)} of {len(trial_prompts)} trial prompts "
                    f"({shown}). Omit `positions` to use each prompt's last "
                    "token, or supply prompts of a consistent length."
                )

        controls = build_control(k=k, seed=control_seed, d_model=service.d_model)

        # ---------------------------------------------------------------- the pass
        # PERTURB, THEN LET THE MODEL RUN. The paper applies the primitive and
        # "allow[s] the forward pass to continue", reading the effect from the
        # model's own output. This used to stop at the lens and report the mean
        # absolute displacement of the transported activation, which measured
        # `s*J(v)` — a quantity independent of the activation, the prompt and the
        # position, because the transport is linear and `apply_additive` is
        # `h + s*v`, so `h` cancels. Two unrelated prompts returned 0.01739214 to
        # seven significant figures.
        #
        # THE HOOK TARGET IS THE WHOLE DECODER LAYER. `structure.layers_module[L]`
        # is resid_post. Hooking the discovered "residual" module instead is a
        # post-attention RMSNorm on LFM2, which renormalises the added vector away —
        # steered output came back byte-identical to unsteered. Same target the
        # serving path uses, deliberately.
        hook_layers = {}
        for L in layers:
            target_module = loaded.structure.layers_module[L]
            if target_module is None:
                raise ValueError(f"No hookable layer module for layer {L} on this model")
            hook_layers[L] = target_module

        skipped = {"n": 0}

        def _perturbing_hook(vector, at_positions, partner=None):
            def hook(_module, _inp, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                if hidden.dim() != 3:
                    return output
                with torch.no_grad():
                    v = vector.to(dtype=hidden.dtype, device=hidden.device)
                    if partner is not None:
                        partner_v = partner.to(
                            dtype=hidden.dtype, device=hidden.device
                        )
                    for pos in at_positions:
                        if pos < 0 or pos >= hidden.shape[1]:
                            # UNREACHABLE after the validation above, and counted
                            # rather than passed over: a silent `continue` here is
                            # exactly what turned "never perturbed" into "perturbed
                            # and had no effect" for 19 of 24 trials.
                            skipped["n"] += 1
                            continue
                        h = hidden[0, pos]
                        # EXHAUSTIVE, WITH NO `else`. This used to be two
                        # branches — projective ablation, and everything else is
                        # additive — so a request for `coordinate_swap` ran an
                        # ADDITIVE steer and the result was labelled
                        # `coordinate_swap` in its `steering_recipe`. That recipe
                        # is written into `interventions.json`, which is built to
                        # travel with the lens, so the mislabelling would have
                        # become false provenance downstream.
                        if chosen is Primitive.PROJECTIVE_ABLATION:
                            hidden[0, pos] = apply_projective_ablation(h, v)
                        elif chosen is Primitive.COORDINATE_SWAP:
                            hidden[0, pos] = apply_coordinate_swap(h, v, partner_v)
                        else:
                            hidden[0, pos] = apply_additive(h, v, strength)
                return output
            return hook

        def final_rank(text, vector, top_k=50):
            """Rank of the target token in the model's REAL next-token distribution.

            `None` when it falls outside `top_k` — distinct from a large rank, so a
            search cutoff is never reported as a measurement.
            """
            ids = service.tokenizer(text, return_tensors="pt")["input_ids"].to(
                loaded.model.device
            )
            n = int(ids.shape[-1])
            sites = _sites_for(n)
            handles = []
            if vector is not None:
                hook = _perturbing_hook(vector, sites, partner=swap_partner)
                for L in layers:
                    handles.append(hook_layers[L].register_forward_hook(hook))
            try:
                with torch.no_grad():
                    logits = loaded.model(input_ids=ids).logits[0, -1]
            finally:
                for handle in handles:
                    handle.remove()
            order = torch.topk(logits.float(), k=min(top_k, int(logits.shape[-1]))).indices
            hit = (order == target_id).nonzero()
            return int(hit[0, 0]) if hit.numel() else None

        # ------------------------------------------------------------- the trials
        trials = []
        for i, text in enumerate(trial_prompts):
            self.update_state(
                state="PROGRESS",
                meta=beat({"stage": "trials", "trial": i + 1, "of": len(trial_prompts)}),
            )
            jlens_progress.update_row(
                self.request.id,
                status="running",
                progress=100.0 * i / max(len(trial_prompts), 1),
            )
            # ONE CONTROL DIRECTION PER TRIAL, rotating through the seeded set. Over
            # N trials this samples N control directions rather than reusing k, and
            # keeps the cost at three forward passes per trial instead of k + 2.
            control_vector = controls[i % max(k, 1)].to(torch.float32)
            trials.append(
                Trial(
                    prompt=text,
                    baseline_rank=final_rank(text, None),
                    intervened_rank=final_rank(text, named),
                    control_rank=final_rank(text, control_vector),
                )
            )

        summary = CausalReport(
            trials=trials,
            target_token=target_token or direction_token or "<vector>",
            primitive=chosen.value,
            layers=list(layers),
            # Same rule as the recipe below: the evidence block and the recipe
            # must not disagree about what was applied.
            strength=strength if chosen is Primitive.ADDITIVE else None,
        ).summary()

        # ---------------------------------------------- record it beside the lens
        # SO IT TRAVELS. A lens is consumed by mounting its directory: published
        # to HuggingFace and pulled down by a serving runtime, it arrives as
        # files and nothing else. Evidence living only in this task result would
        # not make that journey, and the consumer would have a dictionary it can
        # read with no idea which directions actually move the model.
        #
        # ONLY WHEN AN ARTIFACT WAS ACTUALLY USED. An intervention run without
        # `artifact_id` steered along a raw unembedding direction and has
        # nothing to do with any lens, so attaching its result to one would
        # credit the lens for a finding it played no part in.
        if artifact_id:
            record = {
                # WHAT A CONSUMER APPLIES. Scores alone say something worked
                # without saying what to do; this block is the experiment,
                # restated as instructions.
                "steering_recipe": {
                    "primitive": chosen.value,
                    "direction_token": direction_token,
                    "target_token": wanted,
                    "layers": list(layers),
                    "positions": (
                        list(positions) if positions is not None else "last-per-prompt"
                    ),
                    # STRENGTH ONLY WHERE STRENGTH DID SOMETHING. The hook passes
                    # it to `apply_additive` alone; an ablation and a swap ignore
                    # it entirely. Recording 40.0 on an ablation invites a
                    # consumer to apply one at 40, and made two bit-identical
                    # runs look like two experiments to `_recipe_key`.
                    "strength": (
                        strength if chosen is Primitive.ADDITIVE else None
                    ),
                    # WHAT `strength` IS MEASURED IN. The direction is unit-norm
                    # at the point of use, so strength is an absolute
                    # displacement and comparable across tokens. Its absence in a
                    # record marks the older convention, where strength was
                    # scaled by the unembedding row's own norm and was NOT
                    # comparable across tokens or matched to the control.
                    "direction_scaling": "unit",
                    "direction_norm_before_scaling": direction_norm,
                    "hook_target": "layers_module[L] (resid_post)",
                    # RECORDED WHEN IT APPLIES. A consumer reading a recipe that
                    # hooks most of the stack should be able to see that it was
                    # above the scale-derived budget without recomputing it.
                    "over_layer_budget": over_budget,
                    # THE TRIAL SET IS PART OF THE EXPERIMENT. Without it, a
                    # 50-prompt run and a one-click one-prompt run on the same
                    # direction and layers were the same key, and the click
                    # DELETED the 50-prompt record from the file whose whole
                    # purpose is carrying evidence off this machine. A digest
                    # rather than the prompts themselves: the record travels,
                    # and a user's prompt text is not something to publish to
                    # HuggingFace by default.
                    "n_trials": len(trials),
                    "prompts_sha256": hashlib.sha256(
                        "\u0000".join(trial_prompts).encode("utf-8")
                    ).hexdigest(),
                },
                "evidence": summary,
                "evidence_rung": 2,
                "n_trials": len(trials),
                "control": {
                    "k": k,
                    "seed": control_seed,
                    # MATCHED BECAUSE BOTH ARE UNIT (BR-018), not because the
                    # word "matched" appears next to it.
                    "construction": "gaussian_unit_norm",
                    "norm_matched_to_intervention": True,
                },
            }
            try:
                service = JLensArtifactService(settings.jlens_artifacts_dir)
                service.record_intervention_result(loaded.name, record)
            except Exception as exc:  # noqa: BLE001 - narration must not fail the run
                # The measurement succeeded; only its filing did. Losing the
                # result because a directory was read-only would be worse than
                # a warning nobody has to act on immediately.
                logger.warning("Could not record the intervention result: %s", exc)

        jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
        return {
            "model": loaded.name,
            "primitive": chosen.value,
            "parameters": {
                # THE THIRD SITE. The evidence block and the recipe were both
                # nulled for primitives that ignore strength, under a comment
                # saying the two "must not disagree about what was applied" —
                # and this one was left reporting the nominal value, so a swap
                # at strength 40 returned `parameters.strength: 40.0` beside
                # `strength: null` in the same payload.
                "strength": strength if chosen is Primitive.ADDITIVE else None,
                "requested_strength": strength,
                "k": k,
                "artifact_id": artifact_id,
                "lens_type": transport.lens_type,
                "positions": (
                    list(positions) if positions is not None else "last-per-prompt"
                ),
                "over_layer_budget": over_budget,
            },
            "control": {
                "k": k,
                "seed": control_seed,
                "construction": "gaussian_unit_norm",
                "matched": "one direction per trial, rotating through the seeded set",
            },
            **summary,
        # MUST BE ZERO. Non-zero means some trials were scored without the
        # perturbation ever being applied, and their "no effect" is an artefact
        # of the harness rather than a measurement of the model.
        "positions_skipped": skipped["n"],
            # RUNG 2. The perturbation is applied to the residual stream and the
            # model is RUN — this measures the model's behaviour, not the lens's
            # geometry. It remains one model, one direction and one prompt set: it
            # is evidence that this coordinate MOVES this model here, not that it is
            # the only direction that would.
            "evidence_rung": 2,
            "method": (
                "Perturb the residual at the named layers and positions, continue "
                "the forward pass, and score the rank of the target token in the "
                "model's own next-token distribution. Reported as top-1 and top-5 "
                "rates with Wilson 95% intervals, against a matched-norm random "
                "control run on the same prompts."
            ),
            # THE GENERAL CAVEAT, AND THEN THE SPECIFIC ONE IF THERE IS ONE.
            #
            # This key sits AFTER `**summary` in the same dict, so a literal
            # here silently overwrote whatever `summary()` derived. A 1-trial
            # run therefore returned the generic "overlapping intervals mean no
            # effect was demonstrated here" — precisely the reading the
            # sample-size caveat exists to prevent — while the derived text
            # survived only in the on-disk record, and only when an artifact_id
            # was supplied.
            "caveat": " ".join(
                filter(
                    None,
                    [
                        "The FINDING is the separation between the intervened "
                        "and control rates, not the intervened rate alone. "
                        "Overlapping intervals mean no effect was demonstrated "
                        "here — never that none exists. A baseline rate near "
                        "the intervened rate means the prompts were already "
                        "answering that way and the intervention moved nothing.",
                        summary.get("caveat"),
                    ],
                )
            ),
        }

    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # DROP EVERY REFERENCE, NOT JUST `loaded`. `clear_cache` nulls the
            # cache entry then runs gc + `empty_cache()`, so anything still
            # holding the model keeps every block allocated.
            #
            # `loaded = None` alone was NOT enough here and the hardware said so:
            # 2570 MiB stayed resident after a 3-strength sweep. `ReadoutService`
            # is constructed with `model=loaded.model` and holds its own strong
            # reference, entirely independent of the name `loaded`; `hook_layers`
            # holds decoder modules, which are part of the same graph.
            #
            # The fit task's version works because it hands `loaded` to a helper
            # and keeps nothing else. This one builds a service, so it has more
            # to put down. Rebinding a closed-over name updates the cell, so
            # `final_rank` and the hook let go with it.
            loaded = None  # noqa: F841 - the assignment IS the release
            service = None  # noqa: F841 - holds model=loaded.model independently
            hook_layers = None  # noqa: F841 - holds the decoder modules
            transport = None  # noqa: F841
            clear_cache()
            logger.info("Released the intervention model from GPU memory")
