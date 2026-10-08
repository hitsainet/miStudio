"""Dead-latent resampling, repaired (tracker item 1).

Every test drives the real routine on a real SAE. The fixtures are built so the
broken behaviours would show: activations at a resid_post-like scale (per-element
~0.03) with a non-zero b_dec, so raw and normalised directions differ; a latent
killed through its JumpReLU threshold, so a routine that leaves thresholds alone
cannot revive it; and Adam moments FILLED WITH ONES before a resample, so "zeroed"
is distinguishable from "was already zero" (a dead latent's decoder column gets
no gradient, so its moments are zero by construction otherwise).

MUTATION CONTROLS (2026-09-15, WS-LOOP; each applied alone by a runner that checked
the target occurred once, ran this file and the resume-equivalence file, restored the
bytes and verified sha256). All red:
  B1 re-init direction from the RAW input -> revives[jumprelu, skip], keeps firing,
     rows_columns_bias_and_threshold, cpu autocast (5 failed)
  B2 JumpReLU threshold not reset         -> revives[jumprelu], keeps firing, rows_..._threshold,
     cpu autocast (4 failed)
  B3 encoder-row Adam moments not reset   -> moments reset [jumprelu, standard], grads cleared (3)
  B4 decoder column not set               -> rows_columns_bias_and_threshold (1)
  B5 inputs drawn by loss, not loss^2     -> drawn_in_proportion_to_their_squared_loss (1)
  B6 encoder scale over every row         -> rows_columns_bias_and_threshold (1; the dead rows
     are zeroed first, or the two means agree by construction)
  B7 tracker counts firing steps          -> tracker, keeps firing (2)
  B8 resample_due ignores the decay window -> never_inside_the_lr_decay_window (1)
  B9 encoder bias not zeroed              -> revives[standard x2, skip, transcoder], rows_... (5)
Not controlled: the `torch.autocast(enabled=False)` inside the routine. Called from
an autocast region, only the per-row errors run in bf16; the slices written are
float32 either way, and no deterministic test separates the two draws.

MUTATION CONTROLS (2026-09-16, acceptance item 16 — the revived latent set):
  R1 `record_resample` call dropped from the loop -> the end-to-end assertion in
     test_training_resume_equivalence (the checkpoint carries the indices the loop
     resampled) fails
  R2 `resampled` key dropped from `state_dict`   -> round_trips_through_the_saved_state,
     plus the same end-to-end assertion
  R3 `record_resample` stores without cloning    -> does_not_alias_the_callers_tensor
  R4 history bound removed                       -> history_is_bounded_and_keeps_the_newest
"""

import math

import pytest
import torch

from src.ml.sparse_autoencoder import JumpReLUSAE, TopKSAE, Transcoder, create_sae
from src.services import dead_latent_resampling as R

D, LATENTS, CLUSTERS = 16, 24, 4


def _data(seed=0, n=256, scale=0.03):
    """Clustered activations at a small raw scale, so normalisation matters."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(CLUSTERS, D, generator=gen)
    labels = torch.randint(0, CLUSTERS, (n,), generator=gen)
    x = centers[labels] + 0.05 * torch.randn(n, D, generator=gen)
    return (x * scale).float(), labels


def _sae(kind="jumprelu", seed=0):
    torch.manual_seed(seed)
    if kind == "transcoder":
        return create_sae("transcoder", hidden_dim=D, latent_dim=LATENTS, l1_alpha=1e-3)
    if kind == "topk":
        return create_sae("topk", hidden_dim=D, latent_dim=LATENTS, top_k=4)
    return create_sae(kind, hidden_dim=D, latent_dim=LATENTS, l1_alpha=1e-3, sparsity_coeff=1e-3)


def _forward(model, x):
    if isinstance(model, Transcoder):
        return model(x, x, return_loss=True)
    return model(x, return_loss=True)


def _train(model, x, steps=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    for _ in range(steps):
        opt.zero_grad()
        _, _, losses = _forward(model, x)
        losses["loss"].backward()
        opt.step()
    return opt


def _centre_data_mean(model, x):
    """b_dec to the normalised mean, as the task does before training."""
    with torch.no_grad():
        if isinstance(model, Transcoder):
            model.b_enc_center.copy_(x.mean(0))
        else:
            normed, _ = model.normalize(x)
            model.decoder_bias.data.copy_(normed.mean(0))


def _kill(model, latent):
    """Make `latent` unable to fire on anything."""
    with torch.no_grad():
        if isinstance(model, JumpReLUSAE):
            model.activation.log_threshold[latent] = math.log(1e6)
        else:
            model.encoder.bias[latent] = -1e6


def _fires(model, x, latent):
    with torch.no_grad():
        _, z, _ = _forward(model, x)
    return z[:, latent] != 0


def _fill_moments_with_ones(opt):
    for state in opt.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            state[name].fill_(1.0)


RESAMPLING_KINDS = ["jumprelu", "standard_saelens", "standard_anthropic", "skip", "transcoder"]


class TestAKilledLatentRevives:
    @pytest.mark.parametrize("kind", RESAMPLING_KINDS)
    def test_it_fires_on_its_source_input_after_a_resample(self, kind):
        x, _ = _data()
        model = _sae(kind)
        _centre_data_mean(model, x)
        opt = _train(model, x)
        killed = 3
        _kill(model, killed)
        assert not _fires(model, x, killed).any(), "precondition: the latent is dead"

        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[killed] = True
        result = R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(1))

        assert result.skipped_reason is None
        assert result.latents.tolist() == [killed]
        source = x[result.source_rows]
        assert _fires(model, source, killed).all(), f"{kind}: the resampled latent does not fire on its source"

    def test_it_keeps_firing_as_training_continues_on_new_batches(self):
        x, _ = _data(seed=0)
        model = _sae("jumprelu")
        _centre_data_mean(model, x)
        opt = _train(model, x)
        killed = 5
        _kill(model, killed)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[killed] = True
        R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(2))

        tracker = R.DeadLatentTracker(LATENTS)
        tracker.steps_since_fired[killed] = 10_000
        tracker.mark_revived(torch.tensor([killed]))
        fired_batches = 0
        for seed in range(1, 11):
            batch, _ = _data(seed=0, n=64)  # same clusters, fresh draws below
            batch = batch[torch.randperm(64, generator=torch.Generator().manual_seed(seed))]
            opt.zero_grad()
            _, z, losses = model(batch, return_loss=True)
            losses["loss"].backward()
            opt.step()
            model.normalize_decoder()
            tracker.update(z)
            fired_batches += int((z[:, killed] != 0).any())
        assert fired_batches >= 5, f"the revived latent fired in only {fired_batches}/10 batches"
        assert tracker.steps_since_fired[killed] < 5


class TestTheReinitialisationIsInTheEncodersSpace:
    def test_rows_columns_bias_and_threshold(self):
        x, _ = _data()
        model = _sae("jumprelu")
        _centre_data_mean(model, x)
        opt = _train(model, x)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[[2, 7]] = True
        # Dead rows shrink in practice. Zeroing them makes the ALIVE mean differ
        # from the mean over every row, so a reference taken over all rows shows.
        with torch.no_grad():
            model.W_enc[dead] = 0.0
        alive_norm = model.W_enc.detach()[~dead].norm(dim=1).mean()
        assert not torch.isclose(alive_norm, model.W_enc.detach().norm(dim=1).mean())

        result = R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(3))

        normed, _ = model.normalize(x[result.source_rows])
        centred = normed - model.b_dec.detach()
        u = centred / centred.norm(dim=1, keepdim=True)
        rows = model.W_enc.detach()[result.latents]
        # Direction: the NORMALISED, CENTRED input — not the raw activation.
        assert torch.allclose(rows / rows.norm(dim=1, keepdim=True), u, atol=1e-5)
        # Scale: a fifth of the mean ALIVE encoder norm.
        assert torch.allclose(rows.norm(dim=1), torch.full((2,), 0.2 * float(alive_norm)), rtol=1e-4)
        # Decoder: the matching unit direction.
        assert torch.allclose(model.W_dec.detach()[:, result.latents].T, u, atol=1e-5)
        assert torch.all(model.b_enc.detach()[result.latents] == 0)
        # Threshold: half the pre-activation on the source input.
        pre = 0.2 * float(alive_norm) * centred.norm(dim=1)
        assert torch.allclose(model.activation.threshold.detach()[result.latents], 0.5 * pre, rtol=1e-4)

    def test_inputs_are_drawn_in_proportion_to_their_squared_loss(self, monkeypatch):
        model = _sae("jumprelu")
        x, _ = _data(n=2)
        monkeypatch.setattr(
            R, "_encoder_inputs_and_errors",
            lambda m, xs: (xs * 100.0, torch.tensor([1.0, 2.0])),
        )
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[0] = True
        gen = torch.Generator().manual_seed(0)
        picks = [
            int(R.resample_dead_latents(model, None, x, dead, generator=gen).source_rows[0])
            for _ in range(2000)
        ]
        share = sum(picks) / len(picks)
        # Squared: 4/5 = 0.80. Linear would be 2/3 = 0.67; uniform 0.50.
        assert abs(share - 0.8) < 0.03, share

    def test_no_more_latents_are_resampled_than_the_batch_has_rows(self):
        x, _ = _data(n=3)
        model = _sae("jumprelu")
        dead = torch.ones(LATENTS, dtype=torch.bool)
        result = R.resample_dead_latents(model, None, x, dead)
        assert result.latents.tolist() == [0, 1, 2]
        assert sorted(result.source_rows.tolist()) == [0, 1, 2]
        assert result.dead_before == LATENTS


class TestAdamMomentsAreResetOnlyForResampledSlices:
    @pytest.mark.parametrize("kind", ["jumprelu", "standard_saelens"])
    def test_resampled_slices_are_zero_and_the_rest_keep_theirs(self, kind):
        x, _ = _data()
        model = _sae(kind)
        opt = _train(model, x)
        _fill_moments_with_ones(opt)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[[1, 9]] = True

        result = R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(4))
        assert result.latents.tolist() == [1, 9]

        if kind == "jumprelu":
            slices = [(model.W_enc, 0), (model.b_enc, 0), (model.W_dec, 1), (model.activation.log_threshold, 0)]
            untouched = [model.b_dec]
        else:
            slices = [(model.encoder.weight, 0), (model.encoder.bias, 0), (model.decoder.weight, 1)]
            untouched = [model.decoder_bias]
        for param, dim in slices:
            for name in ("exp_avg", "exp_avg_sq"):
                buf = opt.state[param][name]
                picked = buf.index_select(dim, result.latents)
                assert torch.all(picked == 0), f"{name} of a resampled slice was not reset ({tuple(param.shape)})"
                keep = torch.ones(LATENTS, dtype=torch.bool)
                keep[result.latents] = False
                rest = buf.index_select(dim, torch.nonzero(keep).flatten())
                assert torch.all(rest == 1), f"{name} of a slice that was NOT resampled changed"
        for param in untouched:
            for name in ("exp_avg", "exp_avg_sq"):
                assert torch.all(opt.state[param][name] == 1)

    def test_accumulated_gradients_in_resampled_slices_are_cleared(self):
        x, _ = _data()
        model = _sae("jumprelu")
        opt = _train(model, x)
        _, _, losses = model(x, return_loss=True)
        losses["loss"].backward()  # a gradient accumulating mid-cycle
        for p in model.parameters():
            p.grad.fill_(1.0)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[4] = True
        R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(5))
        assert torch.all(model.W_enc.grad[4] == 0) and torch.all(model.W_dec.grad[:, 4] == 0)
        assert torch.all(model.W_enc.grad[3] == 1)

    def test_a_parameter_never_stepped_gets_no_empty_state(self):
        x, _ = _data()
        model = _sae("jumprelu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[0] = True
        R.resample_dead_latents(model, opt, x, dead)
        assert len(opt.state) == 0


class TestUnderMixedPrecision:
    def _amp_steps(self, model, opt, scaler, x, device_type, dtype, steps):
        for _ in range(steps):
            opt.zero_grad()
            with torch.autocast(device_type=device_type, dtype=dtype):
                _, _, losses = model(x, return_loss=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            model.normalize_decoder()

    def test_cpu_autocast_with_a_grad_scaler(self):
        x, _ = _data()
        model = _sae("jumprelu")
        _centre_data_mean(model, x)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
        scaler = torch.amp.GradScaler("cpu")
        self._amp_steps(model, opt, scaler, x, "cpu", torch.bfloat16, 5)
        killed = 6
        _kill(model, killed)
        dead = torch.zeros(LATENTS, dtype=torch.bool)
        dead[killed] = True

        # Called from inside an autocast region: the routine turns it off.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = R.resample_dead_latents(model, opt, x, dead, generator=torch.Generator().manual_seed(6))

        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert _fires(model, x[result.source_rows], killed).all()
        self._amp_steps(model, opt, scaler, x, "cpu", torch.bfloat16, 10)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert math.isfinite(scaler.get_scale())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FP16 autocast")
    def test_cuda_fp16_autocast_with_a_grad_scaler(self):
        x, _ = _data()
        x = x.cuda()
        model = _sae("jumprelu").cuda()
        _centre_data_mean(model, x)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
        scaler = torch.amp.GradScaler("cuda")
        self._amp_steps(model, opt, scaler, x, "cuda", torch.float16, 5)
        _kill(model, 6)
        dead = torch.zeros(LATENTS, dtype=torch.bool, device="cuda")
        dead[6] = True
        result = R.resample_dead_latents(model, opt, x, dead)
        assert _fires(model, x[result.source_rows.cuda()], 6).all()
        self._amp_steps(model, opt, scaler, x, "cuda", torch.float16, 10)
        assert all(torch.isfinite(p).all() for p in model.parameters())


class TestWhatIsNotResampled:
    def test_topk_is_refused_and_left_untouched(self):
        x, _ = _data()
        model = _sae("topk")
        assert isinstance(model, TopKSAE)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        dead = torch.ones(LATENTS, dtype=torch.bool)
        result = R.resample_dead_latents(model, None, x, dead)
        assert result.count == 0 and "auxiliary loss" in result.skipped_reason
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    def test_tied_weights_are_refused(self):
        model = JumpReLUSAE(d_model=D, d_sae=LATENTS, tied_weights=True)
        assert "tied" in R.resampling_unsupported_reason(model)

    def test_nothing_dead_changes_nothing(self):
        x, _ = _data()
        model = _sae("jumprelu")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = R.resample_dead_latents(model, None, x, torch.zeros(LATENTS, dtype=torch.bool))
        assert result.count == 0
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


class TestTheDeadLatentTracker:
    def test_dead_means_no_firing_for_threshold_consecutive_steps(self):
        tracker = R.DeadLatentTracker(3)
        silent = torch.tensor([[0.0, 1.0, 0.0]])
        for _ in range(4):
            tracker.update(silent)
        assert tracker.dead_mask(4).tolist() == [True, False, True]
        assert tracker.dead_mask(5).tolist() == [False, False, False]
        tracker.update(torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))  # fires on one token
        assert tracker.steps_since_fired.tolist() == [0, 1, 5]

    def test_revived_latents_restart_their_count(self):
        tracker = R.DeadLatentTracker(2)
        tracker.steps_since_fired += 7
        tracker.mark_revived(torch.tensor([1]))
        assert tracker.steps_since_fired.tolist() == [7, 0]

    def test_state_round_trips_and_refuses_another_width(self):
        tracker = R.DeadLatentTracker(4)
        tracker.steps_since_fired += torch.tensor([1, 2, 3, 4])
        restored = R.DeadLatentTracker(4)
        restored.load_state_dict(tracker.state_dict())
        assert restored.steps_since_fired.tolist() == [1, 2, 3, 4]
        with pytest.raises(ValueError):
            R.DeadLatentTracker(5).load_state_dict(tracker.state_dict())


class TestItRemembersWhichLatentsAResampleRevived:
    """Acceptance item 16 could not be measured because nothing recorded this.

    A drift measurement asks "did the latents we revived go on to fire?", which
    needs their INDICES. The loop recorded only `result.count` in a log line,
    and no checkpoint could recover the set afterwards — see
    `test_a_latent_that_only_fired_is_indistinguishable_afterwards`.
    """

    def test_it_records_the_step_and_the_latents(self):
        tracker = R.DeadLatentTracker(4)
        tracker.record_resample(7, torch.tensor([1, 3]))
        assert [(s, t.tolist()) for s, t in tracker.resampled] == [(7, [1, 3])]

    def test_a_latent_that_only_fired_is_indistinguishable_afterwards(self):
        """WHY the record exists, stated as a test.

        `update` zeroes the counter for every latent that fired this step and
        `mark_revived` zeroes it for the revived ones, so afterwards latent 0
        (fired) and latent 2 (revived) look identical. Only `resampled` tells
        them apart.
        """
        tracker = R.DeadLatentTracker(3)
        tracker.steps_since_fired += 9
        tracker.update(torch.tensor([[5.0, 0.0, 0.0]]))  # latent 0 fired
        tracker.mark_revived(torch.tensor([2]))
        tracker.record_resample(4, torch.tensor([2]))

        assert tracker.steps_since_fired.tolist() == [0, 10, 0], "precondition: both read zero"
        assert [t.tolist() for _, t in tracker.resampled] == [[2]]

    def test_an_empty_resample_records_nothing(self):
        tracker = R.DeadLatentTracker(2)
        tracker.record_resample(3, torch.empty(0, dtype=torch.long))
        assert tracker.resampled == []

    def test_the_record_does_not_alias_the_callers_tensor(self):
        tracker = R.DeadLatentTracker(3)
        latents = torch.tensor([1, 2])
        tracker.record_resample(2, latents)
        latents[0] = 0
        assert tracker.resampled[0][1].tolist() == [1, 2]

    def test_it_round_trips_through_the_saved_state(self):
        tracker = R.DeadLatentTracker(4)
        tracker.record_resample(5, torch.tensor([0, 2]))
        tracker.record_resample(10, torch.tensor([3]))
        restored = R.DeadLatentTracker(4)
        restored.load_state_dict(tracker.state_dict())
        assert [(s, t.tolist()) for s, t in restored.resampled] == [(5, [0, 2]), (10, [3])]

    def test_a_checkpoint_written_before_this_field_existed_still_loads(self):
        """Every checkpoint on disk today lacks the key; a resume must not refuse."""
        tracker = R.DeadLatentTracker(3)
        tracker.load_state_dict({"steps_since_fired": torch.tensor([1, 2, 3])})
        assert tracker.resampled == []
        assert tracker.steps_since_fired.tolist() == [1, 2, 3]

    def test_the_history_is_bounded_and_keeps_the_newest(self):
        tracker = R.DeadLatentTracker(2)
        for step in range(R.RESAMPLE_HISTORY_EVENTS + 5):
            tracker.record_resample(step, torch.tensor([0]))
        assert len(tracker.resampled) == R.RESAMPLE_HISTORY_EVENTS
        assert [s for s, _ in tracker.resampled] == list(
            range(5, R.RESAMPLE_HISTORY_EVENTS + 5)
        )


class TestWhenTheLoopResamples:
    def test_on_the_interval_after_both_warmups(self):
        kw = dict(interval=100, warmup_steps=150, sparsity_warmup_steps=250, total_steps=1000)
        assert not R.resample_due(0, **kw)
        assert not R.resample_due(200, **kw)  # inside the sparsity warmup
        assert R.resample_due(300, **kw)
        assert not R.resample_due(350, **kw)

    def test_never_inside_the_lr_decay_window(self):
        kw = dict(interval=100, total_steps=1000, lr_decay_steps=250)
        assert R.resample_due(700, **kw)
        assert not R.resample_due(800, **kw)
        assert R.resample_due(800, interval=100, total_steps=1000, lr_decay_steps=0)
