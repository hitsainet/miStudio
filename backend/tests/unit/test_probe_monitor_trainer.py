"""Layer selection, rule training, SAE feature choice and calibration.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M116  `select_layers` ranks by TRAINING AUROC          → the leak test fails
  M117  standardisation is fitted on train+val together  → the scale-leak test fails
  M118  `select_layers` returns only the argmax          → the full-grid test fails
  M119  early stopping switches to validation LOSS       → the AUROC-stopping test fails
  M120  the LAST state is returned instead of the BEST   → the restore test fails
  M121  the attention query is initialised to zeros      → the attention test fails
  M122  `select_sae_features` ranks on train+val         → the structural-leak test fails
  M123  `calibrate` reports the TARGET as realised       → the honest-FPR test fails
  M124  `calibrate` rounds the budget UP                 → the overspend test fails
  M125  `calibrate` returns `inf` for fire-on-nothing    → the serialisable test fails
  M126  `_standardisation` floors a degenerate std at eps → the amplification test fails

⚠ THE FIRST CONFIGURATION OF THIS TRAINER DID NOT TRAIN, AND IT REPORTED A NUMBER.
60 full-batch steps at lr 1e-3 from a zero init reached **0.51 AUROC on linearly
separable data** — the loop ran, early-stopped, and produced a probe that reads as
"this concept is undetectable in this model". That is the worst failure mode a trainer
has, because the output is indistinguishable from a true negative result.
`TestItActuallyLearns` exists so a configuration that cannot converge fails the suite
rather than shipping.
"""
import numpy as np
import pytest
import torch

from src.services.probe_monitor_capture import PooledCapture
from src.services.probe_monitor_trainer import (
    POOLINGS,
    calibrate,
    head_from_trained,
    select_layers,
    select_sae_features,
    train_rule,
)

D = 16


def _pooled(n=200, informative_layer=2, layers=4, seed=0):
    """A pooled capture where exactly one layer carries the signal."""
    rng = np.random.default_rng(seed)
    labels = [i % 2 for i in range(n)]
    y = np.asarray(labels)
    mean, last = {}, {}
    for layer in range(layers):
        base = rng.normal(size=(n, D)).astype(np.float32)
        if layer == informative_layer:
            base[:, 0] += y * 3.0
        mean[layer] = torch.tensor(base)
        last[layer] = torch.tensor(base + rng.normal(scale=0.1, size=(n, D)).astype(np.float32))
    return PooledCapture(mean=mean, last=last), labels


def _token_rows(n=200, signal=2.0, seed=0, constant_channel=False):
    """Variable-length token blocks with the signal in channel 0."""
    rng = np.random.default_rng(seed)
    labels = [i % 2 for i in range(n)]
    rows = []
    for i in range(n):
        t = int(rng.integers(3, 8))
        block = rng.normal(size=(t, D)).astype(np.float32)
        block[:, 0] += labels[i] * signal
        if constant_channel:
            block[:, 1] = 7.0          # zero variance in training
        rows.append(block)
    return rows, labels


class TestLayerSelectionScoresOnHeldOutData:
    def test_it_finds_the_informative_layer(self):
        pooled, labels = _pooled(informative_layer=2)
        selection = select_layers(pooled, labels, range(160), range(160, 200), top_n=1)
        assert selection.chosen == [2]

    def test_the_FULL_grid_is_returned_not_only_the_winner(self):
        """A sweep that records only its argmax cannot be audited for a near-tie, and a
        near-tie is exactly when the chosen layer is arbitrary."""
        pooled, labels = _pooled(layers=4)
        selection = select_layers(pooled, labels, range(160), range(160, 200))
        assert len(selection.grid) == 4 * len(POOLINGS)
        assert {entry.pooling for entry in selection.grid} == set(POOLINGS)

    def test_the_MARGIN_over_the_runner_up_is_reported(self):
        pooled, labels = _pooled(informative_layer=1)
        selection = select_layers(pooled, labels, range(160), range(160, 200), top_n=1)
        assert selection.margin is not None and selection.margin > 0

    def test_it_ranks_by_VALIDATION_not_training_auroc(self):
        """⚠ A layer chosen by its TRAINING AUROC is whichever layer MEMORISES best.

        ⚠ AND MY FIRST FIXTURE FOR THIS COULD NOT TELL THE TWO APART. It gave layer 0 a
        signal only in the training rows and layer 3 a signal only in the validation
        rows — but the regression is FITTED on the training rows, so layer 3's model
        learns nothing and scores ~0.5 on validation, exactly like layer 0's. Both cells
        landed at chance and the assertion failed on a tie-break, not on a leak.

        The fixture that does discriminate: layer 0 is MEMORISABLE — a signal in the
        training rows only, in enough dimensions that a d-dimensional model can fit it —
        so its TRAINING AUROC is ~1.0 while its validation AUROC is chance. Layer 3
        carries a modest signal in BOTH halves, so its training AUROC is lower and its
        validation AUROC is high. A train-scored ranking picks 0; a val-scored one
        picks 3.
        """
        n = 200
        labels = [i % 2 for i in range(n)]
        y = np.asarray(labels)
        rng = np.random.default_rng(3)

        memorisable = rng.normal(size=(n, D)).astype(np.float32)
        for channel in range(D):
            memorisable[:160, channel] += y[:160] * 6.0      # train only, every channel
        generalising = rng.normal(size=(n, D)).astype(np.float32)
        generalising[:, 0] += y * 1.2                        # both halves, modest

        pooled = PooledCapture(
            mean={0: torch.tensor(memorisable), 3: torch.tensor(generalising)},
            last={0: torch.tensor(memorisable), 3: torch.tensor(generalising)},
        )
        selection = select_layers(pooled, labels, range(160), range(160, 200), top_n=1)
        assert selection.chosen == [3], (
            "the memorisable layer was chosen, so selection is scoring on data the "
            "model has already seen"
        )

    def test_that_fixture_really_does_separate_the_two_answers(self):
        """Proves the test above can fail: the memorisable layer must genuinely win on
        TRAINING data, or the assertion is satisfied by chance."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        n = 200
        labels = [i % 2 for i in range(n)]
        y = np.asarray(labels)
        rng = np.random.default_rng(3)
        memorisable = rng.normal(size=(n, D)).astype(np.float32)
        for channel in range(D):
            memorisable[:160, channel] += y[:160] * 6.0
        generalising = rng.normal(size=(n, D)).astype(np.float32)
        generalising[:, 0] += y * 1.2

        train_scores = {}
        for name, matrix in (("memorisable", memorisable), ("generalising", generalising)):
            model = LogisticRegression(max_iter=2000).fit(matrix[:160], y[:160])
            train_scores[name] = roc_auc_score(y[:160], model.decision_function(matrix[:160]))
        assert train_scores["memorisable"] > train_scores["generalising"], (
            f"on TRAINING data {train_scores}; the two rankings would agree, so the "
            f"test above cannot detect a train-scored selection"
        )

    def test_top_n_returns_that_many_layers(self):
        pooled, labels = _pooled(layers=4)
        assert len(select_layers(pooled, labels, range(160), range(160, 200), top_n=3).chosen) == 3

    def test_a_single_class_split_is_REFUSED_not_scored_as_chance(self):
        pooled, labels = _pooled()
        with pytest.raises(ValueError, match="single class"):
            select_layers(pooled, [1] * 200, range(160), range(160, 200))

    def test_the_grid_survives_serialisation(self):
        pooled, labels = _pooled()
        payload = select_layers(pooled, labels, range(160), range(160, 200)).as_dict()
        import json

        json.dumps(payload)     # must not raise: it is stored in JSONB
        assert payload["chosen"] and payload["grid"]


class TestItActuallyLearns:
    """The class the first configuration of this trainer would have failed."""

    @pytest.mark.parametrize(
        "rule", ["mean", "max", "last", "softmax", "attention", "rolling_mean_max"]
    )
    def test_every_rule_solves_a_separable_problem(self, rule):
        """⚠ 0.51 ON SEPARABLE DATA IS WHAT 60 STEPS AT lr 1e-3 PRODUCED. The loop ran,
        early-stopped and reported — output indistinguishable from "this concept is
        undetectable in this model"."""
        rows, labels = _token_rows(n=200, signal=2.0)
        trained = train_rule(rule, rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.val_auroc is not None
        assert trained.val_auroc > 0.80, (
            f"{rule} reached only {trained.val_auroc:.3f} on linearly separable data, so "
            f"this configuration cannot converge and would report a false negative result"
        )

    def test_a_TRUE_negative_result_still_reads_as_chance(self):
        """The counterpart: on data with no signal the trainer must NOT manufacture
        one, or the test above would be satisfied by a leak."""
        rng = np.random.default_rng(9)
        rows = [rng.normal(size=(5, D)).astype(np.float32) for _ in range(200)]
        labels = [i % 2 for i in range(200)]
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.val_auroc is not None
        assert trained.val_auroc < 0.75, (
            f"reached {trained.val_auroc:.3f} on pure noise, so something is leaking"
        )

    def test_a_stronger_signal_gives_a_better_probe(self):
        weak_rows, labels = _token_rows(signal=0.3, seed=5)
        strong_rows, _ = _token_rows(signal=4.0, seed=5)
        weak = train_rule("mean", weak_rows[:160], labels[:160], weak_rows[160:], labels[160:])
        strong = train_rule("mean", strong_rows[:160], labels[:160], strong_rows[160:], labels[160:])
        assert strong.val_auroc > weak.val_auroc


class TestEarlyStoppingAndRestore:
    def test_it_stops_on_validation_AUROC_not_loss(self):
        """Loss and AUROC do not share an argmin, so stopping on loss can hand back a
        probe that ranks worse than one seen three steps earlier — under a number
        nobody looks at."""
        rows, labels = _token_rows()
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        by_auroc = max(
            (h for h in trained.history if h["val_auroc"] is not None),
            key=lambda h: h["val_auroc"],
        )
        assert trained.best_epoch == by_auroc["epoch"]
        assert trained.val_auroc == pytest.approx(by_auroc["val_auroc"])

    def test_the_BEST_state_is_restored_not_the_last(self):
        """Returning the last step's weights while reporting the best step's AUROC
        reports a number the returned probe does not achieve.

        ⚠ AND THE FIRST VERSION OF THIS TEST COULD NOT FAIL. It recomputed the AUROC from
        the returned head and compared it with the reported one — but on a converged fixture
        the AUROC PLATEAUS: measured, the best epoch (60) and the last (100) both score
        0.997500, delta 0.000000. So "kept the last state" is indistinguishable from
        "restored the best" by AUROC, and a mutation disabling the restore survived.

        The WEIGHTS do keep moving. So the discriminating comparison is against a run
        stopped exactly at the best epoch: if the long run returned its final state, the
        two weight vectors would differ. Verified to discriminate before being asserted.
        """
        rows, labels = _token_rows()
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.epochs_run > trained.best_epoch, (
            "training stopped exactly at its best step, so this fixture cannot tell "
            "'restored the best' from 'kept the last'"
        )
        stopped_at_best = train_rule(
            "mean", rows[:160], labels[:160], rows[160:], labels[160:],
            epochs=trained.best_epoch,
        )
        assert stopped_at_best.epochs_run == trained.best_epoch
        assert torch.allclose(trained.weight, stopped_at_best.weight, atol=1e-7), (
            "the returned weights are not the weights at the best epoch, so the reported "
            "AUROC belongs to a probe that was not returned"
        )
        assert trained.bias == pytest.approx(stopped_at_best.bias, abs=1e-7)

    def test_that_comparison_actually_discriminates(self):
        """Prove the weights move after the best epoch, or the test above is vacuous."""
        rows, labels = _token_rows()
        short = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:], epochs=20)
        long = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:], epochs=100)
        assert not torch.allclose(short.weight, long.weight, atol=1e-4), (
            "the weights do not move between step 20 and step 100, so comparing weight "
            "vectors cannot detect a failure to restore"
        )

    def test_the_reported_AUROC_is_the_one_the_returned_head_achieves(self):
        """Still worth asserting — it just cannot, alone, detect a missing restore."""
        from sklearn.metrics import roc_auc_score

        from src.ml.probe_monitor_model import combine

        rows, labels = _token_rows()
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        head = head_from_trained(trained, layer=1)
        widths = [r.shape[0] for r in rows[160:]]
        packed = torch.zeros(len(widths), max(widths), D)
        mask = torch.zeros(len(widths), max(widths), dtype=torch.bool)
        for i, row in enumerate(rows[160:]):
            packed[i, : row.shape[0]] = torch.tensor(row)
            mask[i, : row.shape[0]] = True
        scores = combine("mean", head.token_scores(packed), mask=mask).detach().numpy()
        assert roc_auc_score(labels[160:], scores) == pytest.approx(
            trained.val_auroc, abs=1e-6
        )

    def test_patience_actually_stops_it_early(self):
        rows, labels = _token_rows()
        trained = train_rule(
            "mean", rows[:160], labels[:160], rows[160:], labels[160:],
            epochs=400, patience=3,
        )
        assert trained.epochs_run < 400

    def test_training_is_reproducible_under_the_same_seed(self):
        rows, labels = _token_rows()
        first = train_rule("attention", rows[:160], labels[:160], rows[160:], labels[160:], seed=7)
        second = train_rule("attention", rows[:160], labels[:160], rows[160:], labels[160:], seed=7)
        assert torch.allclose(first.weight, second.weight)
        assert torch.allclose(first.attention_query, second.attention_query)


class TestTheAttentionRuleGetsARealQuery:
    def test_the_query_is_trained_and_not_left_at_its_init(self):
        rows, labels = _token_rows()
        trained = train_rule("attention", rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.attention_query is not None
        assert trained.attention_query.abs().max().item() > 0.05, (
            "the query never moved, so `attention` is weighting every token equally — "
            "which makes it `mean` under a different name"
        )

    def test_a_zero_init_would_make_it_indistinguishable_from_mean(self):
        """Why the init is small-random rather than zeros: an all-zero query gives every
        softmax weight the same value and a symmetric gradient, so the rule would train,
        converge, and silently be a different rule."""
        query = torch.zeros(4)
        weights = torch.softmax(torch.zeros(3, 4) @ query, dim=-1)
        assert torch.allclose(weights, torch.full((3,), 1 / 3), atol=1e-6)

    def test_only_attention_gets_a_query(self):
        rows, labels = _token_rows()
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.attention_query is None


class TestStandardisationDoesNotAmplify:
    def test_a_constant_channel_gets_std_ONE_not_eps(self):
        """⚠ THE OTHER SIDE OF THE SAME DEFECT. `ProbeHead.standardise` zeroes a
        degenerate channel; if the statistics hand it a std of 1e-6 instead of a true
        zero, the head's own guard never triggers and the amplification returns."""
        rows, labels = _token_rows(constant_channel=True)
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        assert trained.std[1].item() == pytest.approx(1.0), (
            f"the zero-variance channel got std {trained.std[1].item()}, which the head "
            f"will divide by"
        )

    def test_a_real_channel_keeps_its_scale(self):
        rows, labels = _token_rows()
        trained = train_rule("mean", rows[:160], labels[:160], rows[160:], labels[160:])
        assert 0.1 < trained.std[5].item() < 10.0


class TestSAEFeatureSelectionRanksOnTrainOnly:
    def test_it_picks_the_separating_features(self):
        rng = np.random.default_rng(1)
        labels = [i % 2 for i in range(100)]
        rows = []
        for i in range(100):
            block = rng.normal(size=(4, 12)).astype(np.float32)
            block[:, 3] += labels[i] * 5.0
            block[:, 7] += labels[i] * 4.0
            rows.append(block)
        chosen = select_sae_features(rows, labels, k=2)
        assert set(chosen.tolist()) == {3, 7}

    def test_the_indices_come_back_ASCENDING_for_a_stable_basis(self):
        """An exported definition's feature list must be comparable between probes."""
        rng = np.random.default_rng(2)
        labels = [i % 2 for i in range(60)]
        rows = [rng.normal(size=(3, 20)).astype(np.float32) for _ in range(60)]
        chosen = select_sae_features(rows, labels, k=5)
        assert list(chosen) == sorted(chosen)

    def test_ranking_uses_ONLY_the_rows_it_is_given(self):
        """⚠ THE LEAK IS STRUCTURAL, NOT NUMERIC. Ranking on validation rows decides
        WHICH FEATURES EXIST using held-out data, so the validation AUROC that then
        drives early stopping is no longer held out — and the reported number is
        optimistic by an amount nobody can estimate afterwards.

        Built so the two answers differ: feature 3 separates the first half, feature 9
        the second.
        """
        # ⚠ THE DISCRIMINATING PROPERTY IS CONSISTENCY, NOT MAGNITUDE, AND TWO EARLIER
        # FIXTURES GOT THAT WRONG. The ranking statistic is a STANDARDISED class-mean
        # difference, so scale cancels: making one feature's signal twenty times larger
        # in half the rows also multiplies its pooled standard deviation, and the score
        # barely moves. Both earlier versions therefore ranked feature 3 first over the
        # full set and the assertion passed for the wrong reason.
        #
        # What does separate the two answers: feature 3 separates the TRAIN half only
        # (strongly, so it wins there), while feature 9 separates EVERY row modestly —
        # so over the full set feature 9's smaller variance makes it the winner.
        rng = np.random.default_rng(4)
        labels = [i % 2 for i in range(100)]
        rows = []
        for i in range(100):
            block = rng.normal(size=(4, 12)).astype(np.float32)
            if i < 50:
                block[:, 3] += labels[i] * 8.0     # train half only, strong there
            block[:, 9] += labels[i] * 2.0         # every row, modest
            rows.append(block)
        train_only = select_sae_features(rows[:50], labels[:50], k=1)
        everything = select_sae_features(rows, labels, k=1)
        assert train_only.tolist() == [3]
        assert everything.tolist() == [9], (
            "this fixture cannot distinguish a train-only ranking from an all-rows one"
        )

    #: A fixture where the correct statistic and the "every row is a positive" variant
    #: DISAGREE. Found by SEARCH over (seed, gaps, noises) using the exact construction
    #: below — twice, because my first two attempts had both definitions choose the same
    #: feature and the negative control caught each one. The search is reproduced in the
    #: review record; what matters here is that the control below keeps it honest.
    #:
    #: Feature 0: a large class gap (8.0) with small within-class noise (0.5) — the
    #: correct winner. Feature 1: a smaller gap (2.0) with tiny noise (0.15). Pooling
    #: every row into the "positive" mean halves both gaps and inflates feature 0's
    #: variance by the gap it just absorbed, which flips the ranking to feature 1.
    _LEAK_PARAMS = (8.0, 0.5, 2.0, 0.15)
    _LEAK_SEED = 3

    def _two_feature_rows(self):
        gap0, noise0, gap1, noise1 = self._LEAK_PARAMS
        rng = np.random.default_rng(self._LEAK_SEED)
        n = 120
        labels = [1] * 60 + [0] * 60
        y = np.asarray(labels)
        pooled = np.zeros((n, 2), dtype=np.float32)
        pooled[:, 0] = rng.normal(scale=noise0, size=n) + np.where(y == 1, gap0, 0.0)
        pooled[:, 1] = rng.normal(scale=noise1, size=n) + np.where(y == 1, gap1, 0.0)
        # One token per row, so the row-pooling step is the identity and the fixture's
        # numbers reach the statistic unchanged.
        rows = [pooled[i : i + 1, :].copy() for i in range(n)]
        return rows, labels, pooled, y

    @staticmethod
    def _rank(pooled, y, positives):
        """The statistic, written out, so a test can compare against the DEFINITION."""
        negatives = pooled[y == 0]
        spread = np.sqrt((positives.var(axis=0) + negatives.var(axis=0)) / 2.0)
        gaps = np.abs(positives.mean(axis=0) - negatives.mean(axis=0))
        return int(np.argmax(gaps / (spread + 1e-6)))

    def test_the_positive_mean_uses_ONLY_positive_rows(self):
        """⚠ THE LEAK TEST ABOVE COMPARES TWO CALLS OF THE SAME FUNCTION, so a change
        INSIDE it — computing the positive mean over every row rather than the positives —
        affects both sides equally and cannot be detected. A mutation doing exactly that
        survived the suite.

        ⚠ AND THIS TEST DOES NOT HARDCODE WHICH FEATURE WINS. My first version asserted a
        remembered index taken from a parameter search, and the search's RNG draws differed
        from the fixture's — so the expected value was simply wrong. Comparing against the
        DEFINITION, computed inline, cannot go stale that way; the companion test below is
        what proves the two definitions actually disagree here.
        """
        rows, labels, pooled, y = self._two_feature_rows()
        expected = self._rank(pooled, y, pooled[y == 1])
        assert select_sae_features(rows, labels, k=1).tolist() == [expected]

    def test_a_MUTATED_statistic_would_give_a_different_answer(self):
        """The negative control: the all-rows variant must choose a DIFFERENT feature, or
        the assertion above passes against the mutation it was written to catch.

        My first fixture failed exactly this — both definitions chose the same feature —
        which is why the control is a test rather than a comment.
        """
        _rows, _labels, pooled, y = self._two_feature_rows()
        correct = self._rank(pooled, y, pooled[y == 1])
        mutated = self._rank(pooled, y, pooled)      # every row treated as a positive
        assert correct != mutated, (
            f"both definitions choose feature {correct}, so the fixture cannot detect "
            f"the mutation"
        )

    def test_k_larger_than_the_dictionary_returns_every_feature(self):
        rng = np.random.default_rng(6)
        labels = [i % 2 for i in range(40)]
        rows = [rng.normal(size=(2, 5)).astype(np.float32) for _ in range(40)]
        assert select_sae_features(rows, labels, k=99).tolist() == [0, 1, 2, 3, 4]

    def test_k_below_one_is_refused(self):
        rng = np.random.default_rng(7)
        rows = [rng.normal(size=(2, 5)).astype(np.float32) for _ in range(4)]
        with pytest.raises(ValueError, match="at least 1"):
            select_sae_features(rows, [0, 1, 0, 1], k=0)

    def test_a_single_class_is_refused(self):
        rng = np.random.default_rng(8)
        rows = [rng.normal(size=(2, 5)).astype(np.float32) for _ in range(4)]
        with pytest.raises(ValueError, match="single class"):
            select_sae_features(rows, [1, 1, 1, 1], k=2)


class TestCalibrationIsHonestAboutWhatItSpends:
    def test_the_realised_rate_is_reported_not_the_target(self):
        """⚠ WITH 100 NEGATIVES THE ACHIEVABLE RATES ARE MULTIPLES OF 0.01. Reporting
        the TARGET as though it were achieved is the honest-absence-into-silent-lie
        shape this estate has already shipped once."""
        scores = list(np.linspace(0, 1, 100))
        result = calibrate(scores, target_fpr=0.037, source="calibration_set")
        assert result.target_fpr == 0.037
        assert result.realised_fpr != 0.037
        assert result.realised_fpr <= 0.037 + 1e-9

    def test_it_never_OVERSPENDS_the_budget(self):
        """Rounding up to hit the target exactly is the wrong direction for a monitor."""
        for n in (7, 13, 20, 99, 100, 1000):
            scores = list(np.linspace(0, 1, n))
            result = calibrate(scores, target_fpr=0.01, source="s")
            assert result.realised_fpr <= 0.01 + 1e-9, f"n={n} spent {result.realised_fpr}"

    def test_fire_on_nothing_is_None_and_NOT_inf(self):
        """A 1% budget with 20 negatives admits no firing threshold, which is a real
        operating point — and `inf` cannot be serialised: Starlette renders JSON with
        `allow_nan=False` and jsonb rejects it, so a report endpoint would 500."""
        import json

        result = calibrate(list(np.linspace(0, 1, 20)), target_fpr=0.01, source="s")
        assert result.threshold is None
        assert result.realised_fpr == 0.0
        json.dumps(result.as_dict())

    def test_the_threshold_actually_achieves_the_reported_rate(self):
        rng = np.random.default_rng(11)
        scores = list(rng.normal(size=500))
        result = calibrate(scores, target_fpr=0.05, source="s")
        fired = sum(1 for s in scores if s >= result.threshold) / len(scores)
        assert fired == pytest.approx(result.realised_fpr)

    def test_ties_at_the_threshold_are_counted(self):
        """Ten identical negatives at the boundary all fire, and the realised rate must
        say so rather than assuming distinct scores."""
        scores = [0.5] * 10 + [0.1] * 90
        result = calibrate(scores, target_fpr=0.05, source="s")
        if result.threshold is not None:
            fired = sum(1 for s in scores if s >= result.threshold) / len(scores)
            assert fired == pytest.approx(result.realised_fpr)

    def test_the_SOURCE_is_recorded(self):
        """A threshold from validation negatives and one from a held-out calibration
        corpus are different claims."""
        scores = list(np.linspace(0, 1, 200))
        assert calibrate(scores, target_fpr=0.05, source="validation_negatives").source == (
            "validation_negatives"
        )

    def test_no_negatives_is_refused(self):
        with pytest.raises(ValueError, match="without negatives"):
            calibrate([], target_fpr=0.01, source="s")

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_an_out_of_range_target_is_refused(self, bad):
        with pytest.raises(ValueError, match="target_fpr"):
            calibrate([0.1, 0.2], target_fpr=bad, source="s")


class TestHeadConstruction:
    def test_the_head_carries_the_layer(self):
        rows, labels = _token_rows(n=60)
        trained = train_rule("mean", rows[:48], labels[:48], rows[48:], labels[48:], epochs=20)
        head = head_from_trained(trained, layer=7)
        assert head.layer == 7

    def test_it_carries_the_query_for_attention_and_not_otherwise(self):
        rows, labels = _token_rows(n=60)
        attention = train_rule("attention", rows[:48], labels[:48], rows[48:], labels[48:], epochs=20)
        plain = train_rule("mean", rows[:48], labels[:48], rows[48:], labels[48:], epochs=20)
        assert head_from_trained(attention, 1).attention_query is not None
        assert head_from_trained(plain, 1).attention_query is None
