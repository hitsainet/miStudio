"""Pins for the validation math (017, A.5/A.7): ES, sign gate, null
threshold, survival/uplift, necessity/sufficiency."""

import numpy as np
import pytest

from src.services.circuit_validation_math import (
    edge_verdict,
    effect_size,
    necessity,
    sign_consistency,
    sufficiency,
    survival_rate,
    uplift,
)


class TestEffectSize:
    def test_es_is_mean_over_sigma(self):
        assert effect_size([2.0, 4.0], sigma_d=2.0) == pytest.approx(1.5)

    def test_zero_sigma_is_zero(self):
        assert effect_size([1.0], 0.0) == 0.0

    def test_empty(self):
        assert effect_size([], 1.0) == 0.0


class TestSignConsistency:
    def test_all_agree(self):
        assert sign_consistency([1.0, 2.0, 3.0]) == 1.0

    def test_split(self):
        assert sign_consistency([1.0, 1.0, 1.0, -1.0]) == pytest.approx(0.75)


class TestEdgeVerdict:
    # A real null needs >= MIN_NULL_SAMPLES (10) samples (R1 Q1) — a tight,
    # adequately-sized null distribution.
    _TIGHT_NULL = [0.1, -0.2, 0.15, -0.05, 0.08, -0.11, 0.05, -0.09, 0.12, -0.03, 0.07]
    _BIG_NULL = [0.5, -0.6, 0.55, -0.5, 0.6, -0.55, 0.52, -0.58, 0.48, -0.62, 0.5]

    def test_strong_consistent_edge_passes(self):
        # ES big vs a tight (adequately-sized) null, all prompts same sign
        v = edge_verdict([3.0, 3.2, 2.8, 3.1], sigma_d=1.0,
                         null_effect_sizes=self._TIGHT_NULL)
        assert v.passed and v.effect_size > 2
        assert v.sign_consistency == 1.0

    def test_below_null_fails(self):
        v = edge_verdict([0.1, 0.1], sigma_d=1.0,
                         null_effect_sizes=self._BIG_NULL)
        assert not v.passed and "null" in v.reason

    def test_sign_inconsistent_fails(self):
        # big magnitude but prompts disagree → not causal
        v = edge_verdict([3.0, -3.0, 3.0, -3.0, 0.1], sigma_d=1.0,
                         null_effect_sizes=[0.01] * 11)
        assert not v.passed and "sign" in v.reason

    def test_underpowered_null_cannot_validate(self):
        # R1 Q1: too few null samples → CANNOT validate (not a silent pass).
        v = edge_verdict([5.0, 5.0], sigma_d=1.0, null_effect_sizes=[0.01, 0.02])
        assert not v.passed and "underpowered" in v.reason

    def test_empty_null_cannot_validate(self):
        v = edge_verdict([5.0, 5.0], sigma_d=1.0, null_effect_sizes=[])
        assert not v.passed and "underpowered" in v.reason


class TestSurvivalUplift:
    def test_survival(self):
        assert survival_rate([True, True, False, False]) == 0.5
        assert survival_rate([]) is None

    def test_uplift(self):
        assert uplift(0.6, 0.4) == 0.2
        assert uplift(None, 0.4) is None

    def test_uplift_can_be_negative_or_zero(self):
        assert uplift(0.3, 0.5) == -0.2
        assert uplift(0.5, 0.5) == 0.0


class TestFaithfulness:
    def test_necessity(self):
        # members account for 80% of the ablatable behavior
        assert necessity(b_clean=1.0, b_ablate_m=0.2, b_ablate_all=0.0) == 0.8

    def test_necessity_zero_denominator(self):
        assert necessity(1.0, 0.5, 1.0) is None

    def test_sufficiency(self):
        assert sufficiency(b_ablate_nonmembers=0.9, b_ablate_all=0.0,
                           b_clean=1.0) == 0.9
