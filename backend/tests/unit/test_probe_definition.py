"""The `mistudio.probe-definition/v1` contract, one test per validator (033 1.3, 1.5).

⚠ EVERY VALIDATOR HERE EXISTS BECAUSE THE FAILURE IT CATCHES IS SILENT. A probe served in the
wrong basis, with the wrong normalisation, or with `attention` falling back to `softmax`, does not
raise and does not look broken: it produces plausible scores about the wrong thing. This estate has
shipped that class of defect repeatedly — a capture at a post-attention norm for months, an SAE
encoded without its training normalisation, a steering vector renormalised away — so the contract
refuses rather than interprets.

MUTATION CONTROLS (each verified to fail this file):
  C1  `_all_vectors_share_one_width` deleted            → the width-agreement tests
  C2  `_no_zero_or_negative_std` deleted                → the degenerate-std test
  C3  the attention-query validator deleted             → both attention tests
  C4  the sae/basis validator deleted                   → both basis tests
  C5  `_the_head_width_matches_the_basis` deleted       → the head-width tests
  C6  `_the_layer_is_inside_the_model` deleted          → the layer test
  C7  `_streamable_agrees_with_the_rule` deleted        → the streamable tests
  C8  `_a_low_rung_needs_an_acknowledgement` deleted    → the rung-gate tests
  C9  `_sorted_distinct_and_non_negative` deleted       → the feature-order tests
  C10 `extra="forbid"` relaxed on the root              → the unknown-field test
  C11 `_refuse_local_path` made a no-op                 → the portability tests
  C12 the `kind` Literal widened                        → the kind test
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.ml.probe_monitor_model import RULES, STREAMABLE
from src.schemas.probe_definition import (
    MAX_D,
    MAX_DEFINITION_BYTES,
    MAX_EVALUATIONS,
    MAX_TEST_VECTORS,
    MAX_VECTOR_TOKENS,
    MIN_TEST_VECTORS,
    PROBE_DEFINITION_KIND,
    ProbeDefinitionV1,
    rule_names_match_the_trainer,
)

D = 4


def _vector(index: int = 0) -> dict:
    return {
        "messages": [{"role": "user", "content": f"row {index}"}],
        "token_ids": [1, 2, 3],
        "token_scores": [0.1, 0.2, 0.3],
        "score": 0.2,
        "verdict": index % 2 == 0,
    }


def definition(**overrides) -> dict:
    """A minimal VALID document. Every test below changes exactly one thing."""
    document = {
        "kind": PROBE_DEFINITION_KIND,
        "name": "high-stakes",
        "description": "fires on high-stakes deployment contexts",
        "concept": "positive = high-stakes",
        "model": {
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
            "d_model": D,
            "n_layers": 32,
            "architecture": "llama",
            "chat_template_sha256": "e" * 64,
            "mistudio_model_id": "m_40e78d80",
        },
        "read": {"layer": 11, "hook_point": "resid_post"},
        "scope": "all",
        "basis": "residual",
        "head": {
            "weights": [0.1] * D,
            "bias": -0.5,
            "norm_mean": [0.0] * D,
            "norm_std": [1.0] * D,
            "attention_query": None,
        },
        "aggregation": {"rule": "mean", "params": {}, "streamable": True},
        "decision": {
            "threshold": 2.38,
            "target_fpr": 0.01,
            "realised_fpr": 0.0097,
            "threshold_source": "validation_negatives",
            "calibration": None,
        },
        "evidence": {
            "rung": 2,
            "rung_language": "detects on unseen tasks",
            "acknowledgement": None,
            "evaluations": [
                {
                    "dataset": {
                        "hf_id": "Arrrlex/models-under-pressure",
                        "config": "mt_balanced",
                        "split": "test",
                        "revision": None,
                    },
                    "distribution": "out_of_distribution",
                    "n_positive": 302,
                    "n_negative": 302,
                    "auroc": 0.9574,
                    "auroc_ci": [0.9415, 0.9722],
                    "recall_at_target_fpr": 0.6159,
                }
            ],
            "judge": None,
        },
        "provenance": {"run_id": "pmr_x", "probe_id": "pm_x"},
        "test_vectors": {
            "tolerance": 0.01,
            "vectors": [_vector(i) for i in range(MIN_TEST_VECTORS)],
        },
    }
    document.update(overrides)
    return document


def _deep(document: dict, path: str, value) -> dict:
    """Set `a.b.0.c` in a copy, so each test changes exactly one field.

    A numeric segment indexes a list — `evidence.evaluations.0.auroc` has to reach into the
    evaluation list, and the first version of this helper raised `TypeError: list indices must be
    integers` on seven tests, which reads as a contract failure rather than a helper bug.
    """
    copy = json.loads(json.dumps(document))
    node = copy
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value
    return copy


class TestTheFixtureIsValid:
    """The control for every test below: if the baseline did not validate, a test asserting a
    REFUSAL would pass for the wrong reason."""

    def test_it_round_trips(self):
        parsed = ProbeDefinitionV1.model_validate(definition())
        again = ProbeDefinitionV1.model_validate(json.loads(parsed.model_dump_json()))
        assert again == parsed

    def test_the_serialised_form_keeps_every_wire_field(self):
        """⚠ AN `alias` RENAMES ON OUTPUT TOO. Doing that once republished this estate's cluster
        schema without its wire field and invalidated every exported document."""
        payload = json.loads(ProbeDefinitionV1.model_validate(definition()).model_dump_json())
        for field in ("kind", "model", "read", "scope", "basis", "head", "aggregation",
                      "decision", "evidence", "provenance", "test_vectors"):
            assert field in payload, f"{field} vanished on serialisation"

    def test_it_is_far_under_the_size_cap(self):
        size = len(ProbeDefinitionV1.model_validate(definition()).model_dump_json())
        assert size < MAX_DEFINITION_BYTES


class TestTheKindCarriesTheVersion:
    def test_the_default_is_the_versioned_kind(self):
        assert ProbeDefinitionV1.model_validate(definition()).kind == PROBE_DEFINITION_KIND
        assert PROBE_DEFINITION_KIND.endswith("/v1")

    def test_another_kind_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "kind", "mistudio.probe/v2"))

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(definition(surprise=1))
        assert "surprise" in str(caught.value)


class TestTheRuleSetComesFrom032:
    def test_the_literal_equals_the_trainers_rules(self):
        """A rule the trainer can produce but the contract cannot express is a probe that cannot
        be exported; the reverse validates a document nothing can serve."""
        assert rule_names_match_the_trainer(), sorted(RULES)

    def test_an_invented_rule_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "aggregation.rule", "median"))

    @pytest.mark.parametrize("rule", sorted(RULES))
    def test_streamable_must_agree_with_the_rule(self, rule):
        expected = rule in STREAMABLE
        body = _deep(definition(), "aggregation.rule", rule)
        body = _deep(body, "aggregation.streamable", expected)
        if rule == "attention":
            body = _deep(body, "head.attention_query", [0.1] * D)
        ProbeDefinitionV1.model_validate(body)          # the truthful combination

        wrong = _deep(body, "aggregation.streamable", not expected)
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(wrong)
        assert "streaming" in str(caught.value) or "stream" in str(caught.value)

    def test_last_is_the_one_rule_that_cannot_stream(self):
        """Pinned as a fact, not inferred: `last` needs the final token, so a consumer told it
        may stream would report a partial score as final."""
        assert "last" not in STREAMABLE
        assert set(RULES) - STREAMABLE == {"last"}


class TestTheHeadAndItsNormalisation:
    @pytest.mark.parametrize("field", ["norm_mean", "norm_std", "attention_query"])
    def test_a_mismatched_vector_width_is_refused(self, field):
        body = definition()
        if field == "attention_query":
            body = _deep(body, "aggregation.rule", "attention")
            body = _deep(body, "aggregation.streamable", True)
        body = _deep(body, f"head.{field}", [0.1] * (D + 1))
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "width" in str(caught.value)

    def test_a_zero_std_is_refused(self):
        """⚠ CLAMPING WAS THE WRONG FIX AND AMPLIFIED INSTEAD. A 1e-6 floor turned a 0.001 drift
        into 1000.0 and fired the monitor on a channel carrying no signal."""
        body = _deep(definition(), "head.norm_std", [1.0, 0.0, 1.0, 1.0])
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "non-positive" in str(caught.value)

    def test_a_negative_std_is_refused(self):
        body = _deep(definition(), "head.norm_std", [1.0, -1.0, 1.0, 1.0])
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)

    def test_a_degenerate_channel_recorded_as_one_is_accepted(self):
        """The positive half: 032 writes 1.0 for a constant channel, which contributes nothing."""
        ProbeDefinitionV1.model_validate(_deep(definition(), "head.norm_std", [1.0] * D))

    def test_a_head_wider_than_the_cap_is_refused(self):
        body = _deep(definition(), "head.weights", [0.1] * (MAX_D + 1))
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)


class TestAttentionNeedsItsQuery:
    def test_attention_without_a_query_is_refused(self):
        body = _deep(definition(), "aggregation.rule", "attention")
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "attention_query" in str(caught.value)

    def test_a_query_on_a_non_attention_rule_is_refused(self):
        """Both directions, because a stray query means the document describes two readouts."""
        body = _deep(definition(), "head.attention_query", [0.1] * D)
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "does not use it" in str(caught.value)

    def test_attention_with_its_query_is_accepted(self):
        body = _deep(definition(), "aggregation.rule", "attention")
        body = _deep(body, "head.attention_query", [0.1] * D)
        assert ProbeDefinitionV1.model_validate(body).head.attention_query == [0.1] * D


class TestTheBasisAndTheSaeBlock:
    @staticmethod
    def _sae(**overrides) -> dict:
        block = {
            "hf_repo": "hitsainet/mistudio-saes",
            "path": "lfm2-l11/sae.safetensors",
            "revision": "a" * 40,
            "weights_sha256": "b" * 64,
            "architecture": "jumprelu",
            "d_model": D,
            "n_features": 16384,
            "normalization": {"mode": "constant_norm_rescale", "target": 1.0},
            "feature_indices": [3, 9, 27],
            "feature_labels": None,
        }
        block.update(overrides)
        return block

    def _sae_definition(self, **sae_overrides) -> dict:
        body = definition()
        body["basis"] = "sae_features"
        body["sae"] = self._sae(**sae_overrides)
        body["head"]["weights"] = [0.1] * 3          # k = 3 selected features
        body["head"]["norm_mean"] = [0.0] * 3
        body["head"]["norm_std"] = [1.0] * 3
        return body

    def test_an_sae_basis_with_its_block_is_accepted(self):
        parsed = ProbeDefinitionV1.model_validate(self._sae_definition())
        assert parsed.sae is not None
        assert len(parsed.head.weights) == len(parsed.sae.feature_indices)

    def test_an_sae_basis_without_the_block_is_refused(self):
        body = _deep(definition(), "basis", "sae_features")
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "no `sae` block" in str(caught.value)

    def test_a_residual_basis_with_an_sae_block_is_refused(self):
        body = definition()
        body["sae"] = self._sae()
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "wrong basis" in str(caught.value) or "residual" in str(caught.value)

    def test_the_head_width_must_match_d_model_on_a_residual_probe(self):
        body = _deep(definition(), "head.weights", [0.1] * (D - 1))
        body = _deep(body, "head.norm_mean", [0.0] * (D - 1))
        body = _deep(body, "head.norm_std", [1.0] * (D - 1))
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "d_model" in str(caught.value)

    def test_the_head_width_must_match_k_on_an_sae_probe(self):
        body = self._sae_definition()
        body["head"]["weights"] = [0.1] * 2          # 2 weights for 3 features
        body["head"]["norm_mean"] = [0.0] * 2
        body["head"]["norm_std"] = [1.0] * 2
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "SAE" in str(caught.value) or "features are selected" in str(caught.value)

    def test_unsorted_feature_indices_are_refused(self):
        """Each weight pairs with one selected feature by POSITION, so a different order silently
        pairs every weight with another feature."""
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(self._sae_definition(feature_indices=[27, 3, 9]))
        assert "sorted" in str(caught.value)

    def test_duplicate_feature_indices_are_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(self._sae_definition(feature_indices=[3, 3, 9]))

    def test_an_index_outside_the_dictionary_is_refused(self):
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(
                self._sae_definition(feature_indices=[3, 9, 99999], n_features=1024)
            )
        assert "outside a dictionary" in str(caught.value)

    def test_an_sae_of_another_width_is_refused(self):
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(self._sae_definition(d_model=D + 1))
        assert "cannot encode" in str(caught.value)

    def test_labels_must_match_the_indices_one_for_one(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(self._sae_definition(feature_labels=["a", "b"]))

    def test_a_short_weights_sha_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(self._sae_definition(weights_sha256="b" * 40))


class TestTheReadPoint:
    def test_a_layer_beyond_the_model_is_refused(self):
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(_deep(definition(), "read.layer", 32))
        assert "outside a 32-layer model" in str(caught.value)

    def test_the_last_layer_is_inside(self):
        ProbeDefinitionV1.model_validate(_deep(definition(), "read.layer", 31))

    def test_a_negative_layer_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "read.layer", -1))

    def test_another_hook_point_is_refused(self):
        """⚠ THE ONE-MEMBER Literal IS THE POINT. This estate captured at a post-attention norm
        for months while every document said "residual"."""
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "read.hook_point", "mlp_out"))


class TestTheEvidenceGateTravelsWithTheFormat:
    @pytest.mark.parametrize("rung", [0, 1])
    def test_a_low_rung_without_an_acknowledgement_is_refused(self, rung):
        body = _deep(definition(), "evidence.rung", rung)
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "acknowledgement" in str(caught.value)

    @pytest.mark.parametrize("rung", [0, 1])
    def test_a_low_rung_with_an_acknowledgement_is_accepted(self, rung):
        body = _deep(definition(), "evidence.rung", rung)
        body = _deep(body, "evidence.acknowledgement", {
            "by": "sean",
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": "exploratory monitor on a known-narrow concept; not for gating",
        })
        parsed = ProbeDefinitionV1.model_validate(body)
        assert parsed.evidence.acknowledgement is not None

    @pytest.mark.parametrize("rung", [2, 3])
    def test_a_sufficient_rung_needs_no_acknowledgement(self, rung):
        ProbeDefinitionV1.model_validate(_deep(definition(), "evidence.rung", rung))

    def test_a_token_reason_is_refused(self):
        """"ok" records nothing. The floor is what makes the acknowledgement evidence."""
        body = _deep(definition(), "evidence.rung", 1)
        body = _deep(body, "evidence.acknowledgement", {
            "by": "sean", "at": datetime.now(timezone.utc).isoformat(), "reason": "ok",
        })
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)

    def test_a_rung_above_the_ladder_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "evidence.rung", 4))

    def test_too_many_evaluations_are_refused(self):
        one = definition()["evidence"]["evaluations"][0]
        body = _deep(definition(), "evidence.evaluations", [one] * (MAX_EVALUATIONS + 1))
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "cap" in str(caught.value)

    def test_an_unordered_confidence_interval_is_refused(self):
        body = _deep(definition(), "evidence.evaluations.0.auroc_ci", [0.9, 0.8])
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)

    def test_an_auroc_outside_zero_one_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(
                _deep(definition(), "evidence.evaluations.0.auroc", 1.4)
            )


class TestPortabilityIsEnforced:
    @pytest.mark.parametrize(
        "path",
        [
            "model.hf_id",
            "evidence.evaluations.0.dataset.hf_id",
        ],
    )
    @pytest.mark.parametrize(
        "bad", ["/data/models/m_x", "~/models/x", "../x", "C:\\models\\x", "file:///data/x"]
    )
    def test_a_filesystem_path_is_refused(self, path, bad):
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(_deep(definition(), path, bad))
        assert "filesystem path" in str(caught.value)

    def test_an_hf_identifier_is_accepted(self):
        ProbeDefinitionV1.model_validate(_deep(definition(), "model.hf_id", "owner/repo"))

    def test_an_sae_repo_path_is_refused(self):
        body = TestTheBasisAndTheSaeBlock()._sae_definition(hf_repo="/mnt/saes")
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "filesystem path" in str(caught.value)


class TestTheTestVectors:
    def test_fewer_than_the_floor_is_refused(self):
        body = _deep(definition(), "test_vectors.vectors", [_vector()] * (MIN_TEST_VECTORS - 1))
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)

    def test_more_than_the_ceiling_is_refused(self):
        body = _deep(definition(), "test_vectors.vectors", [_vector()] * (MAX_TEST_VECTORS + 1))
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(body)

    def test_a_vector_longer_than_the_token_cap_is_refused(self):
        long = _vector()
        long["token_ids"] = list(range(MAX_VECTOR_TOKENS + 1))
        body = _deep(definition(), "test_vectors.vectors", [long] * MIN_TEST_VECTORS)
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "cap" in str(caught.value)

    def test_more_scores_than_tokens_is_refused(self):
        """Scores are per SCORED token, so there can never be more of them than tokens."""
        bad = _vector()
        bad["token_scores"] = [0.1] * 4              # 4 scores, 3 tokens
        body = _deep(definition(), "test_vectors.vectors", [bad] * MIN_TEST_VECTORS)
        with pytest.raises(ValidationError) as caught:
            ProbeDefinitionV1.model_validate(body)
        assert "per SCORED token" in str(caught.value)

    def test_fewer_scores_than_tokens_is_fine(self):
        """The normal case under a narrowed scope: only the assistant's tokens are scored."""
        partial = _vector()
        partial["token_scores"] = [0.3]
        body = _deep(definition(), "test_vectors.vectors", [partial] * MIN_TEST_VECTORS)
        ProbeDefinitionV1.model_validate(body)

    def test_a_non_positive_tolerance_is_refused(self):
        """A zero tolerance demands bit-identical floats across two runtimes, which fp16 vs bf16
        already breaks by about 1.5% relative at resid_post."""
        with pytest.raises(ValidationError):
            ProbeDefinitionV1.model_validate(_deep(definition(), "test_vectors.tolerance", 0.0))


class TestTheContractIsVendoredIdentically:
    """⚠ PIN THE TWO REPOS TO EACH OTHER, NOT EACH TO ITS OWN MIRROR. Regenerating one side passes
    its own sync test while silently drifting from the other — 009 R3 found exactly that."""

    OURS = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-definition-v1.json"
    THEIRS = Path("/home/x-sean/app/miLLM/docs/schemas/probe-definition-v1.json")

    def test_ours_exists(self):
        assert self.OURS.exists(), self.OURS

    def test_byte_identical_to_millms_vendored_copy(self):
        import os

        if not self.THEIRS.exists():
            if os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
                pytest.fail(
                    f"{self.THEIRS} is absent and MISTUDIO_REQUIRE_CROSS_REPO_CHECKS=1. The "
                    f"probe contract must be vendored into miLLM before phase 7 co-releases."
                )
            pytest.skip("miLLM's vendored copy is not present in this environment")
        assert self.OURS.read_bytes() == self.THEIRS.read_bytes(), (
            "the two repos' copies of probe-definition-v1.json differ; copy ours across rather "
            "than regenerating there, or the mirror drifts from the generator"
        )
