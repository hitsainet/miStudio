"""Probe monitor request schemas: what they refuse, and why each refusal matters.

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M80  `_rules_must_exist` accepts an unknown rule      → the unknown-rule test fails
  M81  explicit `layers` no longer clears `stride`      → the ambiguity test fails
  M82  a train set with one class is accepted           → the both-classes test fails
  M83  a calibration set may map 'positive'            → the calibration test fails
  M84  train_dataset_id may appear in eval_dataset_ids  → the leakage test fails
  M85  `ScoreRequest` accepts text AND messages         → the one-input test fails
  M86  the 16k cap truncates instead of refusing        → the cap test fails
  M87  `distribution` accepted on a train set           → the distribution test fails

⚠ THE REFUSALS ARE THE FEATURE. Each one blocks a way of producing a probe that
looks good for the wrong reason: evaluating on the training set reports memorisation
as detection; a single-class set yields an AUROC that is undefined or trivially 1.0;
a calibration set carrying "positive" means somebody invented a label for
conversational data that has none; and a silently truncated input returns a score for
text the caller never sent.
"""
import pytest
from pydantic import ValidationError

from src.ml.probe_monitor_model import RULES
from src.schemas.probe_monitor import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    DEFAULT_STRIDE,
    DEFAULT_TARGET_FPR,
    DEFAULT_TOP_N_LAYERS,
    DEFAULT_VAL_FRACTION,
    JudgeRunCreate,
    KeywordFilterSpec,
    ProbeDatasetCreate,
    ProbeRunConfig,
    ProbeRunCreate,
    ScoreRequest,
)


def _dataset(**overrides):
    base = dict(
        name="t",
        dataset_id="ds",
        input_column="inputs",
        label_column="stakes",
        label_mapping={"high": "positive", "low": "negative"},
    )
    base.update(overrides)
    return ProbeDatasetCreate(**base)


class TestTheDefaultsAreTheDocumentedOnes:
    """FTID §9 fixes these, and FR-15 needs a run reproducible from its record — so
    they are constants here, not environment variables."""

    def test_they_match_the_specification(self):
        assert (DEFAULT_STRIDE, DEFAULT_TOP_N_LAYERS) == (5, 1)
        assert DEFAULT_VAL_FRACTION == 0.15
        assert DEFAULT_SEED == 1337
        assert DEFAULT_MAX_LENGTH == 4096
        assert DEFAULT_TARGET_FPR == 0.01

    def test_a_default_config_is_valid_and_carries_them(self):
        config = ProbeRunConfig()
        assert config.stride == DEFAULT_STRIDE
        assert config.seed == DEFAULT_SEED
        assert config.target_fpr == DEFAULT_TARGET_FPR

    def test_no_default_reads_the_environment(self, monkeypatch):
        """A default a deployment can change is a default no run record explains."""
        monkeypatch.setenv("PROBE_MONITOR_STRIDE", "99")
        monkeypatch.setenv("DEFAULT_STRIDE", "99")
        assert ProbeRunConfig().stride == 5


class TestRuleSelection:
    def test_an_unknown_rule_is_refused_and_the_options_are_named(self):
        with pytest.raises(ValidationError) as caught:
            ProbeRunConfig(rules=["mean", "telepathy"])
        message = str(caught.value)
        assert "telepathy" in message
        assert "mean" in message, "the error must name what IS available"

    @pytest.mark.parametrize("rule", sorted(RULES))
    def test_every_implemented_rule_is_accepted(self, rule):
        """The schema and `ml.probe_monitor_model` must not drift: a rule the trainer
        implements but the schema rejects is unreachable."""
        assert ProbeRunConfig(rules=[rule]).rules == [rule]

    def test_an_empty_rule_list_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeRunConfig(rules=[])

    def test_duplicates_are_collapsed_keeping_order(self):
        """Two identical rules train two identical probes and double the report."""
        assert ProbeRunConfig(rules=["max", "mean", "max"]).rules == ["max", "mean"]


class TestLayersVersusStride:
    def test_explicit_layers_clear_the_stride(self):
        """Both set is ambiguous, and silently preferring one sweeps layers the
        caller never asked for — the stored config would then claim a stride that
        governed nothing."""
        config = ProbeRunConfig(layers=[11, 12, 13], stride=5)
        assert config.layers == [11, 12, 13]
        assert config.stride is None

    def test_neither_is_refused(self):
        with pytest.raises(ValidationError, match="either layers or stride"):
            ProbeRunConfig(stride=None)

    def test_an_empty_layer_list_is_refused_rather_than_read_as_absent(self):
        with pytest.raises(ValidationError):
            ProbeRunConfig(layers=[])

    def test_a_negative_layer_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeRunConfig(layers=[-1])

    def test_a_stride_below_one_is_refused(self):
        with pytest.raises(ValidationError):
            ProbeRunConfig(stride=0)


class TestADatasetViewMustBeScoreable:
    def test_a_train_set_needs_both_classes(self):
        with pytest.raises(ValidationError, match="no negative"):
            _dataset(label_mapping={"high": "positive"}, role="train")

    def test_an_eval_set_needs_both_classes(self):
        with pytest.raises(ValidationError, match="no positive"):
            _dataset(label_mapping={"low": "negative"}, role="eval")

    def test_excluded_values_are_allowed_beside_the_two_classes(self):
        """`ambiguous` is a real label in the reference data and must be excludable
        explicitly — and counted, which the model's `counts` column carries."""
        view = _dataset(
            label_mapping={"high": "positive", "low": "negative", "ambiguous": "excluded"}
        )
        assert view.label_mapping["ambiguous"] == "excluded"

    def test_a_calibration_set_must_NOT_carry_a_positive(self):
        """It supplies negatives for the FPR threshold and is not labelled for the
        concept at all. Accepting a positive here means somebody invented ground
        truth for ordinary conversation."""
        with pytest.raises(ValidationError, match="not labelled"):
            _dataset(label_mapping={"a": "positive", "b": "negative"}, role="calibration")

    def test_a_calibration_set_of_negatives_alone_is_fine(self):
        view = _dataset(label_mapping={"chat": "negative"}, role="calibration")
        assert view.role == "calibration"

    def test_distribution_is_only_meaningful_on_an_eval_set(self):
        """It is what rung 2 turns on, so it may not sit on a set rung 2 never reads."""
        with pytest.raises(ValidationError, match="meaningless"):
            _dataset(role="train", distribution="out_of_distribution")
        assert _dataset(role="eval", distribution="out_of_distribution").distribution


class TestTheKeywordFilterCannotLabel:
    def test_it_has_no_field_that_assigns_a_label(self):
        """BR-003 by construction: there is nothing to set. A spec that could say
        "rows containing 'urgent' are positive" would make the AUROC measure the
        keyword."""
        fields = set(KeywordFilterSpec.model_fields)
        assert fields == {"terms", "mode", "case_sensitive"}
        assert not any(
            "label" in name or "class" in name or "target" in name for name in fields
        )

    def test_an_extra_field_is_refused_outright(self):
        with pytest.raises(ValidationError):
            KeywordFilterSpec(terms=["a"], label="positive")

    def test_a_filter_of_blank_terms_is_refused(self):
        with pytest.raises(ValidationError):
            KeywordFilterSpec(terms=["  ", ""])


class TestRunSubmissionRefusesLeakage:
    def test_the_train_set_cannot_also_be_an_eval_set(self):
        with pytest.raises(ValidationError, match="memorisation"):
            ProbeRunCreate(model_id="m", train_dataset_id="d1", eval_dataset_ids=["d2", "d1"])

    def test_the_train_set_cannot_be_the_calibration_set(self):
        with pytest.raises(ValidationError, match="calibrated on training negatives"):
            ProbeRunCreate(model_id="m", train_dataset_id="d1", calibration_dataset_id="d1")

    def test_a_duplicated_eval_set_is_refused(self):
        """Two copies of one set would be counted twice by the rung, which requires
        EVERY out-of-distribution set to clear 0.5."""
        with pytest.raises(ValidationError, match="duplicate"):
            ProbeRunCreate(model_id="m", train_dataset_id="d1", eval_dataset_ids=["d2", "d2"])

    def test_a_legitimate_submission_is_accepted(self):
        run = ProbeRunCreate(
            model_id="m", train_dataset_id="d1", eval_dataset_ids=["d2", "d3"]
        )
        assert run.gpu == "auto"
        assert run.config.stride == DEFAULT_STRIDE


class TestOfflineScoringTakesExactlyOneInput:
    def test_both_is_refused(self):
        """Two inputs need a precedence rule, and a silently ignored field is how a
        caller scores something other than what they sent."""
        with pytest.raises(ValidationError, match="exactly one"):
            ScoreRequest(text="a", messages=[{"role": "user", "content": "b"}])

    def test_neither_is_refused(self):
        with pytest.raises(ValidationError, match="exactly one"):
            ScoreRequest()

    def test_text_alone_is_accepted(self):
        assert ScoreRequest(text="hello").messages is None

    def test_messages_need_role_and_content(self):
        with pytest.raises(ValidationError, match="role and content"):
            ScoreRequest(messages=[{"role": "user"}])

    def test_empty_messages_are_refused(self):
        with pytest.raises(ValidationError):
            ScoreRequest(messages=[])

    def test_over_the_cap_is_REFUSED_not_truncated(self):
        """A truncated input returns a score for text the caller did not submit."""
        with pytest.raises(ValidationError):
            ScoreRequest(text="x" * 16385)

    def test_the_cap_applies_to_messages_in_TOTAL(self):
        """Per-message caps let a caller split 1 MB across 100 turns."""
        with pytest.raises(ValidationError, match="16384"):
            ScoreRequest(
                messages=[{"role": "user", "content": "x" * 9000} for _ in range(2)]
            )

    def test_just_under_the_cap_is_accepted(self):
        assert ScoreRequest(text="x" * 16384)


class TestJudgeRunBounds:
    def test_the_parse_failure_limit_is_a_fraction_in_the_open_interval(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValidationError):
                JudgeRunCreate(endpoint="http://x", model="m", dataset_ids=["d"],
                               parse_failure_limit=bad)

    def test_it_defaults_to_five_percent(self):
        run = JudgeRunCreate(endpoint="http://x", model="m", dataset_ids=["d"])
        assert run.parse_failure_limit == 0.05

    def test_a_judge_run_needs_at_least_one_set(self):
        with pytest.raises(ValidationError):
            JudgeRunCreate(endpoint="http://x", model="m", dataset_ids=[])

    def test_max_rows_cannot_fall_below_the_scoring_floor(self):
        """Fewer than 20 rows per class cannot be scored at all, so a cap below 20
        guarantees a refusal and wastes a judge run."""
        with pytest.raises(ValidationError):
            JudgeRunCreate(endpoint="http://x", model="m", dataset_ids=["d"], max_rows_per_set=5)
