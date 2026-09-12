"""
Unit tests for LabelingContextFormatter.

Tests the context-based example formatting for different template types:
- miStudio Internal (K=10, basic context)
- Anthropic Style (K=50, with logit effects)
- EleutherAI Detection (K=20, for scoring)
"""

import pytest
from src.services.labeling_context_formatter import LabelingContextFormatter


@pytest.fixture
def sample_examples():
    """Create sample activation examples for testing.

    Uses GPT-2 style BPE tokens where 'Ġ' prefix indicates start of a new word.
    This matches how transformer tokenizers actually encode text.
    """
    return [
        {
            # "The dog was running in the park"
            "prefix_tokens": ["The", "Ġdog", "Ġwas"],
            "prime_token": "Ġrunning",
            "suffix_tokens": ["Ġin", "Ġthe", "Ġpark"],
            "max_activation": 8.5,
        },
        {
            # "She started running every morning"
            "prefix_tokens": ["She", "Ġstarted"],
            "prime_token": "Ġrunning",
            "suffix_tokens": ["Ġevery", "Ġmorning"],
            "max_activation": 7.2,
        },
        {
            # "Keep running the server 24/7"
            "prefix_tokens": ["Keep"],
            "prime_token": "Ġrunning",
            "suffix_tokens": ["Ġthe", "Ġserver", "Ġ24", "/", "7"],
            "max_activation": 6.8,
        },
    ]


@pytest.fixture
def sample_logit_effects():
    """Create sample logit effects for testing."""
    return {
        "promoted": ["running", "jogging", "sprinting", "racing", "moving"],
        "suppressed": ["stopped", "still", "paused", "halted", "stationary"],
    }


@pytest.fixture
def basic_template_config():
    """Create basic template configuration for testing."""
    return {
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_prefix": True,
        "include_suffix": True,
        "template_type": "mistudio_context",
        "max_examples": 10,
        "include_logit_effects": False,
    }


@pytest.fixture
def anthropic_template_config():
    """Create Anthropic-style template configuration for testing."""
    return {
        "prime_token_marker": "<< >>",  # Note: space in middle for proper split
        "include_prefix": True,
        "include_suffix": True,
        "template_type": "anthropic_logit",
        "max_examples": 50,
        "include_logit_effects": True,
        "top_promoted_tokens_count": 10,
        "top_suppressed_tokens_count": 10,
    }


class TestFormatMiStudioInternal:
    """Tests for miStudio Internal template formatting."""

    def test_format_basic(self, sample_examples, basic_template_config):
        """Test basic formatting with default settings."""
        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        assert "Example 1" in result
        assert "Example 2" in result
        assert "Example 3" in result
        assert "<<running>>" in result
        assert "(activation: 8.5" in result
        assert "The dog was" in result
        assert "in the park" in result

    def test_format_with_custom_marker(self, sample_examples, basic_template_config):
        """Test formatting with custom prime token marker."""
        custom_config = basic_template_config.copy()
        custom_config["prime_token_marker"] = "**"  # Splits to * and *

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=custom_config,
            feature_id="feat_123"
        )

        # ** is symmetric and splits to * and *, giving us *running*
        assert "*running*" in result
        assert "<<" not in result

    def test_format_without_prefix(self, sample_examples, basic_template_config):
        """Test formatting without prefix tokens."""
        no_prefix_config = basic_template_config.copy()
        no_prefix_config["include_prefix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=no_prefix_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" not in result  # Prefix should be excluded
        assert "in the park" in result  # Suffix should still be included

    def test_format_without_suffix(self, sample_examples, basic_template_config):
        """Test formatting without suffix tokens."""
        no_suffix_config = basic_template_config.copy()
        no_suffix_config["include_suffix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=no_suffix_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" in result  # Prefix should still be included
        assert "in the park" not in result  # Suffix should be excluded

    def test_format_prime_only(self, sample_examples, basic_template_config):
        """Test formatting with only prime tokens."""
        prime_only_config = basic_template_config.copy()
        prime_only_config["include_prefix"] = False
        prime_only_config["include_suffix"] = False

        result = LabelingContextFormatter.format_mistudio_context(
            examples=sample_examples,
            template_config=prime_only_config,
            feature_id="feat_123"
        )

        assert "<<running>>" in result
        assert "The dog was" not in result
        assert "in the park" not in result
        # Should still show example structure
        assert "Example 1" in result

    def test_format_empty_examples(self, basic_template_config):
        """Test formatting with empty examples list."""
        result = LabelingContextFormatter.format_mistudio_context(
            examples=[],
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        # Empty list returns empty string
        assert result == ""

    def test_format_truncates_long_context(self, basic_template_config):
        """Test that very long contexts are handled properly."""
        long_examples = [
            {
                "prefix_tokens": ["Ġword"] * 100,  # Very long prefix with BPE markers
                "prime_token": "Ġtest",
                "suffix_tokens": ["Ġword"] * 100,  # Very long suffix with BPE markers
                "max_activation": 5.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=long_examples,
            template_config=basic_template_config,
            feature_id="feat_123"
        )

        # Should handle long contexts (may or may not truncate at formatter level)
        assert "<<test>>" in result
        assert "Example 1" in result


class TestFormatAnthropicStyle:
    """Tests for Anthropic Style template formatting with logit effects."""

    def test_format_with_logit_effects(self, sample_examples, sample_logit_effects, anthropic_template_config):
        """Test formatting with logit effects included."""
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=sample_logit_effects,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "<<running>>" in result
        assert "TOP ACTIVATING EXAMPLES" in result
        assert "LOGIT EFFECTS" in result
        assert "Top promoted tokens" in result
        assert "Top suppressed tokens" in result
        assert "running" in result
        assert "jogging" in result
        assert "stopped" in result

    def test_format_without_logit_effects(self, sample_examples, anthropic_template_config):
        """Test formatting without logit effects."""
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=None,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "<<running>>" in result
        assert "TOP ACTIVATING EXAMPLES" in result
        assert "LOGIT EFFECTS" in result
        # Should show "No logit effects available" message
        assert "No logit effects available" in result

    def test_format_with_partial_logit_effects(self, sample_examples, anthropic_template_config):
        """Test formatting with only some logit effects."""
        partial_effects = {
            "promoted": ["running", "jogging"],
            "suppressed": []
        }

        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=partial_effects,
            template_config=anthropic_template_config,
            feature_id="feat_456"
        )

        assert "running" in result
        assert "jogging" in result
        assert "Top promoted tokens" in result


class TestStrengthIsStatedHonestly:
    """How an example's activation is described to the judge.

    Two changes, both consequences of widening the range of examples shown.

    1. `{:.3f}` collapses. This module's own docstring shows two DISTINCT
       examples both printing "activation: 0.007". While every example came
       from the top of the distribution that was harmless; a stratified span
       whose weakest member renders identically to its strongest tells the
       judge nothing changed.

    2. The weak block was labelled "NEGATIVE EXAMPLES (Low Activation)" and
       described as text where the feature "does NOT activate strongly". Those
       rows are the feature's OWN stored activations taken from the bottom of
       what was retained — weak POSITIVES. On L46 the smallest stored
       activation is 0.67, strictly positive, and there is no encode-on-text
       service that could certify any passage as non-activating.

       Nothing asserted the old wording, so the rename broke no test. That is
       the finding: a claim made to the judge on every contrastive prompt was
       uncovered.

    MUTATION CONTROLS:
      C72 restore the unconditional `{:.3f}`
           -> test_percent_display_separates_what_3dp_flattens
      C73 normalise the weak block against its own maximum
           -> test_weak_examples_normalise_against_the_positives_peak
      C74 restore "does NOT activate" framing
           -> test_weak_examples_are_never_called_non_activating
    """

    @staticmethod
    def _example(activation, prime="running"):
        return {
            "prefix_tokens": ["the", "server", "was"],
            "prime_token": prime,
            "suffix_tokens": ["fast"],
            "max_activation": activation,
        }

    @staticmethod
    def _config(display):
        return {
            "template_type": "mistudio_context",
            "prime_token_marker": "<<>>",
            "include_prefix": True,
            "include_suffix": True,
            "activation_display": display,
        }

    def test_percent_display_separates_what_3dp_flattens(self):
        """C72. The docstring's own 0.007 pair must become distinguishable."""
        examples = [
            self._example(6.800, "alpha"),
            self._example(0.0074, "beta"),
            self._example(0.0069, "gamma"),
        ]

        absolute = LabelingContextFormatter.format_mistudio_context(
            examples=examples, template_config=self._config("absolute"),
            feature_id="f1",
        )
        # The defect, pinned: at 3dp the two weak examples read the same.
        assert absolute.count("activation: 0.007") == 2

        percent = LabelingContextFormatter.format_mistudio_context(
            examples=examples, template_config=self._config("percent_of_max"),
            feature_id="f1",
        )
        assert "100% of this feature's peak" in percent
        assert "activation: 0.007" not in percent

    def test_absolute_display_is_unchanged(self):
        """Negative control: existing templates must render exactly as before.

        `absolute` is the default, so a change here would silently alter every
        prompt in the estate and invalidate comparison with past runs.
        """
        out = LabelingContextFormatter.format_mistudio_context(
            examples=[self._example(6.8)],
            template_config=self._config("absolute"), feature_id="f1",
        )
        assert "(activation: 6.800)" in out
        assert "peak" not in out

    def test_a_zero_peak_falls_back_to_absolute(self):
        """Division by zero must not take down a labeling run.

        A feature whose stored examples are all zero should not exist — the
        heap drops activation <= 0 — but a formatter that divides by an
        untested peak is one bad row away from failing every feature in a job.
        """
        out = LabelingContextFormatter.format_mistudio_context(
            examples=[self._example(0.0)],
            template_config=self._config("percent_of_max"), feature_id="f1",
        )
        assert "activation: 0.000" in out

    def test_weak_examples_normalise_against_the_positives_peak(self):
        """C73. The weak block must not renormalise to its own maximum.

        Doing so renders its strongest member as "100% of peak", which is the
        exact opposite of what the block exists to convey.
        """
        out = LabelingContextFormatter.format_mistudio_context(
            examples=[self._example(10.0, "strong")],
            template_config=self._config("percent_of_max"),
            feature_id="f1",
            negative_examples=[self._example(1.0, "weak")],
        )
        assert "10% of this feature's peak" in out, (
            "the weak example was not scaled against the positives' peak"
        )
        assert out.count("100% of this feature's peak") == 1

    def test_weak_examples_are_never_called_non_activating(self):
        """C74. These rows activate the feature. Saying otherwise is false."""
        out = LabelingContextFormatter.format_mistudio_context(
            examples=[self._example(6.8, "strong")],
            template_config=self._config("absolute"),
            feature_id="f1",
            negative_examples=[self._example(0.7, "weak")],
        )

        assert "NEGATIVE EXAMPLES" not in out
        assert "does NOT" not in out
        assert "WEAKER EXAMPLES" in out
        assert "also activate the feature" in out
        assert "NOT non-activating text" in out


class TestLogitEffectsUseTheProducersKeys:
    """The reader asked for a key no producer ever writes.

    `format_anthropic_logit` read `logit_effects['promoted']` while EVERY
    producer writes `top_promoted` — labeling_service.py:1598-1601, and the same
    block duplicated at :1798-1801 for the OPENAI_COMPATIBLE branch. So the
    Anthropic template's logit section has always rendered
    "No logit effects available." in production.

    The quiet kind of wrong: the section still renders, so nothing looks broken.

    Note the fixtures above build `logit_effects` with the LEGACY key, which is
    why four green tests never caught this — they agree with the reader by
    construction and say nothing about the producer. These tests use the shape
    the producer actually emits.

    MUTATION CONTROL C57: drop the `top_promoted` lookup, leaving only
    `.get('promoted')` -> both tests below fail.
    """

    @staticmethod
    def _producer_shape():
        """Exactly what labeling_service.py:1798-1801 builds."""
        return {
            "top_promoted": ["running", "jogging", "sprinting"],
            "top_suppressed": ["stopped", "still", "halted"],
        }

    def test_the_producers_keys_reach_the_prompt(
        self, sample_examples, anthropic_template_config
    ):
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects=self._producer_shape(),
            template_config=anthropic_template_config,
            feature_id="feat_456",
        )

        assert "No logit effects available" not in result, (
            "the producer's logit effects were dropped — this is what "
            "production has been rendering"
        )
        assert "Top promoted tokens" in result
        assert "Top suppressed tokens" in result
        assert "sprinting" in result
        assert "halted" in result

    def test_the_legacy_key_still_works(
        self, sample_examples, anthropic_template_config
    ):
        """Negative control: accepting the new key must not drop the old one.

        Templates and fixtures in the wild carry `promoted`. A strict rename
        would have reddened four existing tests while proving nothing about the
        producer, so both shapes are accepted.
        """
        result = LabelingContextFormatter.format_anthropic_logit(
            examples=sample_examples,
            logit_effects={"promoted": ["alpha"], "suppressed": ["omega"]},
            template_config=anthropic_template_config,
            feature_id="feat_456",
        )

        assert "alpha" in result and "omega" in result
        assert "No logit effects available" not in result


class TestFormatEleutherAIDetection:
    """Tests for EleutherAI Detection template formatting."""

    def test_format_detection_basic(self):
        """Test basic detection template formatting."""
        feature_explanation = {
            "name": "continuous_actions",
            "category": "semantic",
            "description": "This feature detects continuous actions or processes."
        }
        test_examples = [
            "The dog was running in the park",
            "She started running every morning",
            "Keep running the server 24/7"
        ]

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=test_examples
        )

        assert "1. The dog was running in the park" in result
        assert "2. She started running every morning" in result
        assert "3. Keep running the server 24/7" in result

    def test_format_detection_empty_examples(self):
        """Test detection format with empty examples list."""
        feature_explanation = {
            "name": "test_feature",
            "category": "semantic",
            "description": "Test"
        }

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=[]
        )

        # Empty list returns empty string
        assert result == ""

    def test_format_detection_single_example(self):
        """Test detection format with single example."""
        feature_explanation = {
            "name": "test_feature",
            "category": "semantic",
            "description": "Test"
        }
        test_examples = ["Single example text"]

        result = LabelingContextFormatter.format_eleutherai_detection(
            feature_explanation=feature_explanation,
            test_examples=test_examples
        )

        assert "1. Single example text" in result
        assert "2." not in result


class TestFormatEdgeCases:
    """Tests for edge cases and error handling."""

    def test_format_with_special_characters(self, basic_template_config):
        """Test formatting with special characters in tokens."""
        examples = [
            {
                "prefix_tokens": ["Test", "with", "\"quotes\""],
                "prime_token": "<special>",
                "suffix_tokens": ["and", "&", "symbols"],
                "max_activation": 5.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_special"
        )

        # Should handle special characters gracefully
        assert "<<" in result
        assert "&" in result or "symbols" in result

    def test_format_with_missing_token_fields(self, basic_template_config):
        """Test formatting when some token fields are missing."""
        examples = [
            {
                "prime_token": "test",
                "max_activation": 5.0,
                # Missing prefix_tokens and suffix_tokens
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_partial"
        )

        # Should still format with available data
        assert "<<test>>" in result

    def test_format_with_zero_activation(self, basic_template_config):
        """Test formatting with zero activation value."""
        examples = [
            {
                "prefix_tokens": ["Ġno"],
                "prime_token": "Ġactivation",
                "suffix_tokens": ["Ġhere"],
                "max_activation": 0.0,
            }
        ]

        result = LabelingContextFormatter.format_mistudio_context(
            examples=examples,
            template_config=basic_template_config,
            feature_id="feat_zero"
        )

        assert "(activation: 0.0" in result
        assert "<<activation>>" in result


class TestTheContrastBlockIsNeverAboveThePositives:
    """"Lower activation" must be true of every passage under that heading.

    WHY THIS EXISTS
    ---------------
    The weak examples are the feature's lowest RETAINED activations, chosen
    without reference to which rows the prompt is showing. Under `top_k` that is
    safe — positives are ranks 1-10, the weak block is the bottom five.

    Under `stratified` on a 20-row extraction the positives are ranks
    1,3,5…19 and the weak block is 20,18,16,14,12. INTERLEAVED. Measured on the
    eb48 shape: four of five passages captioned "same feature, lower activation"
    are strictly stronger than four of the ten presented as positives, and under
    `percent_of_max` the prompt prints positives ending at 82% above a "weaker"
    block leading with 89%.

    The commit that introduced this block was titled "the contrast block reaches
    the judge, and says something true". On any extraction where the sampled
    positives reach far enough down the stored set, it did not.

    A silent, empty contrast block is honest. A false one is not.

    MUTATION CONTROL:
      C107 remove the ordering check
            -> test_an_interleaved_block_is_dropped
    """

    @staticmethod
    def _ex(activation, prime):
        return {
            "prefix_tokens": ["the"], "prime_token": prime,
            "suffix_tokens": ["ran"], "max_activation": activation,
        }

    @staticmethod
    def _config(display="absolute"):
        return {
            "template_type": "mistudio_context",
            "prime_token_marker": "<<>>",
            "include_prefix": True,
            "include_suffix": True,
            "activation_display": display,
        }

    def test_an_interleaved_block_is_dropped(self):
        """C107. The eb48 shape, with the real numbers.

        Positives at stratified ranks 1,11,13,15,17,19 of a 20-row feature;
        weak block at 20,18,16,14,12. Four of the five 'weaker' rows outrank
        four of the shown ones.
        """
        positives = [self._ex(a, f"p{a}") for a in
                     (0.990, 0.890, 0.870, 0.850, 0.830, 0.810)]
        weak = [self._ex(a, f"w{a}") for a in
                (0.880, 0.860, 0.840, 0.820, 0.800)]

        out = LabelingContextFormatter.format_mistudio_context(
            examples=positives, template_config=self._config(),
            feature_id="feat_eb48", negative_examples=weak,
        )

        assert "WEAKER EXAMPLES" not in out, (
            "a contrast block was rendered whose strongest member (0.880) sits "
            "above four of the shown positives — the caption is false"
        )
        for prime in ("w0.88", "w0.86", "w0.84"):
            assert prime not in out

    def test_a_genuinely_lower_block_is_kept(self):
        """Negative control: the guard must not delete every contrast block.

        A check that dropped the block unconditionally passes the test above
        and silently removes the feature this arc wired.
        """
        positives = [self._ex(a, f"p{a}") for a in (0.99, 0.95, 0.91)]
        weak = [self._ex(a, f"w{a}") for a in (0.30, 0.25, 0.20)]

        out = LabelingContextFormatter.format_mistudio_context(
            examples=positives, template_config=self._config(),
            feature_id="feat_l45", negative_examples=weak,
        )

        assert "WEAKER EXAMPLES" in out
        assert "w0.3" in out

    def test_the_boundary_is_strict(self):
        """Equal activations are not 'lower'.

        A weak example tying the weakest shown one carries no contrast and the
        caption still claims one.
        """
        positives = [self._ex(0.90, "p"), self._ex(0.50, "q")]
        weak = [self._ex(0.50, "w")]

        out = LabelingContextFormatter.format_mistudio_context(
            examples=positives, template_config=self._config(),
            feature_id="f", negative_examples=weak,
        )
        assert "WEAKER EXAMPLES" not in out

    def test_percent_display_cannot_print_a_weaker_row_higher(self):
        """The rendered numbers must not contradict the heading.

        Under percent_of_max the falsehood becomes visible in the prompt: a
        block captioned 'lower activation' printing a higher percentage than
        rows above it.
        """
        positives = [self._ex(a, f"p{a}") for a in (1.00, 0.82)]
        weak = [self._ex(0.89, "w")]

        out = LabelingContextFormatter.format_mistudio_context(
            examples=positives, template_config=self._config("percent_of_max"),
            feature_id="f", negative_examples=weak,
        )
        assert "89%" not in out
        assert "WEAKER EXAMPLES" not in out
