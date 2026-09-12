"""
Unit tests for token filtering utilities.
"""

import pytest
from src.utils.token_filters import (
    is_junk_token,
    clean_token_display,
    analyze_feature_tokens
)


class TestIsJunkToken:
    """Tests for is_junk_token function."""

    def test_special_tokens(self):
        """Special tokens should be filtered."""
        assert is_junk_token('<s>') is True
        assert is_junk_token('</s>') is True
        assert is_junk_token('<pad>') is True
        assert is_junk_token('<unk>') is True
        assert is_junk_token('\ufeff') is True

    def test_single_characters(self):
        """Single characters should be filtered."""
        assert is_junk_token('a') is True
        assert is_junk_token('Z') is True
        assert is_junk_token('1') is True
        assert is_junk_token(',') is True
        assert is_junk_token(' ') is True

    def test_pure_punctuation(self):
        """Pure punctuation should be filtered."""
        assert is_junk_token(',') is True
        assert is_junk_token('.,!') is True
        assert is_junk_token('---') is True
        assert is_junk_token('▁') is True

    def test_short_fragments_without_vowels(self):
        """Short fragments without vowels should be filtered."""
        assert is_junk_token('th') is True
        assert is_junk_token('str') is True
        assert is_junk_token('by') is True
        assert is_junk_token('ps') is True

    def test_pure_numbers(self):
        """Pure numbers should be filtered."""
        assert is_junk_token('123') is True
        assert is_junk_token('2024') is True

    def test_valid_tokens(self):
        """Valid tokens should not be filtered."""
        assert is_junk_token('The') is False
        assert is_junk_token('about') is False
        assert is_junk_token('news') is False
        assert is_junk_token('Pennsylvania') is False
        # Word fragment filtering can be disabled
        assert is_junk_token('ious', filter_fragments=False) is False
        assert is_junk_token('ious', filter_fragments=True) is True

    def test_short_tokens_with_vowels(self):
        """Short tokens with vowels should not be filtered."""
        assert is_junk_token('he') is False
        assert is_junk_token('in') is False
        assert is_junk_token('on') is False
        assert is_junk_token('are') is False


class TestCleanTokenDisplay:
    """Tests for clean_token_display function."""

    def test_removes_space_marker(self):
        """Should remove leading space marker."""
        assert clean_token_display('▁Hello') == 'Hello'
        assert clean_token_display('▁The') == 'The'
        assert clean_token_display('▁news') == 'news'

    def test_preserves_token_if_empty(self):
        """Should preserve original if empty after cleaning."""
        assert clean_token_display('▁') == '▁'

    def test_no_space_marker(self):
        """Should preserve tokens without space marker."""
        assert clean_token_display('world') == 'world'
        assert clean_token_display('123') == '123'

    def test_multiple_space_markers(self):
        """Should remove all space markers."""
        assert clean_token_display('▁▁test') == 'test'


class TestAnalyzeFeatureTokens:
    """Tests for analyze_feature_tokens function."""

    def test_basic_analysis(self):
        """Should count tokens correctly."""
        tokens_list = [
            ['The', 'cat'],
            ['The', 'dog'],
            ['A', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['summary']['total_examples'] == 3
        assert result['summary']['original_token_count'] == 5  # The, cat, dog, A, bird
        assert result['summary']['total_token_occurrences'] == 6

    def test_filtering(self):
        """Should filter junk tokens."""
        tokens_list = [
            ['<s>', 'The', 'cat', ','],
            ['</s>', 'The', 'dog', '1'],
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=True)

        # Should remove: <s>, </s>, ,, 1 (4 junk tokens)
        assert result['summary']['junk_removed'] == 4
        # Should keep: The (2x), cat, dog (3 unique)
        assert result['summary']['filtered_token_count'] == 3

    def test_sorting(self):
        """Should sort by count descending, then alphabetically."""
        tokens_list = [
            ['The', 'cat'],
            ['The', 'dog'],
            ['The', 'bird'],
            ['cat', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        # The: 3, bird: 2, cat: 2, dog: 1
        # bird and cat have same count, should be sorted alphabetically
        assert result['tokens'][0]['token'] == 'The'
        assert result['tokens'][0]['count'] == 3
        assert result['tokens'][1]['token'] == 'bird'
        assert result['tokens'][1]['count'] == 2
        assert result['tokens'][2]['token'] == 'cat'
        assert result['tokens'][2]['count'] == 2

    def test_percentage_calculation(self):
        """Should calculate percentages correctly."""
        tokens_list = [
            ['The', 'The'],  # The: 50%
            ['cat', 'dog']   # cat: 25%, dog: 25%
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['tokens'][0]['percentage'] == 50.0
        assert result['tokens'][1]['percentage'] == 25.0
        assert result['tokens'][2]['percentage'] == 25.0

    def test_rank_assignment(self):
        """Should assign ranks correctly."""
        tokens_list = [
            ['A', 'B', 'C']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        assert result['tokens'][0]['rank'] == 1
        assert result['tokens'][1]['rank'] == 2
        assert result['tokens'][2]['rank'] == 3

    def test_empty_input(self):
        """Should handle empty input."""
        result = analyze_feature_tokens([], apply_filters=True)

        assert result['summary']['total_examples'] == 0
        assert result['summary']['original_token_count'] == 0
        assert result['tokens'] == []

    def test_diversity_calculation(self):
        """Should calculate diversity percentage."""
        # 3 unique tokens out of 6 total = 50% diversity
        tokens_list = [
            ['The', 'The', 'cat'],
            ['The', 'dog', 'bird']
        ]
        result = analyze_feature_tokens(tokens_list, apply_filters=False)

        # 4 unique / 6 total = 66.67%
        assert result['summary']['diversity_percent'] == 66.67


class TestTheSkipDecisionDoesNotDependOnTheDraw:
    """A verdict that permanently removes a feature must be a property of the
    FEATURE, not of which examples happened to be sampled.

    WHY THIS EXISTS
    ---------------
    Four individually-defensible lines composed into irreversible data loss:

      1. `labeling_service` shuffles the examples (to break primacy bias in the
         PROMPT) using the unseeded global RNG.
      2. `is_junk_feature_from_examples` judged `prime_tokens[:10]` — the first
         ten of that shuffled list.
      3. A positive verdict writes `label_status='skipped'`.
      4. `skipped` is in `ADJUDICATED_STATUSES` (never redone) and deliberately
         NOT in `STALEABLE_STATUSES`, on the reasoning that a skip "has no
         fingerprint because no judge was ever asked".

    That reasoning was false: the skip IS computed from the sampled examples.
    So a feature near the threshold was a coin flip whose loser was deleted from
    the labelable estate for good, carrying an error message that states a
    definite fact. At the 0.8 threshold over 10 drawn from 25, a feature whose
    true junk ratio is 0.76 had a ~54% chance of being skipped; anything from
    roughly 0.56 to 0.88 was materially uncertain. Only hand SQL recovers it.

    It also coupled the arc's `example_sampling` switch to which features exist:
    stratified sampling surfaces weaker activations, which peak on affixes more
    often, so turning the feature on would silently skip more features.

    MUTATION CONTROLS:
      C102 judge `prime_tokens[:10]` again
            -> test_the_verdict_is_the_same_for_every_ordering
      C103 divide by `min(len(prime_tokens), 10)` while counting over all
            -> test_a_ratio_can_never_exceed_one
    """

    @staticmethod
    def _filter():
        from src.utils.token_filter import FeatureFilter

        return FeatureFilter(
            junk_ratio_threshold=0.8, min_tokens_for_decision=5,
        )

    @staticmethod
    def _examples(primes):
        return [{"prime_token": t} for t in primes]

    def test_the_verdict_is_the_same_for_every_ordering(self):
        """C102. 25 examples, 16 junk — a 0.64 ratio, safely below 0.8.

        Under `[:10]` the drawn ratio ranges from 0.1 to 1.0 depending on the
        shuffle, so SOME orderings cross the threshold and delete the feature.
        Judging all 25 makes the answer the feature's own.
        """
        import itertools
        import random

        primes = ["." ] * 16 + ["server", "running", "database", "python",
                                "kernel", "matrix", "tensor", "gradient",
                                "cluster"]
        assert len(primes) == 25

        f = self._filter()
        verdicts = set()
        rng = random.Random(20260910)
        for _ in range(200):
            shuffled = primes[:]
            rng.shuffle(shuffled)
            verdicts.add(f.is_junk_feature_from_examples(self._examples(shuffled)))

        assert verdicts == {False}, (
            "the skip verdict changed with the ordering of the examples — a "
            "coin flip that permanently removes a feature from the estate"
        )

        # POSITIVE CONTROL, over every slice length a regression might use.
        #
        # Checking only a 10-window meant re-slicing at `[:20]` — which restores
        # the coin flip at max_examples=50 AND reintroduces a denominator
        # mismatch in the flattering direction — left the suite green. The
        # fixture must be able to exhibit the defect at each length, or the
        # ordering assertion above is vacuous there.
        for width in (10, 15, 20):
            windows = [primes[i:i + width] for i in range(0, 26 - width)]
            above = [w for w in windows if w.count(".") / width >= 0.8]
            assert above, (
                f"no {width}-token window of this fixture crosses the "
                f"threshold, so re-slicing at {width} would not be detected"
            )

    def test_the_verdict_reads_every_retained_example(self):
        """Pins the INVARIANT, not one spelling of the slice.

        `[:10]` was the original defect and `[:20]` is an equally green
        regression. The property is that the count and the denominator cover the
        SAME tokens, and that they cover ALL of them.
        """
        import ast
        import inspect
        import textwrap

        from src.utils.token_filter import FeatureFilter

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(FeatureFilter.is_junk_feature_from_examples)
        ))

        loops = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.For) and "prime_tokens" in ast.dump(node.iter)
        ]
        assert loops, "no loop over prime_tokens found; this test is inert"

        for loop in loops:
            assert not isinstance(loop.iter, ast.Subscript), (
                "the verdict reads a SLICE of the examples again. The caller "
                "shuffles them, so a slice is a random draw — and this decision "
                "writes `skipped`, which is never redone."
            )

        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "total_analyzed"
                for t in node.targets
            )
        ]
        assert assigns, "total_analyzed is not assigned; this test is inert"
        for node in assigns:
            dumped = ast.dump(node.value)
            assert "min" not in dumped, (
                "the denominator is capped while the loop counts every token, "
                "so the ratio can exceed 1.0 and clears every threshold"
            )

    def test_a_genuinely_junk_feature_is_still_skipped(self):
        """Negative control: a filter that never skips passes the test above."""
        f = self._filter()
        assert f.is_junk_feature_from_examples(
            self._examples(["."] * 20 + ["server", "kernel"])
        ) is True

    def test_a_clean_feature_is_never_skipped(self):
        f = self._filter()
        assert f.is_junk_feature_from_examples(
            self._examples(["server", "running", "kernel", "matrix", "tensor"])
        ) is False

    def test_a_ratio_can_never_exceed_one(self):
        """C103. Counting over N and dividing by 10 clears any threshold.

        With 25 examples of which 16 are junk, a mismatched denominator gives
        16/10 = 1.6 — above every possible threshold, so EVERY feature with
        more than ten junk primes would be skipped regardless of its ratio.
        """
        f = self._filter()
        primes = ["."] * 16 + ["server"] * 9
        # 0.64 < 0.8, so the honest answer is False. A ratio above 1.0 cannot
        # produce that.
        assert f.is_junk_feature_from_examples(self._examples(primes)) is False

    def test_too_few_examples_is_conservative(self):
        """Below `min_tokens_for_decision` the answer must be "do not skip".

        A decision this destructive must not be taken on thin evidence.
        """
        f = self._filter()
        assert f.is_junk_feature_from_examples(self._examples(["."] * 3)) is False


class TestTheJunkVerdictIsPinned:
    """`example_sampling` must not decide which features are permanently retired.

    WHY THIS EXISTS
    ---------------
    `filter_features_from_examples` was handed the DISPLAY set — the rows chosen
    by the template's `example_sampling` — and judged junkiness on those. Under
    `stratified` those are ranks 1,11,21…91; under `top_k`, ranks 1-10. Different
    sets, different verdicts, from a template setting.

    That would be survivable if the decision were revisable. It is not:
    `_persist_filtered_out` writes `label_status='skipped'`, which is in
    ADJUDICATED_STATUSES (never redone) and carries no fingerprint, so the
    staleness path cannot reach it either. Flipping the default template to
    stratified would have permanently retired features whose MID-RANGE primes
    are punctuation-heavy — common for register features, which fire strongly on
    a word and weakly at sentence boundaries — with an error message asserting a
    definite fact about the feature.

    MUTATION CONTROLS:
      C108 fall back to `features_examples` for the verdict
            -> test_the_verdict_ignores_what_the_prompt_shows
      C109 drop the length check on verdict_examples
            -> test_a_mismatched_verdict_list_is_refused
    """

    @staticmethod
    def _filter():
        from src.utils.token_filter import FeatureFilter

        return FeatureFilter(
            junk_ratio_threshold=0.8, min_tokens_for_decision=5,
        )

    @staticmethod
    def _ex(primes):
        return [{"prime_token": t} for t in primes]

    class _F:
        def __init__(self, fid):
            self.id = fid

    def test_the_verdict_ignores_what_the_prompt_shows(self):
        """C108. Same feature, two display sets, one verdict.

        The display sets are deliberately opposite: one all punctuation, one all
        words. If the verdict followed them, the two calls would disagree.
        """
        f = self._filter()
        feature = [self._F("feat_1")]
        clean_verdict = [self._ex(["server", "running", "kernel", "matrix", "tensor"])]

        junk_display = [self._ex(["."] * 10)]
        word_display = [self._ex(["server"] * 10)]

        kept_a, _, _, stats_a = f.filter_features_from_examples(
            feature, junk_display, [[]], verdict_examples=clean_verdict)
        kept_b, _, _, stats_b = f.filter_features_from_examples(
            feature, word_display, [[]], verdict_examples=clean_verdict)

        assert stats_a["features_skipped"] == stats_b["features_skipped"] == 0, (
            "the junk verdict changed with the DISPLAY set, so a template's "
            "sampling strategy decides which features are permanently retired"
        )
        assert len(kept_a) == len(kept_b) == 1

    def test_a_junk_feature_is_still_skipped_on_its_pinned_evidence(self):
        """Negative control: the filter must not become inert.

        A verdict that always says "keep" passes the test above and lets every
        punctuation feature through to the judge.
        """
        f = self._filter()
        kept, _, _, stats = f.filter_features_from_examples(
            [self._F("feat_junk")],
            [self._ex(["server"] * 10)],          # display looks fine
            [[]],
            verdict_examples=[self._ex(["."] * 10)],  # evidence says junk
        )
        assert stats["features_skipped"] == 1
        assert kept == []

    def test_a_mismatched_verdict_list_is_refused(self):
        """C109. A positional zip must not judge one feature on another's tokens."""
        f = self._filter()
        with pytest.raises(ValueError, match="positional mismatch"):
            f.filter_features_from_examples(
                [self._F("a"), self._F("b")],
                [self._ex(["x"] * 5), self._ex(["y"] * 5)],
                [[], []],
                verdict_examples=[self._ex(["x"] * 5)],  # one short
            )

    def test_the_service_takes_the_verdict_from_a_pinned_retrieval(self):
        """The wiring: the bulk path must retrieve the verdict set unsampled.

        Bound to the call's arguments, with a self-check, because the property
        is about WHICH retrieval feeds the filter.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_service import LabelingService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingService.label_features_for_extraction)
        ))

        by_target = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_retrieve_top_examples_batch_sync"
            ):
                continue
            target = next(
                (t.id for t in node.targets if isinstance(t, ast.Name)), None
            )
            by_target[target] = {kw.arg: ast.dump(kw.value) for kw in call.keywords}

        assert "junk_verdict_map" in by_target, (
            "the junk verdict is not taken from its own retrieval; if it reads "
            "the display set, a template setting decides which features are "
            "permanently retired"
        )
        assert "None" in by_target["junk_verdict_map"].get("sampling", ""), (
            "the junk verdict retrieval carries a sampling strategy, so the "
            "verdict moves with the arm"
        )
        # AND ITS SIZE MUST BE THE CONSTANT, NOT THE TEMPLATE'S.
        #
        # Pinning `sampling=None` closes the example_sampling axis and leaves
        # the max_examples one wide open: the seeded templates carry 10, 25 and
        # 50, so a verdict sized by the template retires under one what it keeps
        # under another — and `skipped` is never redone.
        size = by_target["junk_verdict_map"].get("max_examples", "")
        assert "JUNK_VERDICT_EXAMPLES" in size, (
            f"the junk verdict is sized by {size!r} rather than the fixed "
            f"constant, so a template setting still decides which features are "
            f"permanently retired"
        )
        assert "max_examples'" not in size and "id='max_examples'" not in size, (
            "the junk verdict reads the template's max_examples"
        )

        # And the filter must actually be handed it.
        filter_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "filter_features_from_examples"
        ]
        assert filter_calls, "no junk filter call found; this test is inert"
        for call in filter_calls:
            kwargs = {kw.arg: ast.dump(kw.value) for kw in call.keywords}
            assert "verdict_examples" in kwargs, (
                "the filter is not given a pinned verdict set, so it falls back "
                "to the display set"
            )
            assert "junk_verdict_map" in kwargs["verdict_examples"]
