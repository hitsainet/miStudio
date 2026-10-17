"""The contrast block must actually reach the judge.

WHY THIS EXISTS
---------------
`_retrieve_bottom_examples_batch` and its sync twin were fully written, fully
tested in isolation, and had ZERO callers. `negative_examples` was `None` at
every call site in `openai_labeling_service`, so
`format_mistudio_context`'s contrast block had never rendered in production —
while `include_negative_examples` is True on every template in the estate.

An inert switch that reports itself as on. This repo's signature failure.

Wiring it also exposed two things the mechanism got wrong:

  * the bottom-K query did not exclude the rows already being shown, so a
    feature with fewer than K+N stored rows presented the SAME passage as both
    a strong and a weak example in one prompt. Every feature in the eb48
    extraction stores exactly 20 rows, so at K=10/N=5 that is the common case,
    not an edge case.
  * `num_negative_examples` is NULL on three of five live templates including
    the default, so resolving it with `or 0` would have kept the switch inert
    on the one template everything uses.

MUTATION CONTROLS:
  C75 drop negative_examples from the serial _build_user_prompt call
       -> test_the_serial_path_renders_the_contrast_block
  C76 drop it from _resolve_user_prompt (the batched path)
       -> test_batched_and_serial_agree
  C77 resolve a NULL num_negative_examples to 0
       -> test_null_means_five_not_zero
  C78 drop the NOT EXISTS exclusion from the SQL
       -> test_a_passage_is_never_both_strong_and_weak
  C79 let batch_negatives mismatch the feature list
       -> test_a_short_negatives_list_is_refused
"""

import pytest
from sqlalchemy import text
from sqlalchemy.pool import NullPool

from src.core.config import settings
from src.services.labeling_service import (
    DEFAULT_NUM_NEGATIVE_EXAMPLES,
    LabelingService,
    WEAKEST_EXAMPLES_SQL,
    resolve_num_negative,
)
from src.services.openai_labeling_service import OpenAILabelingService

CONFIG = {
    "template_type": "mistudio_context",
    "prime_token_marker": "<<>>",
    "include_prefix": True,
    "include_suffix": True,
    "activation_display": "absolute",
}


def _ex(activation, prime):
    return {
        "prefix_tokens": ["the"], "prime_token": prime,
        "suffix_tokens": ["ran"], "max_activation": activation,
        "sample_index": int(activation * 100),
    }


class TestResolveNumNegative:
    def test_null_means_five_not_zero(self):
        """C77. NULL is "the documented default", not "none".

        Three of five live templates leave this NULL, including the default, so
        `or 0` would keep the feature switched off exactly where it matters.
        """
        assert resolve_num_negative({
            "include_negative_examples": True, "num_negative_examples": None,
        }) == DEFAULT_NUM_NEGATIVE_EXAMPLES == 5

    def test_the_switch_still_turns_it_off(self):
        """Negative control: a resolver that always returns 5 passes above."""
        assert resolve_num_negative({
            "include_negative_examples": False, "num_negative_examples": 5,
        }) == 0
        assert resolve_num_negative({}) == 0

    def test_an_explicit_count_is_honoured(self):
        assert resolve_num_negative({
            "include_negative_examples": True, "num_negative_examples": 3,
        }) == 3


class TestThePromptCarriesTheContrast:
    @staticmethod
    def _svc():
        return OpenAILabelingService.__new__(OpenAILabelingService)

    @pytest.mark.asyncio
    async def test_the_serial_path_renders_the_contrast_block(self, monkeypatch):
        """C75. Through the REAL entry point, capturing what the judge is sent.

        The first version of this test called `_build_user_prompt` directly and
        SURVIVED the mutation that removes `negative_examples=` from
        `generate_label_from_examples`'s call to it — because the mutated line
        was never executed. That is precisely the "every test passed by
        importing the module directly" failure this repo has recorded, and it
        is why this now drives the public method and reads the outgoing
        messages.
        """
        svc = OpenAILabelingService.__new__(OpenAILabelingService)
        svc.model = "m"
        svc.temperature = 0.0
        svc.max_tokens = 64
        svc.top_p = 1.0
        svc.save_requests_for_testing = False
        svc.chat_template_kwargs = None

        sent = {}

        class _Msg:
            content = '{"category": "semantic", "specific": "x", "description": "d"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        async def _fake_call(messages, **_kw):
            sent["messages"] = messages
            return _Resp()

        monkeypatch.setattr(svc, "_call_openai", _fake_call, raising=False)

        await svc.generate_label_from_examples(
            examples=[_ex(6.8, "strong")],
            template_config=CONFIG,
            user_prompt_template="Look:\n{examples_block}",
            system_message="sys",
            feature_id="f1",
            negative_examples=[_ex(0.7, "faint")],
        )

        user_prompt = next(
            m["content"] for m in sent["messages"] if m["role"] == "user"
        )
        assert "WEAKER EXAMPLES" in user_prompt
        assert "faint" in user_prompt, (
            "the weak example never reached the prompt the judge was sent"
        )
        assert "strong" in user_prompt

    def test_batched_and_serial_agree(self):
        """C76. The two prompt paths must not diverge.

        `_resolve_user_prompt`'s own docstring says it is shared so a batched
        label is built from the same prompt as a serial one. A parameter
        threaded into one and not the other silently breaks that promise, and
        the difference would show up as an unexplained batch-size effect.
        """
        svc = self._svc()
        kwargs = dict(
            examples=[_ex(6.8, "strong")],
            template_config=CONFIG,
            user_prompt_template="Look:\n{examples_block}",
            feature_id="f1",
        )
        negatives = [_ex(0.7, "faint")]

        serial = svc._build_user_prompt(**kwargs, negative_examples=negatives)
        batched = svc._resolve_user_prompt(**kwargs, negative_examples=negatives)

        assert serial == batched, (
            "the batched and serial paths produced different prompts"
        )
        assert "faint" in batched

    def test_no_contrast_means_no_block(self):
        """Negative control: the block must not appear uninvited."""
        prompt = self._svc()._build_user_prompt(
            examples=[_ex(6.8, "strong")],
            template_config=CONFIG,
            user_prompt_template="Look:\n{examples_block}",
            feature_id="f1",
        )
        assert "WEAKER EXAMPLES" not in prompt


class TestBatchAlignment:
    def test_a_short_negatives_list_is_refused(self):
        """C79. A positional zip must not silently mispair.

        `_label_batch` zips features, examples and negatives by position. A
        short list would attach one feature's weak examples to another
        feature's prompt — a wrong label with no error anywhere.
        """
        svc = LabelingService.__new__(LabelingService)

        class _F:
            id = "f1"
            neuron_index = 0
            nlp_analysis = None

        with pytest.raises(ValueError, match="positional mismatch"):
            svc._label_batch(
                labeling_service=None, loop=None,
                batch_features=[_F(), _F()],
                batch_examples=[[], []],
                batch_all_examples=[[], []],
                batch_negatives=[[]],  # one short
                feature_logit_effects={},
                template_config=CONFIG,
                user_prompt_template="{examples_block}",
                system_message="sys",
            )


@pytest.fixture
def db_session(async_engine):
    """Sync session against the test database (see test_stratified_sampling)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    url = str(settings.database_url_sync)
    if "postgresql" in url and "test" not in url:
        url = url.rsplit("/", 1)[0] + "/mistudio_test"
    engine = create_engine(url, poolclass=NullPool)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        # NO CLEANUP DELETES.
        #
        # `async_engine` is function-scoped and runs `Base.metadata.drop_all`
        # plus an enum drop on teardown, so every row here is removed anyway.
        # Deleting first was worse than redundant: this sync session holds locks
        # on the same tables, and racing them against that drop produced three
        # different failures depending on ordering — a duplicate pg_type key, a
        # missing feature_activations relation, and an outright DeadlockDetected.
        #
        # Two files defining this fixture passed individually and failed
        # together, so "the suite is green" was order-dependent. Closing and
        # disposing promptly is the whole job.
        session.rollback()
        session.close()
        engine.dispose()


class TestTheExclusionIsReal:
    def test_a_passage_is_never_both_strong_and_weak(self, db_session):
        """C78. Tested against real Postgres, on the shape that actually occurs.

        20 stored rows is not a contrived fixture — it is exactly what the eb48
        extraction stores for all 53,088 of its features.
        """
        from src.models.extraction_job import ExtractionJob
        from src.models.feature import Feature
        from src.models.feature_activation import FeatureActivation

        db_session.add(ExtractionJob(
            id="extr_weak_test", config={}, features_extracted=0))
        db_session.flush()
        db_session.add(Feature(
            id="feat_weak_20", extraction_job_id="extr_weak_test",
            neuron_index=0, name="feat_weak_20", activation_frequency=0.1,
            max_activation=1.0, mean_activation=0.5,
            interpretability_score=0.5))
        db_session.flush()
        for r in range(1, 21):
            db_session.add(FeatureActivation(
                feature_id="feat_weak_20", sample_index=r,
                max_activation=1.0 - r / 100.0,
                tokens=["a"], activations=[0.5],
                prefix_tokens=["a"], prime_token="p", suffix_tokens=["b"],
                prime_activation_index=0))
        db_session.commit()

        svc = LabelingService.__new__(LabelingService)
        shown = svc._retrieve_top_examples_batch_sync(
            db_session, ["feat_weak_20"], max_examples=10)
        weak = svc._retrieve_bottom_examples_batch_sync(
            db_session, ["feat_weak_20"], num_negative_examples=5,
            exclude_sample_indices_by_feature={
                fid: [e["sample_index"] for e in rows]
                for fid, rows in shown.items()
            },
        )

        shown_idx = {e["sample_index"] for e in shown["feat_weak_20"]}
        weak_idx = {e["sample_index"] for e in weak["feat_weak_20"]}

        assert len(shown_idx) == 10 and len(weak_idx) == 5
        assert not (shown_idx & weak_idx), (
            f"passages {sorted(shown_idx & weak_idx)} appear as BOTH a strong "
            f"and a weak example in the same prompt"
        )

    def test_without_the_exclusion_they_would_overlap(self, db_session):
        """The positive control that proves the test above can fail.

        With only 20 stored rows, an unexcluded bottom-5 and top-10 do NOT
        overlap (10 + 5 <= 20) — so a naive fixture would pass either way. This
        pins the boundary where they genuinely collide: K=15, N=10 over 20 rows.
        """
        from src.models.extraction_job import ExtractionJob
        from src.models.feature import Feature
        from src.models.feature_activation import FeatureActivation

        db_session.add(ExtractionJob(
            id="extr_weak_test", config={}, features_extracted=0))
        db_session.flush()
        db_session.add(Feature(
            id="feat_weak_ovl", extraction_job_id="extr_weak_test",
            neuron_index=1, name="feat_weak_ovl", activation_frequency=0.1,
            max_activation=1.0, mean_activation=0.5,
            interpretability_score=0.5))
        db_session.flush()
        for r in range(1, 21):
            db_session.add(FeatureActivation(
                feature_id="feat_weak_ovl", sample_index=r,
                max_activation=1.0 - r / 100.0,
                tokens=["a"], activations=[0.5],
                prefix_tokens=["a"], prime_token="p", suffix_tokens=["b"],
                prime_activation_index=0))
        db_session.commit()

        svc = LabelingService.__new__(LabelingService)
        shown = svc._retrieve_top_examples_batch_sync(
            db_session, ["feat_weak_ovl"], max_examples=15)
        shown_idx = {e["sample_index"] for e in shown["feat_weak_ovl"]}

        # Unexcluded: the two sets MUST collide at these sizes.
        unexcluded = db_session.execute(
            text(WEAKEST_EXAMPLES_SQL),
            {"feature_ids": ["feat_weak_ovl"], "num_negative_examples": 10,
             "excl_feature_ids": [], "excl_sample_indices": []},
        ).fetchall()
        assert shown_idx & {r.sample_index for r in unexcluded}, (
            "the fixture cannot exhibit an overlap, so the exclusion test "
            "above would pass with the guard removed"
        )

        # Excluded: they must not.
        excluded = svc._retrieve_bottom_examples_batch_sync(
            db_session, ["feat_weak_ovl"], num_negative_examples=10,
            exclude_sample_indices_by_feature={"feat_weak_ovl": sorted(shown_idx)},
        )
        assert not (shown_idx & {
            e["sample_index"] for e in excluded["feat_weak_ovl"]
        })


class TestWeakExamplesCannotDesynchronise:
    """A feature's weak examples must belong to THAT feature.

    WHY THIS EXISTS
    ---------------
    The first wiring carried weak examples in a list parallel to `features`,
    `features_examples` and `all_features_examples`. Those three are then
    re-indexed by the junk filter — `filter_features_from_examples` returns
    SHORTENED lists — and the fourth was not passed to it.

    So after any junk drop, every surviving feature was shown another feature's
    passages, under a heading that says "same feature, lower activation". A
    prompt whose entire design goal was to stop making false claims to the judge
    would have been making a new one.

    The length guard in `_label_batch` could not catch it: the stale list is
    LONGER than the filtered feature list, so the slice comes back full-length
    and the check passes. It fails open in exactly the case that occurs.

    The fix is structural — a dict keyed by feature id cannot desynchronise —
    and this test pins the property rather than the mechanism, so a future
    refactor back to parallel lists fails here.

    MUTATION CONTROL:
      C86 carry the negatives in a list and slice it positionally
           -> test_a_junk_drop_does_not_shift_weak_examples
    """

    def test_a_junk_drop_does_not_shift_weak_examples(self):
        """CALLS THE PRODUCTION FUNCTION, not a copy of it.

        The first version of this test built its own dict and its own stubs and
        then executed its OWN copy of the comprehension. It imported nothing
        from `labeling_service`, so reverting the production code wholesale to
        the parallel-list form left it green — while claiming in its docstring
        to be "stated over the real assembly step".

        Three features; the junk filter removes the FIRST. Under positional
        slicing the survivors inherit the dropped feature's weak examples.
        """
        from src.services.labeling_service import assemble_batch_negatives

        negatives_by_feature_id = {
            "feat_a": [_ex(0.1, "weak_a")],
            "feat_b": [_ex(0.2, "weak_b")],
            "feat_c": [_ex(0.3, "weak_c")],
        }

        class _F:
            def __init__(self, fid):
                self.id = fid

        # What the batch looks like after the junk filter dropped feat_a.
        surviving = [_F("feat_b"), _F("feat_c")]

        batch_negatives = assemble_batch_negatives(
            surviving, negatives_by_feature_id
        )

        assert batch_negatives[0][0]["prime_token"] == "weak_b", (
            "feat_b was given another feature's weak examples"
        )
        assert batch_negatives[1][0]["prime_token"] == "weak_c"

    def test_a_feature_with_no_weak_examples_gets_an_empty_list(self):
        """The list must stay positionally aligned with `batch_features`.

        Returning nothing for a missing feature would shorten the list and
        re-introduce the mispairing from the other direction.
        """
        from src.services.labeling_service import assemble_batch_negatives

        class _F:
            def __init__(self, fid):
                self.id = fid

        out = assemble_batch_negatives(
            [_F("feat_a"), _F("feat_missing"), _F("feat_c")],
            {"feat_a": [_ex(0.1, "a")], "feat_c": [_ex(0.3, "c")]},
        )
        assert len(out) == 3
        assert out[1] == []
        assert out[2][0]["prime_token"] == "c"

    def test_the_service_assembles_batches_through_that_function(self):
        """C86. Pins that production USES the tested function.

        The behavioural test above proves the function is right; this proves the
        service calls it. Bound to the ASSIGNMENT TARGET rather than to a
        substring, because the previous version asserted three strings and a
        mutation satisfied all three while restoring the defect: keep the dict
        alive for the scrape, populate a parallel list beside it, and slice that.

        The self-check makes the scan fail closed if the pattern moves.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_service import LabelingService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingService.label_features_for_extraction)
        ))

        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "batch_negatives"
                for t in node.targets
            )
        ]
        # SELF-CHECK: no assignment means the pattern moved and every assertion
        # below is vacuous.
        assert assignments, (
            "`batch_negatives` is no longer assigned; this scan is inert"
        )

        for node in assignments:
            dumped = ast.dump(node.value)
            assert "assemble_batch_negatives" in dumped, (
                "batch_negatives is built inline again rather than through the "
                "tested function — and if it is built from a parallel list it "
                "will desynchronise at the junk filter"
            )
            assert "Slice" not in dumped, (
                "batch_negatives is being SLICED positionally; the feature "
                "lists are re-indexed by the junk filter and this one is not"
            )


class TestRetrievalIsSplitWhenNlpIsOn:
    """The judge's examples and the NLP summariser's examples are two questions.

    WHY THIS EXISTS
    ---------------
    One retrieval served both: fetch 100 rows for NLP, then slice the first 10
    for the judge. That broke two capabilities silently.

    * STRATIFICATION BECAME A NO-OP. Banding 100 rows into 100 bands returns
      ranks 1..100 in order, and the first ten are exactly top-K. Two arms of an
      A/B trial would have produced IDENTICAL prompts while reporting distinct
      prompt fingerprints — a null result that looked like a measurement.
    * THE CONTRAST POOL WAS FULLY EXCLUDED. Retention is at most 100 rows per
      feature, so "exclude everything retrieved" left the weak query nothing to
      return, and the block never rendered on any NLP-enabled template.

    MUTATION CONTROLS:
      C87 retrieve once with retrieval_count and slice for display
           -> test_the_display_retrieval_uses_max_examples
      C88 exclude the NLP set instead of the displayed set
           -> test_the_exclusion_is_the_displayed_set
    """

    def test_the_split_is_covered_by_the_sampling_wiring_tests(self):
        """C87 lives in test_sampling_wiring_reachable, bound to the AST.

        The version that lived here asserted `"max_examples=max_examples," in
        source`. Swapping the two retrievals' size arguments — judge gets 100,
        summariser gets 10 — left that literal present on the OTHER call and
        passed, restoring both R1 defects at once. It also pinned a comment
        character-for-character, including its double space, so rewording a
        comment broke the test while changing the behaviour did not.

        Named rather than deleted silently, so the coverage is traceable.
        """
        from tests.unit import test_sampling_wiring_reachable as wiring

        assert hasattr(
            wiring.TestTheCallSitesPassTheStrategy,
            "test_the_bulk_path_forwards_the_config_key_unconditionally",
        ), "the replacement coverage for C87 is gone"

    def test_the_exclusion_is_the_displayed_set(self):
        """C88. Excluding the NLP set removes the entire donor pool."""
        import inspect

        from src.services.labeling_service import LabelingService

        source = inspect.getsource(LabelingService.label_features_for_extraction)

        assert "for fid, rows in display_map.items()" in source, (
            "the weak-example exclusion is not built from the displayed set; "
            "if it is built from the NLP set there is nothing left to return"
        )
