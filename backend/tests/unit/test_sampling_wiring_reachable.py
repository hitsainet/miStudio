"""The sampling strategy must reach the query, and the ruler must not move.

WHY THIS EXISTS
---------------
Review round 1 ran the obvious mutation — make `_retrieve_top_examples_batch_sync`
ignore its `sampling` argument — against 228 labeling tests. All 228 passed.

`test_stratified_sampling.py` proves the SQL is correct and that
`examples_sql()` dispatches. Nothing connected the dispatcher to a caller. So
the arc's HEADLINE capability was an inert switch that reported itself as on —
the exact failure the sibling file `test_weak_examples_wired.py` exists to
prevent, reproduced one function away.

Three separate wirings are pinned here, each by driving the real method and
asserting on what the DATABASE was asked, not on what the source says:

  * the retrieval passes the template's strategy to the SQL selector
  * the trial's PROMPT retrieval uses the arm's strategy
  * the trial's SCORING retrieval is PINNED to top_k regardless of the arm

The third is the one PADR IDL-48 is cited for. If the scoring passages moved
with the arm, a stratified arm would be graded on intrinsically harder text than
its baseline, so a negative result would be an artifact of the instrument. That
guard had zero tests.

MUTATION CONTROLS:
  C89 `text(examples_sql(sampling))` -> `text(examples_sql(None))`
       -> test_the_strategy_reaches_the_query
  C90 drop `sampling=template_config.get('example_sampling')` at the bulk site
       -> test_the_bulk_path_passes_the_templates_strategy
  C91 `sampling="top_k"` -> `frozen.get("example_sampling")` on the scoring
       retrieval
       -> test_the_scoring_retrieval_is_pinned_regardless_of_the_arm
"""

from unittest.mock import MagicMock

import pytest

from src.services.labeling_service import (
    STRATIFIED_EXAMPLES_SQL,
    TOP_K_EXAMPLES_SQL,
    LabelingService,
)


class _CapturingSession:
    """Records what each execute() was given, refusing what a real Session would.

    NO MORE FORGIVING THAN THE REAL THING, and both halves of that were wrong
    first time round:

      * it `str()`d anything, so dropping the `text(...)` wrapper passed here
        and raised `ObjectNotExecutableError` in production;
      * it ignored `params`, so hardcoding `{"max_examples": 10}` — which caps
        the NLP retrieval at ten rows instead of the full stored set — survived
        every test in this file.

    A stand-in that accepts what production rejects is how a green suite ships a
    broken call.
    """

    def __init__(self):
        self.statements = []
        self.params = []

    def execute(self, statement, params=None):
        from sqlalchemy.sql.elements import TextClause

        if not isinstance(statement, TextClause):
            raise TypeError(
                f"execute() was given {type(statement).__name__}, not a "
                f"TextClause. SQLAlchemy raises ObjectNotExecutableError for a "
                f"bare string; a stand-in that accepts one hides that."
            )
        self.statements.append(str(statement))
        self.params.append(params)
        return iter(())


class TestTheStrategyReachesTheQuery:
    def test_the_strategy_reaches_the_query(self):
        """C89. The argument must select the SQL, not be accepted and ignored.

        Asserts on the STATEMENT TEXT the session received — the only evidence
        that distinguishes "the parameter was passed" from "the parameter had
        an effect".
        """
        svc = LabelingService.__new__(LabelingService)
        session = _CapturingSession()

        svc._retrieve_top_examples_batch_sync(
            session, ["feat_1"], max_examples=10, sampling="stratified"
        )

        assert len(session.statements) == 1
        # The size actually asked for must be the size requested, or a
        # hardcoded value silently truncates the retrieval.
        assert session.params[0]["max_examples"] == 10
        assert session.params[0]["feature_ids"] == ["feat_1"]
        assert session.statements[0].strip() == STRATIFIED_EXAMPLES_SQL.strip(), (
            "the retrieval ran the top-K query despite being asked for "
            "stratified sampling; the switch is inert"
        )

    def test_top_k_still_runs_the_top_k_query(self):
        """Negative control: a selector that always returns stratified passes
        the test above and silently changes every existing template's prompt."""
        svc = LabelingService.__new__(LabelingService)
        session = _CapturingSession()

        svc._retrieve_top_examples_batch_sync(
            session, ["feat_1"], max_examples=10, sampling="top_k"
        )
        assert session.statements[0].strip() == TOP_K_EXAMPLES_SQL.strip()

    def test_no_strategy_runs_the_top_k_query(self):
        """The default must be the historical behaviour."""
        svc = LabelingService.__new__(LabelingService)
        session = _CapturingSession()

        svc._retrieve_top_examples_batch_sync(session, ["feat_1"], max_examples=10)
        assert session.statements[0].strip() == TOP_K_EXAMPLES_SQL.strip()

    def test_the_weakest_query_is_used_for_negatives(self):
        """The contrast retrieval must not silently share the positives' SQL."""
        from src.services.labeling_service import WEAKEST_EXAMPLES_SQL

        svc = LabelingService.__new__(LabelingService)
        session = _CapturingSession()

        svc._retrieve_bottom_examples_batch_sync(
            session, ["feat_1"], num_negative_examples=5
        )
        assert session.statements[0].strip() == WEAKEST_EXAMPLES_SQL.strip()


class TestTheCallSitesPassTheStrategy:
    """The strategy must travel from the template to the retrieval.

    The first version of this class defined a `_record_calls` helper and NEVER
    CALLED IT — both tests were pure AST scans, while the docstring claimed they
    were "driven through the real methods with a recording double". A dead
    helper standing in for the coverage it describes.

    The consequence was a live hole: making the forward conditional —

        sampling=(template_config.get('example_sampling') if include_nlp else None)

    — leaves the string `example_sampling` in the AST dump, so the scan passed,
    while stratification became inert on the DEFAULT configuration
    (`include_nlp` is False for every template in the estate). One edit, the
    arc's headline capability switched off, suite green.

    So the value is now asserted where it arrives.
    """

    def test_the_value_reaching_the_retrieval_is_the_templates_strategy(self):
        """Drives the retrieval and reads what the SQL selector was given.

        `examples_sql` is the single point every strategy must pass through, so
        recording its argument catches a conditional, a `None`, or a hardcoded
        value wherever upstream it was introduced.
        """
        from src.services import labeling_service as module

        seen = []
        real = module.examples_sql

        def _recording(sampling):
            seen.append(sampling)
            return real(sampling)

        module.examples_sql = _recording
        try:
            svc = LabelingService.__new__(LabelingService)
            session = _CapturingSession()
            svc._retrieve_top_examples_batch_sync(
                session, ["feat_1"], max_examples=10, sampling="stratified"
            )
        finally:
            module.examples_sql = real

        assert seen == ["stratified"], (
            f"the SQL selector was given {seen!r}; the template's strategy did "
            f"not reach the query"
        )

    def test_the_bulk_path_forwards_the_config_key_unconditionally(self):
        """C90. Bound to the CALL, with comments stripped — and no condition.

        An `in source` backstop matched the argument's name inside a COMMENT, so
        commenting the forward out passed. `ast` ignores comments, which is the
        point of using it; and an `IfExp` in the value is now refused outright,
        because a forward that only happens sometimes is the shape that made
        stratification inert on the default configuration.
        """
        import ast
        import inspect
        import textwrap

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
            by_target[target] = {kw.arg: kw.value for kw in call.keywords}

        # SELF-CHECK: two retrievals is the design — the judge's (sampled) and
        # the NLP summariser's (unsampled).
        for required in ("display_map", "all_examples_map"):
            assert required in by_target, (
                f"no retrieval assigned to {required!r}; the split has moved "
                f"and this test is inert"
            )

        display = by_target["display_map"]
        sampling = display.get("sampling")
        assert sampling is not None, "the judge's retrieval states no strategy"
        assert not isinstance(sampling, ast.IfExp), (
            "the sampling strategy is forwarded CONDITIONALLY. A forward that "
            "only fires on some configurations is an inert switch on the "
            "others — and `include_nlp` is False for every template in the "
            "estate."
        )
        assert "example_sampling" in ast.dump(sampling), (
            "the judge's retrieval does not forward the template's strategy"
        )
        assert "max_examples" in ast.dump(display.get("max_examples")), (
            "the judge's retrieval asks for a different size than it displays, "
            "so the strategy is discarded by the slice"
        )

        nlp = by_target["all_examples_map"]
        assert "None" in ast.dump(nlp.get("sampling")), (
            "the NLP retrieval carries a sampling strategy; the summariser "
            "reads activation order and a stratified span is not that"
        )
        assert "retrieval_count" in ast.dump(nlp.get("max_examples")), (
            "the NLP retrieval no longer asks for the full stored set"
        )


class TestTheSweepCarriesItsOwnPredicate:
    """R4. The sweep must select with the predicate it was SIZED with.

    Two independent mutations each restored the defect with 4545 tests green:
    nulling the arguments in `next_batch_ids`, and reverting the endpoint's
    config merge. The docstring describing the hazard was three lines above the
    code, and nothing asserted either half.

    On a 100-batch ceiling at 2000 features that is up to ~444 GPU-hours booked
    against a set the sweep will not select.

    MUTATION CONTROLS:
      M15 `prompt_fingerprint=config.get(...)` -> `=None`
      M16 revert `config={**..., **fingerprint, **judge}` to `body.config.model_dump()`
    """

    def test_next_batch_ids_passes_the_frozen_predicate(self):
        """M15. Bound to the arguments, resolved through their source."""
        import ast
        import inspect
        import textwrap

        from src.services.labeling_sweep_service import LabelingSweepService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingSweepService.next_batch_ids)
        ))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resume_batch_query"
        ]
        assert calls, "no resume_batch_query call found; this test is inert"

        for call in calls:
            kwargs = {kw.arg: ast.dump(kw.value) for kw in call.keywords}
            for key in ("prompt_fingerprint", "judge_model"):
                assert key in kwargs, (
                    f"the sweep selects without {key}, so every batch takes a "
                    f"different set than the one max_batches was computed from"
                )
                assert "config" in kwargs[key], (
                    f"{key} is not read from the sweep's FROZEN config, so "
                    f"editing the template mid-sweep changes what a running "
                    f"sweep selects"
                )

    def test_the_endpoint_freezes_the_predicate_onto_the_row(self):
        """M16. The request's predicate must reach `config`."""
        import ast
        import inspect
        import textwrap

        from src.api.v1.endpoints import labeling as endpoints

        source = textwrap.dedent(inspect.getsource(endpoints))
        tree = ast.parse(source)

        creates = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
            and any(kw.arg == "max_batches" for kw in node.keywords)
        ]
        assert creates, "no sweep create call found; this test is inert"

        for call in creates:
            config = next(
                (kw.value for kw in call.keywords if kw.arg == "config"), None
            )
            assert config is not None, "the sweep is created without a config"
            dumped = ast.dump(config)
            for key in ("prompt_fingerprint", "judge_model"):
                assert key in dumped, (
                    f"{key} is not frozen onto the sweep row, so the sweep "
                    f"cannot select with the predicate it was sized with"
                )

    def test_the_request_schema_accepts_the_predicate(self):
        """Without these fields the endpoint cannot receive it at all."""
        from src.api.v1.endpoints.labeling import ResumeSweepRequest

        fields = ResumeSweepRequest.model_fields
        assert "prompt_fingerprint" in fields
        assert "judge_model" in fields


class TestTheScoringRulerIsPinned:
    """PADR IDL-48, one level deeper: the passages are half the ruler.

    `_score_detection` takes its positives, its `negative_ceiling` and the
    gate's oracle token from whatever dict it is handed. If that dict came from
    the arm's own sampling, a stratified arm would be graded on different,
    intrinsically harder passages than its baseline — biased downward by
    construction, so a null would be uninterpretable and a negative an artifact.

    This guard had zero tests despite being the arc's most-cited safeguard.
    """

    def test_the_scoring_retrieval_is_pinned_regardless_of_the_arm(self):
        """C91. WHICH retrieval is pinned, not merely that one of them is.

        The first version collected both calls' `sampling=` values and asserted
        that at least one said `top_k`. Swapping the two — pinning the PROMPT
        and letting the RULER follow the arm — left exactly one `top_k` in the
        list and passed. That single edit makes the arm unmeasurable (the prompt
        stops varying) AND moves the ruler with the arm, which is the whole
        failure this guard exists to catch.

        Bound to the assignment targets: `scoring_examples` must be pinned,
        `examples_by_feature` must not be.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import (
            SCORING_POSITIVES_K,
            LabelingTrialService,
        )

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService)
        ))

        sampling_by_target = {}
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
            sampling = next(
                (ast.dump(kw.value) for kw in call.keywords if kw.arg == "sampling"),
                None,
            )
            sampling_by_target[target] = sampling

        # SELF-CHECK: both retrievals must be found by name, or the scan has
        # stopped matching and every assertion below is vacuous.
        for required in ("examples_by_feature", "scoring_examples"):
            assert required in sampling_by_target, (
                f"no retrieval assigned to {required!r} was found; the trial's "
                f"variable/ruler split has moved and this test is inert"
            )

        ruler = sampling_by_target["scoring_examples"]
        assert ruler is not None and "'top_k'" in ruler, (
            f"the SCORING retrieval is not pinned to top_k (it is {ruler!r}). "
            f"A stratified arm would then be graded on its own, intrinsically "
            f"harder passages — biased downward by construction, so a null "
            f"result is uninterpretable and a negative one an artifact."
        )

        variable = sampling_by_target["examples_by_feature"]
        assert variable is not None and "'top_k'" not in variable, (
            f"the PROMPT retrieval is pinned to top_k ({variable!r}), so the "
            f"arm under test does not vary and the trial measures nothing"
        )
        assert "example_sampling" in variable, (
            "the prompt retrieval does not read the arm's sampling strategy"
        )
        assert SCORING_POSITIVES_K == 10

    def test_the_pinned_set_is_what_scoring_consumes(self):
        """M1. The retrieval being pinned is not the same as it being USED.

        The existing guard asserts two retrievals exist and one says `top_k`.
        Changing only the CONSUMPTION site —
        `_score_detection(examples_by_feature=examples_by_feature)` — leaves both
        retrievals in place, one still pinned, and the guard green, while the
        arm is graded on its own passages. The entire validity argument for the
        experiment is gone and the payload still records `top_k`, so the
        provenance field certifies a falsehood.

        Bound to the ARGUMENT of the `_score_detection` call.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import LabelingTrialService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService)
        ))

        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_score_detection"
        ]
        # SELF-CHECK: the call must exist, or this asserts nothing.
        assert calls, "no _score_detection call found; this test is inert"

        for call in calls:
            kwargs = {kw.arg: kw.value for kw in call.keywords}
            examples = kwargs.get("examples_by_feature")
            assert examples is not None, (
                "_score_detection is called without naming its example set"
            )
            assert isinstance(examples, ast.Name), (
                "the scoring set is an expression rather than a named variable; "
                "bind it to `scoring_examples` so the pinning is legible"
            )
            # EITHER PINNED RULER, and nothing else.
            #
            # Two rulers now: top-K and mid-range. Both are retrieved without a
            # sampling strategy, so both are arm-independent. What must never
            # appear here is `examples_by_feature` — the arm's own prompt set —
            # because grading an arm on its own passages biases it downward by
            # construction while payload['scoring'] still records 'top_k'.
            assert examples.id in {"scoring_examples", "midrange_examples"}, (
                f"scoring consumes {examples.id!r}, which is not a pinned "
                f"ruler. If that is the arm's own prompt set, a stratified arm "
                f"is graded on intrinsically harder passages and a negative "
                f"result is an artifact of the instrument."
            )
            prompt_indices = kwargs.get("prompt_sample_indices")
            # A NON-EMPTY COMPREHENSION, not merely the keyword.
            #
            # `prompt_sample_indices={} or {...}` keeps the kwarg and passes an
            # empty dict, so every passage the judge was shown becomes drawable
            # as its own negative again.
            if prompt_indices is not None:
                assert isinstance(prompt_indices, ast.DictComp), (
                    "prompt_sample_indices is not built as a comprehension over "
                    "what the judge was shown; an empty literal restores the "
                    "contamination the exclusion exists to remove"
                )
                dumped = ast.dump(prompt_indices)
                assert "examples_by_feature" in dumped, (
                    "the prompt's positives are not excluded"
                )
                assert "negatives_by_feature" in dumped, (
                    "the prompt's CONTRAST passages are not excluded — the "
                    "judge is asked whether the label describes a passage it "
                    "was told activates the feature, and marked wrong for "
                    "saying yes"
                )
            assert prompt_indices is not None, (
                "M3: the arm's prompt passages are not excluded from its own "
                "negatives, so the judge is asked whether a label describes the "
                "evidence it was derived from — and scored wrong for saying yes"
            )

    def test_the_exclusion_unions_both_sets(self):
        """M3. The union, not either half.

        `exclude_samples` must cover the pinned scoring set AND the arm's own
        prompt set. Excluding only the scoring set leaves ranks 11-91 of a
        stratified arm drawable as that feature's own negatives.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import LabelingTrialService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService._score_detection)
        ))

        excludes = [
            kw.value for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords if kw.arg == "exclude_samples"
        ]
        assert excludes, "no exclude_samples argument found; this test is inert"

        for value in excludes:
            dumped = ast.dump(value)
            assert "positives" in dumped, (
                "the pinned scoring passages are not excluded"
            )
            assert "prompt_sample_indices" in dumped, (
                "the arm's own prompt passages are not excluded, so a label "
                "can be scored against the evidence it was derived from"
            )

    def test_the_trial_sends_the_contrast_block_the_bulk_run_would(self):
        """2f. A trial must not measure a prompt the bulk run will not send.

        `include_negative_examples` was in the frozen template_config and in
        `TEMPLATE_CONFIG_KEYS`, but the trial never passed a `negative_examples`
        LIST — and the formatter renders the block only when handed one. So an
        arm configured for contrast produced a prompt byte-identical to one
        without it: `compare` reported `identical`, the paired delta reported ~0
        with a CI containing zero, and the two frozen fingerprints DID differ,
        so the record looked like two distinct arms.

        A confidently measured null, manufactured. The plan's question "does
        explicit contrast add anything?" would have been answered "no" by an
        experiment that never ran it.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import LabelingTrialService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService)
        ))

        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate_label_from_examples"
        ]
        assert calls, "no label call found in the trial; this test is inert"

        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "negative_examples" in kwargs, (
                "the trial does not pass negative_examples, so a "
                "contrast-enabled template renders no contrast block and the "
                "arm is indistinguishable from the one it is meant to differ "
                "from"
            )

        # And the retrieval that feeds it must exist and exclude what is shown.
        bottoms = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_retrieve_bottom_examples_batch_sync"
        ]
        assert bottoms, "the trial never retrieves weak examples"
        for call in bottoms:
            kwargs = {kw.arg: ast.dump(kw.value) for kw in call.keywords}
            assert "exclude_sample_indices_by_feature" in kwargs, (
                "the trial's contrast block can repeat a passage it is already "
                "showing as a positive"
            )
            # AND THE COUNT MUST COME FROM THE TEMPLATE.
            #
            # Hardcoding 0 leaves `negative_examples=[]` still being passed, so
            # a kwarg-presence check stays green while the arm is byte-identical
            # to its baseline — the manufactured null, with the guard written
            # for it intact.
            count = kwargs.get("num_negative_examples", "")
            assert "n_negative" in count, (
                f"the contrast count is {count!r} rather than the resolved "
                f"template value"
            )

        assigns = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "n_negative"
                for t in node.targets
            )
        ]
        assert assigns, "n_negative is not assigned; this test is inert"
        for node in assigns:
            dumped = ast.dump(node.value)
            assert "resolve_num_negative" in dumped, (
                f"n_negative is set from {dumped!r} rather than resolved from "
                f"the frozen template — a constant here makes every contrast "
                f"arm identical to its baseline"
            )

    def test_the_two_rulers_score_different_passages(self):
        """C117. Two rulers grading the same text are one ruler.

        The mid-range ruler exists because top-K detection rewards the narrowest
        label covering the top decile, while a wider example spread produces
        broader ones. If both `_score_detection` calls receive the same
        passages, the second measurement is a duplicate of the first, the
        combined verdict is derived from one instrument twice, and the
        experiment is back to being unable to detect success.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import LabelingTrialService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService)
        ))

        sets = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_score_detection"
            ):
                continue
            examples = next(
                (kw.value for kw in node.keywords
                 if kw.arg == "examples_by_feature"), None
            )
            assert isinstance(examples, ast.Name), (
                "a scoring set is an expression rather than a named variable; "
                "bind it so the pinning is legible"
            )
            sets.append(examples.id)

        # SELF-CHECK: two scoring calls is the design.
        assert len(sets) == 2, (
            f"expected two scoring passes (top-K and mid-range), found "
            f"{len(sets)}: {sets}"
        )
        assert len(set(sets)) == 2, (
            f"both rulers score the same passages ({sets[0]!r}), so the second "
            f"measurement duplicates the first and the combined verdict rests "
            f"on one instrument counted twice"
        )
        assert "scoring_examples" in sets and "midrange_examples" in sets, (
            f"the two rulers are {sets}, not the pinned top-K and mid-range sets"
        )

    def test_the_midrange_ruler_is_pinned_too(self):
        """It must not carry a sampling strategy either.

        A ruler that follows the arm is not a ruler, whichever band it draws
        from.
        """
        import ast
        import inspect
        import textwrap

        from src.services.labeling_trial_service import LabelingTrialService

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(LabelingTrialService)
        ))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_retrieve_midrange_examples_batch_sync"
        ]
        assert calls, "the mid-range ruler is never retrieved; this is inert"
        for call in calls:
            assert not any(kw.arg == "sampling" for kw in call.keywords), (
                "the mid-range retrieval takes a sampling strategy, so it can "
                "be made to follow the arm under test"
            )

    def test_the_trial_records_which_ruler_it_used(self):
        """A pinned ruler nobody can verify afterwards is not evidence."""
        import inspect

        from src.services.labeling_trial_service import LabelingTrialService

        source = inspect.getsource(LabelingTrialService)
        assert '"positive_sampling"' in source, (
            "the trial payload does not record the scoring strategy, so a "
            "later reader cannot check the ruler held"
        )
        assert "SCORING_POSITIVES_K" in source, (
            "the recorded n_positive is a literal rather than the constant the "
            "retrieval actually used, so the audit field can disagree with the "
            "run it describes"
        )
