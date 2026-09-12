"""The per-feature adjudication state that makes a resume possible.

The defect these guard against is measurable on the live database: 16,824
features across the estate have FAILED and look finished, because a failure was
written as a fake label (`category='error_feature'`, `name='feature_{n}'`) with
`label_source` and `labeled_at` both set. Every "is it labeled?" predicate said
yes, including the one that looked reusable — `has_label` is
`label_source != 'auto'`.
"""

import pytest
from sqlalchemy import String

from src.models.feature import Feature
from src.models.labeling_prompt_template import LabelingPromptTemplate
from src.services.labeling_fingerprint import (
    IDENTITY_FIELDS,
    prompt_fingerprint,
    prompt_fingerprint_fields,
)


def _template(**overrides) -> LabelingPromptTemplate:
    """A fully-populated template. Every column is set explicitly.

    An unset column reads as None on a transient instance, and a fixture where
    most fields are None cannot distinguish "this field is covered by the
    fingerprint" from "both variants hashed None" — the fixtures-agree-by-
    construction trap that has kept this repo's suite green over real bugs.
    """
    base = dict(
        id="lpt_fixture",
        name="Fixture template",
        description="a description",
        system_message="You are a judge.",
        user_prompt_template="Examples:\n{examples_block}",
        temperature=0.3,
        max_tokens=50,
        top_p=0.9,
        template_type="mistudio_context",
        max_examples=10,
        include_prefix=True,
        include_suffix=True,
        prime_token_marker="<<>>",
        include_logit_effects=False,
        top_promoted_tokens_count=10,
        top_suppressed_tokens_count=10,
        include_negative_examples=True,
        num_negative_examples=5,
        is_detection_template=False,
        include_nlp_analysis=False,
        is_default=False,
        is_system=False,
        created_by="tester",
    )
    base.update(overrides)
    return LabelingPromptTemplate(**base)


# ── the columns ──────────────────────────────────────────────────────────────

class TestAdjudicationColumns:
    """The ORM must declare what the migration adds, or the unit suite — which
    builds its schema from `Base.metadata.create_all()` — tests a different
    table from the one production runs against."""

    def test_all_six_columns_are_declared(self):
        cols = Feature.__table__.c
        for name in (
            "label_status",
            "label_attempts",
            "label_error",
            "label_error_at",
            "label_prompt_fingerprint",
            "label_model",
        ):
            assert name in cols, f"features.{name} is not declared on the ORM"

    def test_status_and_attempts_are_not_null_with_server_defaults(self):
        """A NOT NULL column with no server default breaks every raw INSERT and
        every pre-existing row the migration touches."""
        for name, default in (("label_status", "pending"), ("label_attempts", "0")):
            col = Feature.__table__.c[name]
            assert col.nullable is False, f"features.{name} must be NOT NULL"
            assert col.server_default is not None, (
                f"features.{name} is NOT NULL with no server default"
            )
            assert col.server_default.arg == default

    def test_fingerprint_column_holds_a_whole_sha256(self):
        """64 hex characters exactly. A narrower column truncates silently and
        collapses distinct judges onto one identity."""
        col = Feature.__table__.c["label_prompt_fingerprint"]
        assert isinstance(col.type, String)
        assert col.type.length == 64
        assert len(prompt_fingerprint(_template())) == col.type.length

    def test_resume_predicate_columns_are_indexed(self):
        """`label_status` filters every resume; `label_prompt_fingerprint`
        answers the staleness query. Both run over the whole extraction."""
        for name in ("label_status", "label_prompt_fingerprint"):
            assert Feature.__table__.c[name].index is True, (
                f"features.{name} drives a whole-table predicate and is unindexed"
            )


# ── the fingerprint ──────────────────────────────────────────────────────────

# Every column that is not identity/bookkeeping. Derived, so a new column is
# covered without anyone remembering to add it here.
PROMPT_FIELDS = sorted(
    c.name
    for c in LabelingPromptTemplate.__table__.columns
    if c.name not in IDENTITY_FIELDS
)

# A value guaranteed to differ from the fixture's, per field.
_OTHER = {
    "system_message": "You are a different judge.",
    "user_prompt_template": "Nothing:\n{examples_block}",
    "template_type": "anthropic_logit",
    "temperature": 1.0,
    "max_tokens": 500,
    "top_p": 0.5,
    "max_examples": 20,
    "include_prefix": False,
    "include_suffix": False,
    "prime_token_marker": "[[]]",
    "include_logit_effects": True,
    "top_promoted_tokens_count": 3,
    "top_suppressed_tokens_count": 3,
    "include_negative_examples": False,
    "num_negative_examples": 1,
    "is_detection_template": True,
    "include_nlp_analysis": True,
    # WHICH examples the judge is shown, and HOW their strength is stated.
    # Both change what the judge is sent, so both must move the fingerprint —
    # a verdict produced from the top ten is not the same verdict as one
    # produced from a stratified span of the same feature.
    "example_sampling": "stratified",
    "activation_display": "percent_of_max",
}


class TestPromptFingerprint:

    def test_the_denylist_itself_is_pinned(self):
        """PROMPT_FIELDS is derived by SUBTRACTING IDENTITY_FIELDS, so every
        other test here inherits whatever that set says. Moving a real prompt
        field into it would drop that field from the parametrised coverage test
        and no assertion would fire — two views sharing one blind spot.

        Pinning the literal set is what makes the derivation safe: widening it
        must be a deliberate edit here, with a reason.
        """
        assert IDENTITY_FIELDS == {
            "id",
            "name",
            "description",
            "is_default",
            "is_system",
            "created_by",
            "created_at",
            "updated_at",
        }, (
            "IDENTITY_FIELDS changed. A field belongs here ONLY if it cannot "
            "affect what the judge is sent or how it is called — otherwise "
            "verdicts it invalidates will silently stay valid."
        )

    def test_every_column_is_classified(self):
        """The drift guard. A new column on the template is either covered by
        the fingerprint or explicitly declared identity — never neither.

        `freeze_template` next door hand-lists its fields and would silently
        ignore a new one; that is the failure mode this asserts away.
        """
        declared = {c.name for c in LabelingPromptTemplate.__table__.columns}
        classified = set(PROMPT_FIELDS) | set(IDENTITY_FIELDS)
        assert declared == classified, (
            f"unclassified template columns: {sorted(declared - classified)}"
        )

    def test_covered_fields_are_exactly_the_prompt_fields(self):
        assert sorted(prompt_fingerprint_fields(_template())) == PROMPT_FIELDS

    def test_is_deterministic(self):
        assert prompt_fingerprint(_template()) == prompt_fingerprint(_template())

    @pytest.mark.parametrize("field", PROMPT_FIELDS)
    def test_changing_a_prompt_field_changes_the_fingerprint(self, field):
        """Parametrised over every field, not one representative. A single
        spot-check passes against a fingerprint that covers only that field."""
        assert field in _OTHER, f"no differing value defined for {field}"
        before = prompt_fingerprint(_template())
        after = prompt_fingerprint(_template(**{field: _OTHER[field]}))
        assert before != after, f"{field} changes the prompt but not the fingerprint"

    @pytest.mark.parametrize("field", sorted(IDENTITY_FIELDS - {"created_at", "updated_at"}))
    def test_identity_changes_leave_the_fingerprint_alone(self, field):
        """Renaming a template, or marking it default, must not invalidate the
        verdicts it already produced — nothing the judge saw has changed."""
        other = {"id": "lpt_other", "name": "Renamed", "description": "other",
                 "is_default": True, "is_system": True, "created_by": "someone"}[field]
        assert prompt_fingerprint(_template()) == prompt_fingerprint(
            _template(**{field: other})
        )

    def test_absent_template_has_its_own_stable_identity(self):
        """NULL means "we do not know which judge produced this" and is what the
        backfill writes over historical rows. A live run with no template row
        must be distinguishable from that, and from any real template."""
        builtin = prompt_fingerprint(None)
        assert builtin == prompt_fingerprint(None)
        assert len(builtin) == 64
        assert builtin != prompt_fingerprint(_template())

        # Pin the MECHANISM, not just the consequence. Asserting only that the
        # two differ passes against `return {}` — an empty payload is also
        # stable and also unequal, but it is a value another code path could
        # plausibly produce, so "no template" would stop being unforgeable.
        # (Mutation control C7 survived the consequence-only assertion.)
        fields = prompt_fingerprint_fields(None)
        assert fields, "an absent template must hash a named identity, not an empty payload"
        assert "builtin" in str(fields).lower(), (
            f"the absent-template identity must name itself; got {fields!r}"
        )

    def test_refuses_a_value_with_no_stable_textual_form(self):
        """`json.dumps(default=str)` would hash a repr containing a memory
        address, and the fingerprint would then differ between two processes
        reading the same row — presenting as "everything is stale, always"."""
        template = _template()
        template.prime_token_marker = object()
        with pytest.raises(TypeError, match="stable textual form"):
            prompt_fingerprint(template)


class TestFreezeTemplateAgrees:
    """`labeling_trial_service.freeze_template` copies the template body into a
    trial's payload so a later edit cannot re-describe a finished run. It is a
    hand-maintained list of the same fields, so it can drift from the ORM in
    exactly the way the fingerprint cannot."""

    def test_freeze_template_covers_every_prompt_field(self):
        from src.services.labeling_trial_service import freeze_template

        frozen = set(freeze_template(_template()))
        missing = set(PROMPT_FIELDS) - frozen
        assert not missing, (
            f"freeze_template omits prompt-affecting fields: {sorted(missing)} — "
            "a trial payload would understate what it ran with"
        )


# ── the single writer ────────────────────────────────────────────────────────

import inspect
from unittest.mock import MagicMock, patch

from src.models.labeling_job import LabelingJob
from src.services.labeling_service import LabelingService


@pytest.fixture
def writer():
    return LabelingService(db=MagicMock())


@pytest.fixture
def job():
    return LabelingJob(id="lbl_test", extraction_job_id="ext_test")


def _feature(**overrides) -> Feature:
    base = dict(
        id="feat_1",
        extraction_job_id="ext_test",
        neuron_index=42,
        name="feature_42",
        category=None,
        activation_frequency=0.1,
        interpretability_score=0.5,
        max_activation=1.0,
        label_status="pending",
        label_attempts=0,
        star_color=None,
    )
    base.update(overrides)
    return Feature(**base)


LABELED_AT = __import__("datetime").datetime(2026, 9, 8, tzinfo=__import__("datetime").timezone.utc)


def _persist(writer, job, feature, label, examples=None):
    with patch("src.services.labeling_service.emit_labeling_result"):
        return writer._persist_label_outcome(
            feature,
            label,
            examples if examples is not None else [],
            labeling_job=job,
            labeled_at=LABELED_AT,
            label_source_value="openai",
            prompt_fingerprint="f" * 64,
            judge_model="gemma-4-31b-GGUF:IQ4_XS",
        )


class TestPersistLabelOutcomeIsTheOnlyWriter:
    """The three copy-pasted persistence loops are now one call each.

    A capability is not shipped until a test fails when its wiring is removed,
    so this asserts the CALL SITES, not just that the method exists.
    """

    def test_all_three_labeling_paths_call_it(self):
        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert source.count("self._persist_label_outcome(") == 3, (
            "the local, OpenAI and OpenAI-compatible paths must each persist "
            "through the single writer"
        )

    def test_no_path_fabricates_a_label_for_a_failure(self):
        """The three sites that wrote `category='error_feature'` and
        `name='feature_{n}'` over a crash are gone. That fabrication is what
        made 16,824 failures indistinguishable from finished work."""
        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert '"category": "error_feature"' not in source


class TestSuccessfulVerdict:

    def test_records_the_verdict_and_the_judge(self, writer, job):
        feature = _feature()
        status = _persist(writer, job, feature,
                          {"category": "semantic", "specific": "legal_terms",
                           "description": "d"})
        assert status == "succeeded"
        assert feature.label_status == "succeeded"
        assert feature.name == "legal_terms"
        assert feature.category == "semantic"
        assert feature.labeled_at == LABELED_AT
        assert feature.label_attempts == 1
        assert feature.label_error is None
        # Provenance: without both, a resume can only ask whether a verdict
        # exists, not whether it is still the one this judge would produce.
        assert feature.label_prompt_fingerprint == "f" * 64
        assert feature.label_model == "gemma-4-31b-GGUF:IQ4_XS"

    def test_a_refusal_is_a_verdict_not_a_failure(self, writer, job):
        """`_enforce_refusal` produces `uninterpretable` deliberately when the
        fit ratio is below 0.5. It is an adjudication and must never be redone —
        treating it as a gap turns resume into a relabel-everything button."""
        feature = _feature()
        status = _persist(writer, job, feature,
                          {"category": "uninterpretable", "specific": "no_pattern",
                           "description": ""})
        assert status == "succeeded"
        assert feature.label_status == "succeeded"

    def test_clears_a_previous_failure(self, writer, job):
        feature = _feature(label_status="failed", label_error="a previous crash",
                           label_error_at=LABELED_AT, label_attempts=2)
        _persist(writer, job, feature,
                 {"category": "semantic", "specific": "ok", "description": ""})
        assert feature.label_status == "succeeded"
        assert feature.label_error is None
        assert feature.label_error_at is None
        assert feature.label_attempts == 3


# Every shape a judge failure actually arrives in. Parametrised over all of
# them, not one representative: the point of this work is that a failure is
# never again silent, and a single spot-check cannot establish that.
FAILURE_LABELS = {
    "provider_exception": {"category": "error_feature", "specific": "feature_42",
                           "description": "", "error": "APIError: connection reset"},
    "rate_limited": {"category": "rate_limited", "specific": "feature_42",
                      "error": "rate limited by the judge endpoint: 429"},
    "empty_examples": {"category": "empty_features", "specific": "feature_42",
                        "error": "no activating examples were retrieved for this feature"},
    "unparseable_response": {"category": "uncategorized", "specific": "feature_42",
                              "error": "the judge's response could not be parsed as a label"},
    "batch_gap": {"category": "error_feature", "specific": "feature_42",
                   "error": "the batch judge returned no result for this feature"},
    "raised_exception": RuntimeError("the judge process died"),
    "no_label_at_all": {"category": "semantic", "specific": ""},
}


class TestFailureIsNeverSilent:

    @pytest.mark.parametrize("shape", sorted(FAILURE_LABELS))
    def test_records_failed_with_a_reason(self, writer, job, shape):
        feature = _feature()
        status = _persist(writer, job, feature, FAILURE_LABELS[shape])
        assert status == "failed", f"{shape} was not recorded as a failure"
        assert feature.label_status == "failed"
        assert feature.label_error, f"{shape} recorded no reason"
        assert feature.label_error_at == LABELED_AT

    @pytest.mark.parametrize("shape", sorted(FAILURE_LABELS))
    def test_fabricates_no_verdict(self, writer, job, shape):
        """A failure writes NO name, NO category and NO `labeled_at`. The
        feature keeps its placeholder and is truthfully still unlabeled — which
        is exactly what the old code destroyed."""
        feature = _feature(name="feature_42", category=None)
        _persist(writer, job, feature, FAILURE_LABELS[shape])
        assert feature.category is None, f"{shape} fabricated a category"
        assert feature.name == "feature_42", f"{shape} fabricated a name"
        assert feature.labeled_at is None, (
            f"{shape} stamped labeled_at, which is what made a failure look finished"
        )

    def test_counts_the_attempt(self, writer, job):
        feature = _feature(label_attempts=4)
        _persist(writer, job, feature, FAILURE_LABELS["provider_exception"])
        assert feature.label_attempts == 5, (
            "an uncounted attempt lets a permanently-broken feature be retried forever"
        )

    def test_truncates_a_runaway_reason(self, writer, job):
        feature = _feature()
        _persist(writer, job, feature,
                 {"category": "error_feature", "specific": "x", "error": "y" * 50_000})
        assert len(feature.label_error) == LabelingService.MAX_LABEL_ERROR_CHARS


class TestAquaIsSkippedNotLabeled:

    def test_records_skipped_and_leaves_the_label_alone(self, writer, job):
        """The aqua star is a user-visible promise that a hand-verified label
        survives a bulk run."""
        feature = _feature(star_color="aqua", name="hand_verified",
                           category="semantic", label_status="succeeded")
        status = _persist(writer, job, feature,
                          {"category": "noise", "specific": "overwritten",
                           "description": ""})
        assert status == "skipped"
        assert feature.name == "hand_verified"
        assert feature.category == "semantic"

    def test_does_not_spend_an_attempt(self, writer, job):
        """Nothing was asked of the judge, so nothing was attempted."""
        feature = _feature(star_color="aqua", label_attempts=0)
        _persist(writer, job, feature, {"category": "x", "specific": "y"})
        assert feature.label_attempts == 0

    def test_a_skip_is_not_a_success(self, writer, job):
        """Skips used to be `logger.debug`'d and then counted as labeled,
        inflating `features_labeled`."""
        feature = _feature(star_color="aqua")
        assert _persist(writer, job, feature, {"category": "x", "specific": "y"}) != "succeeded"


class TestStatisticsCountWhatWasWritten:

    def test_success_count_no_longer_reads_label_strings(self):
        """`successfully_labeled` tested `specific.startswith("feature_")`, so a
        judge that crashed and a judge that legitimately named a feature
        `feature_store_checkout` scored identically."""
        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert 'startswith("feature_")' not in source
        assert "successfully_labeled = outcome_counts[self.LABEL_STATUS_SUCCEEDED]" in source

    def test_features_labeled_means_labeled_not_attempted(self):
        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert "labeling_job.features_labeled = successfully_labeled" in source
        assert "labeling_job.features_labeled = len(labels)" not in source


class TestEveryJudgeFailurePathCarriesItsReason:
    """The writer keys on `error`, so a judge service that omits it produces a
    failure recorded as a VERDICT — the original defect, reintroduced one file
    away.

    The tests above use synthetic label dicts and cannot see that: mutation
    control C15 (dropping `error` from a real OpenAI failure return) left them
    all green. This reads the actual source instead.
    """

    #: Categories that only ever arise from a failure. `uninterpretable` is
    #: deliberately absent — it is a verdict.
    FAILURE_CATEGORIES = {"error_feature", "rate_limited", "empty_features", "uncategorized"}

    SERVICES = (
        "src/services/local_labeling_service.py",
        "src/services/openai_labeling_service.py",
        "src/services/labeling_service.py",
    )

    def _failure_dicts(self):
        import ast
        import pathlib

        found = []
        for path in self.SERVICES:
            tree = ast.parse(pathlib.Path(path).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Dict):
                    continue
                keys = {
                    k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                category = next(
                    (
                        v.value for k, v in zip(node.keys, node.values)
                        if isinstance(k, ast.Constant) and k.value == "category"
                        and isinstance(v, ast.Constant)
                    ),
                    None,
                )
                if category in self.FAILURE_CATEGORIES:
                    found.append((path, node.lineno, category, keys))
        return found

    def test_the_scan_finds_the_failure_returns(self):
        """A source scan that matches nothing asserts nothing, and this repo has
        shipped guards that failed open exactly that way — twice in one arc."""
        found = self._failure_dicts()
        assert len(found) >= 10, (
            f"expected the judge services' failure returns; found {len(found)}. "
            "If they were refactored, retarget this scan — do not delete it."
        )

    def test_every_failure_return_carries_an_error(self):
        missing = [
            f"{path}:{line} category={category!r}"
            for path, line, category, keys in self._failure_dicts()
            if "error" not in keys
        ]
        assert not missing, (
            "these failure returns carry no reason, so `label_error` would be "
            "empty and the failure would be recorded as a verdict: "
            + "; ".join(missing)
        )


class TestClaimingPreventsDuplicateWork:
    """`in_progress` is excluded from eligibility. Nothing wrote it.

    An exclusion with no writer is decorative: two resumes started against the
    same extraction would compute the same batch from the same `pending` rows
    and the estate would pay twice, at ~8 s a feature, for one result. The plan
    called for claiming; this asserts it actually happens.
    """

    def test_taking_a_batch_marks_it_in_progress(self, writer):
        features = [_feature(id=f"f{i}", neuron_index=i) for i in range(3)]
        claimed = writer._claim_features(features, LABELED_AT)
        assert claimed == 3
        assert all(f.label_status == "in_progress" for f in features)

    def test_an_aqua_feature_is_not_claimed(self, writer):
        """It is going to be skipped. Moving it through `in_progress` would make
        it briefly indistinguishable from work in flight to anything reading
        coverage."""
        aqua = _feature(id="f_aqua", star_color="aqua", label_status="succeeded")
        normal = _feature(id="f_normal")
        assert writer._claim_features([aqua, normal], LABELED_AT) == 1
        assert aqua.label_status == "succeeded"
        assert normal.label_status == "in_progress"

    def test_a_claim_is_committed_before_the_judge_is_called(self, writer):
        """Uncommitted, the claim is invisible to the other worker's session and
        buys nothing."""
        writer._claim_features([_feature()], LABELED_AT)
        assert writer.db.commit.called

    def test_every_labeling_path_claims_before_asking_the_judge(self):
        """Reachability: three paths persist outcomes, so three must claim."""
        import inspect

        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert source.count("self._claim_features(batch_features") == 3
        # And the claim precedes the persist in each path, not the other way
        # round — claiming after the work is done protects nothing.
        assert source.index("self._claim_features(") < source.index(
            "self._persist_label_outcome("
        )


class TestALeftoverCancelRequestCannotOutliveItsJob:
    """`cancel_requested_at` is set by the endpoint and cleared by nobody in the
    labeling path.

    A job that was cancelled and is then resumed reads its predecessor's request
    and abandons on the first guard tick — a resume that reports "cancelled"
    without doing anything, every time. The same omission once meant a cancelled
    model download could never be downloaded again.
    """

    def test_the_task_clears_it_before_any_work(self):
        import inspect

        from src.workers import labeling_tasks

        source = inspect.getsource(labeling_tasks.label_features_task)
        assert 'clear_cancel_request("labeling"' in source, (
            "the labeling task never clears a stale cancel request"
        )
        # BEFORE the work, not after: clearing it afterwards leaves the whole
        # run exposed to a request the previous run earned.
        #
        # Matched on the CALL, not the bare name — the docstring names the
        # method 3 KB earlier, and the looser match compared the clear against
        # a mention in prose and failed for a reason that had nothing to do with
        # the code.
        call = "labeling_service.label_features_for_extraction("
        assert call in source
        assert source.index("clear_cancel_request(\"labeling\"") < source.index(call)


class TestTheTemplateResponseCarriesItsFingerprint:
    """WS4. Without this the staleness filter has NO CALLER — the UI has nothing
    to send to `?prompt_fingerprint=`, and a capability wired to nothing is not
    shipped, however well it is tested in isolation."""

    def _orm(self, **overrides):
        import datetime

        base = dict(
            id="lpt_x", name="n", description=None,
            system_message="s", user_prompt_template="u",
            temperature=0.3, max_tokens=50, top_p=0.9,
            template_type="legacy", max_examples=10,
            include_prefix=True, include_suffix=True, prime_token_marker="<<>>",
            include_logit_effects=False,
            top_promoted_tokens_count=None, top_suppressed_tokens_count=None,
            include_negative_examples=True, num_negative_examples=5,
            is_detection_template=False, include_nlp_analysis=False,
            # Set explicitly: SQLAlchemy `default=` applies at INSERT, so an
            # unflushed ORM object reads None for a NOT NULL column and the
            # response schema — correctly — refuses it. This fixture stands in
            # for a PERSISTED row, so it carries persisted values.
            example_sampling="top_k", activation_display="absolute",
            is_default=False, is_system=False, created_by=None,
            created_at=datetime.datetime.now(), updated_at=datetime.datetime.now(),
        )
        base.update(overrides)
        return LabelingPromptTemplate(**base)

    def _response(self, **overrides):
        from src.schemas.labeling_prompt_template import LabelingPromptTemplateResponse

        return LabelingPromptTemplateResponse.model_validate(self._orm(**overrides))

    def test_the_response_fingerprint_matches_the_writer(self):
        """The one assertion that matters. A second implementation would drift
        silently — showing verdicts as fresh when the template that produced them
        no longer exists in that form."""
        template = self._orm()
        from src.schemas.labeling_prompt_template import LabelingPromptTemplateResponse

        assert (
            LabelingPromptTemplateResponse.model_validate(template).prompt_fingerprint
            == prompt_fingerprint(template)
        )

    def test_it_is_a_whole_sha256(self):
        assert len(self._response().prompt_fingerprint) == 64

    def test_editing_the_prompt_changes_it(self):
        assert (
            self._response().prompt_fingerprint
            != self._response(system_message="different").prompt_fingerprint
        )

    def test_renaming_the_template_does_not(self):
        """Nothing the judge saw has changed, so the verdicts it produced are
        still current."""
        assert (
            self._response().prompt_fingerprint
            == self._response(name="Renamed").prompt_fingerprint
        )

    def test_the_response_exposes_every_field_the_fingerprint_needs(self):
        """`prompt_fingerprint` reads its fields by NAME off whatever it is
        given. If a prompt-affecting column is dropped from this schema the hash
        would change meaning rather than fail — so the field sets must agree."""
        from src.schemas.labeling_prompt_template import LabelingPromptTemplateResponse

        needed = set(prompt_fingerprint_fields(self._orm()))
        exposed = set(LabelingPromptTemplateResponse.model_fields)
        missing = needed - exposed
        assert not missing, (
            f"the response omits prompt-affecting fields {sorted(missing)}, so "
            "its fingerprint is computed over a different template than the "
            "writer's"
        )


class TestTheJunkFilterRecordsWhatItDrops:
    """R4 HARDWARE FINDING. Static review could not see this one.

    A pre-labeling filter removes features whose prime tokens are predominantly
    punctuation or whitespace. Those features were left `pending` with
    `label_attempts` still 0 — so every resume offered them again, every sweep
    batch re-selected them, and the attempt cap could NEVER engage because no
    attempt was ever recorded.

    Observed on the L46 extraction: a sweep batch of 5 labelled 1 and silently
    dropped 4, twice in a row, and would have spent its entire ceiling doing
    that. 38,299 pending features on that extraction all carry 0 attempts.

    It is the same shape as the aqua skip — a deliberate decision NOT to ask the
    judge, which is an outcome and must be written down.
    """

    def test_dropped_features_are_recorded_as_skipped(self, writer):
        dropped = [_feature(id=f"junk{i}", neuron_index=i) for i in range(3)]
        assert writer._persist_filtered_out(dropped, LABELED_AT) == 3
        for f in dropped:
            assert f.label_status == "skipped"

    def test_they_carry_a_reason_an_operator_can_read(self, writer):
        """`skipped` alone says a decision was made; the reason says which."""
        f = _feature()
        writer._persist_filtered_out([f], LABELED_AT)
        assert f.label_error and "punctuation" in f.label_error

    def test_skipped_is_adjudicated_so_they_stop_being_offered(self):
        """The whole point: ineligible from here on."""
        from src.services.labeling_eligibility import (
            ADJUDICATED_STATUSES,
            RETRYABLE_STATUSES,
        )

        assert "skipped" in ADJUDICATED_STATUSES
        assert "skipped" not in RETRYABLE_STATUSES

    def test_an_aqua_feature_keeps_its_own_reason(self, writer):
        """Aqua is already skipped, for a better reason — a hand-verified label
        being protected. Overwriting that record would lose it."""
        aqua = _feature(star_color="aqua", label_status="skipped", label_error=None)
        writer._persist_filtered_out([aqua], LABELED_AT)
        assert aqua.label_error is None

    def test_nothing_dropped_writes_nothing(self, writer):
        assert writer._persist_filtered_out([], LABELED_AT) == 0
        assert not writer.db.commit.called

    def test_the_labeling_path_actually_calls_it(self):
        """Reachability. The writer existing changes nothing on its own — this
        defect WAS a working filter whose outcome nobody recorded."""
        import inspect

        from src.services.labeling_service import LabelingService

        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert "self._persist_filtered_out(" in source
        # …and it must run against what the filter REMOVED, not what it kept.
        #
        # Pinned to the PROPERTY rather than one spelling of it: this asserted
        # the exact comprehension `[f for f in _before_filter if f not in
        # features]`, and broke when that `in` scan — O(N) identity comparison
        # per element, ~1.4 billion comparisons at 53,088 features — was
        # replaced by an id set. The membership test changed; what it computes
        # did not.
        assert "_before_filter" in source, (
            "nothing captures the pre-filter feature list, so the dropped set "
            "cannot be derived at all"
        )
        assert "_kept_ids" in source and "id(f) not in _kept_ids" in source, (
            "the filtered-out set is no longer the complement of what the "
            "filter kept"
        )


# ── a judge that cannot answer is not the feature's failure ──────────────────

from src.services.labeling_judge_health import (  # noqa: E402
    JudgeUnavailable,
    is_job_level_failure,
    judge_identity,
)

#: The exact string production wrote 54 times, once per feature, in fourteen
#: seconds — spending a retry each time.
LIVE_404 = (
    "NotFoundError: Error code: 404 - {'error': {'message': \"The model "
    "'granite-3.3-8b-instruct' does not exist or has not been downloaded. "
    "Download it first.\"}}"
)


class TestJobLevelFailuresAreNotFeatureFailures:
    """REPORTED FROM THE UI, and the deeper of the two bugs behind it.

    Two resume clicks on an extraction whose March-era judge had since been
    removed from the server took 39 features from one attempt to three in
    fourteen seconds of 404s. A third click then reported "Resume 0 of 39" and
    refused to act, because those features had used up retries they never
    actually received.

    Nothing was learned about any of them. The judge was simply absent.
    """

    @pytest.mark.parametrize("reason", [
        LIVE_404,
        "AuthenticationError: 401 - invalid api key",
        "APIConnectionError: connection refused",
        "PermissionDeniedError: 403",
    ])
    def test_these_describe_the_judge(self, reason):
        assert is_job_level_failure(reason)

    @pytest.mark.parametrize("reason", [
        "the judge's response could not be parsed as a label",
        "no activating examples were retrieved for this feature",
        "ReadTimeout: timed out after 120.0s waiting on feat_abc",
        "rate limited by the judge endpoint: 429",
        "(reason not recorded: this failure predates per-feature error capture)",
    ])
    def test_these_describe_the_feature(self, reason):
        """A false positive here aborts a job that should carry on — a slow or
        unparseable feature is a fact about that feature, not the judge."""
        assert not is_job_level_failure(reason)

    def test_no_reason_is_not_a_judge_failure(self):
        assert not is_job_level_failure(None)
        assert not is_job_level_failure("")

    # The classifier has TWO independent mechanisms — the exception TYPE and a
    # message MARKER — and the live 404 happens to match both. So a test using
    # only that string cannot tell them apart, and mutation control C63
    # (deleting `NotFoundError` from the type set) SURVIVED it.
    #
    # These pin each mechanism alone. If the SDK ever rewords its 404, the type
    # check is the only thing left standing.

    @pytest.mark.parametrize("reason", [
        "NotFoundError: 404 - unknown model",          # no marker phrase
        "AuthenticationError: 401",
        "APIConnectionError: no route to host",
    ])
    def test_the_type_alone_is_enough(self, reason):
        assert is_job_level_failure(reason), (
            "classification is relying on the message text; a reworded error "
            "of the same type would be blamed on the feature"
        )

    @pytest.mark.parametrize("reason", [
        "SomeNewSDKError: The model 'x' does not exist or has not been downloaded.",
        "RuntimeError: model not found",
    ])
    def test_the_message_alone_is_enough(self, reason):
        assert is_job_level_failure(reason), (
            "classification is relying on the exception type; an unfamiliar "
            "wrapper around the same failure would be blamed on the feature"
        )

    def test_the_writer_refuses_to_record_it_against_a_feature(self, writer, job):
        feature = _feature(label_attempts=1)
        with pytest.raises(JudgeUnavailable):
            _persist(writer, job, feature,
                     {"category": "error_feature", "specific": "x", "error": LIVE_404})

        assert feature.label_status == "pending", "the feature was blamed for the judge"
        assert feature.label_error is None
        assert feature.labeled_at is None

    def test_it_spends_no_retry(self, writer, job):
        """THE damage. At 14,000 features one wrong endpoint would exhaust an
        entire extraction in about a minute, permanently."""
        feature = _feature(label_attempts=1)
        with pytest.raises(JudgeUnavailable):
            _persist(writer, job, feature,
                     {"category": "error_feature", "specific": "x", "error": LIVE_404})

        assert feature.label_attempts == 1, (
            "a judge-level failure consumed one of the feature's retries"
        )

    def test_a_real_feature_failure_still_counts(self, writer, job):
        """The distinction has to cut both ways, or failures stop being recorded."""
        feature = _feature(label_attempts=1)
        status = _persist(writer, job, feature, {
            "category": "uncategorized", "specific": "x",
            "error": "the judge's response could not be parsed as a label",
        })
        assert status == "failed"
        assert feature.label_attempts == 2
        assert feature.label_error


class TestThePreflightCatchesItBeforeAnyFeature:

    def test_it_refuses_a_judge_the_endpoint_does_not_serve(self, writer):
        import httpx

        job = LabelingJob(
            id="lbl_x", extraction_job_id="ext_1",
            openai_compatible_endpoint="http://millm.invalid/v1",
            openai_compatible_model="granite-3.3-8b-instruct",
        )
        payload = {"data": [{"id": "granite-4.2-8b"}, {"id": "granite-4.0-1b"}]}
        with patch.object(httpx, "get", return_value=MagicMock(
            raise_for_status=lambda: None, json=lambda: payload
        )):
            with pytest.raises(JudgeUnavailable, match="granite-3.3-8b-instruct"):
                writer._preflight_judge(job)

    def test_it_names_what_is_available(self, writer):
        """An operator has to be able to act on it without a second query."""
        import httpx

        job = LabelingJob(
            id="lbl_x", extraction_job_id="ext_1",
            openai_compatible_endpoint="http://millm.invalid/v1",
            openai_compatible_model="gone",
        )
        payload = {"data": [{"id": "granite-4.2-8b"}]}
        with patch.object(httpx, "get", return_value=MagicMock(
            raise_for_status=lambda: None, json=lambda: payload
        )):
            with pytest.raises(JudgeUnavailable, match="granite-4.2-8b"):
                writer._preflight_judge(job)

    def test_it_passes_a_judge_that_is_served(self, writer):
        import httpx

        job = LabelingJob(
            id="lbl_x", extraction_job_id="ext_1",
            openai_compatible_endpoint="http://millm.invalid/v1",
            openai_compatible_model="granite-4.2-8b",
        )
        payload = {"data": [{"id": "granite-4.2-8b"}]}
        with patch.object(httpx, "get", return_value=MagicMock(
            raise_for_status=lambda: None, json=lambda: payload
        )):
            writer._preflight_judge(job)  # must not raise

    def test_an_unreachable_check_does_not_block_the_job(self, writer):
        """A preflight that stops work because it could not confirm anything is
        worse than no preflight."""
        import httpx

        job = LabelingJob(
            id="lbl_x", extraction_job_id="ext_1",
            openai_compatible_endpoint="http://millm.invalid/v1",
            openai_compatible_model="anything",
        )
        with patch.object(httpx, "get", side_effect=OSError("unreachable")):
            writer._preflight_judge(job)  # must not raise

    def test_it_is_skipped_when_there_is_nothing_to_check(self, writer):
        writer._preflight_judge(LabelingJob(id="l", extraction_job_id="e"))

    def test_it_runs_before_any_feature_is_touched(self):
        import inspect

        from src.services.labeling_service import LabelingService

        source = inspect.getsource(LabelingService.label_features_for_extraction)
        assert source.index("_preflight_judge(") < source.index("_claim_features(")


class TestJudgeIdentity:
    def test_it_finds_whichever_method_the_job_uses(self):
        assert judge_identity({"openai_compatible_model": "a"}) == "a"
        assert judge_identity({"openai_model": "b"}) == "b"
        assert judge_identity({"local_model": "c"}) == "c"
        assert judge_identity({}) is None
        assert judge_identity(None) is None
