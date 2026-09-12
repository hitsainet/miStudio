"""`run_trial` is driven end-to-end, against a REAL PostgreSQL.

WHY THIS FILE EXISTS. Round 4 mutated two of round 3's fixes inside `run_trial`
and the full 4545-test suite stayed green on both:

  M12  n_negative = resolve_num_negative(frozen)  ->  n_negative = 0
  M19  prompt_sample_indices={...}                ->  prompt_sample_indices={}

Nothing in the suite executes `run_trial`. `test_labeling_trial_task_session.py`
substitutes its own `run_trial`, and the wiring guards in
`test_sampling_wiring_reachable.py` read the SOURCE for a kwarg NAME — which is
present under both mutations. A source scrape fails open; this file runs the
function.

What each mutation costs:

  M12 makes the contrast arm byte-identical to the baseline. That is the
      "manufactured null" this arc already paid for once: `compare` reports
      `identical`, the paired delta is ~0 with a CI containing zero, and the
      arm was never an arm.

  M19 puts the contrast passages back into the pool the label is scored
      against. The judge was told those passages "also activate the feature";
      at scoring time it is asked whether the label describes one, says yes,
      and is marked WRONG. Only a contrast arm has such passages, so it biases
      the experiment's answer toward "contrast does not help".
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DB = os.environ.get(
    "DATABASE_URL_SYNC", "postgresql://postgres:devpassword@localhost:5432/mistudio"
)

_ENUMS = [
    ("export_status", ["pending", "computing", "packaging", "completed", "failed", "cancelled"]),
    ("label_source_enum", ["auto", "user", "llm", "local_llm", "openai", "enhanced_llm", "mcp_agent"]),
    ("analysis_type_enum", ["logit_lens", "correlations", "ablation", "nlp_analysis"]),
    ("extraction_status_enum", ["queued", "loading", "extracting", "saving", "completed", "failed", "cancelled"]),
]

#: More stored rows than the prompt shows, so there is a genuine weak tail for
#: the contrast block to draw from. With 10 shown and 5 contrast rows, ranks
#: 26-30 are the contrast set and cannot overlap the top 10.
STORED_ROWS = 30
SHOWN = 10
NEGATIVES = 5


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(DB)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"PostgreSQL is unreachable: {exc}")

    from src.core.database import Base
    from src import models  # noqa: F401 - registers every table

    with eng.begin() as c:
        for name, values in _ENUMS:
            vals = ", ".join(f"'{v}'" for v in values)
            c.execute(text(
                f"DO $$ BEGIN CREATE TYPE {name} AS ENUM ({vals}); "
                f"EXCEPTION WHEN duplicate_object THEN NULL; END $$;"))
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(engine):
    """A Session on an outer transaction that is always rolled back.

    `run_trial` commits. `join_transaction_mode="create_savepoint"` turns those
    commits into savepoint releases inside the outer transaction, so the
    function runs for real and this test still writes nothing.
    """
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        conn.close()


@pytest.fixture
def trial(db):
    """One extraction, one feature with a real activation tail, one queued trial."""
    from src.models.external_sae import ExternalSAE
    from src.models.extraction_job import ExtractionJob
    from src.models.feature import Feature
    from src.models.feature_activation import FeatureActivation
    from src.models.labeling_job import LabelingJob, LabelingMode, LabelingStatus
    from src.models.labeling_prompt_template import LabelingPromptTemplate
    from src.models.labeling_trial_run import LabelingTrialRun
    from src.services.labeling_trial_service import freeze_template, panel_id_for

    tag = uuid.uuid4().hex[:8]
    ext_id = f"extr_test_{tag}"
    feat_id = f"feat_test_{tag}_00001"

    sae_id = f"sae_{tag}"
    db.add(ExternalSAE(id=sae_id, name=f"sae {tag}", source="trained"))
    db.add(ExtractionJob(id=ext_id, external_sae_id=sae_id,
                         status="completed", config={}))

    db.add(Feature(
        id=feat_id, extraction_job_id=ext_id, external_sae_id=sae_id,
        neuron_index=1,
        name="unlabeled", max_activation=30.0,
        activation_frequency=0.01, mean_activation=15.0,
        interpretability_score=0.5,
    ))

    # Strictly decreasing activations, so rank is unambiguous and the contrast
    # rows are provably weaker than everything shown.
    for i in range(STORED_ROWS):
        db.add(FeatureActivation(
            feature_id=feat_id,
            sample_index=1000 + i,
            max_activation=float(STORED_ROWS - i),
            prime_token="alpha",
            prefix_tokens=["the", "quick"],
            suffix_tokens=["brown", "fox"],
            tokens=["the", "quick", "alpha", "brown", "fox"],
            activations=[0.0, 0.0, float(STORED_ROWS - i), 0.0, 0.0],
            prime_activation_index=2,
        ))

    template = LabelingPromptTemplate(
        id=f"tmpl_{tag}",
        name=f"contrast-arm-{tag}",
        user_prompt_template="{examples_block}",
        system_message="Answer with JSON.",
        max_examples=SHOWN,
        # THE VARIABLE UNDER TEST: this template asks for a contrast block.
        include_negative_examples=True,
        num_negative_examples=NEGATIVES,
    )
    db.add(template)
    db.flush()

    job_id = f"trial_{ext_id}"
    run_id = f"ltr_{tag}"
    db.add(LabelingJob(
        id=job_id, extraction_job_id=ext_id,
        labeling_method="openai_compatible",
        openai_compatible_endpoint="http://judge.invalid/v1",
        prompt_template_id=template.id,
        mode=LabelingMode.TRIAL.value,
        feature_ids=[feat_id],
        trial_run_id=run_id,
        status=LabelingStatus.QUEUED.value,
        total_features=1, max_tokens=300,
    ))
    db.add(LabelingTrialRun(
        id=run_id, labeling_job_id=job_id, extraction_job_id=ext_id,
        prompt_template_id=template.id,
        panel_id=panel_id_for(ext_id, [feat_id]),
        status="queued",
        payload={
            "panel": {"panel_id": panel_id_for(ext_id, [feat_id]),
                      "extraction_job_id": ext_id,
                      "feature_ids": [feat_id], "size": 1},
            "prompt": freeze_template(template),
            "config": {"labeling_method": "openai_compatible",
                       "model": "judge-x", "batch_size": 1},
            "results": [], "stats": {},
        },
    ))
    db.flush()
    return {"run_id": run_id, "job_id": job_id,
            "feature_id": feat_id, "extraction_job_id": ext_id}


def _drive(db, trial, monkeypatch):
    """Run the trial with a stub judge, capturing every `_score_detection` call."""
    from src.services import labeling_trial_service as mod

    calls = []

    def _fake_score(self, **kwargs):
        calls.append(kwargs)
        return {"scored": False, "reason": "stubbed for this test",
                "coverage": {"scored": 0, "skipped": 1, "panel_size": 1}}

    monkeypatch.setattr(
        mod.LabelingTrialService, "_score_detection", _fake_score
    )

    async def _fake_label(**kwargs):
        _fake_label.seen.append(kwargs)
        return {"category": "semantic", "specific": "alpha_token",
                "description": "d", "fit_count": "9/10", "confidence": "high"}

    _fake_label.seen = []

    class _FakeLabeler:
        def __init__(self, *a, **kw):
            pass

        async def generate_label_from_examples(self, **kwargs):
            return await _fake_label(**kwargs)

    monkeypatch.setattr(
        "src.services.openai_labeling_service.OpenAILabelingService", _FakeLabeler
    )

    mod.LabelingTrialService(db).run_trial(trial["job_id"])
    return calls, _fake_label.seen


class TestTheContrastArmIsActuallyAnArm:
    """M12. `n_negative = 0` makes the contrast arm identical to the baseline."""

    def test_the_judge_is_sent_the_contrast_passages(self, db, trial, monkeypatch):
        _, sent = _drive(db, trial, monkeypatch)

        assert len(sent) == 1, "the panel is one feature; expected one judge call"
        negatives = sent[0].get("negative_examples") or []
        assert len(negatives) == NEGATIVES, (
            f"the template asked for {NEGATIVES} contrast passages and the judge "
            f"was sent {len(negatives)}. At zero the contrast arm is "
            f"byte-identical to the baseline and the comparison is a "
            f"manufactured null."
        )

    def test_the_contrast_passages_are_the_weak_tail_not_the_shown_rows(
        self, db, trial, monkeypatch
    ):
        """NEGATIVE CONTROL for the test above.

        Passing the top rows back as 'negatives' would satisfy a count-only
        assertion while showing the judge the same passage twice — once as
        evidence and once as contrast.
        """
        _, sent = _drive(db, trial, monkeypatch)
        shown = {e["sample_index"] for e in sent[0]["examples"]}
        contrast = {e["sample_index"] for e in sent[0]["negative_examples"]}

        assert not (shown & contrast), (
            f"{len(shown & contrast)} passage(s) appear as BOTH evidence and "
            f"contrast in one prompt"
        )
        weakest_shown = min(e["max_activation"] for e in sent[0]["examples"])
        strongest_contrast = max(e["max_activation"] for e in sent[0]["negative_examples"])
        assert strongest_contrast < weakest_shown, (
            "the contrast block must be drawn from BELOW everything shown"
        )


class TestTheRulerExcludesTheContrastPassages:
    """M19. The scoring exclusion set must cover the contrast block."""

    def test_both_rulers_exclude_every_passage_the_prompt_showed(
        self, db, trial, monkeypatch
    ):
        calls, sent = _drive(db, trial, monkeypatch)

        assert len(calls) == 2, (
            f"expected two rulers (top-K and mid-range), got {len(calls)}"
        )

        shown = {e["sample_index"] for e in sent[0]["examples"]}
        contrast = {e["sample_index"] for e in sent[0]["negative_examples"]}
        feat = trial["feature_id"]

        for i, call in enumerate(calls):
            excluded = set(call["prompt_sample_indices"].get(feat, ()))
            missing_contrast = contrast - excluded
            assert not missing_contrast, (
                f"ruler {i}: {len(missing_contrast)} contrast passage(s) were "
                f"not excluded from the negative pool. The judge was told they "
                f"activate the feature; scoring will ask whether the label "
                f"describes one, get 'yes', and mark it WRONG — biasing the "
                f"contrast arm downward by construction."
            )
            missing_shown = shown - excluded
            assert not missing_shown, (
                f"ruler {i}: {len(missing_shown)} passage(s) shown to the judge "
                f"were not excluded from the pool it is scored against"
            )
