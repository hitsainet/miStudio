"""G38: pooling skips the chat template's fixed prefix (operator decisions G38-1..3, 2026-11-08).

The first P positions of every row are the same tokens in the same context — the start token, the
template's preamble, the first user header — so a decoder that attends only to earlier positions gives them identical activations in
every row and their token scores are a constant. Under `max` that constant is a FLOOR: the D128
digits probe `pm_87db6a6fc80e` sat on exactly 17.65 for 271 of 1,443 test rows and trained to chance
(validation AUROC 0.499). This file proves, on a tiny REAL model:

* the floor exists without the flag and is gone with it (`TestTheFloorIsGone`);
* a probe and a document without the flag are scored and serialised exactly as before;
* the contract still refuses unknown fields and accepts this one, and a skipping document's vectors
  each state their P;
* a new run defaults the flag on, an old run row reads it off;
* the stale-date refresh, truncation, the render summary and the calibration lengths.

The pipeline-level wiring (capture, training, calibration, evaluation, the builder, the offline
score and the forecast on a real run) is `test_template_prefix_pipeline.py`. Mutation controls are
recorded in `0xcc/reviews/g38_template_prefix_2026-11-08.md`.
"""
from __future__ import annotations

import copy
import json
import uuid
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from src.services.probe_monitor_render import (
    TEMPLATE_PREFIX_SCOPES,
    render_all,
    render_messages,
    template_prefix_ids,
)

D_MODEL = 32
LAYER = 1
WORDS = ["<unk>", "<s>", "<turn>", "</turn>", "user", "assistant", "system", "today", "x", "y"]
WORDS += [f"w{i}" for i in range(40)]
#: Llama-3.1-shaped: a start token, an injected system block, then the first user header — P = 7.
TEMPLATE = (
    "<s> <turn> system today </turn> {% for m in messages %}<turn> {{ m['role'] }} "
    "{{ m['content'] }} </turn> {% endfor %}{% if add_generation_prompt %}<turn> assistant {% endif %}"
)
PREFIX = ["<s>", "<turn>", "system", "today", "</turn>", "<turn>", "user"]


def _tokenizer(template=TEMPLATE, words=WORDS):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>")
    tokenizer.chat_template = template
    tokenizer.pad_token = "<unk>"
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return _tokenizer()


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=64, hidden_size=D_MODEL, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
        )
    )
    model.eval()
    return model


def _rows(n=40, seed=3):
    import random

    rng = random.Random(seed)
    return [
        [{"role": "user", "content": " ".join(f"w{rng.randrange(40)}" for _ in range(6))}]
        for _ in range(n)
    ]


# ── the floor ────────────────────────────────────────────────────────────────────────────────


class TestTheFloorIsGone:
    """`max` over a row whose prefix holds the maximum returns the SAME number for every row.

    The head reads the direction of the start token's own activation, so the prefix scores high —
    the condition D128 hit by training, set here by construction. The model and tokenizer are real;
    the rows are scored by `forward_scores`, the path evaluation, calibration and the test vectors use.
    """

    @pytest.fixture(scope="class")
    def head(self, tiny_lm, tokenizer):
        from src.ml.probe_monitor_model import ProbeHead

        ids = render_messages(tokenizer, _rows(1)[0]).input_ids
        with torch.no_grad():
            hidden = tiny_lm(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        start = hidden[LAYER + 1][0, 0]          # the decoder layer's output at position 0
        return ProbeHead(
            weight=start / start.norm(), bias=0.0,
            mean=torch.zeros(D_MODEL), std=torch.ones(D_MODEL), layer=LAYER,
        )

    def _aggregates(self, tiny_lm, tokenizer, head, *, skip):
        from src.services.probe_monitor_capture import forward_scores

        rows = [render_messages(tokenizer, r, skip_template_prefix=skip) for r in _rows()]
        scored = forward_scores(tiny_lm, rows, head, rule="max", scope="all", pad_id=0)
        return rows, scored

    def test_without_the_flag_every_row_sits_on_the_same_floor(self, tiny_lm, tokenizer, head):
        rows, scored = self._aggregates(tiny_lm, tokenizer, head, skip=False)
        values = [round(row.aggregate, 5) for row in scored]
        assert Counter(values).most_common(1)[0][1] == len(rows), (
            "the fixture no longer reproduces the floor, so the test below proves nothing: "
            f"{Counter(values).most_common(3)}"
        )
        # And the floor IS the prefix: it is the maximum over [0, P) of every row.
        for row, result in zip(rows, scored):
            assert max(result.token_scores[: len(PREFIX)]) == pytest.approx(result.aggregate)

    def test_with_the_flag_no_score_is_shared_by_more_than_a_few_rows(self, tiny_lm, tokenizer, head):
        rows, scored = self._aggregates(tiny_lm, tokenizer, head, skip=True)
        assert all(row.template_prefix_tokens == len(PREFIX) for row in rows)
        values = [round(result.aggregate, 5) for result in scored]
        assert Counter(values).most_common(1)[0][1] <= 3, Counter(values).most_common(3)
        # Exactly the positions after the prefix were scored.
        assert [r.n_scored for r in scored] == [len(row.input_ids) - len(PREFIX) for row in rows]

    def test_mean_only_moves_by_leaving_the_constant_out(self, tiny_lm, tokenizer, head):
        """G38-2: on for every rule. `mean` loses a constant term, it is not floored."""
        from src.services.probe_monitor_capture import forward_scores

        for messages in _rows(5):
            plain = render_messages(tokenizer, messages)
            skipped = render_messages(tokenizer, messages, skip_template_prefix=True)
            full = forward_scores(tiny_lm, [plain], head, rule="mean", scope="all", pad_id=0)[0]
            cut = forward_scores(tiny_lm, [skipped], head, rule="mean", scope="all", pad_id=0)[0]
            tail = full.token_scores[len(PREFIX):]
            assert cut.aggregate == pytest.approx(sum(tail) / len(tail), abs=1e-5)


# ── old probes and documents are unchanged ───────────────────────────────────────────────────


class TestOldProbesAreUnchanged:
    def test_a_row_rendered_without_the_flag_scores_every_position(self, tokenizer):
        row = render_messages(tokenizer, _rows(1)[0])
        assert row.skip_template_prefix is False and row.template_prefix_tokens == 0
        assert row.scored_mask("all") == [True] * len(row.input_ids)
        assert row.scored_mask("input") == [True] * len(row.input_ids)

    def test_an_old_run_row_reads_the_flag_as_off(self):
        """A run row without the key was trained over the prefix; it must be re-read that way."""
        from src.services.probe_monitor_run import context_from_row

        old = context_from_row(SimpleNamespace(id="pmr_old", config={"scope": "all", "rules": ["max"]}))
        assert old.skip_template_prefix is False
        new = context_from_row(SimpleNamespace(id="pmr_new", config={"skip_template_prefix": True}))
        assert new.skip_template_prefix is True

    def test_an_old_document_round_trips_byte_identically(self):
        """A document exported before G38 — no field anywhere — validates and re-serialises to the
        same bytes, and keeps both digests (`probe-identity-cases.json` pins the digests too)."""
        from src.schemas.probe_definition import ProbeDefinitionV1
        from src.schemas.probe_identity import document_identity
        from src.services.probe_definition_builder import serialise

        document = _document()
        before = serialise(ProbeDefinitionV1.model_validate(document))
        parsed = ProbeDefinitionV1.model_validate(json.loads(before))
        assert parsed.aggregation.skip_template_prefix is False
        assert serialise(parsed) == before
        assert "skip_template_prefix" not in before and "template_prefix_tokens" not in before
        assert document_identity(json.loads(before)) == document_identity(document)

    def test_push_order_guard_an_old_consumer_cannot_reproduce_a_flagged_documents_digest(self):
        """THE PUSH-ORDER GUARD (review round 2). A consumer built before G38 projects the detector
        without the added-after table, so it CANNOT reproduce the detector digest a flagged document
        carries — deployed miLLM refuses such an import with 409 `identity_mismatch` rather than
        serving the probe without the skip. (A document stripped of `identity` has no such check: that
        gap is why "miLLM first" stays the operational rule.) An unflagged document's digest is the
        same under both projections, so nothing exported before G38 is refused."""
        from src.schemas.probe_definition import ProbeDefinitionV1
        from src.schemas.probe_identity import (
            DETECTOR_V1, _project, defaults_table, digest, identity_for,
        )

        def old_consumer(document):
            return digest("detector", _project(document, "", DETECTOR_V1, {}, defaults_table(), "detector"))

        flagged = ProbeDefinitionV1.model_validate(_skipping(_document()))
        carried = identity_for(flagged, 1).detector_digest
        assert old_consumer(json.loads(flagged.model_dump_json())) != carried
        plain = ProbeDefinitionV1.model_validate(_document())
        assert old_consumer(json.loads(plain.model_dump_json())) == identity_for(plain, 1).detector_digest

    def test_the_flag_moves_the_detector_digest_and_not_the_decision(self):
        from src.schemas.probe_identity import document_identity

        plain = _document()
        skipping = _skipping(_document())
        explicit_false = copy.deepcopy(plain)
        explicit_false["aggregation"]["skip_template_prefix"] = False
        assert document_identity(explicit_false) == document_identity(plain)
        assert document_identity(skipping)[0] != document_identity(plain)[0]
        assert document_identity(skipping)[1] == document_identity(plain)[1]


# ── the contract ─────────────────────────────────────────────────────────────────────────────


def _document():
    """A minimal valid residual `all` document (the identity case file's shape)."""
    vectors = [
        {"messages": [{"role": "user", "content": f"row {i}"}], "token_ids": [1, 2, 3, 4],
         "token_scores": [0.1, 0.2, 0.3, 0.4], "score": 0.4, "verdict": None}
        for i in range(8)
    ]
    return {
        "kind": "mistudio.probe-definition/v1", "name": "g38-example",
        "model": {"hf_id": "example-org/example-model", "revision": "0" * 40, "d_model": 4,
                  "n_layers": 8, "architecture": "ExampleForCausalLM"},
        "read": {"layer": 3, "hook_point": "resid_post"},
        "scope": "all", "basis": "residual",
        "head": {"weights": [0.125, -0.5, 1.0, 0.25], "bias": -0.25,
                 "norm_mean": [0.0, 0.5, -0.5, 0.1], "norm_std": [1.0, 2.0, 0.5, 1.5]},
        "aggregation": {"rule": "max", "params": {}, "streamable": True},
        "decision": {"threshold": 2.0, "target_fpr": 0.01},
        "evidence": {"rung": 2, "rung_language": "example", "evaluations": []},
        "test_vectors": {"tolerance": 0.05, "vectors": vectors},
    }


def _skipping(document, prefix=1):
    document = copy.deepcopy(document)
    document["aggregation"]["skip_template_prefix"] = True
    for vector in document["test_vectors"]["vectors"]:
        vector["template_prefix_tokens"] = prefix
        vector["token_scores"] = vector["token_scores"][prefix:]
    return document


class TestTheContract:
    def test_it_accepts_the_flag_and_writes_it_only_when_true(self):
        from src.schemas.probe_definition import Aggregation, ProbeDefinitionV1

        assert Aggregation(rule="max", streamable=True, skip_template_prefix=True).model_dump()[
            "skip_template_prefix"
        ] is True
        assert "skip_template_prefix" not in Aggregation(rule="max", streamable=True).model_dump()
        parsed = ProbeDefinitionV1.model_validate(_skipping(_document()))
        assert parsed.aggregation.skip_template_prefix is True
        dumped = json.loads(parsed.model_dump_json())
        assert dumped["aggregation"]["skip_template_prefix"] is True
        assert {v["template_prefix_tokens"] for v in dumped["test_vectors"]["vectors"]} == {1}

    def test_it_still_refuses_an_unknown_aggregation_field(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import Aggregation

        with pytest.raises(ValidationError, match="extra"):
            Aggregation(rule="max", streamable=True, skip_template_prefixes=True)
        # STRICT (review round 2): `1` or `"true"` is not the boolean the published schema states,
        # and a consumer reading the raw value would disagree with one reading the coerced one.
        for value in ("sometimes", "true", 1, 0):
            with pytest.raises(ValidationError):
                Aggregation(rule="max", streamable=True, skip_template_prefix=value)

    def test_a_skipping_document_whose_vector_states_no_prefix_is_refused(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import ProbeDefinitionV1

        document = _skipping(_document())
        del document["test_vectors"]["vectors"][3]["template_prefix_tokens"]
        with pytest.raises(ValidationError, match="test vector 3 carries no template_prefix_tokens"):
            ProbeDefinitionV1.model_validate(document)

    def test_a_vector_stating_a_prefix_on_a_probe_that_does_not_skip_is_refused(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import ProbeDefinitionV1

        document = _document()
        document["test_vectors"]["vectors"][0]["template_prefix_tokens"] = 0
        with pytest.raises(ValidationError, match="two different readouts"):
            ProbeDefinitionV1.model_validate(document)

    def test_an_all_vector_must_score_exactly_the_positions_after_the_prefix(self):
        from pydantic import ValidationError

        from src.schemas.probe_definition import ProbeDefinitionV1

        document = _skipping(_document())
        document["test_vectors"]["vectors"][2]["token_scores"] = [0.1, 0.2, 0.3, 0.4]
        with pytest.raises(ValidationError, match="skips 1 prefix positions"):
            ProbeDefinitionV1.model_validate(document)

    def test_the_published_schema_carries_both_fields(self):
        from pathlib import Path

        published = json.loads(
            (Path(__file__).resolve().parents[3] / "docs" / "schemas" / "probe-definition-v1.json").read_text()
        )
        aggregation = published["$defs"]["Aggregation"]
        assert aggregation["properties"]["skip_template_prefix"]["default"] is False
        assert "skip_template_prefix" not in aggregation["required"]
        assert aggregation["additionalProperties"] is False
        assert "template_prefix_tokens" in published["$defs"]["TestVector"]["properties"]


# ── new runs ─────────────────────────────────────────────────────────────────────────────────


class TestNewRunsSkipByDefault:
    def test_the_run_config_defaults_on_and_is_stamped(self):
        """G38-2: on for every rule in a new run; stamped into `run.config` at submit, which
        persists `config.model_dump()` verbatim."""
        from src.schemas.probe_monitor import ProbeRunConfig, ProbeRunCreate

        assert ProbeRunConfig().skip_template_prefix is True
        assert ProbeRunConfig(rules=["mean"]).model_dump()["skip_template_prefix"] is True
        assert ProbeRunConfig(skip_template_prefix=False).model_dump()["skip_template_prefix"] is False
        request = ProbeRunCreate(model_id="m", train_dataset_id="v")
        assert request.config.model_dump()["skip_template_prefix"] is True


async def _submit_world(session):
    from src.models.dataset import Dataset, DatasetStatus
    from src.models.model import Model, ModelStatus
    from src.models.probe_monitor import ProbeMonitorDataset

    model = Model(id=f"m_{uuid.uuid4().hex[:12]}", name="tiny-g38", status=ModelStatus.READY,
                  file_path="/x", architecture="LlamaForCausalLM", params_count=1)
    dataset = Dataset(id=uuid.uuid4(), name="d-g38", source="local", status=DatasetStatus.READY,
                      raw_path="/x", extra_metadata={})
    session.add_all([model, dataset])
    await session.flush()
    train = ProbeMonitorDataset(name="train view", dataset_id=dataset.id, input_column="i",
                                label_column="l", label_mapping={"1": "positive"}, role="train",
                                counts={"positive": 30, "negative": 30})
    session.add(train)
    await session.commit()
    return model.id, train.id


@pytest.mark.parametrize("sent,stored", [(None, True), (False, False), (True, True)])
async def test_the_submit_route_stores_the_flag_on_the_run_row(client, async_session, monkeypatch, sent, stored):
    """DRIVEN, not read (review R5): `POST /runs` is called and the ROW it committed is read back. A
    new run stores `skip_template_prefix: true` unless the caller turned it off. The source-scan this
    replaced passed against `model_dump(exclude={"skip_template_prefix"})`, which stores no key —
    and `context_from_row` reads an absent key as False, so every new run would train over the
    prefix while the config said nothing."""
    from sqlalchemy import select

    from src.api.v1.endpoints import probe_monitors as endpoint
    from src.models.probe_monitor import ProbeMonitorRun

    model_id, train_id = await _submit_world(async_session)
    queued = []
    monkeypatch.setattr(endpoint, "gpu_delay",
                        lambda *a, **k: (lambda **kw: queued.append(kw) or SimpleNamespace(id="task-g38")))
    body = {"model_id": model_id, "train_dataset_id": train_id}
    if sent is not None:
        body["config"] = {"skip_template_prefix": sent}
    response = await client.post("/api/v1/probe-monitors/runs", json=body)
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    assert queued == [{"run_id": run_id}]
    async_session.expire_all()
    row = (await async_session.execute(select(ProbeMonitorRun).where(ProbeMonitorRun.id == run_id))).scalar_one()
    assert row.config["skip_template_prefix"] is stored


# ── the prefix itself ────────────────────────────────────────────────────────────────────────


class TestThePrefix:
    def test_a_stale_date_is_re_read_once(self, monkeypatch):
        """`strftime_now` stamps today's date into the preamble: a prefix cached yesterday matches a
        row rendered today only up to the date. The row re-reads it once and skips the whole prefix."""
        import datetime as real_datetime

        from transformers.utils import chat_template_utils

        days = ["07", "08"]
        tokenizer = _tokenizer(
            "<s> <turn> system {{ strftime_now('%d') }} </turn> {% for m in messages %}<turn> "
            "{{ m['role'] }} {{ m['content'] }} </turn> {% endfor %}"
            "{% if add_generation_prompt %}<turn> assistant {% endif %}",
            WORDS + days,
        )
        clock = {"day": 7}

        class FakeDatetime:
            @staticmethod
            def now():
                return real_datetime.datetime(2026, 11, clock["day"], 12, 0)

        monkeypatch.setattr(chat_template_utils, "datetime", FakeDatetime)
        assert template_prefix_ids(tokenizer, refresh=True)[3] == tokenizer.convert_tokens_to_ids("07")
        clock["day"] = 8
        row = render_messages(tokenizer, _rows(1)[0], skip_template_prefix=True)
        assert tokenizer.convert_ids_to_tokens(row.input_ids)[3] == "08"
        assert row.template_prefix_tokens == 7, "the stale prefix stopped the skip at the date"

    def test_a_row_cut_at_the_front_measures_what_it_still_holds(self, tokenizer):
        """Truncation keeps the TAIL, so a cut row no longer begins with the prefix: P is measured on
        the ids it keeps — the number a consumer computes from the same ids."""
        messages = _rows(1)[0]
        full = render_messages(tokenizer, messages, skip_template_prefix=True)
        assert full.template_prefix_tokens == 7
        cut = render_messages(tokenizer, messages, max_length=len(full.input_ids) - 3, skip_template_prefix=True)
        assert cut.truncated and cut.template_prefix_tokens == 0
        assert cut.scored_mask("all") == [True] * len(cut.input_ids)

    def test_unknown_probe_contents_give_no_prefix(self):
        """A vocabulary that cannot tell `x` from `y` renders both probes identically; the common
        prefix would then run through the content. No prefix is known, and nothing is skipped."""
        tokenizer = _tokenizer(words=[w for w in WORDS if w not in ("x", "y")])
        assert template_prefix_ids(tokenizer, refresh=True) == []
        row = render_messages(tokenizer, _rows(1)[0], skip_template_prefix=True)
        assert row.template_prefix_tokens == 0

    def test_the_skip_applies_to_exactly_the_windows_that_contain_the_prefix(self, tokenizer):
        assert TEMPLATE_PREFIX_SCOPES == {"all", "input", "user", "last_user"}
        messages = [{"role": "user", "content": "w1 w2"}, {"role": "assistant", "content": "w3"}]
        row = render_messages(tokenizer, messages, skip_template_prefix=True)
        plain = render_messages(tokenizer, messages)
        for scope in ("all", "input", "user", "last_user"):
            assert row.scored_mask(scope)[:7] == [False] * 7
            assert row.scored_mask(scope)[7:] == plain.scored_mask(scope)[7:]
        # `last_user` WITHOUT the skip starts at the first user header, inside the prefix.
        assert plain.scored_mask("last_user")[5:7] == [True, True]
        for scope in ("last_assistant", "assistant", "content"):
            assert row.scored_mask(scope) == plain.scored_mask(scope), scope

    def test_the_render_summary_counts_what_was_skipped(self, tokenizer):
        rows = _rows(6)
        _rendered, summary = render_all(tokenizer, rows, scope="all", skip_template_prefix=True)
        assert summary.as_dict()["template_prefix_skipped"] == 6 * 7
        _rendered, plain = render_all(tokenizer, rows, scope="all")
        assert "template_prefix_skipped" not in plain.as_dict()
        assert plain.scored_tokens - summary.scored_tokens == 6 * 7


# ── calibration reads the same masks ─────────────────────────────────────────────────────────


def test_calibration_lengths_and_negatives_leave_the_prefix_out(tiny_lm, tokenizer):
    """The calibrating stage cuts every window's bar and length bands from `_score_calibration_group`.
    On skipping rows `all`, `prompt` (internal `input`) and `last_user` count only positions at or after
    P, so each bar is cut over the same positions miLLM serves."""
    from src.ml.probe_monitor_model import ProbeHead
    from src.services.probe_monitor_run import _score_calibration_group

    torch.manual_seed(1)
    head = ProbeHead(weight=torch.randn(D_MODEL), bias=0.0, mean=torch.zeros(D_MODEL),
                     std=torch.ones(D_MODEL), layer=LAYER)
    trained = SimpleNamespace(rule="max", rule_params={})
    probe = SimpleNamespace(variant="dense")
    messages = _rows(12)
    out = {}
    for skip in (False, True):
        rendered, _summary = render_all(tokenizer, messages, scope="all", skip_template_prefix=skip)
        out[skip] = _score_calibration_group(
            None, (rendered, [0] * len(rendered)), [(probe, head, trained)],
            ["all", "input", "last_user"], tiny_lm, tokenizer, "LlamaForCausalLM",
        )[0]
    for scope in ("all", "input"):
        assert [n - 7 for n in out[False][scope].lengths] == out[True][scope].lengths
        assert out[False][scope].negatives != out[True][scope].negatives
    # `last_user` on these single-turn rows starts at the user header (position 5), inside the prefix:
    # the skip leaves out its first two positions.
    assert [n - 2 for n in out[False]["last_user"].lengths] == out[True]["last_user"].lengths
