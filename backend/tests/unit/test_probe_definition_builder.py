"""The builder: sampling, the gate, the caps and the identity (033 2.1–2.4, 2.7).

⚠ WHAT THESE TESTS ARE FOR. An exported definition is a set of CLAIMS someone else will act on
without being able to check them: which layer, which normalisation, which revision, what evidence.
Every refusal here exists because the corresponding mistake produces a document that parses,
validates, and is wrong — and a consumer has no way to tell.

MUTATION CONTROLS (each verified to fail this file):
  B1  `sample_vectors` stops representing every set        → the coverage test
  B2  the sample stops balancing the classes               → the balance test
  B3  the seed is ignored                                  → the determinism test
  B4  `truncate_tokens` keeps the HEAD instead of the tail  → the truncation test
  B5  the gate's order changed                             → the order tests
  B6  the acknowledgement's reason floor removed           → the token-reason test
  B7  `check_size` stops raising                           → the size test
  B8  `resolve_model_revision` returns "main"              → the revision tests
  B9  `_d_model_of` falls back to the head's width          → the d_model test
  B10 the SAE-without-an-HF-home refusal removed           → the SAE-location test
"""
import hashlib
import pathlib
from types import SimpleNamespace

import pytest

from src.schemas.probe_definition import MAX_VECTOR_TOKENS, MIN_TEST_VECTORS
from src.services import probe_definition_builder as builder
from src.services.probe_definition_builder import (
    DEFAULT_VECTOR_COUNT,
    ProbeExportRefused,
    check_export_gate,
    check_size,
    resolve_model_revision,
    sample_vectors,
    truncate_tokens,
)


def _rows(spec):
    """`spec` is {set_name: (n_positive, n_negative)}."""
    rows = []
    for name, (positives, negatives) in spec.items():
        index = 0
        for _ in range(positives):
            rows.append({"dataset": name, "index": index, "label": 1})
            index += 1
        for _ in range(negatives):
            rows.append({"dataset": name, "index": index, "label": 0})
            index += 1
    return rows


class TestTheSampleSpansEverySet:
    """⚠ THE SMALLEST SET IS THE ONE THAT MATTERS. A proportional sample would omit it, and it is
    the set whose distribution is least like the training data — exactly where an implementation
    difference hides."""

    def test_every_set_contributes(self):
        rows = _rows({"big": (200, 200), "tiny": (3, 3), "middle": (40, 40)})
        chosen = sample_vectors(rows, count=DEFAULT_VECTOR_COUNT, seed=1)
        assert {row["dataset"] for row in chosen} == {"big", "tiny", "middle"}

    def test_a_set_with_one_class_still_contributes(self):
        rows = _rows({"both": (50, 50), "positives_only": (4, 0)})
        chosen = sample_vectors(rows, count=DEFAULT_VECTOR_COUNT, seed=1)
        assert "positives_only" in {row["dataset"] for row in chosen}

    def test_it_returns_exactly_the_requested_count(self):
        rows = _rows({"a": (100, 100), "b": (100, 100)})
        assert len(sample_vectors(rows, count=12, seed=1)) == 12

    def test_a_count_outside_the_contract_is_refused(self):
        rows = _rows({"a": (100, 100)})
        for count in (MIN_TEST_VECTORS - 1, 33):
            with pytest.raises(ProbeExportRefused):
                sample_vectors(rows, count=count, seed=1)

    def test_no_rows_at_all_is_a_409(self):
        with pytest.raises(ProbeExportRefused) as caught:
            sample_vectors([], count=DEFAULT_VECTOR_COUNT, seed=1)
        assert caught.value.status == 409
        assert "has not been evaluated" in str(caught.value)

    def test_too_few_rows_is_a_409(self):
        with pytest.raises(ProbeExportRefused) as caught:
            sample_vectors(_rows({"a": (2, 2)}), count=DEFAULT_VECTOR_COUNT, seed=1)
        assert caught.value.status == 409


class TestTheSampleIsBalanced:
    """A parity check on positives alone passes for a probe that fires on everything."""

    def test_the_classes_are_within_one_of_each_other(self):
        rows = _rows({"a": (100, 100), "b": (100, 100)})
        chosen = sample_vectors(rows, count=16, seed=7)
        positives = sum(1 for row in chosen if row["label"] == 1)
        assert abs(positives - (len(chosen) - positives)) <= 1, positives

    def test_it_stays_balanced_with_many_small_sets(self):
        rows = _rows({name: (5, 5) for name in "abcde"})
        chosen = sample_vectors(rows, count=16, seed=3)
        positives = sum(1 for row in chosen if row["label"] == 1)
        assert abs(positives - (len(chosen) - positives)) <= 2, positives

    def test_an_exhausted_class_does_not_return_short(self):
        """If one class runs out, filling from the other beats handing back too few vectors — the
        contract's floor is 8 and a short list would fail validation instead of exporting."""
        rows = _rows({"a": (2, 40)})
        chosen = sample_vectors(rows, count=12, seed=1)
        assert len(chosen) == 12


class TestTheSampleIsDeterministic:
    def test_the_same_seed_gives_the_same_rows(self):
        rows = _rows({"a": (50, 50), "b": (50, 50)})
        first = sample_vectors(rows, count=16, seed=99)
        second = sample_vectors(rows, count=16, seed=99)
        assert first == second

    def test_a_different_seed_gives_different_rows(self):
        """The control: if the seed did nothing, the determinism test above would pass over a
        sampler that ignores it."""
        rows = _rows({"a": (200, 200)})
        first = sample_vectors(rows, count=16, seed=1)
        second = sample_vectors(rows, count=16, seed=2)
        assert first != second

    def test_the_order_is_stable_regardless_of_input_order(self):
        """Two builds of the same probe must produce the same FILE, so a consumer diffing them is
        comparing implementations rather than dict ordering."""
        rows = _rows({"a": (40, 40), "b": (40, 40)})
        forward = sample_vectors(rows, count=16, seed=5)
        backward = sample_vectors(list(reversed(rows)), count=16, seed=5)
        assert [(r["dataset"], r["index"]) for r in forward] == [
            (r["dataset"], r["index"]) for r in backward
        ]


class TestTruncationKeepsTheTail:
    """⚠ THE TAIL, NOT THE HEAD. `last` reads the final scored token and `rolling_mean_max` weights
    the end, so truncating from the right would change what those rules see — and the recorded score
    would not be the score of the row in the file."""

    def test_a_short_row_is_untouched(self):
        assert truncate_tokens([1, 2, 3]) == [1, 2, 3]

    def test_a_long_row_keeps_its_END(self):
        tokens = list(range(MAX_VECTOR_TOKENS + 50))
        kept = truncate_tokens(tokens)
        assert len(kept) == MAX_VECTOR_TOKENS
        assert kept[-1] == tokens[-1], "the final token is gone, so `last` reads something else"
        assert kept[0] == tokens[50]

    def test_the_limit_is_respected_exactly(self):
        assert len(truncate_tokens(list(range(5000)), limit=100)) == 100


class TestTheGateRefusesInOrder:
    """⚠ THE ORDER IS PART OF THE CONTRACT. Telling someone their evidence is too thin, when the
    real problem is that the run is still going, sends them to do the wrong work."""

    @staticmethod
    def _probe(**overrides):
        probe = SimpleNamespace(
            id="pm_x", variant="dense", weights_path="/data/x.safetensors", rung=2, sae_id=None
        )
        for key, value in overrides.items():
            setattr(probe, key, value)
        return probe

    @staticmethod
    def _run(status="completed"):
        return SimpleNamespace(id="pmr_x", status=status)

    def test_a_missing_probe_is_a_404(self):
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(None, self._run())
        assert caught.value.status == 404

    def test_an_sae_probe_without_an_hf_location_is_a_422(self):
        sae = SimpleNamespace(id="sae_x", hf_repo_id=None, hf_filepath=None)
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(variant="sae", sae_id="sae_x"), self._run(), sae_row=sae)
        assert caught.value.status == 422
        assert "publish the SAE first" in str(caught.value)

    def test_that_refusal_cannot_be_acknowledged_away(self):
        """It is not a judgement call: without the dictionary a consumer cannot encode at all."""
        sae = SimpleNamespace(id="sae_x", hf_repo_id=None, hf_filepath=None)
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(
                self._probe(variant="sae", sae_id="sae_x", rung=0),
                self._run(),
                sae_row=sae,
                acknowledge_below_rung2={"reason": "I accept the thin evidence, publish anyway"},
            )
        assert "publish the SAE first" in str(caught.value)

    def test_an_sae_probe_with_a_location_passes_that_check(self):
        sae = SimpleNamespace(id="sae_x", hf_repo_id="owner/repo", hf_filepath="l11/sae.safetensors")
        assert check_export_gate(
            self._probe(variant="sae", sae_id="sae_x"), self._run(), sae_row=sae
        ) is None

    def test_no_weights_is_a_409(self):
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(weights_path=None), self._run())
        assert caught.value.status == 409
        assert "did not finish training" in str(caught.value)

    def test_the_weights_check_comes_BEFORE_the_rung(self):
        """A probe with no weights and a low rung must report the weights: acknowledging the rung
        would not make it exportable."""
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(weights_path=None, rung=0), self._run())
        assert "did not finish training" in str(caught.value)

    @pytest.mark.parametrize("rung", [0, 1])
    def test_a_low_rung_without_an_acknowledgement_is_a_422(self, rung):
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(rung=rung), self._run())
        assert caught.value.status == 422
        assert "acknowledge_below_rung2" in str(caught.value)

    def test_the_refusal_quotes_the_rung_LANGUAGE(self):
        """The wording comes from 032's ladder, so the refusal says what the rung means rather than
        making the reader look up a number."""
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(rung=1), self._run())
        from src.schemas.evidence_ladder import probe_rung_language

        assert probe_rung_language(1) in str(caught.value)

    @pytest.mark.parametrize("rung", [0, 1])
    def test_an_acknowledgement_is_returned_and_stamped(self, rung):
        acknowledgement = check_export_gate(
            self._probe(rung=rung),
            self._run(),
            acknowledge_below_rung2={"reason": "exploratory monitor, not used for gating"},
            actor="sean",
        )
        assert acknowledgement is not None
        assert acknowledgement.by == "sean"
        assert acknowledgement.at.tzinfo is not None, "the timestamp must be timezone-aware"
        assert "exploratory" in acknowledgement.reason

    def test_a_token_reason_is_refused(self):
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(
                self._probe(rung=0), self._run(), acknowledge_below_rung2={"reason": "ok"}
            )
        assert "at least 10 characters" in str(caught.value)

    @pytest.mark.parametrize("status", ["pending", "running", "cancelling"])
    def test_a_run_in_flight_is_a_409(self, status):
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(), self._run(status))
        assert caught.value.status == 409
        assert "about to be superseded" in str(caught.value)

    def test_the_rung_gate_comes_BEFORE_the_running_check(self):
        """A low-rung probe on a running run reports the RUNG, because that is the thing the caller
        can act on — waiting will not raise it."""
        with pytest.raises(ProbeExportRefused) as caught:
            check_export_gate(self._probe(rung=0), self._run("running"))
        assert "acknowledge_below_rung2" in str(caught.value)

    def test_a_ready_rung_2_probe_on_a_finished_run_passes(self):
        assert check_export_gate(self._probe(), self._run("completed")) is None


class TestTheSizeCapIsMeasuredOnWhatIsDownloaded:
    @staticmethod
    def _definition(vectors=MIN_TEST_VECTORS, tokens=3):
        from tests.unit.test_probe_definition import definition
        from src.schemas.probe_definition import ProbeDefinitionV1

        body = definition()
        body["test_vectors"]["vectors"] = [
            {
                "messages": [{"role": "user", "content": "x" * 10}],
                "token_ids": list(range(tokens)),
                "token_scores": [0.1] * tokens,
                "score": 0.2,
                "verdict": True,
            }
            for _ in range(vectors)
        ]
        return ProbeDefinitionV1.model_validate(body)

    def test_a_small_definition_passes_and_reports_its_size(self):
        size = check_size(self._definition())
        assert size > 0

    def test_the_size_is_of_the_SERIALISED_bytes(self):
        """A cap on field counts would not bound what a consumer downloads.

        ⚠ THIS TEST PINNED THE DEFECT RATHER THAN PREVENTING IT. It asserted the size equalled
        `model_dump_json()` — the COMPACT form — which is exactly what `check_size` was wrongly
        measuring, while the file on disk is written `indent=2` and is about 1.6x larger. The test
        and the code agreed with each other and both disagreed with the file, so the suite was green
        over a cap that did not bound the download and a digest that did not match the document.
        It now asserts the form that is written.
        """
        definition = self._definition()
        assert check_size(definition) == len(builder.serialise(definition).encode("utf-8"))
        assert check_size(definition) == len(definition.model_dump_json(indent=2).encode("utf-8"))

    def test_an_oversized_definition_is_refused_with_advice(self, monkeypatch):
        monkeypatch.setattr(builder, "MAX_DEFINITION_BYTES", 200)
        with pytest.raises(ProbeExportRefused) as caught:
            check_size(self._definition())
        message = str(caught.value)
        assert "over the" in message and "cap" in message
        assert "test-vector count" in message, "the refusal must say what to change"

    def test_the_sha_changes_with_the_content(self):
        from src.services.probe_definition_builder import sha256_of

        one = self._definition(tokens=3)
        two = self._definition(tokens=4)
        assert sha256_of(one) != sha256_of(two)
        assert sha256_of(one) == sha256_of(self._definition(tokens=3))
        assert len(sha256_of(one)) == 64


class TestTheRevisionIsPinnedToACommit:
    """⚠ "main" MOVES. Two LiquidAI models sharing a family name agreed on 0.00% of token ids here,
    and 20 of 22 extractions silently read the wrong tokenizer. A moving reference in an exported
    document is the same class of failure, shipped to someone else.

    ⚠ THE TWO TESTS THAT USED TO OPEN THIS CLASS WERE TESTING FICTION, and they are gone.
    `test_a_recorded_sha_is_used` handed the resolver `SimpleNamespace(hf_revision="a"*40)` and
    `test_a_moving_reference_is_not_accepted_as_the_revision` handed it `hf_revision="main"` —
    but `Model` has no `hf_revision` column, and nothing in this codebase persists a model revision
    anywhere. Both tests exercised a tier that could not fire in production, and the "main is
    refused" one passed for the wrong reason: the real refusal was "no snapshot", not "moving
    reference". A stub that answers a question the ORM cannot be asked will agree with anything.

    What is pinned instead: the revision is the cache snapshot directory name, resolved through
    `resolve_model_snapshot` — the loader's own function — so the pin and the weights cannot come
    from different snapshots.
    """

    def test_the_pin_is_the_snapshot_the_LOADER_resolves(self, tmp_path, monkeypatch):
        """Not merely "a" snapshot name — the same one the weights come from.

        The old implementation globbed the cache itself and took `sorted(...)[-1]`; the loader takes
        `glob(...)[0]`. One snapshot on disk hides the difference, and every model here has one, so
        this is the test that would notice the day one does not.
        """
        from src.services import activation_service

        root = tmp_path / "raw" / "m_x"
        for sha in ("a" * 40, "f" * 40):
            (root / "models--org--name" / "snapshots" / sha).mkdir(parents=True)
        monkeypatch.setattr(
            builder, "settings", SimpleNamespace(resolve_data_path=lambda _p: root)
        )
        row = SimpleNamespace(id="m_x", file_path="/data/models/raw/m_x")
        loader_choice = pathlib.Path(activation_service.resolve_model_snapshot(str(root))).name
        assert resolve_model_revision(row) == loader_choice

    def test_a_flat_download_is_refused_because_it_records_no_commit(self, tmp_path, monkeypatch):
        """A flat directory's name is the model id, not a sha — pinning it would be a lie."""
        root = tmp_path / "raw" / "m_x"
        (root).mkdir(parents=True)
        (root / "config.json").write_text("{}")
        monkeypatch.setattr(
            builder, "settings", SimpleNamespace(resolve_data_path=lambda _p: root)
        )
        row = SimpleNamespace(id="m_x", file_path="/data/models/raw/m_x")
        with pytest.raises(ProbeExportRefused) as caught:
            resolve_model_revision(row)
        assert "flat download records no commit" in str(caught.value)

    def test_a_model_with_no_path_at_all_is_refused(self):
        row = SimpleNamespace(id="m_x", file_path=None)
        with pytest.raises(ProbeExportRefused) as caught:
            resolve_model_revision(row)
        assert "different distribution" in str(caught.value)

    def test_it_falls_back_to_the_cache_snapshot_which_IS_the_sha(self, tmp_path, monkeypatch):
        snapshot = tmp_path / "raw" / "m_x" / "models--org--name" / "snapshots" / ("b" * 40)
        snapshot.mkdir(parents=True)
        # `Settings` is a pydantic model and refuses an unknown attribute, so the MODULE's
        # reference is replaced rather than the object's field.
        monkeypatch.setattr(
            builder, "settings",
            SimpleNamespace(resolve_data_path=lambda _p: tmp_path / "raw" / "m_x"),
        )
        row = SimpleNamespace(id="m_x", file_path="/data/models/raw/m_x")
        assert resolve_model_revision(row) == "b" * 40

    def test_no_revision_and_no_snapshot_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            builder, "settings",
            SimpleNamespace(resolve_data_path=lambda _p: tmp_path / "empty"),
        )
        row = SimpleNamespace(id="m_x", file_path="/data/models/raw/m_x")
        with pytest.raises(ProbeExportRefused) as caught:
            resolve_model_revision(row)
        assert "different distribution" in str(caught.value)


class TestTheIdentityComesFromTheModelRow:
    """⚠ `d_model` MUST NOT BE INFERRED FROM THE HEAD. On a k-sparse probe the head's width is k —
    128, not 2,048 — so a fallback would state a d_model the model does not have, and the contract's
    own width check would then pass over a wrong document."""

    def test_d_model_is_read_from_the_config(self):
        row = SimpleNamespace(id="m_x", architecture_config={"hidden_size": 2048})
        assert builder._d_model_of(row, head=None) == 2048

    def test_the_alternate_config_keys_are_accepted(self):
        for key in ("d_model", "n_embd"):
            row = SimpleNamespace(id="m_x", architecture_config={key: 1536})
            assert builder._d_model_of(row, head=None) == 1536

    def test_a_missing_hidden_size_is_REFUSED_not_guessed(self):
        row = SimpleNamespace(id="m_x", architecture_config={})
        with pytest.raises(ProbeExportRefused) as caught:
            builder._d_model_of(row, head=SimpleNamespace(weight=[0.0] * 128))
        assert "silently wrong" in str(caught.value)

    def test_the_layer_count_is_read_from_the_config(self):
        row = SimpleNamespace(id="m_x", architecture_config={"num_hidden_layers": 16})
        assert builder._n_layers_of(row) == 16

    def test_a_missing_layer_count_is_refused(self):
        with pytest.raises(ProbeExportRefused):
            builder._n_layers_of(SimpleNamespace(id="m_x", architecture_config={}))

    def test_the_chat_template_hash_is_of_the_template(self):
        tokenizer = SimpleNamespace(chat_template="{{ messages }}")
        expected = hashlib.sha256(b"{{ messages }}").hexdigest()
        assert builder.chat_template_sha256(tokenizer) == expected

    def test_no_template_is_None_not_a_hash_of_the_empty_string(self):
        """A hash of "" would claim a template exists and pin it, which is worse than saying
        nothing: a consumer would compare against it and always disagree."""
        assert builder.chat_template_sha256(SimpleNamespace(chat_template=None)) is None
        assert builder.chat_template_sha256(SimpleNamespace()) is None


class TestTheScopeMapsWidenNeverNarrow:
    """032 has `all | assistant | user | last_assistant`; the contract has `all | prompt | response`.
    The mapping is lossy in one direction only, deliberately: a consumer told `response` scores more
    tokens than the probe was trained on rather than fewer, which is the safe error."""

    @pytest.mark.parametrize(
        "internal,contract",
        [
            ("all", "all"),
            ("user", "prompt"),
            ("assistant", "response"),
            ("last_assistant", "response"),
        ],
    )
    def test_the_mapping(self, internal, contract):
        assert builder._contract_scope(internal) == contract

    def test_an_unknown_scope_falls_back_to_the_WIDEST(self):
        """Falling back to a narrow scope would claim the probe reads fewer tokens than it does."""
        assert builder._contract_scope("something_new") == "all"


class TestTheConceptComesFromTheLabelMapping:
    def test_it_names_the_positive_labels(self):
        view = SimpleNamespace(label_mapping={"high-stakes": "positive", "low-stakes": "negative"})
        assert builder._concept_of(view) == "positive = high-stakes"

    def test_several_positives_are_all_named(self):
        view = SimpleNamespace(label_mapping={"a": "positive", "b": "positive", "c": "negative"})
        assert builder._concept_of(view) == "positive = a, b"

    def test_no_mapping_is_None_rather_than_an_invented_summary(self):
        assert builder._concept_of(None) is None
        assert builder._concept_of(SimpleNamespace(label_mapping={})) is None

    def test_a_mapping_with_no_positives_is_None(self):
        view = SimpleNamespace(label_mapping={"x": "negative", "y": "excluded"})
        assert builder._concept_of(view) is None


class TestTheDatasetReferenceIsReadFromTheROW:
    """⚠ THIS FUNCTION HAD NO TEST AND THAT IS WHY IT SHIPPED BROKEN.

    The first version read `view.dataset` — an ORM relationship that does not exist.
    `ProbeMonitorDataset` has a `dataset_id` FOREIGN KEY and nothing beside it, so the attribute
    was always None and every build refused with

        probe dataset pmd_… has no HuggingFace identifier, only a local path

    over a `Dataset` row whose `hf_repo_id` is `Arrrlex/models-under-pressure`, one query away.
    **Three of the four Stage 2 acceptance builds died on it**, and no unit test could see it
    because there was none: the surrounding tests all used hand-built objects that HAD a `.dataset`
    attribute, which is the fixture-agrees-by-construction trap in its purest form.

    So these tests pass a session-like object and a view carrying only `dataset_id`, which is what
    the ORM actually provides.

    MUTATION CONTROLS:
      B11  back to `getattr(view, "dataset", None)`   → the resolution test
      B12  the local-path refusal removed             → the refusal test
      B13  `config`/`split` dropped from the ref      → the fields test
    """

    class _Db:
        """Just enough session for `db.query(Dataset).filter(...).first()`."""

        def __init__(self, dataset):
            self._dataset = dataset

        def query(self, _model):
            return self

        def filter(self, *_args):
            return self

        def first(self):
            return self._dataset

    @staticmethod
    def _dataset(**overrides):
        row = SimpleNamespace(
            id="ds-1",
            hf_repo_id="Arrrlex/models-under-pressure",
            extra_metadata={"config": "mt_balanced", "revision": "abc123"},
        )
        for key, value in overrides.items():
            setattr(row, key, value)
        return row

    @staticmethod
    def _view(**overrides):
        view = SimpleNamespace(
            id="pmd_1", dataset_id="ds-1", config="mt_balanced", split="test"
        )
        for key, value in overrides.items():
            setattr(view, key, value)
        return view

    def test_the_view_has_no_dataset_relationship(self):
        """The premise, asserted so the tests below are not guarding a fiction. If a relationship
        is ever added, this fails and the function can be simplified."""
        from src.models.probe_monitor import ProbeMonitorDataset

        assert not hasattr(ProbeMonitorDataset, "dataset"), (
            "ProbeMonitorDataset now has a `dataset` relationship; `dataset_ref_from_view` can "
            "stop taking a session"
        )
        assert hasattr(ProbeMonitorDataset, "dataset_id")

    def test_it_resolves_the_repo_from_the_dataset_row(self):
        from src.services.probe_definition_builder import dataset_ref_from_view

        ref = dataset_ref_from_view(self._Db(self._dataset()), self._view())
        assert ref.hf_id == "Arrrlex/models-under-pressure"

    def test_it_carries_the_config_and_split_from_the_VIEW(self):
        """The config is what distinguishes five views of one repo, and it lives on the view, not
        on the dataset."""
        from src.services.probe_definition_builder import dataset_ref_from_view

        ref = dataset_ref_from_view(
            self._Db(self._dataset()), self._view(config="toolace_balanced", split="test")
        )
        assert (ref.config, ref.split) == ("toolace_balanced", "test")

    def test_it_takes_the_revision_from_the_metadata_when_there_is_one(self):
        from src.services.probe_definition_builder import dataset_ref_from_view

        ref = dataset_ref_from_view(self._Db(self._dataset()), self._view())
        assert ref.revision == "abc123"

    def test_no_revision_is_None_rather_than_invented(self):
        from src.services.probe_definition_builder import dataset_ref_from_view

        ref = dataset_ref_from_view(
            self._Db(self._dataset(extra_metadata={})), self._view()
        )
        assert ref.revision is None

    def test_a_dataset_with_no_repo_is_REFUSED(self):
        """The real refusal, on the real condition: a locally-ingested dataset genuinely cannot be
        named portably, and that must not become a definition pointing at /data."""
        from src.services.probe_definition_builder import (
            ProbeExportRefused,
            dataset_ref_from_view,
        )

        with pytest.raises(ProbeExportRefused) as caught:
            dataset_ref_from_view(self._Db(self._dataset(hf_repo_id=None)), self._view())
        assert "no HuggingFace identifier" in str(caught.value)

    def test_a_missing_dataset_row_is_refused_too(self):
        from src.services.probe_definition_builder import (
            ProbeExportRefused,
            dataset_ref_from_view,
        )

        with pytest.raises(ProbeExportRefused):
            dataset_ref_from_view(self._Db(None), self._view())

    def test_it_queries_by_the_views_dataset_id(self):
        """A resolution that ignored `dataset_id` would return whatever the first row happened to
        be — right on a one-row fixture, wrong everywhere else."""
        import ast
        import inspect

        from src.services.probe_definition_builder import dataset_ref_from_view

        source = inspect.getsource(dataset_ref_from_view)
        tree = ast.parse(source.lstrip())
        reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in {"dataset_id", "dataset"}
        }
        assert "dataset_id" in reads
        assert "dataset" not in reads, (
            "the function still reads `view.dataset`, a relationship that does not exist"
        )


class TestTheNormalisationIsReadFromWhereItIsRecorded:
    """⚠ THE DEFAULT THAT HID THIS BUG WAS CORRECT FOR THE SAE THAT FOUND IT.

    `sae_reference` read `getattr(sae_row, "normalize_activations", None) or
    "constant_norm_rescale"`. `ExternalSAE` has no such column — the value is at
    `sae_metadata["training_hyperparameters"]["normalize_activations"]` — so the default fired on
    every export. `sae_9a4db34d5dab`, the SAE this was found on, records exactly
    `constant_norm_rescale`, so the exported file was right and nothing anywhere disagreed with it.

    Export an SAE trained with `none` or `anthropic_rescale` and the same code states
    `constant_norm_rescale` with full confidence. A consumer then encodes in the wrong basis, which
    yields plausible features with different meanings and is invisible in every metric — MIS-E2E-083,
    reintroduced inside the field whose docstring warns about it.
    """

    def _sae(self, **metadata):
        return SimpleNamespace(
            id="sae_x",
            hf_repo_id="someone/sae",
            hf_filepath="layer_11",
            hf_revision=None,
            d_model=2048,
            n_features=16384,
            architecture="jumprelu",
            local_path=None,
            sae_metadata=metadata,
        )

    def test_the_mode_comes_from_the_training_hyperparameters(self, monkeypatch):
        monkeypatch.setattr(builder, "_sae_weights_sha256", lambda _row: "c" * 64)
        reference = builder.sae_reference(
            self._sae(training_hyperparameters={"normalize_activations": "anthropic_rescale"}),
            [1, 2, 3],
        )
        assert reference.normalization["mode"] == "anthropic_rescale"
        assert reference.normalization["source"] == "sae_metadata.training_hyperparameters"

    def test_a_mode_of_none_is_a_RECORDED_value_not_an_absent_one(self, monkeypatch):
        """`"none"` is a real normalisation choice, and the string is truthy — so it must survive.

        This is the case a `or DEFAULT` gets wrong in the most damaging direction: an SAE that
        applied no normalisation, exported as one that rescales.
        """
        monkeypatch.setattr(builder, "_sae_weights_sha256", lambda _row: "c" * 64)
        reference = builder.sae_reference(
            self._sae(training_hyperparameters={"normalize_activations": "none"}), [1]
        )
        assert reference.normalization["mode"] == "none"

    def test_an_unrecorded_mode_is_REFUSED_rather_than_guessed(self, monkeypatch):
        monkeypatch.setattr(builder, "_sae_weights_sha256", lambda _row: "c" * 64)
        with pytest.raises(ProbeExportRefused) as caught:
            builder.sae_reference(self._sae(training_hyperparameters={}), [1])
        message = str(caught.value)
        assert "records no activation-normalisation mode" in message
        assert "wrong basis" in message

    def test_the_top_level_metadata_key_is_accepted_as_a_second_source(self, monkeypatch):
        monkeypatch.setattr(builder, "_sae_weights_sha256", lambda _row: "c" * 64)
        reference = builder.sae_reference(
            self._sae(normalize_activations="constant_norm_rescale"), [1]
        )
        assert reference.normalization["mode"] == "constant_norm_rescale"
        assert reference.normalization["source"] == "sae_metadata"

    def test_the_constants_travel_from_the_hyperparameters_too(self, monkeypatch):
        """The mode alone does not reproduce the basis — its constants have to come with it."""
        monkeypatch.setattr(builder, "_sae_weights_sha256", lambda _row: "c" * 64)
        reference = builder.sae_reference(
            self._sae(
                training_hyperparameters={
                    "normalize_activations": "constant_norm_rescale",
                    "normalization_constant": 12.5,
                }
            ),
            [1],
        )
        assert reference.normalization["normalization_constant"] == 12.5


class TestTheModelIdentifierIsTheRepoId:
    """⚠ `Model.hf_repo_id` DOES NOT EXIST; the column is `repo_id`.

    A bare access on the wrong name is an `AttributeError`, so every definition build crashed here.
    The fallback it never reached was the worse half: `model_row.hf_repo_id or model_row.name` would
    have put a DISPLAY NAME in the one field that tells a consumer what to fetch.
    """

    def test_the_repo_id_is_used(self):
        assert builder._model_hf_id(SimpleNamespace(id="m_x", repo_id="org/name", name="Name")) == "org/name"

    def test_a_model_with_no_repo_is_refused_not_named(self):
        with pytest.raises(ProbeExportRefused) as caught:
            builder._model_hf_id(SimpleNamespace(id="m_x", repo_id=None, name="My Local Model"))
        message = str(caught.value)
        assert "no HuggingFace repo id" in message
        assert "My Local Model" in message  # the refusal quotes what it found

    def test_it_does_not_read_hf_repo_id_even_when_one_is_present(self):
        """A stub CAN carry `hf_repo_id`; the production row cannot, so reading it is the bug.

        This is the shape of test that would have caught the original: it asserts the code ignores
        the attribute the old version depended on.
        """
        row = SimpleNamespace(id="m_x", repo_id=None, name="Name", hf_repo_id="org/ghost")
        with pytest.raises(ProbeExportRefused):
            builder._model_hf_id(row)


def _contract_document(**overrides) -> dict:
    """A valid v1 document, borrowed from the contract tests so there is ONE factory.

    Importing it rather than re-declaring one is deliberate: two hand-written documents drift, and
    a builder test passing against a document the contract would reject proves nothing.
    """
    from tests.unit.test_probe_definition import definition as _contract_factory

    return _contract_factory(**overrides)


class TestTheRecordedDigestDESCRIBESTheFile:
    """⚠ THE INTEGRITY PIN DID NOT MATCH THE FILE IT PINNED, and it was found by checking.

    Three places serialised the document independently. The writer used
    `model_dump_json(indent=2)`; `sha256_of` and `check_size` used `model_dump_json()`. For the
    first definition this estate built:

        recorded sha256  b0f9f404…      file's sha256  60a93126…
        recorded bytes     308,552      file's bytes     491,721

    A consumer verifying the document against the sha in its own manifest — what a checksum is FOR
    — would reject a good file, and the publisher ships that sha in the README and the manifest
    entry. A checksum that is always wrong is worse than absent: it teaches its first reader to
    ignore it.

    `check_size`'s docstring also said "measured on what a consumer downloads" while measuring a
    form 1.6x smaller, so a document 1.3 MB compact and 2.1 MB indented passed the 2 MB cap and
    landed over it.

    The fix is structural: `write_definition` returns the measurements of the bytes it wrote, and
    callers are never handed the model to serialise a second time.
    """

    def test_the_three_serialisations_are_ONE(self):
        """Asserted on the functions rather than a document, so it holds for every document."""
        from src.services.probe_definition_builder import serialise

        import inspect

        size_source = inspect.getsource(builder.check_size)
        sha_source = inspect.getsource(builder.sha256_of)
        write_source = inspect.getsource(builder.write_definition)
        for name, source in (("check_size", size_source), ("sha256_of", sha_source),
                             ("write_definition", write_source)):
            assert "serialise(" in source, f"{name} must go through serialise()"
            assert "model_dump_json" not in source, (
                f"{name} serialises the model itself, which is how the digest and the file came to "
                f"describe different byte strings"
            )
        assert "indent=2" in inspect.getsource(serialise)

    def test_what_is_written_is_what_is_measured(self, tmp_path, monkeypatch):
        """The end-to-end claim, on a real document written to a real path."""
        import hashlib

        from src.schemas.probe_definition import ProbeDefinitionV1

        document = _contract_document()
        definition = ProbeDefinitionV1.model_validate(document)
        path = tmp_path / "pm_x.probe.json"
        written_bytes, written_sha = builder.write_definition(definition, path)

        raw = path.read_bytes()
        assert len(raw) == written_bytes
        assert hashlib.sha256(raw).hexdigest() == written_sha
        # And the standalone helpers agree with the file, which is the regression that mattered.
        assert builder.sha256_of(definition) == written_sha
        assert builder.check_size(definition) == written_bytes

    def test_the_partial_file_does_not_survive(self, tmp_path):
        from src.schemas.probe_definition import ProbeDefinitionV1

        definition = ProbeDefinitionV1.model_validate(_contract_document())
        path = tmp_path / "pm_x.probe.json"
        builder.write_definition(definition, path)
        assert not list(tmp_path.glob("*.partial")), "the staging file must be moved, not copied"

    def test_the_size_cap_is_checked_against_the_written_form(self, monkeypatch):
        """A cap measured on a smaller serialisation is not a cap.

        The indented form is ~1.6x the compact one here, so a cap set just above a compact
        measurement lets a file through that is over it on disk. The cap is lowered below the
        document's INDENTED size and above its compact size; only a check on the written form
        refuses.
        """
        from src.schemas.probe_definition import ProbeDefinitionV1

        definition = ProbeDefinitionV1.model_validate(_contract_document())
        indented = len(definition.model_dump_json(indent=2).encode())
        compact = len(definition.model_dump_json().encode())
        assert compact < indented, "the fixture must differ between the two forms to test this"
        monkeypatch.setattr(builder, "MAX_DEFINITION_BYTES", (compact + indented) // 2)
        with pytest.raises(ProbeExportRefused) as caught:
            builder.check_size(definition)
        assert "over the" in str(caught.value)


class TestTheJudgeEvidenceIsNotEmpty:
    """⚠ THE DOCUMENT CLAIMED RUNG 3 AND CARRIED NO COMPARISON.

    `JudgeEvidence.per_set_auroc` was read from `metrics["per_set_auroc"]`, a key the judge never
    writes. The judge writes `metrics["per_set"][<dataset id>]` — a dict per set with `auroc`, `ci`,
    `roc`, `operating_points` and counts — so the lookup found nothing and the field was `{}` on
    every export.

    An empty dict is a legal value for the field, so the document validated perfectly. The rung's
    entire justification is "compared with a judge", and the block asserting that comparison was
    empty while the real numbers sat in the judge run: 0.8827, 0.8189, 0.9895, 0.9553, 0.7255.

    This is the sixth wrong field name in this feature and the worst-placed of them — the other five
    either crashed or changed a value; this one hollowed out the evidence for a badge.
    """

    def test_the_auroc_comes_from_the_per_set_block(self):
        metrics = {
            "per_set": {
                "pmd_a": {"auroc": 0.8826648739658878, "ci": {"low": 0.87}, "roc": [], "scored": True},
                "pmd_b": {"auroc": 0.7255269546882076, "scored": True},
            }
        }
        assert builder._judge_per_set_auroc(metrics) == {
            "pmd_a": 0.8826648739658878,
            "pmd_b": 0.7255269546882076,
        }

    def test_the_OLD_key_is_not_consulted(self):
        """A document must not be filled from a key nothing writes, even if one appeared."""
        assert builder._judge_per_set_auroc({"per_set_auroc": {"pmd_a": 0.9}}) == {}

    def test_an_unscored_set_is_MISSING_not_zero(self):
        """0.0 reads as "the judge performed at chance", which is a different claim from "absent"."""
        metrics = {"per_set": {"pmd_a": {"auroc": 0.88}, "pmd_b": {"scored": False}}}
        assert builder._judge_per_set_auroc(metrics) == {"pmd_a": 0.88}

    def test_a_missing_or_malformed_block_is_empty_not_an_error(self):
        for metrics in ({}, {"per_set": None}, {"per_set": []}, {"per_set": {"a": "nope"}}):
            assert builder._judge_per_set_auroc(metrics) == {}


class TestTheMessagesRoundTripIsMEASURED:
    """⚠ THE PARITY CHECK TOLD A CORRECT CONSUMER IT WAS WRONG, ON EVERY VECTOR.

    Measured in 033 acceptance 8.1 against a real definition, 16 vectors, on the GPU:

        scored from the recorded `token_ids`   max |Δ| = 0.000e+00   exact, all sixteen
        re-rendered from `messages`           max |Δ| = 1.153e+00   23x the 0.05 tolerance

    `messages` is a reconstruction. The corpus is plain prose, not conversations, so `_messages_of`
    wraps each row in a single user turn to give a consumer something sendable — and re-rendering
    that through the chat template adds six tokens of scaffolding the scored row never had, first
    difference at index 2. Two of sixteen rows were truncated to the cap keeping the TAIL, so their
    re-render differs from index 0.

    A consumer following the contract would have scored `messages`, compared against `score` within
    `tolerance`, failed on every row and concluded its implementation was broken. **A parity check
    that reports "incorrect" against a correct implementation is worse than no check**: it is
    believed the first time.

    The document now states `authoritative_input` and carries the measured round-trip, so a consumer
    reads the answer instead of deriving it from a failure.
    """

    def _vectors(self, ids):
        return [SimpleNamespace(messages=[{"role": "user", "content": "hello"}], token_ids=list(ids))]

    def test_it_reports_TRUE_when_the_ids_come_back(self, monkeypatch):
        monkeypatch.setattr(
            "src.services.probe_monitor_render.render_messages",
            lambda _t, _m, **_k: SimpleNamespace(input_ids=[1, 2, 3]),
        )
        assert builder._messages_round_trip(object(), self._vectors([1, 2, 3]), 4096) is True

    def test_it_reports_FALSE_when_the_template_adds_scaffolding(self, monkeypatch):
        """The real case: the re-render is the scored row plus template tokens."""
        monkeypatch.setattr(
            "src.services.probe_monitor_render.render_messages",
            lambda _t, _m, **_k: SimpleNamespace(input_ids=[9, 9, 1, 2, 3, 8]),
        )
        assert builder._messages_round_trip(object(), self._vectors([1, 2, 3]), 4096) is False

    def test_one_bad_vector_is_enough_to_report_FALSE(self, monkeypatch):
        calls = {"n": 0}

        def _render(_t, _m, **_k):
            calls["n"] += 1
            return SimpleNamespace(input_ids=[1, 2, 3] if calls["n"] == 1 else [4, 5])

        monkeypatch.setattr("src.services.probe_monitor_render.render_messages", _render)
        vectors = self._vectors([1, 2, 3]) + self._vectors([1, 2, 3])
        assert builder._messages_round_trip(object(), vectors, 4096) is False

    def test_NOT_CHECKED_is_None_not_False(self, monkeypatch):
        """"not checked" and "checked, and it does not round-trip" are different claims.

        A consumer may reasonably treat them differently, so collapsing them to False would state
        something the exporter does not know.
        """
        assert builder._messages_round_trip(None, self._vectors([1]), 4096) is None

        def _raise(_t, _m, **_k):
            raise ValueError("this conversation cannot be rendered")

        monkeypatch.setattr("src.services.probe_monitor_render.render_messages", _raise)
        assert builder._messages_round_trip(object(), self._vectors([1]), 4096) is None

    def test_the_contract_states_which_field_to_score(self):
        from src.schemas.probe_definition import TestVectors

        field = TestVectors.model_fields["authoritative_input"]
        assert field.default == "token_ids"
        assert "token_ids" in (TestVectors.__doc__ or "")
        # And the measured numbers are recorded where a reader of the contract will find them.
        assert "1.153" in (TestVectors.__doc__ or "")
