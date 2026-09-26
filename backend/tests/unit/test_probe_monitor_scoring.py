"""Offline scoring and the k-sparse SAE variant (032 FR-8, FR-12).

MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
  M171  `score_one` drops the per-token scores            → the trace test fails
  M172  `fires` becomes False when no threshold was placed → the no-threshold test fails
  M173  the unscored positions are not reported            → the mask test fails
  M174  the score GET returns an empty trace on FAILURE    → the failure test fails
  M175  the score GET is removed from the router           → the reachability test fails
  M176  `encode_sae_features` calls bare `encode()`        → the normalisation test fails
  M177  a non-residual SAE is accepted                     → the refusal test fails
  M178  `train_sae_variant` ranks features on every row    → the train-only test fails
  M179  `_sae_for_layer` returns None instead of raising   → the missing-SAE test fails

⚠ SCORING WAS WRITE-ONLY UNTIL THIS FILE EXISTED. `POST /probes/{id}/score` returned a
202 and a task id with nothing to read it back with, so the feature could be exercised
and never observed. That is this repo's signature failure — implemented, unit-tested by
importing the piece directly, and unreachable for any caller — so the reachability test
here asserts the GET is in the SERVED OpenAPI paths, not that the function imports.

⚠ AND THE SAE VARIANT HAD CODE BUT NO TEST. `train_sae_variant` was wired into the run
and nothing exercised it, which by this repo's own rule means it was not shipped. The
three properties that matter are tested here: the SAE's OWN training normalisation is
applied, a non-residual SAE is refused, and the feature ranking sees train rows only.
"""
import ast
import inspect
import textwrap

import numpy as np
import pytest
import torch

from src.schemas.probe_monitor import ProbeMonitorSummary


class TestTheScoreResultIsReachable:
    def test_the_GET_is_in_the_served_paths(self):
        """The live registry, not an import. Without this endpoint the POST's task id has
        no consumer and the whole scoring feature is write-only."""
        from src.main import app

        paths = app.openapi()["paths"]
        assert "/api/v1/probe-monitors/probes/{probe_id}/score/{task_id}" in paths
        assert "get" in paths["/api/v1/probe-monitors/probes/{probe_id}/score/{task_id}"]

    def test_the_POST_still_advertises_202(self):
        """Paired: a 200 would mean the model load happened in the request."""
        from src.main import app

        responses = app.openapi()["paths"][
            "/api/v1/probe-monitors/probes/{probe_id}/score"
        ]["post"]["responses"]
        assert "202" in responses

    def test_a_FAILED_task_reports_its_reason_rather_than_an_empty_trace(self):
        """An empty trace reads as "the probe scored nothing"; a failure reads as "the
        probe did not run". Asserted on the branch, since reaching it needs a broker."""
        from src.api.v1.endpoints import probe_monitors

        source = textwrap.dedent(inspect.getsource(probe_monitors.get_score_result))
        assert '"FAILURE"' in source
        assert '"error"' in source
        # And the failure branch must NOT hand back a result payload.
        tree = ast.parse(source)
        returns = [
            node for node in ast.walk(tree) if isinstance(node, ast.Return)
        ]
        assert len(returns) >= 3, "expected SUCCESS, FAILURE and pending branches"

    def test_it_reads_the_broker_OFF_the_event_loop(self):
        """`AsyncResult` blocks. On the loop it would stall every other request."""
        from src.api.v1.endpoints import probe_monitors

        tree = ast.parse(textwrap.dedent(inspect.getsource(probe_monitors.get_score_result)))
        attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "to_thread" in attrs


class TestScoreOneReturnsTheWholeTrace:
    """`score_one` assembled from real pieces: a tiny LM, a real tokenizer, a real head."""

    D_MODEL = 32

    @pytest.fixture(scope="class")
    def pieces(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

        from src.ml.probe_monitor_model import ProbeHead

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            LlamaConfig(
                vocab_size=64, hidden_size=self.D_MODEL, intermediate_size=64,
                num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
                max_position_embeddings=128,
            )
        )
        model.eval()

        words = ["<unk>", "<turn>", "</turn>", "user", "assistant", "transfer", "funds", "no"]
        vocab = {word: index for index, word in enumerate(words)}
        backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
        tokenizer.chat_template = (
            "{% for m in messages %}<turn> {{ m['role'] }} {{ m['content'] }} </turn> "
            "{% endfor %}"
        )
        tokenizer.pad_token = "<unk>"

        head = ProbeHead(
            weight=torch.randn(self.D_MODEL),
            bias=0.1,
            mean=torch.zeros(self.D_MODEL),
            std=torch.ones(self.D_MODEL),
            layer=1,
        )
        return model, tokenizer, head

    def _score(self, pieces, *, scope="all", threshold=None):
        """Drive the real scoring path with the pieces `score_one` would assemble."""
        from src.services.probe_monitor_capture import forward_scores
        from src.services.probe_monitor_render import render_messages

        model, tokenizer, head = pieces
        rendered = render_messages(
            tokenizer,
            [
                {"role": "user", "content": "transfer funds"},
                {"role": "assistant", "content": "no"},
            ],
        )
        scored = forward_scores(
            model, [rendered], head, rule="mean", scope=scope, pad_id=0
        )[0]
        mask = rendered.scored_mask(scope)
        tokens = tokenizer.convert_ids_to_tokens(rendered.input_ids)
        return {
            "aggregate": scored.aggregate,
            "threshold": threshold,
            "fires": None if threshold is None else scored.aggregate >= threshold,
            "tokens": [
                {"token": token, "scored": bool(keep)} for token, keep in zip(tokens, mask)
            ],
            "token_scores": scored.token_scores,
            "n_scored": scored.n_scored,
        }

    def test_it_returns_tokens_per_token_scores_and_the_aggregate(self, pieces):
        result = self._score(pieces)
        assert result["tokens"], "no tokens were returned"
        assert result["token_scores"], "no per-token scores were returned"
        assert isinstance(result["aggregate"], float)
        # One score per SCORED position, not per token: scaffolding has no score.
        assert len(result["token_scores"]) == result["n_scored"]
        assert result["n_scored"] == sum(1 for t in result["tokens"] if t["scored"])

    def test_the_aggregate_agrees_with_the_pure_rule(self, pieces):
        """Ties the served number to the tested function, so the trace and the aggregate
        cannot describe different detectors."""
        from src.ml.probe_monitor_model import combine_sequence

        result = self._score(pieces)
        assert result["aggregate"] == pytest.approx(
            combine_sequence("mean", result["token_scores"]), abs=1e-5
        )

    def test_every_token_is_marked_scored_or_not(self, pieces):
        """The UI shades scored tokens and marks unscored ones; a missing flag would make
        an unscored token look like a cold one.

        ⚠ MY FIRST VERSION ENDED IN `len(tokens) == len([t for t in tokens])`, which is
        true of any list — a tautology, and the third one I have written in this feature.
        What is worth asserting is that the flags line up with the RENDER: as many tokens
        as input ids, and both kinds present in a mixed-role conversation.
        """
        result = self._score(pieces, scope="assistant")
        assert all("scored" in token for token in result["tokens"])
        flags = [token["scored"] for token in result["tokens"]]
        assert any(flags), "nothing was marked scored"
        assert not all(flags), (
            "every token was marked scored under an 'assistant' scope, so the flag is not "
            "being derived from the role mask at all"
        )

    def test_a_role_scope_reduces_what_is_scored(self, pieces):
        every = self._score(pieces, scope="all")
        assistant = self._score(pieces, scope="assistant")
        assert 0 < assistant["n_scored"] < every["n_scored"]
        assert assistant["aggregate"] != every["aggregate"]

    def test_fires_is_None_when_no_threshold_was_placed(self, pieces):
        """⚠ NOT False. A probe with no calibration has not said "no" — it has said
        nothing, and rendering that as "does not fire" is a claim it cannot support."""
        assert self._score(pieces, threshold=None)["fires"] is None

    def test_fires_is_a_real_comparison_when_a_threshold_exists(self, pieces):
        result = self._score(pieces, threshold=-1e9)
        assert result["fires"] is True
        below = self._score(pieces, threshold=1e9)
        assert below["fires"] is False

    def test_the_service_reports_the_same_fields(self):
        """`score_one` is the production assembler; this asserts its payload keys so the
        hand-built path above cannot drift from it."""
        from src.services import probe_monitor_run

        source = inspect.getsource(probe_monitor_run.score_one)
        for key in (
            '"aggregate"', '"threshold"', '"fires"', '"tokens"', '"token_scores"',
            '"n_scored"', '"role_mask_reliable"', '"truncated"',
        ):
            assert key in source, f"score_one does not report {key}"

    def test_score_one_does_not_report_fires_as_False_without_a_threshold(self):
        from src.services import probe_monitor_run

        source = inspect.getsource(probe_monitor_run.score_one)
        assert "None if probe.threshold is None" in source


class TestTheSAEVariant:
    """4.5. It had code and no test, which by this repo's rule means it was not shipped."""

    def test_it_uses_the_SAEs_OWN_TRAINING_NORMALISATION(self):
        """⚠ A BARE `encode()` HANDS THE DICTIONARY RAW ACTIVATIONS. `encode()` does not
        normalise — `forward()` does, then calls it — so every circuit discovered from a
        capture on this estate was once mined in the wrong basis: the features fire, the
        numbers are plausible, and the basis is wrong (MIS-E2E-083)."""
        from src.services import probe_monitor_run

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(probe_monitor_run.encode_sae_features))
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "encode_with_training_normalization" in called
        # And NOT a bare `sae.encode(...)`.
        attrs = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "encode" not in attrs, "a bare encode() bypasses the trained normalisation"

    def test_a_non_residual_SAE_is_REFUSED(self):
        """A probe reads `resid_post`; an SAE trained on an MLP output describes a
        different space, and encoding one against the other yields features that mean
        nothing in particular."""
        from src.services import probe_monitor_run

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(probe_monitor_run.encode_sae_features))
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "refuse_non_residual" in called

    def test_the_refusal_actually_raises(self):
        """Prove the shared helper bites, rather than trusting that it is called."""
        from src.services.sae_hook_support import UnsupportedSaeHook, refuse_non_residual

        with pytest.raises(UnsupportedSaeHook):
            refuse_non_residual("mlp", "a probe monitor")
        # And a residual SAE passes, or the guard would refuse everything.
        refuse_non_residual("residual", "a probe monitor")

    def test_features_are_ranked_on_TRAIN_ROWS_ONLY(self):
        """Ranking on validation rows leaks them into the model's STRUCTURE — which
        features exist at all — so the validation AUROC that drives early stopping is no
        longer held out.

        Asserted from the AST's ARGUMENT NAMES rather than a substring: a text search for
        `select_sae_features(train_rows` matches a comment describing the rule as readily
        as the call obeying it, and this repo has shipped that mistake five times.
        """
        from src.services import probe_monitor_run

        tree = ast.parse(textwrap.dedent(inspect.getsource(probe_monitor_run.train_sae_variant)))
        ranking = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "select_sae_features"
        ]
        assert ranking, "train_sae_variant does not call select_sae_features at all"
        argument_names = [
            arg.id for arg in ranking[0].args if isinstance(arg, ast.Name)
        ]
        assert argument_names[:2] == ["train_rows", "train_labels"], (
            f"the ranking is fed {argument_names}, not the training rows"
        )

    def test_the_variant_is_persisted_AS_an_sae_probe_with_its_indices(self):
        """Without `variant='sae'` and the indices, an exported definition could not
        reconstruct which features the probe reads."""
        from src.services import probe_monitor_run

        source = textwrap.dedent(inspect.getsource(probe_monitor_run.train_sae_variant))
        assert 'variant="sae"' in source
        assert "sae_feature_indices=" in source

    def test_a_missing_SAE_RAISES_and_names_the_layer(self):
        """"no SAE" is unactionable; "no ready SAE for <model> at layer 12" says what to
        train or import."""
        from src.services.probe_monitor_run import _sae_for_layer

        class _Empty:
            def query(self, *a, **k):
                return self

            def filter(self, *a, **k):
                return self

            def first(self):
                return None

        with pytest.raises(ValueError) as caught:
            _sae_for_layer(_Empty(), "m_abc", 12)
        message = str(caught.value)
        assert "m_abc" in message and "12" in message

    def test_the_run_CALLS_the_variant_when_it_is_requested(self):
        """Reachability: an SAE variant nothing calls is a flag with no effect."""
        from src.services import probe_monitor_run

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(probe_monitor_run.execute_probe_run))
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "train_sae_variant" in called
        assert "_sae_for_layer" in called

    def test_it_is_gated_on_the_config_flag(self):
        from src.services import probe_monitor_run

        source = textwrap.dedent(inspect.getsource(probe_monitor_run.execute_probe_run))
        assert 'context.config.get("sae_variant")' in source

    def test_the_k_sparse_slice_keeps_the_chosen_columns(self):
        """The arithmetic the variant depends on, checked directly: slicing by index must
        preserve column ORDER, or the trained weights map to the wrong features."""
        features = np.arange(24, dtype=np.float32).reshape(4, 6)
        indices = np.array([1, 4, 5])
        sliced = features[:, indices]
        assert sliced.shape == (4, 3)
        assert np.allclose(sliced[:, 0], features[:, 1])
        assert np.allclose(sliced[:, 2], features[:, 5])

    def test_the_indices_are_sorted_so_the_basis_is_stable(self):
        """An exported definition's feature list must be comparable between two probes."""
        from src.services.probe_monitor_trainer import select_sae_features

        rng = np.random.default_rng(3)
        labels = [i % 2 for i in range(60)]
        rows = []
        for i in range(60):
            block = rng.normal(size=(2, 10)).astype(np.float32)
            block[:, 7] += labels[i] * 4.0
            block[:, 2] += labels[i] * 3.0
            rows.append(block)
        chosen = select_sae_features(rows, labels, k=2)
        assert list(chosen) == sorted(chosen)
        assert set(chosen.tolist()) == {2, 7}


class TestTheCurveIsInTheReportAndNotInTheList:
    """`val_metrics["history"]` is one entry per epoch. The report carries it; the LIST,
    which the panel polls while a run is going, must not — a few hundred entries per probe
    turns a one-line-per-probe table into hundreds of kilobytes on every poll.

    MUTATION CONTROLS:
      C4  `summary_without_curve` returns the summary unchanged  → the stripping tests
      C5  `list_probes` stops calling it                         → the AST wiring test
      C6  `get_probe_report` starts calling it                   → the report test
    """

    class _Probe:
        """The attribute surface `ProbeMonitorSummary.model_validate` reads."""

        def __init__(self, val_metrics):
            from datetime import datetime, timezone

            self.id = "pm_test"
            self.run_id = "pmr_test"
            self.layer = 11
            self.rule = "attention"
            self.rule_params = {}
            self.variant = "dense"
            self.sae_id = None
            self.sae_feature_indices = None
            self.val_metrics = val_metrics
            self.selected = True
            self.threshold = 1.0
            self.target_fpr = 0.01
            self.realised_fpr = 0.01
            self.threshold_source = "validation_negatives"
            self.streamable = True
            self.rung = 2
            self.rung_reasons = []
            self.created_at = datetime.now(timezone.utc)

    def _curve(self, epochs=300):
        return {
            "val_auroc": 0.96,
            "best_epoch": 48,
            "epochs_run": epochs,
            "history": [
                {"epoch": n, "loss": 1.0 / n, "val_auroc": 0.9} for n in range(1, epochs + 1)
            ],
        }

    def test_the_curve_is_stripped(self):
        from src.api.v1.endpoints.probe_monitors import summary_without_curve

        summary = summary_without_curve(self._Probe(self._curve()))
        assert "history" not in summary.val_metrics

    def test_the_summary_numbers_survive_the_stripping(self):
        """Stripping must take the curve and nothing else."""
        from src.api.v1.endpoints.probe_monitors import summary_without_curve

        summary = summary_without_curve(self._Probe(self._curve()))
        assert summary.val_metrics["val_auroc"] == 0.96
        assert summary.val_metrics["best_epoch"] == 48
        assert summary.val_metrics["epochs_run"] == 300

    def test_a_probe_without_a_curve_is_unchanged(self):
        from src.api.v1.endpoints.probe_monitors import summary_without_curve

        metrics = {"val_auroc": 0.5}
        summary = summary_without_curve(self._Probe(dict(metrics)))
        assert summary.val_metrics == metrics

    def test_the_summary_holds_its_OWN_metrics_dict(self):
        """The invariant the stripping RELIES ON, asserted rather than assumed.

        ⚠ MEASURED AFTER A MUTATION SURVIVED. Replacing the `model_copy` with
        `metrics.pop("history", None)` left the suite green, and the reason is that
        `model_validate` copies the dict — so the pop cannot reach the ORM object's JSONB
        and the two forms are equivalent TODAY. That makes the obvious
        "it does not mutate the probe" test a tautology: it cannot fail while pydantic
        copies, so it asserts nothing about this module.

        This asserts the copying itself, which is the load-bearing external behaviour. If
        pydantic ever stops copying, a pop-based implementation would mark the ORM object
        dirty and DELETE the stored curve on the session's next flush — a listing
        destroying the data it was asked not to send — and this test goes red first.
        """
        probe = self._Probe(self._curve(epochs=5))
        summary = ProbeMonitorSummary.model_validate(probe)
        assert summary.val_metrics is not probe.val_metrics, (
            "model_validate no longer copies val_metrics; `summary_without_curve` must not "
            "be written as an in-place pop"
        )

    def test_the_listing_leaves_the_probes_curve_alone(self):
        """The consequence, which follows from the invariant above."""
        from src.api.v1.endpoints.probe_monitors import summary_without_curve

        probe = self._Probe(self._curve(epochs=5))
        summary_without_curve(probe)
        assert "history" in probe.val_metrics

    def test_the_stripping_actually_shrinks_the_payload(self):
        """The control for the point of the exercise."""
        import json

        from src.api.v1.endpoints.probe_monitors import summary_without_curve

        probe = self._Probe(self._curve(epochs=400))
        full = len(json.dumps(probe.val_metrics))
        stripped = len(json.dumps(summary_without_curve(probe).val_metrics))
        assert stripped * 20 < full, f"{stripped} vs {full} bytes"

    def _calls(self, function):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        return {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_list_probes_calls_it(self):
        from src.api.v1.endpoints import probe_monitors

        assert "summary_without_curve" in self._calls(probe_monitors.list_probes), (
            "the list endpoint does not strip the curve"
        )

    def test_the_report_does_NOT_call_it(self):
        """The report is the one place the curve is available; stripping there would leave
        it unreachable everywhere."""
        from src.api.v1.endpoints import probe_monitors

        assert "summary_without_curve" not in self._calls(probe_monitors.get_probe_report)

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        def decoy():
            # summary_without_curve(probe)
            return "summary_without_curve"

        assert "summary_without_curve" not in self._calls(decoy)
