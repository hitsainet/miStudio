"""Every mechanism this arc added must be reachable from production.

WHY THIS FILE EXISTS. This project's signature failure is a capability that is
implemented, unit-tested and documented while nothing calls it — an entire
16-tool MCP surface once shipped that way, green suite and all. Four of the
mechanisms in this arc had ZERO production callers when first written
(`pack_token_blocks`, `render_conversation`, `split_documents`,
`spliced_ce_delta`).

These assertions read the AST for a CALL, not the text for a NAME: a substring
search matches the explanatory comments that describe the mechanism, which is a
guard that passes for the wrong reason.
"""

import ast
import inspect

import pytest


def _called_names(module) -> set:
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute):
            names.add(fn.attr)
        elif isinstance(fn, ast.Name):
            names.add(fn.id)
    return names


class TestTheTokenizationPathUsesThem:

    def test_packing_is_called_by_the_worker(self):
        """The STREAMING packer is the production one.

        `pack_token_blocks` returns lists and is what the unit tests exercise
        directly; the worker uses `iter_packed_blocks` so a 1M-document corpus
        does not have to fit in RAM. A test in test_document_packing.py pins the
        two to identical output, which is what makes substituting one for the
        other safe.
        """
        from src.workers import dataset_tasks

        called = _called_names(dataset_tasks)
        assert "iter_packed_blocks" in called, (
            "packing is implemented and tested but nothing calls it; a "
            "3%-occupancy corpus stays at 3%"
        )

    def test_the_renderer_is_called_by_preprocessing(self):
        from src.services import tokenization_service

        assert "render_conversation" in _called_names(tokenization_service), (
            "the chat renderer has no production caller, so conversations are "
            "still flattened with invented <|user|> markers"
        )

    def test_the_endpoint_forwards_the_new_tokenization_fields(self):
        from src.api.v1.endpoints import datasets

        src = inspect.getsource(datasets)
        for field in ("text_column", "chat_format", "pack_sequences"):
            assert f"{field}=request.{field}" in src, (
                f"{field} is accepted by the schema and never forwarded to the "
                f"task — the same defect as remove_all_punctuation"
            )


class TestTheTrainingPathUsesThem:

    def test_the_mask_is_applied(self):
        from src.workers import training_tasks

        called = _called_names(training_tasks)
        assert "load_valid_mask" in called
        assert "resolve_flat_indices" in called, (
            "the loader no longer maps selection indices through the mask"
        )

    def test_the_holdout_split_is_applied(self):
        from src.workers import training_tasks

        assert "split_documents" in _called_names(training_tasks), (
            "the held-out split has no caller, so every reported number is "
            "still in-sample"
        )

    def test_the_mixture_allocation_is_applied(self):
        from src.workers import training_tasks

        assert "allocate_tokens" in _called_names(training_tasks)

    def test_density_monitoring_is_applied(self):
        from src.workers import training_tasks

        called = _called_names(training_tasks)
        assert "update_firing_rate" in called and "density_summary" in called

    def test_the_seed_is_applied_to_torch(self):
        from src.workers import training_tasks

        assert "manual_seed" in _called_names(training_tasks), (
            "torch.manual_seed is not called, so weight init and calibration "
            "remain unseeded and two runs are not comparable"
        )


class TestTheExtractionPathUsesThem:

    def test_the_mask_sidecar_is_written(self):
        from src.services import activation_service

        assert "_write_attention_mask" in _called_names(activation_service)

    def test_tokenization_selection_is_by_model(self):
        from src.workers import model_tasks

        assert "select_tokenization_for_model" in _called_names(model_tasks)

    def test_the_micro_batch_budget_is_applied(self):
        from src.services import activation_service

        assert "micro_batch_size_for_length" in _called_names(activation_service)


class TestTheEvaluationIsNoLongerUnwired:
    """This class used to assert `spliced_ce_delta` had NO caller, and told
    whoever wired it to replace the assertion. This is that replacement.

    It was unwired because the cached-activation path never loads a base model.
    It now runs once after the checkpoint is saved, where the load is affordable
    and a failure cannot cost a finished training.
    """

    def test_spliced_ce_has_a_production_caller(self):
        from src.workers import training_tasks

        assert "spliced_ce_delta" in _called_names(training_tasks), (
            "the spliced-CE evaluation is unreachable again"
        )

    def test_the_evaluator_itself_is_called(self):
        from src.workers import training_tasks

        assert "_evaluate_spliced_ce" in _called_names(training_tasks)


class TestTheNewHyperparametersReachTheWorker:
    """A knob the worker never receives is the same defect as one with no caller.

    `training_service` builds the row with
    `training_data.hyperparameters.model_dump()`, so a field declared on the
    WRONG schema class would validate, appear in the API, and never arrive.
    """

    @pytest.mark.parametrize(
        "field",
        ["seed", "dataset_weights", "holdout_fraction", "ste_bandwidth", "target_l0"],
    )
    def test_the_field_is_on_the_hyperparameters_schema(self, field):
        from src.schemas.training import TrainingHyperparameters

        assert field in TrainingHyperparameters.model_fields, (
            f"{field} is not on TrainingHyperparameters, so model_dump() will "
            f"never carry it into Training.hyperparameters"
        )

    def test_values_survive_the_dump_the_service_performs(self):
        from src.schemas.training import TrainingHyperparameters

        hp = TrainingHyperparameters(
            hidden_dim=2048,
            latent_dim=8192,
            batch_size=2048,
            total_steps=1000,
            learning_rate=7e-5,
            architecture_type="jumprelu",
            seed=1234,
            dataset_weights=[0.7, 0.3],
            holdout_fraction=0.05,
            ste_bandwidth=0.25,
        )
        dumped = hp.model_dump()

        assert dumped["seed"] == 1234
        assert dumped["dataset_weights"] == [0.7, 0.3]
        assert dumped["holdout_fraction"] == 0.05
        assert dumped["ste_bandwidth"] == 0.25

    def test_the_worker_reads_them_from_hp(self):
        """The other half: the names the worker looks up must match."""
        import ast
        import inspect

        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        looked_up = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "hp"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                looked_up.add(node.args[0].value)

        for field in ("seed", "dataset_weights", "holdout_fraction", "ste_bandwidth"):
            assert field in looked_up, (
                f"the worker never reads hp[{field!r}]; it is configurable and inert"
            )


class TestTheSeedIsPersistedNotJustLogged:
    """Round 2, H5: the derived seed was used, logged, and never written back.

    `TrainingHyperparameters.seed` describes itself as "Persisted, so a
    multi-seed comparison is attributable". It was not — `hyperparameters['seed']`
    stayed null and the value was recoverable only by knowing to re-run crc32 on
    the training id.
    """

    def test_the_worker_writes_the_seed_back(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        # Round 3, M5: the previous assertion watched the DICT BUILD. Deleting
        # the assignment below left the suite green — and without it SQLAlchemy
        # never marks the JSON column dirty, so nothing is written at all.
        assert "row.hyperparameters = merged" in src, (
            "the seed is built into a dict and never assigned back, so the JSON "
            "column is never marked dirty and nothing is persisted"
        )
        assert "db.commit()" in src

    def test_the_derivation_is_not_a_constant(self):
        """Round 2's M4 pinned the seed to 42 — the exact defect the arc claims
        to have removed — and the full suite stayed green."""
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "zlib.crc32(training_id" in src, (
            "the seed no longer derives from the training id, so every run "
            "shares one seed and 'independent' runs read identical data"
        )


class TestAnUnhonouredMixtureIsReported:
    """Round 2, M3/M4: weights are a silent no-op on the paths large corpora
    actually take — the CPU-streaming path always, and the on-the-fly path
    entirely."""

    def test_the_cached_path_warns_when_weights_cannot_bind(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "could NOT be honoured" in src, (
            "when every source fits the budget the requested mixture silently "
            "becomes availability-proportional, and the log line reporting the "
            "realised split reads as confirmation"
        )

    def test_the_cached_path_actually_reads_the_weights(self):
        """A key-presence assertion is not enough — and this test exists because
        that mistake was made here.

        `test_the_worker_reads_them_from_hp` asserts hp.get('dataset_weights')
        appears somewhere in the module. Once a WARNING about weights was added
        to the on-the-fly path, that assertion was satisfied by the warning even
        with the cached path's read deleted — so round 2's M5 survived a second
        time. The binding that matters is the one feeding allocate_tokens.
        """
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "requested_weights = hp.get('dataset_weights')" in src, (
            "the cached-activation loader no longer reads dataset_weights, so "
            "allocate_tokens always receives None and the mixture silently "
            "follows availability"
        )
        assert "requested_weights\n" in src or "requested_weights," in src, (
            "the value is read and never passed to the allocator"
        )

    def test_the_on_the_fly_path_warns_that_weights_are_ignored(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "will be IGNORED" in src, (
            "the on-the-fly path never reads dataset_weights and said nothing"
        )


class TestBothTrainingPathsExcludePadding:
    """Round 1, H3: the on-the-fly branch trained on padding, unwarned.

    It is the same defect as the cached path, in the other branch of the same
    function, with the mask already in local scope — built, handed to the
    forward, and dropped. On the Bloomberg tokenization (3.3% real) ~97% of
    every batch drawn there was PAD. The arc's "NEVER SILENT" guarantee applied
    only to the cached branch.
    """

    def test_the_on_the_fly_branch_applies_the_mask(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "real = attention_mask_tensor.reshape(-1).bool()" in src, (
            "the on-the-fly path flattens every position again, so PAD "
            "activations are sampled as training data"
        )

    def test_the_warn_branch_exists_on_the_cached_path(self):
        """Round 1's M3 deleted this and the whole suite stayed green."""
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "PADDING NOT MASKED" in src, (
            "the only warning emitted when a mask cannot be recovered is gone; "
            "training on padding becomes silent again"
        )


class TestTokenizationSelectionIsFixedEverywhere:
    """Round 1, H6: `select_tokenization_for_model` was wired to one of two
    entry points. The on-the-fly training path kept an unordered `.first()` on
    (dataset, model) while the uniqueness constraint is
    (dataset, model, max_length) — several rows per pair are EXPECTED.

    Giving the directory name a max_length made this MORE dangerous, not less:
    before, the arbitrary pick returned the same bytes.
    """

    def test_the_training_path_uses_the_selector_too(self):
        import inspect

        from src.workers import training_tasks

        src = inspect.getsource(training_tasks)
        assert "select_tokenization_for_model(" in src, (
            "the on-the-fly path still picks a tokenization by row order, so "
            "which context window a run trains on is decided by Postgres"
        )
        assert "DatasetTokenization.model_id == training.model_id\n                    ).first()" not in src


class TestTheProvenanceDecisionsAreCalled:
    """Round 3: both were guarded by scrapes and both mutations survived."""

    def test_the_worker_calls_the_bos_resolver(self):
        from src.workers import dataset_tasks

        assert "resolve_add_special_tokens" in _called_names(dataset_tasks), (
            "the BOS suppression is inline again; deleting it is invisible"
        )

    def test_the_worker_calls_the_text_column_resolver(self):
        from src.workers import dataset_tasks

        assert "resolve_recorded_text_column" in _called_names(dataset_tasks), (
            "the recorded text column is computed inline again"
        )
