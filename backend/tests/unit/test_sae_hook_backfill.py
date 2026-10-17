"""The backfill that repairs SAE rows recording no hook and no layer (review R3-B, R3B-11).

A data repair gets one attempt against real rows at deploy time, so its behaviour is
pinned here rather than discovered in production. Every rule R3-B's design names has a
test: only NULL fields are written, a recorded layer that disagrees is REPORTED and not
changed, ``trained`` rows are excluded (their pre-A5 cfg.json says ``resid_post`` for
every hook, so re-resolving one would write a wrong hook over a right one), an in-flight
job blocks the whole apply, and a second run changes nothing.

The resolution itself is NOT re-tested here -- it is the writers' own
``resolve_sae_hook`` / ``resolve_sae_layer`` (pinned by ``test_sae_hooks_from_real_configs``),
and one test asserts the backfill agrees with them on the same inputs rather than
carrying a second copy of the rule.

The CLI is exercised through its own ``run()`` against a real session, not merely
imported: a backfill nobody can invoke repairs nothing.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored from a byte
copy, sha256 and ``git diff`` verified clean):
  BF1 ``candidate_rows`` drops the source / training_id exclusion
        -> test_a_trained_sae_is_never_touched
  BF2 ``apply_backfill`` writes ``plan.new_hook`` without the ``is None`` guard
        -> test_apply_writes_only_the_null_fields
  BF3 ``apply_backfill`` ignores ``plan.in_flight``
        -> test_an_in_flight_job_blocks_the_apply_and_writes_nothing
  BF4 ``resolve_row`` passes no origins to the resolver
        -> test_it_resolves_by_name_when_the_files_are_gone, test_the_hook_comes_from_the_writers_resolver
  BF5 ``plan_backfill`` prefers the resolved layer over a recorded one
        -> test_apply_writes_only_the_null_fields (the disagreement case)
  BF6 the CLI's ``run()`` applies even without ``--apply``
        -> test_the_cli_dry_run_writes_nothing
  BF7 ``apply_backfill`` stops recording ``hook_backfill`` provenance
        -> test_it_records_what_it_changed_and_what_was_there_before
"""

import ast
import importlib.util
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from src.db import sae_hook_backfill as mod
from src.models.external_sae import ExternalSAE
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature

BACKEND = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND / "scripts" / "backfill_sae_hooks.py"


# ── SAE directories as they arrive on disk (the shapes the resolver reads) ──


def _saelens_dir(root: Path, hook_name: Optional[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"d_in": 8, "d_sae": 16, "architecture": "standard", "model_name": "tiny"}
    if hook_name is not None:
        cfg["hook_name"] = hook_name
    (root / "cfg.json").write_text(json.dumps(cfg))
    (root / "sae_weights.safetensors").write_bytes(b"")
    return root


def _gemma_scope_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    np.savez(
        root / "params.npz",
        W_enc=np.zeros((8, 16), dtype=np.float32), W_dec=np.zeros((16, 8), dtype=np.float32),
        b_enc=np.zeros(16, dtype=np.float32), b_dec=np.zeros(8, dtype=np.float32),
        threshold=np.zeros(16, dtype=np.float32),
    )
    return root


def _row(sae_id: str, **kwargs) -> ExternalSAE:
    fields = dict(
        name=sae_id, source="huggingface", status="ready", hook_type=None, layer=None,
        progress=100.0, sae_metadata={},
    )
    fields.update(kwargs)
    return ExternalSAE(id=sae_id, **fields)


async def _plan(async_session):
    return await async_session.run_sync(mod.plan_backfill)


async def _apply(async_session, plans):
    return await async_session.run_sync(lambda session: mod.apply_backfill(session, plans))


def _by_id(plans):
    return {plan.sae_id: plan for plan in plans}


# ── which rows it considers ─────────────────────────────────────────────────


class TestWhichRowsItConsiders:
    @pytest.mark.asyncio
    async def test_a_row_recording_both_is_not_a_candidate(self, async_session):
        async_session.add(_row("sae_complete", hook_type="residual", layer=20))
        async_session.add(_row("sae_no_hook", layer=20))
        async_session.add(_row("sae_no_layer", hook_type="residual"))
        await async_session.commit()

        assert set(_by_id(await _plan(async_session))) == {"sae_no_hook", "sae_no_layer"}

    @pytest.mark.asyncio
    async def test_a_deleted_row_is_still_repaired(self, async_session):
        """A deleted row can be restored, and would then be read at a fabricated layer 0."""
        async_session.add(_row("sae_deleted", status="deleted"))
        await async_session.commit()

        assert "sae_deleted" in _by_id(await _plan(async_session))

    @pytest.mark.asyncio
    async def test_a_trained_sae_is_never_touched(self, async_session, tmp_path):
        """BOTH halves of the exclusion: ``source='trained'`` and a training import.

        A pre-A5 training export carries a cfg.json naming ``resid_post`` for every hook,
        so resolving one would record ``residual`` over an MLP SAE's real hook.
        """
        from src.models.model import Model, ModelStatus, QuantizationFormat
        from src.models.training import Training, TrainingStatus

        exported = _saelens_dir(tmp_path / "layer_3_mlp", "blocks.3.hook_resid_post")
        async_session.add(Model(
            id="m_bf", name="tiny", architecture="llama", params_count=1_000,
            quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
        ))
        await async_session.flush()
        async_session.add(Training(
            id="train_bf", model_id="m_bf", dataset_id=str(uuid.uuid4()),
            status=TrainingStatus.COMPLETED.value, total_steps=10,
            hyperparameters=dict(latent_dim=16, hidden_dim=8),
        ))
        await async_session.flush()
        async_session.add(_row("sae_trained_source", source="trained", local_path=str(exported)))
        async_session.add(_row("sae_trained_link", source="local", training_id="train_bf",
                               local_path=str(exported)))
        await async_session.commit()

        assert _by_id(await _plan(async_session)) == {}


# ── resolution: the writers' resolver, never a second copy ──────────────────


class TestResolution:
    @pytest.mark.asyncio
    async def test_the_hook_comes_from_the_writers_resolver(self, async_session, tmp_path):
        """A second resolver would drift from the one the download task calls."""
        from src.services.sae_manager_service import resolve_sae_hook, resolve_sae_layer

        directory = _saelens_dir(tmp_path / "sae", "blocks.3.hook_mlp_out")
        async_session.add(_row(
            "sae_cfg", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_cfg"]
        expected = resolve_sae_hook(
            directory, "google/gemma-scope-2b-pt-res", "layer_20/width_16k"
        )
        assert (plan.new_hook, plan.hook_source) == (expected.hook_type, expected.source)
        assert plan.new_layer == resolve_sae_layer(
            directory, expected.hook_type, "google/gemma-scope-2b-pt-res", "layer_20/width_16k"
        )
        # The config outranks the repository name, so this -res row is an MLP SAE and the
        # report must say the consumers will now refuse it.
        assert plan.new_hook == "blocks.3.hook_mlp_out" and plan.new_layer == 3
        assert plan.refusal_after and "MLP" in plan.refusal_after

    @pytest.mark.asyncio
    async def test_a_cfgless_gemma_scope_set_resolves_from_its_files_and_names(
        self, async_session, tmp_path
    ):
        directory = _gemma_scope_dir(tmp_path / "layer_20" / "width_16k" / "average_l0_71")
        async_session.add(_row(
            "sae_gs", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_gs"]
        assert (plan.new_hook, plan.new_layer, plan.files_present) == ("residual", 20, True)
        assert plan.refusal_after is None

    @pytest.mark.asyncio
    async def test_it_resolves_by_name_when_the_files_are_gone(self, async_session, tmp_path):
        """A deleted row's files are gone; its repository and path still name the set."""
        async_session.add(_row(
            "sae_absent", status="deleted", local_path=str(tmp_path / "gone"),
            hf_repo_id="google/gemma-scope-2b-pt-att",
            hf_filepath="layer_7/width_16k/average_l0_71",
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_absent"]
        assert plan.files_present is False
        assert (plan.new_hook, plan.new_layer) == ("attention", 7)

    @pytest.mark.asyncio
    async def test_a_local_import_resolves_from_the_path_it_came_from(
        self, async_session, tmp_path
    ):
        source = _gemma_scope_dir(
            tmp_path / "gemma-scope-2b-pt-mlp" / "layer_9" / "width_16k" / "average_l0_71"
        )
        async_session.add(_row(
            "sae_local", source="local", local_path=str(tmp_path / "copied"),
            sae_metadata={"original_path": str(source)},
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_local"]
        assert (plan.new_hook, plan.new_layer) == ("mlp", 9)

    @pytest.mark.asyncio
    async def test_a_row_nothing_records_keeps_its_nulls(self, async_session, tmp_path):
        """Nothing is invented. An unresolvable row is reported, not filled with a guess."""
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "sae.safetensors").write_bytes(b"")
        async_session.add(_row(
            "sae_unknown", local_path=str(bare), hf_repo_id="someone/saes", hf_filepath="sae",
        ))
        await async_session.commit()

        plans = await _plan(async_session)
        plan = _by_id(plans)["sae_unknown"]
        assert (plan.new_hook, plan.new_layer) == (None, None)
        assert plan.changes is False and plan.unresolved == ["hook_type", "layer"]

        assert await _apply(async_session, plans) == 0
        row = await async_session.get(ExternalSAE, "sae_unknown")
        await async_session.refresh(row)
        assert (row.hook_type, row.layer) == (None, None)


# ── write rules ─────────────────────────────────────────────────────────────


class TestWriteRules:
    @pytest.mark.asyncio
    async def test_a_plan_writes_nothing(self, async_session, tmp_path):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_dry", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_dry"]
        assert plan.changes is True  # there IS something to write, and planning did not

        row = await async_session.get(ExternalSAE, "sae_dry")
        await async_session.refresh(row)
        assert (row.hook_type, row.layer) == (None, None)
        assert "hook_backfill" not in (row.sae_metadata or {})

    @pytest.mark.asyncio
    async def test_apply_writes_only_the_null_fields(self, async_session, tmp_path):
        """A recorded layer that disagrees with the files is REPORTED, never overwritten.

        The row is a claim someone made; silently correcting it would hide exactly the
        disagreement an operator needs to see.
        """
        directory = _gemma_scope_dir(tmp_path / "layer_20" / "width_16k")
        async_session.add(_row(
            "sae_partial", layer=7, local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        plans = await _plan(async_session)
        plan = _by_id(plans)["sae_partial"]
        assert (plan.old_layer, plan.new_layer, plan.layer_disagreement) == (7, 7, 20)
        assert plan.writes_hook is True and plan.writes_layer is False

        assert await _apply(async_session, plans) == 1
        row = await async_session.get(ExternalSAE, "sae_partial")
        await async_session.refresh(row)
        assert row.hook_type == "residual"
        assert row.layer == 7, "the recorded layer was overwritten by the resolved one"

    @pytest.mark.asyncio
    async def test_a_recorded_hook_is_never_replaced(self, async_session, tmp_path):
        """Asserted at BOTH levels, because either guard alone leaves the row correct.

        Mutation control BF2 -- the PLAN preferring the resolved hook over the recorded one --
        SURVIVED the first run: the apply-time ``is None`` guard caught it, so the row still
        ended up right and a test that only read the row stayed green. Two guards agreeing by
        construction is the same trap as two fixtures agreeing by construction, so the plan is
        asserted here and the apply-time guard has its own test below.
        """
        directory = _saelens_dir(tmp_path / "sae", "blocks.3.hook_mlp_out")
        async_session.add(_row("sae_hooked", hook_type="attention", local_path=str(directory)))
        await async_session.commit()

        plans = await _plan(async_session)
        plan = _by_id(plans)["sae_hooked"]
        assert plan.new_hook == "attention", "the plan proposes replacing a recorded hook"
        assert plan.writes_hook is False

        await _apply(async_session, plans)
        row = await async_session.get(ExternalSAE, "sae_hooked")
        await async_session.refresh(row)
        assert row.hook_type == "attention" and row.layer == 3

    @pytest.mark.asyncio
    async def test_a_stale_plan_cannot_overwrite_a_value_recorded_since(
        self, async_session, tmp_path
    ):
        """The apply re-checks the LIVE row, not the plan it was handed.

        Between the dry run an operator reads and the ``--apply`` they then run, someone can
        record the hook by hand -- and the plan in flight still says NULL. Without this, the
        apply-time guard was unfalsifiable: every path to it was already protected by the
        plan, so no mutation of it could turn a test red.
        """
        directory = _gemma_scope_dir(tmp_path / "layer_20" / "width_16k")
        async_session.add(_row(
            "sae_stale", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        plans = await _plan(async_session)
        assert _by_id(plans)["sae_stale"].new_hook == "residual"

        row = await async_session.get(ExternalSAE, "sae_stale")
        row.hook_type = "attention"
        await async_session.commit()

        await _apply(async_session, plans)
        await async_session.refresh(row)
        assert row.hook_type == "attention", "a stale plan overwrote a hook recorded since"
        assert row.layer == 20, "the layer was still NULL and should have been filled"

    @pytest.mark.asyncio
    async def test_it_records_what_it_changed_and_what_was_there_before(
        self, async_session, tmp_path
    ):
        directory = _gemma_scope_dir(tmp_path / "layer_20" / "width_16k")
        async_session.add(_row(
            "sae_prov", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        await _apply(async_session, await _plan(async_session))
        row = await async_session.get(ExternalSAE, "sae_prov")
        await async_session.refresh(row)
        record = row.sae_metadata["hook_backfill"]
        assert record["by"] == mod.BACKFILL_BY
        assert record["previous"] == {"hook_type": None, "layer": None}
        assert record["at"]
        assert row.sae_metadata["hook_source"] == "gemma_scope_name"

    @pytest.mark.asyncio
    async def test_a_second_run_changes_nothing(self, async_session, tmp_path):
        directory = _gemma_scope_dir(tmp_path / "layer_20" / "width_16k")
        async_session.add(_row(
            "sae_twice", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        assert await _apply(async_session, await _plan(async_session)) == 1
        assert await _plan(async_session) == []
        assert await _apply(async_session, await _plan(async_session)) == 0


# ── dependents and in-flight jobs ───────────────────────────────────────────


def _extraction(job_id: str, sae_id: str, status: ExtractionStatus) -> ExtractionJob:
    return ExtractionJob(id=job_id, external_sae_id=sae_id, status=status, config={})


class TestDependentsAndInFlightJobs:
    @pytest.mark.asyncio
    async def test_the_plan_lists_what_depends_on_the_row(self, async_session, tmp_path):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_deps", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        await async_session.flush()
        async_session.add(_extraction("ext_done", "sae_deps", ExtractionStatus.COMPLETED))
        await async_session.flush()
        async_session.add(Feature(
            id="feat_deps_0", external_sae_id="sae_deps", extraction_job_id="ext_done",
            neuron_index=0, name="feature_0", activation_frequency=0.01,
            interpretability_score=0.5, max_activation=1.0, mean_activation=0.1,
        ))
        await async_session.commit()

        plan = _by_id(await _plan(async_session))["sae_deps"]
        assert plan.dependents == {"features": 1, "extraction_jobs": 1}
        assert plan.in_flight == [], "a completed job is not in flight"
        assert "features=1" in mod.format_report([plan])

    @pytest.mark.asyncio
    async def test_an_in_flight_job_blocks_the_apply_and_writes_nothing(
        self, async_session, tmp_path
    ):
        """A running extraction re-reads the row's layer as it works."""
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_busy", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        async_session.add(_row(
            "sae_idle", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-att", hf_filepath="layer_5/width_16k",
        ))
        await async_session.flush()
        async_session.add(_extraction("ext_live", "sae_busy", ExtractionStatus.EXTRACTING))
        await async_session.commit()

        plans = await _plan(async_session)
        assert _by_id(plans)["sae_busy"].in_flight == ["extraction job ext_live (extracting)"]

        with pytest.raises(mod.BackfillBlocked, match="ext_live"):
            await _apply(async_session, plans)

        # NOTHING is written -- not even the row nobody is reading. A half repair while a
        # job reads the other half is worse than no repair.
        for sae_id in ("sae_busy", "sae_idle"):
            row = await async_session.get(ExternalSAE, sae_id)
            await async_session.refresh(row)
            assert (row.hook_type, row.layer) == (None, None), sae_id

    @pytest.mark.asyncio
    async def test_a_job_on_another_sae_does_not_block(self, async_session, tmp_path):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_free", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        async_session.add(_row("sae_other", hook_type="residual", layer=1))
        await async_session.flush()
        async_session.add(_extraction("ext_elsewhere", "sae_other", ExtractionStatus.EXTRACTING))
        await async_session.commit()

        assert await _apply(async_session, await _plan(async_session)) == 1


# ── the CLI reaches the module ──────────────────────────────────────────────


def _load_script():
    spec = importlib.util.spec_from_file_location("backfill_sae_hooks_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def _fixed_session(session):
    yield session


class TestTheCliReachesTheBackfill:
    def test_it_calls_the_module_rather_than_reimplementing_it(self):
        """The AST, not a substring: the docstring names both functions too.

        Both reach the database through ``run_sync`` -- ``plan_backfill`` handed to it by
        name, ``apply_backfill`` called inside a lambda -- so the guard looks at what
        ``run_sync`` is given, which is the link that actually has to exist.
        """
        tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
        through_run_sync = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run_sync"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    through_run_sync.add(arg.id)
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                        through_run_sync.add(inner.func.id)
        assert {"plan_backfill", "apply_backfill"} <= through_run_sync, through_run_sync

        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "format_report" in called, "the CLI never prints the report"

    @pytest.mark.asyncio
    async def test_the_cli_dry_run_writes_nothing(self, async_session, tmp_path, capsys):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_cli", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        await async_session.commit()

        cli = _load_script()
        cli.AsyncSessionLocal = lambda: _fixed_session(async_session)
        assert await cli.run(apply=False) == 0

        row = await async_session.get(ExternalSAE, "sae_cli")
        await async_session.refresh(row)
        assert (row.hook_type, row.layer) == (None, None)
        assert "Dry run" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_the_cli_applies_when_asked(self, async_session, tmp_path, capsys):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_cli_apply", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res",
            hf_filepath="layer_20/width_16k/average_l0_71",
        ))
        await async_session.commit()

        cli = _load_script()
        cli.AsyncSessionLocal = lambda: _fixed_session(async_session)
        assert await cli.run(apply=True) == 0

        row = await async_session.get(ExternalSAE, "sae_cli_apply")
        await async_session.refresh(row)
        assert (row.hook_type, row.layer) == ("residual", 20)
        assert "Wrote 1 row(s)." in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_the_cli_reports_a_refusal_and_exits_nonzero(
        self, async_session, tmp_path, capsys
    ):
        directory = _gemma_scope_dir(tmp_path / "layer_20")
        async_session.add(_row(
            "sae_cli_busy", local_path=str(directory),
            hf_repo_id="google/gemma-scope-2b-pt-res", hf_filepath="layer_20/width_16k",
        ))
        await async_session.flush()
        async_session.add(_extraction("ext_cli", "sae_cli_busy", ExtractionStatus.QUEUED))
        await async_session.commit()

        cli = _load_script()
        cli.AsyncSessionLocal = lambda: _fixed_session(async_session)
        assert await cli.run(apply=True) == 1

        out = capsys.readouterr().out
        assert "REFUSED" in out and "ext_cli" in out
        row = await async_session.get(ExternalSAE, "sae_cli_busy")
        await async_session.refresh(row)
        assert (row.hook_type, row.layer) == (None, None)

    def test_apply_is_not_the_default(self):
        """A backfill that writes by default is one typo from an unreviewed repair."""
        tree = ast.parse(SCRIPT.read_text(), filename=str(SCRIPT))
        flags = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(isinstance(a, ast.Constant) and a.value == "--apply" for a in node.args)
        ]
        assert len(flags) == 1, "the --apply flag is gone or duplicated"
        action = {kw.arg: getattr(kw.value, "value", None) for kw in flags[0].keywords}
        assert action.get("action") == "store_true", action
