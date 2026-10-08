"""Every writer of external_saes records the SAE's hook, so the refusals can see it (review R2, B5).

Feature extraction, the Neuronpedia export and the local push refuse an MLP or attention SAE by
reading ``external_saes.hook_type`` (review R1-A, A5), and a NULL there reads as residual. Only
SAEs imported from a miStudio training recorded a hook: the HuggingFace download task read the
SAE's ``hook_point`` through ``get_sae_info`` and threw it away, and local import never looked. So
a downloaded Gemma Scope MLP SAE passed every refusal as residual and its features described the
wrong activations.

Each writer now records, in order of authority: the SAE's own cfg.json hook (``hook_name``,
``hook_point``, or either under SAELens 6's ``metadata``), exactly as written; else a Gemma Scope
set's -res/-mlp/-att name, in miStudio's hook vocabulary; else NULL. The loaders' invented
``blocks.L.hook_resid_post`` for cfg-less files is never consulted.

Only HuggingFace, the WebSocket emitter and (for the task) the sync session are stood in for; the
real task body, the real service, the real endpoints and the real refusal helper run.

MUTATION CONTROLS (one line broken at a time, this file run, bytes restored, sha256 and
``git diff`` verified clean; the table is in review_sae_remediation_R2_E_2026-09-15.md):
  H1 download task: ``sae.hook_type = recorded_hook.hook_type`` deleted
        -> the four task tests that record a hook (cfg-over-name included), the E2E MLP case
  H2 download task: resolve with location None (the cfg.json is never read)
        -> the two cfg task tests, the cfg-over-name task test, the E2E MLP case
  H3 initiate_download: ``hook_type=None`` instead of the resolved kind
        -> test_initiate_download_records_a_gemma_scope_kind
  H4 import_from_file: ``hook_type=None`` instead of the resolved hook
        -> the three recording local-import tests (the AST test stays green, as it must)
  H5 import_from_file: the ``hook_type`` keyword deleted -> the AST test
  H6 resolver: cfg.json and name precedence swapped -> test_the_config_outranks_the_set_name,
        test_the_task_takes_the_config_over_the_repository_name
  H7 hook_name_from_sae_config ignores ``metadata`` -> the SAELens 6 case
  H8 Gemma Scope ``att`` recorded raw as "att" -> the three att cases of the kind table, the
        refusal-reads-it test, the broken-cfg test, test_initiate_download_records_a_gemma_scope_kind
  H9 A5's training import (_import_single_sae): the ``hook_type`` keyword deleted -> the AST test
  H10 A5's training import: ``hook_type=None`` -> SURVIVED the first run (71 green across this
        file, test_multi_sae_import.py and test_non_residual_sae_refusals.py; nothing in tests/
        called import_from_training). test_a_training_import_records_each_saes_hook_and_the_mlp_one_is_refused
        was added; re-run as a negative control -> RED. (Its first draft failed on the unmutated
        tree -- a non-UUID dataset id -- so that first "red" was void and is not counted.)
"""

import ast
import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.config import settings
from src.models.external_sae import ExternalSAE
from src.services.sae_hook_support import non_residual_hook_reason
from src.services.sae_manager_service import (
    HOOK_SOURCE_CFG,
    HOOK_SOURCE_GEMMA_SCOPE_NAME,
    RecordedHook,
    SAEManagerService,
    gemma_scope_hook_kind,
    hook_name_from_sae_config,
    resolve_sae_hook,
)

PHRASE = "supports residual-stream SAEs only"
SRC = Path(__file__).resolve().parents[2] / "src"


# ── SAE directories as they arrive on disk ──────────────────────────────────


def _saelens_dir(root: Path, hook_name: Optional[str], key: str = "hook_name") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"d_in": 8, "d_sae": 16, "architecture": "standard", "model_name": "tiny", "hook_point_layer": 3}
    if hook_name is not None:
        cfg[key] = hook_name
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


def _cfgless_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


# ── the resolver ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (dict(hook_name="blocks.3.hook_mlp_out"), "blocks.3.hook_mlp_out"),
        (dict(hook_point="blocks.7.hook_attn_out"), "blocks.7.hook_attn_out"),
        (dict(metadata=dict(hook_name="blocks.3.hook_mlp_out")), "blocks.3.hook_mlp_out"),
        (dict(hook_name="", hook_point="blocks.2.hook_resid_pre"), "blocks.2.hook_resid_pre"),
        (dict(d_in=8), None),
        (dict(hook_name=7), None),
    ],
)
def test_the_config_hook_is_taken_exactly_as_written(cfg, expected):
    assert hook_name_from_sae_config(cfg) == expected


@pytest.mark.parametrize(
    "origin, expected",
    [
        ("google/gemma-scope-2b-pt-res", "residual"),
        ("google/gemma-scope-2b-pt-mlp", "mlp"),
        ("google/gemma-scope-9b-it-att", "attention"),
        ("google/gemma-scope-2b-pt-mlp-canonical", "mlp"),
        ("google/gemma-scope-2b-pt-att-canonical", "attention"),
        ("/data/imports/gemma-scope-27b-pt-att/layer_3/width_16k/average_l0_71", "attention"),
        # Review R3-B: was None (debt 2 of R2-E), which reads as residual. A transcoder is
        # recorded as ``transcoder``: the set type, not a guessed hook, and refused as MLP-side.
        ("google/gemma-scope-2b-pt-transcoders", "transcoder"),
        ("google/gemma-2-2b", None),
        ("someone/my-mlp-saes", None),
        ("layer_20/width_16k/average_l0_71", None),
        (None, None),
    ],
)
def test_a_gemma_scope_set_name_gives_its_kind(origin, expected):
    assert gemma_scope_hook_kind(origin) == expected


def test_what_is_recorded_for_mlp_and_attention_is_what_the_refusal_reads():
    """A recorded value the live helper does not recognise protects nothing."""
    for recorded in (
        gemma_scope_hook_kind("google/gemma-scope-2b-pt-mlp"),
        gemma_scope_hook_kind("google/gemma-scope-2b-pt-att"),
        "blocks.3.hook_mlp_out",
        "blocks.7.hook_attn_out",
    ):
        assert non_residual_hook_reason(recorded, "X") is not None, recorded
    assert non_residual_hook_reason(gemma_scope_hook_kind("google/gemma-scope-2b-pt-res"), "X") is None


def test_the_config_outranks_the_set_name(tmp_path):
    sae = _saelens_dir(tmp_path / "gemma-scope-2b-pt-res" / "layer_3", "blocks.3.hook_mlp_out")
    assert resolve_sae_hook(sae, "google/gemma-scope-2b-pt-res") == RecordedHook("blocks.3.hook_mlp_out", HOOK_SOURCE_CFG)


def test_a_cfgless_gemma_scope_set_records_its_kind_and_anything_else_records_nothing(tmp_path):
    gemma = _gemma_scope_dir(tmp_path / "a" / "layer_3" / "width_16k" / "average_l0_71")
    assert resolve_sae_hook(gemma, "google/gemma-scope-2b-pt-mlp") == RecordedHook("mlp", HOOK_SOURCE_GEMMA_SCOPE_NAME)
    # The same params.npz from a repository that does not say: the loaders would call it
    # hook_resid_post, and that is a default, not a record.
    assert resolve_sae_hook(gemma, "someone/saes", "layer_3/width_16k") == RecordedHook(None, None)


def test_a_file_reads_the_config_beside_it_and_an_unreadable_config_is_not_fatal(tmp_path):
    sae = _saelens_dir(tmp_path / "saelens", "blocks.5.hook_attn_out")
    assert resolve_sae_hook(sae / "sae_weights.safetensors").hook_type == "blocks.5.hook_attn_out"

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "cfg.json").write_text("{not json")
    assert resolve_sae_hook(broken, "google/gemma-scope-2b-pt-att") == RecordedHook("attention", HOOK_SOURCE_GEMMA_SCOPE_NAME)
    assert resolve_sae_hook(broken) == RecordedHook(None, None)


# ── writer 1: the HuggingFace download task ─────────────────────────────────


class _Query:
    def __init__(self, row):
        self.row = row

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.row


class _Session:
    """The sync session the task writes the row through; the row is a real ORM object."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    def query(self, model):
        assert model is ExternalSAE
        return _Query(self.row)

    def commit(self):
        self.commits += 1


def _run_download(monkeypatch, tmp_path, row, repo_id: str, filepath: str, lay_out: Callable[[Path], None]):
    """Run the real download task body; HuggingFace, the emitter and the session are stand-ins."""
    from src.services.huggingface_sae_service import HuggingFaceSAEService
    from src.workers import sae_tasks

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    session = _Session(row)

    @contextmanager
    def get_sync_db():
        yield session

    calls = []

    async def download_sae(repo_id, filepath, local_dir, **kwargs):
        calls.append((repo_id, filepath))
        target = Path(local_dir) / filepath
        lay_out(target)
        return {"local_path": str(target)}

    monkeypatch.setattr(sae_tasks, "get_sync_db", get_sync_db)
    monkeypatch.setattr(sae_tasks, "emit_sae_download_progress", lambda **kwargs: None)
    monkeypatch.setattr(HuggingFaceSAEService, "download_sae", staticmethod(download_sae))

    result = sae_tasks.download_sae_task.run(sae_id=row.id, repo_id=repo_id, filepath=filepath)
    assert result["status"] == "success" and calls == [(repo_id, filepath)]
    assert row.status == "ready" and session.commits > 0
    return row


def _pending_row(hook_type=None):
    return ExternalSAE(
        id="sae_dl", name="download", source="huggingface", status="pending",
        hook_type=hook_type, progress=0.0, sae_metadata={},
    )


def test_the_task_records_a_saelens_config_hook(monkeypatch, tmp_path):
    row = _run_download(
        monkeypatch, tmp_path, _pending_row(), "someone/saes", "blocks.3.hook_mlp_out",
        lambda target: _saelens_dir(target, "blocks.3.hook_mlp_out"),
    )
    assert row.hook_type == "blocks.3.hook_mlp_out"
    assert row.sae_metadata["hook_source"] == HOOK_SOURCE_CFG
    assert non_residual_hook_reason(row.hook_type, "Feature extraction") is not None


def test_the_task_records_a_legacy_hook_point(monkeypatch, tmp_path):
    row = _run_download(
        monkeypatch, tmp_path, _pending_row(), "someone/saes", "layer_7",
        lambda target: _saelens_dir(target, "blocks.7.hook_attn_out", key="hook_point"),
    )
    assert row.hook_type == "blocks.7.hook_attn_out"


def test_the_task_records_a_gemma_scope_mlp_set(monkeypatch, tmp_path):
    row = _run_download(
        monkeypatch, tmp_path, _pending_row(), "google/gemma-scope-2b-pt-mlp",
        "layer_3/width_16k/average_l0_71", _gemma_scope_dir,
    )
    assert row.hook_type == "mlp"
    assert row.sae_metadata["hook_source"] == HOOK_SOURCE_GEMMA_SCOPE_NAME
    assert non_residual_hook_reason(row.hook_type, "Feature extraction") is not None


def test_the_task_records_nothing_for_a_cfgless_file(monkeypatch, tmp_path):
    row = _run_download(
        monkeypatch, tmp_path, _pending_row(), "someone/saes", "layer_3/sae.safetensors", _cfgless_file,
    )
    assert row.hook_type is None
    assert row.sae_metadata["hook_source"] is None


def test_the_task_takes_the_config_over_the_repository_name(monkeypatch, tmp_path):
    """Also: the task's value replaces what initiate_download recorded from the name alone."""
    row = _run_download(
        monkeypatch, tmp_path, _pending_row(hook_type="residual"), "google/gemma-scope-2b-pt-res",
        "layer_3", lambda target: _saelens_dir(target, "blocks.3.hook_mlp_out"),
    )
    assert row.hook_type == "blocks.3.hook_mlp_out"


# ── writer 2: the PENDING row the download endpoint creates ─────────────────


@pytest.mark.asyncio
async def test_initiate_download_records_a_gemma_scope_kind(async_session, monkeypatch, tmp_path):
    from src.schemas.sae import SAEDownloadRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    att = await SAEManagerService.initiate_download(
        async_session, SAEDownloadRequest(repo_id="google/gemma-scope-2b-pt-att", filepath="layer_3/width_16k/average_l0_71"),
    )
    plain = await SAEManagerService.initiate_download(
        async_session, SAEDownloadRequest(repo_id="someone/saes", filepath="layer_3"),
    )
    assert (att.hook_type, att.sae_metadata["hook_source"]) == ("attention", HOOK_SOURCE_GEMMA_SCOPE_NAME)
    assert (plain.hook_type, plain.sae_metadata["hook_source"]) == (None, None)


# ── writer 3: local import ──────────────────────────────────────────────────


async def _import(async_session, monkeypatch, tmp_path, relative: str):
    from src.schemas.sae import SAEImportFromFileRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return await SAEManagerService.import_from_file(async_session, SAEImportFromFileRequest(file_path=relative, name="imported"))


@pytest.mark.asyncio
async def test_local_import_of_a_saelens_directory_records_its_config_hook(async_session, monkeypatch, tmp_path):
    _saelens_dir(tmp_path / "data" / "imports" / "saelens", "blocks.5.hook_attn_out")
    row = await _import(async_session, monkeypatch, tmp_path, "imports/saelens")
    assert (row.hook_type, row.sae_metadata["hook_source"]) == ("blocks.5.hook_attn_out", HOOK_SOURCE_CFG)


@pytest.mark.asyncio
async def test_local_import_of_a_file_reads_the_config_beside_it(async_session, monkeypatch, tmp_path):
    _saelens_dir(tmp_path / "data" / "imports" / "beside", "blocks.2.hook_mlp_out")
    row = await _import(async_session, monkeypatch, tmp_path, "imports/beside/sae_weights.safetensors")
    assert row.hook_type == "blocks.2.hook_mlp_out"


@pytest.mark.asyncio
async def test_local_import_of_a_gemma_scope_set_records_its_kind(async_session, monkeypatch, tmp_path):
    _gemma_scope_dir(tmp_path / "data" / "imports" / "gemma-scope-2b-pt-mlp" / "layer_3" / "width_16k" / "average_l0_71")
    row = await _import(async_session, monkeypatch, tmp_path, "imports/gemma-scope-2b-pt-mlp/layer_3/width_16k/average_l0_71")
    assert (row.hook_type, row.sae_metadata["hook_source"]) == ("mlp", HOOK_SOURCE_GEMMA_SCOPE_NAME)


@pytest.mark.asyncio
async def test_local_import_of_a_cfgless_file_records_nothing(async_session, monkeypatch, tmp_path):
    _cfgless_file(tmp_path / "data" / "imports" / "bare" / "sae.safetensors")
    row = await _import(async_session, monkeypatch, tmp_path, "imports/bare/sae.safetensors")
    assert row.hook_type is None and row.sae_metadata["hook_source"] is None


# ── end to end: a downloaded MLP SAE meets feature extraction's refusal ─────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_name, refused",
    [("blocks.3.hook_mlp_out", True), ("blocks.3.hook_resid_post", False)],
)
async def test_a_downloaded_sae_is_refused_by_feature_extraction_by_its_recorded_hook(
    client, async_session, monkeypatch, tmp_path, hook_name, refused,
):
    """POST /saes/download -> the real task -> POST /saes/{id}/extract-features."""
    from src.api.v1.endpoints import saes as endpoint
    from src.models.model import Model, ModelStatus, QuantizationFormat

    async_session.add(Model(
        id="m_dl", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    await async_session.commit()

    task = MagicMock()
    monkeypatch.setattr(endpoint, "download_sae_task", task)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    created = await client.post("/api/v1/saes/download", json=dict(
        repo_id="someone/saes", filepath="layer_3", model_id="m_dl",
    ))
    assert created.status_code == 200, created.text
    sae_id = created.json()["id"]
    assert task.delay.call_count == 1
    kwargs = task.delay.call_args.kwargs
    assert (kwargs["sae_id"], kwargs["repo_id"], kwargs["filepath"]) == (sae_id, "someone/saes", "layer_3")

    row = await async_session.get(ExternalSAE, sae_id)
    assert row.hook_type is None  # the repository name says nothing yet
    await asyncio.to_thread(
        _run_download, monkeypatch, tmp_path, row, "someone/saes", "layer_3",
        lambda target: _saelens_dir(target, hook_name),
    )
    await async_session.commit()

    # Review R3-B: the dataset id was "ds_none", not a UUID, so the residual case reached a 500 from
    # the dataset query and passed on "not 422". A well-formed id that does not exist makes the
    # residual case reach the NEXT check, which answers 400 with its own reason.
    missing_dataset = "00000000-0000-4000-8000-00000000abcd"
    response = await client.post(f"/api/v1/saes/{sae_id}/extract-features?dataset_id={missing_dataset}", json=dict())
    if refused:
        assert response.status_code == 422 and PHRASE in response.text, response.text
    else:
        assert response.status_code == 400 and f"Dataset {missing_dataset} not found" in response.text, response.text
        assert PHRASE not in response.text


# ── every construction passes hook_type ─────────────────────────────────────


def _external_sae_constructions():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name != "ExternalSAE":
                continue
            enclosing = parents.get(node)
            while enclosing is not None and not isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enclosing = parents.get(enclosing)
            found.append((str(path.relative_to(SRC)), node, enclosing.name if enclosing else None))
    return found


def test_every_external_sae_construction_in_src_passes_hook_type():
    found = _external_sae_constructions()
    writers = {(path, function) for path, _, function in found}
    # A scan that finds nothing asserts nothing: the three known writers must be among them.
    assert writers >= {
        ("services/sae_manager_service.py", "initiate_download"),
        ("services/sae_manager_service.py", "_import_single_sae"),
        ("services/sae_manager_service.py", "import_from_file"),
    }, writers
    unrecorded = [
        f"{path}:{node.lineno} ({function})"
        for path, node, function in found
        if any(keyword.arg is None for keyword in node.keywords)
        or "hook_type" not in {keyword.arg for keyword in node.keywords}
    ]
    assert not unrecorded, "ExternalSAE rows constructed without hook_type: " + ", ".join(unrecorded)


# ── writer 4 (A5): import from a miStudio training ─────────────────────────


@pytest.mark.asyncio
async def test_a_training_import_records_each_saes_hook_and_the_mlp_one_is_refused(async_session, monkeypatch, tmp_path):
    """A5's writer. No test anywhere called import_from_training, so its recorded hook was
    unpinned: H10 (``hook_type=None`` in _import_single_sae) left the whole suite green."""
    from src.models.model import Model, ModelStatus, QuantizationFormat
    from src.models.training import Training, TrainingStatus
    from src.schemas.sae import SAEImportFromTrainingRequest

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    async_session.add(Model(
        id="m_ti", name="tiny", architecture="llama", params_count=1_000,
        quantization=QuantizationFormat.FP16, status=ModelStatus.READY,
    ))
    await async_session.flush()
    async_session.add(Training(
        # datasets.id is a UUID column; the import looks the names up by it.
        id="train_ti", model_id="m_ti", dataset_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        dataset_ids=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
        status=TrainingStatus.COMPLETED.value, total_steps=10,
        hyperparameters=dict(latent_dim=16, hidden_dim=8),
    ))
    await async_session.commit()
    community = tmp_path / "data" / "trainings" / "train_ti" / "community_format"
    _saelens_dir(community / "layer_3_residual", "blocks.3.hook_resid_post")
    _saelens_dir(community / "layer_3_mlp", "blocks.3.hook_mlp_out")

    response = await SAEManagerService.import_from_training(
        async_session, SAEImportFromTrainingRequest(training_id="train_ti", import_all=True),
    )
    by_hook = {sae.hook_type: sae for sae in response.saes}
    assert response.imported_count == 2 and set(by_hook) == {"residual", "mlp"}, by_hook
    assert non_residual_hook_reason(by_hook["mlp"].hook_type, "Feature extraction") is not None
    assert non_residual_hook_reason(by_hook["residual"].hook_type, "Feature extraction") is None
