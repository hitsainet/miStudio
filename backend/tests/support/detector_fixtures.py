"""Real-file fixtures for the behaviour-detector reports (Feature 040 S2; FTID §8.2).

* Splits are tiny Hugging Face `datasets` saved under the test's data root and read through the REAL
  `open_split` — so a column snapshot is taken the way production takes it.
* Evaluation arrays are written by the writer's own naming (`evaluation_array_paths`) under
  `<data_dir>/probe_monitors/<run>`, so the reader's root check sees what production writes.
* Positions are NON-CONTIGUOUS on purpose (filtered rows leave gaps), so a join by array index instead of
  by position gives different figures (the FTID's fixture rule).

Keys, names and column names are random (`random_behaviour`); nothing names a behaviour.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def write_split(root: Path, columns: Dict[str, Sequence[Any]], name: Optional[str] = None) -> str:
    """Save a one-split `datasets.Dataset` under `root/datasets/<name>`; returns the raw_path."""
    from datasets import Dataset as HFDataset

    path = Path(root) / "datasets" / (name or f"ds_{uuid.uuid4().hex[:10]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    HFDataset.from_dict({k: list(v) for k, v in columns.items()}).save_to_disk(str(path))
    return str(path)


async def dataset_row(session, raw_path: str):
    from src.models.dataset import Dataset, DatasetStatus

    row = Dataset(id=uuid.uuid4(), name=f"corpus-{uuid.uuid4().hex[:6]}", source="local",
                  status=DatasetStatus.READY, raw_path=raw_path, extra_metadata={})
    session.add(row)
    await session.flush()
    return row


async def model_row(session, model_id: Optional[str] = None, repo_id: Optional[str] = None):
    from src.models.model import Model, ModelStatus

    row = Model(id=model_id or f"m_{uuid.uuid4().hex[:12]}", name="tiny", repo_id=repo_id,
                status=ModelStatus.READY, file_path="/nonexistent", architecture="LlamaForCausalLM",
                params_count=1)
    session.add(row)
    await session.flush()
    return row


async def view_row(session, dataset, *, role="eval", distribution=None, behaviour=None, name=None,
                   counts=None, sub_question=None, behaviour_version=None, dataset_version=None):
    from src.models.probe_monitor import ProbeMonitorDataset

    view = ProbeMonitorDataset(
        name=name or f"view-{uuid.uuid4().hex[:6]}", dataset_id=dataset.id, input_column="text",
        label_column="label", label_mapping={"1": "positive", "0": "negative"}, role=role,
        distribution=distribution, counts=counts or {}, behaviour=behaviour, sub_question=sub_question,
        behaviour_version=behaviour_version, dataset_version=dataset_version,
    )
    session.add(view)
    await session.flush()
    return view


async def declare(session, view, columns: Sequence[str]):
    """Declare and snapshot columns on a view through the REAL snapshot writer."""
    from src.services import view_columns

    limits = view_columns.ColumnLimits(max_values=10_000, max_columns=32)
    entries = await view_columns.snapshot_columns(session, view, list(columns), limits)
    view.group_columns = list(columns)
    view.column_snapshots = entries
    await session.flush()
    return view


@dataclass
class Arrays:
    scores: np.ndarray
    positions: np.ndarray
    n_scored: np.ndarray
    labels: np.ndarray


async def evaluation_row(session, probe, run, view, arrays: Arrays, *, status="completed", auroc=0.8,
                         with_positions=True):
    from src.models.probe_monitor import ProbeMonitorEvaluation
    from src.services.probe_monitor_run import artifact_dir_for, evaluation_array_paths

    evaluation = ProbeMonitorEvaluation(
        id=f"pme_{uuid.uuid4().hex[:12]}", probe_id=probe.id, dataset_id=view.id, status=status,
        metrics={"auroc": auroc}, n_positive=int((arrays.labels == 1).sum()),
        n_negative=int((arrays.labels == 0).sum()),
    )
    directory = artifact_dir_for(run.id)
    directory.mkdir(parents=True, exist_ok=True)
    paths = evaluation_array_paths(directory, probe.id, view.id, evaluation.id)
    for column, values in (("scores_path", arrays.scores.astype(np.float32)),
                           ("positions_path", arrays.positions.astype(np.int64)),
                           ("n_scored_path", arrays.n_scored.astype(np.int32)),
                           ("labels_path", arrays.labels.astype(np.int8))):
        if column == "positions_path" and not with_positions:
            continue
        np.save(paths[column], values)
        setattr(evaluation, column, str(paths[column]))
    session.add(evaluation)
    await session.flush()
    return evaluation


async def run_and_probe(session, model, train_view, *, eval_views=(), calibration_view=None, behaviour=None,
                        scope="all", threshold=0.5, length_bands=None, window_decisions=None,
                        target_fpr=0.01, n_negatives=None, selection_rule=None, selected=True,
                        calibration_scores: Optional[Sequence[float]] = None, history=None):
    from src.models.probe_monitor import ProbeMonitor, ProbeMonitorRun
    from src.services.probe_monitor_run import artifact_dir_for

    config = {"scope": scope}
    if selection_rule:
        config["selection_rule"] = selection_rule
    run = ProbeMonitorRun(model_id=model.id, train_dataset_id=train_view.id,
                          eval_dataset_ids=[v.id for v in eval_views],
                          calibration_dataset_id=calibration_view.id if calibration_view else None,
                          config=config, status="completed", environment={}, behaviour=behaviour)
    session.add(run)
    await session.flush()
    probe = ProbeMonitor(run_id=run.id, layer=1, rule="mean", rule_params={}, variant="dense",
                         val_metrics={}, selected=selected, behaviour=behaviour, threshold=threshold,
                         target_fpr=target_fpr, length_bands=length_bands, window_decisions=window_decisions,
                         threshold_source="calibration_set" if calibration_view else "validation_negatives",
                         calibration_dataset_id=calibration_view.id if calibration_view else None,
                         calibration_n_negatives=n_negatives, calibration_history=history)
    session.add(probe)
    await session.flush()
    if calibration_scores is not None:
        directory = artifact_dir_for(run.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{probe.id}__calibration_negatives.npy"
        np.save(path, np.asarray(calibration_scores, dtype=np.float32))
        probe.calibration_scores_path = str(path)
        await session.flush()
    return run, probe
