"""Passing everyday-text, nonsense-text and catch-rate checks for fixtures that test OTHER parts of the probe
ladder (ENH-006 D82, D84). `add_rung2_checks_pass` adds all three.

Rung 2 and 3 need a completed evaluation on an `everyday_text` view of at least 1,000 rows with few false
alarms at the shipped bar. A fixture whose subject is the out-of-distribution rule or the judge rule adds
this so its rung is decided by the rule under test; the everyday-text rule itself is tested in
`test_everyday_text_rung.py`. The count is RECORDED at the probe's bar (no score files), which
`everyday_set_of` accepts only while the bar is unchanged.
"""
from __future__ import annotations

from typing import Any


def add_everyday_pass(db: Any, probe: Any, dataset_id: Any, *, rows: int = 1000, directory: Any = None) -> Any:
    """Give `probe` a shipped bar if it has none, and a 0-in-`rows` everyday-text evaluation. Sync session.

    With `directory`, the per-row scores are written too (every row far below any bar), so the check
    survives a re-cut: `everyday_set_of` re-reads them at whatever bar is in force.
    """
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation

    if probe.threshold is None:
        probe.threshold = 1.0
    if probe.target_fpr is None:
        probe.target_fpr = 0.01
    view = ProbeMonitorDataset(name="everyday", dataset_id=dataset_id, input_column="text",
                               label_column="text", label_mapping={"*": "negative"},
                               role="everyday_text", counts={"negative": rows})
    db.add(view)
    db.flush()
    paths = {}
    if directory is not None:
        import uuid
        from pathlib import Path

        import numpy as np

        stem = Path(directory) / f"everyday_{uuid.uuid4().hex[:8]}"
        np.save(f"{stem}.npy", np.full(rows, -1e9, dtype=np.float32))
        np.save(f"{stem}__labels.npy", np.zeros(rows, dtype=np.int8))
        paths = {"scores_path": f"{stem}.npy", "labels_path": f"{stem}__labels.npy"}
    db.add(ProbeMonitorEvaluation(
        probe_id=probe.id, dataset_id=view.id, status="refused", n_positive=0, n_negative=rows,
        metrics={"scored": False, "everyday_text": {"threshold": float(probe.threshold), "rows": rows,
                                                    "false_alarms": 0}}, **paths))
    db.flush()
    return view


def add_nonsense_pass(db: Any, probe: Any, dataset_id: Any, *, rows: int = 1000, directory: Any = None) -> Any:
    """`add_everyday_pass` for the NONSENSE-TEXT check (ENH-006 D84): a 0-in-`rows` `nonsense_text` evaluation."""
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation

    if probe.threshold is None:
        probe.threshold = 1.0
    if probe.target_fpr is None:
        probe.target_fpr = 0.01
    view = ProbeMonitorDataset(name="nonsense", dataset_id=dataset_id, input_column="text",
                               label_column="kind", label_mapping={"*": "negative"},
                               role="nonsense_text", counts={"negative": rows})
    db.add(view)
    db.flush()
    paths = {}
    if directory is not None:
        import uuid
        from pathlib import Path

        import numpy as np

        stem = Path(directory) / f"nonsense_{uuid.uuid4().hex[:8]}"
        np.save(f"{stem}.npy", np.full(rows, -1e9, dtype=np.float32))
        np.save(f"{stem}__labels.npy", np.zeros(rows, dtype=np.int8))
        paths = {"scores_path": f"{stem}.npy", "labels_path": f"{stem}__labels.npy"}
    db.add(ProbeMonitorEvaluation(
        probe_id=probe.id, dataset_id=view.id, status="refused", n_positive=0, n_negative=rows,
        metrics={"scored": False, "nonsense_text": {"threshold": float(probe.threshold), "rows": rows,
                                                     "false_alarms": 0}}, **paths))
    db.flush()
    return view


def add_catch_rate_pass(db: Any, probe: Any, *, positives: int = 100, directory: Any = None) -> int:
    """Give every out-of-distribution evaluation of `probe` that has no score files a full catch rate (D84).

    Without `directory` the count is RECORDED at the probe's bar (`metrics.catch_rate`), which
    `recall_set_of` accepts only while the bar is unchanged; with it, per-row scores far ABOVE any bar are
    written, so the catch rate survives a re-cut. Returns how many evaluations it touched. Sync session.
    """
    from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorEvaluation

    if probe.threshold is None:
        probe.threshold = 1.0
    db.flush()
    touched = 0
    rows = (db.query(ProbeMonitorEvaluation, ProbeMonitorDataset)
            .join(ProbeMonitorDataset, ProbeMonitorDataset.id == ProbeMonitorEvaluation.dataset_id)
            .filter(ProbeMonitorEvaluation.probe_id == probe.id,
                    ProbeMonitorDataset.distribution == "out_of_distribution").all())
    for evaluation, _view in rows:
        if evaluation.scores_path:
            continue
        if directory is not None:
            import uuid
            from pathlib import Path

            import numpy as np

            stem = Path(directory) / f"ood_{uuid.uuid4().hex[:8]}"
            np.save(f"{stem}.npy", np.full(positives, 1e9, dtype=np.float32))
            np.save(f"{stem}__labels.npy", np.ones(positives, dtype=np.int8))
            evaluation.scores_path, evaluation.labels_path = f"{stem}.npy", f"{stem}__labels.npy"
        else:
            evaluation.metrics = {**(evaluation.metrics or {}), "catch_rate": {
                "threshold": float(probe.threshold), "positives": positives, "caught": positives}}
        touched += 1
    db.flush()
    return touched


def add_rung2_checks_pass(db: Any, probe: Any, dataset_id: Any, *, directory: Any = None) -> None:
    """Every rung-2 condition OTHER than the AUROC one (ENH-006 D82, D84): everyday text, nonsense text and the
    catch rate on the probe's existing out-of-distribution evaluations. For fixtures whose subject is another
    rule; each of these rules is tested on its own in `test_everyday_text_rung.py` / `test_rung2_tighten.py`."""
    add_everyday_pass(db, probe, dataset_id, directory=directory)
    add_nonsense_pass(db, probe, dataset_id, directory=directory)
    add_catch_rate_pass(db, probe, directory=directory)
