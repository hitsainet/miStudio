"""
The replication report (BR-001, Phase 5.5).

PUBLISHED WHETHER FAVOURABLE OR NOT. That is the requirement, and it is the
whole point: a replication that only gets recorded when it succeeds is not a
replication, it is a press release. `ReplicationReport` has no "publish if"
switch and no draft state — building one records it.

WHAT IT MEASURES, per lens (logit / J-lens / tuned where available):
  * normalised pass@k AUC for intermediate-concept recovery, over the six
    evaluation distributions
  * ablation KL divergence
  * lens-coordinate swap success rate

WHAT IT MUST NEVER MEASURE. Next-token agreement is not a quality metric here
either (BR-004). The J-lens is DELIBERATELY worse on it than the logit lens
through most of the network, so a replication that scored it would report the
J-lens as a failure and be wrong. An AST guard over this module enforces that,
the same way it does for the gate.

PROVENANCE IS PART OF THE RESULT. The reference implementation is vendored at a
recorded commit and is explicitly unmaintained, so "which commit" is not
bookkeeping — nobody upstream will fix a discrepancy, and a figure without its
commit cannot be compared to anything later.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPLICATION_FILENAME = "replication-report.json"

# The six evaluation distributions the reference harness ships. Named here so a
# partial run is VISIBLE as partial rather than reported as a complete result
# over whatever happened to run.
EVALUATION_SETS = (
    "multihop",
    "arithmetic",
    "modulation",
    "entity_tracking",
    "list_recall",
    "counterfactual",
)


@dataclass
class LensResult:
    """One lens's figures. Absent means NOT MEASURED, never zero."""

    lens: str
    pass_at_k_auc: Dict[str, Optional[float]] = field(default_factory=dict)
    ablation_kl: Optional[float] = None
    swap_success_rate: Optional[float] = None

    @property
    def coverage(self) -> List[str]:
        """Which evaluation sets actually produced a figure."""
        return sorted(k for k, v in self.pass_at_k_auc.items() if v is not None)

    @property
    def is_complete(self) -> bool:
        return set(self.coverage) == set(EVALUATION_SETS)


@dataclass
class ReplicationReport:
    """A recorded replication attempt. Existence IS publication (BR-001)."""

    model_id: str
    reference_commit: str
    results: List[LensResult]
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.reference_commit.strip():
            raise ValueError(
                "a replication report requires the reference implementation's "
                "commit. The upstream repository is unmaintained, so a figure "
                "without its commit cannot be compared to anything later."
            )

    @property
    def is_complete(self) -> bool:
        """Every lens measured on every evaluation set.

        Reported rather than enforced: an incomplete replication is a real
        result and must still be published. What is forbidden is presenting one
        AS complete.
        """
        return bool(self.results) and all(r.is_complete for r in self.results)

    @property
    def missing(self) -> Dict[str, List[str]]:
        return {
            r.lens: sorted(set(EVALUATION_SETS) - set(r.coverage))
            for r in self.results
            if not r.is_complete
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "reference_commit": self.reference_commit,
            "complete": self.is_complete,
            # Named explicitly so a partial run cannot be read as a full one.
            "missing": self.missing,
            "evaluation_sets": list(EVALUATION_SETS),
            "notes": self.notes,
            "results": [
                {
                    "lens": r.lens,
                    # Nulls preserved: "not measured" is not "scored zero".
                    "pass_at_k_auc": {k: r.pass_at_k_auc.get(k) for k in EVALUATION_SETS},
                    "ablation_kl": r.ablation_kl,
                    "swap_success_rate": r.swap_success_rate,
                    "coverage": r.coverage,
                }
                for r in self.results
            ],
        }


def save_replication_report(directory: Path, report: ReplicationReport) -> Path:
    """Write the report. There is no favourable/unfavourable branch here.

    Deliberately so: the only way to make BR-001's "published whether
    favourable or not" structural is for the writer to have no opinion about
    the contents.
    """
    path = Path(directory) / REPLICATION_FILENAME
    path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info(
        "Replication report for %s recorded (complete=%s)",
        report.model_id,
        report.is_complete,
    )
    return path


def load_replication_report(directory: Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / REPLICATION_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001 - reported
        logger.warning("Replication report at %s is unreadable: %s", path, exc)
        return None
