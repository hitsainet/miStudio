"""
Checkpoint retention policy.

Training writes a checkpoint every ``checkpoint_interval`` steps and nothing
ever removes them, so a long run accumulates dozens of multi-GB snapshots that
are dead weight once the run has produced its Community Standard export.

The policy is deliberately conservative and DISABLED by default: pruning is
irreversible, so an operator opts in (see the ``checkpoint_prune_*`` settings)
and the first runs are dry-run reports.

GROUPING: a multi-layer training writes ONE ROW PER (step, layer, hook) sharing
a single ``checkpoint_{step}/`` directory. Retention therefore selects whole
STEPS — counting rows would keep an arbitrary subset of a step's layers and
produce a half-deleted, unloadable checkpoint.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set

from ..models.checkpoint import Checkpoint
from ..models.training import Training, TrainingStatus

logger = logging.getLogger(__name__)

# Settings keys (app_settings table, category "general").
SETTING_ENABLED = "checkpoint_prune_enabled"
SETTING_DRY_RUN = "checkpoint_prune_dry_run"
SETTING_KEEP_LAST = "checkpoint_prune_keep_last"
SETTING_KEEP_BEST = "checkpoint_prune_keep_best"
SETTING_MIN_AGE_HOURS = "checkpoint_prune_min_age_hours"

# Conservative defaults, used whenever the setting row is absent or unparseable.
DEFAULT_ENABLED = False
DEFAULT_DRY_RUN = True
DEFAULT_KEEP_LAST = 2
DEFAULT_KEEP_BEST = True
DEFAULT_MIN_AGE_HOURS = 24

# A training in any of these states may still write to or resume from its
# checkpoints, so its directory is off limits regardless of policy.
ACTIVE_TRAINING_STATUSES = frozenset({
    TrainingStatus.PENDING.value,
    TrainingStatus.INITIALIZING.value,
    TrainingStatus.RUNNING.value,
    TrainingStatus.PAUSED.value,
})


@dataclass
class RetentionPolicy:
    """Resolved retention configuration."""

    enabled: bool = DEFAULT_ENABLED
    dry_run: bool = DEFAULT_DRY_RUN
    keep_last: int = DEFAULT_KEEP_LAST
    keep_best: bool = DEFAULT_KEEP_BEST
    min_age_hours: int = DEFAULT_MIN_AGE_HOURS


@dataclass
class PrunePlan:
    """What a prune WOULD do for one training. Produced without deleting."""

    training_id: str
    prunable_steps: List[int] = field(default_factory=list)
    kept_steps: List[int] = field(default_factory=list)
    checkpoint_ids: List[str] = field(default_factory=list)
    estimated_bytes: int = 0
    skipped_reason: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        return self.skipped_reason is None and bool(self.checkpoint_ids)


_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _parse_bool(raw: str, default: bool) -> bool:
    """Parse a settings flag, falling back to ``default`` on anything unknown.

    Failing to the default (rather than to False) matters: for
    ``checkpoint_prune_dry_run`` a False means DELETE FILES, so a typo like
    "t"/"enabled"/"Y" written through the settings API must not silently arm
    deletion.
    """
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    logger.warning(
        "Unrecognised boolean setting value %r; falling back to %s", raw, default
    )
    return default


def _parse_int(
    raw: str, default: int, minimum: int, maximum: Optional[int] = None
) -> int:
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer setting value %r; falling back to %s", raw, default
        )
        return default
    if value < minimum:
        logger.warning(
            "Setting value %s below minimum %s; clamping", value, minimum
        )
        return minimum
    if maximum is not None and value > maximum:
        # The UI caps these, but the settings API accepts any string — clamp at
        # the parse site so an out-of-range write cannot weaken the policy.
        logger.warning(
            "Setting value %s above maximum %s; clamping", value, maximum
        )
        return maximum
    return value


def policy_from_values(values: Dict[str, str]) -> RetentionPolicy:
    """Build a policy from raw settings values.

    The single place the keys and parsers are applied, so the sync worker and
    the async API endpoint cannot drift: adding a setting here reaches both.
    """
    return RetentionPolicy(
        enabled=_parse_bool(values.get(SETTING_ENABLED), DEFAULT_ENABLED),
        dry_run=_parse_bool(values.get(SETTING_DRY_RUN), DEFAULT_DRY_RUN),
        keep_last=_parse_int(
            values.get(SETTING_KEEP_LAST), DEFAULT_KEEP_LAST, minimum=1, maximum=50
        ),
        keep_best=_parse_bool(values.get(SETTING_KEEP_BEST), DEFAULT_KEEP_BEST),
        min_age_hours=_parse_int(
            values.get(SETTING_MIN_AGE_HOURS), DEFAULT_MIN_AGE_HOURS,
            minimum=0, maximum=8760,
        ),
    )


SETTING_KEYS = [
    SETTING_ENABLED, SETTING_DRY_RUN, SETTING_KEEP_LAST,
    SETTING_KEEP_BEST, SETTING_MIN_AGE_HOURS,
]


def load_policy(db) -> RetentionPolicy:
    """Read the retention policy from app_settings using a SYNC session.

    ``AppSettingService`` is async and unusable from a Celery worker, so this
    follows the established in-worker idiom of querying ``AppSetting`` directly.
    Absent rows fall back to the conservative defaults above (the settings table
    has no seeding mechanism — a key simply does not exist until written).
    """
    from ..models.app_setting import AppSetting

    rows = db.query(AppSetting).filter(AppSetting.key.in_(SETTING_KEYS)).all()
    return policy_from_values({r.key: r.value for r in rows})


def select_prunable_steps(
    checkpoints: Sequence[Checkpoint],
    policy: RetentionPolicy,
    now: Optional[datetime] = None,
) -> tuple[List[int], List[int]]:
    """Split a training's checkpoint STEPS into (prunable, kept).

    Guards applied, in order:
      1. the newest step is always kept (resume target)
      2. the ``keep_last`` newest steps are kept
      3. any step containing an ``is_best`` row is kept when ``keep_best``
      4. steps younger than ``min_age_hours`` are kept

    Returns:
        (prunable_steps, kept_steps), both ascending.
    """
    if not checkpoints:
        return [], []

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=policy.min_age_hours)

    steps: Set[int] = {c.step for c in checkpoints}
    ordered = sorted(steps)

    # keep_last >= 1 is enforced by _parse_int, so this also covers guard 1.
    keep: Set[int] = set(ordered[-policy.keep_last:]) if policy.keep_last else {ordered[-1]}

    if policy.keep_best:
        keep |= {c.step for c in checkpoints if c.is_best}

    # Youngest-row-wins: a step is only old enough to prune when EVERY row in it
    # is older than the cutoff.
    newest_created: Dict[int, datetime] = {}
    for c in checkpoints:
        created = c.created_at
        if created is None:
            # Unknown age -> treat as brand new and keep it.
            newest_created[c.step] = now
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        prev = newest_created.get(c.step)
        if prev is None or created > prev:
            newest_created[c.step] = created

    for step, created in newest_created.items():
        if created > cutoff:
            keep.add(step)

    prunable = [s for s in ordered if s not in keep]
    kept = [s for s in ordered if s in keep]
    return prunable, kept


def plan_from_checkpoints(
    training_id: str,
    training_status: str,
    checkpoints: Sequence[Checkpoint],
    policy: RetentionPolicy,
    now: Optional[datetime] = None,
) -> PrunePlan:
    """Pure planning core — no database access.

    Kept session-free so the sync Celery worker and the async API endpoint share
    one implementation (and so the policy is testable without a database).
    """
    plan = PrunePlan(training_id=training_id)

    if training_status in ACTIVE_TRAINING_STATUSES:
        plan.skipped_reason = f"training is {training_status}"
        return plan

    if not checkpoints:
        plan.skipped_reason = "no checkpoints"
        return plan

    prunable, kept = select_prunable_steps(checkpoints, policy, now=now)
    plan.prunable_steps = prunable
    plan.kept_steps = kept

    if not prunable:
        return plan

    prunable_set = set(prunable)
    total = 0
    for c in checkpoints:
        if c.step not in prunable_set:
            continue
        # Defence in depth: even if step selection were wrong, never queue a
        # best checkpoint for deletion while keep_best is on.
        if policy.keep_best and c.is_best:
            continue
        plan.checkpoint_ids.append(c.id)
        total += _checkpoint_size(c)

    plan.estimated_bytes = total
    return plan


def build_plan(
    db,
    training: Training,
    policy: RetentionPolicy,
    now: Optional[datetime] = None,
) -> PrunePlan:
    """Sync-session wrapper around :func:`plan_from_checkpoints`."""
    checkpoints: List[Checkpoint] = (
        db.query(Checkpoint).filter(Checkpoint.training_id == training.id).all()
    )
    return plan_from_checkpoints(
        training_id=training.id,
        training_status=training.status,
        checkpoints=checkpoints,
        policy=policy,
        now=now,
    )


def _checkpoint_size(checkpoint: Checkpoint) -> int:
    """Best-effort size for reporting.

    ``file_size_bytes`` is NULL for every row written by the training loop, so
    fall back to stat()ing the file. Returns 0 when neither is available —
    reporting must never be the thing that breaks a prune.
    """
    if checkpoint.file_size_bytes:
        return int(checkpoint.file_size_bytes)
    try:
        from pathlib import Path

        p = Path(checkpoint.storage_path)
        if p.is_file():
            return p.stat().st_size
    except OSError:
        pass
    return 0


def iter_prunable_trainings(db) -> Iterable[Training]:
    """Trainings eligible for pruning (terminal states only)."""
    return (
        db.query(Training)
        .filter(~Training.status.in_(sorted(ACTIVE_TRAINING_STATUSES)))
        .all()
    )
