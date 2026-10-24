/**
 * Archive helpers for probe dataset views (operator decision 2026-10-21).
 *
 * Nothing here decides what can be deleted: the server's purge plan is the authority
 * (`backend/src/services/probe_dataset_lifecycle.py`), so a blocker the server knows is a
 * blocker the screen shows.
 */
import type { ProbeDataset, ProbeDatasetPurgePlan } from '../../types/probeMonitor';

export function isArchived(view: Pick<ProbeDataset, 'archived_at'>): boolean {
  return Boolean(view.archived_at);
}

/** The views a NEW run, evaluation or judge baseline may use: archived ones are out of use
 * (the server refuses them with 409, so offering them would only produce that refusal). */
export function activeViews(views: ProbeDataset[]): ProbeDataset[] {
  return views.filter((view) => !isArchived(view));
}

/**
 * Why a view cannot be deleted, in one line; `null` when it can.
 *
 * `afterArchive`: the one-step "Archive and delete" path from a visible row. Not being archived
 * is then not a blocker — the dialog archives first — but every other blocker still is.
 */
export function blockedReason(
  plan: ProbeDatasetPurgePlan | undefined,
  afterArchive = false
): string | null {
  if (plan === undefined) return null;
  if (!plan.found) return plan.blockers[0]?.message ?? 'This view no longer exists.';
  const [first, ...rest] = plan.blockers;
  if (first !== undefined) {
    return rest.length === 0 ? first.message : `${first.message} (+${rest.length} more)`;
  }
  if (plan.not_archived !== null && !afterArchive) return plan.not_archived.message;
  return null;
}

/** True when the plan allows deletion (now, or right after archiving when `afterArchive`). */
export function deletable(plan: ProbeDatasetPurgePlan | undefined, afterArchive = false): boolean {
  if (plan === undefined || !plan.found) return false;
  return blockedReason(plan, afterArchive) === null;
}
