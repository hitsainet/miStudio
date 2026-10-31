/**
 * The shared approvals vocabulary (ENH-008 §2.1, §2.2, §2.5), mapped onto miStudio's tools and
 * statuses. Every string the operator reads about a request's kind or outcome comes from here.
 */
import type { ApprovalRequest } from '../../types/featureGroups';

export const STEERING_TOOLS = new Set(['steer_compare', 'steer_sweep', 'steer_combined']);
/** Approving WRITES a durable record (an operator approval per code hash); never consumed. */
export const RECORD_TOOLS = new Set(['code_rule_approve']);
export const PROBE_RUN_TOOL = 'start_probe_training_run';
export const CODE_RULE_TOOL = 'code_rule_approve';

/** Plain-words label per tool. Old `millm_*` rows (before miLLM decided its own) stay readable in History. */
export const ACTION_LABELS: Record<string, string> = {
  steer_compare: 'Compare steering on several features',
  steer_sweep: "Sweep a feature's steering strength",
  steer_combined: 'Steer several features together',
  start_probe_training_run: 'Train probes on a graphics card',
  evaluate_probe_monitor: 'Evaluate a probe on a graphics card',
  start_probe_judge_run: 'Judge probe datasets with miLLM',
  start_probe_spot_check: 'Spot-check a behaviour with miLLM',
  recalibrate_probe_monitor: "Re-score a probe's calibration on a graphics card",
  code_rule_approve: 'Run operator code in the isolated runner',
  millm_load_model: 'Load a model in miLLM',
  millm_request_probe_limits: "Change a model's probe limits in miLLM",
  millm_measure_probe_capacity: "Measure a model's probe capacity in miLLM",
  millm_import_rule_set: 'Import a rule set with operator code into miLLM',
};

export function actionLabel(tool: string): string {
  return ACTION_LABELS[tool] ?? tool;
}

/** "Approve {action, lower case}" (§2.1 row 8). */
export function approveLabel(tool: string): string {
  const label = actionLabel(tool);
  return `Approve ${label.charAt(0).toLowerCase()}${label.slice(1)}`;
}

/** "asked by agent mcp" / "asked by an agent" / "asked by the operator" (§2.1 row 2). */
export function askedBy(requestedBy: string | null | undefined): string {
  if (!requestedBy) return 'asked by an agent';
  if (requestedBy === 'operator') return 'asked by the operator';
  return `asked by agent ${requestedBy}`;
}

/** The §2.2 outcome label for one of miStudio's rows. */
export function outcomeLabel(r: ApprovalRequest): string {
  switch (r.status) {
    case 'pending':
      return 'Waiting';
    case 'consumed':
      return 'Approved and used';
    case 'denied':
      return 'Rejected';
    case 'expired':
      return 'Expired';
    case 'approved':
      if (STEERING_TOOLS.has(r.tool_name)) return 'Approved and run';
      if (RECORD_TOOLS.has(r.tool_name)) return 'Approved (recorded)';
      return 'Approved, waiting for the agent';
    default:
      return String(r.status);
  }
}

/** True when nobody decided: the request lapsed while pending (or was drained at switch-over). */
export function expiredUndecided(r: ApprovalRequest): boolean {
  if (r.status !== 'expired') return false;
  if (r.decided_by === 'system') return true;
  if (r.decided_by === 'operator') return false;
  return !(r.reason ?? '').startsWith('expired: approved');
}

/** "Decided by" (§2.2): "operator", or "nobody (expired)" for an expiry. */
export function decidedBy(r: ApprovalRequest): string {
  if (r.status === 'pending') return '—';
  if (expiredUndecided(r)) return 'nobody (expired)';
  return 'operator';
}

/** "Result" (§2.2): what the decision produced. */
export function resultText(r: ApprovalRequest): string {
  if (r.steering_task_id) return `Steering task ${r.steering_task_id}`;
  if (r.status === 'consumed') return 'used by the agent';
  if (r.status === 'expired') {
    return expiredUndecided(r) ? 'expired without a decision' : 'approved, then expired unused';
  }
  // ENH-006 D81: each request of an "Approve all" batch is its own row; the shared batch id ties them together.
  if (r.batch_id) return `approved together, batch ${r.batch_id}`;
  return '—';
}
