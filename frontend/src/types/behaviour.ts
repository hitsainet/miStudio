/**
 * Behaviour catalogue shapes (Feature 037). Mirrors `services/behaviour_catalogue.py` and
 * `services/behaviour_checklist.py` exactly — nothing here is derived on the client.
 */

// ── Feature 040 FR-1: the detector data a version carries. Each field is ABSENT when the version does
// not declare it (a version written before 040 has none); `[]` is "declared empty". ──
export type DetectorForm = 'rules_or_blend' | 'blend_alone' | 'best_single' | 'rules_alone' | 'named_probe';
export const DETECTOR_FORMS: DetectorForm[] = ['rules_or_blend', 'blend_alone', 'best_single', 'rules_alone', 'named_probe'];
export type NamedSubset = 'all' | 'rule_invisible_positives' | 'rule_visible_positives';
export const NAMED_SUBSETS: NamedSubset[] = ['all', 'rule_invisible_positives', 'rule_visible_positives'];

export interface SubQuestion {
  key: string;
  question: string;
  positive_meaning: string;
}

export interface SuccessCriterion {
  id: string;
  detector: DetectorForm;
  subset: NamedSubset | { column: string; value: string | number | boolean };
  held_out_test?: string;
  target_fpr?: number;
  min_recall: number;
  probe_id?: string;
}

export interface ModelRef {
  model_id: string;
}

/** How a version's rule set reads. `unvalidated` until the rule engine is available (S3). */
export interface RuleSetInfo {
  digest: string;
  digest_scheme: string;
  status: string;
  reason?: string;
  source?: 'referenced' | 'inline';
  source_ref?: { name: string; version: number; digest: string } | null;
  behaviour_key?: string;
}

export interface DetectorFields {
  sub_questions?: SubQuestion[];
  success_criteria?: SuccessCriterion[];
  held_out_columns?: string[];
  models?: ModelRef[];
  calibration_floor?: number;
}

export interface BehaviourDefinition extends DetectorFields {
  version: number;
  name: string;
  description: string | null;
  positive_meaning: string;
  negative_meaning: string | null;
  target_fpr: number | null;
  created_by: string;
  created_at: string | null;
  rule_set?: RuleSetInfo;
}

export interface BehaviourCounts {
  prompts: number;
  runs: number;
  probes: number;
  views: number;
}

export type BehaviourStatus = 'draft' | 'validated' | 'monitoring' | 'unknown' | 'archived';

export interface Behaviour {
  key: string;
  catalogued: true;
  current_version: number;
  archived: boolean;
  archived_at: string | null;
  archived_by: string | null;
  created_by: string;
  created_at: string | null;
  updated_at: string | null;
  definition: BehaviourDefinition;
  counts: BehaviourCounts;
  status?: BehaviourStatus;
}

export interface UncataloguedKey extends BehaviourCounts {
  key: string;
  catalogued: false;
}

export interface BehaviourList {
  behaviours: Behaviour[];
  uncatalogued: UncataloguedKey[];
  /** Feature 040 FR-6: the list is paged; follow `next_cursor` until it is null. */
  next_cursor?: string | null;
  total?: number | null;
  total_exact?: boolean;
  uncatalogued_total?: number;
  limit?: number;
}

export type ProbeRole = 'primary' | 'candidate' | 'retired';

export interface AdoptedProbe {
  id: string;
  run_id: string;
  layer: number;
  rule: string;
  variant: string;
  rung: number;
  behaviour_label: string | null;
  definition_sha256: string | null;
  // Feature 039: the identity of the built definition (the summary's six fields).
  detector_digest?: string | null;
  decision_digest?: string | null;
  decision_revision?: number | null;
  identity_current?: boolean | null;
  identity_stale_reason?: string | null;
  identity_source?: string | null;
  role: ProbeRole;
  training_view_confirmed: boolean;
  created_at: string | null;
}

export interface AdoptedView {
  id: string;
  name: string;
  role: string;
  use: string;
  linked_by: 'key' | 'link' | 'confirmation';
  behaviour: string | null;
  positive_meaning: string | null;
  positive_labels_confirmed: Record<string, unknown> | null;
  confirmed_for_this_behaviour: boolean;
  archived: boolean;
  dataset_version: { version_id: string | null; version_number: number | null; dataset_name: string | null; split: string | null } | null;
}

export interface BehaviourDetail extends Behaviour {
  adopted: {
    prompts: { id: string; key: string; title: string; locked: boolean; archived: boolean }[];
    runs: { id: string; status: string; model_id: string; train_dataset_id: string; created_at: string | null }[];
    probes: AdoptedProbe[];
    views: AdoptedView[];
  };
}

/** `stale` (Feature 039 FR-19, D-T7): the primary's definition exists but states an old decision. */
/** `partly_done` (Feature 041 FR-5): some members of the family are done, the step names the rest. */
export type StepState = 'done' | 'not_done' | 'unknown' | 'stale' | 'partly_done';

export interface ChecklistStep {
  id: string;
  label: string;
  state: StepState;
  reason: string;
  evidence: string[];
  link: { tab?: string; app?: string };
  /** The tool calls that clear the step (a stale export: `build_probe_definition`). */
  next?: Array<{ tool: string; arguments: Record<string, unknown>; why: string; needs?: string[] }>;
  /** 039: the exported step's identity of the built definition, and where it came from. */
  identity?: Record<string, unknown> | null;
  identity_source?: string | null;
}

export interface BehaviourChecklist {
  key: string;
  status: BehaviourStatus;
  steps: ChecklistStep[];
}

export interface BehaviourHistoryEntry {
  id: number;
  subject_kind: string;
  subject_id: string | null;
  field: string;
  old_value: unknown;
  new_value: unknown;
  actor: string;
  approval_id: string | null;
  reason: string | null;
  created_at: string | null;
}

export interface BehaviourDefinitionBody {
  name: string;
  positive_meaning: string;
  description?: string | null;
  negative_meaning?: string | null;
  target_fpr?: number | null;
  sub_questions?: SubQuestion[] | null;
  success_criteria?: SuccessCriterion[] | null;
  held_out_columns?: string[] | null;
  models?: ModelRef[] | null;
  calibration_floor?: number | null;
}

/** A change against a base version (FR-3): every omitted field carries over; `null` removes it. */
export type BehaviourVersionChange = Partial<Omit<BehaviourDefinitionBody, 'name' | 'positive_meaning'>> & {
  base_version: number;
  name?: string;
  positive_meaning?: string;
};

/** `field_diff` of the server: one entry per changed field. */
export interface DefinitionDiffEntry {
  path: string;
  before: unknown;
  after: unknown;
}

export interface BehaviourWriteResult extends BehaviourDetail {
  warnings: Array<{ code: string; field?: string; column?: string; message: string; next: ChecklistStep['next'] }>;
  diff?: DefinitionDiffEntry[];
  base_version?: number;
}
