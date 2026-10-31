/**
 * The precision a model ran at when an artifact was built, as the backend describes it
 * (`backend/src/services/artifact_dtype.py`).
 *
 * `source` is what makes this honest: "recorded" is a fact the artifact wrote down; "inferred"
 * is a deduction (every miStudio load path ran float16 before 2026-10-03, and recorded nothing);
 * "unknown" is neither. The database is never written with an inferred value.
 */
export type PrecisionSource = 'recorded' | 'inferred' | 'unknown';

export interface PrecisionLabel {
  value: string | null;
  source: PrecisionSource;
  note: string | null;
}
