/**
 * J-space claims vocabulary — the UI mirror of backend/src/schemas/jspace_claims.py.
 *
 * These strings are PINNED to the backend by
 * backend/tests/unit/test_jspace_claims_discipline.py::test_ts_mirror_in_sync.
 * Edit BOTH files together or that test fails the build.
 *
 * They are mirrored rather than fetched because they must render before any
 * request completes — the caveat has to be on screen next to an empty readout,
 * which is exactly when a user is most likely to over-read a negative result.
 *
 * One definition per statement. Two copies drift, and the drifted copy is
 * always the one on the surface nobody re-read.
 */

/** BR-020. Both mechanisms named: either alone reads as a hedge. */
export const ABSENCE_CAVEAT =
  'Absence of a signal is not evidence that the computation did not occur. ' +
  'Sufficiently automatic or well-practiced computation proceeds without ' +
  'engaging the workspace, and a concept with no single-token name may not ' +
  'surface even when it is represented.';

/** BR-020's second half: honest about one negative, still implying total coverage. */
export const NO_COVERAGE_CLAIM =
  'This is not a comprehensive account of what the model is doing. Workspace ' +
  'evidence covers what the lens can name, not everything the model computes.';

/** BR-011. Bounds what any readout can contain at all. */
export const READOUT_LIMITS =
  'Readouts are limited to concepts with single-token names, and a readout ' +
  'that resists interpretation is not a null result.';

/** BR-019. What a rung-0 surface says instead of an intervention claim. */
export const READOUT_NOT_CAUSAL =
  'A concept appearing in a readout is not a causal claim: it was present, ' +
  'which is not the same as having been used.';
