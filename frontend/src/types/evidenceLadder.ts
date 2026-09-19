/**
 * Evidence ladder — the UI mirror of backend/src/schemas/evidence_ladder.py
 * (IDL-35). Literal values are pinned to the backend by
 * backend/tests/unit/test_evidence_ladder.py::test_ts_mirror_in_sync — edit
 * BOTH files together or that test fails the build.
 *
 * Language strings come from the SERVER (`rung_language` fields in API/MCP
 * responses) — this module deliberately exports no user-facing causal
 * language of its own.
 */

export enum EvidenceRung {
  MINED = 0,
  ATTRIBUTION_SUPPORTED = 1,
  CAUSALLY_VALIDATED = 2,
  FAITHFULNESS_TESTED = 3,
}

export interface EdgeEvidence {
  rung: EvidenceRung;
  tested_and_failed: EvidenceRung[];
}

// NOTE deliberately NO client-side rung computation here: the server computes
// and serves rungs + language (IDL-35). A client recomputation invited drift
// (review R1 finding #13) and was removed.
