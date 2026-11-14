/**
 * The six identity fields of a probe summary (Feature 039 FR-16), picked out for the export section.
 * The server computes every one of them (`identity_fields`); nothing here derives or defaults a value.
 */
import type { ProbeMonitorSummary } from '../../types/probeMonitor';
import type { ProbeDefinitionIdentity } from './ProbeExportSection';

export function identityOf(probe: ProbeMonitorSummary): ProbeDefinitionIdentity {
  return {
    detector_digest: probe.detector_digest ?? null,
    decision_digest: probe.decision_digest ?? null,
    decision_revision: probe.decision_revision ?? null,
    identity_current: probe.identity_current ?? null,
    identity_stale_reason: probe.identity_stale_reason ?? null,
    identity_source: probe.identity_source ?? null,
  };
}
