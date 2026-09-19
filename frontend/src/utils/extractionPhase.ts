/**
 * What an activation extraction card says it is doing.
 *
 * The backend keeps `status` at 'extracting' through three steps: the GPU pass,
 * merging the per-batch files, and computing statistics. `phase` says which,
 * and it is stored on the row. These helpers read whatever the store holds:
 * the value `checkActiveExtraction` restored after a reload, or the latest
 * WebSocket message. Before the phase was stored, a reload during the merge or
 * the statistics fell back to "Extracting 20000/20000".
 */

/** Label for the card's progress header, or null when nothing is running. */
export function extractionStageLabel(
  status?: string | null,
  phase?: string | null,
): string | null {
  switch (status) {
    case 'starting':
      return 'Starting Extraction';
    case 'loading':
      return 'Loading Model';
    case 'extracting':
      if (phase === 'merging') return 'Merging…';
      if (phase === 'statistics') return 'Computing statistics…';
      return 'Extracting Activations';
    case 'saving':
      return 'Saving Results';
    default:
      return null;
  }
}

/** The fields of an `/extractions/active` response the message is built from. */
export interface ActiveExtractionSummary {
  status: string;
  phase?: string | null;
  samples_processed?: number | null;
  max_samples?: number | null;
}

/** The card's detail line after a reload, before any WebSocket message. */
export function activeExtractionMessage(active: ActiveExtractionSummary): string {
  const samples = `${active.samples_processed ?? 0}/${active.max_samples ?? 0}`;
  if (active.status === 'extracting' && active.phase === 'merging') {
    return `Merging batch files (${samples} samples extracted)`;
  }
  if (active.status === 'extracting' && active.phase === 'statistics') {
    return `Computing statistics (${samples} samples extracted)`;
  }
  return `${active.status} (${samples} samples)`;
}
