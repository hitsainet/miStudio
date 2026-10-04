/**
 * Provenance and the interpretability caveat (BR-007, BR-011, BR-020).
 *
 * The logit lens involves NO ARTIFACT, and the strip says so explicitly rather
 * than rendering blank — a blank strip reads as "provenance unknown", which is
 * a different and worse claim than "there is nothing to have provenance over".
 *
 * The caveat is not decoration. All three sentences are FPRD §3.7 requirements,
 * and the last one — absence is not evidence of absence — is the one a reader
 * is most likely to get wrong on their own.
 */

import { Database, Info } from 'lucide-react';
import { ABSENCE_CAVEAT, READOUT_LIMITS } from '../../config/jspaceClaims';
import type { ReadoutProvenance } from '../../types/jlens';

interface ProvenanceStripProps {
  provenance: ReadoutProvenance | null;
  bandsAvailable: boolean;
}

export function ProvenanceStrip({ provenance, bandsAvailable }: ProvenanceStripProps) {
  return (
    <footer className="mt-4 rounded-lg border border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-slate-500 dark:text-slate-500">
        <span className="flex items-center gap-1 text-slate-600 dark:text-slate-400">
          <Database className="h-3 w-3" /> provenance
        </span>
        {provenance == null ? (
          <span>no readout yet</span>
        ) : provenance.artifact_id == null ? (
          <span data-testid="jlens-no-artifact">
            logit lens · no artifact involved · readout computed from the model's
            own unembedding and final norm
          </span>
        ) : (
          <>
            <span>{provenance.artifact_id}</span>
            {provenance.target_layer && <span>target={provenance.target_layer}</span>}
            {provenance.attention_gradients && (
              <span>qk={provenance.attention_gradients}</span>
            )}
            {provenance.target_position_scope && (
              <span>positions={provenance.target_position_scope}</span>
            )}
            {provenance.aggregation && <span>agg={provenance.aggregation}</span>}
            {provenance.corpus && <span>corpus={provenance.corpus}</span>}
            {provenance.n_prompts != null && <span>n_prompts={provenance.n_prompts}</span>}
            {provenance.seq_len != null && <span>seq_len={provenance.seq_len}</span>}
            {provenance.dtype && <span>dtype={provenance.dtype}</span>}
          </>
        )}
        {provenance != null && !bandsAvailable && (
          <span>bands=none (no report for this model)</span>
        )}
      </div>
      <div className="mt-1 flex items-start gap-1.5 text-[10px] leading-snug text-slate-500 dark:text-slate-600">
        <Info className="mt-px h-3 w-3 shrink-0" />
        <span>
          {/* Imported, never restated. A hardcoded sentence here is a copy the
              backend's sync test cannot see, and a drifted caveat is worse
              than a missing one: the surface still looks like it is warning
              the user while saying something weaker than the requirement. */}
          {READOUT_LIMITS} {ABSENCE_CAVEAT}
        </span>
      </div>
    </footer>
  );
}
