import type { CircuitEdge } from '../types/circuits';

/*
 * THE CLASSIFIER'S TOKEN AND LABEL SIGNALS, WITH "NOT MEASURED" SAID OUT LOUD.
 *
 * Live 2026-10-20 (tracker 3.4 A1/A2): edges between unlabelled features with
 * no top tokens recorded `token_overlap: 0.0`, a measured-looking zero. The
 * classifier now records `null` with `token_method: "absent"`; labels already
 * carried `label_method: "absent"`. Either way the row must not print a number
 * nobody measured. Old definitions (no `token_method`) print what they hold.
 */
export function edgeSignalParts(edge: CircuitEdge): string[] {
  const s = edge.type_signals as
    | { classified?: boolean; token_overlap?: number | null; token_method?: string;
        label_similarity?: number | null; label_method?: string }
    | null
    | undefined;
  if (!s || s.classified === false) return [];
  const parts: string[] = [];
  if ('token_overlap' in s) {
    parts.push(
      s.token_overlap == null || s.token_method === 'absent'
        ? 'tokens not measured'
        : `tokens ${s.token_overlap.toFixed(2)}`,
    );
  }
  if ('label_similarity' in s) {
    parts.push(
      s.label_similarity == null || s.label_method === 'absent'
        ? 'labels not measured'
        : `labels ${s.label_similarity.toFixed(2)}`,
    );
  }
  return parts;
}
