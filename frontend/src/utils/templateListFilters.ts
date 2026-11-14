/**
 * Template list filters -> list-route query params.
 *
 * The training and extraction template lists are paginated server-side (50 per
 * page), so their filters are sent to the list routes rather than applied to
 * the page in hand: a browser-side filter would filter one page and report it
 * as the whole set.
 */

import type { TrainingTemplateListParams } from '../types/trainingTemplate';
import type { ExtractionHookType, ExtractionTemplateListParams } from '../types/extractionTemplate';

export interface TrainingTemplateFilters {
  search?: string;
  /** SAE architecture (encoder_type); undefined = any */
  encoder_type?: string;
  /** Inclusive first day, YYYY-MM-DD (the viewer's local calendar day) */
  created_from?: string;
  /** Inclusive last day, YYYY-MM-DD (the viewer's local calendar day) */
  created_to?: string;
}

export type ExtractionSortKey = 'newest' | 'oldest' | 'name_asc' | 'name_desc' | 'updated';

export const EXTRACTION_SORTS: Record<ExtractionSortKey, {
  label: string;
  sort_by: NonNullable<ExtractionTemplateListParams['sort_by']>;
  order: 'asc' | 'desc';
}> = {
  newest: { label: 'Newest first', sort_by: 'created_at', order: 'desc' },
  oldest: { label: 'Oldest first', sort_by: 'created_at', order: 'asc' },
  name_asc: { label: 'Name A–Z', sort_by: 'name', order: 'asc' },
  name_desc: { label: 'Name Z–A', sort_by: 'name', order: 'desc' },
  updated: { label: 'Recently updated', sort_by: 'updated_at', order: 'desc' },
};

export interface ExtractionTemplateFilters {
  search?: string;
  hook_type?: ExtractionHookType;
  sort?: ExtractionSortKey;
}

/** Local midnight of a YYYY-MM-DD day, shifted by `addDays`, as an ISO instant. */
function localDayStartIso(day: string, addDays = 0): string | undefined {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  if (!m) return undefined;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]) + addDays);
  return d.toISOString();
}

export function trainingFiltersToParams(f: TrainingTemplateFilters): TrainingTemplateListParams {
  const search = f.search?.trim();
  return {
    search: search || undefined,
    encoder_type: f.encoder_type || undefined,
    created_after: f.created_from ? localDayStartIso(f.created_from) : undefined,
    // "To" is inclusive of its whole day: the route's bound is exclusive, so
    // send the start of the NEXT day.
    created_before: f.created_to ? localDayStartIso(f.created_to, 1) : undefined,
  };
}

export function extractionFiltersToParams(f: ExtractionTemplateFilters): ExtractionTemplateListParams {
  const search = f.search?.trim();
  const sort = EXTRACTION_SORTS[f.sort ?? 'newest'];
  return {
    search: search || undefined,
    hook_type: f.hook_type,
    sort_by: sort.sort_by,
    order: sort.order,
  };
}

export function trainingFiltersActive(f: TrainingTemplateFilters): boolean {
  return !!(f.search?.trim() || f.encoder_type || f.created_from || f.created_to);
}

export function extractionFiltersActive(f: ExtractionTemplateFilters): boolean {
  return !!(f.search?.trim() || f.hook_type);
}
