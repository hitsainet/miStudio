/**
 * Duplicating an extraction template copies the WHOLE template.
 *
 * `handleDuplicate` built its copy from a hand-maintained field list that had
 * drifted from `ExtractionTemplateCreate`: `micro_batch_size`,
 * `context_prefix_tokens` and `context_suffix_tokens` were declared there and
 * omitted here, so a duplicate silently reset the GPU micro-batch and both
 * context windows to the backend's defaults while calling itself a copy. Two of
 * the three are REQUIRED on `ExtractionTemplate`, so the copy did not merely lose
 * an optional extra — it changed what the extraction would do.
 *
 * Same shape as the template-save defect review round 2 lane R2-F fixed for
 * training templates: a list of keys maintained by hand beside a type that grew.
 *
 * The expected body is written out literally rather than spread from the fixture.
 * A spread would agree with the copy by construction and pass however many fields
 * `handleDuplicate` dropped.
 *
 * MUTATION CONTROL (applied alone, source restored by byte copy, sha256 verified):
 *   D1 the three fields removed from handleDuplicate again -> RED here
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ExtractionTemplatesPanel } from './ExtractionTemplatesPanel';
import { useExtractionTemplatesStore } from '../../stores/extractionTemplatesStore';
import * as api from '../../api/extractionTemplates';
import type { ExtractionTemplate } from '../../types/extractionTemplate';

vi.mock('../../api/extractionTemplates');

const TEMPLATE: ExtractionTemplate = {
  id: 'et_0001',
  name: 'LFM2.5-1.2B · L11-13 residual',
  description: 'the stored description',
  layer_indices: [11, 12, 13],
  hook_types: ['residual'],
  max_samples: 10000,
  batch_size: 8,
  // The three that were dropped. Deliberately NOT the backend defaults (which are
  // batch_size for micro_batch_size, and 25/25 for the context windows), so a copy
  // that loses them is distinguishable from one that keeps them.
  micro_batch_size: 2,
  context_prefix_tokens: 40,
  context_suffix_tokens: 12,
  top_k_examples: 100,
  is_favorite: true,
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
} as ExtractionTemplate;

beforeEach(() => {
  vi.mocked(api.getExtractionTemplates).mockReset().mockResolvedValue({
    data: [TEMPLATE as any],
    pagination: { page: 1, limit: 50, total: 1, total_pages: 1, has_next: false, has_prev: false },
  } as any);
  vi.mocked(api.getFavoriteExtractionTemplates).mockReset().mockResolvedValue({ data: [] } as any);
  vi.mocked(api.createExtractionTemplate)
    .mockReset()
    .mockResolvedValue({ ...TEMPLATE, id: 'et_copy', name: `${TEMPLATE.name} (Copy)` } as any);
  vi.mocked(api.updateExtractionTemplate).mockReset().mockResolvedValue(TEMPLATE as any);
  useExtractionTemplatesStore.setState({ templates: [], favorites: [], loading: false, error: null } as any);
});

describe('duplicating an extraction template', () => {
  it('carries the micro-batch and both context windows into the copy', async () => {
    render(<ExtractionTemplatesPanel />);

    fireEvent.click(await screen.findByTitle(/Duplicate/i));
    fireEvent.click(screen.getByRole('button', { name: /Create Template|Save Template|Update Template/ }));

    await waitFor(() => expect(api.createExtractionTemplate).toHaveBeenCalledTimes(1));
    const [body] = vi.mocked(api.createExtractionTemplate).mock.calls[0];

    expect(body.name).toBe(`${TEMPLATE.name} (Copy)`);
    // The three the hand-written list had dropped.
    expect(body.micro_batch_size).toBe(2);
    expect(body.context_prefix_tokens).toBe(40);
    expect(body.context_suffix_tokens).toBe(12);
    // ...and the rest of the template still arrives, so this is a copy.
    expect(body.layer_indices).toEqual([11, 12, 13]);
    expect(body.hook_types).toEqual(['residual']);
    expect(body.max_samples).toBe(10000);
    expect(body.batch_size).toBe(8);
    expect(body.top_k_examples).toBe(100);
    // A copy starts unfavourited: the original's star is not a property of the copy.
    expect(body.is_favorite).toBe(false);
  });
});
