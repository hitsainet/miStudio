/**
 * A description the user clears is actually cleared.
 *
 * The form sent `description.trim() || undefined`. `undefined` is dropped by
 * JSON.stringify, so the key never reached the backend — and the update service
 * uses `model_dump(exclude_unset=True)`, where a MISSING key means "leave
 * unchanged". Emptying the field therefore did nothing: the old text came back on
 * the next load, with no error and nothing to suggest the edit had been ignored.
 *
 * `null` is the value that clears it (the column is `Text, nullable=True`), and is
 * the convention `TrainingTemplateForm` already established for this exact defect
 * — review round 2 lane R2-F, control F5: "`description.trim() || undefined`: a
 * cleared description comes back". Extraction templates were left carrying the bug
 * that had already been diagnosed and fixed next door.
 *
 * MUTATION CONTROL (applied alone, source restored by byte copy, sha256 verified):
 *   D2 `|| null` reverted to `|| undefined` -> RED here
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ExtractionTemplateForm } from './ExtractionTemplateForm';
import type { ExtractionTemplate } from '../../types/extractionTemplate';

const STORED: ExtractionTemplate = {
  id: 'et_0001',
  name: 'L11-13 residual',
  description: 'the old description',
  layer_indices: [11, 12, 13],
  hook_types: ['residual'],
  max_samples: 10000,
  batch_size: 8,
  micro_batch_size: 2,
  context_prefix_tokens: 40,
  context_suffix_tokens: 12,
  top_k_examples: 100,
  is_favorite: false,
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
} as ExtractionTemplate;

function renderForm() {
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(<ExtractionTemplateForm template={STORED} onSubmit={onSubmit} onCancel={vi.fn()} />);
  return onSubmit;
}

async function save(onSubmit: ReturnType<typeof vi.fn>) {
  fireEvent.click(screen.getByRole('button', { name: /Update Template|Create Template|Save/ }));
  await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  return onSubmit.mock.calls[0][0];
}

describe('clearing an extraction template description', () => {
  it('sends null, not an absent key, so the stored text is replaced', async () => {
    const onSubmit = renderForm();

    fireEvent.change(screen.getByLabelText(/Description/i), { target: { value: '' } });
    const payload = await save(onSubmit);

    // `toBeNull` and not `toBeUndefined`: an absent key is exactly the defect.
    expect(payload.description).toBeNull();
    expect('description' in payload).toBe(true);
  });

  it('still sends an edited description unchanged', async () => {
    const onSubmit = renderForm();

    fireEvent.change(screen.getByLabelText(/Description/i), { target: { value: '  a new one  ' } });
    const payload = await save(onSubmit);

    expect(payload.description).toBe('a new one');
  });
});
