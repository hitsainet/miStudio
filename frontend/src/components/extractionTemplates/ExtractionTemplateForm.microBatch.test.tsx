/**
 * OSD-15 — a micro-batch size can now be SET, not only carried.
 *
 * The field was deliberately "carried, not shown": an earlier fix stopped every
 * save silently dropping the key (a template tuned down to fit a card came back
 * untuned after any edit), but left the form with state and no setter, so a tuned
 * template could only ever be created through the API.
 *
 * Clearing the input must restore `undefined`, which the backend reads as "use
 * batch_size" — that is what keeps a template that never set one unaffected.
 *
 * MUTATION CONTROLS (each applied alone, source restored by byte copy):
 *   the input's onChange removed          -> RED (the value never changes)
 *   '' mapped to 0 instead of undefined   -> RED (0 is not "use batch_size")
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ExtractionTemplateForm } from './ExtractionTemplateForm';
import type { ExtractionTemplate } from '../../types/extractionTemplate';

const STORED: ExtractionTemplate = {
  id: 'et_0002',
  name: 'L11-13 residual',
  description: 'tuned to fit the 3090',
  layer_indices: [11, 12, 13],
  hook_types: ['residual'],
  max_samples: 10000,
  batch_size: 8,
  micro_batch_size: 2,
  context_prefix_tokens: 25,
  context_suffix_tokens: 25,
  top_k_examples: 25,
  is_favorite: false,
  created_at: '2026-09-25T00:00:00Z',
  updated_at: '2026-09-25T00:00:00Z',
} as ExtractionTemplate;

function renderForm(template?: ExtractionTemplate) {
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(
    <ExtractionTemplateForm template={template} onSubmit={onSubmit} onCancel={vi.fn()} />
  );
  return onSubmit;
}

async function save(onSubmit: ReturnType<typeof vi.fn>) {
  fireEvent.click(
    screen.getByRole('button', { name: /Update Template|Create Template|Save/ })
  );
  await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
  return onSubmit.mock.calls[0][0];
}

function openAdvanced() {
  // The control lives with the context-window settings behind the disclosure.
  const toggles = screen.getAllByRole('button');
  const disclosure = toggles.find((b) => /context/i.test(b.textContent || ''));
  if (disclosure) fireEvent.click(disclosure);
}

describe('the extraction template micro-batch control', () => {
  it('is present and shows the stored value', () => {
    renderForm(STORED);
    openAdvanced();
    const input = screen.getByLabelText(/Micro-Batch Size/i) as HTMLInputElement;
    expect(input.value).toBe('2');
  });

  it('sends a value the operator typed, so a tuned template can be CREATED', async () => {
    const onSubmit = renderForm();
    // A new template needs its required fields before the form will submit.
    fireEvent.change(screen.getByLabelText(/Template Name/i), { target: { value: 'new' } });
    fireEvent.change(screen.getByLabelText(/Layer Indices/i), { target: { value: '11, 12' } });
    openAdvanced();
    fireEvent.change(screen.getByLabelText(/Micro-Batch Size/i), {
      target: { value: '4' },
    });
    const payload = await save(onSubmit);
    expect(payload.micro_batch_size).toBe(4);
  });

  it('sends undefined when cleared, which the backend reads as "use batch_size"', async () => {
    const onSubmit = renderForm(STORED);
    openAdvanced();
    fireEvent.change(screen.getByLabelText(/Micro-Batch Size/i), {
      target: { value: '' },
    });
    const payload = await save(onSubmit);
    expect(payload.micro_batch_size).toBeUndefined();
    expect(payload.micro_batch_size).not.toBe(0);
  });

  it('still carries a stored value through an untouched save', async () => {
    const onSubmit = renderForm(STORED);
    const payload = await save(onSubmit);
    expect(payload.micro_batch_size).toBe(2);
  });

  it('leaves a template that never set one alone', async () => {
    const onSubmit = renderForm();
    fireEvent.change(screen.getByLabelText(/Template Name/i), { target: { value: 'new' } });
    fireEvent.change(screen.getByLabelText(/Layer Indices/i), { target: { value: '11, 12' } });
    const payload = await save(onSubmit);
    expect(payload.micro_batch_size).toBeUndefined();
  });
});

/**
 * OSD-48 — a stepped number input silently blocks the values this estate uses.
 *
 * Found while writing the micro-batch tests above: the form would not submit at
 * all with `top_k_examples: 25`. HTML5 validates a number against `min` + `step`,
 * and a non-conforming value makes the browser refuse to submit WITHOUT firing
 * React's onSubmit and without rendering any message. `min=10 step=10` accepted
 * only 10, 20, 30 …, so a template holding 25 — the top_k every extraction on
 * this estate uses — could not be saved through the form. `Evaluation Samples`
 * carried the same trap at `min=100 step=100`.
 *
 * MUTATION CONTROLS: restoring step={10} on Top-K -> RED; restoring step={100} on
 * Evaluation Samples -> RED.
 */
describe('numeric inputs accept the values the backend accepts', () => {
  it('saves a template whose top_k is 25, the value extractions actually use', async () => {
    const onSubmit = renderForm({ ...STORED, top_k_examples: 25 } as ExtractionTemplate);
    const payload = await save(onSubmit);
    expect(payload.top_k_examples).toBe(25);
  });

  it('saves an evaluation-sample count that is not a multiple of 100', async () => {
    const onSubmit = renderForm({ ...STORED, max_samples: 10050 } as ExtractionTemplate);
    const payload = await save(onSubmit);
    expect(payload.max_samples).toBe(10050);
  });

  it('a typed top_k of 25 is accepted too', async () => {
    const onSubmit = renderForm(STORED);
    fireEvent.change(screen.getByLabelText(/Top-K Examples/i), { target: { value: '25' } });
    const payload = await save(onSubmit);
    expect(payload.top_k_examples).toBe(25);
  });
});
