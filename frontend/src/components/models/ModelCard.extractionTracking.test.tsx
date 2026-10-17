/**
 * The card that shows the progress bar must be the thing subscribed to it.
 *
 * Reported 2026-08-26: starting an extraction left the Models list reading
 * "Starting Extraction — Extraction job queued, waiting for worker..." and it
 * never advanced. A browser refresh jumped straight to "Extracting
 * Activations, 552/10000".
 *
 * `useModelExtractionProgress` was mounted ONLY inside the extraction modal,
 * and only while `extractionStarted` was true:
 *
 *     useModelExtractionProgress(extractionStarted ? model.id : undefined)
 *
 * Closing the modal unmounted the hook, which unsubscribed from
 * `models/{id}/extraction`. The card kept rendering `model.extraction_status`
 * from a store nothing was updating any more. Refresh worked only because a
 * remount re-fetched the state once.
 *
 * Same shape as the tokenization tracking bug fixed the day before, in a
 * different component -- which is why this is asserted as WIRING, not as a
 * rendered string.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { ModelStatus } from '../../types/model';

const subscribed: string[] = [];
const listeners: string[] = [];

vi.mock('../../hooks/useModelProgress', async () => {
  const actual = await vi.importActual<typeof import('../../hooks/useModelProgress')>(
    '../../hooks/useModelProgress'
  );
  return {
    ...actual,
    useModelExtractionProgress: (modelId?: string) => {
      if (modelId) subscribed.push(modelId);
      listeners.push('extraction:progress');
    },
  };
});

const model = {
  id: 'm_b55c6926',
  name: 'gemma-4-12B-it',
  repo_id: 'google/gemma-4-12B-it',
  status: ModelStatus.READY,
  quantization: 'q4',
  params_count: 12_000_000_000,
  memory_required_bytes: 3_600_000_000,
} as never;

describe('ModelCard extraction tracking', () => {
  beforeEach(() => {
    subscribed.length = 0;
    listeners.length = 0;
  });

  it('subscribes for its own model as soon as it renders', () => {
    render(
      <ModelCard
        model={model}
        onClick={vi.fn()}
        onExtract={vi.fn()}
        onViewExtractions={vi.fn()}
        onDelete={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(subscribed).toContain('m_b55c6926');
  });

  it('does not wait for an extraction to already be in progress', () => {
    // The chicken-and-egg that caused the bug: subscribing only once progress
    // exists means the first update is the one you miss.
    const idle = { ...(model as object), extraction_status: undefined } as never;

    render(
      <ModelCard
        model={idle}
        onClick={vi.fn()}
        onExtract={vi.fn()}
        onViewExtractions={vi.fn()}
        onDelete={vi.fn()}
        onCancel={vi.fn()}
      />
    );

    expect(subscribed).toContain('m_b55c6926');
  });
});
