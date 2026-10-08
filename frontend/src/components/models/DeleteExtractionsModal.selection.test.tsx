/**
 * The confirmation must delete what you picked, not everything.
 *
 * Reported 2026-08-25: selecting one extraction in Extraction History and
 * pressing "Delete Selected (1)" opened a confirmation reading "Select All
 * Deletable (3 of 3) - 3 selected", so the user had to un-tick the two they
 * had never chosen. The modal seeded its own state with every deletable row
 * and the caller's selection was never passed in at all.
 *
 * The dangerous direction is silent: a confirmation that pre-ticks rows the
 * user did not choose turns one careless click into 73 GB of deleted work.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { DeleteExtractionsModal } from './DeleteExtractionsModal';
import { getModelExtractions } from '../../api/models';

vi.mock('../../api/models', () => ({
  getModelExtractions: vi.fn(),
}));

const model = { id: 'm_b55c6926', name: 'gemma-4-12B-it' } as never;

const EXTRACTIONS = [
  { extraction_id: 'ext_a', status: 'failed', num_samples: 16, can_delete: true },
  { extraction_id: 'ext_b', status: 'completed', num_samples: 10000, can_delete: true },
  { extraction_id: 'ext_c', status: 'completed', num_samples: 10000, can_delete: true },
];

function boxes() {
  return screen.getAllByRole('checkbox') as HTMLInputElement[];
}

/** Checked ids, ignoring the "select all" control which has no id of its own. */
async function checkedCount() {
  await waitFor(() => expect(boxes().length).toBeGreaterThan(EXTRACTIONS.length - 1));
  // the header select-all is the first checkbox; the rows follow
  return boxes().slice(1).filter((b) => b.checked).length;
}

describe('DeleteExtractionsModal initial selection', () => {
  beforeEach(() => {
    vi.mocked(getModelExtractions).mockResolvedValue({
      extractions: EXTRACTIONS,
    } as never);
  });

  it('carries over only the extraction the user picked', async () => {
    render(
      <DeleteExtractionsModal
        model={model}
        onClose={vi.fn()}
        onDelete={vi.fn()}
        initialSelectedIds={new Set(['ext_a'])}
      />
    );

    await waitFor(async () => expect(await checkedCount()).toBe(1));
    expect(await screen.findByText(/1 selected/i)).toBeInTheDocument();
  });

  it('carries over a two-row selection intact', async () => {
    render(
      <DeleteExtractionsModal
        model={model}
        onClose={vi.fn()}
        onDelete={vi.fn()}
        initialSelectedIds={new Set(['ext_a', 'ext_c'])}
      />
    );

    await waitFor(async () => expect(await checkedCount()).toBe(2));
  });

  it('ignores ids that cannot be deleted', async () => {
    vi.mocked(getModelExtractions).mockResolvedValue({
      extractions: [
        EXTRACTIONS[0],
        { ...EXTRACTIONS[1], can_delete: false },
        EXTRACTIONS[2],
      ],
    } as never);

    render(
      <DeleteExtractionsModal
        model={model}
        onClose={vi.fn()}
        onDelete={vi.fn()}
        initialSelectedIds={new Set(['ext_a', 'ext_b'])}
      />
    );

    await waitFor(async () => expect(await checkedCount()).toBe(1));
  });

  it('still selects everything deletable when opened with no selection', async () => {
    render(
      <DeleteExtractionsModal model={model} onClose={vi.fn()} onDelete={vi.fn()} />
    );

    await waitFor(async () => expect(await checkedCount()).toBe(3));
  });

  it('falls back to select-all when the whole selection is undeletable', async () => {
    render(
      <DeleteExtractionsModal
        model={model}
        onClose={vi.fn()}
        onDelete={vi.fn()}
        initialSelectedIds={new Set(['ext_gone'])}
      />
    );

    await waitFor(async () => expect(await checkedCount()).toBe(3));
  });
});
