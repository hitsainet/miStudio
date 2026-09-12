/**
 * Reachability for the carried selection: the journey, not the prop.
 *
 * `DeleteExtractionsModal` accepting `initialSelectedIds` is worth nothing if
 * the screen that shows "Delete Selected (N)" never passes it -- which is
 * exactly how the bug shipped. These tests drive the real path: tick one row
 * in Extraction History, press Delete Selected, and read what the confirmation
 * arrives with.
 *
 * Removing `initialSelectedIds` from either call site must go red.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ActivationExtractionHistory } from './ActivationExtractionHistory';
import { ExtractionListModal } from './ExtractionListModal';
import { getModelExtractions } from '../../api/models';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useModelsStore } from '../../stores/modelsStore';

vi.mock('../../api/models', async (orig) => ({
  ...(await orig<typeof import('../../api/models')>()),
  getModelExtractions: vi.fn(),
  deleteExtractions: vi.fn().mockResolvedValue({}),
}));

const model = { id: 'm_b55c6926', name: 'gemma-4-12B-it' } as never;

function extraction(id: string, samples: number, status: string) {
  return {
    extraction_id: id,
    model_id: 'm_b55c6926',
    dataset_id: `ds_${id}`,
    architecture: 'gemma3',
    quantization: 'q4',
    dataset_path: `/data/${id}`,
    layer_indices: [10],
    hook_types: ['residual'],
    max_samples: samples,
    batch_size: 8,
    num_samples: samples,
    num_samples_processed: samples,
    created_at: '2026-08-25T00:11:20Z',
    completed_at: '2026-08-25T00:57:03Z',
    saved_files: [`/data/${id}/L10.pt`],
    statistics: {},
    status,
    can_delete: true,
  };
}

const EXTRACTIONS = [
  extraction('ext_a', 16, 'failed'),
  extraction('ext_b', 10000, 'completed'),
  extraction('ext_c', 10000, 'completed'),
];

/** Row checkboxes inside the confirmation dialog, excluding its select-all. */
async function confirmationSelectedCount() {
  const heading = await screen.findByText(/select extractions to delete/i);
  const dialog = heading.closest('div[class*="fixed"]') as HTMLElement;
  const boxes = within(dialog).getAllByRole('checkbox') as HTMLInputElement[];
  return boxes.slice(1).filter((b) => b.checked).length;
}

// Each screen gets its own render so both keep their real prop types --
// sharing one element shape would need a cast, and a cast here could hide the
// very wiring these tests exist to check.
describe.each([
  [
    'ActivationExtractionHistory',
    () => <ActivationExtractionHistory model={model} onClose={vi.fn()} />,
  ],
  [
    'ExtractionListModal',
    () => (
      <ExtractionListModal
        model={model}
        onClose={vi.fn()}
        onSelectExtraction={vi.fn()}
      />
    ),
  ],
])('%s hands its selection to the confirmation', (_name, renderScreen) => {
  beforeEach(() => {
    vi.mocked(getModelExtractions).mockResolvedValue({
      extractions: EXTRACTIONS,
    } as never);
    useDatasetsStore.setState({
      datasets: [],
      fetchDatasets: vi.fn().mockResolvedValue(undefined),
    } as never);
    useModelsStore.setState({
      models: [],
      fetchModels: vi.fn().mockResolvedValue(undefined),
    } as never);
  });

  it('opens the confirmation with only the row that was ticked', async () => {
    render(renderScreen());

    // one row, chosen deliberately
    const rowBoxes = (await screen.findAllByRole('checkbox')) as HTMLInputElement[];
    await userEvent.click(rowBoxes[1]);

    const deleteButton = await screen.findByRole('button', {
      name: /delete selected \(1\)/i,
    });
    await userEvent.click(deleteButton);

    await waitFor(async () =>
      expect(await confirmationSelectedCount()).toBe(1)
    );
  });

  it('carries two ticked rows, not all three', async () => {
    render(renderScreen());

    const rowBoxes = (await screen.findAllByRole('checkbox')) as HTMLInputElement[];
    await userEvent.click(rowBoxes[1]);
    await userEvent.click(rowBoxes[2]);

    await userEvent.click(
      await screen.findByRole('button', { name: /delete selected \(2\)/i })
    );

    await waitFor(async () =>
      expect(await confirmationSelectedCount()).toBe(2)
    );
  });
});
