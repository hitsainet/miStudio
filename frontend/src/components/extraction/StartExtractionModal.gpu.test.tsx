/**
 * SAE feature extraction carries the GPU the user chose — single and batch.
 *
 * The extraction service passed "cuda" (card 0) for every job. The backend now
 * places each job by the request's `gpu`. There are two request builders here —
 * one SAE, and a batch over several — and a batch that forgot the field would
 * put every job in it on the backend default with nothing on screen to say so.
 *
 * MUTATION CONTROLS:
 *   * delete `gpu,` from the single-SAE config   -> "single SAE" tests fail
 *   * delete `gpu,` from the batch request       -> "batch" test fails
 *   F3 drop allowSplit from the picker (2026-09-14, restored by sha256)
 *                                                -> "single SAE: offers "All GPUs""
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../../api/saes', () => ({
  getReadySAEs: vi.fn(),
  startSAEExtraction: vi.fn(),
  startBatchSAEExtraction: vi.fn(),
}));
vi.mock('../../stores/featuresStore', () => ({
  useFeaturesStore: () => ({ isLoadingExtraction: false, extractionError: null }),
}));
vi.mock('../../stores/datasetsStore', () => ({
  useDatasetsStore: () => ({
    datasets: [{ id: 'ds_owt', name: 'OpenWebText-2M', status: 'ready' }],
    fetchDatasets: vi.fn(),
  }),
}));
vi.mock('../../stores/modelsStore', () => ({
  useModelsStore: () => ({ models: [], fetchModels: vi.fn() }),
}));
vi.mock('../../stores/extractionTemplatesStore', () => ({
  useExtractionTemplatesStore: () => ({
    templates: [], fetchTemplates: vi.fn(), createTemplate: vi.fn(),
  }),
}));

import { getReadySAEs, startBatchSAEExtraction, startSAEExtraction } from '../../api/saes';
import { StartExtractionModal } from './StartExtractionModal';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';

const sae = (id: string, layer: number) => ({
  id, name: `LFM2.5 L${layer}`, layer, status: 'ready', model_name: 'LFM2.5-1.2B-Instruct',
  architecture: 'jumprelu', n_features: 16384,
});

async function openWith(saeCount: number) {
  vi.mocked(getReadySAEs).mockResolvedValue({
    data: [sae('sae_11', 11), sae('sae_12', 12)].slice(0, saeCount),
  } as never);
  render(<StartExtractionModal isOpen onClose={vi.fn()} />);
  await screen.findByText('LFM2.5 L11');
  fireEvent.click(screen.getByText('LFM2.5 L11'));
  if (saeCount > 1) fireEvent.click(screen.getByText('LFM2.5 L12'));
  // The dataset control is a multi-select grid now, not a <select>, so that a
  // mixture can be expressed at all. Target the checkbox by role rather than
  // clicking the label's <span>: in jsdom a click on a label descendant does
  // not reliably fire the input's onChange.
  fireEvent.click(screen.getByRole('checkbox', { name: 'OpenWebText-2M' }));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(startSAEExtraction).mockResolvedValue({ id: 'ext_1' } as never);
  vi.mocked(startBatchSAEExtraction).mockResolvedValue({
    batch_id: 'b1', total_created: 2, total_skipped: 0, created_jobs: [], skipped_saes: [],
  } as never);
  useSystemMonitorStore.setState({
    gpuList: {
      gpu_count: 2,
      gpus: [
        { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251', total_memory_gb: 12 },
        { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 },
      ],
    },
  } as never);
});

describe('StartExtractionModal GPU placement', () => {
  it('single SAE: sends gpu "auto" when no card was chosen', async () => {
    await openWith(1);
    fireEvent.click(screen.getByRole('button', { name: 'Start Extraction' }));

    await waitFor(() => expect(startSAEExtraction).toHaveBeenCalledTimes(1));
    const [saeId, datasetId, config] = vi.mocked(startSAEExtraction).mock.calls[0];
    expect(saeId).toBe('sae_11');
    expect(datasetId).toBe('ds_owt');
    expect(config.gpu).toBe('auto');
  });

  it('single SAE: sends the chosen card UUID', async () => {
    await openWith(1);
    fireEvent.change(screen.getByLabelText('GPU'), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: 'Start Extraction' }));

    await waitFor(() => expect(startSAEExtraction).toHaveBeenCalledTimes(1));
    expect(vi.mocked(startSAEExtraction).mock.calls[0][2].gpu).toBe(RTX_3090_UUID);
  });

  it('batch: every job is placed by the same choice', async () => {
    await openWith(2);
    fireEvent.change(screen.getByLabelText('GPU'), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: /Start Batch \(2 SAEs\)/ }));

    await waitFor(() => expect(startBatchSAEExtraction).toHaveBeenCalledTimes(1));
    const request = vi.mocked(startBatchSAEExtraction).mock.calls[0][0];
    expect(request.sae_ids).toEqual(['sae_11', 'sae_12']);
    expect(request.gpu).toBe(RTX_3090_UUID);
    expect(startSAEExtraction).not.toHaveBeenCalled();
  });

  it('single SAE: offers "All GPUs" and sends gpu "all"', async () => {
    await openWith(1);

    const picker = screen.getByLabelText('GPU') as HTMLSelectElement;
    expect(Array.from(picker.options).map((o) => o.value)).toEqual([
      'auto', 'GPU-f47ba814-49a2-603f-3595-275284140251', RTX_3090_UUID, 'all',
    ]);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start Extraction' }));

    await waitFor(() => expect(startSAEExtraction).toHaveBeenCalledTimes(1));
    expect(vi.mocked(startSAEExtraction).mock.calls[0][2].gpu).toBe('all');
  });
});
