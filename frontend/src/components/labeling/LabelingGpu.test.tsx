/**
 * Local labeling carries a GPU; remote labeling does not.
 *
 * The LOCAL method loads a HuggingFace model onto a card in this app's worker,
 * which ran on card 0. The OpenAI and OpenAI-compatible methods call an endpoint
 * and hold no card of ours, so a `gpu` on those requests would describe work
 * that is not done here. The picker therefore appears for LOCAL only, and the
 * field is sent for LOCAL only.
 *
 * Local labeling can run its model split across GPUs (multi-GPU Phase 2), so
 * the picker offers "All GPUs".
 *
 * MUTATION CONTROLS:
 *   * send `gpu` unconditionally            -> "remote sends no gpu" fails
 *   * drop `gpu` from the LOCAL payload     -> "local sends gpu" fails
 *   * render the picker outside LOCAL       -> "no picker for remote" fails
 *   F4 drop allowSplit from the LOCAL picker (2026-09-14, restored by sha256)
 *                                           -> "local method: offers "All GPUs""
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

import { StartLabelingButton } from './StartLabelingButton';
import * as labelingAPI from '../../api/labeling';
import * as modelsAPI from '../../api/models';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

vi.mock('../../api/labeling', async (orig) => ({
  ...(await orig<typeof labelingAPI>()),
  startLabeling: vi.fn(),
}));
vi.mock('../../api/models', async (orig) => ({
  ...(await orig<typeof modelsAPI>()),
  getLocalModels: vi.fn(),
}));

const TWO_CARDS = {
  gpu_count: 2,
  gpus: [
    { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251', total_memory_gb: 12 },
    { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57', total_memory_gb: 24 },
  ],
};

async function openForm() {
  render(<StartLabelingButton extractionId="extr_1" />);
  fireEvent.click(await screen.findByText(/Label Features/i));
  return screen.findByDisplayValue('OpenAI (requires api-key)');
}

async function submit() {
  fireEvent.click(screen.getByRole('button', { name: /Start Labeling/i }));
  await waitFor(() => expect(labelingAPI.startLabeling).toHaveBeenCalledTimes(1));
  return vi.mocked(labelingAPI.startLabeling).mock.calls[0][0];
}

describe('labeling GPU placement', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(labelingAPI.startLabeling).mockResolvedValue({
      id: 'lbl_1', extraction_job_id: 'extr_1', status: 'queued', progress: 0,
    } as never);
    vi.mocked(modelsAPI.getLocalModels).mockResolvedValue({ models: ['meta-llama/Llama-3.2-1B'] } as never);
  });

  it('local method: shows the picker and sends gpu "auto"', async () => {
    const method = await openForm();
    fireEvent.change(method, { target: { value: 'local' } });

    expect(await screen.findByLabelText('GPU')).toBeInTheDocument();
    const sent = await submit();
    expect(sent.labeling_method).toBe('local');
    expect(sent.gpu).toBe('auto');
  });

  it('remote method: no picker, and no gpu on the request', async () => {
    // OpenAI, not OpenAI-compatible: starting the compatible method also saves
    // its endpoint to Settings, a real network write that has nothing to do
    // with GPU placement and left unhandled rejections behind this test.
    // Switched to local first so the picker is known to have been on screen.
    const method = await openForm();
    fireEvent.change(method, { target: { value: 'local' } });
    expect(await screen.findByLabelText('GPU')).toBeInTheDocument();
    fireEvent.change(method, { target: { value: 'openai' } });

    expect(screen.queryByLabelText('GPU')).not.toBeInTheDocument();
    const sent = await submit();
    expect(sent.labeling_method).toBe('openai');
    expect(sent.gpu).toBeUndefined();
  });

  it('local method: offers "All GPUs" and sends gpu "all"', async () => {
    useSystemMonitorStore.setState({ gpuList: TWO_CARDS } as never);
    const method = await openForm();
    fireEvent.change(method, { target: { value: 'local' } });

    const picker = (await screen.findByLabelText('GPU')) as HTMLSelectElement;
    expect(Array.from(picker.options).map((o) => o.value)).toEqual([
      'auto', TWO_CARDS.gpus[0].uuid, TWO_CARDS.gpus[1].uuid, 'all',
    ]);
    fireEvent.change(picker, { target: { value: 'all' } });

    const sent = await submit();
    expect(sent.labeling_method).toBe('local');
    expect(sent.gpu).toBe('all');
  });
});
