/**
 * Every J-Lens GPU job carries the card the user chose.
 *
 * Fit, acquire (which validates by reading out through the lens), intervention
 * and readout all passed "cuda" as the device map — card 0, the 12 GB card since
 * 2026-09-13, where gemma-4-12B does not fit. The backend now places each by the
 * request's `gpu`. Publishing is an upload and takes no card, so it sends none.
 *
 * MUTATION CONTROLS:
 *   * delete `gpu,` from FitLensCard's fit request        -> the fit tests fail
 *   * delete `gpu,` from InterventionCard's request       -> the intervention test fails
 *   * delete `gpu: get().gpu` from the store readout      -> the readout test fails
 *
 * "All GPUs" (multi-GPU Phase 2): an intervention can run split; a fit stays on
 * one card. Controls (2026-09-14, each restored byte-identically, sha256):
 *   F8  drop allowSplit from InterventionCard's picker -> "offers "All GPUs" and sends gpu "all""
 *   F13 add allowSplit to FitLensCard's picker         -> "does not offer "All GPUs" — a fit stays
 *                                                          on one card"
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('../../api/jlens', () => ({
  jlensApi: {
    fit: vi.fn(),
    intervene: vi.fn(),
    checkTokens: vi.fn(),
    readout: vi.fn(),
    readoutResult: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
  },
}));
vi.mock('../../api/models', () => ({ getTaskStatus: vi.fn() }));

import { jlensApi } from '../../api/jlens';
import { FitLensCard, MIN_FIT_PROMPTS } from './FitLensCard';
import { InterventionCard } from './InterventionCard';
import { useJLensStore } from '../../stores/jlensStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
const TWO_CARDS = {
  gpu_count: 2,
  gpus: [
    { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251', total_memory_gb: 12 },
    { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 },
  ],
};

const queued = { task_id: 't-1', model_id: 'm_1', queue: 'extraction' };

beforeEach(() => {
  vi.clearAllMocks();
  useSystemMonitorStore.setState({ gpuList: TWO_CARDS } as never);
});

async function fillFit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /fit a lens/i }));
  await user.click(screen.getByLabelText(/one prompt per line/i));
  await user.paste(Array.from({ length: MIN_FIT_PROMPTS }, (_, i) => `prompt ${i}`).join('\n'));
  await user.type(screen.getByLabelText(/corpus name/i), 'acceptance-100');
  await user.type(screen.getByLabelText(/probe prompt/i), 'The capital of France is');
  await user.type(screen.getByLabelText(/expected intermediate/i), 'Paris');
}

describe('FitLensCard', () => {
  it('sends gpu "auto" when no card was chosen', async () => {
    const user = userEvent.setup();
    vi.mocked(jlensApi.fit).mockResolvedValue(queued);
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await fillFit(user);
    await user.click(screen.getByRole('button', { name: /^fit$/i }));

    await waitFor(() => expect(jlensApi.fit).toHaveBeenCalledTimes(1));
    expect(vi.mocked(jlensApi.fit).mock.calls[0][0].gpu).toBe('auto');
  });

  it('sends the chosen card UUID', async () => {
    const user = userEvent.setup();
    vi.mocked(jlensApi.fit).mockResolvedValue(queued);
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await fillFit(user);
    fireEvent.change(screen.getByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    await user.click(screen.getByRole('button', { name: /^fit$/i }));

    await waitFor(() => expect(jlensApi.fit).toHaveBeenCalledTimes(1));
    expect(vi.mocked(jlensApi.fit).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
  });

  it('does not offer "All GPUs" — a fit stays on one card', async () => {
    const user = userEvent.setup();
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await fillFit(user);

    const picker = screen.getByLabelText(/^GPU/) as HTMLSelectElement;
    // Both cards are listed, so the missing "all" is not an empty picker.
    expect(Array.from(picker.options).map((o) => o.value)).toEqual([
      'auto', TWO_CARDS.gpus[0].uuid, RTX_3090_UUID,
    ]);
  });
});

describe('InterventionCard', () => {
  it('offers "All GPUs" and sends gpu "all" with the intervention', async () => {
    vi.mocked(jlensApi.intervene).mockResolvedValue(queued);
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));

    const picker = screen.getByLabelText('GPU') as HTMLSelectElement;
    expect(Array.from(picker.options).map((o) => o.value)).toEqual([
      'auto', TWO_CARDS.gpus[0].uuid, RTX_3090_UUID, 'all',
    ]);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    expect(vi.mocked(jlensApi.intervene).mock.calls[0][0].gpu).toBe('all');
  });

  it('sends the chosen card UUID with the intervention', async () => {
    vi.mocked(jlensApi.intervene).mockResolvedValue(queued);
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByLabelText('GPU'), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    expect(vi.mocked(jlensApi.intervene).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
  });
});

describe('jlensStore readout', () => {
  it('sends the store GPU, and "auto" until one is set', async () => {
    vi.mocked(jlensApi.readout).mockResolvedValue({ task_id: 't1', model_id: 'm_1', status: 'queued' } as never);
    vi.mocked(jlensApi.readoutResult).mockResolvedValue({ task_id: 't1', status: 'FAILURE', error: 'x' } as never);

    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null, gpu: 'auto' });
    await useJLensStore.getState().fetchReadout();
    expect(vi.mocked(jlensApi.readout).mock.calls[0][0].gpu).toBe('auto');

    useJLensStore.getState().setGpu(RTX_3090_UUID);
    await useJLensStore.getState().fetchReadout();
    expect(vi.mocked(jlensApi.readout).mock.calls[1][0].gpu).toBe(RTX_3090_UUID);
  });

  it('sends unload_after false by default, and true once the user asks to unload', async () => {
    // MUTATION CONTROL: drop `unload_after` from the store's readout request -> fails.
    vi.mocked(jlensApi.readout).mockResolvedValue({ task_id: 't1', model_id: 'm_1', status: 'queued' } as never);
    vi.mocked(jlensApi.readoutResult).mockResolvedValue({ task_id: 't1', status: 'FAILURE', error: 'x' } as never);

    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null, unloadAfterReadout: false });
    await useJLensStore.getState().fetchReadout();
    expect(vi.mocked(jlensApi.readout).mock.calls[0][0].unload_after).toBe(false);

    useJLensStore.getState().setUnloadAfterReadout(true);
    await useJLensStore.getState().fetchReadout();
    expect(vi.mocked(jlensApi.readout).mock.calls[1][0].unload_after).toBe(true);
  });

  it('does not persist the GPU — a saved UUID can name a removed card', () => {
    useJLensStore.getState().setGpu(RTX_3090_UUID);
    // The store's PERSIST name ('jlens-store' is its devtools name). The first
    // assertion proves the key is being written, so the second cannot pass
    // just because nothing was persisted at all.
    const persisted = localStorage.getItem('miStudio-jlens') ?? '';
    expect(persisted).toContain('"modelId"');
    expect(persisted).not.toContain(RTX_3090_UUID);
  });
});
