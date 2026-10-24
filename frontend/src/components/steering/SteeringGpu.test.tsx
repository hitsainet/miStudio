/**
 * Every steering generation carries the GPU chosen in Generation Config.
 *
 * The steering worker was spawned with CUDA_VISIBLE_DEVICES="0", which since
 * 2026-09-13 is the 12 GB card. The backend now places compare, sweep and
 * combined work by the request's `gpu`; this pins that the panel control sets
 * it and that EVERY request builder sends it — five builders, so one that
 * forgets is not hidden by the others.
 *
 * MUTATION CONTROLS:
 *   * delete `gpu: get().gpu,` from generateComparison      -> compare tests fail
 *   * delete it from the batch Blended builder              -> batch blended test fails
 *   * drop setGpu from GenerationConfig's onChange          -> the control test fails
 *   F5 drop allowSplit from GenerationConfig's picker (2026-09-14, restored by sha256)
 *                                                           -> "offers "All GPUs", and a compare
 *                                                               then sends gpu "all""
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';

vi.mock('../../api/steering', () => ({
  submitAsyncComparison: vi.fn(),
  submitAsyncCombined: vi.fn(),
  computeClusterAllocation: vi.fn(),
  submitAsyncSweep: vi.fn(),
  cancelTask: vi.fn().mockResolvedValue({}),
  getExperiments: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
}));

import * as steeringApi from '../../api/steering';
import { GenerationConfig } from './GenerationConfig';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import type { SAE } from '../../types/sae';

const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
const TWO_CARDS = {
  gpu_count: 2,
  gpus: [
    { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251', total_memory_gb: 12 },
    { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 },
  ],
};

const SAE_FIXTURE = {
  id: 'sae-123',
  name: 'Test SAE',
  model_id: 'model-456',
  architecture: 'standard',
  layer: 6,
  d_model: 768,
  n_features: 4096,
  status: 'ready',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
} as unknown as SAE;

const accepted = { task_id: 't-1', status: 'pending' } as never;

/** Let a submit reach the API mock; the resolver it then waits on never settles here. */
const settle = () => act(async () => { await new Promise((r) => setTimeout(r, 80)); });

function select(prompts: string[]) {
  act(() => {
    const store = useSteeringStore.getState();
    store.selectSAE(SAE_FIXTURE);
    store.setPrompts(prompts);
    // Only the fields the request builders read; the rest of AddFeatureInput is display metadata.
    store.addFeature({ feature_idx: 100, layer: 6, strength: 2.0 } as Parameters<typeof store.addFeature>[0]);
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  act(() => {
    useSteeringStore.setState({
      selectedSAE: null,
      selectedFeatures: [],
      prompts: [''],
      isGenerating: false,
      isCombinedGenerating: false,
      isSweeping: false,
      batchState: null,
      taskId: null,
      clusterBudget: null,
      layerBudgets: null,
      combinedMode: false,
      gpu: 'auto',
      error: null,
    });
  });
  useSystemMonitorStore.setState({ gpuList: TWO_CARDS } as never);
  vi.mocked(steeringApi.submitAsyncComparison).mockResolvedValue(accepted);
  vi.mocked(steeringApi.submitAsyncCombined).mockResolvedValue(accepted);
  vi.mocked(steeringApi.submitAsyncSweep).mockResolvedValue(accepted);
});

describe('the Generation Config GPU control', () => {
  it('sets the store GPU to the chosen card UUID', () => {
    render(<GenerationConfig />);
    fireEvent.change(screen.getByLabelText('GPU'), { target: { value: RTX_3090_UUID } });
    expect(useSteeringStore.getState().gpu).toBe(RTX_3090_UUID);
  });

  it('is disabled while a generation runs', () => {
    act(() => { useSteeringStore.setState({ isGenerating: true }); });
    render(<GenerationConfig />);
    expect(screen.getByLabelText('GPU')).toBeDisabled();
  });

  it('offers "All GPUs", and a compare then sends gpu "all"', async () => {
    select(['P']);
    render(<GenerationConfig />);

    const picker = screen.getByLabelText('GPU') as HTMLSelectElement;
    expect(Array.from(picker.options).map((o) => o.value)).toEqual([
      'auto', TWO_CARDS.gpus[0].uuid, RTX_3090_UUID, 'all',
    ]);
    fireEvent.change(picker, { target: { value: 'all' } });
    expect(useSteeringStore.getState().gpu).toBe('all');

    void useSteeringStore.getState().generateComparison(true, false).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncComparison).toHaveBeenCalledTimes(1);
    expect(vi.mocked(steeringApi.submitAsyncComparison).mock.calls[0][0].gpu).toBe('all');
  });
});

describe('every steering request carries gpu', () => {
  it('compare sends "auto" by default', async () => {
    select(['P']);
    void useSteeringStore.getState().generateComparison(true, false).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncComparison).toHaveBeenCalledTimes(1);
    expect(vi.mocked(steeringApi.submitAsyncComparison).mock.calls[0][0].gpu).toBe('auto');
  });

  it('compare sends the UUID picked in the panel control', async () => {
    select(['P']);
    render(<GenerationConfig />);
    fireEvent.change(screen.getByLabelText('GPU'), { target: { value: RTX_3090_UUID } });
    void useSteeringStore.getState().generateComparison(true, false).catch(() => {});
    await settle();
    expect(vi.mocked(steeringApi.submitAsyncComparison).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
  });

  it('combined sends it', async () => {
    select(['P']);
    act(() => { useSteeringStore.getState().setGpu(RTX_3090_UUID); });
    void useSteeringStore.getState().generateCombined(true, false).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncCombined).toHaveBeenCalledTimes(1);
    expect(vi.mocked(steeringApi.submitAsyncCombined).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
    act(() => { useSteeringStore.setState({ isCombinedGenerating: false }); });
  });

  it('sweep sends it', async () => {
    select(['P']);
    void useSteeringStore.getState().runStrengthSweep(100, 6, [0.5, 1.0]).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncSweep).toHaveBeenCalledTimes(1);
    expect(vi.mocked(steeringApi.submitAsyncSweep).mock.calls[0][0].gpu).toBe('auto');
    act(() => { useSteeringStore.setState({ isSweeping: false }); });
  });

  it('batch Compare sends it on each prompt', async () => {
    select(['A', 'B']);
    act(() => { useSteeringStore.getState().setGpu(RTX_3090_UUID); });
    void useSteeringStore.getState().generateBatchComparison(true, false).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncComparison).toHaveBeenCalled();
    expect(vi.mocked(steeringApi.submitAsyncComparison).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
    act(() => { useSteeringStore.getState().abortBatch(); });
  });

  it('batch Blended sends it on each prompt', async () => {
    select(['A', 'B']);
    act(() => {
      useSteeringStore.setState({ combinedMode: true });
      useSteeringStore.getState().setGpu(RTX_3090_UUID);
    });
    void useSteeringStore.getState().generateBatchComparison(true, false).catch(() => {});
    await settle();
    expect(steeringApi.submitAsyncCombined).toHaveBeenCalled();
    expect(vi.mocked(steeringApi.submitAsyncCombined).mock.calls[0][0].gpu).toBe(RTX_3090_UUID);
    act(() => { useSteeringStore.getState().abortBatch(); });
  });
});
