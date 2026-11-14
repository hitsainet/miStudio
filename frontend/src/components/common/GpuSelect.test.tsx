/**
 * The GPU picker offers Auto and every card, and sends a card's UUID.
 *
 * An index is not an identity: adding an RTX 3080 Ti on 2026-09-13 moved the
 * RTX 3090 from GPU 0 to GPU 1, so a saved `0` changed meaning overnight.
 *
 * "All GPUs" (multi-GPU Phase 2) is offered only where the job can run split,
 * and only with more than one card.
 *
 * MUTATION CONTROLS (2026-09-14; each alone on GpuSelect.tsx, restored
 * byte-identically) — all three went red:
 *   G1 offered with a single card              -> "does not offer it with a single card"
 *   G2 offered without allowSplit              -> "offers Auto first", "does not offer it to a job that
 *                                                  cannot run split", "keeps a saved all visible"
 *   G3 a saved "all" shown as an unavailable card -> "keeps a saved all visible ... not supported"
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GpuSelect } from './GpuSelect';

const store = vi.hoisted(() => ({
  gpuList: null as unknown,
  fetchGPUList: (() => undefined) as () => void,
}));

vi.mock('../../stores/systemMonitorStore', () => ({
  useSystemMonitorStore: () => store,
}));

const card = (gpu_id: number, name: string, uuid: string, total_memory_gb: number) => ({
  gpu_id,
  name,
  uuid,
  pci_bus_id: '',
  driver_version: '580.173.02',
  cuda_version: '13.0',
  compute_capability: '8.6',
  total_memory_gb,
});

const TWO_CARDS = {
  gpu_count: 2,
  gpus: [
    card(0, 'NVIDIA GeForce RTX 3080 Ti', 'GPU-f47ba814-49a2-603f-3595-275284140251', 12),
    card(1, 'NVIDIA GeForce RTX 3090', 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57', 24),
  ],
};

const optionValues = () =>
  Array.from((screen.getByLabelText(/gpu/i) as HTMLSelectElement).options).map((o) => o.value);

describe('GpuSelect', () => {
  beforeEach(() => {
    store.gpuList = TWO_CARDS;
    store.fetchGPUList = vi.fn();
  });

  it('offers Auto first, then every card by UUID', () => {
    render(<GpuSelect id="gpu" value="auto" onChange={vi.fn()} />);

    expect(optionValues()).toEqual([
      'auto',
      'GPU-f47ba814-49a2-603f-3595-275284140251',
      'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
    ]);
    expect(screen.getByRole('option', { name: /GPU 1: NVIDIA GeForce RTX 3090 \(24 GB\)/ })).toBeTruthy();
  });

  it('sends the chosen card UUID, not its index', () => {
    const onChange = vi.fn();
    render(<GpuSelect id="gpu" value="auto" onChange={onChange} />);

    fireEvent.change(screen.getByLabelText(/gpu/i), {
      target: { value: 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57' },
    });

    expect(onChange).toHaveBeenCalledWith('GPU-247aa582-0d1b-e161-8156-983ed1fefc57');
  });

  it('asks for the GPU list when it has none, and still offers Auto', () => {
    store.gpuList = null;
    render(<GpuSelect id="gpu" value="auto" onChange={vi.fn()} />);

    expect(optionValues()).toEqual(['auto']);
    expect(store.fetchGPUList).toHaveBeenCalledTimes(1);
  });

  it('shows a saved card that is no longer installed as unavailable', () => {
    render(<GpuSelect id="gpu" value="GPU-00000000-gone" onChange={vi.fn()} />);

    expect(screen.getByRole('option', { name: /Unavailable GPU/ })).toBeTruthy();
    expect((screen.getByLabelText(/gpu/i) as HTMLSelectElement).value).toBe('GPU-00000000-gone');
  });

  describe('splitting a model across GPUs (multi-GPU Phase 2)', () => {
    it('offers "All GPUs" after the cards when the job can run split', () => {
      const onChange = vi.fn();
      render(<GpuSelect id="gpu" value="auto" onChange={onChange} allowSplit />);

      expect(optionValues()).toEqual([
        'auto',
        'GPU-f47ba814-49a2-603f-3595-275284140251',
        'GPU-247aa582-0d1b-e161-8156-983ed1fefc57',
        'all',
      ]);
      fireEvent.change(screen.getByLabelText(/gpu/i), { target: { value: 'all' } });
      expect(onChange).toHaveBeenCalledWith('all');
    });

    it('does not offer it to a job that cannot run split', () => {
      render(<GpuSelect id="gpu" value="auto" onChange={vi.fn()} />);

      expect(optionValues()).not.toContain('all');
    });

    it('does not offer it with a single card, where there is nothing to split across', () => {
      store.gpuList = { gpu_count: 1, gpus: [TWO_CARDS.gpus[1]] };
      render(<GpuSelect id="gpu" value="auto" onChange={vi.fn()} allowSplit />);

      expect(optionValues()).toEqual(['auto', 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57']);
    });

    it('keeps a saved "all" visible on a form that cannot split, marked as not supported', () => {
      render(<GpuSelect id="gpu" value="all" onChange={vi.fn()} />);

      expect(screen.getByRole('option', { name: /not supported/i })).toBeTruthy();
      expect(screen.queryByRole('option', { name: /Unavailable GPU/ })).toBeNull();
    });
  });
});
