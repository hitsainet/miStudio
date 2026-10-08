/**
 * The System Monitor never assumes card 0.
 *
 * `/system/all` used to be called with `gpu_id=0` by default, and the server
 * used to answer an unknown id with card 0's numbers. With two cards that
 * described the 12 GB 3080 Ti (which became index 0 on 2026-09-13) under
 * whatever id the page thought it was showing. The server now requires an id on
 * every per-card endpoint and returns every card from `/system/all` when none
 * is given; the store must read that shape and pick from what was LISTED.
 *
 * MUTATION CONTROLS:
 *   * call getAllMonitoringData(get().selectedGPU ?? 0)  -> "asks for every card" fails
 *   * pickDisplayedCard returns cards.find(...) ?? null   -> "stale id" fails
 *   * drop the allGpuMetrics fill from fetchAllMetrics   -> "records every card" fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api/system', () => ({
  getAllMonitoringData: vi.fn(),
  getGPUList: vi.fn(),
  getAllGPUMetrics: vi.fn(),
}));

import { getAllMonitoringData } from '../api/system';
import { cardsOf, pickDisplayedCard, useSystemMonitorStore } from './systemMonitorStore';
import type { AllMonitoringDataResponse, MonitoringGpuCard } from '../types/system';

function card(gpu_id: number, name: string, utilization: number): MonitoringGpuCard {
  return {
    gpu_id,
    metrics: {
      gpu_id,
      utilization: { gpu: utilization, memory: 0 },
      memory: { used: 0, total: 0, free: 0, used_gb: 0, total_gb: 0, used_percent: 0 },
      temperature: 40 + gpu_id,
      power: { usage: 0, limit: 0, usage_percent: 0 },
      fan_speed: 0,
      clock_speed: { gpu: 0, memory: 0 },
      pcie: { tx: 0, rx: 0 },
    } as unknown as MonitoringGpuCard['metrics'],
    info: { gpu_id, name } as unknown as MonitoringGpuCard['info'],
    processes: [],
  };
}

// Utilisation differs per card, so a test reading the wrong card's block
// cannot pass by coincidence.
const TI = card(0, 'NVIDIA GeForce RTX 3080 Ti', 11);
const RTX_3090 = card(1, 'NVIDIA GeForce RTX 3090', 87);

function response(cards: MonitoringGpuCard[]): AllMonitoringDataResponse {
  return {
    gpu_available: true,
    system: {} as AllMonitoringDataResponse['system'],
    disk_usage: [],
    network_rates: {} as AllMonitoringDataResponse['network_rates'],
    disk_rates: {} as AllMonitoringDataResponse['disk_rates'],
    gpu: { gpu_count: cards.length, selected_gpu_id: null, cards },
  };
}

beforeEach(() => {
  vi.mocked(getAllMonitoringData).mockReset();
  useSystemMonitorStore.setState({
    selectedGPU: null,
    allGpuMetrics: {},
    allGpuInfo: {},
    gpuMetrics: null,
    gpuInfo: null,
    consecutiveErrors: 0,
  });
});

describe('pickDisplayedCard', () => {
  it('shows the chosen card when it is listed', () => {
    expect(pickDisplayedCard([TI, RTX_3090], 1)).toBe(RTX_3090);
  });

  it('shows the first LISTED card when nothing is chosen', () => {
    // The first listed card, not a literal 0: a list that starts at 1 must
    // not produce a request for a card that is not there.
    expect(pickDisplayedCard([RTX_3090], null)).toBe(RTX_3090);
  });

  it('never keeps a stale id the server no longer lists', () => {
    expect(pickDisplayedCard([TI, RTX_3090], 5)).toBe(TI);
  });

  it('shows nothing when there are no cards', () => {
    expect(pickDisplayedCard([], 0)).toBeNull();
  });
});

describe('cardsOf', () => {
  it('reads the single-card shape that /system/all?gpu_id=N returns', () => {
    const single: AllMonitoringDataResponse = {
      ...response([]),
      gpu: {
        gpu_count: 2,
        selected_gpu_id: 1,
        metrics: RTX_3090.metrics,
        info: RTX_3090.info,
        processes: [],
      },
    };
    expect(cardsOf(single).map((c) => c.gpu_id)).toEqual([1]);
  });
});

describe('fetchAllMetrics', () => {
  it('asks for every card — no gpu_id — and shows the first listed one', async () => {
    vi.mocked(getAllMonitoringData).mockResolvedValue(response([TI, RTX_3090]));

    await useSystemMonitorStore.getState().fetchAllMetrics();

    expect(getAllMonitoringData).toHaveBeenCalledTimes(1);
    expect(vi.mocked(getAllMonitoringData).mock.calls[0]).toEqual([]);
    const state = useSystemMonitorStore.getState();
    expect(state.selectedGPU).toBe(0);
    expect(state.gpuMetrics?.utilization.gpu).toBe(11);
  });

  it('keeps showing the card the user chose', async () => {
    vi.mocked(getAllMonitoringData).mockResolvedValue(response([TI, RTX_3090]));
    useSystemMonitorStore.setState({ selectedGPU: 1 });

    await useSystemMonitorStore.getState().fetchAllMetrics();

    expect(vi.mocked(getAllMonitoringData).mock.calls[0]).toEqual([]);
    const state = useSystemMonitorStore.getState();
    expect(state.selectedGPU).toBe(1);
    expect(state.gpuMetrics?.utilization.gpu).toBe(87);
  });

  it('records every card, so the compare view and header show both', async () => {
    vi.mocked(getAllMonitoringData).mockResolvedValue(response([TI, RTX_3090]));

    await useSystemMonitorStore.getState().fetchAllMetrics();

    const { allGpuMetrics } = useSystemMonitorStore.getState();
    expect(Object.keys(allGpuMetrics).sort()).toEqual(['0', '1']);
    expect(allGpuMetrics[1].utilization.gpu).toBe(87);
  });
});
