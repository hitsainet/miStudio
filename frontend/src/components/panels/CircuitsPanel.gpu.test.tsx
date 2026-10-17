/**
 * Every circuits GPU job carries the card the user chose, reached through the
 * real panel and its tabs — not by rendering an unexported helper.
 *
 * Capture, attribution, validation, faithfulness and calibration all passed
 * "cuda" (card 0, the 12 GB card since 2026-09-13). The backend now places each
 * by the request's `gpu`. Discovery is statistics over a stored capture and
 * takes no card, so it is deliberately not here.
 *
 * MUTATION CONTROLS:
 *   * delete `gpu,` from ValidationTab's body              -> validation tests fail
 *   * send confirmCapture(id, { gpu: AUTO_GPU })            -> "confirm follows the picker" fails
 *   * delete `gpu` from the faithfulness body              -> faithfulness test fails
 *
 * "All GPUs" (multi-GPU Phase 2): capture, attribution, validation, faithfulness
 * and calibration all run split, so every circuits GPU picker offers it.
 * Controls (2026-09-14, restored by sha256):
 *   F9  drop allowSplit from the capture picker      -> "capture offers it, and confirm sends gpu "all""
 *   F10 drop allowSplit from the attribution picker  -> "the attribution pass offers it ..."
 *   F11 drop allowSplit from the validation picker   -> "validation offers it and sends gpu "all""
 *   F24 drop allowSplit from the faithfulness &
 *       calibration picker                           -> "the faithfulness & calibration picker offers
 *                                                        it ...", "calibration, from the same picker,
 *                                                        sends gpu "all""
 *   (F12, the earlier "add allowSplit" control, pinned the old decision that
 *    calibration could not split; the backend now splits it.)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../../api/circuits', () => ({
  circuitsApi: {
    list: vi.fn(),
    get: vi.fn(),
    exportUrl: vi.fn(() => '#'),
    listCaptures: vi.fn(),
    getCapture: vi.fn(),
    createCapture: vi.fn(),
    confirmCapture: vi.fn(),
    listDiscoveries: vi.fn(),
    getDiscovery: vi.fn(),
    createDiscovery: vi.fn(),
    startAttribution: vi.fn(),
    startValidation: vi.fn(),
    startFaithfulness: vi.fn(),
    startCalibration: vi.fn(),
    listManifests: vi.fn(),
  },
}));

import { circuitsApi } from '../../api/circuits';
import { CircuitsPanel } from './CircuitsPanel';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useSAEsStore } from '../../stores/saesStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { SAEStatus } from '../../types/sae';
import type { Circuit, CircuitCapture, DiscoveryRun } from '../../types/circuits';

const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';

const RUN = {
  id: 'run-12345678',
  capture_run_id: 'cap-1',
  status: 'completed',
  progress: 100,
  params: null,
  report: null,
  candidate_count: 3,
  candidates: [],
  attribution_status: null,
  validation_status: null,
  created_at: '',
  updated_at: '',
} as unknown as DiscoveryRun;

const CIRCUIT = {
  id: 'crc-1',
  name: 'Humor circuit',
  granularity: 'feature',
  layers: [11, 12],
  member_count: 1,
  edge_count: 0,
  rung: 2,
  rung_language: 'causally validated',
  rung_next_step: '',
  promoted: false,
  model_id: null,
  version: 1,
  updated_at: '',
  narrative: null,
  saes: [],
  members: [{ layer: 11, feature_idx: 5, sae_id: 'sae_11' }],
  edges: [],
  budget: null,
  faithfulness: null,
  faithfulness_status: null,
  calibration: null,
  calibration_status: null,
  discovery: null,
  discovery_run_id: 'run-12345678',
  created_at: '',
} as unknown as Circuit;

const accepted = { id: 'x', task_id: 't', status: 'queued' };

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(circuitsApi.list).mockResolvedValue({ circuits: [CIRCUIT], total: 1 });
  vi.mocked(circuitsApi.get).mockResolvedValue(CIRCUIT);
  vi.mocked(circuitsApi.listCaptures).mockResolvedValue({ captures: [], limit: 50, offset: 0 });
  vi.mocked(circuitsApi.listDiscoveries).mockResolvedValue({ discoveries: [RUN], limit: 50, offset: 0 });
  vi.mocked(circuitsApi.getDiscovery).mockResolvedValue(RUN);
  vi.mocked(circuitsApi.listManifests).mockResolvedValue({ manifests: [] });
  vi.mocked(circuitsApi.startAttribution).mockResolvedValue(accepted);
  vi.mocked(circuitsApi.startValidation).mockResolvedValue(accepted);
  vi.mocked(circuitsApi.startFaithfulness).mockResolvedValue({ circuit_id: 'crc-1', task_id: 't', status: 'queued' });
  vi.mocked(circuitsApi.startCalibration).mockResolvedValue({ circuit_id: 'crc-1', task_id: 't', status: 'queued' });
  vi.mocked(circuitsApi.createCapture).mockResolvedValue({ id: 'cap-1', task_id: 't', status: 'queued', confirmed: false });
  vi.mocked(circuitsApi.getCapture).mockResolvedValue({
    id: 'cap-1', status: 'estimated', estimate: { events: 10, bytes: 1024, minutes: 1 }, stale: false,
  } as unknown as CircuitCapture);
  vi.mocked(circuitsApi.confirmCapture).mockResolvedValue(accepted);

  useDatasetsStore.setState({ datasets: [{ id: 'ds_owt', name: 'OpenWebText-2M' }] } as never);
  useSAEsStore.setState({
    saes: [{ id: 'sae_11', name: 'LFM2.5 L11', layer: 11, status: SAEStatus.READY }],
  } as never);
  useSystemMonitorStore.setState({
    gpuList: {
      gpu_count: 1,
      gpus: [{ gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 }],
    },
  } as never);
});

const openTab = (label: string) => fireEvent.click(screen.getByRole('button', { name: label }));

describe('Capture', () => {
  it('estimate sends gpu "auto"; confirm follows the picker', async () => {
    render(<CircuitsPanel />);
    openTab('Capture');
    fireEvent.change(screen.getByDisplayValue('Select a dataset…'), { target: { value: 'ds_owt' } });
    fireEvent.change(screen.getByDisplayValue('Select an SAE…'), { target: { value: 'sae_11' } });
    fireEvent.click(screen.getByRole('button', { name: /Estimate/ }));

    await waitFor(() => expect(circuitsApi.createCapture).toHaveBeenCalledTimes(1));
    expect(vi.mocked(circuitsApi.createCapture).mock.calls[0][0].gpu).toBe('auto');

    fireEvent.change(screen.getByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    fireEvent.click(await screen.findByRole('button', { name: /Run capture/ }));
    await waitFor(() => expect(circuitsApi.confirmCapture).toHaveBeenCalledTimes(1));
    expect(circuitsApi.confirmCapture).toHaveBeenCalledWith('cap-1', { gpu: RTX_3090_UUID });
  });
});

describe('Discovery → attribution', () => {
  it('the attribution pass sends the chosen card', async () => {
    render(<CircuitsPanel />);
    openTab('Discovery');
    fireEvent.click(await screen.findByText('run-1234'));
    fireEvent.change(await screen.findByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: /Run attribution pass/ }));

    await waitFor(() => expect(circuitsApi.startAttribution).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startAttribution).toHaveBeenCalledWith('run-12345678', { gpu: RTX_3090_UUID });
  });
});

describe('Validation', () => {
  async function selectRun() {
    render(<CircuitsPanel />);
    openTab('Validation');
    const picker = await screen.findByDisplayValue('Select a completed discovery run…');
    await screen.findByRole('option', { name: /3 candidates/ });
    fireEvent.change(picker, { target: { value: 'run-12345678' } });
    await waitFor(() => expect(circuitsApi.getDiscovery).toHaveBeenCalled());
  }

  it('sends gpu "auto" by default', async () => {
    await selectRun();
    const button = screen.getByRole('button', { name: /Validate top-K/ });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(circuitsApi.startValidation).toHaveBeenCalledTimes(1));
    const [runId, body] = vi.mocked(circuitsApi.startValidation).mock.calls[0];
    expect(runId).toBe('run-12345678');
    expect(body.gpu).toBe('auto');
    expect(body.k).toBe(20);
  });

  it('sends the chosen card UUID', async () => {
    await selectRun();
    fireEvent.change(screen.getByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    const button = screen.getByRole('button', { name: /Validate top-K/ });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(circuitsApi.startValidation).toHaveBeenCalledTimes(1));
    expect(vi.mocked(circuitsApi.startValidation).mock.calls[0][1].gpu).toBe(RTX_3090_UUID);
  });
});

describe('Circuit detail — faithfulness and calibration', () => {
  async function openCircuit() {
    render(<CircuitsPanel />);
    fireEvent.click(await screen.findByText('Humor circuit'));
    await screen.findByRole('button', { name: /Run faithfulness/ });
  }

  it('faithfulness sends the chosen card', async () => {
    await openCircuit();
    fireEvent.change(screen.getByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: /Run faithfulness/ }));

    await waitFor(() => expect(circuitsApi.startFaithfulness).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startFaithfulness).toHaveBeenCalledWith('crc-1', {
      mode: 'necessity',
      gpu: RTX_3090_UUID,
    });
  });

  it('calibration sends gpu "auto" by default', async () => {
    await openCircuit();
    fireEvent.change(screen.getByLabelText('Judge endpoint'), { target: { value: 'http://millm/v1' } });
    fireEvent.change(screen.getByLabelText('Judge model'), { target: { value: 'gemma' } });
    fireEvent.click(screen.getByRole('button', { name: /Calibrate strength/ }));

    await waitFor(() => expect(circuitsApi.startCalibration).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startCalibration).toHaveBeenCalledWith('crc-1', {
      judge_endpoint: 'http://millm/v1',
      judge_model: 'gemma',
      gpu: 'auto',
    });
  });
});

describe('"All GPUs" only where the circuit run can split (multi-GPU Phase 2)', () => {
  const RTX_3080_TI_UUID = 'GPU-f47ba814-49a2-603f-3595-275284140251';
  const optionValues = (picker: HTMLElement) =>
    Array.from((picker as HTMLSelectElement).options).map((o) => o.value);

  beforeEach(() => {
    useSystemMonitorStore.setState({
      gpuList: {
        gpu_count: 2,
        gpus: [
          { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: RTX_3080_TI_UUID, total_memory_gb: 12 },
          { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 },
        ],
      },
    } as never);
  });

  it('capture offers it, and confirm sends gpu "all"', async () => {
    render(<CircuitsPanel />);
    openTab('Capture');
    fireEvent.change(screen.getByDisplayValue('Select a dataset…'), { target: { value: 'ds_owt' } });
    fireEvent.change(screen.getByDisplayValue('Select an SAE…'), { target: { value: 'sae_11' } });
    fireEvent.click(screen.getByRole('button', { name: /Estimate/ }));
    await waitFor(() => expect(circuitsApi.createCapture).toHaveBeenCalledTimes(1));

    const picker = screen.getByLabelText(/^GPU/);
    expect(optionValues(picker)).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(await screen.findByRole('button', { name: /Run capture/ }));

    await waitFor(() => expect(circuitsApi.confirmCapture).toHaveBeenCalledTimes(1));
    expect(circuitsApi.confirmCapture).toHaveBeenCalledWith('cap-1', { gpu: 'all' });
  });

  it('the attribution pass offers it and sends gpu "all"', async () => {
    render(<CircuitsPanel />);
    openTab('Discovery');
    fireEvent.click(await screen.findByText('run-1234'));

    const picker = await screen.findByLabelText(/^GPU/);
    expect(optionValues(picker)).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(screen.getByRole('button', { name: /Run attribution pass/ }));

    await waitFor(() => expect(circuitsApi.startAttribution).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startAttribution).toHaveBeenCalledWith('run-12345678', { gpu: 'all' });
  });

  it('validation offers it and sends gpu "all"', async () => {
    render(<CircuitsPanel />);
    openTab('Validation');
    const runPicker = await screen.findByDisplayValue('Select a completed discovery run…');
    await screen.findByRole('option', { name: /3 candidates/ });
    fireEvent.change(runPicker, { target: { value: 'run-12345678' } });
    await waitFor(() => expect(circuitsApi.getDiscovery).toHaveBeenCalled());

    const picker = screen.getByLabelText(/^GPU/);
    expect(optionValues(picker)).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(picker, { target: { value: 'all' } });
    const button = screen.getByRole('button', { name: /Validate top-K/ });
    await waitFor(() => expect(button).not.toBeDisabled());
    fireEvent.click(button);

    await waitFor(() => expect(circuitsApi.startValidation).toHaveBeenCalledTimes(1));
    expect(vi.mocked(circuitsApi.startValidation).mock.calls[0][1].gpu).toBe('all');
  });

  it('the faithfulness & calibration picker offers it, and faithfulness sends gpu "all"', async () => {
    render(<CircuitsPanel />);
    fireEvent.click(await screen.findByText('Humor circuit'));
    await screen.findByRole('button', { name: /Run faithfulness/ });

    const picker = screen.getByLabelText(/^GPU/);
    expect(optionValues(picker)).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(screen.getByRole('button', { name: /Run faithfulness/ }));

    await waitFor(() => expect(circuitsApi.startFaithfulness).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startFaithfulness).toHaveBeenCalledWith('crc-1', { mode: 'necessity', gpu: 'all' });
  });

  it('calibration, from the same picker, sends gpu "all"', async () => {
    render(<CircuitsPanel />);
    fireEvent.click(await screen.findByText('Humor circuit'));
    await screen.findByRole('button', { name: /Calibrate strength/ });

    fireEvent.change(screen.getByLabelText(/^GPU/), { target: { value: 'all' } });
    fireEvent.change(screen.getByLabelText('Judge endpoint'), { target: { value: 'http://millm/v1' } });
    fireEvent.change(screen.getByLabelText('Judge model'), { target: { value: 'gemma' } });
    fireEvent.click(screen.getByRole('button', { name: /Calibrate strength/ }));

    await waitFor(() => expect(circuitsApi.startCalibration).toHaveBeenCalledTimes(1));
    expect(circuitsApi.startCalibration).toHaveBeenCalledWith('crc-1', {
      judge_endpoint: 'http://millm/v1',
      judge_model: 'gemma',
      gpu: 'all',
    });
  });
});
