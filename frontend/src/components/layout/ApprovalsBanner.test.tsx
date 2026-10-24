/**
 * The agent-approvals banner renders every approvable tool (Feature 034, FR-20).
 *
 * A model load evicts the model miLLM serves, so its line must name the model
 * AND the eviction. A tool the banner does not know must still be readable — a
 * request the operator cannot read is one they approve blind.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

const listApprovals = vi.fn();
vi.mock('../../api/featureGroups', () => ({
  listApprovals: (...args: unknown[]) => listApprovals(...args),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
}));
const getModel = vi.fn();
vi.mock('../../api/models', () => ({ getModel: (...args: unknown[]) => getModel(...args) }));
const listDatasets = vi.fn();
vi.mock('../../api/probeMonitors', () => ({
  probeMonitorsApi: { listDatasets: (...args: unknown[]) => listDatasets(...args) },
}));
const getGPUList = vi.fn();
vi.mock('../../api/system', () => ({ getGPUList: (...args: unknown[]) => getGPUList(...args) }));
vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: false,
  }),
}));

import { ApprovalsBanner, renderSummary } from './ApprovalsBanner';

function request(tool_name: string, payload: Record<string, unknown>): ApprovalRequest {
  return {
    id: `id-${tool_name}`, tool_name, payload, status: 'pending', reason: null,
    steering_task_id: null, created_at: null, resolved_at: null,
  };
}

beforeEach(() => {
  listApprovals.mockReset();
  getModel.mockReset();
  listDatasets.mockReset();
  getGPUList.mockReset();
});

function view(id: string, name: string, role: string, positive: number, negative: number) {
  return {
    id, name, role, dataset_id: 'ds', config: null, split: null, input_column: 'text',
    label_column: 'label', label_mapping: {}, keyword_filter: null, pair_column: null,
    distribution: null, created_at: '2026-10-20T00:00:00Z',
    counts: { positive, negative, excluded: 0, filtered_out: 0, unparseable: 0 },
  };
}

const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
const GPUS = [
  { gpu_id: 0, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, pci_bus_id: '', driver_version: '',
    cuda_version: '', compute_capability: '', total_memory_gb: 24 },
  { gpu_id: 1, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: 'GPU-f47ba814-49a2-603f-3595-275284140251',
    pci_bus_id: '', driver_version: '', cuda_version: '', compute_capability: '', total_memory_gb: 12 },
];

describe('ApprovalsBanner', () => {
  it('renders a model-load request with the model and the eviction', async () => {
    listApprovals.mockResolvedValue({
      approvals: [request('millm_load_model', { model_id: 7 })], total: 1,
    });
    render(<ApprovalsBanner />);
    expect(
      await screen.findByText('Load miLLM model #7 — evicts the resident model'),
    ).toBeInTheDocument();
    expect(screen.getByText('1 agent request awaiting approval')).toBeInTheDocument();
    expect(listApprovals).toHaveBeenCalledWith('pending');
  });

  it('keeps the steering line', async () => {
    listApprovals.mockResolvedValue({
      approvals: [
        request('steer_compare', { prompt: 'hello', selected_features: [1, 2] }),
        request('millm_load_model', { model_id: 3 }),
      ],
      total: 2,
    });
    render(<ApprovalsBanner />);
    expect(await screen.findByText('2 features · "hello"')).toBeInTheDocument();
    expect(screen.getByText('2 agent requests awaiting approval')).toBeInTheDocument();
  });

  it('renders NOTHING when the queue is empty — no wrapper, no padding', async () => {
    listApprovals.mockResolvedValue({ approvals: [], total: 0 });
    const { container } = render(<ApprovalsBanner />);
    await waitFor(() => expect(listApprovals).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it('has no one-line summary for an UNKNOWN tool — the banner shows its arguments instead', () => {
    expect(renderSummary(request('future_tool', { a: 1 }))).toBeNull();
  });

  // F-5: an approval for a GPU probe run showed raw JSON cut at 120 characters, which could
  // hide the card and the evaluation/calibration views. NEGATIVE CONTROLS (2026-10-20):
  // deleting the start_probe_training_run branch in RequestDetails -> the probe-run tests fail;
  // reinstating `.slice(0, 120)` on the fallback -> the fallback test fails.
  it('names the model, the GPU by name AND identifier, and every view with its row counts', async () => {
    listApprovals.mockResolvedValue({
      approvals: [
        request('start_probe_training_run', {
          model_id: 'm_abc',
          train_dataset_id: 'pdv_train',
          eval_dataset_ids: ['pdv_eval1', 'pdv_eval2'],
          calibration_dataset_id: 'pdv_cal',
          gpu: RTX_3090_UUID,
          config: { layers: [11, 12], target_fpr: 0.01 },
        }),
      ],
      total: 1,
    });
    getModel.mockResolvedValue({ id: 'm_abc', name: 'LFM2.5 1.2B Instruct' });
    listDatasets.mockResolvedValue([
      view('pdv_train', 'Deception train', 'train', 1200, 1300),
      view('pdv_eval1', 'Deception eval (in distribution)', 'eval', 200, 210),
      view('pdv_eval2', 'Insider trading (out of distribution)', 'eval', 50, 60),
      view('pdv_cal', 'Benign chat calibration', 'calibration', 0, 5000),
    ]);
    getGPUList.mockResolvedValue({ gpu_count: 2, gpus: GPUS });
    render(<ApprovalsBanner />);
    const summary = await screen.findByTestId('probe-run-summary');
    await waitFor(() => expect(summary).toHaveTextContent('“LFM2.5 1.2B Instruct”'));
    expect(summary).toHaveTextContent(`Graphics card: NVIDIA GeForce RTX 3090 (${RTX_3090_UUID})`);
    expect(summary).toHaveTextContent('Training set: “Deception train” — 1,200 positive, 1,300 negative rows');
    expect(summary).toHaveTextContent('“Deception eval (in distribution)” — 200 positive, 210 negative rows');
    expect(summary).toHaveTextContent('“Insider trading (out of distribution)” — 50 positive, 60 negative rows');
    expect(summary).toHaveTextContent('Calibration set: “Benign chat calibration” — 0 positive, 5,000 negative rows');
    expect(summary).toHaveTextContent('Layers: [11,12] · Target false-positive rate: 0.01');
    expect(summary).toHaveTextContent('graphics processing unit (GPU)');
    expect(getModel).toHaveBeenCalledTimes(1);
    expect(getModel).toHaveBeenCalledWith('m_abc');
  });

  // Operator, 2026-10-21: four pending runs on one model read alike ("Train probes on
  // LFM2.5…") and could not be told apart. The behaviour leads each request now.
  it('leads each probe-run request with what it detects, so several runs on one model differ', async () => {
    listApprovals.mockResolvedValue({
      approvals: [
        request('start_probe_training_run', {
          model_id: 'm_abc', train_dataset_id: 'pdv_train', eval_dataset_ids: [], gpu: 'auto',
          behaviour: 'dlp-ssn-us', behaviour_label: 'US Social Security Number (DLP high-confidence rule)',
        }),
        request('start_probe_training_run', {
          model_id: 'm_abc', train_dataset_id: 'pdv_train', eval_dataset_ids: [], gpu: 'auto',
          behaviour: 'dlp-iban',
        }),
        request('start_probe_training_run', {
          model_id: 'm_abc', train_dataset_id: 'pdv_train', eval_dataset_ids: [], gpu: 'auto',
        }),
      ],
      total: 3,
    });
    getModel.mockResolvedValue({ id: 'm_abc', name: 'LFM2.5 1.2B Instruct' });
    listDatasets.mockResolvedValue([]);
    getGPUList.mockResolvedValue({ gpu_count: 2, gpus: GPUS });
    render(<ApprovalsBanner />);
    const lines = await screen.findAllByTestId('probe-run-behaviour');
    expect(lines.map((l) => l.textContent)).toEqual([
      'Detects: US Social Security Number (DLP high-confidence rule) (behaviour key “dlp-ssn-us”)',
      'Detects: dlp-iban',
      'Detects: no behaviour given — tell these probes apart by their training set below',
    ]);
  });

  it('says plainly when a probe-run card or view cannot be looked up, and names "auto"', async () => {
    listApprovals.mockResolvedValue({
      approvals: [
        request('start_probe_training_run', {
          model_id: 'm_abc', train_dataset_id: 'pdv_missing', eval_dataset_ids: [], gpu: 'auto',
        }),
      ],
      total: 1,
    });
    getModel.mockRejectedValue(new Error('404'));
    listDatasets.mockRejectedValue(new Error('down'));
    getGPUList.mockRejectedValue(new Error('down'));
    render(<ApprovalsBanner />);
    const summary = await screen.findByTestId('probe-run-summary');
    await waitFor(() => expect(listDatasets).toHaveBeenCalled());
    expect(summary).toHaveTextContent('a model the banner could not look up (m_abc)');
    expect(summary).toHaveTextContent('Automatic — whichever card has the most free memory');
    expect(summary).toHaveTextContent('a probe dataset view the banner could not look up (pdv_missing)');
    expect(summary).toHaveTextContent('Evaluation sets: none');
    expect(summary).toHaveTextContent('Calibration set: none');
    expect(summary).toHaveTextContent('Settings: the defaults');
  });

  it('makes no lookups when no probe run is pending', async () => {
    listApprovals.mockResolvedValue({
      approvals: [request('millm_load_model', { model_id: 7 })], total: 1,
    });
    render(<ApprovalsBanner />);
    await screen.findByText('Load miLLM model #7 — evicts the resident model');
    expect(getModel).not.toHaveBeenCalled();
    expect(listDatasets).not.toHaveBeenCalled();
    expect(getGPUList).not.toHaveBeenCalled();
  });

  it('shows EVERY argument of an unknown tool in full — nothing cut at 120 characters', async () => {
    const tail = 'the part past one hundred and twenty characters: GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
    const long = `${'x'.repeat(150)} ${tail}`;
    listApprovals.mockResolvedValue({
      approvals: [request('future_gpu_tool', { notes: long, gpu: 'GPU-f47ba814', count: 3 })],
      total: 1,
    });
    render(<ApprovalsBanner />);
    const args = await screen.findByTestId('approval-arguments');
    expect(args).toHaveAttribute('open');
    expect(args).toHaveTextContent('All 3 arguments the agent sent');
    expect(args).toHaveTextContent(tail);
    expect(args).toHaveTextContent('GPU-f47ba814');
    expect(args).toHaveTextContent('count3');
  });
});
