/**
 * The empty Circuits list points at the paths that exist, and a discovery run
 * builds a circuit from the panel.
 *
 * The empty state said circuits "arrive from discovery runs (coming with the
 * mining feature)" — discovery shipped in 016, and the REST/MCP producer
 * `POST /circuit-discovery/{id}/build-circuit` existed, but nothing in the UI
 * called it, so a GUI user could run discovery and never get a circuit out of
 * it. The empty state now opens the Discovery tab and the import dialog, and
 * a completed run has a Build circuit action.
 *
 * MUTATION CONTROLS (2026-10-18, each restored by sha256):
 *   C1  drop `onOpenDiscovery={() => setTab('discovery')}` from CircuitsPanel  -> 5 red
 *   C2  send `candidate_keys: buildable` (ignore the selection)                -> 1 red
 *   C3  delete the buildFromDiscovery call in buildCircuit                     -> 3 red
 *   C4  drop `void fetchCircuits();` after a build (list stays stale)          -> 1 red
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../../api/circuits', () => ({
  circuitsApi: {
    list: vi.fn(),
    get: vi.fn(),
    exportUrl: vi.fn(() => '#'),
    listCaptures: vi.fn(),
    listDiscoveries: vi.fn(),
    getDiscovery: vi.fn(),
    buildFromDiscovery: vi.fn(),
    importDefinition: vi.fn(),
    listManifests: vi.fn(),
  },
}));

import { circuitsApi } from '../../api/circuits';
import { CircuitsPanel } from './CircuitsPanel';
import type { DiscoveryRun } from '../../types/circuits';

const api = circuitsApi as unknown as Record<string, ReturnType<typeof vi.fn>>;

const node = (layer: number, feature_idx: number | null, cluster?: string) => ({
  layer, feature_idx, cluster_profile_id: cluster ?? null, cluster_name: null,
});
const cand = (up: ReturnType<typeof node>, down: ReturnType<typeof node>) => ({
  up, down, granularity: 'feature', stats: { pmi: 1.2, support: 30, null_pct: 99 },
  replicated_heldout: true, attribution: null, orderings: null,
});

const RUN = {
  id: 'run-abcdef12',
  capture_run_id: 'cap-1',
  status: 'completed',
  progress: 100,
  params: null,
  report: null,
  candidate_count: 3,
  candidates: [
    cand(node(11, 5), node(12, 9)),
    cand(node(12, 9), node(13, 4)),
    // A cluster endpoint has no feature_idx: it cannot be a build key.
    cand(node(13, null, 'profile-1234567890'), node(14, 1)),
  ],
  attribution_status: null,
  validation_status: null,
  created_at: '',
  updated_at: '',
} as unknown as DiscoveryRun;

beforeEach(() => {
  vi.clearAllMocks();
  api.list.mockResolvedValue({ circuits: [], total: 0 });
  api.listCaptures.mockResolvedValue({ captures: [] });
  api.listDiscoveries.mockResolvedValue({ discoveries: [RUN] });
  api.getDiscovery.mockResolvedValue(RUN);
  api.listManifests.mockResolvedValue({ manifests: [] });
  api.buildFromDiscovery.mockResolvedValue({ id: 'crc-new', name: 'My circuit' });
});

async function openRun() {
  fireEvent.click(await screen.findByText('Open discovery runs'));
  fireEvent.click(await screen.findByText('run-abcd'));
  return screen.findByTestId('build-circuit');
}

describe('Circuits empty state', () => {
  it('names the real paths and no unfinished feature', async () => {
    render(<CircuitsPanel />);
    expect(await screen.findByText('No circuits yet')).toBeInTheDocument();
    expect(screen.queryByText(/coming with/i)).not.toBeInTheDocument();
    expect(screen.getByText('Import a definition')).toBeInTheDocument();
    expect(screen.getByText('build_circuit_from_discovery')).toBeInTheDocument();
  });

  it('"Open discovery runs" switches to the Discovery tab', async () => {
    render(<CircuitsPanel />);
    fireEvent.click(await screen.findByText('Open discovery runs'));
    expect(await screen.findByText('Discovery runs')).toBeInTheDocument();
    expect(screen.getByText('Run discovery')).toBeInTheDocument();
  });
});

describe('Build circuit from a discovery run', () => {
  it('builds from every feature-level candidate by default, once, with the payload', async () => {
    render(<CircuitsPanel />);
    await openRun();
    fireEvent.change(screen.getByLabelText('Circuit name'), { target: { value: 'My circuit' } });
    fireEvent.click(screen.getByText('Build circuit from all 2 candidates'));

    await waitFor(() => expect(api.buildFromDiscovery).toHaveBeenCalledTimes(1));
    expect(api.buildFromDiscovery).toHaveBeenCalledWith('run-abcdef12', {
      name: 'My circuit',
      candidate_keys: [[11, 5, 12, 9], [12, 9, 13, 4]],
    });
    expect(await screen.findByRole('status')).toHaveTextContent('Built “My circuit” with 2 edges.');
  });

  it('builds from the ticked subset only, with a default name', async () => {
    render(<CircuitsPanel />);
    await openRun();
    fireEvent.click(screen.getByLabelText('Select L12:f9 → L13:f4'));
    fireEvent.click(screen.getByText('Build circuit from 1 selected'));

    await waitFor(() => expect(api.buildFromDiscovery).toHaveBeenCalledTimes(1));
    expect(api.buildFromDiscovery).toHaveBeenCalledWith('run-abcdef12', {
      name: 'Discovery run-abcd',
      candidate_keys: [[12, 9, 13, 4]],
    });
    // The cluster-endpoint row offers no checkbox.
    expect(screen.queryByLabelText(/Select L13:profile-/)).not.toBeInTheDocument();
  });

  it('after a build, "Open in Circuits" shows the refreshed list', async () => {
    render(<CircuitsPanel />);
    await openRun();
    const listCallsBefore = api.list.mock.calls.length;
    api.list.mockResolvedValue({
      circuits: [{
        id: 'crc-new', name: 'Fresh circuit', rung: 0, rung_language: 'mined',
        rung_next_step: null, promoted: false, member_count: 3, edge_count: 2,
        layers: [11, 12, 13], updated_at: '',
      }],
      total: 1,
    });
    fireEvent.click(screen.getByText('Build circuit from all 2 candidates'));
    await waitFor(() => expect(api.list.mock.calls.length).toBeGreaterThan(listCallsBefore));
    fireEvent.click(await screen.findByText('Open in Circuits'));
    expect(await screen.findByText('Fresh circuit')).toBeInTheDocument();
  });

  it('a refused build says why', async () => {
    api.buildFromDiscovery.mockRejectedValueOnce(new Error('No matching candidates selected'));
    render(<CircuitsPanel />);
    await openRun();
    fireEvent.click(screen.getByText('Build circuit from all 2 candidates'));
    expect(await screen.findByText('Build failed: No matching candidates selected')).toBeInTheDocument();
  });
});
