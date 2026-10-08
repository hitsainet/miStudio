/**
 * Reproducing a manifest re-runs its interventions on a GPU, so the request
 * carries the card the user chose ("auto" by default).
 *
 * "All GPUs" (multi-GPU Phase 2) is offered for a validation (edge batch)
 * manifest, whose reproduction runs split. This drawer reproduces only through
 * the validation endpoint, so a calibration manifest gets no picker and no
 * Reproduce here (calibration has its own reproduce endpoint, not called from
 * the UI) — it is never offered "All GPUs" by this drawer.
 *
 * MUTATION CONTROL: send `reproduceManifest(manifestId, { gpu: 'auto' })`
 * regardless of the picker -> the UUID test fails; drop the body -> both fail.
 *
 * "All GPUs" CONTROLS (2026-09-14; each restored byte-identically, sha256):
 *   F14 allowSplit kind test flipped (=== -> !==)      -> "a validation (edge batch) manifest offers it"
 *   F15 picker rendered for every kind, allowSplit left
 *       kind-derived                                   -> "a calibration manifest is never offered it"
 *                                                         (red on the picker-absent assertion; "all"
 *                                                         stayed absent — the kind test held)
 *   F16 allowSplit forced on, render guard intact      -> GREEN, by design: the picker is not rendered
 *                                                         for a calibration manifest, so the prop is
 *                                                         unobservable there (an equivalent mutant)
 *   F17 F15 + F16 together                             -> "a calibration manifest is never offered it"
 *                                                         (red on the option[value="all"] assertion)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';

vi.mock('../../api/circuits', () => ({
  circuitsApi: {
    getManifest: vi.fn(),
    reproduceManifest: vi.fn(),
    listManifests: vi.fn().mockResolvedValue({ manifests: [] }),
  },
}));

import { circuitsApi } from '../../api/circuits';
import { ManifestDrawer } from './ManifestDrawer';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import type { ValidationManifest } from '../../types/circuits';

const RTX_3080_TI_UUID = 'GPU-f47ba814-49a2-603f-3595-275284140251';
const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';

const MANIFEST = {
  id: 'man-1',
  kind: 'edge_batch',
  discovery_run_id: 'run-1',
  circuit_id: null,
  parent_manifest_id: null,
  payload: { edges: [], seeds: [0] },
  created_at: '',
} as unknown as ValidationManifest;

/** The backend's calibration manifest kind (endpoints/circuits.py reproduce_calibration). */
const CALIBRATION_MANIFEST = {
  id: 'cman-1',
  kind: 'calibration',
  discovery_run_id: null,
  circuit_id: 'crc-1',
  parent_manifest_id: null,
  payload: { gpu: 'auto' },
  created_at: '',
} as unknown as ValidationManifest;

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(circuitsApi.getManifest).mockResolvedValue(MANIFEST);
  vi.mocked(circuitsApi.reproduceManifest).mockResolvedValue({
    reproduce_of: 'man-1', task_id: 't', status: 'queued',
  });
  useSystemMonitorStore.setState({
    gpuList: {
      gpu_count: 1,
      gpus: [{ gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 }],
    },
  } as never);
});

describe('ManifestDrawer reproduce', () => {
  it('sends gpu "auto" by default', async () => {
    render(<ManifestDrawer manifestId="man-1" onClose={() => {}} />);
    fireEvent.click(await screen.findByRole('button', { name: /Reproduce/ }));
    await waitFor(() => expect(circuitsApi.reproduceManifest).toHaveBeenCalledTimes(1));
    expect(circuitsApi.reproduceManifest).toHaveBeenCalledWith('man-1', { gpu: 'auto' });
  });

  it('sends the chosen card UUID', async () => {
    render(<ManifestDrawer manifestId="man-1" onClose={() => {}} />);
    fireEvent.change(await screen.findByLabelText(/^GPU/), { target: { value: RTX_3090_UUID } });
    fireEvent.click(screen.getByRole('button', { name: /Reproduce/ }));
    await waitFor(() => expect(circuitsApi.reproduceManifest).toHaveBeenCalledTimes(1));
    expect(circuitsApi.reproduceManifest).toHaveBeenCalledWith('man-1', { gpu: RTX_3090_UUID });
  });
});

describe('"All GPUs" for a reproduction depends on the manifest kind (multi-GPU Phase 2)', () => {
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

  it('a validation (edge batch) manifest offers it and sends gpu "all"', async () => {
    render(<ManifestDrawer manifestId="man-1" onClose={() => {}} />);
    const picker = (await screen.findByLabelText(/^GPU/)) as HTMLSelectElement;

    expect(Array.from(picker.options).map((o) => o.value))
      .toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(picker, { target: { value: 'all' } });
    fireEvent.click(screen.getByRole('button', { name: /Reproduce/ }));

    await waitFor(() => expect(circuitsApi.reproduceManifest).toHaveBeenCalledTimes(1));
    expect(circuitsApi.reproduceManifest).toHaveBeenCalledWith('man-1', { gpu: 'all' });
  });

  it('a calibration manifest is never offered it', async () => {
    vi.mocked(circuitsApi.getManifest).mockResolvedValue(CALIBRATION_MANIFEST);
    const { container } = render(<ManifestDrawer manifestId="cman-1" onClose={() => {}} />);
    // Loaded: the kind is on screen, so the assertions below read a rendered drawer.
    await screen.findByText('calibration');

    // Split first, then presence, so a picker that renders here with
    // allowSplit forced on fails on "all" rather than on the picker existing.
    expect(container.querySelector('option[value="all"]')).toBeNull();
    expect(screen.queryByLabelText(/^GPU/)).toBeNull();
  });
});
