/**
 * The Steering panel's λ dial follows a calibrated circuit's usable band (Feature 20,
 * files 019): min/max are [onset, cliff], the three calibrated values are marked, and
 * a "Sweet spot" control resets to the default. GPU block 2 (2026-10-17) found the panel
 * ignoring the band — it re-allocated the circuit with its default budget and the
 * output degenerated.
 */

import { describe, it, expect, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import type { SAE } from '../../types/sae';
import { useSteeringStore } from '../../stores/steeringStore';

vi.mock('../../api/steering', () => ({
  getTaskResult: vi.fn(),
  getSteeringModeStatus: vi.fn().mockResolvedValue({ active: true }),
  enterSteeringMode: vi.fn(),
  exitSteeringMode: vi.fn(),
  submitAsyncComparison: vi.fn(),
  submitAsyncCombined: vi.fn(),
  submitAsyncSweep: vi.fn(),
  computeClusterAllocation: vi.fn(),
  cancelTask: vi.fn().mockResolvedValue({}),
  abortComparison: vi.fn(),
  getComparisonStatus: vi.fn(),
  getExperiments: vi.fn(),
  getExperiment: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
  cleanupGPU: vi.fn(),
}));

// STABLE identities: a mock that rebuilds its object per call hands back a new
// `fetchSAEs` on every render, turning an effect keyed on it into a loop.
const SAES_STATE = { saes: [], fetchSAEs: () => {} };
vi.mock('../../stores/saesStore', () => ({
  useSAEsStore: (selector?: (s: typeof SAES_STATE) => unknown) =>
    selector ? selector(SAES_STATE) : SAES_STATE,
}));
const TEMPLATES_STATE = { templates: [], fetchTemplates: () => {}, createTemplate: () => {} };
vi.mock('../../stores/promptTemplatesStore', () => ({
  usePromptTemplatesStore: (selector?: (s: typeof TEMPLATES_STATE) => unknown) =>
    selector ? selector(TEMPLATES_STATE) : TEMPLATES_STATE,
}));
vi.mock('../../hooks/useSteeringWebSocket', () => ({ useSteeringWebSocket: () => ({}) }));

// The panel's own sidebar and config widgets are not under test, and mounting
// them drags their fetches in. ComparisonResults is deliberately NOT stubbed:
// `vi.mock` is file-wide, and the stub would replace the component the first
// half of this file is testing. The Blended card is the only output surface the
// panel renders here anyway, since currentComparison and batchState are null.
vi.mock('../steering/FeatureSelector', () => ({ FeatureSelector: () => <div /> }));
vi.mock('../steering/GenerationConfig', () => ({ GenerationConfig: () => <div /> }));
vi.mock('../steering/PromptListEditor', () => ({ PromptListEditor: () => <div /> }));
vi.mock('../steering/HazardBanner', () => ({ HazardBanner: () => <div /> }));
vi.mock('../steering/ComparisonPreview', () => ({ ComparisonPreview: () => <div /> }));

import { SteeringPanel } from '../panels/SteeringPanel';

const SAE_FIXTURE = {
  id: 'sae-1',
  name: 'Test SAE',
  model_id: 'm-1',
  layer: 12,
  status: 'ready',
} as unknown as SAE;


const BAND = { onset: 0.4, sweet_spot: 0.85, cliff: 1.0, provisional: true };

async function mount(calibration: typeof BAND | null, budget: unknown = null) {
  useSteeringStore.setState({
    selectedSAE: SAE_FIXTURE,
    selectedFeatures: [],
    prompts: ['P'],
    isGenerating: false,
    isCombinedGenerating: false,
    currentComparison: null,
    batchState: null,
    combinedMode: true,
    combinedResults: null,
    recentComparisons: [],
    error: null,
    clusterBudget: null,
    layerBudgets: budget,
    intensity: calibration ? calibration.sweet_spot : 1,
    clusterContext: { group_id: 'crc', display_token: 'crc', circuit_id: 'crc', calibration },
  } as never);
  await act(async () => {
    render(<SteeringPanel />);
  });
}

describe('the λ dial on a calibrated circuit', () => {
  it('is bounded by [onset, cliff] and starts at the sweet spot', async () => {
    await mount(BAND);
    const dial = screen.getByLabelText('Cluster intensity') as HTMLInputElement;
    expect(dial.min).toBe('0.4');
    expect(dial.max).toBe('1');
    expect(dial.value).toBe('0.85');
    expect(screen.getByText('λ 0.85')).toBeInTheDocument();
  });

  it('marks onset, sweet spot and cliff on the track', async () => {
    await mount(BAND);
    const dial = screen.getByLabelText('Cluster intensity') as HTMLInputElement;
    const marks = screen.getByTestId('calibration-band-marks');
    expect(dial.getAttribute('list')).toBe(marks.id);
    expect(Array.from(marks.querySelectorAll('option')).map((o) => [o.getAttribute('label'), o.value]))
      .toEqual([['onset', '0.4'], ['sweet spot', '0.85'], ['cliff', '1']]);
    expect(screen.getByTestId('calibration-band-readout').textContent).toBe('0.4 · 0.85 · 1');
    expect(screen.getByText('provisional')).toBeInTheDocument();
  });

  it('moving the dial past the cliff stops at the cliff; Sweet spot resets it', async () => {
    await mount(BAND);
    const dial = screen.getByLabelText('Cluster intensity') as HTMLInputElement;
    act(() => { fireEvent.change(dial, { target: { value: '1' } }); });
    expect(useSteeringStore.getState().intensity).toBe(1);
    act(() => { useSteeringStore.getState().setIntensity(1.8); });
    expect(useSteeringStore.getState().intensity).toBe(1);
    act(() => { fireEvent.click(screen.getByRole('button', { name: 'Sweet spot' })); });
    expect(useSteeringStore.getState().intensity).toBe(0.85);
  });

  it('shows the dial for a calibrated circuit even with no budget', async () => {
    await mount(BAND, null);
    expect(screen.getByLabelText('Cluster intensity')).toBeInTheDocument();
  });

  it('an uncalibrated budget keeps the 0–2 dial with no band marks', async () => {
    await mount(null, { 13: { B: 1 } });
    const dial = screen.getByLabelText('Cluster intensity') as HTMLInputElement;
    expect([dial.min, dial.max]).toEqual(['0', '2']);
    expect(dial.getAttribute('list')).toBeNull();
    expect(screen.queryByTestId('calibration-band-readout')).toBeNull();
  });
});
