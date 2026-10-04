/**
 * Degenerate steered output must WRAP, not overflow its card.
 *
 * A high steering coefficient produces exactly this: long unbroken character
 * runs with no spaces ("AIRAIRAIRAIR…"). That output is the signal the operator
 * is looking at, so the panel has to render it legibly. It did not:
 * `whitespace-pre-wrap` alone offers no break opportunity inside a run that
 * contains no spaces, so the line ran past the right edge of the result card
 * and put a horizontal scrollbar across the whole results panel.
 *
 * The fix is a styling change, so the class list IS the behaviour — these tests
 * assert the class list of the element that actually renders the text, in every
 * place steered or baseline output is rendered:
 *   - ComparisonResults.renderOutput        (Compare mode + batch, baseline AND steered)
 *   - ComparisonResults.renderMultiStrength (the strength-sweep stack)
 *   - SteeringPanel's Blended card          (side-by-side baseline/combined, and combined-only)
 *
 * `break-all` is deliberately refused: it would break ordinary words mid-word
 * and make non-degenerate output harder to read. `break-words`
 * (overflow-wrap: break-word) only breaks a run that cannot fit on a line.
 *
 * MUTATION CONTROLS (2026-09-27, every one run and verified RED):
 *   C1 drop `break-words` from ComparisonResults renderOutput's <p>
 *        -> 2 failed / 3 passed (the steered and the baseline case)
 *   C2 drop `break-words` from the multi-strength output div
 *        -> 1 failed / 4 passed (the multi-strength stack)
 *   C3 drop `break-words` from SteeringPanel's Blended BASELINE container
 *        -> 1 failed / 4 passed (only the side-by-side case; the combined-only
 *           card has no baseline, which is why it stays green)
 *   C4 drop `break-words` from SteeringPanel's two Blended COMBINED containers
 *        -> 2 failed / 3 passed
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import type { BatchPromptResult, SteeringComparisonResponse, CombinedSteeringResponse } from '../../types/steering';
import type { SAE } from '../../types/sae';

/** The real thing, from the screenshot that prompted this test. */
const DEGENERATE =
  '**LOASWERETNIGGTEHROUNWHTIHONWMEORITNIYOPPNIECTISNTENP[SINTECRNTIOSNEIRTNCIESTIONSIHNCSOTNSOSTSNTSNTSONTSEANTANATNAAITAINTAIANAIATNATAIAITAIIAIR' +
  'AIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIRAIR';
const DEGENERATE_BASELINE = 'BASELINE' + 'X'.repeat(200);

/**
 * A wrapped element must keep the newlines (they carry the markdown structure)
 * AND break inside a space-free run; it must not reach for `break-all`.
 */
function expectWrapsLongRuns(el: HTMLElement) {
  expect(el.className).toContain('whitespace-pre-wrap');
  expect(el.className).toContain('break-words');
  expect(el.className).not.toContain('break-all');
}

// ---------------------------------------------------------------------------
// ComparisonResults — Compare mode, batch mode, multi-strength stack
// ---------------------------------------------------------------------------

import { ComparisonResults } from './ComparisonResults';
import { useSteeringStore } from '../../stores/steeringStore';

const comparison: SteeringComparisonResponse = {
  comparison_id: 'cmp_wrap',
  sae_id: 'sae-1',
  model_id: 'm-1',
  prompt: 'Write a short poem.',
  unsteered: { text: DEGENERATE_BASELINE, metrics: null },
  steered: [
    {
      text: DEGENERATE,
      feature_config: {
        instance_id: 'x',
        feature_idx: 10091,
        layer: 12,
        strength: 240,
        label: 'Blended (2 features)',
        color: 'teal',
        feature_id: null,
      },
      metrics: null,
    },
  ],
  steered_multi: null,
  applied_features: null,
  metrics_summary: null,
  total_time_ms: 1000,
  created_at: new Date().toISOString(),
} as unknown as SteeringComparisonResponse;

const multiStrengthBatch: BatchPromptResult = {
  prompt: 'Write a short poem.',
  promptIndex: 0,
  status: 'completed',
  error: null,
  comparison: {
    ...comparison,
    steered: [],
    steered_multi: [
      {
        feature_config: {
          instance_id: 'x',
          feature_idx: 10091,
          layer: 12,
          strength: 240,
          label: null,
          color: 'teal',
          feature_id: null,
        },
        primary_result: { strength: 240, text: DEGENERATE, metrics: null },
        additional_results: [],
      },
    ],
  },
} as unknown as BatchPromptResult;

describe('steered output wrapping — ComparisonResults', () => {
  beforeEach(() => {
    useSteeringStore.setState({ selectedFeatures: [] } as never);
  });

  it('wraps a degenerate steered run instead of overflowing the card', () => {
    render(<ComparisonResults comparison={comparison} />);
    expectWrapsLongRuns(screen.getByText(DEGENERATE));
  });

  it('wraps the unsteered baseline too', () => {
    render(<ComparisonResults comparison={comparison} />);
    expectWrapsLongRuns(screen.getByText(DEGENERATE_BASELINE));
  });

  it('wraps degenerate output in the multi-strength stack', () => {
    render(<ComparisonResults batchResults={[multiStrengthBatch]} />);
    expectWrapsLongRuns(screen.getByText(DEGENERATE));
  });
});

// ---------------------------------------------------------------------------
// SteeringPanel — the Blended ("Blended (N features)") result card
// ---------------------------------------------------------------------------

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
vi.mock('../steering/ApprovalsBanner', () => ({ ApprovalsBanner: () => <div /> }));
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

const combined = (withBaseline: boolean): CombinedSteeringResponse =>
  ({
    combined_id: 'cmb_wrap',
    sae_id: 'sae-1',
    model_id: 'm-1',
    prompt: 'Write a short poem.',
    combined_output: DEGENERATE,
    features_applied: [
      { feature_idx: 10091, layer: 12, strength: 240, label: null, color: 'teal' },
      { feature_idx: 2262, layer: 12, strength: 0.7, label: null, color: 'blue' },
    ],
    baseline_output: withBaseline ? DEGENERATE_BASELINE : null,
    combined_metrics: null,
    baseline_metrics: null,
    total_steering_strength: 240.7,
    total_time_ms: 1234,
    created_at: new Date().toISOString(),
  }) as unknown as CombinedSteeringResponse;

async function mountBlended(withBaseline: boolean) {
  useSteeringStore.setState({
    selectedSAE: SAE_FIXTURE,
    selectedFeatures: [],
    prompts: ['Write a short poem.'],
    isGenerating: false,
    isCombinedGenerating: false,
    currentComparison: null,
    batchState: null,
    combinedMode: true,
    combinedResults: combined(withBaseline),
    combinedResultsTitle: 'Blended (2 features)',
    recentComparisons: [],
    error: null,
  } as never);
  // `act` so the mount-time steering-mode probe settles inside the render
  // rather than warning after the assertion.
  await act(async () => {
    render(<SteeringPanel />);
  });
}

describe('steered output wrapping — SteeringPanel Blended card', () => {
  it('wraps the combined output when a baseline is shown beside it', async () => {
    await mountBlended(true);
    expect(screen.getByText('Blended (2 features)')).toBeInTheDocument();
    expectWrapsLongRuns(screen.getByText(DEGENERATE));
    expectWrapsLongRuns(screen.getByText(DEGENERATE_BASELINE));
  });

  it('wraps the combined output when it is rendered alone', async () => {
    await mountBlended(false);
    expectWrapsLongRuns(screen.getByText(DEGENERATE));
  });
});
