/**
 * Component tests for the Feature 017 validation surface.
 *
 * Copy discipline is the point here: validation IS the rung-2 causal tier, so a
 * PASS may say "causally validated (rung 2)"; a tested-and-failed edge must say
 * "tested, did not validate" and must NEVER carry a causal claim. Also pins
 * that the ManifestDrawer renders per-edge effect sizes.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ValidationResults } from './CircuitsPanel';
import { ManifestDrawer } from '../circuits/ManifestDrawer';
import type {
  DiscoveryRun, DiscoveryCandidate, ValidationManifest,
} from '../../types/circuits';

// The drawer fetches the manifest through the api client.
vi.mock('../../api/circuits', () => ({
  circuitsApi: {
    getManifest: vi.fn(),
    reproduceManifest: vi.fn(),
  },
}));
import { circuitsApi } from '../../api/circuits';

function makeCandidate(overrides: Partial<DiscoveryCandidate> = {}): DiscoveryCandidate {
  return {
    up: { layer: 4, feature_idx: 11 },
    down: { layer: 6, feature_idx: 22 },
    granularity: 'feature',
    stats: {},
    replicated_heldout: false,
    ...overrides,
  } as DiscoveryCandidate;
}

function makeRun(
  candidates: DiscoveryCandidate[],
  report: DiscoveryRun['report'] = null,
): DiscoveryRun {
  return {
    id: 'run-1',
    capture_run_id: 'cap-1',
    status: 'completed',
    progress: 100,
    params: null,
    report,
    candidate_count: candidates.length,
    candidates,
    validation_status: 'completed',
    created_at: '',
    updated_at: '',
  } as DiscoveryRun;
}

describe('ValidationResults (copy discipline)', () => {
  it('renders a causally-validated rung-2 chip for a PASS edge', () => {
    // No report → only the per-edge chip carries the rung-2 phrase.
    const run = makeRun([
      makeCandidate({
        validation: { ordering: 'coact', effect_size: 0.42, passed: true, manifest_id: 'man-1' },
        validated_rung: 2,
      }),
    ]);
    render(<ValidationResults run={run} onOpenManifest={() => {}} />);
    expect(screen.getByText(/causally validated \(rung 2\)/)).toBeInTheDocument();
  });

  it('shows the batch banner survival summary from report.validation', () => {
    const run = makeRun(
      [makeCandidate({
        validation: { ordering: 'coact', effect_size: 0.42, passed: true, manifest_id: 'man-1' },
        validated_rung: 2,
      })],
      { validation: { ordering: 'coact', k: 2, survival: 0.5, passed: 1, manifest_id: 'man-1' } } as unknown as DiscoveryRun['report'],
    );
    render(<ValidationResults run={run} onOpenManifest={() => {}} />);
    expect(screen.getByText(/1\/2 edges causally validated/)).toBeInTheDocument();
  });

  it('renders both-orderings survival and the attribution re-ranking uplift', () => {
    const run = makeRun(
      [makeCandidate({
        validation: { ordering: 'attr', effect_size: 0.42, passed: true, manifest_id: 'man-attr' },
        validated_rung: 2,
      })],
      {
        validation: {
          ordering: 'attr', k: 4, survival: 0.75, passed: 3, manifest_id: 'man-attr',
          by_ordering: {
            coact: { survival: 0.5, passed: 2, k: 4, manifest_id: 'man-coact' },
            attr: { survival: 0.75, passed: 3, k: 4, manifest_id: 'man-attr' },
          },
          uplift: 0.25,
        },
      } as unknown as DiscoveryRun['report'],
    );
    render(<ValidationResults run={run} onOpenManifest={() => {}} />);
    // both survival rates + the uplift number appear prominently
    expect(screen.getByText(/attribution re-ranking uplift: \+25%/)).toBeInTheDocument();
    expect(screen.getByText(/raised the causal survival rate/)).toBeInTheDocument();
  });

  it('hints to validate the other ordering when only one exists', () => {
    const run = makeRun(
      [makeCandidate({
        validation: { ordering: 'coact', effect_size: 0.42, passed: true, manifest_id: 'man-coact' },
        validated_rung: 2,
      })],
      {
        validation: {
          ordering: 'coact', k: 4, survival: 0.5, passed: 2, manifest_id: 'man-coact',
          by_ordering: { coact: { survival: 0.5, passed: 2, k: 4, manifest_id: 'man-coact' } },
        },
      } as unknown as DiscoveryRun['report'],
    );
    render(<ValidationResults run={run} onOpenManifest={() => {}} />);
    expect(screen.getByText(/Validate the attribution ordering too/)).toBeInTheDocument();
  });

  it('renders "tested, did not validate" and NO causal claim for a tested_and_failed edge', () => {
    const run = makeRun([
      makeCandidate({
        validation: { ordering: 'coact', effect_size: 0.01, passed: false, manifest_id: 'man-1' },
        tested_and_failed_history: [{ ordering: 'coact', reason: 'ES below null p95' }],
      }),
    ]);
    render(<ValidationResults run={run} onOpenManifest={() => {}} />);
    expect(screen.getByText(/tested, did not validate/)).toBeInTheDocument();
    // A failed edge must never claim to be causally validated.
    expect(screen.queryByText(/causally validated/)).not.toBeInTheDocument();
  });
});

describe('ManifestDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders per-edge effect sizes from the manifest payload', async () => {
    const manifest: ValidationManifest = {
      id: 'man-1',
      kind: 'edge_batch',
      discovery_run_id: 'run-1',
      circuit_id: null,
      parent_manifest_id: null,
      payload: {
        intervention: { kind: 'directional_suppression', baseline: 'zero' },
        ordering: 'coact',
        k: 2,
        seeds: [0],
        survival: 0.5,
        edges: [
          {
            up: { layer: 4, feature_idx: 11 },
            down: { layer: 6, feature_idx: 22 },
            effect_size: 0.42731,
            sign_consistency: 0.9,
            sigma_d: 0.1,
            n_prompts: 8,
            null_percentile_value: 0.12,
            verdict: { passed: true, reason: 'ES > null p95, sign-consistent' },
            rung: 2,
            tested_and_failed: false,
          },
        ],
      },
      created_at: '',
    };
    (circuitsApi.getManifest as ReturnType<typeof vi.fn>).mockResolvedValue(manifest);

    render(<ManifestDrawer manifestId="man-1" onClose={() => {}} />);

    await waitFor(() => {
      expect(screen.getByText('0.427')).toBeInTheDocument();
    });
    // the verdict.reason string renders verbatim
    expect(screen.getByText(/ES > null p95, sign-consistent/)).toBeInTheDocument();
  });
});
