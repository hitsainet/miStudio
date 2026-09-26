/**
 * Probe Monitors panel: reachability, and the honesty rules the components carry.
 *
 * MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
 *   M155  'probe-monitors' removed from PANEL_IDS        -> the registry test fails
 *   M156  the sidebar navItem removed                    -> the nav test fails
 *   M157  the App render branch removed                  -> the reachability test fails
 *   M158  MetricsTable renders 0 for a refusal           -> the refusal test fails
 *   M159  RungChip derives its phrase from the number    -> the wording test fails
 *   M160  SweepGrid renders only the chosen layer        -> the full-grid test fails
 *   M161  TokenTrace shades an unscored token            -> the unscored test fails
 *
 * ⚠ THE REFUSAL TESTS ARE THE POINT. A set that could not be scored must render its
 * REASON — not a 0, not a blank, not a 0.5. Each of those reads as a different fact:
 * "measured and terrible", "not run yet", "measured and chance". None is what happened.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

// Vite's `?raw`, not node's `fs`: the test tsconfig has no node types, so importing
// `fs`/`path` adds three errors to a DOWN-ONLY type-check ratchet — which is how this
// file first failed the suite. The idiom is already used by `frameworkConfigs.test.ts`.
import appSource from '../../../App.tsx?raw';

import { MetricsTable } from '../MetricsTable';
import { RungChip } from '../RungChip';
import { SweepGrid, formatMargin } from '../SweepGrid';
import { TokenTrace } from '../TokenTrace';
import { PANEL_IDS, isActivePanel } from '../../../config/panels';
import { navItems } from '../../layout/Sidebar';
import type { ProbeEvaluation, ProbeScoreResult } from '../../../types/probeMonitor';

describe('the panel is reachable', () => {
  it('is in the panel registry', () => {
    expect(PANEL_IDS).toContain('probe-monitors');
    expect(isActivePanel('probe-monitors')).toBe(true);
  });

  it('has a sidebar nav item', () => {
    const item = navItems.find((entry) => entry.id === 'probe-monitors');
    expect(item).toBeDefined();
    expect(item?.label).toBe('Probes');
  });

  it('sits immediately after Circuits', () => {
    // FTID §6. Order IS array order — there is no sort key — so this pins placement.
    const ids = navItems.map((entry) => entry.id);
    expect(ids[ids.indexOf('circuits') + 1]).toBe('probe-monitors');
  });

  it('App renders it for its panel id', () => {
    // ⚠ THE THIRD REGISTRATION POINT. Omitting it produced a panel that worked when
    // clicked and vanished on reload, with no type error anywhere — because the two
    // unions agreed and the guard array was plain strings. Read from App.tsx's source,
    // because the branch is a JSX conditional with no runtime handle to assert on.
    expect(appSource).toContain("activePanel === 'probe-monitors'");
    expect(appSource).toContain('<ProbeMonitorsPanel />');
  });
});

function evaluation(metrics: ProbeEvaluation['metrics']): ProbeEvaluation {
  return {
    id: 'pme_1',
    probe_id: 'pm_1',
    dataset_id: 'pmd_1',
    status: metrics.scored ? 'completed' : 'refused',
    n_positive: metrics.n_positive ?? null,
    n_negative: metrics.n_negative ?? null,
    metrics,
    created_at: '2026-09-25T00:00:00Z',
  };
}

describe('a refusal renders its reason', () => {
  it('shows the sentence, not a zero', () => {
    render(
      <MetricsTable
        evaluations={[
          evaluation({
            scored: false,
            reason: 'fewer than 20 examples of a class (4 positive, 40 negative)',
            n_positive: 4,
            n_negative: 40,
          }),
        ]}
      />
    );
    expect(screen.getByTestId('refusal-reason')).toHaveTextContent('fewer than 20 examples');
    expect(screen.queryByText('0.000')).toBeNull();
    expect(screen.queryByText('0.500')).toBeNull();
  });

  it('a scored row shows its AUROC and its REALISED fpr', () => {
    render(
      <MetricsTable
        evaluations={[
          evaluation({
            scored: true,
            name: 'unseen',
            auroc: 0.874,
            ci: { low: 0.81, high: 0.93, resamples: 2000, alpha: 0.05 },
            n_positive: 50,
            n_negative: 50,
            operating_points: [
              { target_fpr: 0.01, threshold: 1.2, realised_fpr: 0.02, recall: 0.4 },
            ],
          }),
        ]}
      />
    );
    expect(screen.getByText('0.874')).toBeInTheDocument();
    // The REALISED rate, which is usually not the target.
    expect(screen.getByText(/2\.0% spent/)).toBeInTheDocument();
  });

  it('an empty list says so rather than rendering an empty table', () => {
    render(<MetricsTable evaluations={[]} />);
    expect(screen.getByTestId('metrics-empty')).toBeInTheDocument();
  });
});

describe('the rung wording comes from the server', () => {
  it('renders the language it is given', () => {
    render(<RungChip rung={2} language="detects on unseen tasks" />);
    expect(screen.getByTestId('rung-language')).toHaveTextContent('detects on unseen tasks');
  });

  it('does NOT invent a phrase for a rung number', () => {
    // The guarantee: an unexpected phrase for rung 2 still renders verbatim, which
    // proves there is no client-side rung→phrase map overriding the server.
    render(<RungChip rung={2} language="WHATEVER THE SERVER SAID" />);
    expect(screen.getByTestId('rung-language')).toHaveTextContent('WHATEVER THE SERVER SAID');
  });

  it('shows the next step and the reasons', () => {
    render(
      <RungChip
        rung={1}
        language="detects on held-out data"
        nextStep="evaluate on every out-of-distribution set"
        reasons={['CI lower bound above 0.5 in-distribution on: held_out']}
      />
    );
    expect(screen.getByTestId('rung-next-step')).toHaveTextContent('out-of-distribution');
    expect(screen.getByText(/CI lower bound/)).toBeInTheDocument();
  });
});

describe('the sweep grid shows the whole grid', () => {
  const selection = {
    grid: [
      { layer: 5, pooling: 'mean', val_auroc: 0.71, n_train: 80, n_val: 20 },
      { layer: 5, pooling: 'last', val_auroc: null, n_train: 80, n_val: 20 },
      { layer: 10, pooling: 'mean', val_auroc: 0.92, n_train: 80, n_val: 20 },
      { layer: 10, pooling: 'last', val_auroc: 0.88, n_train: 80, n_val: 20 },
    ],
    chosen: [10],
    margin: 0.21,
    poolings: ['mean', 'last'],
  };

  it('renders every cell, not only the winner', () => {
    render(<SweepGrid selection={selection} />);
    expect(screen.getByText('0.710')).toBeInTheDocument();
    expect(screen.getByText('0.920')).toBeInTheDocument();
    expect(screen.getByText('0.880')).toBeInTheDocument();
  });

  it('an unscored cell is a dash, never a 0', () => {
    render(<SweepGrid selection={selection} />);
    expect(screen.getByText('—')).toBeInTheDocument();
    expect(screen.queryByText('0.000')).toBeNull();
  });

  it('warns when the margin makes the choice arbitrary', () => {
    render(<SweepGrid selection={{ ...selection, margin: 0.002 }} />);
    expect(screen.getByTestId('sweep-margin')).toHaveTextContent('near-tie');
  });

  it('does not warn on a clear win', () => {
    render(<SweepGrid selection={selection} />);
    expect(screen.getByTestId('sweep-margin')).not.toHaveTextContent('near-tie');
  });

  it('never renders a non-zero margin as a flat 0.0000', () => {
    // ⚠ THE FIRST REAL SWEEP. Llama-3.1-8B chose L11 at 0.9984371 over L16 at
    // 0.9983898: a margin of 0.0000473, which toFixed(4) rendered as "0.0000" — beside
    // the words "a near-tie", that reads as an exact tie the sweep could not resolve.
    expect(formatMargin(0.0000473)).toBe('<0.0001');
    expect(formatMargin(0)).toBe('0.0000');
    expect(formatMargin(0.0123)).toBe('0.0123');
    expect(formatMargin(0.0001)).toBe('0.0001');
    expect(formatMargin(-0.00002)).toBe('>-0.0001');
  });

  it('shows the honest margin in the rendered grid', () => {
    render(
      <SweepGrid
        selection={{
          chosen: [11],
          poolings: ['mean'],
          margin: 0.0000473,
          grid: [
            { layer: 11, pooling: 'mean', val_auroc: 0.9984371, n_train: 6800, n_val: 1200 },
            { layer: 16, pooling: 'mean', val_auroc: 0.9983898, n_train: 6800, n_val: 1200 },
          ],
        }}
      />,
    );
    const line = screen.getByTestId('sweep-margin');
    expect(line).toHaveTextContent('<0.0001');
    expect(line).not.toHaveTextContent('0.0000');
    // The warning still fires: it is a near-tie, and that is the point.
    expect(line).toHaveTextContent('near-tie');
  });

  it('says so when there is no sweep', () => {
    render(<SweepGrid selection={null} />);
    expect(screen.getByTestId('sweep-empty')).toBeInTheDocument();
  });
});

describe('the token trace distinguishes unscored from low-scoring', () => {
  const result: ProbeScoreResult = {
    probe_id: 'pm_1',
    aggregate: 1.5,
    threshold: 1.0,
    fires: true,
    tokens: [
      { token: 'user', scored: false },
      { token: 'transfer', scored: true },
      { token: 'funds', scored: true },
    ],
    token_scores: [2.0, -0.5],
    n_scored: 2,
    role_mask_reliable: true,
    truncated: false,
  };

  it('marks an unscored token rather than shading it cold', () => {
    render(<TokenTrace result={result} />);
    const unscored = screen.getAllByTestId('token-unscored');
    expect(unscored).toHaveLength(1);
    expect(unscored[0]).toHaveTextContent('user');
  });

  it('shades the scored tokens', () => {
    render(<TokenTrace result={result} />);
    expect(screen.getAllByTestId('token-scored')).toHaveLength(2);
  });

  it('says NO THRESHOLD rather than "does not fire" when none was placed', () => {
    render(<TokenTrace result={{ ...result, threshold: null, fires: null }} />);
    expect(screen.getByTestId('no-threshold')).toHaveTextContent('no threshold placed');
    expect(screen.queryByText(/below threshold/)).toBeNull();
  });

  it('surfaces an unreliable role mask', () => {
    render(<TokenTrace result={{ ...result, role_mask_reliable: false }} />);
    expect(screen.getByTestId('mask-unreliable')).toBeInTheDocument();
  });

  it('surfaces truncation', () => {
    render(<TokenTrace result={{ ...result, truncated: true }} />);
    expect(screen.getByTestId('truncated')).toBeInTheDocument();
  });
});
