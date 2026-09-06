/**
 * Component tests for the Circuits Discovery report card (Feature 016).
 *
 * Pins the R1/R2 trust-surface fixes so a refactor can't silently
 * reintroduce them:
 *  - replication.rate === null renders "n/a", NOT a confident "0%" (R1 Prod-P2)
 *  - caps-hit renders the truncation banner, never hides truncation (R1 QA)
 *  - fdr.p_resolution is surfaced (R2 Prod-P3)
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RunReportCard } from './CircuitsPanel';
import type { DiscoveryReport } from '../../types/circuits';

function makeReport(overrides: Partial<DiscoveryReport> = {}): DiscoveryReport {
  return {
    granularity: 'feature',
    mode: 'open',
    supernode_activation: null,
    lag0_disclosure: 'All co-activation is lag-0 (same token position).',
    null_summary: { method: 'within_document_circular_shift', shuffles: 100, percentile: 99 },
    fdr: { discipline: 'benjamini_hochberg', p_source: 'pooled_standardized_empirical_null', q: 0.05, tested: 50, passed: 3 },
    replication: { tested: 3, replicated: 2, rate: 0.6667 },
    counts_by_stage: { pairs_considered: 900, post_support: 120, null_tested: 50, post_fdr: 3, candidates_persisted: 3 },
    caps: { units_per_layer: 500, unit_cap_hit_layers: [], null_tested_cap: 2000, null_cap_hit: false, candidate_cap: 2000, candidates_truncated: false },
    uncovered_seeds: [],
    attribution: null,
    uplift: null,
    echo_filter: null,
    wall_clock_seconds: 12.3,
    ...overrides,
  } as DiscoveryReport;
}

describe('RunReportCard', () => {
  it('renders the replication rate as a percentage when present', () => {
    render(<RunReportCard report={makeReport({ replication: { tested: 3, replicated: 2, rate: 0.6667 } })} />);
    expect(screen.getByText('67%')).toBeInTheDocument();
  });

  it('renders "n/a" — NOT a false "0%" — when replication rate is null (R1 Prod-P2)', () => {
    render(<RunReportCard report={makeReport({ replication: { tested: 0, replicated: 0, rate: null } })} />);
    expect(screen.getByText('n/a')).toBeInTheDocument();
    expect(screen.queryByText('0%')).not.toBeInTheDocument();
    expect(screen.getByText(/no held-out candidates tested/)).toBeInTheDocument();
  });

  it('surfaces a caps-hit warning when candidates were truncated (never hides truncation)', () => {
    render(<RunReportCard report={makeReport({
      caps: { units_per_layer: 500, unit_cap_hit_layers: [13], null_tested_cap: 2000, null_cap_hit: true, candidate_cap: 2000, candidates_truncated: true },
    })} />);
    expect(screen.getByText(/Caps hit/)).toBeInTheDocument();
    expect(screen.getByText(/Candidate list was truncated/)).toBeInTheDocument();
  });

  it('does not show a caps warning when no cap was hit', () => {
    render(<RunReportCard report={makeReport()} />);
    expect(screen.queryByText(/Caps hit/)).not.toBeInTheDocument();
  });

  it('surfaces the FDR p-resolution when present (R2 Prod-P3)', () => {
    render(<RunReportCard report={makeReport({
      fdr: { discipline: 'benjamini_hochberg', p_source: 'pooled_standardized_empirical_null', p_resolution: 0.0002, q: 0.05, tested: 50, passed: 3 },
    })} />);
    expect(screen.getByText(/p-res/)).toBeInTheDocument();
  });
});
