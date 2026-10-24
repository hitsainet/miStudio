/**
 * The score panel says WHICH bar it judged against, not just a number.
 *
 * ⚠ A BARE THRESHOLD LEFT THE READER COMPARING AGAINST THE WRONG BAR. On `pm_f463a8a235ae` a ~50-token
 * passage showed "threshold 17.9802" — the probe's global bar — while the live monitor judges that
 * length against the 0–203 band at 12.806. The backend now applies the band; this panel has to name
 * it, or a reader sees a number that changed without explanation and cannot tell why two inputs
 * with similar scores got different verdicts.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TokenTrace } from '../TokenTrace';
import type { ProbeScoreResult } from '../../../types/probeMonitor';

function result(over: Partial<ProbeScoreResult> = {}): ProbeScoreResult {
  return {
    probe_id: 'pm_f463a8a235ae',
    aggregate: 28.9189,
    threshold: 12.806,
    global_threshold: 17.9802,
    threshold_band: { min_tokens: 0, max_tokens: 203, threshold_source: 'band' },
    fires: true,
    tokens: [{ token: 'worried', special: false, scored: true }],
    token_scores: [1.0],
    n_scored: 50,
    role_mask_reliable: true,
    truncated: false,
    ...over,
  };
}

describe('the score panel names the bar it applied', () => {
  it('shows the band the input fell in', () => {
    render(<TokenTrace result={result()} />);
    const band = screen.getByTestId('score-threshold-band');
    expect(band.textContent).toMatch(/0–203 tokens/);
  });

  it('shows the global bar beside it when the band moved the threshold', () => {
    render(<TokenTrace result={result()} />);
    expect(screen.getByTestId('score-threshold-band').textContent).toMatch(/global 17\.9802/);
  });

  it('renders an open-ended final band as ∞, not "null"', () => {
    render(
      <TokenTrace
        result={result({
          threshold: 20.3118,
          threshold_band: { min_tokens: 519, max_tokens: null, threshold_source: 'band' },
        })}
      />
    );
    const band = screen.getByTestId('score-threshold-band').textContent ?? '';
    expect(band).toMatch(/519–∞ tokens/);
    expect(band).not.toMatch(/null/);
  });

  it('says a THIN band inherited the global bar, rather than presenting it as measured', () => {
    render(
      <TokenTrace
        result={result({
          threshold: 17.9802,
          threshold_band: { min_tokens: 0, max_tokens: null, threshold_source: 'global' },
        })}
      />
    );
    const band = screen.getByTestId('score-threshold-band').textContent ?? '';
    expect(band).toMatch(/inherits the global bar/);
    // And does not repeat the global as if it were a different number.
    expect(band).not.toMatch(/; global/);
  });

  it('shows no band at all for a probe without a band table', () => {
    render(<TokenTrace result={result({ threshold_band: null, threshold: 17.9802 })} />);
    expect(screen.queryByTestId('score-threshold-band')).toBeNull();
  });

  it('⚠ renders against an older backend that sends neither new field', () => {
    const old = result();
    delete (old as Partial<ProbeScoreResult>).threshold_band;
    delete (old as Partial<ProbeScoreResult>).global_threshold;
    render(<TokenTrace result={old} />);
    expect(screen.queryByTestId('score-threshold-band')).toBeNull();
    expect(screen.getByText(/fires/)).toBeTruthy();
  });
});
