/**
 * A watchlist without its scoring definition is a different detector.
 *
 * The server refuses one; this form must mirror that rather than letting the
 * user discover it after export. And a concept line with no threshold is
 * DROPPED, never defaulted — a zero threshold fires on everything, which is a
 * detector that always says yes.
 *
 * MUTATION CONTROLS:
 *   * allow submit without a scoring definition -> "refuses" fails
 *   * default a missing threshold to 0          -> "drops" fails
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WatchlistCard, parseConcepts } from './WatchlistCard';

vi.mock('../../api/jlens', () => ({ jlensApi: { createWatchlist: vi.fn() } }));

describe('parseConcepts', () => {
  it('parses token,threshold pairs', () => {
    expect(parseConcepts(' evaluation, 0.4\n test, 0.35')).toEqual([
      { token: 'evaluation', threshold: 0.4 },
      { token: 'test', threshold: 0.35 },
    ]);
  });

  it('DROPS a line with no threshold rather than defaulting it to zero', () => {
    // A zero threshold fires on everything — a detector that always says yes
    // is worse than one that is absent, because it looks like it is working.
    expect(parseConcepts('evaluation\n test, 0.3')).toEqual([
      { token: 'test', threshold: 0.3 },
    ]);
  });

  it('drops a non-numeric threshold', () => {
    expect(parseConcepts('a, soon')).toEqual([]);
  });
});

describe('WatchlistCard', () => {
  it('refuses to save without a scoring definition', () => {
    render(<WatchlistCard artifactId="gemma-2-2b-it" />);
    fireEvent.click(screen.getByRole('button', { name: /author/i }));
    fireEvent.change(screen.getByPlaceholderText('evaluation-awareness'), {
      target: { value: 'w1' },
    });
    fireEvent.change(screen.getByRole('textbox', { name: /concepts/i }), {
      target: { value: 'a, 0.3' },
    });

    expect(screen.getByRole('button', { name: /^validate$/i })).toBeDisabled();
  });

  it('cannot be opened without an artifact to score through', () => {
    render(<WatchlistCard artifactId={null} />);
    expect(screen.getByRole('button', { name: /author/i })).toBeDisabled();
  });
});
