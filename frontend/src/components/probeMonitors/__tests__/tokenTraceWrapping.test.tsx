/**
 * A pasted run with no spaces must WRAP inside the token trace, not run off it.
 *
 * ⚠ THIS ONE IS DIFFERENT FROM THE OTHER WRAPPING SITES, AND THAT IS WHY IT HAS
 * ITS OWN FILE. The trace's DOM is a `<span>` per token, not one text node, and
 * an inline element boundary is NOT a soft-wrap opportunity. So "Try it on your
 * own text" fed a JWT, a minified line or a degenerate generation produces
 * dozens of adjacent spans that the line breaker treats as ONE unbreakable
 * word — the case pre-wrap cannot help with, multiplied by the span count.
 *
 * `overflow-wrap` is an inherited property, so the fix belongs on the `<p>` that
 * holds the spans: one declaration reaches every token. These tests therefore
 * assert the class list of the SPANS' PARENT, reached through the spans rather
 * than by selecting on the class under test — selecting `p.break-words` would
 * assert nothing once `break-words` were gone.
 *
 * MUTATION CONTROL: recorded in the commit message.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TokenTrace } from '../TokenTrace';
import type { ProbeScoreResult } from '../../../types/probeMonitor';
import { expectWrapsLongRuns } from '../../../test/expectWrapsLongRuns';

/**
 * What the tokenizer does to a space-free paste: many short tokens, no space in
 * any of them, so there is no break opportunity anywhere in the paragraph.
 */
const RUN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9eyJzdWIiOiIxMjM0NTY3ODkwIn0';

function unbrokenPaste(): ProbeScoreResult {
  const pieces = RUN.match(/.{1,4}/g) as string[];
  return {
    probe_id: 'pm_wrap',
    aggregate: 2.5,
    threshold: 1.0,
    fires: true,
    tokens: [
      { token: '<|im_start|>', special: true, scored: false },
      ...pieces.map((t) => ({ token: t, scored: true })),
    ],
    token_scores: pieces.map((_, i) => (i % 2 === 0 ? 3.0 : -1.2)),
    n_scored: pieces.length,
    role_mask_reliable: true,
    truncated: false,
  } as ProbeScoreResult;
}

describe('the token trace wraps a space-free paste', () => {
  it('breaks inside the run instead of overflowing the trace', () => {
    render(<TokenTrace result={unbrokenPaste()} />);

    // Reached through the tokens, never by selecting on the class being asserted.
    const paragraph = screen.getAllByTestId('token-scored')[0].parentElement;
    expect(paragraph?.tagName).toBe('P');
    expectWrapsLongRuns(paragraph as HTMLElement);
  });

  it('puts the rule on the container, so every token span inherits it', () => {
    render(<TokenTrace result={unbrokenPaste()} />);

    // The spans deliberately carry no wrapping class of their own: overflow-wrap
    // inherits, so declaring it per-span would be the same rule written N times.
    const spans = [
      ...screen.getAllByTestId('token-scored'),
      ...screen.getAllByTestId('token-special'),
    ];
    expect(spans.length).toBeGreaterThan(4);
    const paragraph = spans[0].parentElement as HTMLElement;
    for (const span of spans) {
      expect(span.parentElement).toBe(paragraph);
    }
    expectWrapsLongRuns(paragraph);
  });
});
