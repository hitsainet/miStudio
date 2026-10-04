import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PrecisionBadge } from './PrecisionBadge';

describe('PrecisionBadge', () => {
  it('shows a recorded precision as recorded', () => {
    render(<PrecisionBadge label={{ value: 'bfloat16', source: 'recorded', note: null }} />);
    const badge = screen.getByTestId('precision-badge');
    expect(badge.textContent).toBe('bfloat16');
    expect(badge.getAttribute('data-source')).toBe('recorded');
  });

  it('never shows an inferred precision without saying it is inferred', () => {
    // Before 2026-10-03 every load path ran float16 and recorded nothing; the backend labels
    // that "inferred". Dropping the word would present a deduction as a fact.
    render(<PrecisionBadge label={{ value: 'float16', source: 'inferred', note: 'inferred: read before…' }} />);
    const badge = screen.getByTestId('precision-badge');
    expect(badge.textContent).toContain('float16');
    expect(badge.textContent).toContain('(inferred)');
  });

  it('calls out a precision that differs from what the model loads at now', () => {
    render(
      <PrecisionBadge label={{ value: 'float16', source: 'inferred', note: null }} servedAs="bfloat16" />,
    );
    expect(screen.getByTestId('precision-badge').textContent).toContain('now bfloat16');
  });

  it('says nothing was recorded when nothing was', () => {
    render(<PrecisionBadge label={{ value: null, source: 'unknown', note: null }} />);
    expect(screen.getByTestId('precision-badge').textContent).toBe('precision not recorded');
  });
});
