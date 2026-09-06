/**
 * Tests for AppliedFeaturesSummary (Feature 012).
 *
 * The component is the trust surface proving every cluster member contributed
 * to a blended run — it must render exactly the server-returned list.
 */

import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AppliedFeaturesSummary } from './AppliedFeaturesSummary';
import type { CombinedFeatureApplied } from '../../types/steering';

const applied: CombinedFeatureApplied[] = [
  { feature_idx: 10091, layer: 12, strength: 2.5, label: 'fear response', color: 'teal' },
  { feature_idx: 2262, layer: 12, strength: -1.8, label: null, color: 'blue' },
  // Unknown color name from a future server — must not crash (falls back to teal)
  { feature_idx: 7, layer: 12, strength: 0.5, label: 'odd', color: 'not-a-color' as never },
];

describe('AppliedFeaturesSummary', () => {
  it('renders nothing for an empty list', () => {
    const { container } = render(<AppliedFeaturesSummary applied={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows the count collapsed by default and expands to server data', () => {
    render(<AppliedFeaturesSummary applied={applied} />);
    const toggle = screen.getByRole('button', { name: /Applied features \(3\)/ });
    expect(toggle).toBeInTheDocument();
    // Collapsed: member chips not rendered yet
    expect(screen.queryByText('#10091')).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(screen.getByText('#10091')).toBeInTheDocument();
    expect(screen.getByText('fear response')).toBeInTheDocument();
    expect(screen.getByText('#2262')).toBeInTheDocument();
    expect(screen.getByText('#7')).toBeInTheDocument();
  });

  it('formats strength signs (+ for positive, bare minus for negative)', () => {
    render(<AppliedFeaturesSummary applied={applied} />);
    fireEvent.click(screen.getByRole('button', { name: /Applied features/ }));
    expect(screen.getByText('@ +2.5')).toBeInTheDocument();
    expect(screen.getByText('@ -1.8')).toBeInTheDocument();
  });

  it('survives unknown color names (falls back, no crash)', () => {
    render(<AppliedFeaturesSummary applied={[applied[2]]} />);
    fireEvent.click(screen.getByRole('button', { name: /Applied features \(1\)/ }));
    expect(screen.getByText('#7')).toBeInTheDocument();
  });
});
