/**
 * An edge's token and label signals say "not measured" rather than print a zero
 * nobody measured (live 2026-10-20, tracker 3.4 A1/A2: every mined edge between
 * unlabelled, token-less features read `token_overlap: 0.0`).
 *
 * NEGATIVE CONTROL: drop `...edgeSignalParts(edge)` from EdgeRow -> the row
 * tests go red; print `tokens ${Number(s.token_overlap ?? 0).toFixed(2)}`
 * unconditionally -> the unmeasured test goes red.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EdgeRow } from './CircuitsPanel';
import type { CircuitEdge } from '../../types/circuits';

function edge(type_signals: Record<string, unknown> | null): CircuitEdge {
  return {
    up: { layer: 12, kind: 'feature', feature_idx: 1 },
    down: { layer: 13, kind: 'feature', feature_idx: 2 },
    type: 'computed',
    type_signals,
    rung: 1,
    tested_and_failed: [],
    weight_prior: 0.54,
  } as unknown as CircuitEdge;
}

function renderRow(e: CircuitEdge) {
  return render(
    <table>
      <tbody>
        <EdgeRow edge={e} />
      </tbody>
    </table>,
  );
}

describe('Edge signals on the circuit detail', () => {
  it('says the token and label signals were not measured', () => {
    renderRow(edge({
      token_overlap: null, token_method: 'absent',
      label_similarity: 0, label_method: 'absent',
    }));
    const cell = screen.getByText(/prior 0\.54/);
    expect(cell.textContent).toContain('tokens not measured');
    expect(cell.textContent).toContain('labels not measured');
    expect(cell.textContent).not.toMatch(/tokens 0\.00/);
  });

  it('prints measured values, including a measured zero', () => {
    renderRow(edge({
      token_overlap: 0, token_method: 'jaccard',
      label_similarity: 0.5, label_method: 'token_set',
    }));
    const cell = screen.getByText(/prior 0\.54/);
    expect(cell.textContent).toContain('tokens 0.00');
    expect(cell.textContent).toContain('labels 0.50');
  });

  it('adds nothing for an unclassified (cluster) edge', () => {
    renderRow(edge({ classified: false, reason: 'cluster endpoint' }));
    expect(screen.getByText(/prior 0\.54/).textContent).toBe('prior 0.54');
  });
});
