/**
 * The Monitor renders task.progress as a percentage, verbatim.
 *
 * This pins the CONSUMING side of the 0-100 contract that
 * backend/tests/unit/test_monitor_federation_contract.py pins on the producing
 * side. The 2026-07-26 bug was a scale mismatch across that boundary — the
 * extraction federator passed a 0-1 fraction — and the wrong repair would be to
 * multiply here, which would then double-scale every OTHER source (trainings,
 * pushes, tokenizations) that already sends 0-100.
 *
 * MUTATION CONTROL:
 *   * multiply task.progress by 100 in the component -> these fail
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@testing-library/react';
import { ActiveOperationsSection } from './ActiveOperationsSection';
import { useTaskQueueStore } from '../../stores/taskQueueStore';

vi.mock('../../stores/taskQueueStore');

function mountWith(progress: number, taskType = 'extraction') {
  (useTaskQueueStore as any).mockReturnValue({
    activeTasks: [
      {
        id: 't1',
        task_id: 't1',
        task_type: taskType,
        entity_id: 'e1',
        entity_type: 'extraction',
        status: 'running',
        progress,
        can_retry: false,
        retry_count: 0,
        entity_info: { name: 'Extraction (features)' },
        created_at: new Date().toISOString(),
      },
    ],
    activeLoading: false,
    activeError: null,
    fetchActiveTasks: vi.fn(),
  });
  return render(<ActiveOperationsSection />);
}

describe('ActiveOperationsSection progress scale', () => {
  beforeEach(() => vi.clearAllMocks());

  it('renders 98 as 98.0%, not 9800%', () => {
    mountWith(98);
    expect(screen.getByText('98.0%')).toBeInTheDocument();
  });

  it('renders a training percentage unchanged', () => {
    // trainings.progress has always been 0-100; the fix must not double-scale it.
    mountWith(42.5, 'training');
    expect(screen.getByText('42.5%')).toBeInTheDocument();
  });

  it('would have shown the reported "1.0%" for a raw fraction', () => {
    // Negative control documenting the original symptom: 0.98 arriving here is
    // the bug, and it must be fixed upstream, not by scaling in this component.
    mountWith(0.98);
    expect(screen.getByText('1.0%')).toBeInTheDocument();
  });
});
