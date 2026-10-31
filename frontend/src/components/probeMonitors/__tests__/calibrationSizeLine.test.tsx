/** Feature 040 S2 (FR-19, FTASKS 13.2): the calibration size line renders the server's sentence and flag. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { CalibrationSizeLine } from '../CalibrationSizeLine';

const base = { view_id: 'pmd_c', threshold_source: 'calibration_set', n_negatives_source: 'recorded' as const,
  target_fpr: 0.01, floor_source: 'setting' as const };

describe('CalibrationSizeLine', () => {
  it('states the sentence and flags a small calibration with the floor and its source', () => {
    render(<CalibrationSizeLine size={{ ...base, n_negatives: 540, floor: 2000, few_negatives: true,
      allowed_false_positives: 5, sentence: '1% of 540 = 5 rows' }} />);
    const line = screen.getByTestId('calibration-size');
    expect(line).toHaveTextContent('Calibration: 1% of 540 = 5 rows');
    expect(line).toHaveTextContent('fewer than the floor of 2,000 (the deployment setting)');
  });

  it('says nothing more when the calibration is large enough, and nothing at all without a count', () => {
    const { rerender } = render(<CalibrationSizeLine size={{ ...base, n_negatives: 5000, floor: 2000, few_negatives: false,
      allowed_false_positives: 50, sentence: '1% of 5,000 = 50 rows' }} />);
    expect(screen.getByTestId('calibration-size')).not.toHaveTextContent('fewer than');
    rerender(<CalibrationSizeLine size={null} />);
    expect(screen.queryByTestId('calibration-size')).toBeNull();
  });
});
