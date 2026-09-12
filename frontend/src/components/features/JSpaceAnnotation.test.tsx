/**
 * Annotation lives where features live, and asks for nothing the modal lacks.
 *
 * The endpoint used to demand model_id, sae_id, layer AND a d_model decoder
 * direction. The feature modal has none of those — it has a feature id — which
 * is why this capability shipped with no UI for its entire life. All four are
 * now resolved server-side from the feature row.
 *
 * MUTATION CONTROLS:
 *   * send anything beyond feature_id/label_tokens -> "asks for nothing" fails
 *   * hide the UNKNOWN workspace class             -> "unknown is an answer" fails
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { JSpaceAnnotation } from './JSpaceAnnotation';

vi.mock('../../api/jlens', () => ({ jlensApi: { annotate: vi.fn() } }));
import { jlensApi } from '../../api/jlens';

beforeEach(() => vi.clearAllMocks());

const ANNOTATION = {
  feature_id: 'feat_sae_x_7',
  layer: 12,
  lens_kurtosis: 4.25,
  workspace_class: 'UNKNOWN',
  top_tokens: [' Paris', ' France'],
  disagreement_score: 0.9,
  has_disagreement: true,
};

describe('JSpaceAnnotation', () => {
  it('asks the server for nothing the modal does not have', async () => {
    vi.mocked(jlensApi.annotate).mockResolvedValue(ANNOTATION);
    render(<JSpaceAnnotation featureId="feat_sae_x_7" labelTokens={['capital']} />);

    await userEvent.click(screen.getByRole('button', { name: /annotate in j-space/i }));
    await waitFor(() => expect(jlensApi.annotate).toHaveBeenCalledTimes(1));

    const sent = vi.mocked(jlensApi.annotate).mock.calls[0][0];
    expect(sent.feature_id).toBe('feat_sae_x_7');
    // The SAE, its model and the layer live on the feature row. Restating them
    // here would mean asking the user to retype what they are looking at.
    expect(sent.sae_id).toBeUndefined();
    expect(sent.model_id).toBeUndefined();
    expect(sent.layer).toBeUndefined();
  });

  it('shows UNKNOWN as a real answer, with the reason', async () => {
    vi.mocked(jlensApi.annotate).mockResolvedValue(ANNOTATION);
    render(<JSpaceAnnotation featureId="f1" />);
    await userEvent.click(screen.getByRole('button', { name: /annotate in j-space/i }));

    await waitFor(() => expect(screen.getByText('UNKNOWN')).toBeInTheDocument());
    // Without boundaries measured on THIS model there is no principled middle
    // of the stack, and the published ones were measured elsewhere.
    expect(
      screen.getByText(/boundaries from another model do not transfer/i)
    ).toBeInTheDocument();
  });

  it('keeps the geometric and behavioural fields separate', async () => {
    vi.mocked(jlensApi.annotate).mockResolvedValue(ANNOTATION);
    render(<JSpaceAnnotation featureId="f1" />);
    await userEvent.click(screen.getByRole('button', { name: /annotate in j-space/i }));

    await waitFor(() => expect(screen.getByText('4.250')).toBeInTheDocument());
    // A sharp direction is not evidence of workspace membership — a motor
    // direction is sharp too (BR-012).
    expect(
      screen.getByText(/A sharp direction is not evidence of workspace membership/i)
    ).toBeInTheDocument();
  });

  it('surfaces a label disagreement without adjudicating it', async () => {
    vi.mocked(jlensApi.annotate).mockResolvedValue(ANNOTATION);
    render(<JSpaceAnnotation featureId="f1" />);
    await userEvent.click(screen.getByRole('button', { name: /annotate in j-space/i }));

    await waitFor(() => expect(screen.getByText('0.90')).toBeInTheDocument());
    expect(screen.getByText(/neither is\s+automatically right/i)).toBeInTheDocument();
  });
});
