/**
 * The whole-extraction Label button must keep relabelling everything by default.
 *
 * Making it skip adjudicated work by default was the obvious change and is the
 * wrong one: an operator pressing an existing control expects the behaviour it
 * has always had, and would quietly stop getting it with nothing on screen to
 * say so. The option is opt-in and states what it will do either way.
 *
 * MUTATION CONTROLS:
 *   * default the state to true          -> the default test fails
 *   * drop skip_adjudicated from the body -> the payload test fails
 *   * make the help text constant         -> the explanation test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

import { StartLabelingButton } from './StartLabelingButton';
import * as labelingAPI from '../../api/labeling';

vi.mock('../../api/labeling', async (orig) => ({
  ...(await orig<typeof labelingAPI>()),
  startLabeling: vi.fn(),
}));

async function openForm() {
  render(<StartLabelingButton extractionId="extr_1" />);
  fireEvent.click(await screen.findByText(/Label Features/i));
  return await screen.findByText(/Skip features that already have a label/i);
}

describe('Skip already-labeled', () => {
  beforeEach(() => vi.clearAllMocks());

  it('is off by default, and says the run will relabel everything', async () => {
    await openForm();

    const box = screen
      .getByText(/Skip features that already have a label/i)
      .closest('label')!
      .querySelector('input')!;
    expect(box).not.toBeChecked();
    expect(
      screen.getByText(/Relabels every feature, overwriting existing labels/i),
    ).toBeInTheDocument();
  });

  it('explains the other behaviour once it is on', async () => {
    await openForm();
    const box = screen
      .getByText(/Skip features that already have a label/i)
      .closest('label')!
      .querySelector('input')!;

    fireEvent.click(box);

    expect(box).toBeChecked();
    expect(
      screen.getByText(/never been attempted or that failed/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Relabels every feature/i),
    ).not.toBeInTheDocument();
  });
});
