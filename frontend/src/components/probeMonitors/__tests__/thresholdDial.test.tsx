/**
 * The operating-point dial: what it shows before it moves anything, and what it never shows.
 *
 * ⚠ THE TWO PROPERTIES WORTH PINNING ARE BOTH ABOUT HONESTY, NOT MECHANICS.
 *
 *   1. **A preview writes nothing, and a commit is a separate, deliberate act.** The server's
 *      `preview` defaults true; this asserts the component SENDS it true on the first click and
 *      false only on the commit button. A dial that committed on preview would move a live
 *      monitor's bar because somebody wanted to see a number.
 *   2. **`null` is "fires on nothing", never 0.0000.** Rendering the tightest possible bar as the
 *      loosest is the NULL-is-not-zero confusion the probe model warns about, one layer out —
 *      and it is the exact state an unaffordable target produces.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import panelSource from '../../panels/ProbeMonitorsPanel.tsx?raw';
import { ThresholdDial } from '../ThresholdDial';
import { TransferCaution } from '../TransferCaution';
import type { ProbeRecalibration, ThresholdTransfer } from '../../../types/probeMonitor';

function transfer(over: Partial<ThresholdTransfer> = {}): ThresholdTransfer {
  return {
    shipped_threshold: 11.9144,
    threshold_source: 'calibration_set',
    per_set: [
      {
        name: 'mental_health_balanced',
        auroc: 0.773,
        own_threshold_at_1pct: 18.77,
        max_score: 40.1,
        recall_at_shipped: 0.5,
        fpr_at_shipped: 0.426,
        unreachable: false,
      },
      {
        name: 'toolace_balanced',
        auroc: 0.95,
        own_threshold_at_1pct: -5.44,
        max_score: 1.17,
        recall_at_shipped: 0.0,
        fpr_at_shipped: 0.0,
        unreachable: true,
      },
    ],
    own_threshold_spread: 24.21,
    unreachable_sets: ['toolace_balanced'],
    caution: null,
    ...over,
  };
}

function proposal(over: Partial<ProbeRecalibration> = {}): ProbeRecalibration {
  return {
    probe_id: 'pm_1',
    current: { threshold: 11.9144, target_fpr: 0.01, realised_fpr: 0.01, revision: 1 },
    proposed: {
      threshold: 14.5201,
      target_fpr: 0.005,
      realised_fpr: 0.005,
      revision: 2,
      fires_on_nothing: false,
    },
    window_decisions: null,
    length_bands: null,
    transfer_current: transfer(),
    transfer_proposed: transfer({ shipped_threshold: 14.5201 }),
    published_copies_go_stale: false,
    committed: false,
    definition_invalidated: false,
    ...over,
  };
}

function setup(outcome: ProbeRecalibration | null = proposal()) {
  const onRecalibrate = vi.fn().mockResolvedValue(outcome);
  render(
    <ThresholdDial
      probeId="pm_1"
      currentThreshold={11.9144}
      currentTargetFpr={0.01}
      onRecalibrate={onRecalibrate}
    />
  );
  return { onRecalibrate, user: userEvent.setup() };
}

describe('ThresholdDial', () => {
  it('previews with preview=true and writes nothing', async () => {
    const { onRecalibrate, user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    await waitFor(() => expect(onRecalibrate).toHaveBeenCalledTimes(1));
    expect(onRecalibrate.mock.calls[0][0]).toMatchObject({ preview: true });
  });

  it('only COMMITS when the commit button is pressed, and sends preview=false', async () => {
    const { onRecalibrate, user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    await screen.findByTestId('recalibrate-proposal');
    await user.click(screen.getByRole('button', { name: /commit this bar/i }));
    await waitFor(() => expect(onRecalibrate).toHaveBeenCalledTimes(2));
    expect(onRecalibrate.mock.calls[1][0]).toMatchObject({ preview: false });
  });

  it('offers no commit button until a preview has been read', () => {
    setup();
    expect(screen.queryByRole('button', { name: /commit this bar/i })).toBeNull();
  });

  it('sends the typed target rather than the preset it started on', async () => {
    const { onRecalibrate, user } = setup();
    const input = screen.getByLabelText(/target fpr/i);
    await user.clear(input);
    await user.type(input, '0.005');
    await user.click(screen.getByRole('button', { name: /preview/i }));
    await waitFor(() => expect(onRecalibrate).toHaveBeenCalled());
    expect(onRecalibrate.mock.calls[0][0].target_fpr).toBe(0.005);
  });

  it('refuses to send a target outside the open interval', async () => {
    const { onRecalibrate, user } = setup();
    const input = screen.getByLabelText(/target fpr/i);
    await user.clear(input);
    await user.type(input, '1.5');
    expect(screen.getByRole('button', { name: /preview/i })).toBeDisabled();
    expect(onRecalibrate).not.toHaveBeenCalled();
  });

  it('shows both ends of the move, not just the new number', async () => {
    const { user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    const panel = await screen.findByTestId('recalibrate-proposal');
    expect(panel.textContent).toContain('11.9144');
    expect(panel.textContent).toContain('14.5201');
  });

  it('renders a fire-on-nothing bar as that, NEVER as 0.0000', async () => {
    const { user } = setup(
      proposal({
        proposed: {
          threshold: null,
          target_fpr: 0.0001,
          realised_fpr: 0.0,
          revision: 2,
          fires_on_nothing: true,
        },
      })
    );
    await user.click(screen.getByRole('button', { name: /preview/i }));
    const panel = await screen.findByTestId('recalibrate-proposal');
    expect(panel.textContent).toContain('fires on nothing');
    expect(panel.textContent).not.toContain('0.0000');
    expect(panel.textContent).toMatch(/silence/i);
  });

  it("surfaces the server's caution rather than composing one", async () => {
    const warning =
      "the evaluation sets' own 1% thresholds span 24.21, wider than the shipped threshold itself";
    const { user } = setup(
      proposal({ transfer_proposed: transfer({ caution: warning }) })
    );
    await user.click(screen.getByRole('button', { name: /preview/i }));
    expect((await screen.findByTestId('transfer-caution')).textContent).toContain(warning);
  });

  it('shows what the new bar does on EVERY set, beside the current bar', async () => {
    const { user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    const panel = await screen.findByTestId('recalibrate-proposal');
    expect(panel.textContent).toContain('mental_health_balanced');
    expect(panel.textContent).toContain('toolace_balanced');
    expect(panel.textContent).toContain('at the current bar');
    expect(panel.textContent).toContain('at the proposed bar');
  });

  it('marks an unreachable set as unreachable, not as zero recall', async () => {
    const { user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    const panel = await screen.findByTestId('recalibrate-proposal');
    expect(panel.textContent).toContain('unreachable');
  });

  it('says a published copy will keep stating the old bar', async () => {
    const { user } = setup(proposal({ published_copies_go_stale: true }));
    await user.click(screen.getByRole('button', { name: /preview/i }));
    const panel = await screen.findByTestId('recalibrate-proposal');
    expect(panel.textContent).toMatch(/published/i);
    expect(panel.textContent).toMatch(/old bar/i);
  });

  it('reports a refusal instead of leaving a stale proposal looking applied', async () => {
    const { user } = setup(null);
    await user.click(screen.getByRole('button', { name: /preview/i }));
    expect((await screen.findByTestId('recalibrate-refusal')).textContent).toMatch(/refused/i);
    expect(screen.queryByTestId('recalibrate-proposal')).toBeNull();
  });

  it('tells the operator the cached definition went stale on commit', async () => {
    const { user } = setup();
    await user.click(screen.getByRole('button', { name: /preview/i }));
    await screen.findByTestId('recalibrate-proposal');
    const committed = proposal({ committed: true, definition_invalidated: true });
    const input = screen.getByLabelText(/target fpr/i);
    expect(input).toBeTruthy();
    // Re-render the committed outcome through the same handler.
    const onRecalibrate = vi.fn().mockResolvedValue(committed);
    render(
      <ThresholdDial
        probeId="pm_2"
        currentThreshold={11.9144}
        currentTargetFpr={0.01}
        onRecalibrate={onRecalibrate}
      />
    );
    const previews = screen.getAllByRole('button', { name: /preview/i });
    await user.click(previews[previews.length - 1]);
    const banners = await screen.findAllByTestId('recalibrate-committed');
    expect(banners[0].textContent).toMatch(/invalidated/i);
  });
});

/**
 * ⚠ THE GUARDRAIL THAT WAS INVISIBLE FOR A MONTH.
 *
 * `threshold_transfer` is assembled on every probe report server-side, has 16 backend tests, and
 * the frontend's `ProbeReport` interface did not even declare the field — so the one statement
 * that a probe's absolute score does not transfer between distributions reached nobody using the
 * UI. Shipping a fast threshold dial without it would have been backwards: the dial makes it
 * trivially easy to lower a bar until the probe fires, and this is the sentence that says why
 * that is not calibration.
 */
describe('TransferCaution', () => {
  it('shows the caution the server wrote, verbatim', () => {
    const warning =
      "the evaluation sets' own 1% thresholds span 24.21, wider than the shipped threshold itself";
    render(<TransferCaution transfer={transfer({ caution: warning })} />);
    expect(screen.getByTestId('report-threshold-transfer-caution').textContent).toContain(warning);
  });

  it('names the sets the bar can never reach', () => {
    render(<TransferCaution transfer={transfer()} />);
    const row = screen.getByTestId('report-unreachable-sets');
    expect(row.textContent).toContain('toolace_balanced');
    expect(row.textContent).toMatch(/recall there is zero/i);
  });

  it('renders NOTHING when there is nothing measured to warn about', () => {
    /* An absent warning is not a reassurance. A probe with one evaluation set has no spread to
       report, and inventing "this threshold transfers fine" for it would be the honest-absence-
       into-silent-claim shape this estate has shipped before. */
    render(<TransferCaution transfer={transfer({ caution: null, unreachable_sets: [] })} />);
    expect(screen.queryByTestId('report-threshold-transfer')).toBeNull();
  });

  it('renders nothing rather than crashing when the server sent no transfer at all', () => {
    render(<TransferCaution transfer={null} />);
    expect(screen.queryByTestId('report-threshold-transfer')).toBeNull();
  });

  it('is CALLED by the report panel, not merely importable', () => {
    /* ⚠ THE CALL, NOT THE IMPORT. The 16-unregistered-MCP-tools defect passed every test by
       importing the module directly. An unused import type-errors, but a component imported and
       rendered behind a condition that is never true would not. */
    expect(panelSource).toMatch(/<TransferCaution\s+transfer=\{report\.threshold_transfer\}/);
    expect(panelSource).toMatch(/<ThresholdDial/);
    expect(panelSource).toMatch(/onRecalibrate=\{\(body\) => recalibrateProbe\(/);
  });
});
