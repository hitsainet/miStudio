/**
 * The export flow: the gate, the download, publication, and the refusals (033 task 6.4).
 *
 * ⚠ WHAT THESE PIN, AND WHY EACH ONE MATTERS.
 *
 *   * **the gate is REQUIRED below rung 2.** A build that skipped it would export a probe stating
 *     evidence it does not have, and the acknowledgement is the only record that somebody decided
 *     that was acceptable.
 *   * **an SAE probe with no published dictionary cannot be exported at all**, and the section says
 *     why rather than disabling a button. That refusal is not waivable: without the dictionary's
 *     location a consumer cannot encode.
 *   * **no token field exists.** A credential typed into this component would sit in browser state
 *     and in any screenshot of the dialog.
 *
 * MUTATION CONTROLS (each verified to fail this file):
 *   X1  the build skips the dialog below rung 2        → the gate test
 *   X2  the reason floor removed                       → the short-reason test
 *   X3  the SAE refusal removed                        → the sae-blocked tests
 *   X4  download enabled before a build                → the download test
 *   X5  publish enabled before a build                 → the publish-order test
 *   X6  a token field added to the publish dialog       → the secrecy test
 *   X7  the invalidation notice removed                 → the invalidated test
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { AcknowledgeDialog, MIN_REASON_LENGTH } from '../AcknowledgeDialog';
import { ProbeExportSection } from '../ProbeExportSection';
import { PublishProbeDialog } from '../PublishProbeDialog';

const buildDefinition = vi.hoisted(() => vi.fn());
const publishDefinition = vi.hoisted(() => vi.fn());

vi.mock('../../../api/probeMonitors', () => ({
  probeMonitorsApi: {
    buildDefinition,
    publishDefinition,
    exportUrl: (id: string) => `/api/v1/probe-monitors/probes/${id}/export`,
  },
}));

function probe(overrides: Record<string, unknown> = {}) {
  return {
    id: 'pm_1',
    run_id: 'pmr_1',
    layer: 11,
    rule: 'mean',
    rule_params: {},
    variant: 'dense',
    sae_id: null,
    sae_feature_indices: null,
    val_metrics: {},
    selected: true,
    threshold: 2.4,
    target_fpr: 0.01,
    realised_fpr: 0.0097,
    threshold_source: 'validation_negatives',
    streamable: true,
    rung: 2,
    rung_reasons: [],
    created_at: '2026-09-26T00:00:00Z',
    ...overrides,
  } as never;
}

beforeEach(() => {
  buildDefinition.mockReset().mockResolvedValue({ task_id: 'task-1', status: 'queued' });
  publishDefinition.mockReset().mockResolvedValue({ task_id: 'task-2', status: 'queued' });
});

describe('the evidence gate', () => {
  it('builds straight away at rung 2', async () => {
    render(
      <ProbeExportSection probe={probe({ rung: 2 })} rungLanguage="detects on unseen tasks" />
    );
    await userEvent.click(screen.getByTestId('probe-export-build'));
    await waitFor(() => expect(buildDefinition).toHaveBeenCalledTimes(1));
    expect(buildDefinition.mock.calls[0][1]).toEqual({ acknowledgeReason: undefined });
    expect(screen.queryByTestId('probe-acknowledge-dialog')).toBeNull();
  });

  it.each([0, 1])('demands an acknowledgement at rung %i', async (rung) => {
    render(
      <ProbeExportSection probe={probe({ rung })} rungLanguage="trained only" />
    );
    await userEvent.click(screen.getByTestId('probe-export-build'));
    expect(screen.getByTestId('probe-acknowledge-dialog')).toBeInTheDocument();
    // ⚠ NOTHING WAS SENT. A dialog that appears after the request is theatre.
    expect(buildDefinition).not.toHaveBeenCalled();
  });

  it('sends the reason once it is given', async () => {
    render(<ProbeExportSection probe={probe({ rung: 1 })} rungLanguage="detects on held-out data" />);
    await userEvent.click(screen.getByTestId('probe-export-build'));
    await userEvent.type(
      screen.getByTestId('probe-acknowledge-reason'),
      'exploratory monitor, not used for gating'
    );
    await userEvent.click(screen.getByTestId('probe-acknowledge-confirm'));
    await waitFor(() => expect(buildDefinition).toHaveBeenCalledTimes(1));
    expect(buildDefinition.mock.calls[0][1].acknowledgeReason).toContain('exploratory');
  });

  it('shows the server rung wording, not a phrase of its own', () => {
    render(
      <AcknowledgeDialog
        rung={1}
        rungLanguage="detects on held-out data"
        nextStep="evaluate on an out-of-distribution set"
        onCancel={() => {}}
        onConfirm={() => {}}
      />
    );
    expect(screen.getByText(/detects on held-out data/)).toBeInTheDocument();
    expect(screen.getByText(/out-of-distribution set/)).toBeInTheDocument();
  });

  it('refuses a reason under the floor', async () => {
    const onConfirm = vi.fn();
    render(
      <AcknowledgeDialog rung={1} rungLanguage="x" onCancel={() => {}} onConfirm={onConfirm} />
    );
    const confirm = screen.getByTestId('probe-acknowledge-confirm');
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByTestId('probe-acknowledge-reason'), 'a'.repeat(MIN_REASON_LENGTH - 1));
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByTestId('probe-acknowledge-reason'), 'aa');
    expect(confirm).toBeEnabled();
    await userEvent.click(confirm);
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it('whitespace does not satisfy the floor', async () => {
    render(<AcknowledgeDialog rung={1} rungLanguage="x" onCancel={() => {}} onConfirm={vi.fn()} />);
    await userEvent.type(screen.getByTestId('probe-acknowledge-reason'), '          ');
    expect(screen.getByTestId('probe-acknowledge-confirm')).toBeDisabled();
  });
});

describe('an SAE probe with no published dictionary', () => {
  const saeProbe = probe({ variant: 'sae', sae_id: 'sae_1', rung: 2 });

  it('explains the refusal instead of just disabling the button', () => {
    render(<ProbeExportSection probe={saeProbe} rungLanguage="detects on unseen tasks" />);
    const notice = screen.getByTestId('probe-export-sae-blocked');
    expect(notice).toHaveTextContent(/cannot encode without it/);
    expect(notice).toHaveTextContent(/acknowledgement does not waive/);
  });

  it('cannot be built', () => {
    render(<ProbeExportSection probe={saeProbe} rungLanguage="x" />);
    expect(screen.getByTestId('probe-export-build')).toBeDisabled();
  });

  it('offers the action that fixes it', async () => {
    const onPushSae = vi.fn();
    render(<ProbeExportSection probe={saeProbe} rungLanguage="x" onPushSae={onPushSae} />);
    await userEvent.click(screen.getByTestId('probe-export-push-sae'));
    expect(onPushSae).toHaveBeenCalledWith('sae_1');
  });

  it('is exportable once the dictionary has a home', () => {
    render(
      <ProbeExportSection probe={saeProbe} rungLanguage="x" saeHfRepo="owner/saes" />
    );
    expect(screen.queryByTestId('probe-export-sae-blocked')).toBeNull();
    expect(screen.getByTestId('probe-export-build')).toBeEnabled();
  });

  it('a DENSE probe is never blocked for this reason', () => {
    render(<ProbeExportSection probe={probe()} rungLanguage="x" />);
    expect(screen.queryByTestId('probe-export-sae-blocked')).toBeNull();
  });
});

describe('download and publish follow the build', () => {
  it('download is inert before a build', () => {
    render(<ProbeExportSection probe={probe()} rungLanguage="x" />);
    const link = screen.getByTestId('probe-export-download');
    expect(link).toHaveAttribute('aria-disabled', 'true');
    expect(link).not.toHaveAttribute('href');
  });

  it('download points at the export route once built', () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuiltAt="2026-09-26T10:00:00Z"
        definitionSha256={'a'.repeat(64)}
      />
    );
    expect(screen.getByTestId('probe-export-download')).toHaveAttribute(
      'href',
      '/api/v1/probe-monitors/probes/pm_1/export'
    );
  });

  it('publish is disabled before a build', () => {
    render(<ProbeExportSection probe={probe()} rungLanguage="x" />);
    expect(screen.getByTestId('probe-export-publish')).toBeDisabled();
  });

  it('publish sends the repo and privacy once built', async () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuiltAt="2026-09-26T10:00:00Z"
      />
    );
    await userEvent.click(screen.getByTestId('probe-export-publish'));
    await userEvent.type(screen.getByTestId('probe-publish-repo'), 'owner/probes');
    await userEvent.click(screen.getByTestId('probe-publish-confirm'));
    await waitFor(() => expect(publishDefinition).toHaveBeenCalledTimes(1));
    expect(publishDefinition.mock.calls[0][1]).toEqual({
      repo_id: 'owner/probes',
      private: true,
    });
  });

  it('a malformed repo id cannot be submitted', async () => {
    render(
      <PublishProbeDialog
        probeName="pm_1"
        rung={2}
        rungLanguage="x"
        onCancel={() => {}}
        onConfirm={vi.fn()}
      />
    );
    const confirm = screen.getByTestId('probe-publish-confirm');
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByTestId('probe-publish-repo'), 'no-slash');
    expect(confirm).toBeDisabled();
    await userEvent.type(screen.getByTestId('probe-publish-repo'), '/probes');
    expect(confirm).toBeEnabled();
  });

  it('shows the revision link after a publication', () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuiltAt="2026-09-26T10:00:00Z"
        published={[
          {
            repo_id: 'owner/probes',
            revision: 'abcdef1234567890',
            path: 'p.probe.json',
            private: false,
            at: '2026-09-26T11:00:00Z',
          },
        ]}
      />
    );
    const list = screen.getByTestId('probe-export-published');
    expect(list).toHaveTextContent('owner/probes');
    expect(list).toHaveTextContent('abcdef12');
    expect(screen.getByRole('link', { name: 'owner/probes' })).toHaveAttribute(
      'href',
      'https://huggingface.co/owner/probes'
    );
  });
});

describe('the token never reaches the browser', () => {
  it('the publish dialog has no token field', () => {
    render(
      <PublishProbeDialog
        probeName="pm_1"
        rung={2}
        rungLanguage="x"
        onCancel={() => {}}
        onConfirm={vi.fn()}
      />
    );
    const inputs = screen.getAllByRole('textbox');
    for (const input of inputs) {
      expect(input.getAttribute('data-testid')).not.toMatch(/token/i);
    }
    expect(screen.getByTestId('probe-publish-dialog')).toHaveTextContent(
      /token comes from Settings/i
    );
  });

  it('the publish body carries no credential', async () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuiltAt="2026-09-26T10:00:00Z"
      />
    );
    await userEvent.click(screen.getByTestId('probe-export-publish'));
    await userEvent.type(screen.getByTestId('probe-publish-repo'), 'owner/probes');
    await userEvent.click(screen.getByTestId('probe-publish-confirm'));
    await waitFor(() => expect(publishDefinition).toHaveBeenCalled());
    expect(JSON.stringify(publishDefinition.mock.calls[0])).not.toMatch(/token/i);
  });
});

describe('server errors and invalidation are visible', () => {
  it('a build failure is rendered rather than swallowed', async () => {
    buildDefinition.mockRejectedValue(new Error('422: rung 1 is below the rung 2 an export claims'));
    render(<ProbeExportSection probe={probe({ rung: 2 })} rungLanguage="x" />);
    await userEvent.click(screen.getByTestId('probe-export-build'));
    await waitFor(() =>
      expect(screen.getByTestId('probe-export-error')).toHaveTextContent(/below the rung 2/)
    );
  });

  it('an invalidated definition says so and says why', () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuild={{ invalidated: { reason: 'rung changed 2 -> 3' } }}
      />
    );
    expect(screen.getByTestId('probe-export-invalidated')).toHaveTextContent('rung changed 2 -> 3');
  });

  it('a fresh build shows no invalidation notice', () => {
    render(
      <ProbeExportSection
        probe={probe()}
        rungLanguage="x"
        definitionBuiltAt="2026-09-26T10:00:00Z"
        definitionBuild={{ resolved: { vectors: 16, bytes: 41234 } }}
      />
    );
    expect(screen.queryByTestId('probe-export-invalidated')).toBeNull();
    expect(screen.getByTestId('probe-export-built')).toHaveTextContent('16');
    expect(screen.getByTestId('probe-export-built')).toHaveTextContent('41234');
  });
});
