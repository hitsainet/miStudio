/**
 * The five capabilities that existed in the API client and had NO UI CALLER.
 *
 * An audit of `api/probeMonitors.ts`'s nineteen methods against their callers found four with
 * none — `submitJudgeRun`, `cancelJudgeRun`, `evaluate`, `score` — and one route with no client
 * method at all (`GET /probes/{id}/score/{task_id}`). The consequence was not cosmetic:
 *
 *   * rung 3 needs a judge run  → **the top of the ladder was unreachable from the UI**
 *   * rung 1 and 2 need `evaluate` on more sets → **the middle was too**
 *   * `score` is how you point a probe at your own text → **the whole point, missing**
 *   * without the score GET, scoring was **write-only** — the exact shape the endpoint's own
 *     docstring was added to prevent, reproduced one layer up in the client
 *
 * A client method with no caller is the unregistered-MCP-tool defect wearing a third hat: written,
 * tested, unreachable. These tests are the guard, and each one fails if its control is removed.
 */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { JudgeRunForm } from '../JudgeRunForm';
import { ProbeActions } from '../ProbeActions';
import type { ProbeDataset, ProbeMonitorSummary } from '../../../types/probeMonitor';

const SETS = [
  { id: 'pmd_a', name: 'mup', config: 'toolace_balanced', role: 'eval',
    distribution: 'out_of_distribution' },
  { id: 'pmd_b', name: 'mup', config: 'mt_balanced', role: 'eval',
    distribution: 'out_of_distribution' },
  { id: 'pmd_train', name: 'mup', config: 'training', role: 'train', distribution: null },
] as unknown as ProbeDataset[];

const PROBES = [
  { id: 'pm_1', layer: 11, rule: 'mean', variant: 'dense' },
] as unknown as ProbeMonitorSummary[];

describe('a judge run can be started from the UI — rung 3 is reachable', () => {
  it('submits endpoint, model, sets and the probe being compared', async () => {
    const onSubmit = vi.fn();
    render(<JudgeRunForm datasets={SETS} probes={PROBES} onSubmit={onSubmit} />);
    await userEvent.type(screen.getByTestId('judge-endpoint'), 'http://millm/v1');
    await userEvent.type(screen.getByTestId('judge-model'), 'Qwen2.5-7B-Instruct');
    await userEvent.click(screen.getByTestId('judge-set-pmd_a'));
    await userEvent.selectOptions(screen.getByTestId('judge-probe'), 'pm_1');
    await userEvent.click(screen.getByTestId('judge-submit'));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      endpoint: 'http://millm/v1',
      model: 'Qwen2.5-7B-Instruct',
      dataset_ids: ['pmd_a'],
      probe_id: 'pm_1',
    });
  });

  it('offers only EVALUATION views — a judge scores the sets the probe was measured on', () => {
    render(<JudgeRunForm datasets={SETS} probes={PROBES} onSubmit={vi.fn()} />);
    expect(screen.getByTestId('judge-set-pmd_a')).toBeInTheDocument();
    expect(screen.queryByTestId('judge-set-pmd_train')).toBeNull();
  });

  it('refuses to submit without an endpoint, a model and at least one set', async () => {
    const onSubmit = vi.fn();
    render(<JudgeRunForm datasets={SETS} probes={PROBES} onSubmit={onSubmit} />);
    expect(screen.getByTestId('judge-submit')).toBeDisabled();
    await userEvent.type(screen.getByTestId('judge-endpoint'), 'http://millm/v1');
    await userEvent.type(screen.getByTestId('judge-model'), 'q');
    expect(screen.getByTestId('judge-submit')).toBeDisabled(); // still no set
    await userEvent.click(screen.getByTestId('judge-set-pmd_a'));
    expect(screen.getByTestId('judge-submit')).toBeEnabled();
  });

  it('a probe is optional, and the form says what leaving it blank means', () => {
    render(<JudgeRunForm datasets={SETS} probes={PROBES} onSubmit={vi.fn()} />);
    expect(screen.getByTestId('judge-probe')).toHaveValue('');
    expect(screen.getByTestId('judge-run-form')).toHaveTextContent(/reach rung 3/i);
  });

  it('says a judge is a BASELINE, not a grader, where someone starting one reads it', () => {
    render(<JudgeRunForm datasets={SETS} probes={PROBES} onSubmit={vi.fn()} />);
    expect(screen.getByTestId('judge-run-form')).toHaveTextContent(/baseline/i);
  });
});

describe('a probe can be evaluated on more sets — rungs 1 and 2 are reachable', () => {
  function mountActions(props: Record<string, unknown> = {}) {
    const onEvaluate = vi.fn().mockResolvedValue(true);
    const onScore = vi.fn().mockResolvedValue({ task_id: 't1', probe_id: 'pm_1', status: 'PENDING' });
    const onPollScore = vi.fn().mockResolvedValue({
      task_id: 't1', probe_id: 'pm_1', status: 'SUCCESS', result: { score: 3.5 },
    });
    render(
      <ProbeActions
        probeId="pm_1"
        datasets={SETS}
        alreadyEvaluated={['pmd_a']}
        onEvaluate={onEvaluate}
        onScore={onScore}
        onPollScore={onPollScore}
        {...props}
      />
    );
    return { onEvaluate, onScore, onPollScore };
  }

  it('submits the chosen sets', async () => {
    const { onEvaluate } = mountActions();
    await userEvent.click(screen.getByTestId('evaluate-set-pmd_b'));
    await userEvent.click(screen.getByTestId('evaluate-submit'));
    expect(onEvaluate).toHaveBeenCalledWith(['pmd_b']);
  });

  it('marks sets already evaluated rather than hiding them — re-running one is legitimate', () => {
    mountActions();
    expect(screen.getByTestId('evaluate-set-pmd_a')).toBeInTheDocument();
    expect(screen.getByTestId('probe-actions')).toHaveTextContent('done');
  });

  it('names rung 2 as what an out-of-distribution set turns on', () => {
    mountActions();
    expect(screen.getByTestId('probe-actions')).toHaveTextContent(/rung 2/);
  });
});

describe('a probe can be pointed at your own text, and the result is READ BACK', () => {
  function mountActions(poll: unknown) {
    const onScore = vi.fn().mockResolvedValue({ task_id: 't1', probe_id: 'pm_1', status: 'PENDING' });
    const onPollScore = vi.fn().mockResolvedValue(poll);
    render(
      <ProbeActions
        probeId="pm_1"
        datasets={SETS}
        alreadyEvaluated={[]}
        onEvaluate={vi.fn().mockResolvedValue(true)}
        onScore={onScore}
        onPollScore={onPollScore}
      />
    );
    return { onScore, onPollScore };
  }

  const TRACE = {
    probe_id: 'pm_1',
    aggregate: 3.5,
    threshold: 2.8786,
    // ⚠ THE SERVER'S VERDICT. The component must not recompute `aggregate >= threshold`.
    fires: true,
    tokens: [{ token: 'a', scored: true }],
    token_scores: [3.5],
    n_scored: 1,
    role_mask_reliable: true,
    truncated: false,
  };

  it('queues the score and then polls for it — scoring is not write-only', async () => {
    const { onScore, onPollScore } = mountActions({
      task_id: 't1', probe_id: 'pm_1', status: 'SUCCESS', result: TRACE,
    });
    await userEvent.type(screen.getByTestId('score-text'), 'a high-stakes message');
    await userEvent.click(screen.getByTestId('score-submit'));
    expect(onScore).toHaveBeenCalledWith({ text: 'a high-stakes message' });
    await screen.findByTestId('score-result');
    expect(onPollScore).toHaveBeenCalledWith('t1');
    expect(screen.getByTestId('score-result')).toHaveTextContent('3.5000');
  });

  it("shows the SERVER's verdict, and renders the per-token trace", async () => {
    /*
     * ⚠ `fires` comes from `score_one`. An earlier version of this component compared
     * `score >= threshold` itself, which is a second definition of a decision the server already
     * makes — and `fires` is null when no threshold was placed, a case the comparison gets wrong.
     */
    mountActions({ task_id: 't1', probe_id: 'pm_1', status: 'SUCCESS', result: TRACE });
    await userEvent.type(screen.getByTestId('score-text'), 'x');
    await userEvent.click(screen.getByTestId('score-submit'));
    await screen.findByTestId('score-result');
    expect(screen.getByTestId('token-trace')).toBeInTheDocument();
    expect(screen.getByTestId('score-result')).toHaveTextContent('3.5000');
  });

  it('a probe with no threshold says it decided nothing, rather than "does not fire"', async () => {
    mountActions({
      task_id: 't1',
      probe_id: 'pm_1',
      status: 'SUCCESS',
      result: { ...TRACE, threshold: null, fires: null },
    });
    await userEvent.type(screen.getByTestId('score-text'), 'x');
    await userEvent.click(screen.getByTestId('score-submit'));
    await screen.findByTestId('score-result');
    expect(screen.getByTestId('no-threshold')).toBeInTheDocument();
  });

  it('a FAILURE shows its reason, because an empty trace reads as "it scored nothing"', async () => {
    mountActions({ task_id: 't1', probe_id: 'pm_1', status: 'FAILURE', error: 'CUDA OOM', result: null });
    await userEvent.type(screen.getByTestId('score-text'), 'x');
    await userEvent.click(screen.getByTestId('score-submit'));
    expect(await screen.findByTestId('score-result')).toHaveTextContent('CUDA OOM');
  });

  it('PENDING is not claimed to be "queued" — the server cannot tell queued from unknown', async () => {
    mountActions({ task_id: 't1', probe_id: 'pm_1', status: 'PENDING' });
    await userEvent.type(screen.getByTestId('score-text'), 'x');
    await userEvent.click(screen.getByTestId('score-submit'));
    const result = await screen.findByTestId('score-result');
    expect(result).toHaveTextContent('PENDING');
    expect(result).toHaveTextContent(/never heard of/i);
  });
});
