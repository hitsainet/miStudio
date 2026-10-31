/**
 * Steering results survive a reload (operator bug, 2026-10-28).
 *
 * Reloading the Steering page used to reset it and drop the results of earlier
 * steering sessions. A user clears results only on purpose, with the reset and
 * clear buttons. Pinned here:
 *
 * - a reload (a fresh module instance hydrating from localStorage, as a new tab
 *   does) keeps the single comparison, a batch's results, a sweep and a blended
 *   result;
 * - a batch interrupted by the reload is never "running" afterwards, and the
 *   prompt that was running reads failed, not pending forever (MIS-E2E-123);
 * - the reset button (first click) and the clear buttons clear the persisted
 *   copy too, so a reload after a reset shows nothing;
 * - an in-flight task resumes after a reload — even when "Recent" is not empty
 *   (the old recovery skipped every task once one recent comparison existed);
 * - a finished task's id is not persisted as in-flight;
 * - the persisted results are bounded and storage failures never crash.
 *
 * Defects of the first cut (db96fda6), pinned since the 2026-10-28 incident review:
 * - a pre-v1 payload's task id (v0 never cleared a finished one) was "recovered":
 *   an expired result reads "pending", so the panel sat generating forever;
 * - a recovered task the server reports cancelled ("revoked") read as running;
 * - a late completion or failure between batch prompts ended the batch's
 *   generating state and cleared the task id the loop had just published;
 * - a recovery while a batch runs in this tab marked the batch finished.
 * MUTATION CONTROLS (2026-10-28, each verified RED): M7 no running-batch guard
 * in handleAsyncCompleted; M8 no "something is waiting" guard before the
 * sweep/blended fall-through; M9 no running-batch guard in handleAsyncFailed;
 * M10 recovery does not route to a waiting loop; M11 revoked treated as
 * running; M12 a v0 task id kept by migrate.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type {
  BatchState,
  CombinedSteeringResponse,
  SteeringComparisonResponse,
  StrengthSweepResponse,
} from '../types/steering';

vi.mock('../api/steering', () => ({
  submitAsyncComparison: vi.fn(),
  submitAsyncCombined: vi.fn(),
  submitAsyncSweep: vi.fn(),
  cancelTask: vi.fn(),
  getTaskResult: vi.fn(),
  getExperiments: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
}));

const KEY = 'miStudio-steering';

const comparison: SteeringComparisonResponse = {
  comparison_id: 'cmp-reload-1',
  prompt: 'Tell me about the sea',
  unsteered_output: 'The sea is large.',
  steered_outputs: [{ feature_idx: 7, layer: 3, strength: 2, output: 'The SEA!', color: 'emerald' }],
  created_at: '2026-10-28T12:00:00Z',
} as unknown as SteeringComparisonResponse;

const sweep = {
  sweep_id: 'sweep-reload-1',
  prompt: 'p',
  feature_idx: 7,
  layer: 3,
  strength_values: [1, 2],
  outputs: [{ strength: 1, output: 'a' }, { strength: 2, output: 'b' }],
  created_at: '2026-10-28T12:00:00Z',
} as unknown as StrengthSweepResponse;

const combined = {
  combined_id: 'cmb-reload-1',
  prompt: 'p',
  combined_output: 'blended text',
  baseline_output: 'base text',
  features_applied: [
    { feature_idx: 7, layer: 3, strength: 2, label: 'sea' },
    { feature_idx: 9, layer: 3, strength: 1, label: 'sky' },
  ],
  total_steering_strength: 3,
  total_time_ms: 12,
  created_at: '2026-10-28T12:00:00Z',
} as unknown as CombinedSteeringResponse;

const batch: BatchState = {
  isRunning: true,
  currentIndex: 1,
  totalPrompts: 3,
  aborted: false,
  results: [
    { prompt: 'one', status: 'completed', comparison },
    { prompt: 'two', status: 'running' },
    { prompt: 'three', status: 'pending' },
  ],
} as unknown as BatchState;

/** A new tab: fresh modules, the store hydrating from whatever localStorage holds. */
async function reload() {
  vi.resetModules();
  const mod = await import('./steeringStore');
  const api = await import('../api/steering');
  await mod.useSteeringStore.persist.rehydrate();
  return { ...mod, api: vi.mocked(api) };
}

function persisted(): Record<string, unknown> {
  return (JSON.parse(localStorage.getItem(KEY) ?? '{}')?.state ?? {}) as Record<string, unknown>;
}

describe('steering results survive a reload', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
    (window as unknown as { _steeringLastReset?: number })._steeringLastReset = 0;
  });

  afterEach(() => {
    localStorage.clear();
  });

  it('keeps the comparison, the batch, the sweep and the blended result', async () => {
    const first = await reload();
    first.useSteeringStore.setState({
      currentComparison: comparison,
      comparisonId: comparison.comparison_id,
      sweepResults: sweep,
      combinedResults: combined,
      combinedResultsTitle: 'Blended: sea + sky',
      batchState: batch,
      prompts: ['Tell me about the sea'],
    });

    const { useSteeringStore } = await reload();
    const s = useSteeringStore.getState();
    expect(s.currentComparison).toEqual(comparison);
    expect(s.comparisonId).toBe('cmp-reload-1');
    expect(s.sweepResults).toEqual(sweep);
    expect(s.combinedResults).toEqual(combined);
    expect(s.combinedResultsTitle).toBe('Blended: sea + sky');
    expect(s.prompts).toEqual(['Tell me about the sea']);
    // the batch's finished result is kept; its loop is gone, so it is not running
    expect(s.batchState?.isRunning).toBe(false);
    expect(s.batchState?.currentIndex).toBe(1);
    expect(s.batchState?.results[0]).toEqual({ prompt: 'one', status: 'completed', comparison });
    expect(s.batchState?.results.slice(1).map((r) => [r.status, r.error])).toEqual([
      ['failed', 'Interrupted by a page reload'],
      ['failed', 'Interrupted by a page reload'],
    ]);
    expect(s.isGenerating).toBe(false);
  });

  it('a batch never persists under batchState and a legacy running batchState is dropped (MIS-E2E-123)', async () => {
    const first = await reload();
    first.useSteeringStore.setState({ batchState: batch });
    expect(persisted().batchState).toBeUndefined();
    expect((persisted().batchResults as BatchState).results).toHaveLength(3);

    localStorage.setItem(KEY, JSON.stringify({
      version: 0,
      state: { prompts: ['legacy'], batchState: { ...batch }, isGenerating: true },
    }));
    const { useSteeringStore, selectCanGenerateBatch } = await reload();
    const s = useSteeringStore.getState();
    expect(s.prompts).toEqual(['legacy']); // a v0 payload's configuration is kept
    expect(s.batchState).toBeNull();
    expect(s.isGenerating).toBe(false);
    useSteeringStore.setState({ selectedSAE: { id: 's' } as never, selectedFeatures: [{}] as never, prompts: ['a', 'b'] });
    expect(selectCanGenerateBatch(useSteeringStore.getState())).toBe(true);
  });

  it('the reset button clears the results and the persisted copy, and they stay gone after a reload', async () => {
    const first = await reload();
    first.useSteeringStore.setState({
      currentComparison: comparison,
      comparisonId: comparison.comparison_id,
      sweepResults: sweep,
      combinedResults: combined,
      combinedResultsTitle: 't',
      batchState: { ...batch, isRunning: false },
      prompts: ['kept'],
    });
    first.useSteeringStore.getState().resetSession(); // first click: results only

    const p = persisted();
    expect(p.currentComparison).toBeNull();
    expect(p.sweepResults).toBeNull();
    expect(p.combinedResults).toBeNull();
    expect(p.batchResults).toBeNull();

    const { useSteeringStore } = await reload();
    const s = useSteeringStore.getState();
    expect([s.currentComparison, s.sweepResults, s.combinedResults, s.batchState]).toEqual([null, null, null, null]);
    expect(s.prompts).toEqual(['kept']); // the first click keeps the configuration
  });

  it('a full reset (second click) leaves nothing to restore', async () => {
    const first = await reload();
    first.useSteeringStore.setState({ currentComparison: comparison, prompts: ['gone'] });
    first.useSteeringStore.getState().resetSession();
    first.useSteeringStore.getState().resetSession();
    const { useSteeringStore } = await reload();
    expect(useSteeringStore.getState().currentComparison).toBeNull();
    expect(useSteeringStore.getState().prompts).toEqual(['']);
  });

  it('each clear button clears its persisted copy', async () => {
    const first = await reload();
    first.useSteeringStore.setState({
      currentComparison: comparison,
      comparisonId: comparison.comparison_id,
      sweepResults: sweep,
      combinedResults: combined,
      batchState: { ...batch, isRunning: false },
    });
    const st = first.useSteeringStore.getState();
    st.clearBatchResults();
    expect(persisted().batchResults).toBeNull();
    st.clearCombinedResults();
    expect(persisted().combinedResults).toBeNull();
    st.clearSweepResults();
    expect(persisted().sweepResults).toBeNull();
    st.clearComparison();
    expect(persisted().currentComparison).toBeNull();
  });

  it('an in-flight task resumes after a reload even with recent comparisons', async () => {
    const first = await reload();
    first.useSteeringStore.setState({
      taskId: 'task-running-1',
      isGenerating: true,
      recentComparisons: [{ id: 'old', prompt: 'x', timestamp: 't', result: comparison }],
    });

    const { useSteeringStore, api } = await reload();
    expect(useSteeringStore.getState().isGenerating).toBe(false); // nothing runs in a fresh tab…
    api.getTaskResult.mockResolvedValueOnce({
      task_id: 'task-running-1',
      status: { status: 'progress', percent: 40, message: 'Generating' },
      result: null,
    } as never);
    await useSteeringStore.getState().recoverActiveTask();
    expect(api.getTaskResult).toHaveBeenCalledTimes(1);
    expect(api.getTaskResult).toHaveBeenCalledWith('task-running-1');
    // …until the recovery reconnects: the panel shows it running and polls it
    let s = useSteeringStore.getState();
    expect([s.isGenerating, s.taskId, s.progress]).toEqual([true, 'task-running-1', 40]);

    // it finishes: the result is shown and the finished task is no longer in flight
    s.handleAsyncCompleted(comparison);
    s = useSteeringStore.getState();
    expect(s.currentComparison).toEqual(comparison);
    expect([s.isGenerating, s.taskId]).toEqual([false, null]);
    expect(persisted().taskId).toBeNull();
  });

  it('a task that finished while the page was away is restored, a sweep as a sweep', async () => {
    const first = await reload();
    first.useSteeringStore.setState({ taskId: 'task-sweep-1' });
    const { useSteeringStore, api } = await reload();
    api.getTaskResult.mockResolvedValueOnce({
      task_id: 'task-sweep-1', status: { status: 'success' }, result: sweep,
    } as never);
    await useSteeringStore.getState().recoverActiveTask();
    const s = useSteeringStore.getState();
    expect(s.sweepResults).toEqual(sweep);
    expect(s.currentComparison).toBeNull(); // never cast into a comparison
    expect([s.isGenerating, s.taskId]).toEqual([false, null]);
  });

  it('a resumed blended task lands in combinedResults when no resolver is waiting', async () => {
    const { useSteeringStore } = await reload();
    useSteeringStore.setState({ taskId: 'task-cmb', isGenerating: true });
    useSteeringStore.getState().handleAsyncCompleted(combined);
    const s = useSteeringStore.getState();
    expect(s.combinedResults).toEqual(combined);
    expect([s.isGenerating, s.isCombinedGenerating, s.taskId]).toEqual([false, false, null]);
  });

  it('results over the size bound are not written; the configuration still is', async () => {
    const { useSteeringStore, PERSISTED_RESULTS_MAX_CHARS } = await reload();
    const huge = { ...comparison, unsteered_output: 'x'.repeat(PERSISTED_RESULTS_MAX_CHARS + 1) };
    useSteeringStore.setState({ currentComparison: huge, prompts: ['still kept'] });
    const p = persisted();
    expect(p.currentComparison).toBeNull();
    expect(p.prompts).toEqual(['still kept']);
    expect(useSteeringStore.getState().currentComparison).toEqual(huge); // the open tab keeps showing it
  });

  it('a storage that refuses writes never breaks the store', async () => {
    const { useSteeringStore } = await reload();
    const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota', 'QuotaExceededError');
    });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(() => useSteeringStore.setState({ currentComparison: comparison })).not.toThrow();
    expect(useSteeringStore.getState().currentComparison).toEqual(comparison);
    expect(spy).toHaveBeenCalled();
    expect(warn).toHaveBeenCalled();
  });

  // ---- defects of the first cut (db96fda6), found in the 2026-10-28 incident review ----

  it("a pre-v1 payload's task id is dropped: it names the LAST task, not a running one", async () => {
    // v0 code never cleared a finished task's id. Recovering it asked the backend
    // about a long-finished (or expired → "pending") task and marked the panel
    // generating forever, so Generate refused every new run.
    localStorage.setItem(
      KEY,
      JSON.stringify({
        version: 0,
        state: {
          taskId: 'last-task-from-v0',
          prompts: ['kept'],
          recentComparisons: [{ id: 'old', prompt: 'x', timestamp: 't', result: comparison }],
        },
      }),
    );
    const { useSteeringStore, api } = await reload();
    const s = useSteeringStore.getState();
    expect(s.taskId).toBeNull();
    expect(s.prompts).toEqual(['kept']); // the configuration survives the migration
    expect(s.recentComparisons).toHaveLength(1);
    await s.recoverActiveTask();
    expect(api.getTaskResult).not.toHaveBeenCalled();
    expect(useSteeringStore.getState().isGenerating).toBe(false);
  });

  it('a recovered task the server reports cancelled is finished, not running', async () => {
    const first = await reload();
    first.useSteeringStore.setState({ taskId: 'task-revoked' });
    const { useSteeringStore, api } = await reload();
    api.getTaskResult.mockResolvedValueOnce({
      task_id: 'task-revoked', status: { status: 'revoked', percent: 0, message: '' }, result: null,
    } as never);
    await useSteeringStore.getState().recoverActiveTask();
    const s = useSteeringStore.getState();
    expect([s.isGenerating, s.taskId]).toEqual([false, null]);
  });

  it('recovery while a batch runs in this tab hands the result to the batch instead of ending it', async () => {
    const { useSteeringStore, api } = await reload();
    useSteeringStore.setState({
      selectedSAE: { id: 'sae-1' } as never,
      prompts: ['one', 'two'],
      isGenerating: false,
      batchState: null,
      taskId: null,
    });
    useSteeringStore.getState().addFeature({ feature_idx: 7, layer: 3, strength: 2 } as never);
    let n = 0;
    api.submitAsyncComparison.mockImplementation((async () => ({ task_id: `bt-${++n}`, status: 'pending' })) as never);
    const batch = useSteeringStore.getState().generateBatchComparison(true, false);
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('bt-1'));

    // prompt 1 finished; a recovery (e.g. a remount) learns it from the result endpoint
    api.getTaskResult.mockResolvedValueOnce({
      task_id: 'bt-1', status: { status: 'success' }, result: { ...comparison, comparison_id: 'cmp-bt-1' },
    } as never);
    await useSteeringStore.getState().recoverActiveTask();

    // the batch was NOT marked finished: prompt 1 completed and prompt 2 was submitted
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('bt-2'));
    let s = useSteeringStore.getState();
    expect(s.batchState?.isRunning).toBe(true);
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'running']);
    expect(s.batchState?.results[0].comparison?.comparison_id).toBe('cmp-bt-1');

    s.handleAsyncCompleted({ ...comparison, comparison_id: 'cmp-bt-2' });
    await batch;
    s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed']);
    expect([s.batchState?.isRunning, s.isGenerating, s.taskId]).toEqual([false, false, null]);
  });

  it('a late completion of the previous batch task, between prompts, does not end the batch', async () => {
    const { useSteeringStore, api } = await reload();
    useSteeringStore.setState({
      selectedSAE: { id: 'sae-1' } as never, prompts: ['one', 'two'], isGenerating: false, batchState: null, taskId: null,
    });
    useSteeringStore.getState().addFeature({ feature_idx: 7, layer: 3, strength: 2 } as never);
    let releaseSecondSubmit: () => void = () => {};
    let n = 0;
    api.submitAsyncComparison.mockImplementation((async () => {
      n += 1;
      if (n === 2) await new Promise<void>((r) => { releaseSecondSubmit = r; });
      return { task_id: `lt-${n}`, status: 'pending' };
    }) as never);
    const batch = useSteeringStore.getState().generateBatchComparison(true, false);
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('lt-1'));
    useSteeringStore.getState().handleAsyncCompleted({ ...comparison, comparison_id: 'cmp-lt-1' });
    await vi.waitFor(() => expect(n).toBe(2)); // prompt 2's submit is out; nothing waits on lt-1 any more

    // a duplicate of prompt 1's completion arrives now (a second emit, a slow poll)
    useSteeringStore.getState().handleAsyncCompleted({ ...comparison, comparison_id: 'cmp-lt-1' });
    let s = useSteeringStore.getState();
    // the first cut ended the batch's generating state and cleared the task id here,
    // which stopped the fallback poll for every later prompt
    expect([s.isGenerating, s.batchState?.isRunning, s.currentComparison]).toEqual([true, true, null]);

    releaseSecondSubmit();
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('lt-2'));
    useSteeringStore.getState().handleAsyncCompleted({ ...comparison, comparison_id: 'cmp-lt-2' });
    await batch;
    s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.comparison?.comparison_id)).toEqual(['cmp-lt-1', 'cmp-lt-2']);
    expect([s.batchState?.isRunning, s.isGenerating, s.taskId]).toEqual([false, false, null]);
  });

  it('between prompts, a late failure or a late blended result does not end a running batch', async () => {
    const { useSteeringStore, api } = await reload();
    useSteeringStore.setState({
      selectedSAE: { id: 'sae-1' } as never, prompts: ['one', 'two'], isGenerating: false, batchState: null,
      taskId: null, combinedMode: true,
    });
    useSteeringStore.getState().addFeature({ feature_idx: 7, layer: 3, strength: 2 } as never);
    useSteeringStore.getState().addFeature({ feature_idx: 9, layer: 3, strength: 1 } as never);
    let releaseSecondSubmit: () => void = () => {};
    let n = 0;
    api.submitAsyncCombined.mockImplementation((async () => {
      n += 1;
      if (n === 2) await new Promise<void>((r) => { releaseSecondSubmit = r; });
      return { task_id: `bb-${n}`, status: 'pending' };
    }) as never);
    const batch = useSteeringStore.getState().generateBatchComparison(true, false);
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('bb-1'));
    useSteeringStore.getState().handleAsyncCompleted({ ...combined, combined_id: 'cmb-bb-1' });
    await vi.waitFor(() => expect(n).toBe(2)); // prompt 2's submit is out; nothing waits now

    // late events for prompt 1's task: a duplicate blended result, then a failure
    useSteeringStore.getState().handleAsyncCompleted({ ...combined, combined_id: 'cmb-bb-1' });
    useSteeringStore.getState().handleAsyncFailed('late failure of a finished task');
    let s = useSteeringStore.getState();
    expect([s.isGenerating, s.batchState?.isRunning, s.combinedResults, s.error]).toEqual([true, true, null, null]);

    releaseSecondSubmit();
    await vi.waitFor(() => expect(useSteeringStore.getState().taskId).toBe('bb-2'));
    useSteeringStore.getState().handleAsyncCompleted({ ...combined, combined_id: 'cmb-bb-2' });
    await batch;
    s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed']);
  });

  it('the persisted slice is versioned', async () => {
    const { useSteeringStore, STEERING_PERSIST_VERSION } = await reload();
    useSteeringStore.setState({ prompts: ['v'] });
    expect(JSON.parse(localStorage.getItem(KEY)!).version).toBe(STEERING_PERSIST_VERSION);
    expect(STEERING_PERSIST_VERSION).toBeGreaterThanOrEqual(1);
  });
});
