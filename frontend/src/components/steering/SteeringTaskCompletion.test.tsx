/**
 * Steering tasks complete in the page — even when the Steering panel is not on
 * screen (incident 2026-10-28).
 *
 * THE INCIDENT. A 5-prompt batch was started on the Steering panel at 17:02:21Z;
 * at 17:02:49Z the operator opened Probe Monitors, then Circuits. From then on
 * every prompt waited out PROMPT_TIMEOUT_MS (5 min), was cancelled
 * (`DELETE /async/task/{id}`) and the next one started, although the server
 * finished each task in ~50 s and emitted `steering:completed`.
 *
 * WHY. The batch loop lives in the store and waits on a per-task resolver, but
 * the ONLY things that resolve it — the `steering/{task_id}` WebSocket
 * subscription and the `/async/result` fallback poll — lived inside
 * SteeringPanel. App renders one panel at a time, so leaving the Steering panel
 * unmounted both, and the loop could only time out. Single compares, sweeps and
 * blended runs lost their completion the same way.
 *
 * db96fda6 ("results survive a page reload") was deployed at the same time and
 * was blamed and reverted, but the session that hung ran a cached bundle that
 * predated it (the db96fda6 bundle was first downloaded at 17:32:40Z), and
 * these tests fail against the revert (7bb2bc3a) exactly as against db96fda6.
 *
 * WHAT IS PINNED. The real App, the real store, the real useSteeringWebSocket
 * over a fake socket bus that only delivers on subscribed channels, and a fake
 * server that finishes every task 50 s after submission. Fake timers: every
 * task must finish at the server's pace with no cancel, on the Steering panel
 * AND after navigating away, over the WebSocket AND (WebSocket dead) through the
 * fallback poll alone.
 *
 * RED/GREEN (2026-10-28, this file + steeringStore.reload.test.ts, 32 tests):
 *   7bb2bc3a (the revert) ........ 30 failed / 2 passed (every leave-the-panel case red)
 *   db96fda6 (the blamed commit) . 20 failed / 12 passed (the same leave-the-panel cases red)
 *   this fix ..................... 32 passed
 *
 * MUTATION CONTROLS (2026-10-28, each run and verified RED, restored from saved bytes):
 *   M1  App does not render <SteeringTaskWatcher />                  -> every leave-the-panel case red
 *   M2  ROOT CAUSE: watcher rendered only while the Steering panel is
 *       (`activePanel === 'steering' && <SteeringTaskWatcher />`)    -> every leave-the-panel case red
 *   M3  poll answer not re-checked against the store's task id       -> slow-poll case red
 *   M4  "in flight" = isGenerating only (sweep / blended not polled) -> 3 red
 *   M5  no WebSocket subscription (poll only)                        -> 3 red
 *   M6  no reload recovery in the watcher                            -> reload case red
 *   M13b batch publishes its task id, waits 50 ms, THEN creates the
 *        resolver (the old order)                                    -> 5 red
 *   M14 Cancel button needs `!batchState` again                      -> cancel case red
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import type { ReactNode } from 'react';
import type { SAE } from '../../types/sae';

// ---- a fake socket: an event reaches a handler only on a subscribed channel
type Handler = (data: unknown) => void;
const bus = {
  handlers: new Map<string, Set<Handler>>(),
  channels: new Map<string, number>(),
  reset() {
    this.handlers.clear();
    this.channels.clear();
  },
  subscribed(channel: string) {
    return (this.channels.get(channel) ?? 0) > 0;
  },
  emit(channel: string, event: string, data: unknown) {
    if (!this.subscribed(channel)) return; // the server sends only to subscribers
    for (const h of [...(this.handlers.get(event) ?? [])]) h(data);
  },
};
const wsContext = {
  isConnected: true,
  socket: null,
  on: (event: string, h: Handler) => {
    if (!bus.handlers.has(event)) bus.handlers.set(event, new Set());
    bus.handlers.get(event)!.add(h);
  },
  off: (event: string, h?: Handler) => {
    if (h) bus.handlers.get(event)?.delete(h);
    else bus.handlers.delete(event);
  },
  subscribe: (c: string) => {
    bus.channels.set(c, (bus.channels.get(c) ?? 0) + 1);
  },
  unsubscribe: (c: string) => {
    bus.channels.set(c, Math.max(0, (bus.channels.get(c) ?? 0) - 1));
  },
};
vi.mock('../../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: ReactNode }) => children,
  useWebSocketContext: () => wsContext,
}));

vi.mock('../../api/steering', () => ({
  getTaskResult: vi.fn(),
  getSteeringModeStatus: vi.fn().mockResolvedValue({ active: true }),
  enterSteeringMode: vi.fn(),
  exitSteeringMode: vi.fn(),
  submitAsyncComparison: vi.fn(),
  submitAsyncCombined: vi.fn(),
  submitAsyncSweep: vi.fn(),
  computeClusterAllocation: vi.fn(),
  cancelTask: vi.fn(),
  getExperiments: vi.fn(),
  getExperiment: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
  cleanupGPU: vi.fn(),
}));

// Every panel but Steering is a stub: they are the places the operator navigates TO.
vi.mock('../panels/DatasetsPanel', () => ({ DatasetsPanel: () => null }));
vi.mock('../panels/ModelsPanel', () => ({ ModelsPanel: () => null }));
vi.mock('../panels/TemplatesPanel', () => ({ TemplatesPanel: () => null }));
vi.mock('../panels/TrainingPanel', () => ({ TrainingPanel: () => null }));
vi.mock('../panels/ExtractionsPanel', () => ({ ExtractionsPanel: () => null }));
vi.mock('../panels/LabelingPanel', () => ({ LabelingPanel: () => null }));
vi.mock('../panels/CircuitsPanel', () => ({ CircuitsPanel: () => <div data-testid="circuits" /> }));
vi.mock('../panels/JLensPanel', () => ({ JLensPanel: () => null }));
vi.mock('../panels/ProbeMonitorsPanel', () => ({ ProbeMonitorsPanel: () => null }));
vi.mock('../panels/SAEsPanel', () => ({ SAEsPanel: () => null }));
vi.mock('../panels/FeatureGroupsPanel', () => ({ FeatureGroupsPanel: () => null }));
vi.mock('../SystemMonitor/SystemMonitor', () => ({ SystemMonitor: () => null }));
vi.mock('../panels/SettingsPanel', () => ({ SettingsPanel: () => null }));
vi.mock('../help/HelpPanel', () => ({ HelpPanel: () => null }));
vi.mock('../layout/Sidebar', () => ({ Sidebar: () => null }));
vi.mock('../layout/Header', () => ({ Header: () => null }));
vi.mock('../layout/ApprovalsBanner', () => ({ ApprovalsBanner: () => null }));
vi.mock('../../hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));
// The Steering panel's own widgets are not under test, and mounting them drags their fetches in.
const SAES_STATE = { saes: [], fetchSAEs: () => {} };
vi.mock('../../stores/saesStore', () => ({
  useSAEsStore: (s?: (x: typeof SAES_STATE) => unknown) => (s ? s(SAES_STATE) : SAES_STATE),
}));
const TEMPLATES_STATE = { templates: [], fetchTemplates: () => {}, createTemplate: () => {} };
vi.mock('../../stores/promptTemplatesStore', () => ({
  usePromptTemplatesStore: (s?: (x: typeof TEMPLATES_STATE) => unknown) => (s ? s(TEMPLATES_STATE) : TEMPLATES_STATE),
}));
vi.mock('./FeatureSelector', () => ({ FeatureSelector: () => null }));
vi.mock('./GenerationConfig', () => ({ GenerationConfig: () => null }));
vi.mock('./PromptListEditor', () => ({ PromptListEditor: () => null }));
vi.mock('./HazardBanner', () => ({ HazardBanner: () => null }));
vi.mock('./ComparisonPreview', () => ({ ComparisonPreview: () => null }));
vi.mock('./ComparisonResults', () => ({ ComparisonResults: () => null }));

import * as api from '../../api/steering';
import App from '../../App';
import { useSteeringStore } from '../../stores/steeringStore';
import { useUIStore } from '../../stores/uiStore';

const SERVER_MS = 50_000; // the server finishes each task in ~50 s
const PROMPT_TIMEOUT_MS = 5 * 60 * 1000;
const sae = { id: 'sae-1', name: 'SAE', model_id: 'm-1', layer: 6, status: 'ready' } as unknown as SAE;

const comparison = (id: string, prompt: string) => ({
  comparison_id: `cmp-${id}`,
  sae_id: 'sae-1',
  model_id: 'm-1',
  prompt,
  unsteered: { text: 'base', metrics: null },
  steered: [],
  total_time_ms: 1,
  created_at: '2026-10-28T12:00:00Z',
});
const combined = (id: string, prompt: string) => ({
  combined_id: `cmb-${id}`,
  prompt,
  combined_output: 'blend',
  baseline_output: 'base',
  features_applied: [{ feature_idx: 100, layer: 6, strength: 1.5, label: 'a' }],
  total_steering_strength: 1.5,
  total_time_ms: 1,
  created_at: '2026-10-28T12:00:00Z',
});
const sweep = (id: string) => ({
  sweep_id: `sweep-${id}`,
  prompt: 'p',
  feature_idx: 100,
  layer: 6,
  strength_values: [1, 2],
  outputs: [],
  created_at: '2026-10-28T12:00:00Z',
});

/**
 * The fake server: a task finishes SERVER_MS after submission. With `ws` it emits
 * `steering:completed` on the task's channel (lost if nobody is subscribed); the
 * result endpoint answers by the clock either way.
 */
function startServer({ ws }: { ws: boolean }) {
  const tasks = new Map<string, { doneAt: number; result: unknown }>();
  let n = 0;
  const submit = (make: (id: string, prompt: string) => unknown) =>
    (async (req: { prompt?: string }) => {
      n += 1;
      const id = `task-${n}`;
      const result = make(id, req?.prompt ?? '');
      tasks.set(id, { doneAt: Date.now() + SERVER_MS, result });
      if (ws) {
        setTimeout(() => {
          bus.emit(`steering/${id}`, 'steering:completed', { task_id: id, percent: 100, message: 'Complete', result });
        }, SERVER_MS);
      }
      return { task_id: id, status: 'pending' };
    }) as never;
  vi.mocked(api.submitAsyncComparison).mockImplementation(submit(comparison));
  vi.mocked(api.submitAsyncCombined).mockImplementation(submit(combined));
  vi.mocked(api.submitAsyncSweep).mockImplementation(submit((id) => sweep(id)));
  vi.mocked(api.getTaskResult).mockImplementation((async (id: string) => {
    const t = tasks.get(id);
    if (!t) return { task_id: id, status: { task_id: id, status: 'pending', percent: 0, message: '' }, result: null };
    if (Date.now() >= t.doneAt) {
      return { task_id: id, status: { task_id: id, status: 'success', percent: 100, message: 'Complete' }, result: t.result };
    }
    return { task_id: id, status: { task_id: id, status: 'progress', percent: 50, message: 'Generating' }, result: null };
  }) as never);
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function mountOnSteering() {
  localStorage.setItem('activePanel', 'steering');
  await act(async () => {
    render(<App />);
  });
  expect(useUIStore.getState().activePanel).toBe('steering');
}

function goTo(panel: 'circuits' | 'steering') {
  act(() => useUIStore.getState().setActivePanel(panel));
}

function configure(prompts: string[], features = 1) {
  act(() => {
    useSteeringStore.setState({
      selectedSAE: sae,
      selectedFeatures: [],
      prompts,
      isGenerating: false,
      isSweeping: false,
      isCombinedGenerating: false,
      combinedMode: false,
      taskId: null,
      batchState: null,
      currentComparison: null,
      comparisonId: null,
      sweepResults: null,
      combinedResults: null,
      recentComparisons: [],
      error: null,
    } as never);
    const s = useSteeringStore.getState();
    s.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 } as never);
    if (features > 1) s.addFeature({ feature_idx: 200, layer: 6, strength: 2 } as never);
  });
}

describe('steering tasks complete in the page wherever the operator is (incident 2026-10-28)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    bus.reset();
    vi.mocked(api.cancelTask).mockReset().mockResolvedValue({} as never);
    vi.mocked(api.getSteeringModeStatus).mockResolvedValue({ active: true } as never);
  });

  afterEach(() => {
    vi.useRealTimers();
    localStorage.clear();
  });

  for (const ws of [true, false]) {
    const via = ws ? 'over the WebSocket' : 'through the fallback poll (WebSocket dead)';

    it(`a Compare batch completes every prompt at the server pace after leaving the panel, ${via}`, async () => {
      startServer({ ws });
      await mountOnSteering();
      configure(['one', 'two', 'three']);

      let done = false;
      const batch = useSteeringStore.getState().generateBatchComparison(true, true).then(() => {
        done = true;
      });
      await advance(1_000);
      goTo('circuits'); // the operator looks at something else while it runs
      expect(useUIStore.getState().activePanel).toBe('circuits');

      // three tasks of ~50 s (+ the poll's ~20 s lag without a socket): well inside ONE prompt timeout
      const budget = 3 * (SERVER_MS + 25_000);
      expect(budget).toBeLessThan(PROMPT_TIMEOUT_MS);
      await advance(budget);

      const s = useSteeringStore.getState();
      expect(s.batchState?.results.map((r) => [r.status, r.error ?? null])).toEqual([
        ['completed', null],
        ['completed', null],
        ['completed', null],
      ]);
      expect(s.batchState?.results.map((r) => r.comparison?.comparison_id)).toEqual([
        'cmp-task-1',
        'cmp-task-2',
        'cmp-task-3',
      ]);
      expect(done).toBe(true);
      expect(s.batchState?.isRunning).toBe(false);
      expect(s.isGenerating).toBe(false);
      expect(api.submitAsyncComparison).toHaveBeenCalledTimes(3);
      expect(api.cancelTask).not.toHaveBeenCalled();
      await batch;
    });

    it(`a single compare lands its result after leaving the panel, ${via}`, async () => {
      startServer({ ws });
      await mountOnSteering();
      configure(['only']);

      await act(async () => {
        await useSteeringStore.getState().generateComparison(true, false);
      });
      goTo('circuits');
      await advance(SERVER_MS + 25_000);

      const s = useSteeringStore.getState();
      expect(s.isGenerating).toBe(false);
      expect(s.currentComparison?.comparison_id).toBe('cmp-task-1');
      expect(s.recentComparisons[0]?.id).toBe('cmp-task-1');
      expect(api.cancelTask).not.toHaveBeenCalled();
    });

    it(`a Blended batch completes every prompt after leaving the panel, ${via}`, async () => {
      startServer({ ws });
      await mountOnSteering();
      configure(['one', 'two'], 2);
      act(() => useSteeringStore.getState().setCombinedMode(true));

      const batch = useSteeringStore.getState().generateBatchComparison(true, true);
      await advance(1_000);
      goTo('circuits');
      await advance(2 * (SERVER_MS + 25_000));

      const s = useSteeringStore.getState();
      expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed']);
      expect(s.batchState?.isRunning).toBe(false);
      expect(api.submitAsyncCombined).toHaveBeenCalledTimes(2);
      expect(api.cancelTask).not.toHaveBeenCalled();
      await batch;
    });

    it(`a single blended run lands after leaving the panel, ${via}`, async () => {
      startServer({ ws });
      await mountOnSteering();
      configure(['only'], 2);

      const run = useSteeringStore.getState().generateCombined(true, false);
      await advance(1_000);
      goTo('circuits');
      await advance(SERVER_MS + 25_000);

      const s = useSteeringStore.getState();
      expect(s.combinedResults?.combined_id).toBe('cmb-task-1');
      expect(s.isCombinedGenerating).toBe(false);
      expect(api.cancelTask).not.toHaveBeenCalled();
      await run;
    });

    it(`a sweep lands after leaving the panel, ${via}`, async () => {
      startServer({ ws });
      await mountOnSteering();
      configure(['only']);

      const run = useSteeringStore.getState().runStrengthSweep(100, 6, [1, 2]);
      await advance(1_000);
      goTo('circuits');
      await advance(SERVER_MS + 25_000);

      const s = useSteeringStore.getState();
      expect(s.sweepResults?.sweep_id).toBe('sweep-task-1');
      expect(s.isSweeping).toBe(false);
      expect(api.cancelTask).not.toHaveBeenCalled();
      await run;
    });
  }

  it('a batch on the panel the whole time completes every prompt (the path that worked)', async () => {
    startServer({ ws: true });
    await mountOnSteering();
    configure(['one', 'two', 'three']);

    const batch = useSteeringStore.getState().generateBatchComparison(true, true);
    await advance(3 * SERVER_MS + 5_000);
    const s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed', 'completed']);
    expect(api.cancelTask).not.toHaveBeenCalled();
    await batch;
  });

  it("a slow poll answer for the previous prompt's task is never handed to the next prompt", async () => {
    // Prompt 1's task is done at 48 s; the poll that starts at 50 s answers at 53 s.
    // The socket delivers the same completion at 50.5 s, so by 53 s the batch is
    // waiting on prompt 2. The late answer is prompt 1's result and must be dropped.
    const tasks = new Map<string, { doneAt: number; result: unknown }>();
    let n = 0;
    vi.mocked(api.submitAsyncComparison).mockImplementation((async (req: { prompt: string }) => {
      n += 1;
      const id = `task-${n}`;
      const result = comparison(id, req.prompt);
      tasks.set(id, { doneAt: Date.now() + 48_000, result });
      setTimeout(() => {
        bus.emit(`steering/${id}`, 'steering:completed', { task_id: id, percent: 100, message: 'Complete', result });
      }, 50_500);
      return { task_id: id, status: 'pending' };
    }) as never);
    vi.mocked(api.getTaskResult).mockImplementation((async (id: string) => {
      const t = tasks.get(id)!;
      const done = Date.now() >= t.doneAt;
      await new Promise((r) => setTimeout(r, 3_000));
      return done
        ? { task_id: id, status: { task_id: id, status: 'success', percent: 100, message: '' }, result: t.result }
        : { task_id: id, status: { task_id: id, status: 'progress', percent: 50, message: '' }, result: null };
    }) as never);

    await mountOnSteering();
    configure(['one', 'two']);
    const batch = useSteeringStore.getState().generateBatchComparison(true, true);
    await advance(54_000);
    let s = useSteeringStore.getState();
    expect(s.taskId).toBe('task-2');
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'running']);

    await advance(60_000);
    s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.comparison?.comparison_id)).toEqual(['cmp-task-1', 'cmp-task-2']);
    expect(api.cancelTask).not.toHaveBeenCalled();
    await batch;
  });

  it('tasks that finish almost at once still complete (the wait exists before the id is published)', async () => {
    let n = 0;
    vi.mocked(api.submitAsyncComparison).mockImplementation((async (req: { prompt: string }) => {
      n += 1;
      const id = `task-${n}`;
      setTimeout(() => {
        bus.emit(`steering/${id}`, 'steering:completed', {
          task_id: id, percent: 100, message: 'Complete', result: comparison(id, req.prompt),
        });
      }, 10);
      return { task_id: id, status: 'pending' };
    }) as never);
    vi.mocked(api.getTaskResult).mockResolvedValue({ status: { status: 'progress' }, result: null } as never);

    await mountOnSteering();
    configure(['one', 'two', 'three']);
    const batch = useSteeringStore.getState().generateBatchComparison(true, true);
    await advance(2_000); // far less than the fallback poll's 15 s silence window
    const s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed', 'completed']);
    await batch;
  });

  it('a task in flight across a reload is recovered and lands, whichever panel the page opens on', async () => {
    startServer({ ws: true });
    // the server already runs task-1 (submitted by the page before the reload)
    await vi.mocked(api.submitAsyncComparison)({ prompt: 'before reload' } as never);
    localStorage.setItem(
      'miStudio-steering',
      JSON.stringify({ version: 1, state: { taskId: 'task-1', prompts: ['before reload'] } }),
    );
    await act(async () => {
      await useSteeringStore.persist.rehydrate();
    });
    localStorage.setItem('activePanel', 'circuits');
    await act(async () => {
      render(<App />);
    });
    expect(useUIStore.getState().activePanel).toBe('circuits');
    expect(api.getTaskResult).toHaveBeenCalledWith('task-1'); // recovered without visiting Steering
    expect(useSteeringStore.getState().isGenerating).toBe(true);

    await advance(SERVER_MS + 1_000);
    const s = useSteeringStore.getState();
    expect(s.currentComparison?.comparison_id).toBe('cmp-task-1');
    expect([s.isGenerating, s.taskId]).toEqual([false, null]);
  });

  it('a running single task can be cancelled while a finished batch from before a reload is still shown', async () => {
    // The first cut rehydrated the batch (never running) and the Cancel button
    // required `!batchState`: a recovered task left the panel generating with no
    // way to stop it.
    startServer({ ws: true });
    configure(['only']);
    act(() => {
      useSteeringStore.setState({
        isGenerating: true,
        taskId: 'task-from-before-reload',
        batchState: { isRunning: false, currentIndex: 0, totalPrompts: 1, aborted: false, results: [] },
      } as never);
    });
    await mountOnSteering();
    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
  });

  it('leaving and coming back mid-batch does not stall the batch', async () => {
    startServer({ ws: true });
    await mountOnSteering();
    configure(['one', 'two', 'three']);

    const batch = useSteeringStore.getState().generateBatchComparison(true, true);
    await advance(1_000);
    goTo('circuits');
    await advance(SERVER_MS + 20_000); // prompt 1 finished while away
    goTo('steering'); // remount: must not clobber the running batch
    await advance(2 * SERVER_MS + 20_000);

    const s = useSteeringStore.getState();
    expect(s.batchState?.results.map((r) => r.status)).toEqual(['completed', 'completed', 'completed']);
    expect(s.batchState?.isRunning).toBe(false);
    expect(api.cancelTask).not.toHaveBeenCalled();
    await batch;
  });
});
