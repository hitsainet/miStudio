/**
 * MIS-E2E-121 / -122 / -123 / -124 — frontend state defects, all permanent.
 *
 * Each is reachable by ordinary UI use and none of them surfaces an error.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('MIS-E2E-122 · rebalance must not flip a suppressing feature', () => {
  /**
   * The budget rebalance derived a member's sign from `strength`, and its
   * over-budget branch zeroes unpinned members first. So dragging a slider past
   * the budget and back read the sign off a ZERO — `0 < 0` is false — and a
   * suppressing feature came back amplifying, at a strength the budget model
   * chose, with no error and no visual cue.
   *
   * Negative strength is canonical here: the cluster contract carries
   * `sign ∈ {1,-1}` and a member's negative strength IS its direction.
   */
  it('derives direction from the persisted sign, not the current magnitude', async () => {
    const { directionOf } = await import('./steeringStore');
    // A feature the budget has zeroed still knows which way it points.
    expect(directionOf({ strength: 0, sign: -1 })).toBe(-1);
    expect(directionOf({ strength: 0, sign: 1 })).toBe(1);
  });

  it('falls back to the magnitude when no sign was ever recorded', async () => {
    // Back-compatibility: a feature that has never been zeroed carries no sign.
    const { directionOf } = await import('./steeringStore');
    expect(directionOf({ strength: -2.5 })).toBe(-1);
    expect(directionOf({ strength: 2.5 })).toBe(1);
  });

  it('a zero with no sign reads as positive — which is why the sign is stored', async () => {
    // The defect itself, as a property. This is the value the old code got,
    // and it is why the direction cannot be recovered from magnitude alone.
    const { directionOf } = await import('./steeringStore');
    expect(directionOf({ strength: 0 })).toBe(1);
  });
});

describe('MIS-E2E-123 · in-flight state must not persist across a refresh', () => {
  /**
   * `isGenerating` and `batchState` were written into the persisted slice and
   * nothing cleared them on rehydration, so refreshing mid-batch left
   * `selectCanGenerateBatch` false forever — the panel's primary action
   * disabled, and `abortBatch` unable to help because it drives an in-memory
   * loop that no longer exists. Recovery required clearing localStorage.
   */
  it('the persisted slice carries no in-flight flags', async () => {
    const mod = await import('./steeringStore');
    const src = mod as unknown as Record<string, unknown>;
    expect(src).toBeTruthy();

    // Read the partialize output for a state that IS generating.
    const { useSteeringStore } = mod;
    useSteeringStore.setState({ isGenerating: true, batchState: null } as never);

    const persisted = JSON.parse(
      localStorage.getItem('miStudio-steering') ?? '{}',
    );
    const state = persisted?.state ?? {};
    expect(state.isGenerating).toBeUndefined();
    expect(state.batchState).toBeUndefined();
  });

  it('a durable task id IS still persisted — that is the real recovery path', async () => {
    const { useSteeringStore } = await import('./steeringStore');
    useSteeringStore.setState({ taskId: 'task-abc' } as never);
    const persisted = JSON.parse(localStorage.getItem('miStudio-steering') ?? '{}');
    expect(persisted?.state?.taskId).toBe('task-abc');
  });
});

describe('MIS-E2E-124 · generateCombined needs the guard its sibling has', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it('refuses a second concurrent combined generation', async () => {
    const { useSteeringStore } = await import('./steeringStore');
    useSteeringStore.setState({
      isCombinedGenerating: true,
      selectedSAE: { id: 'sae_1' },
      selectedFeatures: [{ instance_id: 'i1', strength: 1 }],
      prompts: ['hello'],
    } as never);

    await expect(
      useSteeringStore.getState().generateCombined(),
    ).rejects.toThrow(/already running/i);
  });
});

describe('MIS-E2E-121 · a stale request must not disable cancellation forever', () => {
  /**
   * The cleanup request's completion handlers nulled the SHARED abort
   * controller and timeout refs without checking they still owned them. An
   * older request finishing after a newer one started therefore cleared the
   * newer one's controller — and from then on the controller was `null` for
   * the rest of the session, so nothing could be cancelled and the 5-second
   * hard timeout never fired again.
   *
   * Two rapid feature switches are enough, and clicking through features
   * quickly is the primary interaction of the Feature Browser. Permanent,
   * silent, and caused by ordinary use.
   */
  it('a slow first request does not clear the second request‘s controller', async () => {
    vi.resetModules();

    const aborted: string[] = [];
    let resolveFirst: ((v: unknown) => void) | undefined;
    let call = 0;

    vi.doMock('axios', () => ({
      default: {
        post: (_u: string, _b: unknown, cfg: { signal: AbortSignal }) => {
          call += 1;
          const which = call === 1 ? 'first' : 'second';
          cfg.signal.addEventListener('abort', () => aborted.push(which));
          if (call === 1) {
            return new Promise((res) => {
              resolveFirst = res;
            });
          }
          return new Promise(() => {});   // second stays pending
        },
      },
    }));

    const { useFeaturesStore } = await import('./featuresStore');

    // Switch features twice, quickly.
    useFeaturesStore.getState().clearSelectedFeature();   // starts request 1
    useFeaturesStore.getState().clearSelectedFeature();   // aborts 1, starts 2

    expect(aborted).toContain('first');

    // Request 1's handler now completes, LATE.
    resolveFirst?.({ data: { vram_freed_gb: 0 } });
    await Promise.resolve();
    await Promise.resolve();

    // A third switch must still be able to abort request 2. If the late
    // handler cleared the shared ref, this abort never happens.
    useFeaturesStore.getState().clearSelectedFeature();

    expect(aborted).toContain('second');
  });
});
