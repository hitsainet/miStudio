/**
 * The card must reconcile against the server, not only listen to it.
 *
 * Reported 2026-08-27: a completed extraction displayed "Extracting
 * Activations 90.0%" for 17 minutes. The worker had emitted
 * `progress=100, status="complete"` — the log line `fully completed and saved`
 * only runs after that emit — but the browser was not connected to hear it.
 *
 * WebSocket events are fire-and-forget with no replay, so anything sent while
 * the socket is down is lost permanently. `checkActiveExtraction` ran on MOUNT
 * only, and when the server reported nothing active it returned false and
 * changed nothing, so the stale value survived. Reloading appeared to fix it
 * only because a fresh page starts with an empty store.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useModelsStore } from './modelsStore';

const MODEL_ID = 'm_b55c6926';

function seed(extraction: Record<string, unknown>) {
  useModelsStore.setState({
    models: [{ id: MODEL_ID, name: 'gemma-4-12B-it', ...extraction }],
  } as never);
}

function serverSays(data: unknown) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ data }),
  }) as never;
}

describe('checkActiveExtraction reconciles stale state', () => {
  beforeEach(() => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  afterEach(() => vi.restoreAllMocks());

  it('clears an unfinished extraction the server no longer reports', async () => {
    // exactly the observed state: stuck at the "saving" event
    seed({
      extraction_id: 'ext_x',
      extraction_progress: 90,
      extraction_status: 'saving',
      extraction_message: 'Saved 2 activation files',
    });
    serverSays(null);

    const active = await useModelsStore.getState().checkActiveExtraction(MODEL_ID);

    expect(active).toBe(false);
    const model = useModelsStore.getState().models[0] as never as Record<string, unknown>;
    expect(model.extraction_status).toBeUndefined();
    expect(model.extraction_progress).toBeUndefined();
    expect(model.extraction_id).toBeUndefined();
  });

  it('leaves an already-settled model alone', async () => {
    seed({ extraction_status: 'complete', extraction_progress: 100 });
    serverSays(null);

    await useModelsStore.getState().checkActiveExtraction(MODEL_ID);

    const model = useModelsStore.getState().models[0] as never as Record<string, unknown>;
    expect(model.extraction_status).toBe('complete');
  });

  it('does not touch a model with no extraction state', async () => {
    seed({});
    serverSays(null);

    await useModelsStore.getState().checkActiveExtraction(MODEL_ID);

    const model = useModelsStore.getState().models[0] as never as Record<string, unknown>;
    expect(model.extraction_status).toBeUndefined();
    expect(model.name).toBe('gemma-4-12B-it');
  });

  it('restores state when the server DOES report an active extraction', async () => {
    seed({});
    serverSays({
      extraction_id: 'ext_live',
      progress: 44.7,
      status: 'extracting',
      samples_processed: 4336,
      max_samples: 10000,
    });

    const active = await useModelsStore.getState().checkActiveExtraction(MODEL_ID);

    expect(active).toBe(true);
    const model = useModelsStore.getState().models[0] as never as Record<string, unknown>;
    expect(model.extraction_status).toBe('extracting');
    expect(model.extraction_progress).toBe(44.7);
  });

  it('does not clear another model', async () => {
    useModelsStore.setState({
      models: [
        { id: MODEL_ID, extraction_status: 'saving', extraction_progress: 90 },
        { id: 'm_other', extraction_status: 'extracting', extraction_progress: 12 },
      ],
    } as never);
    serverSays(null);

    await useModelsStore.getState().checkActiveExtraction(MODEL_ID);

    const other = useModelsStore.getState().models[1] as never as Record<string, unknown>;
    expect(other.extraction_status).toBe('extracting');
    expect(other.extraction_progress).toBe(12);
  });
});
