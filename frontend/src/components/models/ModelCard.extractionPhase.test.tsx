/**
 * The card shows the persisted step after a reload, and a live message still moves it.
 *
 * Reported 2026-09-13: "I just saw a more detailed status that said it was in
 * the statistics phase. But when I left the page and came back, it was
 * 'Extracting 20000/20000' again."
 *
 * The step existed only in WebSocket messages. After the GPU pass reaches N/N
 * the row's status stays 'extracting' through the merge and the statistics, so
 * a remount rebuilt the card from `status` and `samples_processed` alone. The
 * backend now stores `phase`, `/extractions/active` returns it, and
 * `checkActiveExtraction` copies it into the store.
 *
 * These tests mount the REAL card, the REAL `useModelExtractionProgress` hook
 * and the REAL store. Only the socket context and `fetch` are faked: a fresh
 * page with an empty store, the server's answer, and a socket that has sent
 * nothing until the test says so.
 *
 * MUTATION CONTROLS, 2026-09-13 (applied, this file run, original bytes restored
 * and hash-checked, `git status` clean afterwards):
 *   M5  checkActiveExtraction stops copying `phase` into the store, so the card
 *       reads the phase only from WebSocket messages  -> 3 failed (both reload
 *       labels, and the live-update test, which never reaches "Merging…")
 *   M9  the WebSocket hook stops passing `data.phase` to the store
 *                                                     -> 1 failed (the live
 *       message no longer moves the card from "Merging…")
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, render, screen, waitFor } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { useModelsStore } from '../../stores/modelsStore';
import { ModelStatus, QuantizationFormat } from '../../types/model';
import { extractionStageLabel } from '../../utils/extractionPhase';

const ws = vi.hoisted(() => {
  const handlers: Record<string, Array<(event: unknown) => void>> = {};
  const context = {
    isConnected: true,
    subscribe: () => undefined,
    unsubscribe: () => undefined,
    on: (event: string, handler: (event: unknown) => void) => {
      (handlers[event] ??= []).push(handler);
    },
    off: (event: string, handler: (event: unknown) => void) => {
      handlers[event] = (handlers[event] ?? []).filter((h) => h !== handler);
    },
  };
  return { handlers, context };
});

vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ws.context,
}));

const MODEL_ID = 'm_b5911e07';

function seedFreshPage() {
  useModelsStore.setState({
    models: [
      {
        id: MODEL_ID,
        name: 'LFM2.5-1.2B-Instruct',
        repo_id: 'LiquidAI/LFM2.5-1.2B-Instruct',
        status: ModelStatus.READY,
        quantization: QuantizationFormat.FP16,
        params_count: 1_200_000_000,
        created_at: '2026-09-12T00:00:00Z',
        updated_at: '2026-09-12T00:00:00Z',
      },
    ],
  } as never);
}

function serverSays(data: Record<string, unknown>) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ data }),
  }) as never;
}

function StoreBackedCard() {
  const model = useModelsStore((state) => state.models[0]);
  return (
    <ModelCard
      model={model}
      onClick={() => undefined}
      onExtract={() => undefined}
      onDelete={() => undefined}
      onCancel={() => undefined}
    />
  );
}

function sendSocketMessage(message: Record<string, unknown>) {
  act(() => {
    for (const handler of ws.handlers['extraction:progress'] ?? []) handler(message);
  });
}

const PERSISTED = {
  extraction_id: 'ext_m_b5911e07_20260913_101500',
  model_id: MODEL_ID,
  status: 'extracting',
  progress: 90,
  samples_processed: 20000,
  max_samples: 20000,
};

describe('ModelCard extraction phase after a reload', () => {
  beforeEach(() => {
    for (const key of Object.keys(ws.handlers)) delete ws.handlers[key];
    vi.spyOn(console, 'log').mockImplementation(() => undefined);
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    seedFreshPage();
  });

  afterEach(() => vi.restoreAllMocks());

  it('shows "Merging…" from the persisted phase with no WebSocket message', async () => {
    serverSays({ ...PERSISTED, phase: 'merging' });

    render(<StoreBackedCard />);

    await waitFor(() => expect(screen.getByText('Merging…')).toBeInTheDocument());
    expect(screen.queryByText('Extracting Activations')).not.toBeInTheDocument();
    expect(screen.getByText('Merging batch files (20000/20000 samples extracted)')).toBeInTheDocument();
  });

  it('shows "Computing statistics…" from the persisted phase with no WebSocket message', async () => {
    serverSays({ ...PERSISTED, phase: 'statistics' });

    render(<StoreBackedCard />);

    await waitFor(() => expect(screen.getByText('Computing statistics…')).toBeInTheDocument());
    expect(screen.queryByText('Extracting Activations')).not.toBeInTheDocument();
  });

  it('a WebSocket message still moves the card on from the persisted phase', async () => {
    serverSays({ ...PERSISTED, phase: 'merging' });
    render(<StoreBackedCard />);
    await waitFor(() => expect(screen.getByText('Merging…')).toBeInTheDocument());

    sendSocketMessage({
      model_id: MODEL_ID,
      extraction_id: PERSISTED.extraction_id,
      progress: 90,
      status: 'extracting',
      message: 'Computing statistics: layer 1 of 3 (0% of the statistics phase)',
      phase: 'statistics',
    });

    expect(screen.getByText('Computing statistics…')).toBeInTheDocument();
    expect(screen.queryByText('Merging…')).not.toBeInTheDocument();
    expect(useModelsStore.getState().models[0].extraction_phase).toBe('statistics');
  });

  it('a row with no recorded phase still reads as extracting', async () => {
    serverSays({ ...PERSISTED, samples_processed: 4336, phase: null });

    render(<StoreBackedCard />);

    await waitFor(() => expect(screen.getByText('Extracting Activations')).toBeInTheDocument());
  });
});

describe('extractionStageLabel', () => {
  it('uses the phase only while the status is extracting', () => {
    expect(extractionStageLabel('extracting', 'merging')).toBe('Merging…');
    expect(extractionStageLabel('extracting', 'statistics')).toBe('Computing statistics…');
    expect(extractionStageLabel('extracting', 'extracting')).toBe('Extracting Activations');
    expect(extractionStageLabel('extracting', null)).toBe('Extracting Activations');
    expect(extractionStageLabel('saving', 'statistics')).toBe('Saving Results');
    expect(extractionStageLabel('loading', 'merging')).toBe('Loading Model');
    expect(extractionStageLabel('complete', 'merging')).toBeNull();
  });
});
