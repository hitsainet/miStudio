/**
 * The training-metrics client and the types the training panel reads off the wire
 * (review R2-D, R2D-9; fixed in round 3, R3-C).
 *
 * `api/trainings.ts` declared its own `TrainingMetric` and it drifted from the
 * backend's `TrainingMetricResponse`: `id` was typed a string (the backend sends an
 * int), and `fvu`, `l0_sparsity`, `dead_neurons` and `learning_rate` were typed never
 * null (each is Optional). `TrainingEvaluation` lacked `progress`, `hook_point` and
 * `gpu_request`, which the backend writes, so no reader could show the running stage.
 *
 * The runtime test pins the request and that rows arrive unchanged. The type
 * assertions are checked by `npm run type-check:test`, whose error count
 * `src/test/typeCheckRatchet.test.ts` holds at or below its baseline in every vitest
 * run, so a type that drifts back turns the suite red.
 *
 * MUTATION CONTROLS (results in the R3-C record):
 *   T1  `TrainingMetric.id` typed `string` again             -> the ratchet goes red
 *   T2  `progress` removed from `TrainingEvaluation`         -> the ratchet goes red
 *   T3  the endpoint loses `aggregate_only`                  -> red here
 */

import { describe, it, expect, expectTypeOf, vi, beforeEach } from 'vitest';
import type { TrainingEvaluation, TrainingEvaluationProgress } from '../types/training';

vi.mock('./client', async () => {
  const actual = await vi.importActual<typeof import('./client')>('./client');
  return { ...actual, fetchAPI: vi.fn() };
});

import { fetchAPI } from './client';
import { fetchTrainingMetrics, type TrainingMetric } from './trainings';

describe('fetchTrainingMetrics', () => {
  beforeEach(() => vi.mocked(fetchAPI).mockReset());

  it('asks once for the aggregate window and returns the rows as the backend sent them', async () => {
    // Shaped like TrainingMetricResponse: an int id, and nulls where a value is absent.
    const rows = [
      {
        id: 4812,
        training_id: 'train_24e5e7d3',
        step: 3800,
        timestamp: '2026-09-14T16:04:51Z',
        layer_idx: null,
        hook_type: null,
        loss: 0.0013,
        loss_reconstructed: null,
        loss_zero: null,
        l0_sparsity: null,
        l1_sparsity: null,
        dead_neurons: 411,
        fvu: 0.783,
        fvu_centred: null,
        learning_rate: 7e-5,
        grad_norm: null,
        gpu_memory_used_mb: null,
        samples_per_second: null,
      },
    ];
    vi.mocked(fetchAPI).mockResolvedValue({ data: rows });

    const result = await fetchTrainingMetrics('train_24e5e7d3', { limit: 20, aggregate_only: true });

    expect(fetchAPI).toHaveBeenCalledTimes(1);
    expect(vi.mocked(fetchAPI).mock.calls[0]).toStrictEqual([
      '/trainings/train_24e5e7d3/metrics?limit=20&aggregate_only=true',
    ]);
    expect(result).toStrictEqual(rows);
    expect(typeof result[0].id).toBe('number');
  });
});

describe('wire types', () => {
  it('types a metric row as the backend sends it', () => {
    expectTypeOf<TrainingMetric['id']>().toEqualTypeOf<number>();
    expectTypeOf<TrainingMetric['fvu']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['fvu_centred']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['l0_sparsity']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['dead_neurons']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['learning_rate']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['layer_idx']>().toEqualTypeOf<number | null | undefined>();
    expectTypeOf<TrainingMetric['hook_type']>().toEqualTypeOf<string | null | undefined>();
  });

  it('types the evaluation fields the backend writes', () => {
    expectTypeOf<TrainingEvaluation['progress']>().toEqualTypeOf<TrainingEvaluationProgress | null | undefined>();
    expectTypeOf<TrainingEvaluationProgress['batches_done']>().toEqualTypeOf<number | undefined>();
    expectTypeOf<TrainingEvaluation['hook_point']>().toEqualTypeOf<string | undefined>();
    expectTypeOf<TrainingEvaluation['gpu_request']>().toEqualTypeOf<string | null | undefined>();

    // What POST /trainings/{id}/evaluate writes, then a running heartbeat.
    const running: TrainingEvaluation = {
      version: 1,
      status: 'running',
      trigger: 'rerun',
      task_id: 'b7c2',
      gpu_request: 'auto',
      hook_point: 'resid_post',
      placement: 'GPU 1 (NVIDIA GeForce RTX 3090, 20,112 of 24,576 MB free)',
      progress: { stage: 'cross_entropy', batches_done: 12, batches: 32 },
    };
    expect(running.progress?.stage).toBe('cross_entropy');
  });
});
