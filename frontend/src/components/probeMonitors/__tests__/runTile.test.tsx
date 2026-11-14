/**
 * What a run tile has to tell an operator, and what it lets them do.
 *
 * ⚠ EVERY ONE OF THESE WAS MISSING, AND THE PANEL LOOKED FINE. A tile showed an id, a status and a
 * percentage: five runs against different models and corpora were visually identical apart from
 * twelve hex characters, a live run looked exactly like a finished one, and a failed run could
 * never be cleared — `DELETE /runs/{id}` existed and nothing in the UI called it. The gap is not
 * that the information was wrong; it is that the screen could not answer "which of these is the
 * LFM2 one", "is this still going" or "how long did it take".
 *
 * MUTATION CONTROLS (each verified to fail this file):
 *   R1  the model/dataset rows removed from the tile     → the identification tests
 *   R2  ids shown instead of names                       → the name-resolution test
 *   R3  the live spinner removed                         → the in-progress tests
 *   R4  the elapsed clock frozen at mount                → the ticking test
 *   R5  delete offered without a confirmation            → the confirm test
 *   R6  delete offered on a RUNNING run                  → the live-delete test
 *   R7  `runDuration` measures a live run as 0            → the live-duration test
 */
import { describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { RunTile, formatDuration, runDuration } from '../RunTile';

function run(overrides: Record<string, unknown> = {}) {
  return {
    id: 'pmr_abc123',
    model_id: 'm_40e78d80',
    train_dataset_id: 'pmd_train',
    eval_dataset_ids: ['pmd_a', 'pmd_b'],
    calibration_dataset_id: null,
    config: {},
    stage: 'rung',
    status: 'completed',
    progress: 100,
    error_message: null,
    celery_task_id: null,
    gpu_request: 'auto',
    gpu_uuid: null,
    artifact_dir: null,
    environment: {},
    layer_selection: null,
    created_at: '2026-09-26T09:00:00.000Z',
    updated_at: '2026-09-26T10:24:00.000Z',
    completed_at: '2026-09-26T10:24:00.000Z',
    ...overrides,
  } as never;
}

const NAMES = {
  models: { m_40e78d80: 'Llama-3.1-8B-Instruct' },
  datasets: { pmd_train: 'training', pmd_a: 'mt_balanced', pmd_b: 'toolace_balanced' },
};

function mount(overrides = {}, handlers = {}) {
  const onCancel = vi.fn();
  const onDelete = vi.fn();
  render(
    <ul>
      <RunTile
        run={run(overrides)}
        modelNames={NAMES.models}
        datasetNames={NAMES.datasets}
        onCancel={onCancel}
        onDelete={onDelete}
        {...handlers}
      />
    </ul>
  );
  return { onCancel, onDelete };
}

describe('the tile says what the run ran on', () => {
  it('names the model, not its id', () => {
    mount();
    const details = screen.getByTestId('run-details');
    expect(details).toHaveTextContent('Llama-3.1-8B-Instruct');
    expect(details).not.toHaveTextContent('m_40e78d80');
  });

  it('names the training view', () => {
    mount();
    expect(screen.getByTestId('run-details')).toHaveTextContent('training');
  });

  it('names EVERY evaluation view', () => {
    mount();
    const details = screen.getByTestId('run-details');
    expect(details).toHaveTextContent('mt_balanced');
    expect(details).toHaveTextContent('toolace_balanced');
  });

  it('falls back to the id when a name is unknown', () => {
    /** Better a raw id than a blank: the id is still resolvable by hand. */
    render(
      <ul>
        <RunTile run={run({ model_id: 'm_unknown' })} onCancel={vi.fn()} onDelete={vi.fn()} />
      </ul>
    );
    expect(screen.getByTestId('run-details')).toHaveTextContent('m_unknown');
  });

  it('shows an em dash for an absent evaluation list rather than nothing', () => {
    mount({ eval_dataset_ids: [] });
    expect(screen.getByTestId('run-details')).toHaveTextContent('—');
  });
});

describe('the tile says when, and for how long', () => {
  it('shows the start and finish stamps', () => {
    mount();
    const details = screen.getByTestId('run-details');
    expect(details).toHaveTextContent('Started');
    expect(details).toHaveTextContent('Finished');
  });

  it('shows the total run time', () => {
    mount();
    // 09:00:00 -> 10:24:00 is 1h 24m 0s
    expect(screen.getByTestId('run-duration')).toHaveTextContent('1h 24m 0s');
  });

  it.each([
    [0, '0s'],
    [45_000, '45s'],
    [90_000, '1m 30s'],
    [3_600_000, '1h 0m 0s'],
    [5_031_000, '1h 23m 51s'],
  ])('formats %ims as %s', (ms, expected) => {
    expect(formatDuration(ms)).toBe(expected);
  });

  it('refuses to invent a duration it cannot compute', () => {
    expect(formatDuration(null)).toBeNull();
    expect(formatDuration(-1)).toBeNull();
    expect(runDuration(run({ created_at: null }) as never)).toBeNull();
    expect(runDuration(run({ created_at: 'not a date' }) as never)).toBeNull();
  });

  it('measures an UNFINISHED run against now, not as zero', () => {
    /** "started 40 minutes ago" is the thing an operator wants; a blank reads as no information
        when the information is there. */
    const started = Date.parse('2026-09-26T09:00:00.000Z');
    const elapsed = runDuration(
      run({ status: 'running', completed_at: null }) as never,
      started + 600_000
    );
    expect(elapsed).toBe(600_000);
    expect(formatDuration(elapsed)).toBe('10m 0s');
  });

  /*
   * ⚠ FOUND BY LOOKING AT THE LIVE PANEL, NOT BY READING THE CODE.
   *
   * `pmr_5d81ad3b81f2` failed within minutes and the tile displayed "Took 6h 36m 19s" beside
   * "Finished —", growing on every refresh. Only the SUCCESS path stamped `completed_at`, so a
   * failed or cancelled run had none, and `completed_at ?? now` is right for a live run and a
   * fabrication for a terminal one.
   *
   * The backend now stamps every terminal path, but the rows already in the database never will be,
   * so this function has to hold the line by itself. "—" is the honest answer; an invented number
   * is not, and a reader has no way to tell which they are looking at.
   */
  describe('a run that has ENDED without recording when', () => {
    it.each(['failed', 'cancelled', 'completed'])(
      'reports nothing rather than counting up to now (%s)',
      (status) => {
        const started = Date.parse('2026-09-26T09:00:00.000Z');
        expect(
          runDuration(
            run({ status, created_at: '2026-09-26T09:00:00.000Z', completed_at: null }) as never,
            started + 6 * 3_600_000
          )
        ).toBeNull();
      }
    );

    it('the tile shows an em dash for it, not a number', () => {
      mount({
        status: 'failed',
        created_at: '2026-09-26T09:00:00.000Z',
        completed_at: null,
      });
      expect(screen.getByTestId('run-duration')).toHaveTextContent('—');
    });

    it('but a LIVE run with no end is still measured against now', () => {
      /** The distinction this rests on: unfinished and ended-without-a-stamp are different facts. */
      const started = Date.parse('2026-09-26T09:00:00.000Z');
      expect(
        runDuration(
          run({ status: 'running', created_at: '2026-09-26T09:00:00.000Z', completed_at: null }) as never,
          started + 600_000
        )
      ).toBe(600_000);
    });
  });
});

describe('a run in progress is visibly in progress', () => {
  it.each(['pending', 'running', 'cancelling'])('%s shows a spinner', (status) => {
    mount({ status, completed_at: null });
    expect(screen.getByTestId('run-live-spinner')).toBeInTheDocument();
  });

  it('a finished run shows none', () => {
    mount({ status: 'completed' });
    expect(screen.queryByTestId('run-live-spinner')).toBeNull();
  });

  it('a failed run shows none', () => {
    mount({ status: 'failed', error_message: 'boom' });
    expect(screen.queryByTestId('run-live-spinner')).toBeNull();
    expect(screen.getByTestId('run-error')).toHaveTextContent('boom');
  });

  it('marks the tile as live for styling and for tests', () => {
    mount({ status: 'running', completed_at: null });
    expect(screen.getByTestId('run-row')).toHaveAttribute('data-live', 'true');
  });

  it('keeps the stage beside the percentage', () => {
    /** 40% means nothing without "layer_selection" beside it. */
    mount({ status: 'running', stage: 'pooled_capture', progress: 34.9, completed_at: null });
    expect(screen.getByTestId('run-status')).toHaveTextContent('pooled_capture');
    expect(screen.getByTestId('run-status')).toHaveTextContent('35%');
  });

  it('reports elapsed time in the status line while live', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-26T09:10:00.000Z'));
    try {
      mount({ status: 'running', completed_at: null, progress: 30 });
      expect(screen.getByTestId('run-status')).toHaveTextContent('running 10m 0s');
    } finally {
      vi.useRealTimers();
    }
  });

  it('the elapsed clock TICKS', async () => {
    /** The control for the one above: a duration computed once at mount would freeze, and a
        frozen clock on a live run is exactly the "is it stuck?" question this is here to
        answer. */
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-26T09:10:00.000Z'));
    try {
      mount({ status: 'running', completed_at: null });
      expect(screen.getByTestId('run-status')).toHaveTextContent('10m 0s');
      // Wrapped: the interval's setState is a React update, and an unwrapped one
      // warns. A warning in the suite is noise that hides a real one.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(5000);
      });
      expect(screen.getByTestId('run-status')).toHaveTextContent('10m 5s');
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('a run can be deleted, carefully', () => {
  it('offers delete on a finished run', () => {
    mount();
    expect(screen.getByTestId('run-delete')).toBeInTheDocument();
  });

  it('does NOT delete on the first click', async () => {
    const { onDelete } = mount();
    await userEvent.click(screen.getByTestId('run-delete'));
    expect(onDelete).not.toHaveBeenCalled();
    expect(screen.getByTestId('run-delete-confirm')).toBeInTheDocument();
  });

  it('says what will be lost', async () => {
    mount();
    await userEvent.click(screen.getByTestId('run-delete'));
    const tile = screen.getByTestId('run-row');
    expect(tile).toHaveTextContent(/artifact directory/i);
    expect(tile).toHaveTextContent(/gigabytes/i);
    expect(tile).toHaveTextContent(/cannot be undone/i);
  });

  it('deletes on confirmation', async () => {
    const { onDelete } = mount();
    await userEvent.click(screen.getByTestId('run-delete'));
    await userEvent.click(screen.getByTestId('run-delete-confirm'));
    expect(onDelete).toHaveBeenCalledWith('pmr_abc123');
  });

  it('can be backed out of', async () => {
    const { onDelete } = mount();
    await userEvent.click(screen.getByTestId('run-delete'));
    await userEvent.click(screen.getByTestId('run-delete-cancel'));
    expect(onDelete).not.toHaveBeenCalled();
    expect(screen.getByTestId('run-delete')).toBeInTheDocument();
  });

  it.each(['pending', 'running', 'cancelling'])(
    'does not offer delete on a %s run',
    (status) => {
      /** Deleting a live run removes the row its worker reads, and the cancel scope treats a
          missing row as a stop — so it WORKS, and that is the problem: it would hide a
          cancellation inside a delete. "Stop" is the honest button for that intent. */
      mount({ status, completed_at: null });
      expect(screen.queryByTestId('run-delete')).toBeNull();
      expect(screen.getByTestId('run-stop')).toBeInTheDocument();
    }
  );

  it('offers stop only while live', () => {
    mount({ status: 'completed' });
    expect(screen.queryByTestId('run-stop')).toBeNull();
  });

  it('stops on click', async () => {
    const { onCancel } = mount({ status: 'running', completed_at: null });
    await userEvent.click(screen.getByTestId('run-stop'));
    expect(onCancel).toHaveBeenCalledWith('pmr_abc123');
  });
});


/*
 * The tile's SHAPE, asked for directly: "less vertical in height … a lot of real estate
 * horizontally … a horizontal progress bar while a run is in progress would be consistent with
 * other running job tiles."
 *
 * Both halves are pinned because both are easy to lose in a later refactor and neither shows up as
 * a type error. The grid can silently revert to a stacked list, and the bar can silently stop
 * rendering for a live run.
 */
describe('the tile uses its width, and shows progress as a bar', () => {
  it('lays the facts out across columns rather than down a list', () => {
    mount({ status: 'completed' });
    const details = screen.getByTestId('run-details');
    // The defect this replaces was a two-column `auto 1fr` list — six facts, six stacked rows.
    expect(details.className).not.toMatch(/grid-cols-\[auto_1fr\]/);
    expect(details.className).toMatch(/grid-cols-2/);
    // Three at `sm` and above, NOT four: the facts share their row with the layer sweep from `lg`
    // up, which leaves them about 55% of the tile, and a fourth column wraps a timestamp.
    expect(details.className).toMatch(/sm:grid-cols-3/);
  });

  it('puts the layer sweep BESIDE the facts on a wide screen, not under them', () => {
    /*
     * The other half of "use the width". The sweep table was stacked under the facts, so a tile
     * spent ~70px on facts and ~110px on a narrow table while half its width sat empty. Measured on
     * the live panel: 284px before any of this, 240px after the facts went across, and the sweep is
     * the remaining block.
     */
    // `mount`'s second argument is spread as props; React takes `children` as one.
    mount(
      { status: 'completed' },
      { children: <div data-testid="sweep">the layer sweep</div> }
    );
    const aside = screen.getByTestId('run-aside');
    expect(aside).toContainElement(screen.getByTestId('sweep'));
    const row = aside.parentElement as HTMLElement;
    expect(row.className).toMatch(/lg:flex-row/);
    expect(row.className).toMatch(/flex-col/); // and stacks below `lg`
    expect(row).toContainElement(screen.getByTestId('run-details'));
  });

  it('renders no aside when there is nothing to put beside the facts', () => {
    mount({ status: 'completed' });
    expect(screen.queryByTestId('run-aside')).toBeNull();
  });

  it('still names every fact it named before — the layout changed, not the content', () => {
    mount({ status: 'completed', completed_at: '2026-09-26T10:30:00Z' });
    const details = screen.getByTestId('run-details');
    for (const label of ['Model', 'Train', 'Evaluate', 'Started', 'Finished', 'Took']) {
      expect(details).toHaveTextContent(label);
    }
  });

  it('says Elapsed rather than Finished while the run is live', () => {
    mount({ status: 'running', completed_at: null });
    const details = screen.getByTestId('run-details');
    expect(details).toHaveTextContent('Elapsed');
    expect(details).not.toHaveTextContent('Finished');
  });

  describe('the progress bar', () => {
    it('renders for a live run with a reported percentage', () => {
      mount({ status: 'running', progress: 42, completed_at: null });
      const bar = screen.getByTestId('run-progress');
      expect(bar).toBeInTheDocument();
      // The fill is the shared `common/ProgressBar`, so its width carries the number.
      const fill = bar.querySelector('[style*="width"]') as HTMLElement | null;
      expect(fill).not.toBeNull();
      expect(fill!.style.width).toBe('42%');
    });

    it('is absent once the run has finished', () => {
      mount({ status: 'completed', progress: 100 });
      expect(screen.queryByTestId('run-progress')).toBeNull();
    });

    it('is ABSENT for a live run that has reported no percentage yet', () => {
      /*
       * A bar needs a number. Drawing an empty one would read as "nothing has happened" when the
       * truth is "nothing has been said yet" — the spinner and the stage carry it until then.
       */
      mount({ status: 'running', progress: null, completed_at: null });
      expect(screen.queryByTestId('run-progress')).toBeNull();
      expect(screen.getByTestId('run-live-spinner')).toBeInTheDocument();
    });

    it('does not overflow the tile on a percentage outside 0..100', () => {
      /*
       * ⚠ THE CLAMP IS NOT IN THIS COMPONENT, and a mutation control is what established that.
       * Deleting RunTile's own `Math.max(0, Math.min(100, …))` changed nothing: `common/ProgressBar`
       * clamps, so the duplicate was dead code. This test stays because the BEHAVIOUR is what
       * matters to the tile; the rule lives with the component that draws the bar.
       */
      mount({ status: 'running', progress: 137, completed_at: null });
      const fill = screen
        .getByTestId('run-progress')
        .querySelector('[style*="width"]') as HTMLElement;
      expect(fill.style.width).toBe('100%');
    });

    it('does not repeat the percentage the status line already carries', () => {
      /*
       * Two copies of one number is noise, and they can disagree mid-render.
       *
       * ⚠ THIS ASSERTION USED TO BE `not.toHaveTextContent('42%')` AND A MUTATION WALKED THROUGH IT:
       * `ProgressBar`'s own label renders `toFixed(1)` — "42.0%" — which does not contain the
       * substring "42%". Turning `showPercentage` back on left the suite green. The assertion is now
       * on the absence of a percent SIGN, which no formatting can slip past.
       */
      mount({ status: 'running', progress: 42, completed_at: null });
      expect(screen.getByTestId('run-status')).toHaveTextContent('42%');
      expect(screen.getByTestId('run-progress').textContent ?? '').not.toContain('%');
    });
  });
});
