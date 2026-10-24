/**
 * How far a live probe job has got, in words a person reads at a glance.
 *
 * ⚠ A JUDGE RUN SAT AT 0% FOR ITS WHOLE FIRST SET (operator, 2026-10-21: "The probe run does not
 * show progress"). It wrote progress once per set, so a two-set run read 0% for twenty minutes
 * while it answered about two calls a second. The server now counts every row and says what it is
 * counting (`progress_detail`); this renders that reading — rows of rows, the set, the speed and
 * the time left — and a training run's stage, units and the item inside the stage.
 *
 * Every figure is the server's. Speed and time left show only when the server has measured them;
 * until then the line says it is still measuring, never a guessed number.
 */
import { ProgressBar } from '../common/ProgressBar';
import type {
  JudgeRunProgressDetail,
  ProbeJudgeRun,
  ProbeRun,
  ProbeRunProgressDetail,
} from '../../types/probeMonitor';

const count = (value: number) => Math.round(value).toLocaleString('en-US');

/** A number of units, without a trailing ".0" — calibration counts fractional layer forwards. */
const units = (value: number) =>
  Number.isInteger(value) ? count(value) : value.toLocaleString('en-US', { maximumFractionDigits: 1 });

/** "2.1 rows/s", or a per-minute figure when a unit takes longer than a second or so. */
export function formatRate(rate: number | null | undefined, unit: string): string | null {
  if (rate == null || !Number.isFinite(rate) || rate <= 0) return null;
  if (rate < 0.5) {
    const perMinute = rate * 60;
    return `${perMinute.toFixed(perMinute < 10 ? 1 : 0)} ${unit}/min`;
  }
  return `${rate.toFixed(rate < 10 ? 1 : 0)} ${unit}/s`;
}

/** "about 9 min left" — plain words, rounded the way a person would say it. */
export function formatTimeLeft(seconds: number | null | undefined): string | null {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return null;
  if (seconds < 60) return 'under a minute left';
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `about ${minutes} min left`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `about ${hours} h ${rest} min left` : `about ${hours} h left`;
}

export function describeJudgeProgress(detail: JudgeRunProgressDetail): string {
  const parts: string[] = [];
  if (detail.set_name) parts.push(detail.set_name);
  parts.push(`${count(detail.rows_done)} of ${count(detail.rows_total)}`);
  const rate = formatRate(detail.rate_per_second, 'rows');
  const left = formatTimeLeft(detail.eta_seconds);
  if (rate) parts.push(rate);
  if (left && detail.rows_done < detail.rows_total) parts.push(left);
  if (!rate && detail.rows_done < detail.rows_total) parts.push('measuring speed');
  return parts.join(' · ');
}

const STAGE_WORDS: Record<string, string> = {
  rendering: 'Preparing the rows',
  pooled_capture: 'Reading the model at every layer',
  layer_selection: 'Choosing the layer',
  token_capture: 'Capturing token activations',
  training: 'Training probes',
  calibrating: 'Setting thresholds',
  evaluating: 'Evaluating on the test sets',
  rung: 'Recording the evidence level',
};

export function describeRunProgress(detail: ProbeRunProgressDetail): string {
  const stage = STAGE_WORDS[detail.stage] ?? detail.stage.replace(/_/g, ' ');
  const parts = [`${stage} (step ${detail.stage_index + 1} of ${detail.stage_count})`];
  if (detail.done != null && detail.total != null && detail.unit) {
    parts.push(`${units(detail.done)} of ${units(detail.total)} ${detail.unit}`);
  }
  if (detail.current) {
    const c = detail.current;
    parts.push(`${c.name}: ${units(c.done)} of ${units(c.total)} ${c.unit}`);
  }
  const rate = detail.unit ? formatRate(detail.rate_per_second, detail.unit) : null;
  if (rate) parts.push(rate);
  const left = formatTimeLeft(detail.eta_seconds);
  if (left && detail.done != null) parts.push(`${left} in this step`);
  return parts.join(' · ');
}

/** A live judge run: the bar, then "US test · 812 of 1,370 · 2.1 rows/s · about 9 min left". */
export function JudgeRunLiveProgress({ run }: { run: ProbeJudgeRun }) {
  if (run.status !== 'running' && run.status !== 'pending') return null;
  const detail = run.progress_detail;
  if (!detail || detail.kind !== 'judge') {
    return (
      <p className="mt-1 text-xs text-slate-400" data-testid="judge-live-progress">
        {run.status === 'pending' ? 'Waiting to start' : 'Reading the sets'}
      </p>
    );
  }
  return (
    <div className="mt-2" data-testid="judge-live-progress">
      <ProgressBar progress={run.progress ?? 0} showPercentage={false} />
      <p className="mt-1 text-xs text-slate-300" data-testid="judge-live-line">
        {describeJudgeProgress(detail)}
      </p>
      <p className="text-[11px] text-slate-500">
        {detail.set_index != null && detail.set_rows_total != null
          ? `Set ${detail.set_index + 1} of ${detail.set_count}: ${count(detail.set_rows_done ?? 0)} of ${count(detail.set_rows_total)} rows`
          : `${detail.set_count} ${detail.set_count === 1 ? 'set' : 'sets'} planned`}
        {detail.parse_failures > 0
          ? ` · ${count(detail.parse_failures)} ${detail.parse_failures === 1 ? 'reply' : 'replies'} unreadable so far`
          : ''}
      </p>
    </div>
  );
}

/** A live training run's stage reading, under its bar. */
export function RunLiveProgress({ run }: { run: ProbeRun }) {
  const detail = run.progress_detail;
  if (!detail || detail.kind !== 'training') return null;
  return (
    <p className="mt-1 text-xs text-slate-300" data-testid="run-live-line">
      {describeRunProgress(detail)}
    </p>
  );
}
