/**
 * "Spot-checks" on a behaviour's detail (Feature 038 FR-15): the start form, the runs, and the report.
 * Without miLLM's recorded-session routes the form is replaced by the server's reason (FR-17).
 */
import { useCallback, useEffect, useState } from 'react';

import { spotChecksApi, type SpotCheck } from '../../api/spotChecks';
import { SpotCheckReport } from './SpotCheckReport';

interface Props {
  behaviourKey: string;
  prompts: { id: string; key: string; title: string; archived: boolean }[];
}

export function SpotCheckSection({ behaviourKey, prompts }: Props) {
  const [availability, setAvailability] = useState<{ available: boolean; reason: string | null } | null>(null);
  const [checks, setChecks] = useState<SpotCheck[]>([]);
  const [open, setOpen] = useState<SpotCheck | null>(null);
  const usable = prompts.filter((p) => !p.archived);
  const [promptId, setPromptId] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [model, setModel] = useState('');
  const [size, setSize] = useState('100');
  const [days, setDays] = useState('7');
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [a, l] = await Promise.all([spotChecksApi.availability(), spotChecksApi.list(behaviourKey)]);
      setAvailability(a);
      setChecks(l.spot_checks);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [behaviourKey]);

  useEffect(() => {
    void load();
  }, [load]);

  const running = checks.some((c) => c.status === 'pending' || c.status === 'running');
  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(timer);
  }, [running, load]);

  async function start(event: React.FormEvent) {
    event.preventDefault();
    try {
      const since = new Date(Date.now() - Number(days) * 86_400_000).toISOString();
      await spotChecksApi.start({ behaviour: behaviourKey, prompt_id: promptId || usable[0]?.id, endpoint, model, size: Number(size), since });
      setError(null);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const input = 'rounded-control border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-100';
  return (
    <section aria-label="Spot-checks" className="space-y-2">
      <h4 className="text-sm font-semibold text-slate-200">Spot-checks</h4>
      <p className="text-xs text-slate-400">
        A judge model re-reads a seeded sample of recorded miLLM conversation turns and is compared with this behaviour&apos;s probes.
      </p>
      {error ? <p role="alert" className="text-xs text-red-400">{error}</p> : null}
      {availability && !availability.available ? (
        <p className="text-xs text-amber-300" data-testid="spot-unavailable">{availability.reason}</p>
      ) : availability ? (
        <form onSubmit={start} className="flex flex-wrap gap-2" aria-label="Start a spot-check">
          <select aria-label="Judge prompt" className={input} value={promptId} onChange={(e) => setPromptId(e.target.value)}>
            {usable.map((p) => <option key={p.id} value={p.id}>{p.title} ({p.key})</option>)}
          </select>
          <input aria-label="Judge endpoint" className={input} placeholder="Judge address, e.g. http://…/v1" value={endpoint} onChange={(e) => setEndpoint(e.target.value)} />
          <input aria-label="Judge model" className={input} placeholder="Judge model" value={model} onChange={(e) => setModel(e.target.value)} />
          <input aria-label="Turns" className={input} value={size} onChange={(e) => setSize(e.target.value)} />
          <input aria-label="Days back" className={input} value={days} onChange={(e) => setDays(e.target.value)} />
          <button type="submit" disabled={running || usable.length === 0} className="rounded-control bg-emerald-700 px-2 py-1 text-xs text-white">Start spot-check</button>
          {/millm/i.test(endpoint) ? (
            <p className="w-full text-xs text-amber-300">The judge is miLLM: live chats wait behind each judge call while the check runs.</p>
          ) : null}
        </form>
      ) : null}
      <ul aria-label="Spot-check runs" className="space-y-1">
        {checks.map((c) => (
          <li key={c.id} className="text-xs text-slate-300">
            <button type="button" className="underline" onClick={() => setOpen(c)}>{c.id}</button>{' '}
            {c.status}{c.progress != null && c.status === 'running' ? ` ${(c.progress * 100).toFixed(0)}%` : ''}{c.partial ? ' (partial)' : ''} — {c.prompt_key}
          </li>
        ))}
      </ul>
      {open ? <SpotCheckReport check={checks.find((c) => c.id === open.id) ?? open} /> : null}
    </section>
  );
}
