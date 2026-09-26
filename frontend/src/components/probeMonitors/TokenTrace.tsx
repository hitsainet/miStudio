/**
 * Per-token scores for one scored input (032 FR-12).
 *
 * ⚠ UNSCORED TOKENS ARE VISIBLY DIFFERENT FROM LOW-SCORING ONES. A token outside the
 * probe's scope has NO score — it is not a zero — so it renders without a heat tint and
 * is marked in the legend. Shading it as "cold" would tell a reader the probe looked at
 * the user's turn and found nothing, when it never looked at all.
 */
import type { ProbeScoreResult } from '../../types/probeMonitor';

interface TokenTraceProps {
  result: ProbeScoreResult;
}

export function TokenTrace({ result }: TokenTraceProps) {
  const scored = result.token_scores;
  const max = scored.length ? Math.max(...scored.map(Math.abs)) : 1;
  let scoredIndex = 0;

  return (
    <div data-testid="token-trace">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs text-slate-400">
        <span>
          aggregate{' '}
          <span className="tabular-nums text-slate-100">{result.aggregate.toFixed(4)}</span>
        </span>
        {result.threshold === null ? (
          // NOT "does not fire". No threshold was placed, so the probe has said nothing.
          <span className="text-amber-300" data-testid="no-threshold">
            no threshold placed — this probe ranks but does not decide
          </span>
        ) : (
          <span>
            threshold <span className="tabular-nums">{result.threshold.toFixed(4)}</span> ·{' '}
            <span className={result.fires ? 'text-rose-300' : 'text-emerald-300'}>
              {result.fires ? 'fires' : 'below threshold'}
            </span>
          </span>
        )}
        {!result.role_mask_reliable ? (
          <span className="text-amber-300" data-testid="mask-unreliable">
            role mask unreliable for this input — every token was scored
          </span>
        ) : null}
        {result.truncated ? (
          <span className="text-amber-300" data-testid="truncated">
            truncated from the front to fit the window
          </span>
        ) : null}
      </div>
      <p className="font-mono text-sm leading-7">
        {result.tokens.map((token, index) => {
          if (!token.scored) {
            return (
              <span
                key={index}
                className="rounded px-0.5 text-slate-500 underline decoration-dotted"
                title="outside this probe's scope — not scored"
                data-testid="token-unscored"
              >
                {token.token}
              </span>
            );
          }
          const value = scored[scoredIndex] ?? 0;
          scoredIndex += 1;
          const intensity = max > 0 ? Math.min(Math.abs(value) / max, 1) : 0;
          const positive = value >= 0;
          return (
            <span
              key={index}
              className="rounded px-0.5 text-slate-100"
              style={{
                backgroundColor: positive
                  ? `rgba(244, 63, 94, ${(intensity * 0.65).toFixed(3)})`
                  : `rgba(16, 185, 129, ${(intensity * 0.65).toFixed(3)})`,
              }}
              title={`${token.token}: ${value.toFixed(4)}`}
              data-testid="token-scored"
            >
              {token.token}
            </span>
          );
        })}
      </p>
      <p className="mt-2 text-[11px] text-slate-500">
        Red is toward the concept, green away from it; intensity is relative to this
        input's largest magnitude. Dotted-underlined tokens were not scored.
      </p>
    </div>
  );
}
