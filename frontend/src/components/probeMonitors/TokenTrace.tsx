/**
 * Per-token scores for one scored input (032 FR-12).
 *
 * ⚠ UNSCORED TOKENS ARE VISIBLY DIFFERENT FROM LOW-SCORING ONES. A token outside the
 * probe's scope has NO score — it is not a zero — so it renders without a heat tint and
 * is marked in the legend. Shading it as "cold" would tell a reader the probe looked at
 * the user's turn and found nothing, when it never looked at all.
 */
import type { ProbeScoreResult } from '../../types/probeMonitor';

/*
 * ⚠ ALPHA ALONE IS NOT A LEGIBLE ENCODING, AND THIS FILE LEARNED IT TWICE. A continuous
 * alpha ramp over forty tokens of flowing prose collapses into one wash: almost every
 * token lands in the bottom third of the range, and the eye cannot order them. The
 * operator reported exactly that, twice, and the second report is what sent me to look
 * at `features/TokenHighlight.tsx` — the component in this app that has always worked.
 *
 * It does NOT rely on alpha. It moves three channels together: background alpha, TEXT
 * COLOUR above intensity 0.6, and A BORDER above 0.7. Strength is encoded redundantly,
 * so a reader discriminates on whichever channel their eye catches first.
 *
 * This takes that approach and makes it explicit as four BANDS. Banding is the point:
 * continuous shading answers "is this one slightly warmer than that one", which nobody
 * can read, where a band answers "is this one strong", which anybody can. Ordering within
 * a band is lost on purpose — the exact score is one hover away, and the legend says the
 * scale is per input, so fine gradations were never comparable anyway.
 *
 * The inset box-shadow draws the rule rather than `border-bottom`, because a real border
 * changes the box height and makes the line rhythm stagger in a wrapped paragraph.
 */
const VIOLET = '167, 139, 250';   // toward the concept
const EMERALD = '110, 231, 183';  // away from it

const BAND_NAME = ['faint', 'weak', 'moderate', 'strong'] as const;
//  A floor at band 0: a token the probe SCORED at ~0 must not render as one it never
//  looked at. Those are different facts and the trace's whole job is keeping them apart.
const BAND_ALPHA = [0.13, 0.3, 0.52, 0.75];
const BAND_TEXT = [
  'text-slate-300',
  'text-slate-100',
  'text-white',
  'text-white font-semibold',
];
const BAND_RULE = [0, 1, 2, 2];
const BAND_RULE_ALPHA = [0, 0.55, 0.85, 1];

/**
 * The scale the bands are cut against.
 *
 * ⚠ NOT THE MAXIMUM, AND THE MAXIMUM IS WHAT MADE THIS TRACE UNREADABLE TWICE. Per-token
 * scores are logit-scale and heavy-tailed: on a real 90-token input measured through the
 * live API, max|score| was 33.24 while the median was 4.23. Dividing by the max put 86 of
 * those 90 tokens in the bottom two bands — one outlier flattened the entire passage, and
 * no amount of restyling fixes that, because the numbers reaching the styles were already
 * crushed. Switching to the 90th percentile moved the same input to 14 / 29 / 25 / 22.
 *
 * The 10-token floor is there because a percentile over four values is not a percentile;
 * below it the maximum is the honest scale.
 *
 * This is a WITHIN-INPUT ranking and the legend says so. Absolute meaning lives in the
 * aggregate against the threshold, never in how dark a token is.
 */
function heatScale(scores: number[]): number {
  const magnitudes = scores.map(Math.abs);
  if (magnitudes.length === 0) return 1;
  if (magnitudes.length < 10) return Math.max(...magnitudes);
  const sorted = [...magnitudes].sort((a, b) => a - b);
  const position = (sorted.length - 1) * 0.9;
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  const p90 = sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  // An all-zero input has no scale; 1 keeps every intensity at 0 rather than dividing by 0.
  return p90 > 0 ? p90 : Math.max(...magnitudes) || 1;
}

function bandOf(intensity: number): number {
  if (intensity >= 0.7) return 3;
  if (intensity >= 0.4) return 2;
  if (intensity >= 0.15) return 1;
  return 0;
}

interface TokenTraceProps {
  result: ProbeScoreResult;
  /** What the probe detects, for the legend. Optional: an older caller still renders. */
  concept?: string;
  /** The pooling rule that turned these per-token scores into the aggregate. */
  rule?: string;
}

export function TokenTrace({ result, concept, rule }: TokenTraceProps) {
  const scored = result.token_scores;
  const max = heatScale(scored);
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
            threshold <span className="tabular-nums">{result.threshold.toFixed(4)}</span>
            {/* ⚠ WHICH BAR, AND WHY THAT ONE. A probe with a length table is judged against the
                band its scored-token count falls in, which can sit well above or below the
                headline threshold — on one probe 12.81 for 0–203 tokens and 26.60 for 204–339
                against a global 17.98. A bare number left a reader comparing against the wrong
                bar, and the live monitor uses the band. */}
            {result.threshold_band ? (
              <span className="text-slate-400" data-testid="score-threshold-band">
                {' '}
                ({result.threshold_band.min_tokens}–
                {result.threshold_band.max_tokens ?? '∞'} tokens
                {result.threshold_band.threshold_source === 'global'
                  ? ', too few negatives to cut — inherits the global bar'
                  : ''}
                {result.global_threshold != null &&
                result.threshold_band.threshold_source !== 'global' &&
                result.global_threshold !== result.threshold
                  ? `; global ${result.global_threshold.toFixed(4)}`
                  : ''}
                )
              </span>
            ) : null}{' '}
            ·{' '}
            <span className={result.fires ? 'text-violet-300' : 'text-emerald-300'}>
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
      {/*
        ⚠ THE DISPLAY TEXT, NOT THE VOCABULARY STRING. `convert_ids_to_tokens` returns how the
        vocabulary SPELLS a token, so on byte-level BPE this read `<|im_start|>ĠuserĊRetĠire` —
        the encoding of the input rather than the input. The backend now decodes with the
        tokenizer that produced it (`Ġ`→space, `Ċ`→newline) and sends `text` beside `token`;
        `?? token.token` keeps a trace from an older build readable.

        Whitespace is rendered inside the highlight, because the space before a word is part of
        the token that was scored — colouring the word and not its space would misattribute the
        heat by one character.
      */}
      {/*
        ⚠ `break-words` IS ON THE CONTAINER BECAUSE THE TOKENS ARE SPANS. An inline element
        boundary is not a soft-wrap opportunity, so a pasted run with no spaces
        ("eyJhbGciOi…", a minified line, a degenerate generation) tokenizes into dozens of
        adjacent spans that the line breaker sees as ONE unbreakable word — and
        `whitespace-pre-wrap` only wraps where an opportunity already exists. `overflow-wrap`
        is an inherited property, so setting it here reaches every token span; putting it on
        the spans instead would be the same declaration written N times.

        NOT `break-all`: that breaks ordinary prose mid-word, and prose is the normal input.
      */}
      <p className="whitespace-pre-wrap break-words font-mono text-sm leading-7">
        {result.tokens.map((token, index) => {
          const shown = token.text ?? token.token;
          if (token.special) {
            /*
             * Scaffolding keeps its literal spelling and is visibly not content. It is NOT hidden:
             * a probe that fires on `<|im_start|>` rather than on the text is a finding, and
             * filtering these would delete the evidence for it. It is still scored or not on its
             * own merits below.
             */
            const value = token.scored ? scored[scoredIndex] : null;
            if (token.scored) scoredIndex += 1;
            return (
              <span
                key={index}
                className="mx-0.5 rounded border border-slate-600 bg-slate-800 px-1 text-[11px] text-slate-400"
                title={
                  value === null
                    ? 'chat-template scaffolding — not scored'
                    : `chat-template scaffolding, scored ${value.toFixed(4)}`
                }
                data-testid="token-special"
              >
                {shown}
              </span>
            );
          }
          if (!token.scored) {
            return (
              <span
                key={index}
                className="rounded px-0.5 text-slate-500 underline decoration-dotted"
                title="outside this probe's scope — not scored"
                data-testid="token-unscored"
              >
                {shown}
              </span>
            );
          }
          const value = scored[scoredIndex] ?? 0;
          scoredIndex += 1;
          const intensity = max > 0 ? Math.min(Math.abs(value) / max, 1) : 0;
          const positive = value >= 0;
          const band = bandOf(intensity);
          const hue = positive ? VIOLET : EMERALD;
          return (
            <span
              key={index}
              className={`rounded-sm px-0.5 ${BAND_TEXT[band]}`}
              style={{
                backgroundColor: `rgba(${hue}, ${BAND_ALPHA[band]})`,
                boxShadow: BAND_RULE[band]
                  ? `inset 0 -${BAND_RULE[band]}px 0 0 rgba(${hue}, ${BAND_RULE_ALPHA[band]})`
                  : undefined,
              }}
              title={`${token.token}: ${value.toFixed(4)} (${BAND_NAME[band]} ${
                positive ? 'toward' : 'away'
              })`}
              data-testid="token-scored"
              data-band={BAND_NAME[band]}
            >
              {shown}
            </span>
          );
        })}
      </p>
      {/*
        ⚠ A LEGEND THAT NAMES THE COLOURS IS NOT AN EXPLANATION. The first version said
        "red is toward the concept, green away from it" and stopped there, which tells a
        reader which swatch is which and nothing about what the probe DID. What was missing
        is the arithmetic: every token gets its own score, the rule pools them into one
        aggregate, and the aggregate alone meets the threshold. Without that, the colours
        look like a verdict per token — and a reader reasonably concludes that a single
        violet token means the probe fired, which is not what any of this says.
      */}
      <div className="mt-3 space-y-2 rounded border border-slate-700 bg-slate-800/40 p-3 text-[11px] leading-5 text-slate-400">
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {/* The ramp itself, so a reader can match a token against the four bands. */}
          <span className="flex items-center gap-1.5" data-testid="legend-up">
            <span className="flex">
              {BAND_NAME.map((name, band) => (
                <span
                  key={name}
                  className={`inline-block w-5 px-0.5 text-center text-[9px] leading-4 ${BAND_TEXT[band]}`}
                  style={{
                    backgroundColor: `rgba(${VIOLET}, ${BAND_ALPHA[band]})`,
                    boxShadow: BAND_RULE[band]
                      ? `inset 0 -${BAND_RULE[band]}px 0 0 rgba(${VIOLET}, ${BAND_RULE_ALPHA[band]})`
                      : undefined,
                  }}
                >
                  {name[0]}
                </span>
              ))}
            </span>
            <span className="text-slate-200">up</span>
            {concept ? <>— toward <span className="text-slate-200">{concept}</span></> : null}
          </span>
          <span className="flex items-center gap-1.5" data-testid="legend-down">
            <span className="flex">
              {BAND_NAME.map((name, band) => (
                <span
                  key={name}
                  className={`inline-block w-5 px-0.5 text-center text-[9px] leading-4 ${BAND_TEXT[band]}`}
                  style={{
                    backgroundColor: `rgba(${EMERALD}, ${BAND_ALPHA[band]})`,
                    boxShadow: BAND_RULE[band]
                      ? `inset 0 -${BAND_RULE[band]}px 0 0 rgba(${EMERALD}, ${BAND_RULE_ALPHA[band]})`
                      : undefined,
                  }}
                >
                  {name[0]}
                </span>
              ))}
            </span>
            <span className="text-slate-200">down</span> — away from it
          </span>
          <span className="flex items-center gap-1.5">
            <span className="rounded border border-slate-600 bg-slate-800 px-1 text-slate-400">
              &lt;|…|&gt;
            </span>
            chat-template scaffolding, not your words
          </span>
          <span className="flex items-center gap-1.5">
            <span className="text-slate-500 underline decoration-dotted">not scored</span>
            outside this probe&apos;s scope
          </span>
        </div>
        <p>
          Every token gets its own score — the probe&apos;s vector read against that token&apos;s
          activation. The{' '}
          {rule ? <span className="font-mono text-slate-200">{rule}</span> : 'pooling'} rule
          combines them into the single <span className="text-slate-200">aggregate</span> above,
          and <span className="text-slate-200">only the aggregate meets the threshold</span>. One
          strongly-coloured token does not mean the probe fired, and a fired probe does not mean
          any particular token caused it.
        </p>
        <p>
          Four bands — <span className="text-slate-300">faint, weak, moderate, strong</span> —
          rank the tokens <em>against each other within this input</em>, so the shading
          always spreads even when one token dominates. It is a ranking, not a measurement:
          bands do not compare across two inputs, and a dark token in a passage that scored
          low is still a low score. Hover any token for its exact number.
        </p>
      </div>
    </div>
  );
}
