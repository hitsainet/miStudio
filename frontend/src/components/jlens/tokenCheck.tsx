/**
 * "Is this one token?" — asked once, in one place, for every hand-typed token.
 *
 * A direction is a row of the unembedding matrix. A single token has exactly
 * one; a string the tokenizer SPLITS is steered along the MEAN of its pieces'
 * rows, and only its first piece is scored — usable, and weaker. Which case you
 * are in is not answerable by looking at the string: ' football' is one token
 * for one tokenizer and three for another, and the only honest answer comes
 * from the server (`POST /jlens/token-check`).
 *
 * So this is ADVICE, not a gate — with one exception. A string that encodes to
 * NOTHING has no row at all and cannot run; the callers block that case and
 * warn about the other.
 *
 * EXTRACTED SO THE VERBATIM RULE HAS ONE HOME. Both the intervention card and
 * the swap composer take a typed token, and the rule that the string must be
 * checked and sent WITHOUT trimming is the kind that survives in one copy and
 * quietly dies in the second. A leading space is a different token, not
 * whitespace to tidy: ' Paris' and 'Paris' are different rows of `W_U`, and
 * trimming here would check a different string than the one the run uses.
 */

import { useEffect, useState } from 'react';

import { jlensApi } from '../../api/jlens';
import type { JLensTokenCheck } from '../../types/jlens';

/**
 * What the model's vocabulary says about a hand-typed token.
 *
 * SHOWN EVEN WHEN IT PASSES. A silent success is indistinguishable from a check
 * that never ran, and the whole reason this control exists is that "is this one
 * token" is not answerable by looking at the string.
 */
export function TokenVerdict({
  check,
  busy,
}: {
  check?: JLensTokenCheck;
  busy: boolean;
}) {
  if (busy && !check) {
    return (
      <span className="text-[10px] text-slate-500 dark:text-slate-500">
        checking the model's vocabulary…
      </span>
    );
  }
  if (!check) return null;
  return (
    <span
      data-testid="token-verdict"
      className={`text-[10px] ${
        check.usable
          ? 'text-emerald-700 dark:text-emerald-400'
          : 'text-amber-700 dark:text-amber-400'
      }`}
    >
      {check.usable ? `id ${check.ids[0]} — ` : ''}
      {check.detail}
    </span>
  );
}

export interface TokenChecker {
  /** Verdicts so far, keyed by the VERBATIM string that was checked. */
  checks: Record<string, JLensTokenCheck>;
  /** A check is in flight. */
  checking: boolean;
  /** Check one token against the vocabulary. Safe to call repeatedly. */
  check: (raw: string) => Promise<void>;
  /** A verdict that says NO. Undefined when unchecked or fine. */
  rejected: (token: string) => JLensTokenCheck | undefined;
}

export function useTokenCheck(modelId: string): TokenChecker {
  const [checks, setChecks] = useState<Record<string, JLensTokenCheck>>({});
  const [checking, setChecking] = useState(false);

  // A VERDICT BELONGS TO ONE VOCABULARY. `usable`, the piece count and the
  // ids are all answers about a specific tokenizer, and the cache is keyed by
  // STRING alone. The intervention card is never remounted on a model change,
  // so ' Rome' checked against one model kept showing its green "one token"
  // badge for the next — suppressing the weaker-evidence warning, or firing the
  // encodes-to-nothing block, on a model where neither was true.
  useEffect(() => {
    setChecks({});
  }, [modelId]);

  /**
   * Check a hand-typed token against the model's vocabulary, on blur.
   *
   * ON BLUR, NOT ON EVERY KEYSTROKE: mid-word text is almost always multi-token
   * and an error that appears while you are still typing trains you to ignore
   * it. And the verdict is CACHED by string, so re-checking the same token is
   * free.
   */
  const check = async (raw: string) => {
    // KEYED AND SENT VERBATIM, whitespace included, because that is what will
    // be tokenised. `.trim()` here would check a different string than the one
    // the run uses.
    const t = raw;
    if (!t.trim() || !modelId || checks[t]) return;
    setChecking(true);
    try {
      const [verdict] = await jlensApi.checkTokens(modelId, [t]);
      if (verdict) setChecks((prev) => ({ ...prev, [t]: verdict }));
    } catch {
      // A FAILED CHECK MUST NOT BLOCK THE RUN. This is an early warning, not
      // a gate that can strand the form when the endpoint is unreachable — and
      // a multi-piece string is legal anyway, so the worst an unreachable
      // check costs is running without the weaker-evidence note. A string that
      // encodes to nothing still dies in the worker, which is the one case
      // this check would have caught.
    } finally {
      setChecking(false);
    }
  };

  const rejected = (t: string) => {
    const v = checks[t];
    return v && !v.usable ? v : undefined;
  };

  return { checks, checking, check, rejected };
}
