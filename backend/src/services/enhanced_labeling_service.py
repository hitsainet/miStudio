"""
Enhanced per-feature labeling service.

Two-pass LLM strategy:
  Pass 1 — parallel per-example summarization
            For each of the top-N activation examples, asks the LLM:
            "What is this token doing in this specific context? One sentence."
  Pass 2 — synthesis
            Feeds all per-example summaries back and asks:
            "What is the unifying concept? Produce a structured label with reasoning."

The notes field receives:
    <synthesis reasoning paragraph>

    ---

    <per-example summary table (markdown)>
"""

import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import httpx
from openai import OpenAI, BadRequestError, OpenAIError

logger = logging.getLogger(__name__)

PER_EXAMPLE_MAX_TOKENS = 80
SYNTHESIS_MAX_TOKENS = 500
# Reasoning models (gpt-5*, o1*, o3*, o4*) consume max_completion_tokens
# for *both* internal reasoning and output. The 80/500 budgets above only
# cover OUTPUT — for reasoning models we add headroom so the JSON answer
# isn't truncated to empty by the reasoning trace.
REASONING_PER_EXAMPLE_MAX_TOKENS = 4000
REASONING_SYNTHESIS_MAX_TOKENS = 16000
_BPE_SPACE = "\u0120"  # Ġ — BPE space-prefix marker


class EnhancedLabelingError(Exception):
    pass


class EnhancedLabelingService:
    """
    Two-pass enhanced labeling for a single SAE feature.

    Args:
        endpoint: OpenAI-compatible base URL (e.g. http://k8s-millm.hitsai.local/v1,
                  or https://api.openai.com/v1 for the real OpenAI API)
        model:    Model identifier (e.g. gemma-4-E4B-it or gpt-4o-mini)
        workers:  Max concurrent HTTP calls during pass 1
        api_key:  Optional Bearer token. Required for api.openai.com; ignored by
                  unauthenticated local endpoints (miLLM/Ollama/vLLM).
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        workers: int = 8,
        api_key: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.workers = workers
        # Use the official OpenAI Python SDK. It handles retries, parameter
        # quirks per model family, error parsing, and connection pooling out
        # of the box — so we don't hand-roll any of that.
        self._openai = OpenAI(
            api_key=api_key or "not-needed",
            base_url=self.endpoint,
            timeout=httpx.Timeout(90.0, connect=10.0),
            max_retries=0,  # we handle retries at this layer for our own logging
        )

    def close(self) -> None:
        try:
            self._openai.close()
        except Exception:
            pass

    # ── LLM call ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        """
        OpenAI reasoning-class models (o1*, o3*, o4*, gpt-5*) reject ``temperature``
        and require ``max_completion_tokens`` instead of ``max_tokens``. Detect by
        prefix so we send the right shape on /chat/completions.
        """
        m = model_name.lower()
        return (
            m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
            or m.startswith("gpt-5")
        )

    def _call_llm(self, prompt: str, max_tokens: int, retries: int = 3) -> str:
        """
        Single chat-completion call via the official OpenAI Python SDK.

        The SDK handles per-model parameter validation (e.g. reasoning models
        rejecting ``temperature``, accepting ``max_completion_tokens`` etc.)
        and surfaces real error text on failure. We do NOT hand-roll
        model-family heuristics anymore — let the API tell us what it wants.
        """
        is_reasoning = self._is_reasoning_model(self.model)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_reasoning:
            # Reasoning-class models: budget is shared between reasoning trace
            # and output, so use the larger configured cap. Don't pass
            # temperature (rejected) or reasoning_effort (value set varies by
            # model — let the model use its default).
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0.1

        last_err: str | None = None
        for attempt in range(retries):
            try:
                resp = self._openai.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    finish = resp.choices[0].finish_reason
                    usage = resp.usage.model_dump() if resp.usage else {}
                    last_err = (
                        f"OpenAI returned empty content (finish_reason={finish}, "
                        f"usage={usage}). For reasoning models this usually means "
                        f"max_completion_tokens={max_tokens} was too small — the "
                        f"reasoning trace consumed the budget. Increase the budget "
                        f"or pick a non-reasoning model (e.g. gpt-4o-mini)."
                    )
                    raise EnhancedLabelingError(last_err)
                return content
            except BadRequestError as exc:
                # 400 from the API — don't retry, the request shape is wrong
                # and won't get better next attempt. Surface OpenAI's exact
                # error message so the caller can fix configuration.
                detail = getattr(exc, "message", None) or str(exc)
                last_err = f"OpenAI 400: {detail}"
                logger.warning(
                    "Enhanced labeling LLM call rejected: %s (not retrying)",
                    last_err,
                )
                break
            except EnhancedLabelingError:
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    "Enhanced labeling LLM call attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, retries, last_err, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                # Catches OpenAIError (rate limits, server errors, timeouts)
                # and generic network failures. Retry with backoff.
                last_err = f"{type(exc).__name__}: {exc}"
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    "Enhanced labeling LLM call attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, retries, last_err, wait,
                )
                time.sleep(wait)
        raise EnhancedLabelingError(
            f"LLM call failed after {retries} retries: {last_err}"
        )

    # ── token cleaning ────────────────────────────────────────────────────────

    @staticmethod
    def _clean_token(t: str) -> str:
        """Replace BPE space-prefix marker and common mojibake."""
        _MOJIBAKE = [
            ("\u00e2\u0080\u0099", "'"), ("\u00e2\u0080\u009c", '"'),
            ("\u00e2\u0080\u009d", '"'), ("\u00e2\u0080\u0093", "–"),
            ("\u00e2\u0080\u0094", "—"), ("\u00e2\u0080\u00a6", "…"),
        ]
        for bad, good in _MOJIBAKE:
            t = t.replace(bad, good)
        return t.replace(_BPE_SPACE, " ")

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        return "".join(EnhancedLabelingService._clean_token(t) for t in tokens)

    # ── pass 1: per-example summarization ────────────────────────────────────

    def _summarize_example(self, row: dict) -> str:
        prime = self._clean_token(row["prime_token"] or "?")
        prefix = self._join_tokens(row.get("prefix_tokens") or [])
        suffix = self._join_tokens(row.get("suffix_tokens") or [])
        act = row.get("max_activation") or 0.0

        # Trim long contexts
        if len(prefix) > 120:
            prefix = "..." + prefix[-120:]
        if len(suffix) > 120:
            suffix = suffix[:120] + "..."

        prompt = (
            f"A language-model feature fires on the token [{prime.strip()}] "
            f"(activation: {act:.2f}) in this passage:\n\n"
            f"  {prefix}[{prime.strip()}]{suffix}\n\n"
            f"In ONE sentence, describe the precise linguistic or semantic role "
            f"[{prime.strip()}] is playing here. Be concrete."
        )
        max_t = REASONING_PER_EXAMPLE_MAX_TOKENS if self._is_reasoning_model(self.model) else PER_EXAMPLE_MAX_TOKENS
        raw = self._call_llm(prompt, max_t)
        # Strip wrapping quotes/fences the model sometimes adds
        return re.sub(r'^["\'`]+|["\'`]+$', "", raw.strip())

    # ── pass 2: synthesis ─────────────────────────────────────────────────────

    def _synthesize(
        self,
        activation_rows: list[dict],
        summaries: list[tuple[dict, str]],
    ) -> dict[str, str]:
        # Prime token frequency table
        all_primes = [
            self._clean_token(r["prime_token"] or "").strip()
            for r in activation_rows
            if r.get("prime_token")
        ]
        counter = Counter(all_primes)
        freq_lines = [
            f"  {cnt:3d}x ({100*cnt/len(activation_rows):4.0f}%)  [{tok}]"
            for tok, cnt in counter.most_common(15)
        ]

        summary_block = "\n".join(
            f"{i+1:3d}. [act={ex.get('max_activation',0):.2f}, "
            f"token={self._clean_token(ex.get('prime_token','') or '').strip()!r}] {note}"
            for i, (ex, note) in enumerate(summaries)
        )

        prompt = (
            f"You are analyzing a sparse autoencoder (SAE) feature from a language model.\n"
            f"The feature fires on specific tokens. You examined {len(summaries)} examples "
            f"and wrote one-sentence observations for each.\n\n"
            f"PRIME TOKEN FREQUENCIES (across all {len(activation_rows)} examples):\n"
            + "\n".join(freq_lines)
            + f"\n\nPER-EXAMPLE OBSERVATIONS:\n{summary_block}\n\n"
            f"Identify the single unifying concept this feature has learned.\n"
            f"Return ONLY this JSON object:\n"
            f'{{\n'
            f'  "name": "snake_case_slug_max_5_words",\n'
            f'  "category": "snake_case_category_slug",\n'
            f'  "description": "One precise sentence describing the firing pattern.",\n'
            f'  "confidence": "high|medium|low",\n'
            f'  "reasoning": "One paragraph explaining the unifying pattern and any notable edge cases."\n'
            f'}}'
        )
        max_t = REASONING_SYNTHESIS_MAX_TOKENS if self._is_reasoning_model(self.model) else SYNTHESIS_MAX_TOKENS
        raw = self._call_llm(prompt, max_t)
        return self._parse_json(raw), raw

    @staticmethod
    def _parse_json(text: str) -> dict[str, str]:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        result: dict[str, str] = {}
        for field in ("name", "category", "description", "confidence", "reasoning"):
            m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if m:
                result[field] = m.group(1).replace('\\"', '"')
        if not result:
            raise EnhancedLabelingError(f"Could not parse JSON from synthesis response:\n{text}")
        return result

    # ── notes formatter ───────────────────────────────────────────────────────

    @staticmethod
    def _build_notes(reasoning: str, summaries: list[tuple[dict, str]]) -> str:
        rows = []
        for ex, note in summaries:
            prime = EnhancedLabelingService._clean_token(
                ex.get("prime_token") or ""
            ).strip()
            act = ex.get("max_activation") or 0.0
            note_escaped = note.replace("|", "\\|")
            rows.append(f"| {act:.2f} | `{prime}` | {note_escaped} |")

        table = (
            "| Act | Token | Summary |\n"
            "|-----|-------|---------|\n"
            + "\n".join(rows)
        )
        return f"{reasoning}\n\n---\n\n{table}"

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        activation_rows: list[dict],
        max_examples: int,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute both passes and return a dict with all label fields plus metadata.

        Args:
            activation_rows: Feature activation rows from DB, pre-sorted by
                             max_activation DESC.
            max_examples:    How many of the top rows to summarize in pass 1.
            progress_cb:     Called with (n_completed, total) after each pass-1 future.

        Returns:
            {name, category, description, notes, raw_synthesis, pass1_summaries}
        """
        top_rows = activation_rows[:max_examples]
        total = len(top_rows)

        if not top_rows:
            raise EnhancedLabelingError("No activation examples available for this feature")

        # ── Pass 1 ────────────────────────────────────────────────────────────
        logger.info("Enhanced labeling pass 1: summarizing %d examples (%d workers)", total, self.workers)
        results: dict[int, tuple[dict, str]] = {}

        # NOT a `with` block. `Executor.__exit__` calls `shutdown(wait=True)`
        # WITHOUT `cancel_futures`, so raising out of the progress callback
        # still waits for every already-queued example to finish — on a slow
        # endpoint that is the entire remaining bill, which is precisely the
        # cost the cancel route claims to avoid.
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            future_to_idx = {
                pool.submit(self._summarize_example, row): i
                for i, row in enumerate(top_rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    note = future.result()
                except Exception as exc:
                    logger.warning("Pass-1 example %d failed: %s", idx, exc)
                    note = "(summarization failed)"
                results[idx] = (top_rows[idx], note)
                if progress_cb:
                    progress_cb(len(results), total)
        finally:
            # cancel_futures=True is the whole point. Without it — and
            # `with ThreadPoolExecutor(...)` gives exactly that — a cancellation
            # raised from `progress_cb` still waits for every example already
            # submitted, so a 100-example job on a slow endpoint pays for all
            # 100 and only then reports "cancelled".
            pool.shutdown(wait=False, cancel_futures=True)

        summaries = [results[i] for i in range(total)]
        pass1_summaries = [
            {
                "n": i + 1,
                "prime": self._clean_token(row.get("prime_token") or "").strip(),
                "activation": row.get("max_activation"),
                "summary": note,
            }
            for i, (row, note) in enumerate(summaries)
        ]

        # ── Pass 2 ────────────────────────────────────────────────────────────
        logger.info("Enhanced labeling pass 2: synthesis")
        label, raw_synthesis = self._synthesize(activation_rows, summaries)

        name = label.get("name", "").strip().replace(" ", "_").lower()
        category = label.get("category", "").strip().replace(" ", "_").lower()
        description = label.get("description", "").strip()
        reasoning = label.get("reasoning", "").strip()
        notes = self._build_notes(reasoning, summaries)

        return {
            "name": name,
            "category": category,
            "description": description,
            "notes": notes,
            "raw_synthesis": raw_synthesis,
            "pass1_summaries": pass1_summaries,
        }
