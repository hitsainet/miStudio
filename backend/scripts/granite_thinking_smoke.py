"""Does granite-4.2-8b return a LABEL, or its own deliberation?

Runs a real feature through the real labeling path (OpenAILabelingService, the
batched call bulk labeling uses) against granite, and reports whether reasoning
leaked into the parsed output.

The check that matters is NOT `'<think>' in content`. granite's template opens
the think tag in the GENERATION PROMPT, so the completion carries reasoning with
no opening tag -- that test returns False on a response that is pure
deliberation, which is exactly how this was missed the first time. So this
script looks for the shape of reasoning prose as well, and prints the raw
content for a human to judge.

Usage, inside the mistudio backend pod:
    env PYTHONPATH=/app python3 /tmp/granite_thinking_smoke.py [model] [n]
"""

import asyncio
import os
import sys
import time

from sqlalchemy import text

from src.core.database import SyncSessionLocal
from src.services.openai_labeling_service import OpenAILabelingService

EXTRACTION = os.environ.get("EXTRACTION", "extr_20260901_084144_sae_sae_eb48")
TEMPLATE_ID = "lpt_95fb74cb61354eb5"
ENDPOINT = os.environ.get(
    "MILLM_ENDPOINT", "http://millm-backend.millm.svc.cluster.local:8000/v1")

# Phrases that mark deliberation rather than an answer. Cheap and imperfect,
# but it catches the failure this script exists for.
TELLS = ("okay, let", "let's see", "wait,", "hmm", "i need to", "the user",
         "but the", "however", "first, i", "looking at")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "granite-4.2-8b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    db = SyncSessionLocal()

    t = db.execute(text("""
        SELECT system_message, user_prompt_template, temperature, max_tokens,
               top_p, template_type, max_examples, include_prefix,
               include_suffix, prime_token_marker
        FROM labeling_prompt_templates WHERE id = :i
    """), {"i": TEMPLATE_ID}).fetchone()
    (system_message, user_prompt_template, temperature, max_tokens, top_p,
     template_type, max_examples, inc_pre, inc_suf, marker) = t

    tc = {"template_type": template_type, "max_examples": max_examples,
          "include_prefix": inc_pre, "include_suffix": inc_suf,
          "prime_token_marker": marker, "include_logit_effects": False,
          "top_promoted_tokens_count": None, "top_suppressed_tokens_count": None,
          "include_negative_examples": False, "num_negative_examples": None}

    total = db.execute(text("SELECT count(*) FROM features WHERE extraction_job_id=:e"),
                       {"e": EXTRACTION}).scalar()
    stride = max(1, total // n)
    feats = db.execute(text("""
        SELECT id, neuron_index FROM features
        WHERE extraction_job_id = :e AND MOD(neuron_index, :s) = 0
        ORDER BY neuron_index LIMIT :n
    """), {"e": EXTRACTION, "s": stride, "n": n}).fetchall()

    reqs = []
    for fid, idx in feats:
        ex = db.execute(text("""
            SELECT prefix_tokens, prime_token, suffix_tokens, max_activation
            FROM feature_activations WHERE feature_id = :f
            ORDER BY max_activation DESC LIMIT :k
        """), {"f": fid, "k": max_examples or 10}).fetchall()
        if not ex:
            continue
        reqs.append({"examples": [{"prefix_tokens": r[0], "prime_token": r[1],
                                   "suffix_tokens": r[2], "max_activation": r[3]}
                                  for r in ex],
                     "template_config": tc,
                     "user_prompt_template": user_prompt_template,
                     "system_message": system_message,
                     "feature_id": fid, "neuron_index": idx})

    # Granite mandates temperature=1.0 / top_p=0.95 "across all tasks and
    # serving backends" (model card, Generation Parameters). Our labeling
    # template runs 0.2/0.9, which is off-spec and could degrade granite for
    # reasons that have nothing to do with it being a good or bad judge. Both
    # are measured so the big run is not configured on a guess.
    arms = (
        ("template sampling  t=%.1f p=%.2f" % (temperature, top_p),
         temperature, top_p, max_tokens or 300),
        ("granite spec       t=1.0 p=0.95", 1.0, 0.95, 2048),
    )
    for tag, temp, tp, mt in arms:
        print(f"\n{'='*74}\n### {model} — {tag}\n{'='*74}", flush=True)
        svc = OpenAILabelingService(
            api_key="unused", base_url=ENDPOINT, model=model,
            temperature=temp, max_tokens=mt, top_p=tp)

        # Compatibility shim: this pod may predate the chat_template_kwargs
        # support. Rather than wait for a rollout, inject it here. Contained to
        # this process -- the serving code on the pod is untouched.
        CTK = {"enable_thinking": False}
        if not getattr(svc, "chat_template_kwargs", None):
            svc.chat_template_kwargs = CTK
            orig = svc._call_openai_batched

            async def _patched(message_sets, _orig=orig, _svc=svc):
                real = _svc.client.chat.completions.with_raw_response.create

                async def _wrap(**kw):
                    kw.setdefault("extra_body", {})
                    kw["extra_body"]["chat_template_kwargs"] = CTK
                    return await real(**kw)

                _svc.client.chat.completions.with_raw_response.create = _wrap
                try:
                    return await _orig(message_sets)
                finally:
                    _svc.client.chat.completions.with_raw_response.create = real

            svc._call_openai_batched = _patched
            print("  (shim active: service predates chat_template_kwargs)")
        print("  sending chat_template_kwargs:", CTK)
        loop = asyncio.new_event_loop()
        try:
            t0 = time.time()
            out = loop.run_until_complete(
                svc.generate_labels_from_examples_batched(reqs))
            dt = time.time() - t0
        finally:
            loop.close()
        leaked = 0
        for r, label in zip(reqs, out):
            cat = (label or {}).get("category")
            spec = (label or {}).get("specific")
            blob = f"{cat} {spec}".lower()
            bad = [w for w in TELLS if w in blob]
            leaked += bool(bad)
            print(f"  idx {r['neuron_index']:<6} category={str(cat)[:28]:<28} "
                  f"specific={str(spec)[:40]:<40}{'  <-- REASONING LEAK' if bad else ''}")
        print(f"  {len(reqs)} features in {dt:.0f}s; reasoning leaked into "
              f"{leaked}/{len(reqs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
