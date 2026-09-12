"""Build a SOUND ground-truth set by consensus, since single verdicts are not.

Measured on 2026-08-30: gemma-4-12B-it changes its refuse/label verdict on 47%
of features when the SAME ten passages are merely reordered, and reproduces its
own stored label only 30% of the time. A single verdict is therefore not ground
truth, and anything validated against one is validated against a coin flip.

This runs the same feature N times under deliberate perturbation and keeps only
the features where every run agrees. Those are the ones whose verdict is a
property of the feature rather than of the presentation. The set is smaller —
roughly half — but it is real, and it is what cohesion (or any other predictor)
should be measured against.

TWO perturbations are applied per run, deliberately:
  1. example ORDER is shuffled
  2. panel order is shuffled, which changes BATCH COMPOSITION — and batch
     composition changes greedy output under int8 (see inference_service
     _create_batched_chat_completion). Consensus that survives both is stronger
     than consensus over order alone.

Results are written after EVERY run, so a pod roll costs one run rather than the
whole job. That is not hypothetical: an ArgoCD rollout killed a labeling job at
7,504/30,719 earlier today, and Celery does not drain.

Usage, inside the mistudio backend pod:
    env PYTHONPATH=/app python3 /tmp/build_consensus_set.py [n_features] [runs]
"""

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict

from sqlalchemy import text

from src.core.database import SyncSessionLocal
from src.services.openai_labeling_service import OpenAILabelingService

EXTRACTION = os.environ.get("EXTRACTION", "extr_20260828_080834_sae_sae_39cc_002")
TEMPLATE_ID = "lpt_95fb74cb61354eb5"
ENDPOINT = os.environ.get(
    "MILLM_ENDPOINT",
    "http://millm-backend.millm.svc.cluster.local:8000/v1",
)
MODEL = os.environ.get("CONSENSUS_MODEL", "gemma-4-12B-it")
OUT = os.environ.get("CONSENSUS_OUT", "/data/consensus_set.json")
# Truncate stored context to N tokens either side. 0 = use whatever the
# extraction captured. This exists to separate two variables that changed
# together on 2026-09-01: expansion factor (8x -> 16x) AND context (25+25 ->
# 100+100). Truncating the RICH extraction back to 25+25 holds the SAE fixed
# and moves only the context, which is the only way to attribute the collapse
# in refusal rate (42% -> 3.5%) to one of them.
CONTEXT_TOKENS = int(os.environ.get("CONTEXT_TOKENS", "0"))
# Sampling overrides. Default to the TEMPLATE's values so a judge swap is the
# only variable versus an earlier run; granite-4.2-8b's card mandates
# temperature=1.0 / top_p=0.95 "across all tasks", which is worth measuring as
# a separate arm rather than silently mixing into the comparison.
TEMPERATURE = os.environ.get("TEMPERATURE")
TOP_P = os.environ.get("TOP_P")
REFUSAL = {"uninterpretable", "noise", "none", "unknown", ""}


def is_refusal(category, specific):
    return (category or "").lower() == "uninterpretable" or \
           (specific or "").lower() in REFUSAL


def main():
    n_features = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    db = SyncSessionLocal()

    t = db.execute(text("""
        SELECT system_message, user_prompt_template, temperature, max_tokens,
               top_p, template_type, max_examples, include_prefix,
               include_suffix, prime_token_marker
        FROM labeling_prompt_templates WHERE id = :i
    """), {"i": TEMPLATE_ID}).fetchone()
    (system_message, user_prompt_template, temperature, max_tokens, top_p,
     template_type, max_examples, inc_pre, inc_suf, marker) = t

    template_config = {
        "template_type": template_type, "max_examples": max_examples,
        "include_prefix": inc_pre, "include_suffix": inc_suf,
        "prime_token_marker": marker, "include_logit_effects": False,
        "top_promoted_tokens_count": None, "top_suppressed_tokens_count": None,
        "include_negative_examples": False, "num_negative_examples": None,
    }

    # Even stride across the WHOLE extraction, not just the labeled prefix.
    # Sampling only labeled features would bias the set toward whatever the
    # interrupted job happened to reach first.
    total = db.execute(text("SELECT count(*) FROM features WHERE extraction_job_id=:e"),
                       {"e": EXTRACTION}).scalar()
    stride = max(1, total // n_features)
    rows = db.execute(text("""
        SELECT id, neuron_index, category, name FROM features
        WHERE extraction_job_id = :e AND MOD(neuron_index, :s) = 0
        ORDER BY neuron_index LIMIT :n
    """), {"e": EXTRACTION, "s": stride, "n": n_features}).fetchall()
    panel = [{"fid": r[0], "idx": r[1],
              "stored": (None if r[2] is None else
                         ("refused" if is_refusal(r[2], r[3]) else "labeled")),
              "stored_name": r[3]} for r in rows]

    examples = {}
    for p in panel:
        ex = db.execute(text("""
            SELECT prefix_tokens, prime_token, suffix_tokens, max_activation
            FROM feature_activations WHERE feature_id = :f
            ORDER BY max_activation DESC LIMIT :k
        """), {"f": p["fid"], "k": max_examples or 10}).fetchall()
        def _cut(pre, suf):
            if not CONTEXT_TOKENS:
                return pre, suf
            # Nearest tokens to the prime are the ones a short window keeps:
            # the TAIL of the prefix and the HEAD of the suffix.
            return ((pre or [])[-CONTEXT_TOKENS:], (suf or [])[:CONTEXT_TOKENS])

        examples[p["fid"]] = [
            {"prefix_tokens": _cut(r[0], r[2])[0], "prime_token": r[1],
             "suffix_tokens": _cut(r[0], r[2])[1], "max_activation": r[3]}
            for r in ex
        ]
    panel = [p for p in panel if examples.get(p["fid"])]

    print(f"model      : {MODEL}")
    print(f"extraction : {EXTRACTION} ({total:,} features, stride {stride})")
    print(f"panel      : {len(panel)} features")
    print(f"runs       : {runs} (example order AND batch composition perturbed)")
    print(f"context    : {'%d+%d tokens (TRUNCATED)' % (CONTEXT_TOKENS, CONTEXT_TOKENS) if CONTEXT_TOKENS else 'as captured'}")
    print(f"output     : {OUT}\n", flush=True)

    _temp = float(TEMPERATURE) if TEMPERATURE else temperature
    _topp = float(TOP_P) if TOP_P else top_p
    print(f"sampling   : temperature={_temp} top_p={_topp}"
          f"{'  (OVERRIDDEN)' if (TEMPERATURE or TOP_P) else '  (from template)'}",
          flush=True)
    svc = OpenAILabelingService(
        api_key="unused", base_url=ENDPOINT, model=MODEL,
        temperature=_temp, max_tokens=max_tokens or 300, top_p=_topp,
    )
    print(f"chat_template_kwargs: {svc.chat_template_kwargs}", flush=True)

    verdicts = defaultdict(list)
    labels = defaultdict(list)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t0 = time.time()
    try:
        for run in range(runs):
            order = list(panel)
            random.Random(f"panel:{run}").shuffle(order)   # varies batching
            reqs = []
            for p in order:
                ex = list(examples[p["fid"]])
                random.Random(f'{p["fid"]}:{run}').shuffle(ex)   # varies order
                reqs.append({
                    "examples": ex, "template_config": template_config,
                    "user_prompt_template": user_prompt_template,
                    "system_message": system_message,
                    "feature_id": p["fid"], "neuron_index": p["idx"],
                })
            out = loop.run_until_complete(
                svc.generate_labels_from_examples_batched(reqs))
            for p, label in zip(order, out):
                verdicts[p["fid"]].append(
                    bool(is_refusal(label.get("category"), label.get("specific"))))
                labels[p["fid"]].append(label.get("specific"))

            done = sum(1 for p in panel if len(verdicts[p["fid"]]) == run + 1)
            print(f"  run {run+1}/{runs} done ({time.time()-t0:.0f}s, "
                  f"{done}/{len(panel)} features)", flush=True)

            # Persist after EVERY run — a rollout costs one run, not the job.
            _write(OUT, panel, verdicts, labels, runs, run + 1)
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    _report(panel, verdicts, labels, runs)
    return 0


def _write(path, panel, verdicts, labels, runs, completed):
    payload = {
        "extraction": EXTRACTION, "model": MODEL, "template": TEMPLATE_ID,
        "runs_planned": runs, "runs_completed": completed,
        "features": [
            {
                "feature_id": p["fid"], "neuron_index": p["idx"],
                "stored_verdict": p["stored"], "stored_name": p["stored_name"],
                "verdicts": verdicts[p["fid"]], "labels": labels[p["fid"]],
                "unanimous": (len(verdicts[p["fid"]]) == completed
                              and len(set(verdicts[p["fid"]])) == 1),
                "consensus": (verdicts[p["fid"]][0]
                              if verdicts[p["fid"]] and
                              len(set(verdicts[p["fid"]])) == 1 else None),
            }
            for p in panel
        ],
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)   # atomic; a killed write cannot corrupt the set


def _report(panel, verdicts, labels, runs):
    complete = [p for p in panel if len(verdicts[p["fid"]]) == runs]
    unan = [p for p in complete if len(set(verdicts[p["fid"]])) == 1]
    ref = [p for p in unan if verdicts[p["fid"]][0]]
    lab = [p for p in unan if not verdicts[p["fid"]][0]]
    print(f"\n{'='*74}")
    print(f"complete   : {len(complete)}/{len(panel)}")
    print(f"UNANIMOUS  : {len(unan)} ({100*len(unan)/max(len(complete),1):.0f}%)"
          f"  -> this is the usable ground truth")
    print(f"  refused  : {len(ref)}")
    print(f"  labeled  : {len(lab)}")
    print(f"discarded  : {len(complete)-len(unan)} features the model could not "
          f"decide consistently")

    agree = [p for p in unan if p["stored"] and
             p["stored"] == ("refused" if verdicts[p["fid"]][0] else "labeled")]
    scored = [p for p in unan if p["stored"]]
    if scored:
        print(f"\nof the unanimous features that ALSO have a stored verdict, "
              f"{len(agree)}/{len(scored)} "
              f"({100*len(agree)/len(scored):.0f}%) match it")
    if len(unan) < 40:
        print("\nWARNING: fewer than 40 unanimous features. That is a thin basis "
              "for validating\nanything — widen the panel before trusting a "
              "separation number computed on it.")


if __name__ == "__main__":
    raise SystemExit(main())
