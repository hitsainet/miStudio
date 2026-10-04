"""Is gemma-4's refusal verdict stable, or is it a coin flip?

This decides whether the 7,504 existing L46 labels are usable as ground truth.
It matters because NO measurable property of these features predicts those
refusals — across every column in the features table plus every NLP statistic,
the best AUC was 0.595, barely off chance. Two readings fit that:

  A. interpretability is semantic and the statistics are too shallow to see it
  B. the refusals are substantially arbitrary

Only this test separates them. If gemma-4 refuses the SAME features when shown
the SAME evidence in a different order, refusals are a stable property and
reading A holds. If it flips, reading B holds and 42% of the labeling output is
noise — which no template change would fix.

Deliberately uses the PRODUCTION path: the template that actually produced the
labels (frozen from the DB), OpenAILabelingService.generate_label_from_examples,
and therefore the real _parse_dual_label and _enforce_refusal. A hand-rolled
prompt would measure a different thing.

Example ORDER is the only variable. Runs are serial, never batched: batch
composition changes greedy output under int8, which would confound the very
thing being measured.

Usage, inside the mistudio backend pod:
    env PYTHONPATH=/app python3 /tmp/judge_consistency.py [model] [runs] [n_per_class]
"""

import asyncio
import os
import random
import sys
import time
from collections import defaultdict

from sqlalchemy import text

from src.core.database import SyncSessionLocal
from src.services.openai_labeling_service import OpenAILabelingService

EXTRACTION = "extr_20260828_080834_sae_sae_39cc_002"
TEMPLATE_ID = "lpt_95fb74cb61354eb5"          # the one that made the labels
ENDPOINT = os.environ.get(
    "MILLM_ENDPOINT",
    "http://millm-backend.millm.svc.cluster.local:8000/v1",
)
REFUSAL = {"uninterpretable", "noise", "none", "unknown", ""}


def is_refusal(category, specific):
    return (category or "").lower() == "uninterpretable" or \
           (specific or "").lower() in REFUSAL


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-12B-it"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    per_class = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    db = SyncSessionLocal()

    t = db.execute(text("""
        SELECT system_message, user_prompt_template, temperature, max_tokens,
               top_p, template_type, max_examples, include_prefix,
               include_suffix, prime_token_marker
        FROM labeling_prompt_templates WHERE id = :i
    """), {"i": TEMPLATE_ID}).fetchone()
    if not t:
        print(f"template {TEMPLATE_ID} not found")
        return 1
    (system_message, user_prompt_template, temperature, max_tokens, top_p,
     template_type, max_examples, inc_pre, inc_suf, marker) = t

    template_config = {
        "template_type": template_type, "max_examples": max_examples,
        "include_prefix": inc_pre, "include_suffix": inc_suf,
        "prime_token_marker": marker, "include_logit_effects": False,
        "top_promoted_tokens_count": None, "top_suppressed_tokens_count": None,
        "include_negative_examples": False, "num_negative_examples": None,
    }

    panel = []
    for cls, cond in (("refused", "category = 'uninterpretable'"),
                      ("labeled", "category <> 'uninterpretable'")):
        for r in db.execute(text(f"""
            SELECT id, neuron_index, name FROM features
            WHERE extraction_job_id = :e AND labeled_at IS NOT NULL AND {cond}
            ORDER BY neuron_index LIMIT :n
        """), {"e": EXTRACTION, "n": per_class}).fetchall():
            panel.append({"fid": r[0], "idx": r[1], "stored_cls": cls,
                          "stored_name": r[2]})

    examples = {}
    for p in panel:
        rows = db.execute(text("""
            SELECT prefix_tokens, prime_token, suffix_tokens, max_activation
            FROM feature_activations WHERE feature_id = :f
            ORDER BY max_activation DESC LIMIT :k
        """), {"f": p["fid"], "k": max_examples or 10}).fetchall()
        examples[p["fid"]] = [
            {"prefix_tokens": r[0], "prime_token": r[1], "suffix_tokens": r[2],
             "max_activation": r[3]}
            for r in rows
        ]

    print(f"model     : {model}")
    print(f"template  : {TEMPLATE_ID} (the one that produced the stored labels)")
    print(f"panel     : {len(panel)} features "
          f"({per_class} stored-refused / {per_class} stored-labeled)")
    print(f"runs      : {runs}, example ORDER shuffled each run, serial (never "
          f"batched)\n")

    svc = OpenAILabelingService(
        api_key="unused", base_url=ENDPOINT, model=model,
        temperature=temperature, max_tokens=max_tokens or 300, top_p=top_p,
    )

    verdicts = defaultdict(list)   # fid -> [bool is_refusal]
    labels = defaultdict(list)     # fid -> [specific]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    t0 = time.time()
    try:
        for run in range(runs):
            for p in panel:
                ex = list(examples[p["fid"]])
                random.Random(f'{p["fid"]}:{run}').shuffle(ex)
                try:
                    label = loop.run_until_complete(
                        svc.generate_label_from_examples(
                            examples=ex,
                            template_config=template_config,
                            user_prompt_template=user_prompt_template,
                            system_message=system_message,
                            feature_id=p["fid"],
                            neuron_index=p["idx"],
                        )
                    )
                except Exception as exc:
                    print(f"   ! {p['idx']} run{run}: {type(exc).__name__}: {exc}")
                    continue
                verdicts[p["fid"]].append(
                    is_refusal(label.get("category"), label.get("specific")))
                labels[p["fid"]].append(label.get("specific"))
            print(f"  run {run+1}/{runs} done ({time.time()-t0:.0f}s)")
    finally:
        loop.close()
        asyncio.set_event_loop(None)

    complete = [p for p in panel if len(verdicts[p["fid"]]) == runs]
    print(f"\n{'='*74}")
    print(f"features with all {runs} runs complete: {len(complete)}/{len(panel)}")
    if not complete:
        print("nothing to measure")
        return 1

    stable = [p for p in complete if len(set(verdicts[p["fid"]])) == 1]
    flipped = [p for p in complete if len(set(verdicts[p["fid"]])) > 1]
    print(f"\n1. REFUSAL STABILITY across runs (order is the only difference)")
    print(f"   same verdict every run : {len(stable)}/{len(complete)} "
          f"({100*len(stable)/len(complete):.0f}%)")
    print(f"   FLIPPED                : {len(flipped)}/{len(complete)} "
          f"({100*len(flipped)/len(complete):.0f}%)")

    if flipped:
        print(f"\n   features that flipped (stored verdict -> this run's verdicts):")
        for p in flipped[:10]:
            seq = "".join("R" if v else "L" for v in verdicts[p["fid"]])
            print(f"     idx {p['idx']:<6} stored={p['stored_cls']:<8} runs={seq}")

    agree = [p for p in complete
             if all(v == (p["stored_cls"] == "refused") for v in verdicts[p["fid"]])]
    print(f"\n2. AGREEMENT WITH THE STORED VERDICT")
    print(f"   all runs match the stored label : {len(agree)}/{len(complete)} "
          f"({100*len(agree)/len(complete):.0f}%)")

    same_label = sum(1 for p in complete
                     if len(set(x for x in labels[p["fid"]] if x)) == 1)
    print(f"\n3. LABEL STABILITY (the exact `specific` string)")
    print(f"   identical every run : {same_label}/{len(complete)} "
          f"({100*same_label/len(complete):.0f}%)")

    pct = 100 * len(stable) / len(complete)
    print(f"\n{'='*74}")
    if pct >= 85:
        print(f"VERDICT: refusals are STABLE ({pct:.0f}%). Reading A — the "
              f"statistics are\ntoo shallow to predict a real semantic property. "
              f"The stored labels are\nusable as ground truth.")
    elif pct >= 65:
        print(f"VERDICT: refusals are PARTLY stable ({pct:.0f}%). Usable as a "
              f"weak target, but\nexpect a ceiling on any predictor trained "
              f"against them.")
    else:
        print(f"VERDICT: refusals are UNSTABLE ({pct:.0f}%). Reading B — a large "
              f"share of the\n42% refusal rate is noise, the stored labels are "
              f"NOT sound ground truth,\nand no template change fixes that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
