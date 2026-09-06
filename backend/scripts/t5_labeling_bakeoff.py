"""Can a 60M summarisation model produce a usable feature topic hint?

Falconsai/text_summarization is T5-small: encoder-decoder, 60.5M parameters,
fine-tuned for summarisation only. It cannot follow instructions and cannot emit
JSON, so every structured approach tried so far (category/specific/description,
or fit/topic) is impossible by construction.

What it CAN do maps onto one job: summarise the activating passages, and take
the summary as the coarse topic hint. That is a legitimate framing, not a
downgrade — a summary of ten passages is exactly "what these have in common".

Four prompt structures are compared on real features, because for a seq2seq
model the input format is the only lever there is.

Metrics, in order of how much they matter:
  DEGENERACY   - is the output empty, or just the input copied back?
  DISCRIMINATION - do different features get different summaries? (a model that
                   emits the same sentence for everything is useless however
                   fluent it is)
  STABILITY    - same passages, shuffled order: same summary?
  SEPARATION   - do consensus-labeled features differ from consensus-refused?

Runs the model DIRECTLY rather than through miLLM, so it does not depend on the
seq2seq generation fix having deployed.

Usage, inside the mistudio backend pod:
    env PYTHONPATH=/app python3 /tmp/t5_labeling_bakeoff.py [n_features] [runs]
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

from sqlalchemy import text

from src.core.database import SyncSessionLocal

MODEL = "Falconsai/text_summarization"
CACHE = "/data/hf_cache"
CONSENSUS = "/data/consensus_set.json"
MAX_INPUT = 512          # T5-small's positional limit
N_PASSAGES = 10


def clean(t):
    return str(t).replace("▁", " ").replace("Ġ", " ").replace("##", "")


def passage(row, marked=False):
    pre = "".join(clean(t) for t in (row[0] or [])[-12:])
    prime = clean(row[1] or "")
    suf = "".join(clean(t) for t in (row[2] or [])[:12])
    if marked:
        prime = f" *{prime.strip()}* "
    return " ".join((pre + prime + suf).split())


# The four structures. For a seq2seq model with no instruction following, the
# input format IS the prompt engineering — there is nothing else to tune.
def build_prompts(rows):
    plain = [passage(r) for r in rows]
    marked = [passage(r, marked=True) for r in rows]
    primes = [clean(r[1] or "").strip() for r in rows]
    return {
        # 1. T5's native task prefix over the raw passages.
        "summarize_prefix": "summarize: " + " ".join(plain),
        # 2. No prefix at all — the fine-tune may have baked the task in.
        "no_prefix": " ".join(plain),
        # 3. Prime tokens emphasised, so the shared token is salient in the
        #    input rather than buried in surrounding prose.
        "marked_primes": "summarize: " + " ".join(marked),
        # 4. Numbered passages: an explicit list structure, which summarisation
        #    training data often resembles more than a run-on paragraph.
        "numbered": "summarize: " + " ".join(
            f"({i+1}) {p}" for i, p in enumerate(plain)),
    }


def main():
    n_features = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    db = SyncSessionLocal()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL, cache_dir=CACHE)
    mdl.eval()
    print(f"model  : {MODEL} ({sum(p.numel() for p in mdl.parameters())/1e6:.1f}M, "
          f"encoder-decoder={mdl.config.is_encoder_decoder})")

    cons = json.load(open(CONSENSUS))
    feats = [f for f in cons["features"] if f.get("unanimous")]
    random.Random(7).shuffle(feats)
    # Balance the panel so separation is measurable at all.
    ref = [f for f in feats if f["consensus"]][:n_features // 2]
    lab = [f for f in feats if not f["consensus"]][:n_features // 2]
    panel = ref + lab
    print(f"panel  : {len(panel)} consensus features "
          f"({len(ref)} refused / {len(lab)} labeled)")
    print(f"runs   : {runs} (passage order shuffled)\n")

    rows_by_feat = {}
    for f in panel:
        rows_by_feat[f["feature_id"]] = db.execute(text("""
            SELECT prefix_tokens, prime_token, suffix_tokens
            FROM feature_activations WHERE feature_id = :f
            ORDER BY max_activation DESC LIMIT :k
        """), {"f": f["feature_id"], "k": N_PASSAGES}).fetchall()

    variants = list(build_prompts(rows_by_feat[panel[0]["feature_id"]]).keys())
    out = {v: defaultdict(list) for v in variants}
    truncated = defaultdict(int)

    t0 = time.time()
    for run in range(runs):
        for f in panel:
            rows = list(rows_by_feat[f["feature_id"]])
            random.Random(f'{f["feature_id"]}:{run}').shuffle(rows)
            for name, prompt in build_prompts(rows).items():
                enc = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=MAX_INPUT)
                if len(tok(prompt)["input_ids"]) > MAX_INPUT:
                    truncated[name] += 1
                with torch.no_grad():
                    gen = mdl.generate(**enc, max_new_tokens=40, num_beams=2)
                out[name][f["feature_id"]].append(
                    tok.decode(gen[0], skip_special_tokens=True).strip())
        print(f"  run {run+1}/{runs} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*78}")
    for name in variants:
        res = out[name]
        alls = [s for v in res.values() for s in v]
        uniq = len(set(alls))
        empty = sum(1 for s in alls if not s)
        # Stability: identical summary across runs for the same feature.
        stable = sum(1 for v in res.values() if len(set(v)) == 1)
        # Discrimination: distinct summaries across FEATURES within one run.
        first = [v[0] for v in res.values() if v]
        disc = len(set(first))
        print(f"\n### {name}")
        print(f"  truncated inputs : {truncated[name]}/{len(panel)*runs}")
        print(f"  empty outputs    : {empty}/{len(alls)}")
        print(f"  distinct summaries across features : {disc}/{len(first)}")
        print(f"  stable under reorder               : {stable}/{len(res)}")
        for f in panel[:3]:
            cls = "refused" if f["consensus"] else "labeled"
            print(f"    [{cls}] {res[f['feature_id']][0][:110]!r}")

    print(f"\n{'='*78}")
    print("A summary that is identical across features carries no information,")
    print("however fluent it reads. Discrimination is the first thing to check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
