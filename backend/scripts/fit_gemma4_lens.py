"""Fit a J-lens for gemma-4-12B-it, following the Neuronpedia reference recipe.

Reference (neuronpedia/jacobian-lens, gemma-4-31b config.yaml):
    --n_prompts 1000 --max_seq_len 128 --max_chars 2000
    --dtype bfloat16 --stop_at_delta 0.002

Two deliberate departures, both recorded in corpus_name so the artifact never
misrepresents how it was made:

  CORPUS. Reference uses wikitext-103; this uses the locally registered
  OpenWebText-2M, which is what miStudio's existing gemma-2-2b lens was fitted
  on (`openwebtext-2m-1200docs`). Both are general text. Using the local one
  avoids a download and keeps this comparable to the lens already on disk.

  PRECISION. Reference fits in bfloat16 with 178 GB free. This card has 24 GB
  and the model is 23 GB in bf16, so the fit runs at Q8. The lens therefore
  describes quantised weights -- which is also what miLLM serves.

freeze_qk=True: the source paper reports frozen-Q/K directions respond MORE
strongly to intervention, and this lens is being built for coordinate-swap work
rather than reading.
"""

import json
import os
import sys

import httpx
from sqlalchemy import text

from src.core.database import SyncSessionLocal

MODEL_ID = "m_b55c6926"
N_PROMPTS = int(os.environ.get("N_PROMPTS", "1200"))
# Reference passes --max_chars 2000 but --max_seq_len 128, so the model only
# ever sees ~128 tokens; the char cap is just an upper bound before tokenisation
# truncates. 550 chars is ~128 tokens, which matches the reference's EFFECTIVE
# input and keeps the payload under the API's 1024 KB body cap -- 1200 x 1812
# chars was 2.2 MB and got a 413.
MAX_CHARS = int(os.environ.get("MAX_CHARS", "550"))
API = "http://localhost:8000/api/v1/jlens/fit"


def main():
    db = SyncSessionLocal()
    raw = db.execute(text(
        "SELECT raw_path FROM datasets WHERE name='OpenWebText-2M'")).scalar()
    if not raw:
        print("OpenWebText-2M not registered"); return 1

    from datasets import load_from_disk
    ds = load_from_disk(raw)
    if hasattr(ds, "keys"):
        ds = ds[list(ds.keys())[0]]
    field = "text" if "text" in ds.column_names else ds.column_names[0]

    prompts, seen = [], 0
    for row in ds:
        seen += 1
        t = (row.get(field) or "").strip()
        # Skip stubs: a near-empty prompt contributes a linearisation around
        # almost no context and drags the mean J toward the null input.
        if len(t) < 400:
            continue
        prompts.append(t[:MAX_CHARS])
        if len(prompts) >= N_PROMPTS:
            break

    print(f"corpus   : OpenWebText-2M ({raw})")
    print(f"prompts  : {len(prompts)} kept from {seen} scanned, <=%d chars" % MAX_CHARS)
    print(f"mean len : {sum(len(p) for p in prompts)//max(len(prompts),1)} chars")

    body = {
        "model_id": MODEL_ID,
        "prompts": prompts,
        "freeze_qk": True,
        "freeze_norms": False,
        "target_layer": "penultimate",
        "convergence_delta": 0.002,
        "corpus_name": f"openwebtext-2m-{len(prompts)}docs-q8-freezeqk",
        # The intermediate must not appear in the prompt, and the control is a
        # prompt where it would be absurd -- a lens that answers the same thing
        # to everything must FAIL rather than pass.
        "semantic_probe": {
            "prompt": "The capital of France is",
            "expected_intermediate": " Paris",
            "control_prompt": "The recipe calls for two cups of",
            "top_k": 10,
        },
    }
    r = httpx.post(API, json=body, timeout=600)
    print(f"\nHTTP {r.status_code}")
    print(json.dumps(r.json(), indent=2)[:600] if r.status_code < 400 else r.text[:600])
    return 0 if r.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
