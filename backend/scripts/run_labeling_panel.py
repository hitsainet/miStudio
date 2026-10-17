"""Launch a labeling run over a stratified panel, chunked to fit Celery.

Why chunking is not optional: celery_app.py sets task_time_limit=43200 (12 h)
and task_soft_time_limit=36000 (10 h). At the ~16 s/feature measured on
gemma-4-12B-it after the miLLM stop-token fix, the soft limit lands at ~2,250
features. A single 2,898-feature job would be killed mid-run.

Chunks are taken in the panel file's EXISTING order, which is md5(id||'order-v1').
That ordering matters: every stratum's members carry i.i.d. uniform hash values,
so any prefix of the file is simultaneously a random subsample of every stratum.
An interrupted run therefore leaves a scaled-down copy of the whole design, and
the reference sample stays unbiased on the prefix. Do NOT sort by anything that
correlates with expected quality — a run killed at hour 9 would then leave a
maximally cherry-picked set whose bias cannot be undone.

Usage:
  python scripts/run_labeling_panel.py PANEL.csv --extraction extr_... \
      --template lpt_... --model gemma-4-12B-it [--chunk 1500] [--dry-run]

The panel CSV needs a `feature_id` column; a `stratum` column is carried into
the run manifest when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request

CHUNK_DEFAULT = 1500  # ~6.7 h at 16 s/feature — inside the 10 h soft limit


def load_panel(path: str) -> list[tuple[str, str]]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "feature_id" not in rows[0]:
        raise SystemExit(f"{path}: expected a CSV with a feature_id column")
    return [(r["feature_id"], r.get("stratum", "")) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--extraction", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api", default="http://k8s-mistudio.hitsai.local")
    ap.add_argument("--endpoint",
                    default="http://millm-backend.millm.svc.cluster.local:8000/v1")
    ap.add_argument("--chunk", type=int, default=CHUNK_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-examples", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    panel = load_panel(a.panel)
    chunks = [panel[i:i + a.chunk] for i in range(0, len(panel), a.chunk)]
    est_h = len(panel) * 16 / 3600

    print(f"panel      : {len(panel)} features from {a.panel}")
    print(f"chunks     : {len(chunks)} x <= {a.chunk}")
    print(f"estimate   : ~{est_h:.1f} h total at 16 s/feature, SERIAL")
    print(f"             (miLLM MAX_CONCURRENT_REQUESTS=1 — measured 1.02x at")
    print(f"              concurrency 4, so batch_size buys latency hiding only)")
    strata: dict[str, int] = {}
    for _, s in panel:
        strata[s] = strata.get(s, 0) + 1
    for s in sorted(strata):
        print(f"  {s or '(none)':22s}{strata[s]}")

    if a.dry_run:
        print("\n--dry-run: nothing submitted")
        return 0

    print("\nNOTE: one chunk at a time. The apply path holds a per-extraction")
    print("409 lock, so a second chunk is refused until the first finishes.")
    for n, chunk in enumerate(chunks, 1):
        body = {
            "extraction_job_id": a.extraction,
            "feature_ids": [fid for fid, _ in chunk],
            "prompt_template_id": a.template,
            "labeling_method": "openai_compatible",
            "openai_compatible_endpoint": a.endpoint,
            "openai_compatible_model": a.model,
            "batch_size": a.batch_size,
            "max_examples": a.max_examples,
        }
        req = urllib.request.Request(
            f"{a.api}/api/v1/labeling/panel",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=120))
            print(f"  chunk {n}/{len(chunks)}: {r.get('id')} status={r.get('status')} "
                  f"total_features={r.get('total_features')}")
        except urllib.error.HTTPError as e:
            print(f"  chunk {n}/{len(chunks)}: HTTP {e.code} {e.read()[:300]!r}")
            return 1
        if n < len(chunks):
            print("     submit the next chunk once this one completes")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
