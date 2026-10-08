#!/usr/bin/env python3
"""Phase 2 hardware acceptance: cancelling a real activation extraction.

RUN FROM THE WORKSTATION. It drives the deployed k8s backend over `kubectl
exec`, because the two things it must observe live in different containers:

  * the API (start / poll / cancel)  -> the `backend` container, :8000
  * the GPU                          -> the `celery-worker` container, which is
                                        the only one with the card

The unit suite cannot demonstrate any of the four criteria. The repo's record
is that GPU bugs are found only on GPUs — Feature 20's `get_hookable_module`
arg-order crash survived three static review rounds and died on the first
hardware run.

  1. work STOPS at the next 10-sample boundary
  2. VRAM is FREED
  3. the row stays CANCELLED after the last in-flight progress write
  4. the worker picks up the next queued job

(3) is the one the unit suite fundamentally cannot reach: it is a race between
the endpoint's write and the worker's next `update_progress`, on a real
solo-pool worker with real timing. It is checked as an INVARIANT over every
observation, not as a final state — a final-state check would pass even if the
row flickered back to EXTRACTING for one write, which is the Phase 1 failure
mode exactly.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time

NS = "mistudio"


def _sh(cmd: str, timeout: int = 120) -> str:
    out = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "sean@192.168.244.61", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return out.stdout.strip()


def pod() -> str:
    """Resolve the pod name ON EVERY CALL.

    Captured once, this invalidated an entire acceptance run: ArgoCD replaced
    the pod mid-run (its Image Updater rolled the deployment for a new digest),
    every later `kubectl exec` addressed a name that no longer existed, and the
    script read empty responses as `None` statuses and a -1 GPU. The run
    reported FAIL for four criteria while the feature underneath was working.

    A test that cannot tell "the thing is broken" from "I lost my connection to
    it" produces a verdict about nothing.
    """
    name = _sh(
        f"kubectl get pods -n {NS} --no-headers "
        f"| awk '/mistudio-backend/ && / Running /{{print $1}}'"
    )
    return name.split()[0] if name.split() else ""


def api(_p, method: str, path: str, body: dict | None = None) -> dict:
    if body is None:
        cmd = (f"kubectl exec -n {NS} {pod()} -c backend -- "
               f"curl -s -X {method} http://localhost:8000{path}")
    else:
        payload = shlex.quote(json.dumps(body))
        cmd = (f"kubectl exec -n {NS} {pod()} -c backend -- "
               f"curl -s -X {method} -H 'Content-Type: application/json' "
               f"-d {payload} http://localhost:8000{path}")
    raw = _sh(cmd)
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"_raw": raw[:400]}


def gpu_used_gb(_p=None) -> float:
    """Used VRAM, read from the container that actually holds the card."""
    raw = _sh(
        f"kubectl exec -n {NS} {pod()} -c celery-worker -- python3 -c "
        f"\"import torch;f,t=torch.cuda.mem_get_info();print(round((t-f)/1024**3,2))\" "
        f"2>/dev/null | tail -1"
    )
    try:
        return float(raw)
    except ValueError:
        return -1.0


def newest_row_id(model: str, after: str = "") -> str:
    """The id of the newest extraction row for this model, from the DATABASE.

    The endpoint's returned id cannot be trusted on a deployment predating the
    fix: it was generated from its own `datetime.now()` and never handed to the
    task, which generated another. Resolving from the row makes this harness
    correct on both sides of that fix, and keeps it measuring WORKER behaviour
    rather than an API contract it is not trying to test.
    """
    q = (f"select id from activation_extractions where model_id='{model}'"
         f"{f" and id > '{after}'" if after else ''} "
         f"order by created_at desc limit 1")
    return _sh(
        f"kubectl exec -n {NS} deploy/postgres -- psql -U mistudio -d mistudio "
        f"-tAc \"{q}\""
    ).strip()


def extraction(p: str, model: str, ext_id: str) -> dict:
    body = api(p, "GET", f"/api/v1/models/{model}/extractions")
    for e in body.get("extractions", []):
        if e.get("extraction_id") == ext_id or e.get("id") == ext_id:
            return e
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--cancel-after", type=float, default=60.0)
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()

    p = pod()
    print(f"[pod] {p}")
    baseline = gpu_used_gb(p)
    print(f"[baseline] GPU {baseline} GB used")

    started = api(p, "POST", f"/api/v1/models/{args.model}/extract-activations", {
        "dataset_id": args.dataset,
        "layer_indices": [args.layer],
        "hook_types": ["residual"],
        "max_samples": args.samples,
        "batch_size": 4,
    })
    claimed = started.get("extraction_id") or started.get("id")
    if not claimed:
        print("FAILED to start:", json.dumps(started)[:400])
        return 1
    time.sleep(4)
    ext_id = newest_row_id(args.model) or claimed
    if ext_id != claimed:
        print(f"[!] endpoint returned {claimed}, the row is {ext_id} "
              f"— pre-fix deployment")
    print(f"[started] {ext_id}")

    # A SECOND job, queued behind it — criterion 4 needs something to pick up.
    # SLEEP FIRST: the extraction id is `ext_{model}_{YYYYmmdd_HHMMSS}`, so two
    # POSTs inside the same second collide and the "second" job is the first
    # one under another name. That made criterion 4 report on the row it was
    # supposed to be independent of.
    time.sleep(2)
    second = api(p, "POST", f"/api/v1/models/{args.model}/extract-activations", {
        "dataset_id": args.dataset,
        "layer_indices": [args.layer],
        "hook_types": ["residual"],
        "max_samples": 200,
        "batch_size": 4,
    })
    # ITS ROW DOES NOT EXIST YET. `extract_activations` creates the row when
    # the WORKER picks the task up, not when the endpoint queues it — so a
    # queued job has no row at all until the one ahead of it finishes. Looking
    # for it now returns nothing and falls back to the endpoint's guessed id,
    # which is what made criterion 4 report None while the worker had in fact
    # taken the next job two seconds after the cancel landed.
    print(f"[queued behind it] (no row until the worker starts it)")
    second_id = None

    peak, samples_at_cancel = baseline, 0
    deadline = time.time() + args.cancel_after
    while time.time() < deadline:
        time.sleep(5)
        peak = max(peak, gpu_used_gb(p))
        e = extraction(p, args.model, ext_id)
        samples_at_cancel = e.get("samples_processed") or samples_at_cancel
        print(f"  status={e.get('status')} samples={samples_at_cancel} "
              f"gpu={gpu_used_gb(p)} GB")

    # READ THE COUNT IMMEDIATELY BEFORE THE POST. The polling loop above is on
    # a 5-second tick and this job advances ~450 samples per tick, so measuring
    # from the last poll attributes up to a tick of ORDINARY PROGRESS to the
    # cancellation. The first run of this script failed criterion 1 at +132
    # samples for exactly that reason — the number was real, the attribution
    # was not.
    samples_at_cancel = (
        extraction(p, args.model, ext_id).get("samples_processed")
        or samples_at_cancel
    )
    print(f"[cancel] at {samples_at_cancel} samples")
    t0 = time.time()
    resp = api(p, "POST",
               f"/api/v1/models/{args.model}/extractions/{ext_id}/cancel")
    print(f"[cancel] {resp.get('message') or json.dumps(resp)[:200]}")

    observations, saw_cancelled = [], False
    for _ in range(40):
        time.sleep(3)
        e = extraction(p, args.model, ext_id)
        obs = (round(time.time() - t0, 1), str(e.get("status")),
               e.get("samples_processed"), gpu_used_gb(p))
        observations.append(obs)
        print(f"  +{obs[0]}s status={obs[1]} samples={obs[2]} gpu={obs[3]} GB")
        if obs[1].lower().endswith("cancelled"):
            saw_cancelled = True
        if saw_cancelled and time.time() - t0 > 45:
            break

    time.sleep(25)
    after = gpu_used_gb(p)
    final = observations[-1] if observations else (0, "?", None, -1)
    # Resolve it AFTER the wait: by now the worker has either taken it or not,
    # and that is precisely the question.
    second_id = newest_row_id(args.model)
    if second_id == ext_id:
        second_id = None  # nothing new appeared; the worker took nothing
    second_row = extraction(p, args.model, second_id) if second_id else {}

    print("\n=== ACCEPTANCE ===")
    ok = True

    delta = (final[2] or 0) - (samples_at_cancel or 0)
    c1 = delta <= 40
    print(f"1. STOPS AT THE NEXT BOUNDARY : {'PASS' if c1 else 'FAIL'}  "
          f"({samples_at_cancel} -> {final[2]}, +{delta} samples)")
    ok &= c1

    c2 = after <= baseline + 0.6
    print(f"2. VRAM FREED                 : {'PASS' if c2 else 'FAIL'}  "
          f"(baseline {baseline}, peak {peak}, after {after} GB)")
    ok &= c2

    # THE INVARIANT: once cancelled, never anything else.
    tail = []
    for o in observations:
        if o[1].lower().endswith("cancelled"):
            tail = observations[observations.index(o):]
            break
    c3 = bool(tail) and all(o[1].lower().endswith("cancelled") for o in tail)
    print(f"3. ROW STAYS CANCELLED        : {'PASS' if c3 else 'FAIL'}  "
          f"(final={final[1]}, {len(tail)} observations after it first read "
          f"cancelled)")
    ok &= c3

    c4 = bool(second_id) and str(
        second_row.get("status", "")
    ).lower() not in ("", "queued")
    print(f"4. WORKER TAKES THE NEXT JOB  : {'PASS' if c4 else 'FAIL'}  "
          f"(second job status={second_row.get('status')})")
    ok &= c4

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print(json.dumps({
        "extraction": ext_id, "second": second_id,
        "baseline_gb": baseline, "peak_gb": peak, "after_gb": after,
        "samples_at_cancel": samples_at_cancel, "final": final,
        "observations": observations,
    }, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
