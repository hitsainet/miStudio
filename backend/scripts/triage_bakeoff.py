"""Can a small model do TRIAGE, and is it self-consistent?

Triage is not labeling. It asks for two cheap things — how many of ten examples
share one pattern, and a rough topic — instead of a discriminative snake_case
label. LFM2.5-1.2B-Instruct failed the LABELING bar (it copied the category enum
verbatim, see smoke_test_labeling_model.py) but has never been measured on this
much lower bar.

Three things are measured, and the third is the one that matters most:

1. PARSEABILITY  - does it emit a usable JSON object at all?
2. SEPARATION    - does its `fit` distinguish features gemma-4-12B refused from
                   ones gemma-4-12B labeled? (checkable, since those verdicts
                   already exist for this extraction)
3. SELF-AGREEMENT- run identically N times: does it give the same answer?

Self-agreement needs NO ground truth, which matters because the ground truth is
itself in question: no measurable property of these features predicts gemma-4's
refusals (best AUC 0.595 across every column in the table), so gemma-4's verdict
may be a noisy target rather than a reliable one. A model that cannot agree with
ITSELF cannot be used to rank 23,000 features whatever it agrees with.

Usage, inside the mistudio backend pod (it needs the DB and reachability to
miLLM):
    env PYTHONPATH=/app python3 /tmp/triage_bakeoff.py <model-name> [runs]
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import httpx
from sqlalchemy import text

from src.core.database import SyncSessionLocal

EXTRACTION = "extr_20260828_080834_sae_sae_39cc_002"
ENDPOINT = os.environ.get(
    "MILLM_ENDPOINT",
    "http://millm-backend.millm.svc.cluster.local:8000/v1",
)
N_PER_CLASS = int(os.environ.get('N_PER_CLASS', '5'))
N_EXAMPLES = 10

# The schema example uses CONCRETE plausible values, not descriptions of what
# to write. The first version showed {"topic": "two or three words"} and the
# model returned exactly that string — an instruction sitting where an answer
# goes is an invitation to copy it, and this model's documented failure mode is
# copying the prompt. Words that appear in the instructions ("pattern",
# "examples") are likewise avoided as bait.
SYSTEM = (
    "Judge one sparse-autoencoder feature.\n"
    "Reply with a JSON object containing exactly two keys and nothing else:\n"
    "  fit   - an integer from 0 to 10: how many of the ten passages below "
    "share one theme\n"
    "  topic - a lowercase snake_case name for that theme, or the word none "
    "when fit is under 5\n"
    "Do not restate this instruction. No prose, no extra keys."
)


def render(ex):
    def clean(t):
        return str(t).replace("▁", " ").replace("Ġ", " ").replace("##", "")
    pre = "".join(clean(t) for t in (ex.get("prefix_tokens") or [])[-12:])
    suf = "".join(clean(t) for t in (ex.get("suffix_tokens") or [])[:12])
    return f"{pre}<<{clean(ex.get('prime_token') or '')}>>{suf}".strip()


def strip_think(s):
    """LFM2.5 chat templates open <think> in the PROMPT, so the reply may carry
    only the closing tag. Handle both shapes plus a truncated trace."""
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL).strip()
    if "</think>" in s and "<think>" not in s:
        s = s.rsplit("</think>", 1)[1].strip()
    if s.startswith("<think>"):
        return ""
    return s


def parse(raw):
    """Return (fit_int, topic) or (None, None). Tolerant of fences and prose."""
    s = strip_think(raw or "")
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.MULTILINE).strip()
    obj = None
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = json.JSONDecoder().raw_decode(s[i:])
                break
            except json.JSONDecodeError:
                continue
    if not isinstance(obj, dict):
        return None, None
    m = re.search(r"(\d+)", str(obj.get("fit", "")))
    fit = int(m.group(1)) if m else None
    if fit is not None and not (0 <= fit <= 10):
        fit = None
    topic = str(obj.get("topic", "")).strip().lower() or None
    return fit, topic


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "LFM2.5-1.2B-Instruct"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    db = SyncSessionLocal()

    # Deterministic mixed panel: features gemma-4 REFUSED and features it LABELED.
    panel = []
    for cls, cond in (("refused", "category = 'uninterpretable'"),
                      ("labeled", "category <> 'uninterpretable'")):
        rows = db.execute(text(f"""
            SELECT id, neuron_index, name FROM features
            WHERE extraction_job_id = :e AND labeled_at IS NOT NULL AND {cond}
            ORDER BY neuron_index LIMIT :n
        """), {"e": EXTRACTION, "n": N_PER_CLASS}).fetchall()
        panel += [(r[0], r[1], cls, r[2]) for r in rows]

    print(f"model    : {model}")
    print(f"panel    : {len(panel)} features "
          f"({sum(1 for p in panel if p[2] == 'refused')} refused / "
          f"{sum(1 for p in panel if p[2] == 'labeled')} labeled by gemma-4)")
    print(f"runs     : {runs} passes per feature, example ORDER shuffled each pass\n")

    prompts = {}
    for fid, idx, cls, name in panel:
        ex = db.execute(text("""
            SELECT prefix_tokens, prime_token, suffix_tokens FROM feature_activations
            WHERE feature_id = :f ORDER BY max_activation DESC LIMIT :k
        """), {"f": fid, "k": N_EXAMPLES}).fetchall()
        prompts[fid] = [
            render(dict(prefix_tokens=r[0], prime_token=r[1], suffix_tokens=r[2]))
            for r in ex
        ]

    results = defaultdict(list)   # fid -> [(fit, topic, raw_len)]
    unparsed = []
    t0 = time.time()
    with httpx.Client(timeout=180.0) as client:
        for run in range(runs):
            for fid, idx, cls, name in panel:
                try:
                    shuffled = list(prompts[fid])
                    random.Random(f"{fid}:{run}").shuffle(shuffled)
                    body = "\n".join(f"{i+1}. {p}" for i, p in enumerate(shuffled))
                    user = f"Passages:\n{body}\n\nJSON:"
                    r = client.post(f"{ENDPOINT}/chat/completions", json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 200,
                    })
                    raw = ""
                    if r.status_code == 200:
                        raw = (r.json()["choices"][0]["message"]["content"] or "")
                    else:
                        unparsed.append((fid, run, f"HTTP {r.status_code}: {r.text[:120]}"))
                        results[fid].append((None, None, 0))
                        continue
                except Exception as exc:
                    unparsed.append((fid, run, f"{type(exc).__name__}: {exc}"))
                    results[fid].append((None, None, 0))
                    continue
                fit, topic = parse(raw)
                if fit is None:
                    unparsed.append((fid, run, raw[:160].replace("\n", " ")))
                results[fid].append((fit, topic, len(raw)))
            print(f"  run {run+1}/{runs} done ({time.time()-t0:.0f}s)")

    total = len(panel) * runs
    parsed = sum(1 for v in results.values() for f, _, _ in v if f is not None)
    print(f"\n{'='*74}\n1. PARSEABILITY : {parsed}/{total} "
          f"({100*parsed/max(total,1):.0f}%)")
    if unparsed:
        print("   first unparseable replies:")
        for fid, run, snippet in unparsed[:3]:
            print(f"     [{fid[-5:]} run{run}] {snippet!r}")

    print(f"\n2. SEPARATION (does `fit` track gemma-4's verdict?)")
    print(f"   {'feat':<7} {'gemma-4':<9} {'fits':<14} {'topics'}")
    by_cls = defaultdict(list)
    for fid, idx, cls, name in panel:
        fits = [f for f, _, _ in results[fid] if f is not None]
        tops = [t for _, t, _ in results[fid] if t]
        if fits:
            by_cls[cls].append(sum(fits) / len(fits))
        print(f"   {idx:<7} {cls:<9} {str(fits):<14} {list(dict.fromkeys(tops))[:2]}")
    if by_cls["refused"] and by_cls["labeled"]:
        mr = sum(by_cls["refused"]) / len(by_cls["refused"])
        ml = sum(by_cls["labeled"]) / len(by_cls["labeled"])
        print(f"\n   mean fit  refused={mr:.2f}   labeled={ml:.2f}   gap={ml-mr:+.2f}")
        print("   (a useful triage signal needs labeled > refused by a clear margin)")

    print(f"\n3. SELF-AGREEMENT ({runs} passes, shuffled order - tests JUDGEMENT,\n   not decoder determinism; a canned constant would score 100% at temp 0)")
    exact_fit = same_topic = scored = 0
    for fid, idx, cls, name in panel:
        fits = [f for f, _, _ in results[fid] if f is not None]
        tops = [t for _, t, _ in results[fid] if t]
        if len(fits) == runs:
            scored += 1
            if len(set(fits)) == 1:
                exact_fit += 1
        if len(tops) == runs and len(set(tops)) == 1:
            same_topic += 1
    if scored:
        print(f"   identical fit across all runs   : {exact_fit}/{scored} "
              f"({100*exact_fit/scored:.0f}%)")
        print(f"   identical topic across all runs : {same_topic}/{len(panel)} "
              f"({100*same_topic/len(panel):.0f}%)")
    else:
        print("   not enough parsed runs to measure agreement")

    BAIT = {"two or three words", "snake_case name", "activation examples",
            "patterns", "pattern", "passages", "theme", "court_sentencing",
            "one theme", "json"}
    echoed = sorted({t for v in results.values() for _, t, _ in v
                     if t and t in BAIT})
    def _auc(pos, neg):
        if not pos or not neg:
            return None
        wins = 0.0
        for a in pos:
            for b in neg:
                wins += 1.0 if a > b else (0.5 if a == b else 0.0)
        return wins / (len(pos) * len(neg))

    mean_auc = _auc(by_cls["labeled"], by_cls["refused"])
    print(f"\n3b. DOES AVERAGING RESCUE IT? (AUC on the mean of {runs} runs)")
    if mean_auc is None:
        print("   not enough data")
    else:
        print(f"   AUC on mean fit = {mean_auc:.3f}   "
              f"({'USABLE' if mean_auc >= 0.65 else 'no better than chance'})")
        print(f"   n = {len(by_cls['labeled'])} labeled vs "
              f"{len(by_cls['refused'])} refused")

    print(f"\n4. PROMPT ECHO (this model's documented failure mode)")
    print(f"   topics copied from the prompt: {echoed if echoed else 'none'}")

    print(f"\n{'='*74}")
    ok_parse = parsed / max(total, 1) >= 0.90
    ok_agree = scored and exact_fit / scored >= 0.80
    gap = (sum(by_cls['labeled'])/len(by_cls['labeled']) -
           sum(by_cls['refused'])/len(by_cls['refused'])) \
        if by_cls['refused'] and by_cls['labeled'] else 0.0
    print(f"parseable >=90% : {'PASS' if ok_parse else 'FAIL'}")
    print(f"self-agree >=80%: {'PASS' if ok_agree else 'FAIL'}")
    print(f"prompt echo     : {'FAIL - ' + str(echoed) if echoed else 'PASS'}")
    print(f"separation gap  : {gap:+.2f} "
          f"({'usable' if gap >= 1.5 else 'weak — but see note'})")
    print("\nNOTE: a weak gap is NOT decisive on its own. No measurable property "
          "of these\nfeatures predicts gemma-4's refusals (best AUC 0.595), so "
          "gemma-4's verdict may\nbe a noisy target. Self-agreement is the "
          "signal that needs no ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
