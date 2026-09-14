"""Smoke-test a serving model against a labeling template BEFORE running a job.

Written after five models were evaluated for labeling a 30k-feature SAE
extraction. Four were unusable, each in a different way that a casual look would
have missed, and two of them looked fine until this script was made stricter:

  LFM2.5-1.2B-Instruct   copied the category ENUM verbatim instead of choosing
  granite-4.0-1b         returned sentences from the system prompt as the answer
  LFM2.5-2.6B            a pure reasoning model; never reached the JSON
  gemma-3-12b-it         schema-perfect, and confidently mislabelled incoherent
                         features -- 0 refusals in 8, `confidence: high` on all

Schema-valid output is NOT the bar. The checks below encode what actually
separated a usable model from an unusable one.

Usage (from the repo root):
  ssh -o BatchMode=yes sean@192.168.244.61 \
    "kubectl exec -i -n mistudio $(POD) -c backend -- python - <MODEL>" < smoke.py

Checks the three things that actually decide whether a 30k-feature run works:
  1. does it emit parseable JSON at all
  2. does it emit the fields the template asks for
  3. how long does one call take -> ETA for the full extraction
"""
import json, sys, time, urllib.request
from sqlalchemy import text
from src.core.database import SyncSessionLocal
from src.services.openai_labeling_service import OpenAILabelingService

MODEL = sys.argv[1] if len(sys.argv) > 1 else "LFM2.5-2.6B"
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else None
TEMPLATE = "lpt_95fb74cb61354eb5"
EXTRACTION = "extr_20260828_080834_sae_sae_39cc_002"
N = 3

t = json.load(urllib.request.urlopen(
    f"http://localhost:8000/api/v1/labeling-prompt-templates/{TEMPLATE}", timeout=30))
db = SyncSessionLocal()
feats = [r[0] for r in db.execute(text(
    "SELECT id FROM features WHERE extraction_job_id=:e ORDER BY neuron_index LIMIT :n OFFSET 200"),
    {"e": EXTRACTION, "n": N})]
svc = OpenAILabelingService(api_key="x",
        base_url="http://millm-backend.millm.svc.cluster.local:8000/v1",
        model=MODEL, temperature=t["temperature"], max_tokens=(MAXTOK or t["max_tokens"]), top_p=t["top_p"])
cfg = {k: t[k] for k in ("template_type","max_examples","include_prefix","include_suffix",
                         "prime_token_marker","include_logit_effects",
                         "top_promoted_tokens_count","top_suppressed_tokens_count")}

print(f"model: {MODEL}   max_tokens: {MAXTOK or t['max_tokens']}")
print("out | secs | finish | JSON | label")
ok_n, lat = 0, []
for fid in feats:
    rows = [dict(r._mapping) for r in db.execute(text(
        "SELECT prefix_tokens, prime_token, suffix_tokens, max_activation FROM feature_activations "
        "WHERE feature_id=:f ORDER BY max_activation DESC LIMIT 10"), {"f": fid})]
    u = svc._build_user_prompt(examples=rows, template_config=cfg,
                               user_prompt_template=t["user_prompt_template"], feature_id=fid)
    body = {"model": MODEL, "max_tokens": (MAXTOK or t["max_tokens"]), "temperature": t["temperature"],
            "messages": [{"role":"system","content":t["system_message"]},
                         {"role":"user","content":u}]}
    s = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://millm-backend.millm.svc.cluster.local:8000/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=300))
    except Exception as e:
        print(f"    - |    - | ERROR  | no   | {type(e).__name__}: model not loaded?")
        continue
    dt = time.time()-s; lat.append(dt)
    import re
    ch = r["choices"][0]; txt = ch["message"]["content"].strip()
    # Mirror the production think-stripping, including the no-opening-tag shape
    # a chat template can produce.
    txt = re.sub(r"<think>.*?</think>\s*", "", txt, flags=re.DOTALL).strip()
    if "</think>" in txt and "<think>" not in txt:
        txt = txt.rsplit("</think>", 1)[1].strip()
    # Production strips markdown fences too; mirror it or a fenced-but-valid
    # reply is scored as a failure.
    txt = re.sub(r"^```(?:json)?\s*", "", txt).strip()
    txt = re.sub(r"\s*```$", "", txt).strip()
    m = re.search(r'\{.*?"specific".*?\}', txt, re.S)
    good, shown = "no", txt[:52].replace("\n", " ")
    if m:
        try:
            j = json.loads(m.group(0))
            faults = []
            missing = {"specific","category","description","distinguisher",
                       "fit_count","confidence"} - set(j)
            if missing:
                faults.append(f"missing:{','.join(sorted(missing))}")
            # A weak model COPIES the enum instead of choosing from it. This is
            # the failure that made an earlier version of this test report
            # "parsed 3/3" for output that was unusable.
            cat = str(j.get("category", ""))
            if "|" in cat:
                faults.append("copied-enum")
            elif cat not in ("semantic","structural","language","uninterpretable"):
                faults.append(f"bad-category:{cat[:20]}")
            extra = set(j) - {"specific","category","description","distinguisher",
                              "fit_count","confidence"}
            if extra:
                faults.append(f"invented:{','.join(sorted(extra))}")
            # A small model regurgitates the INSTRUCTIONS as the answer. Seen
            # live: description "A label that fits many features looks like
            # knowledge and carries none." and specific "a mid-word fragment" —
            # both phrases lifted from the system prompt. Schema-valid, useless.
            # A CORRECT refusal legitimately puts "uninterpretable" in
            # `specific`, which is both one word and a word from the prompt.
            # Penalising that punishes the one behaviour the template most wants.
            refusing = str(j.get("category","")).strip().lower() == "uninterpretable"
            sys_l = t["system_message"].lower()
            for fld in (() if refusing else ("specific", "description", "distinguisher")):
                val = str(j.get(fld, "")).strip().lower().replace("_", " ")
                if len(val) > 12 and val.rstrip(".") in sys_l:
                    faults.append(f"echoes-prompt:{fld}")
                    break
            # The template asks for 2-5 words. A one-word label is a token echo.
            words = str(j.get("specific", "")).replace("_", " ").split()
            if len(words) < 2 and not refusing:
                faults.append("one-word-label")
            # Latching onto example 1's prime token instead of synthesising.
            if rows and str(j.get("specific","")).strip().lower() == \
                    str(rows[0].get("prime_token","")).strip().strip("\u2581").lower():
                faults.append("echoes-example-1")
            good = "yes" if not faults else "; ".join(faults)[:34]
            shown = f'{j.get("specific")} [{cat[:22]}]'
            if not faults:
                ok_n += 1
        except Exception:
            pass
    print(f'{r["usage"]["completion_tokens"]:3d} | {dt:4.1f} | '
          f'{str(ch.get("finish_reason")):6s} | {good:4s} | {shown}')
db.close()

print()
if ok_n == 0:
    print("VERDICT: UNUSABLE. No reply was both parseable AND schema-correct.")
    print("  A model that returns JSON is not enough: it must CHOOSE from the")
    print("  category enum rather than copy it, emit all seven fields, invent")
    print("  none, and synthesise across the ten examples rather than echoing")
    print("  the first one's prime token. Try a larger model.")
elif lat:
    mean = sum(lat)/len(lat)
    print(f"parsed {ok_n}/{N}   mean latency {mean:.1f}s")
    print(f"ETA for 30,712 features at batch_size 10: ~{30712*mean/10/3600:.1f} h")
    print("(granite-4.1-8b previously did 32,601 features in 24h)")
