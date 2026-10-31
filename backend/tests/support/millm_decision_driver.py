"""Run decision-precedence cases through miLLM's REAL arming and serving path.

The probe is armed from a definition's `decision` block through the parsers `probe_arming` uses
(`window_thresholds_from_definition`, `length_bands_from_definition`, `window_length_bands_from_definition`),
then judged by `ProbeRequestContext._verdict_for`. ⚠ Arming is load-bearing (review R1, 040 S1-S2): handing
the case's raw tables to `ArmedProbe` skipped miLLM's whole-or-nothing table rule, and two cases then
"agreed" with miLLM at bars it never serves.

Executed by miStudio's test in miLLM's own venv, cwd = the miLLM checkout, as a subprocess: nothing of
miLLM is imported into miStudio. Reads the case file path from argv[1]; prints {case name: threshold}.
"""
import json
import sys

import torch

from millm.ml.probe_head import ProbeHead
from millm.services.probe_arming import (
    length_bands_from_definition,
    window_length_bands_from_definition,
    window_thresholds_from_definition,
)
from millm.services.probe_runtime import ArmedProbe, ProbeRequestContext

D = 2


def run_case(case):
    probe = case["probe"]
    window = case["window"] if case["window"] is not None else probe["scope"]
    entry = case["window_entry"]
    definition = {"decision": {"threshold": probe["threshold"], "length_bands": probe["length_bands"],
                               "windows": {window: entry} if entry is not None else {}}}
    armed = ArmedProbe(
        probe_id="p", name="p", head=ProbeHead(weight=torch.ones(D), bias=0.0, layer=1),
        rule="mean", scope=probe["scope"], layer=1, rung=2, rung_language="x",
        threshold=probe["threshold"], windows=(window,),
        window_thresholds=window_thresholds_from_definition(definition),
        length_bands=length_bands_from_definition(definition),
        window_length_bands=window_length_bands_from_definition(definition),
    )
    n = int(case["n_scored"])
    # Token layout so that THIS window scores exactly n tokens.
    if window == "all":
        n_prompt, generated = n, 0
    elif window == "prompt":
        n_prompt, generated = n, 1
    elif window == "response":
        n_prompt, generated = 1, n
    else:
        raise SystemExit(f"case {case['name']}: window {window!r} has no token layout here")
    ctx = ProbeRequestContext("r", [armed])
    ctx.set_prompt_length(n_prompt)
    ctx.observe(1, torch.ones((1, n_prompt, D)))
    for _ in range(generated):
        ctx.observe(1, torch.ones((1, 1, D)))
    [verdict] = [v for v in ctx.finish() if v.window == window]
    if not verdict.scored or verdict.n_scored_tokens != n:
        raise SystemExit(f"case {case['name']}: not scored as {n} tokens ({verdict})")
    return verdict.threshold


cases = json.loads(open(sys.argv[1]).read())["cases"]
print("DECISIONS " + json.dumps({case["name"]: run_case(case) for case in cases}))
