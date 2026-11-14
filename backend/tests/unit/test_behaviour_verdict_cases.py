"""`docs/schemas/behaviour-verdict-cases.json` — the served behaviour verdict, pinned by data (ENH-006 G19, AC 2).

miStudio OWNS the file; miLLM vendors it byte-identically and runs every case through its own combiner. Here
every case runs through two independent readings that must both agree with the hand-written expectation:

1. `behaviour_verdict.combine` — miStudio's combiner, built on the blend engine's own `apply` / `fires_of` and
   040's `union_fires` (asserted by spy below, so the cases pin THOSE functions, not a copy);
2. a standard-library checker written from the file's own `rule` text, importing nothing from `src` — a
   consumer reproduces every case from the file alone.

And, once miLLM has vendored it, its copy must be byte-identical (skips loudly without the sibling; fails under
the cross-repo require flag).
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from src.services import behaviour_verdict as bv
from src.services.probe_blend import engine
from tests.support.cross_repo import require_cross_repo
from tests.support.sibling_repos import miserver_repo

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "docs" / "schemas" / "behaviour-verdict-cases.json"
SCHEMA = REPO / "docs" / "schemas" / "behaviour-verdict-v1.json"


def load(path: Path = CASES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fragment(case: dict) -> dict:
    return {"candidate": case["candidate"], "members": case["members"]}


def test_the_file_is_well_formed():
    doc = load()
    assert doc["format"] == bv.CASES_FORMAT
    names = [c["name"] for c in doc["cases"]]
    assert len(names) == len(set(names)) and len(names) >= 30
    assert doc["tolerance"]["value_abs"] == bv.VALUE_TOLERANCE
    # every branch the design names is covered (G19 §The contract)
    by = {c["expect"]["decided_by"] for c in doc["cases"]}
    assert by == set(bv.DECIDED_BY)
    kinds = {c["candidate"]["kind"] for c in doc["cases"]}
    assert kinds == {"any_of", "all_of", "k_of_n", "stack", "max", "best_single"}
    assert any(c["expect"]["fires"] is None for c in doc["cases"])
    assert any(not m["veto_applies"] for c in doc["cases"] if c["candidate"]["rules"] for m in c["members"])


def test_every_expectation_validates_against_the_definitions_verdict_shape():
    schema = json.loads(SCHEMA.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    verdict = {**schema["$defs"]["Verdict"], "$defs": schema["$defs"]}
    for case in load()["cases"]:
        jsonschema.validate(case["expect"], verdict)


@pytest.mark.parametrize("case", load()["cases"], ids=lambda c: c["name"])
def test_mistudios_combiner_reproduces_each_case(case):
    got = bv.combine(fragment(case), case["inputs"]["members"], case["inputs"]["rules"])
    assert bv.agrees(got, case["expect"]), (got, case["expect"])


def test_the_combiner_calls_the_engines_own_functions(monkeypatch):
    """The cases pin the blend engine's `apply` / `fires_of` and 040's `union_fires`, which the 041 report used —
    not a second copy of the arithmetic. Control: a combiner with its own `>=` stops calling `fires_of` → red."""
    calls = {"apply": 0, "fires_of": 0, "union_fires": 0}
    for name in calls:
        real = getattr(engine, name)

        def spy(*a, _real=real, _name=name, **k):
            calls[_name] += 1
            return _real(*a, **k)

        monkeypatch.setattr(engine, name, spy)
    for case in load()["cases"]:
        bv.combine(fragment(case), case["inputs"]["members"], case["inputs"]["rules"])
    rules_cases = sum(1 for c in load()["cases"] if c["candidate"]["rules"])
    assert calls["union_fires"] == rules_cases
    assert calls["apply"] >= 25 and calls["fires_of"] >= 25


# ── a consumer's reading: the standard library and the file's own rule text ─────────────────────


def stdlib_verdict(case: dict) -> dict:
    """Written from `doc["rule"]`, importing nothing from the app."""
    c, members = case["candidate"], case["members"]
    ins, rules = case["inputs"]["members"], case["inputs"]["rules"]
    digests = [m["detector_digest"] for m in members]
    scoped = {m["detector_digest"]: m["veto_applies"] for m in members}
    every = all(scoped.values())
    unknown = c["rules"] and (rules is None or rules["unknown"])
    match = bool(c["rules"] and not unknown and rules["match"])
    veto = bool(c["rules"] and not unknown and rules["veto"] and not match)
    score_kind = c["kind"] in ("stack", "max", "best_single")
    need = "score" if score_kind else "fires"
    missing = [d for d in digests if ins.get(d) is None or ins[d][need] is None]
    fired = [d for d in digests if ins.get(d) is not None and ins[d]["fires"] is True]
    vetoed = [d for d in fired if veto and scoped[d]]
    if not score_kind:
        value = float(len([d for d in fired if every or d not in vetoed]))
        base = value >= c["k"]
    elif missing:
        value, base = None, False
    else:
        p = c["params"]
        z = [(ins[d]["score"] - p["mean"][j]) / p["std"][j] for j, d in enumerate(digests)]
        if c["kind"] == "stack":
            value = 0.0
            for j, zj in enumerate(z):
                value += zj * p["weights"][j]
            value += p["intercept"]
        elif c["kind"] == "max":
            value = max(z)
        else:
            value = z[p["member_index"]]
        base = value >= c["threshold"]
    final = (match or (base and not (veto and every))) if c["rules"] else base
    if final:
        fires, by = True, ("rule_match" if match else "members")
    elif missing:
        fires, by = None, "members_missing"
    elif veto:
        fires, by = False, "vetoed"
    else:
        fires, by = False, "members"
    return {"fires": fires, "decided_by": by, "members_fired": fired, "members_vetoed": vetoed,
            "members_missing": missing, "value": value,
            "rules": ({"match": None if unknown else match, "veto": None if unknown else veto, "unknown": bool(unknown)}
                      if c["rules"] else None)}


@pytest.mark.parametrize("case", load()["cases"], ids=lambda c: c["name"])
def test_a_consumer_reproduces_each_case_from_the_rule_text(case):
    tol = load()["tolerance"]["value_abs"]
    got = stdlib_verdict(case)
    expect = case["expect"]
    assert {k: v for k, v in got.items() if k != "value"} == {k: v for k, v in expect.items() if k != "value"}
    assert (got["value"] is None) == (expect["value"] is None)
    if got["value"] is not None:
        assert abs(got["value"] - expect["value"]) <= tol


# ── the vendored copies ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["behaviour-verdict-cases.json", "behaviour-verdict-v1.json"])
def test_millms_copy_is_byte_identical(name):
    repo = miserver_repo()
    if not (repo / "docs" / "schemas").is_dir():
        if require_cross_repo():
            pytest.fail(f"miLLM is not checked out at {repo}; {name} cannot be compared")
        pytest.skip(f"miLLM sibling absent at {repo}")
    theirs = repo / "docs" / "schemas" / name
    if not theirs.exists():
        if require_cross_repo():
            pytest.fail(f"miLLM has not vendored {name} (push order: miStudio first, then miLLM)")
        pytest.skip(f"miLLM has not vendored {name} yet (push order: miStudio first)")
    assert theirs.read_bytes() == (REPO / "docs" / "schemas" / name).read_bytes()
