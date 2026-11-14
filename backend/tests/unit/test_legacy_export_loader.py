"""The export loader refuses every malformed export loudly (042 FR-042.6; FTID §3.13; FTASKS 6.2).

Each case starts from the COMMITTED export (so a fixture cannot agree with the loader by construction),
changes one record or header key, and requires `ExportError` naming the rule. The positive cases load
both committed variants. Controls K-6.13b and K-6.15 mutate this loader and require these tests red.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.support import legacy_export as le

FULL, PUBLIC = le.paths()


def _base() -> Path:
    # The full variant in the private repo; on the mirror only the public one exists.
    return FULL if FULL.exists() else PUBLIC


def _records(path: Path) -> tuple[str, str, list[dict], str]:
    text = path.read_text(encoding="utf-8")
    head, _, rest = text.partition("```jsonl\n")
    body, _, tail = rest.partition("```")
    return head, "```" + tail, [json.loads(line) for line in body.splitlines() if line.strip()], path.name


def _write(tmp_path: Path, name: str, head: str, tail: str, records: list[dict]) -> Path:
    out = tmp_path / name
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    out.write_text(head + "```jsonl\n" + body + tail, encoding="utf-8")
    return out


def _mutated(tmp_path: Path, mutate) -> Path:
    head, tail, records, name = _records(_base())
    mutate(records)
    return _write(tmp_path, name, head, tail, records)


def _first(records: list[dict], section: str, pred=lambda r: True) -> dict:
    return next(r for r in records if r.get("section") == section and pred(r))


def _refused(path: Path, *words: str) -> str:
    with pytest.raises(le.ExportError) as info:
        le.load(path=path)
    message = str(info.value)
    for w in words:
        assert w in message, f"refusal does not name {w!r}: {message}"
    return message


def test_refuses_format_2(tmp_path):
    path = _mutated(tmp_path, lambda rs: rs[0].update(format=2))
    _refused(path, "format must be 1", "2")


def test_refuses_unknown_section(tmp_path):
    path = _mutated(tmp_path, lambda rs: rs.append({"section": "alias", "old": "a", "new": "b"}))
    _refused(path, "unknown section", "alias")


def test_refuses_unknown_flag(tmp_path):
    path = _mutated(tmp_path, lambda rs: _first(rs, "entry")["flags"].append("bogus"))
    _refused(path, "unknown flag", "bogus")


@pytest.mark.parametrize("section,status", [("entry", "transition"), ("env", "frozen"), ("host", "scheduled"),
                                            ("repo", "frozen")])
def test_refuses_status_outside_set(tmp_path, section, status):
    path = _mutated(tmp_path, lambda rs: _first(rs, section).update(status=status))
    _refused(path, "status must be", status)


@pytest.mark.parametrize("change,word", [
    ({"effect": "warn"}, "effect"),
    ({"from_phase": "P9"}, "from_phase"),
    ({"until_phase": "p5"}, "until_phase"),
])
def test_refuses_bad_guard_effect_and_phase(tmp_path, change, word):
    path = _mutated(tmp_path, lambda rs: _first(rs, "guard").update(change))
    _refused(path, word)


@pytest.mark.parametrize("header_change", [{"phase": "P9"}, {"closed": "done"}, {"phase": "P2", "closed": None}])
def test_refuses_bad_header_phase(tmp_path, header_change):
    path = _mutated(tmp_path, lambda rs: rs[0].update(header_change))
    _refused(path, "phase" if "phase" in header_change else "closed")


@pytest.mark.parametrize("flags", ["g", "gi", "m", "u"])
def test_refuses_js_flags_g(tmp_path, flags):
    path = _mutated(tmp_path, lambda rs: _first(rs, "entry").update(js_flags=flags))
    _refused(path, "js_flags", flags)


@pytest.mark.parametrize("bad", [r"(?i)mistudio", r"(?i:mistudio)", r"x(?s)y"])
def test_refuses_inline_flag_group_accepts_lookaround(tmp_path, bad):
    path = _mutated(tmp_path, lambda rs: _first(rs, "guard").update(pattern_js=bad))
    _refused(path, "inline flag group")
    # Lookarounds and non-capturing groups have the same meaning in JavaScript and must load.
    ok = r"(?<![\w.])mi(?:Studio|LLM)(?=[^\w])(?!x)"
    good = _mutated(tmp_path, lambda rs: _first(rs, "guard").update(pattern=ok, pattern_js=ok))
    assert le.load(path=good).guards()


@pytest.mark.parametrize("construct", ["\\A", "\\Z", "(?P<n>x)", "(?#note)"])
def test_refuses_A_Z_P_hash_constructs(tmp_path, construct):
    path = _mutated(tmp_path, lambda rs: _first(rs, "entry").update(pattern="mistudio" + construct))
    _refused(path, "no identical JavaScript meaning")


def test_requires_closed_and_observed(tmp_path):
    path = _mutated(tmp_path, lambda rs: rs[0].pop("closed"))
    _refused(path, "closed")
    path = _mutated(tmp_path, lambda rs: _first(rs, "host").pop("observed"))
    _refused(path, "boolean `observed`", "CF-24")
    path = _mutated(tmp_path, lambda rs: _first(rs, "host").update(observed="yes"))
    _refused(path, "boolean `observed`")


@pytest.mark.parametrize("section", ["entry", "env", "host", "repo", "guard"])
def test_refuses_locations_key(tmp_path, section):
    path = _mutated(tmp_path, lambda rs: _first(rs, section).update(locations=["backend/src/x.py:1"]))
    _refused(path, "locations", "CF-1")


def test_literal_must_match_pattern(tmp_path):
    def plant(rs):
        e = _first(rs, "entry", lambda r: r["literals"])
        e["literals"].append("zz-not-matched-zz")
    _refused(_mutated(tmp_path, plant), "a literal its own pattern misses", "zz-not-matched-zz")


def test_accepts_repo_and_guard_sections():
    export = le.load()
    assert export.repos(), "the export lists no repo record"
    assert export.guards(), "the export lists no guard record"
    assert export.header["repo"] == "miWorkbench"
    assert {g["effect"] for g in export.guards()} <= {"allow", "deny"}
    # Activity keys to `phase`: G-STUDIO is active throughout; the P5 allow-guards (the suite is named
    # miStudio from P5) are active exactly when the committed export's phase is P5 or later.
    active = {g["id"] for g in export.active_guards()}
    phase = int(export.header["phase"].lstrip("P"))
    assert "G-STUDIO" in active
    assert ("G-SUITE-1" in active) == (phase >= 5), (export.header["phase"], sorted(active))


def test_reads_public_and_skips_withheld_by_name(tmp_path):
    if not PUBLIC.exists():
        pytest.fail(f"{PUBLIC} is missing: the mirror keeps only that file")
    public = le.load(path=PUBLIC)
    assert public.is_public and public.withheld
    # Every withheld record is ABSENT from the public variant, named by id or by its stand-in.
    present_ids = {r.get("id") for r in public.records}
    for name in public.withheld:
        if ":sha256:" in name:
            assert le.STAND_IN.match(name), name
        else:
            assert name not in present_ids, name
    # A stand-in is the first 16 hex digits of the sha256 of the old name (CF-2).
    if FULL.exists():
        full = le.load(path=FULL)
        hashed = {n for n in public.withheld if ":sha256:" in n}
        # Every hashed name is the loader's stand-in of exactly one full record, and is_withheld sees it.
        derived = {}
        for r in [*full.hosts(), *full.repos(), *full.env(), *full.history_tools()]:
            derived[le.stand_in(r["section"], r["old"])] = r
            digest = hashlib.sha256(r["old"].encode()).hexdigest()[:16]
            assert le.stand_in(r["section"], r["old"]) == f"{r['section']}:sha256:{digest}"
        assert hashed and hashed <= set(derived), sorted(hashed - set(derived))
        for name in hashed:
            r = derived[name]
            assert public.is_withheld(r["section"], r["old"]), name
            assert r["old"] not in {x.get("old") for x in public.records}, name
        assert full.withheld == (), "the full variant withholds nothing"
    # The ONE skip: withheld records skip by name; present ones do not.
    withheld_entry = next(n for n in public.withheld if ":" not in n)
    with pytest.raises(pytest.skip.Exception, match=withheld_entry):
        le.require(public, "entry", withheld_entry)
    le.require(public, "entry", public.entries()[0]["id"])  # no skip


def test_forbidden_list_unread_only_on_the_public_variant(tmp_path):
    head, tail, records, _ = _records(_base())
    records[0]["forbidden_list"] = "unread"
    records[0]["variant"] = "full"
    full_named = _write(tmp_path, le.FULL_NAME, head, tail, records)
    _refused(full_named, "forbidden_list", "public")
    records[0]["forbidden_list"] = "read"
    records[0]["variant"] = "public"
    records[0]["withheld"] = []
    pub = _write(tmp_path, le.PUBLIC_NAME, head, tail,
                 [r for r in records if "public_withheld" not in r.get("flags", [])])
    _refused(pub, "forbidden_list", "read")


def test_exercise_ids_name_this_repositorys_suites(tmp_path):
    path = _mutated(tmp_path, lambda rs: _first(rs, "entry", lambda r: r["exercise"]).update(
        exercise="pytest:tests/unit/test_x.py::test_y"))
    _refused(path, "exercise must be")
    assert le.parse_exercise("vitest:frontend/src/test/a.test.ts::DR-25 x y") == (
        "vitest", "frontend/src/test/a.test.ts", "DR-25 x y")
