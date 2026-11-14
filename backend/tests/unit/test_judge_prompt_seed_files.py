"""Judge-prompt seeds are DATA FILES, loaded at startup, and no prompt text lives in Python.

Operator request (2026-10-21): adding or changing a behaviour-monitoring topic must take
configuration, never code. The starter prompts are `mistudio.judge-prompt/v1` files in
``src/data/judge_prompt_seeds/``; ``JUDGE_PROMPT_SEED_DIR`` adds more; `judge_prompt_seeds` is a
loader with no prompt text in it.

Negative controls: ``0xcc/reviews/judge_prompt_seed_files_2026-10-21.md``.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path

import pytest
from sqlalchemy import func, select

from src.models.probe_judge_prompt import ProbeJudgePrompt
from src.services import judge_prompt_library as library
from src.services import judge_prompt_seeds as seeds
from src.services.probe_monitor_judge import JUDGE_SYSTEM_PROMPT, prompt_hash

BACKEND = Path(__file__).resolve().parents[2]
SRC = BACKEND / "src"
MIGRATION = BACKEND / "alembic" / "versions" / "d5f8a2c41e7b_probe_judge_prompt_library.py"

#: PRODUCTION values, read 2026-10-21 from GET /api/v1/probe-monitors/judge-prompts. Both versions
#: are locked there by real judge runs; a different hash here would orphan those runs.
STAKES_HASH = "31dde28dcd8b75d216d65a47e6f102715f4ddf0a7ba9cd7a04d03a229a46023c"
PII_HASH = "10d20d2dd018ada3334d3d354c12143af671a54bf9ec60a093df04079a62fb6a"

BUILTIN = [(seeds.BUILTIN_SEED_DIR, True)]


def _doc(name: str, version: int = 1) -> dict:
    return seeds.builtin_document(name, version)


def _write(folder: Path, filename: str, document) -> Path:
    path = folder / filename
    path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
    return path


async def _count(db) -> int:
    return (await db.execute(select(func.count()).select_from(ProbeJudgePrompt))).scalar_one()


async def _row(db, name: str, version: int = 1):
    return (
        await db.execute(
            select(ProbeJudgePrompt).where(
                ProbeJudgePrompt.name == name, ProbeJudgePrompt.version == version
            )
        )
    ).scalar_one_or_none()


# ── the built-in files are the production instruments, byte for byte ─────────────────────────


class TestTheBuiltInFiles:
    def test_there_are_exactly_the_two_starter_files_in_export_format(self):
        files = seeds.seed_files(seeds.BUILTIN_SEED_DIR)
        assert [p.name for p in files] == ["pii-presence.v1.json", "stakes-rating.v1.json"]
        for path in files:
            document = seeds.read_seed_file(path)
            assert document["format"] == library.EXPORT_FORMAT
            # The SAME shape Export writes: an exported row and the file have the same keys.
            assert set(document) == set(library.export_document(ProbeJudgePrompt(**{
                k: v for k, v in document.items() if k != "format"}))), path.name

    def test_stakes_rating_v1_hashes_exactly_as_every_earlier_run_recorded(self):
        parsed = library.parse_document(_doc("stakes-rating"))
        assert parsed.content_hash == STAKES_HASH
        assert prompt_hash() == STAKES_HASH
        assert JUDGE_SYSTEM_PROMPT == parsed.content["system_prompt"], (
            "probe_monitor_judge must read the stakes text from the seed file, not hold a copy"
        )

    def test_pii_presence_v1_hashes_exactly_as_production_stores_it(self):
        parsed = library.parse_document(_doc("pii-presence"))
        assert parsed.content_hash == PII_HASH
        assert parsed.behaviour == "pii" and parsed.content["answer_form"]["kind"] == "binary"

    def test_each_file_declares_the_hash_its_text_produces(self):
        for path in seeds.seed_files(seeds.BUILTIN_SEED_DIR):
            document = seeds.read_seed_file(path)
            assert document["content_hash"] == library.parse_document(document).content_hash

    def test_the_seed_ids_are_the_ids_the_migration_wrote(self):
        assert seeds.seed_id("stakes-rating", 1) == "jp_stakes_rating_v1"
        assert seeds.seed_id("pii-presence", 1) == "jp_pii_presence_v1"
        long_name = "a" * 64
        assert len(seeds.seed_id(long_name, 12)) <= 36
        assert seeds.seed_id(long_name, 12) != seeds.seed_id(long_name, 13)


# ── the loader ────────────────────────────────────────────────────────────────────────────────


class TestTheLoader:
    async def test_it_creates_the_missing_seeds_with_their_fixed_ids(self, async_session):
        assert await _count(async_session) == 0
        report = await seeds.ensure_seeds(async_session, BUILTIN)
        assert sorted(report.created) == ["pii-presence/v1", "stakes-rating/v1"]
        assert report.refused == [] and report.invalid == []
        stakes = await _row(async_session, "stakes-rating")
        pii = await _row(async_session, "pii-presence")
        assert (stakes.id, stakes.content_hash, stakes.source) == ("jp_stakes_rating_v1", STAKES_HASH, "seed")
        assert (pii.id, pii.content_hash) == ("jp_pii_presence_v1", PII_HASH)
        assert stakes.system_prompt == JUDGE_SYSTEM_PROMPT

    async def test_a_second_pass_is_a_no_op(self, async_session):
        await seeds.ensure_seeds(async_session, BUILTIN)
        before = await _count(async_session)
        report = await seeds.ensure_seeds(async_session, BUILTIN)
        assert report.created == [] and report.refused == [] and report.invalid == []
        assert sorted(report.unchanged) == ["pii-presence/v1", "stakes-rating/v1"]
        assert await _count(async_session) == before == 2

    async def test_a_CHANGED_file_for_a_stored_version_is_refused_and_the_row_kept(
        self, async_session, tmp_path, caplog
    ):
        await seeds.ensure_seeds(async_session, BUILTIN)
        changed = dict(_doc("pii-presence"))
        changed["system_prompt"] += " Also count nicknames."
        changed.pop("content_hash")
        path = _write(tmp_path, "pii-edited.json", changed)

        with caplog.at_level(logging.WARNING, logger=seeds.__name__):
            report = await seeds.ensure_seeds(async_session, [(tmp_path, False)])

        assert report.created == [] and len(report.refused) == 1
        refused = report.refused[0]
        assert refused["file"] == str(path) and refused["prompt"] == "pii-presence/v1"
        assert refused["existing_hash"] == PII_HASH and refused["file_hash"] != PII_HASH
        warning = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert str(path) in warning and PII_HASH in warning
        row = await _row(async_session, "pii-presence")
        await async_session.refresh(row)
        assert row.content_hash == PII_HASH and "nicknames" not in row.system_prompt

    async def test_it_reads_the_extra_folder(self, async_session, tmp_path):
        document = {
            "format": "mistudio.judge-prompt/v1",
            "name": "refusal-presence",
            "version": 1,
            "title": "Refusal present",
            "behaviour": "refusal",
            "system_prompt": 'Did the assistant refuse? Reply {"answer": "yes" or "no"}.',
            "answer_form": {"kind": "binary"},
        }
        _write(tmp_path, "refusal.json", document)
        _write(tmp_path, "README.txt", "not a prompt")

        report = await seeds.ensure_seeds(async_session, [*BUILTIN, (tmp_path, False)])

        assert "refusal-presence/v1" in report.created and report.invalid == []
        row = await _row(async_session, "refusal-presence")
        assert row.source == "seed" and row.source_ref == {"seed_file": "refusal.json"}
        assert row.user_template == "{interaction}"  # New prompt's default, as documented
        assert row.content_hash == library.parse_document(document).content_hash

    async def test_a_malformed_file_is_skipped_by_name_and_the_rest_still_load(
        self, async_session, tmp_path, caplog
    ):
        broken = _write(tmp_path, "a-broken.json", '{"format": "mistudio.judge-prompt/v1", ')
        bad_form = dict(_doc("pii-presence"), name="pii-other", answer_form={"kind": "dice"})
        bad_form.pop("content_hash")
        bad = _write(tmp_path, "b-bad-form.json", bad_form)
        listed = _write(tmp_path, "c-list.json", "[]")
        good = dict(_doc("pii-presence"), name="pii-copy")
        good.pop("content_hash")
        _write(tmp_path, "d-good.json", good)

        with caplog.at_level(logging.ERROR, logger=seeds.__name__):
            report = await seeds.ensure_seeds(async_session, [(tmp_path, False)])

        assert report.created == ["pii-copy/v1"]
        errors = {entry["file"]: entry["error"] for entry in report.invalid}
        assert set(errors) == {str(broken), str(bad), str(listed)}
        assert "not valid JSON" in errors[str(broken)]
        assert "invalid_answer_form" in errors[str(bad)]
        assert "one JSON object" in errors[str(listed)]
        logged = " ".join(r.getMessage() for r in caplog.records)
        for path in (broken, bad, listed):
            assert str(path) in logged

    async def test_a_missing_extra_folder_is_reported_not_fatal(self, async_session, tmp_path):
        report = await seeds.ensure_seeds(async_session, [*BUILTIN, (tmp_path / "nope", False)])
        assert len(report.created) == 2
        assert report.invalid == [
            {"file": str(tmp_path / "nope"), "error": "is not a folder (JUDGE_PROMPT_SEED_DIR)"}
        ]

    async def test_losing_an_insert_race_compares_against_the_winner(
        self, async_session, monkeypatch
    ):
        """Two replicas start together: both see the key absent, one commits first. The other's
        insert hits the unique constraint and must re-read and compare — not crash startup."""
        await seeds.ensure_seeds(async_session, BUILTIN)
        real = seeds._stored
        calls = {"n": 0}

        async def stale_first(db, name, version):
            calls["n"] += 1
            return None if calls["n"] == 1 else await real(db, name, version)

        monkeypatch.setattr(seeds, "_stored", stale_first)
        report = seeds.SeedReport()
        await seeds._load_one(
            async_session, seeds.BUILTIN_SEED_DIR / "stakes-rating.v1.json", True, report
        )
        assert calls["n"] == 2, "the loser must re-read the winner's row"
        assert report.unchanged == ["stakes-rating/v1"] and report.created == []
        assert await _count(async_session) == 2

    def test_the_extra_folder_comes_from_JUDGE_PROMPT_SEED_DIR(self, monkeypatch, tmp_path):
        from src.core.config import Settings, settings

        monkeypatch.setattr(settings, "judge_prompt_seed_dir", tmp_path)
        assert seeds.seed_directories() == [(seeds.BUILTIN_SEED_DIR, True), (tmp_path, False)]
        monkeypatch.setattr(settings, "judge_prompt_seed_dir", None)
        assert seeds.seed_directories() == [(seeds.BUILTIN_SEED_DIR, True)]
        monkeypatch.setenv("JUDGE_PROMPT_SEED_DIR", str(tmp_path))
        assert Settings().judge_prompt_seed_dir == tmp_path
        monkeypatch.setenv("JUDGE_PROMPT_SEED_DIR", "")
        assert Settings().judge_prompt_seed_dir is None, "blank must not mean the working directory"


# ── wiring: startup loads them, and a run with no prompt gets the default ─────────────────────


async def test_the_API_STARTUP_loads_the_seed_files(async_engine, monkeypatch, tmp_path):
    """The real lifespan, against an empty library. Removing the `ensure_seeds` call in
    `main.lifespan` makes this red (control S-W1)."""
    from src import main
    from src.core.config import settings
    from src.core.database import AsyncSessionLocal

    class Inert:
        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(main, "get_background_monitor", lambda: Inert())
    extra = dict(_doc("pii-presence"), name="pii-from-configmap")
    extra.pop("content_hash")
    _write(tmp_path, "extra.json", extra)
    monkeypatch.setattr(settings, "judge_prompt_seed_dir", tmp_path)

    async with AsyncSessionLocal() as db:
        assert await _count(db) == 0
    async with main.lifespan(main.app):
        pass
    async with AsyncSessionLocal() as db:
        rows = {(r.name, r.version): r.content_hash for r in (await db.execute(select(ProbeJudgePrompt))).scalars()}
    assert rows[("stakes-rating", 1)] == STAKES_HASH
    assert rows[("pii-presence", 1)] == PII_HASH
    assert ("pii-from-configmap", 1) in rows, "JUDGE_PROMPT_SEED_DIR was not read at startup"
    assert len(rows) == 3


async def test_a_run_naming_no_prompt_gets_stakes_rating_v1_even_on_an_empty_library(async_session):
    row = await library.resolve_for_run(async_session, None)
    assert (row.name, row.version, row.content_hash) == ("stakes-rating", 1, STAKES_HASH)


# ── the guard: no judge-prompt instruction text in any Python module ─────────────────────────


def _needles() -> dict[str, list[str]]:
    """Distinctive sentences of every built-in prompt, FROM THE DATA FILES (the authority)."""
    out = {}
    for path in seeds.seed_files(seeds.BUILTIN_SEED_DIR):
        text = " ".join(seeds.read_seed_file(path)["system_prompt"].split())
        pieces = [p.strip() for p in re.split(r"(?<=[.:;])\s+", text)]
        out[path.name] = [p for p in pieces if len(p) >= 30]
    return out


def _string_literals(source: str):
    """Every string literal as Python sees it — implicit concatenation across lines already
    joined by the parser, and the literal parts of f-strings."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, " ".join(node.value.split())
        elif isinstance(node, ast.JoinedStr):
            literal = "".join(
                v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            yield node.lineno, " ".join(literal.split())


def _offenders(source: str, needles: dict[str, list[str]]) -> list[tuple[int, str, str]]:
    hits = []
    for lineno, literal in _string_literals(source):
        for seed_file, sentences in needles.items():
            for sentence in sentences:
                if sentence in literal:
                    hits.append((lineno, seed_file, sentence[:60]))
    return hits


class TestNoPromptTextInCode:
    def test_the_needles_are_real(self):
        needles = _needles()
        assert set(needles) == {"pii-presence.v1.json", "stakes-rating.v1.json"}
        for seed_file, sentences in needles.items():
            assert len(sentences) >= 4, (seed_file, sentences)

    def test_the_scanner_FINDS_prompt_text_laid_out_as_code_used_to_hold_it(self):
        """Not fail-open: the migration legitimately carries frozen copies, written as split,
        implicitly concatenated literals across lines — exactly how the seeds module did. The
        scanner must flag every seed there."""
        hits = _offenders(MIGRATION.read_text(encoding="utf-8"), _needles())
        assert {seed_file for _, seed_file, _ in hits} == {"pii-presence.v1.json", "stakes-rating.v1.json"}

    def test_no_module_under_src_holds_judge_prompt_instructions(self):
        needles = _needles()
        modules = sorted(SRC.rglob("*.py"))
        assert len(modules) > 200, "the scan did not reach the source tree"
        offenders = []
        for module in modules:
            for lineno, seed_file, sentence in _offenders(module.read_text(encoding="utf-8"), needles):
                offenders.append(f"{module.relative_to(BACKEND)}:{lineno} holds {seed_file}: {sentence!r}")
        assert offenders == [], (
            "judge-prompt text belongs in a seed FILE (src/data/judge_prompt_seeds/), never in "
            "code:\n" + "\n".join(offenders)
        )

    def test_the_judge_modules_bind_no_instruction_literal(self):
        """Beyond the two known texts: in the judge-prompt modules, no `system_prompt=` or
        `instructions=` keyword and no `*_PROMPT` name is bound to a string literal. A new
        behaviour's prompt written into code would land here first."""
        judge_modules = [
            *sorted((SRC / "services").glob("judge_prompt*.py")),
            SRC / "services" / "probe_monitor_judge.py",
            SRC / "api" / "v1" / "endpoints" / "probe_judge_prompts.py",
            SRC / "mcp_server" / "tools" / "judge_prompts.py",
        ]
        assert all(m.is_file() for m in judge_modules), judge_modules
        offenders = []
        for module in judge_modules:
            for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
                if isinstance(node, ast.keyword) and node.arg in ("system_prompt", "instructions"):
                    if isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                        offenders.append(f"{module.name}:{node.value.lineno} {node.arg}=<literal>")
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    if any(n.endswith("_PROMPT") for n in names) and isinstance(
                        node.value, (ast.Constant, ast.JoinedStr)
                    ):
                        offenders.append(f"{module.name}:{node.lineno} {names} = <literal>")
        assert offenders == [], offenders
