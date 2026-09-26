"""OSD-9 — two extractions started in the same second must not share an id.

The id was `extr_<YYYYmmdd_HHMMSS>_sae_<id8>`, second-granular. Two starts inside
one second collided on the primary key AND on the on-disk directory name, because
the directory is derived from the id.
"""
import ast
import re
from pathlib import Path

from src.services.extraction_service import _mint_extraction_id

SERVICE = Path(__file__).resolve().parents[2] / "src" / "services" / "extraction_service.py"


def test_two_ids_minted_in_the_same_second_differ():
    """200 mints, which is why the suffix is 8 hex characters and not 4.

    With 4 (65,536 values) this test had a ~26% chance of failing by the birthday
    bound, and it duly passed alone and failed in a parallel full-suite run — a
    flake inside the fix for a collision bug. At 2^32 the same 200 draws collide
    with probability ~5e-6.
    """
    ids = {_mint_extraction_id(sae_suffix="9a4db34d") for _ in range(200)}
    assert len(ids) == 200, "a same-second collision is a lost extraction directory"


def test_the_shape_stays_parseable_and_sortable():
    minted = _mint_extraction_id(sae_suffix="9a4db34d")
    assert re.fullmatch(r"extr_\d{8}_\d{6}_[0-9a-f]{8}_sae_9a4db34d", minted), minted


def test_the_timestamp_still_leads_so_ids_sort_chronologically():
    """Directory listings and greps rely on the timestamp being the prefix."""
    minted = _mint_extraction_id(sae_suffix="abcd1234")
    assert minted.startswith("extr_2"), minted
    assert minted.split("_")[1].isdigit() and minted.split("_")[2].isdigit()


def test_the_sae_suffix_is_preserved_verbatim():
    assert _mint_extraction_id(sae_suffix="deadbeef").endswith("_sae_deadbeef")


def test_the_position_variant_is_also_collision_free():
    """The batch site carried a position suffix, so a same-second pair at
    DIFFERENT positions was distinct and the site looked safe. Two runs of the
    SAME position inside one second still collided."""
    ids = {_mint_extraction_id(sae_suffix="9a4db34d", position=3) for _ in range(200)}
    assert len(ids) == 200
    assert all(i.endswith("_003") for i in ids)


class TestNoIdIsBuiltAnywhereElse:
    """A mutation that reverted the call site left every other test in this file
    green: they exercised the minter while production built its own id inline.
    There were also TWO mint sites and the first fix changed one. So assert
    reachability structurally — no f-string may build an extraction id outside
    the minter."""

    def _fstrings_producing_ids(self):
        tree = ast.parse(SERVICE.read_text())
        minter = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_mint_extraction_id"
        )
        inside_minter = {id(n) for n in ast.walk(minter)}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr) or id(node) in inside_minter:
                continue
            literal = "".join(
                v.value for v in node.values if isinstance(v, ast.Constant)
                and isinstance(v.value, str)
            )
            if literal.startswith("extr_"):
                offenders.append(node.lineno)
        return offenders

    def test_the_scan_can_see_the_minter(self):
        """Prove the scan works before trusting its silence."""
        tree = ast.parse(SERVICE.read_text())
        assert any(
            isinstance(node, ast.FunctionDef) and node.name == "_mint_extraction_id"
            for node in ast.walk(tree)
        ), "the minter is gone — this guard is now vacuous"

    def test_every_extraction_id_comes_from_the_minter(self):
        offenders = self._fstrings_producing_ids()
        assert not offenders, (
            "extraction ids are built inline at line(s) "
            f"{offenders} instead of via _mint_extraction_id, so those ids are "
            "still second-granular and collide"
        )

    def test_both_creation_sites_call_it(self):
        tree = ast.parse(SERVICE.read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_mint_extraction_id"
        ]
        assert len(calls) >= 2, (
            f"only {len(calls)} call site(s) found; there are two places an "
            "ExtractionJob id is created and both must mint through the helper"
        )
