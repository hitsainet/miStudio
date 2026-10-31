"""Every `Settings` field has a reader, and every field is classified once (Feature 036; ENH-003 D7).

D7 deleted 21 fields that were declared, parsed and read by nothing — a page of controls that would
have done nothing. This keeps it from happening again:

* ``test_every_field_has_a_reader`` — a field is READ when some module in ``src/`` other than
  ``core/config.py`` accesses it on a settings object: ``settings.<f>``, ``effective().<f>``,
  ``get_settings().<f>`` or ``getattr(settings, "<f>")``, found by ``ast`` (comments and docstrings
  never count; f-string expressions do, which a tokenizer on Python 3.11 cannot see). A field only a
  ``Settings`` property reads counts when that property is read the same way.
* ``test_every_field_is_classified`` — exactly one of ``RUNTIME_KEYS``, ``RESTART_KEYS``,
  ``SHOWN_READ_ONLY``, ``HIDDEN_FIELDS``, so a new setting forces the decision the spec asks for.

Controls (``0xcc/reviews/036_implementation_controls_2026-10-20.md``):
  C-DEAD    add an unread field to ``Settings``                     → both tests red
  C-READER  revert ``labeling_service``'s reader to a dead name     → the reader test red
"""
from __future__ import annotations

import ast
import pathlib

from src.core import config

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
CONFIG = SRC / "core" / "config.py"

#: Receivers that hold a Settings object (the module singleton, the override accessor's result).
_SETTINGS_NAMES = {"settings", "_settings"}
_SETTINGS_CALLS = {"effective", "get_settings"}

#: A field whose reader is not a settings object. Each carries its reason; keep this empty if you can.
EXEMPT: dict[str, str] = {}


def _is_settings(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _SETTINGS_NAMES
    if isinstance(node, ast.Attribute):  # e.g. `config.settings`, `module.settings`
        return node.attr in _SETTINGS_NAMES
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in _SETTINGS_CALLS
    return False


def _reads(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_settings(node.value):
            found.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_settings(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            found.add(node.args[1].value)
    return found


def _property_fields() -> dict[str, set[str]]:
    """``Settings`` property name -> the fields it reads through ``self``."""
    tree = ast.parse(CONFIG.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")
    out: dict[str, set[str]] = {}
    for item in cls.body:
        if isinstance(item, ast.FunctionDef):
            used = {
                n.attr for n in ast.walk(item)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
            }
            out[item.name] = used
    return out


def _all_reads() -> set[str]:
    reads: set[str] = set()
    for path in SRC.rglob("*.py"):
        if path == CONFIG:
            continue
        reads |= _reads(ast.parse(path.read_text(), filename=str(path)))
    return reads


def test_the_scan_sees_known_readers():
    """The scan itself works: fields read in each style are found (a broken scan finds nothing and
    every field would read as dead, or — worse — a too-loose one finds everything)."""
    reads = _all_reads()
    assert {"data_dir", "labeling_batch_size", "internal_api_url", "gpu_worker_mode"} <= reads
    assert "debug" not in reads, "`logger.debug` must not count as a settings read"


def test_every_field_has_a_reader():
    reads = _all_reads()
    via_property = {
        field
        for prop, fields in _property_fields().items()
        if prop in reads
        for field in fields
    }
    dead = sorted(set(config.Settings.model_fields) - reads - via_property - set(EXEMPT))
    assert dead == [], (
        f"Settings fields nothing reads: {dead}. Wire them, or delete them with their "
        ".env.example lines and docs (ENH-003 D7)."
    )


def test_every_field_is_classified():
    groups = {
        "RUNTIME_KEYS": set(config.RUNTIME_KEYS),
        "RESTART_KEYS": set(config.RESTART_KEYS),
        "SHOWN_READ_ONLY": set(config.SHOWN_READ_ONLY),
        "HIDDEN_FIELDS": set(config.HIDDEN_FIELDS),
    }
    fields = set(config.Settings.model_fields)
    seen: dict[str, list[str]] = {}
    for name, members in groups.items():
        for key in members:
            seen.setdefault(key, []).append(name)
    unclassified = sorted(fields - set(seen))
    twice = sorted(k for k, v in seen.items() if len(v) > 1)
    stale = sorted(set(seen) - fields)
    assert (unclassified, twice, stale) == ([], [], []), (
        f"unclassified={unclassified} classified twice={twice} not a field={stale}"
    )


def test_secrets_and_never_writable_are_consistent():
    assert config.SECRET_FIELDS <= config.SHOWN_READ_ONLY | config.HIDDEN_FIELDS
    assert config.SECRET_FIELDS <= config.NEVER_WRITABLE
    assert not set(config.RUNTIME_KEYS) & (config.NEVER_WRITABLE | config.SECRET_FIELDS)
    assert config.NEVER_WRITABLE <= set(config.Settings.model_fields)


def test_the_dead_fields_stay_deleted():
    """D7's 21, by name. The variables may still be set (``extra="ignore"``); the fields must not
    come back without a reader."""
    deleted = {
        "database_url_sync", "hf_home", "api_host", "api_port", "api_workers", "api_reload",
        "api_base_url", "access_token_expire_minutes", "extraction_progress_interval_seconds",
        "tokenization_filter_enabled", "tokenization_filter_mode", "tokenization_junk_ratio_threshold",
        "extraction_filter_enabled", "extraction_filter_mode", "pre_labeling_filter_enabled",
        "pre_labeling_junk_ratio_threshold", "pre_labeling_single_char_threshold",
        "rate_limit_enabled", "rate_limit_per_minute", "rate_limit_downloads_per_hour", "debug",
    }
    assert len(deleted) == 21
    assert not deleted & set(config.Settings.model_fields)
