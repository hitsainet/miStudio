"""Old names, read new-first and reported (042 FR-042.8, FR-042.9, FR-042.17; FTDD §6.1; FTID §3.1; TPD-1).

ENH-009 renames miStudio to miWorkbench and miLLM to miServer. Until each alias is removed at P7 on
evidence, an operator's old environment variable, an old host name and an old MCP category key keep
working, and every use is RECORDED so S001's P7 sampler can tell "unused for 7 days" from "never
measured". This module is the per-process half: the hand-kept tables, the readers, the recorder.

* **Standard library only** (`test_legacy_names_table.py::test_module_is_stdlib_only`): the MCP pod has
  no database and the rule-runner image copies only `src/rule_runner/`.
* **The tables are literals** (CF-3, CF-25 m13): no file is read here. `test_legacy_names_table.py`
  compares `ENV` and `HOSTS` with the census export's `env` and `host` records, so the two are kept
  equal by a test, not by construction (a copy that agreed by construction would prove nothing).
* **Evidence is bounded.** `record()` accepts only a `(kind, old)` pair that one of the tables below
  lists, so a process holds at most `len(ENV) + len(HOSTS) + len(CATEGORY_ALIASES)` uses and the
  backend's `legacy_name_uses` table (key `kind, old`) at most as many rows. It grows only when a row
  is added to a table here (and to the export); never per request, per day or per value.
* **Names only, never values** (BR-023): nothing here returns or logs an environment value.

The sink (`set_sink`) is where uses go: the backend API and the Celery processes write to the database
(`services/legacy_names_store.py`), the MCP pod pushes over HTTP (`mcp_server/legacy_names_push.py`).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

KINDS = ("env", "host", "category")
IN_USE_WINDOW = timedelta(hours=24)
#: A process writes a given name at most once a minute, and refreshes at most once a minute.
WRITE_INTERVAL_S = 60.0
REFRESH_INTERVAL_S = 60.0
REPORTER_PATTERN = re.compile(r"^[a-z]+(:[A-Za-z0-9@._-]{1,56})?$")
#: The only process that reports over HTTP (`POST /api/v1/system/legacy-names`): the MCP pod, which has no
#: database. The API and the workers write the store directly, so the route refuses every other name (review
#: rev-mistudio-p0 F5: an open reporter name let any caller grow `legacy_name_reporters` and `_remote_pairs`
#: without bound, and a forged heartbeat under a real reporter's name could make an unmeasured window read
#: as measured).
HTTP_REPORTER = "mcp"


@dataclass(frozen=True)
class EnvName:
    new: str
    old: str
    status: str  # "transition" | "retired"


@dataclass(frozen=True)
class HostAlias:
    old: str
    new: str
    status: str  # "transition" | "retired"
    observed: bool = True


#: FTID §3.1: the runtime names only (test-only switches are not operator configuration, TI-3).
#: ENH-009 P7 (042 FTASKS 18.4, group G1): every row is `retired` on the suite's post-sweep evidence
#: (`legacy-window --all --since 2026-11-13T10:11:41Z`: each env alias `eligible`). A retired old name is
#: never read; when it is still set it is reported in `retired_but_set`. The rows go with the export's
#: `env` records at P7's close (FTASKS 18.15).
#: Renamed outright at P3 and therefore NOT rows (PD-10): the worker-card and image-version variables, now
#: MIWORKBENCH_WORKER_GPU_UUID and MIWORKBENCH_VERSION (set and read by this image only).
ENV: tuple[EnvName, ...] = (
    EnvName(new="MISERVER_API_URL", old="MILLM_API_URL", status="retired"),
    EnvName(new="MIWORKBENCH_API_URL", old="MISTUDIO_API_URL", status="retired"),
    EnvName(new="MIWORKBENCH_BYPASS_PIN", old="MISTUDIO_BYPASS_PIN", status="retired"),
    EnvName(new="MIWORKBENCH_GPU_RECYCLE_DIR", old="MISTUDIO_GPU_RECYCLE_DIR", status="retired"),
    EnvName(new="MIWORKBENCH_GPU_RECYCLE_RSS_GIB", old="MISTUDIO_GPU_RECYCLE_RSS_GIB", status="retired"),
    EnvName(new="MIWORKBENCH_GPU_WORKER_MEMORY_DIR", old="MISTUDIO_GPU_WORKER_MEMORY_DIR", status="retired"),
)

#: FTID §3.1 (R20: no public k8s-*.hitsai.net name exists; the public tunnel names are not host records
#: because cloudflared rewrites Host before the request reaches the Ingress, TI-8). `retired` since ENH-009 P7
#: (the export's `host` records): the Ingress stops serving them with group G5; a request that still
#: arrives under one is recorded, as before, until the rows go at P7's close.
HOSTS: tuple[HostAlias, ...] = (
    HostAlias(old="k8s-mistudio.hitsai.local", new="k8s-miworkbench.hitsai.local", status="retired"),
    HostAlias(old="k8s-mistudio.hitsai.net", new="miworkbench.hitsai.net", status="retired"),
    HostAlias(old="mcp-mistudio.hitsai.local", new="mcp-workbench.hitsai.local", status="retired"),
    HostAlias(old="mcp-mistudio.hitsai.net", new="mcp-workbench.hitsai.net", status="retired"),
    HostAlias(old="mcp-studio.hitsai.local", new="mcp-workbench.hitsai.local", status="retired"),
    HostAlias(old="mcp-studio.hitsai.net", new="mcp-workbench.hitsai.net", status="retired"),
    HostAlias(old="k8s-mcp-studio.hitsai.local", new="k8s-miworkbench.hitsai.local", status="retired"),
    HostAlias(old="mistudio-backend.mistudio", new="miworkbench-backend.miworkbench", status="retired"),
    HostAlias(old="mistudio-backend.mistudio.svc", new="miworkbench-backend.miworkbench.svc", status="retired"),
    HostAlias(old="mistudio-backend.mistudio.svc.cluster.local",
              new="miworkbench-backend.miworkbench.svc.cluster.local", status="retired"),
)

HOSTS_OLD: frozenset[str] = frozenset(h.old for h in HOSTS)
HOSTS_NEW: frozenset[str] = frozenset(h.new for h in HOSTS)
_HOST_NEW_OF: dict[str, str] = {h.old: h.new for h in HOSTS}
_ENV_BY_NEW: dict[str, EnvName] = {r.new: r for r in ENV}
_ENV_BY_OLD: dict[str, EnvName] = {r.old: r for r in ENV}

#: Kept after P7: the judge gate and other "is this host a sibling" questions use both stems.
SIBLING_STEMS: dict[str, tuple[str, ...]] = {
    "miserver": ("miserver", "millm"),
    "miworkbench": ("miworkbench", "mistudio"),
}

#: TPD-5 / FTDD §6.4. From the P3 cut the canonical keys are `miserver_*`, and until ENH-009 P7 the OLD `millm_*`
#: keys mapped to them and were recorded. P7 (042 FTASKS 18.5, group G2) removed the seven aliases on evidence: an
#: old key is now an unknown category (reported on /health as `unknown_categories`) and switches nothing on. The
#: map is kept as a mechanism; the old keys stay as history in `tests/support/mcp_rename_map.json`.
CATEGORY_ALIASES: dict[str, str] = {}
CATEGORY_ALIAS_RECORDS: bool = True

#: One report, the only thing a sink ever receives: names and times, never a value.
Report = dict[str, Any]
Sink = Callable[[Report], None]


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Registry:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.uses: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_write: dict[tuple[str, str], float] = {}
        self.logged: set[tuple[str, str]] = set()
        self.retired_but_set: set[str] = set()
        #: Old names read once at start-up from a source that is not `os.environ` (a `.env` file): in force
        #: for the process's life, so `refresh()` keeps re-stamping them (review rev-mistudio-p0 F7).
        self.in_force: set[tuple[str, str]] = set()
        self.sink: Sink | None = None
        self.reporter: str | None = None
        self.last_refresh: float | None = None
        self.last_sink_error_log: float | None = None
        self.clock: Callable[[], datetime] | None = None
        self.mono: Callable[[], float] | None = None
        self.observing_since = datetime.now(timezone.utc)


_R = _Registry()


def _now() -> datetime:
    return _R.clock() if _R.clock is not None else datetime.now(timezone.utc)


def _mono() -> float:
    return _R.mono() if _R.mono is not None else time.monotonic()


def _set(name: str) -> bool:
    return bool(os.environ.get(name))


# ── readers ────────────────────────────────────────────────────────────────────────────────────────
def env(new: str, default: str | None = None) -> str | None:
    """The NEW variable when set and non-empty; else the OLD one (recorded); else `default`.

    A row whose status is `retired` never reads the old name; when the old one is set it is reported in
    `retired_but_set` instead. An unknown `new` raises `KeyError` (a typo must not read as "unset")."""
    row = _ENV_BY_NEW[new]
    value = os.environ.get(new)
    if value:
        # The old name is NOT in use when the new one is set (the new value wins); if it is still in the
        # environment, `also_set` (computed from os.environ) reports the pair. Recording it here would put a
        # harmless leftover into `in_use` and hold its alias alive past P7 (departure from FTID §3.2 step 4).
        return value
    if row.status == "retired":
        if _set(row.old):
            mark_retired_but_set(row.old)
        return default
    old_value = os.environ.get(row.old)
    if old_value:
        record("env", row.old, row.new)
        return old_value
    return default


def is_retired(new: str) -> bool:
    """True when the row whose new name is `new` is `retired` (its old name is never read)."""
    return _ENV_BY_NEW[new].status == "retired"


def mark_retired_but_set(old: str) -> None:
    """Report a retired old name that is still set somewhere this process reads configuration from (the
    environment, or a `.env` file the Settings read, where `os.environ` never shows it). Unknown names are
    ignored: the set stays bounded by `ENV`."""
    row = _ENV_BY_OLD.get(old)
    if row is None or row.status != "retired":
        return
    with _R.lock:
        _R.retired_but_set.add(old)


def env_from(environ: Any, new: str, default: str | None = None) -> str | None:
    """`env(new)` for code that takes an injectable environment: the real process environment (None or
    `os.environ`) goes through `env()` (recorded); an injected mapping is read new-first, then old, and
    never recorded (it is a test's or a child's dict, not this process's configuration)."""
    if environ is None or environ is os.environ:
        return env(new, default)
    row = _ENV_BY_NEW[new]
    return environ.get(new) or (environ.get(row.old) if row.status == "transition" else None) or default


def env_flag(new: str) -> bool:
    return (env(new) or "").strip().lower() in ("1", "true", "yes", "on")


def host_is_old(host: str | None) -> str | None:
    """The new host an OLD host maps to, else None. Lower-cased; a `:port` and a trailing dot ignored."""
    if not host:
        return None
    h = host.strip().lower()
    if h.startswith("["):  # an IPv6 literal is never an old name
        return None
    h = h.rsplit(":", 1)[0] if ":" in h else h
    return _HOST_NEW_OF.get(h.rstrip("."))


def names_sibling(host: str | None, sibling: str) -> bool:
    """A substring test of `host` against every stem of `sibling` (both the new and the old stem)."""
    if not host:
        return False
    h = host.lower()
    return any(stem in h for stem in SIBLING_STEMS[sibling])


def map_category(name: str) -> str:
    """The canonical key a requested MCP category maps to; records it when CATEGORY_ALIAS_RECORDS."""
    new = CATEGORY_ALIASES.get(name)
    if new is None:
        return name
    if CATEGORY_ALIAS_RECORDS:
        record("category", name, new)
    return new


# ── recorder ───────────────────────────────────────────────────────────────────────────────────────
def _known(kind: str, old: str, new: str) -> bool:
    if kind == "env":
        row = _ENV_BY_OLD.get(old)
        return row is not None and row.new == new
    if kind == "host":
        return _HOST_NEW_OF.get(old) == new
    if kind == "category":
        return CATEGORY_ALIASES.get(old) == new
    return False


def known_use(kind: str, old: str, new: str) -> bool:
    """True when `(kind, old) -> new` is a row of this module's tables (the only uses ever recorded)."""
    return _known(kind, old, new)


def record(kind: str, old: str, new: str) -> None:
    """Remember one use of an old name; hand the process's report to the sink at most once a minute per
    name. Never raises: an unknown pair is logged and dropped (evidence stays bounded by the tables)."""
    if not _known(kind, old, new):
        logger.warning("legacy_names: ignoring an unknown use (%s, %s -> %s)", kind, old, new)
        return
    now = _now()
    key = (kind, old)
    due = False
    with _R.lock:
        use = _R.uses.get(key)
        if use is None:
            _R.uses[key] = {"kind": kind, "old": old, "new": new, "first_seen": now, "last_seen": now}
        else:
            use["last_seen"] = max(use["last_seen"], now)
            use["new"] = new
        first_log = key not in _R.logged
        _R.logged.add(key)
        mono = _mono()
        last = _R.last_write.get(key)
        if _R.sink is not None and (last is None or mono - last >= WRITE_INTERVAL_S):
            _R.last_write[key] = mono
            due = True
    if first_log:
        logger.info("old name %s read (%s); set %s instead", old, kind, new)
    if due:
        _deliver()


def record_in_force(kind: str, old: str, new: str) -> None:
    """`record()` for a value read ONCE and in force for the process's life (a `Settings` field whose old
    variable came from a `.env` file, where `os.environ` never shows it): `refresh()` keeps re-stamping it,
    as it does an old variable still in the environment. Without this a dotenv old name aged out of the
    24-hour window while still in use (review rev-mistudio-p0 F7)."""
    if _known(kind, old, new):
        with _R.lock:
            _R.in_force.add((kind, old))
    record(kind, old, new)


def _env_pairs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    also_set, conflicts = [], []
    for row in ENV:
        new_v, old_v = os.environ.get(row.new), os.environ.get(row.old)
        if new_v and old_v:
            also_set.append({"old": row.old, "new": row.new})
            if new_v != old_v:  # compared, never returned
                conflicts.append({"old": row.old, "new": row.new})
    return also_set, conflicts


def _in_force_env() -> None:
    """An old env name effective NOW (old set, new unset) is in use even if no code path has read it yet."""
    for row in ENV:
        if row.status == "transition" and _set(row.old) and not _set(row.new):
            key = ("env", row.old)
            now = _now()
            with _R.lock:
                use = _R.uses.get(key)
                if use is None:
                    _R.uses[key] = {"kind": "env", "old": row.old, "new": row.new, "first_seen": now,
                                    "last_seen": now}
                else:
                    use["last_seen"] = max(use["last_seen"], now)


def report() -> Report:
    """This process's full report: every recorded use, and the environment's also_set / conflicts."""
    also_set, conflicts = _env_pairs()
    with _R.lock:
        uses = [{"kind": u["kind"], "old": u["old"], "new": u["new"], "last_seen": _iso(u["last_seen"])}
                for u in sorted(_R.uses.values(), key=lambda u: (u["kind"], u["old"]))]
        reporter = _R.reporter
    return {"reporter": reporter, "uses": uses, "also_set": also_set, "conflicts": conflicts}


def _deliver() -> None:
    sink = _R.sink
    if sink is None:
        return
    rep = report()
    try:
        sink(rep)
    except Exception as exc:  # the sink never fails configuration or a request
        mono = _mono()
        if _R.last_sink_error_log is None or mono - _R.last_sink_error_log >= WRITE_INTERVAL_S:
            _R.last_sink_error_log = mono
            logger.warning("legacy_names: the sink failed (%s); evidence kept in memory", type(exc).__name__)


def refresh(now: datetime | None = None) -> bool:
    """Re-stamp every recorded name still in force and hand the full report to the sink — an EMPTY report
    included (a heartbeat moves this reporter's `last_report`, CF-5). Throttled to once a minute per
    process; returns True when it ran. Env: in force while the old name is set and the new one is not.
    Category: in force for the process's life (configuration is read once). Host: never re-stamped here,
    only by a request."""
    mono = _mono()
    with _R.lock:
        if _R.last_refresh is not None and mono - _R.last_refresh < REFRESH_INTERVAL_S:
            return False
        _R.last_refresh = mono
    stamp = now or _now()
    _in_force_env()
    with _R.lock:
        for (kind, old), use in _R.uses.items():
            if kind == "env":
                row = _ENV_BY_OLD[old]
                if (_set(row.old) and not _set(row.new)) or (kind, old) in _R.in_force:
                    use["last_seen"] = max(use["last_seen"], stamp)
            elif kind == "category":
                use["last_seen"] = max(use["last_seen"], stamp)
    _deliver()
    return True


# ── views ──────────────────────────────────────────────────────────────────────────────────────────
def snapshot(now: datetime | None = None) -> dict[str, list[dict[str, str]]]:
    """BR-023's block for THIS process: exactly `in_use`, `also_set`, `conflicts`, `retired_but_set`.
    `in_use` entries carry exactly `kind, old, new, last_seen`; pairs exactly `old, new`."""
    _in_force_env()
    cutoff = (now or _now()) - IN_USE_WINDOW
    also_set, conflicts = _env_pairs()
    with _R.lock:
        in_use = [{"kind": u["kind"], "old": u["old"], "new": u["new"], "last_seen": _iso(u["last_seen"])}
                  for u in sorted(_R.uses.values(), key=lambda u: (u["kind"], u["old"]))
                  if u["last_seen"] >= cutoff]
        retired = sorted(_R.retired_but_set)
    for row in ENV:
        if row.status == "retired" and _set(row.old) and row.old not in retired:
            retired.append(row.old)
    retired_pairs = [{"old": o, "new": _ENV_BY_OLD[o].new} for o in sorted(retired)]
    return {"in_use": in_use, "also_set": also_set, "conflicts": conflicts, "retired_but_set": retired_pairs}


def watching() -> dict[str, int]:
    return {"env": len(ENV), "host": sum(1 for h in HOSTS if h.observed), "category": len(CATEGORY_ALIASES)}


def unobserved_hosts() -> list[str]:
    return sorted(h.old for h in HOSTS if not h.observed)


def meta() -> dict[str, Any]:
    """This process's `legacy_names_meta` part; `error` is always present (CF-24)."""
    return {"observing_since": _iso(_R.observing_since), "in_use_window_hours": 24, "watching": watching(),
            "unobserved_hosts": unobserved_hosts(), "error": None}


# ── wiring ─────────────────────────────────────────────────────────────────────────────────────────
def set_sink(fn: Sink | None, reporter: str) -> None:
    """Set where this process's reports go and the name it reports under (`api`, `beat`, `mcp`,
    `worker:<celery node name>`)."""
    if not REPORTER_PATTERN.match(reporter):
        raise ValueError(f"reporter name {reporter!r} does not match {REPORTER_PATTERN.pattern}")
    with _R.lock:
        _R.sink = fn
        _R.reporter = reporter


def reporter() -> str | None:
    return _R.reporter


def reset(clock: Callable[[], datetime] | None = None, mono: Callable[[], float] | None = None) -> None:
    """TESTS ONLY (an AST check keeps it out of `src/` callers): an empty registry, optional clocks."""
    global _R
    _R = _Registry()
    _R.clock = clock
    _R.mono = mono
    if clock is not None:
        _R.observing_since = clock()
