"""The runner's configuration: ``RUNNER_*`` environment variables only (Feature 040 S4; FTDD Appendix C.4).

A frozen dataclass, no pydantic-settings: the image stays small and the import guard simple. Ceilings
(``RUNNER_MAX_*``) bound what a caller may ask for; the FLOOR comes from the contract file, never from here (the
runner carries no second copy of a contract value). The runner reads nothing else from its environment — and its
manifest gives it nothing else (Property 1, `test_rule_runner_manifest.py`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .contract import CONTRACT_PATH


def _int(env: Dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


@dataclass(frozen=True)
class RunnerConfig:
    bundle_dir: Path
    contract_path: Path
    host: str
    port: int
    interactive_slots: int
    batch_slots: int
    batch_texts: int
    document_cache: int
    max_call_fuel: int
    max_load_fuel: int
    max_prelude_fuel: int
    max_memory_bytes: int
    max_input_code_points: int
    max_output_bytes: int
    max_wall_ms: int
    max_source_bytes: int

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "RunnerConfig":
        e = dict(os.environ if env is None else env)
        return cls(
            bundle_dir=Path(e.get("RUNNER_BUNDLE_DIR", "/app/bundle")),
            contract_path=Path(e.get("RUNNER_CONTRACT_PATH", str(CONTRACT_PATH))),
            host=e.get("RUNNER_HOST", "0.0.0.0"),  # noqa: S104 - the Service's port, admitted by the NetworkPolicy
            port=_int(e, "RUNNER_PORT", 8080),
            interactive_slots=max(1, _int(e, "RUNNER_INTERACTIVE_SLOTS", 2)),
            batch_slots=max(1, _int(e, "RUNNER_BATCH_SLOTS", 2)),
            batch_texts=max(1, _int(e, "RUNNER_BATCH_TEXTS", 256)),
            document_cache=max(1, _int(e, "RUNNER_DOCUMENT_CACHE", 64)),
            max_call_fuel=_int(e, "RUNNER_MAX_CALL_FUEL", 1 << 35),
            max_load_fuel=_int(e, "RUNNER_MAX_LOAD_FUEL", 1 << 31),
            max_prelude_fuel=_int(e, "RUNNER_MAX_PRELUDE_FUEL", 1 << 31),
            max_memory_bytes=_int(e, "RUNNER_MAX_MEMORY_BYTES", 512 << 20),
            max_input_code_points=_int(e, "RUNNER_MAX_INPUT_CODE_POINTS", 1 << 20),
            max_output_bytes=_int(e, "RUNNER_MAX_OUTPUT_BYTES", 1 << 16),
            max_wall_ms=_int(e, "RUNNER_MAX_WALL_MS", 60_000),
            max_source_bytes=_int(e, "RUNNER_MAX_SOURCE_BYTES", 1 << 20),
        )
