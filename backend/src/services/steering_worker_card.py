"""One steering worker per GPU: its queue, hostname, PID file, log and busy marker (Phase 3).

The steering worker is spawned on demand by the API process (see
``api/v1/endpoints/steering.py``) and exits after each generation. With one
worker for the node, two steering requests for different cards waited on each
other. In per-card mode the API spawns one per card, and everything that names a
worker is derived from its card HERE, in one place, so the API (which spawns,
checks and kills by PID) and the worker (which writes its busy marker and checks
that the PID file is its own) cannot disagree:

* queue ``steering.<card>``; the worker also drains the legacy ``steering``
  queue, where messages queued before the switch (or rescued from a removed
  card's queue) wait;
* hostname ``steering-<8 hex>@%h``, unique per card;
* PID file, log and busy marker suffixed with the card.

``card=None`` names the single pre-Phase-3 worker, exactly as before.

NEVER A PATTERN KILL. A per-card worker is found by its PID file and the PIDs
the API recorded when it spawned it — never by matching a command line.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .gpu_placement import normalise_uuid

LEGACY_QUEUE = "steering"


def _run_dir() -> Path:
    from ..core.config import settings

    return settings.run_dir


def card_tag(card: str) -> str:
    """``gpu-247aa582-…`` — the spelling queues use (``gpu_claim.queue_for``)."""
    return normalise_uuid(card)


def queue(card: Optional[str] = None) -> str:
    return LEGACY_QUEUE if card is None else f"{LEGACY_QUEUE}.{card_tag(card)}"


def consumed_queues(card: Optional[str] = None) -> str:
    """The ``-Q`` value: a card's own queue, then the legacy one it drains."""
    return LEGACY_QUEUE if card is None else f"{queue(card)},{LEGACY_QUEUE}"


def hostname(card: Optional[str] = None) -> str:
    return "steering@%h" if card is None else f"steering-{card_tag(card)[4:12]}@%h"


def pid_file(card: Optional[str] = None) -> Path:
    name = "mistudio-celery-steering.pid" if card is None else f"mistudio-celery-steering-{card_tag(card)}.pid"
    return _run_dir() / name


def log_file(card: Optional[str] = None) -> Path:
    name = "celery-steering.log" if card is None else f"celery-steering-{card_tag(card)}.log"
    return _run_dir() / name


def busy_marker(card: Optional[str] = None) -> Path:
    name = "steering-worker-busy.json" if card is None else f"steering-worker-busy-{card_tag(card)}.json"
    return _run_dir() / name


def this_workers_card() -> Optional[str]:
    """The card of the steering worker running in this process, from its spawn env."""
    from ..workers.gpu_supervisor import WORKER_GPU_ENV

    return os.environ.get(WORKER_GPU_ENV) or None
