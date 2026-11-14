"""Apply `settings.log_level` / `settings.log_format` to the API process.

THE SETTING WAS DECLARED AND NEVER READ (found live 2026-10-20, tracker 3.4 A1).
`LOG_LEVEL=INFO` is set in the Dockerfile, both compose files and the k8s
manifests, and `Settings.log_level` parses it — but nothing configured logging
in the API process. Uvicorn configures only its own `uvicorn.*` loggers, so the
root logger stayed at Python's default WARNING with no handler, and every
`logger.info` under `src.*` was dropped. The runbook's
`[CircuitService] typed N edge(s)` line could never appear; only uvicorn access
lines did. (Celery workers were unaffected: Celery configures the root logger
from `--loglevel`.)
"""

import json
import logging
import sys

#: Marks the handler this module installs, so a second call is a no-op.
_HANDLER_NAME = "mistudio-root"


class _CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever `sys.stderr` is at emit time.

    A plain StreamHandler binds the stream it was built with; under a test
    runner that swaps `sys.stderr` that is a closed capture object a moment
    later. This mirrors `logging.lastResort`.
    """

    def __init__(self):
        super().__init__(stream=None)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, _value):
        pass


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, fmt: str = "text") -> bool:
    """Install one root handler at `level`. Returns True when it configured.

    Leaves a root logger that someone else already configured alone (a Celery
    worker, a test runner, an embedding application): their handlers are the
    decision, and adding ours would print every record twice.
    """
    root = logging.getLogger()
    if any(getattr(h, "name", None) == _HANDLER_NAME for h in root.handlers):
        root.setLevel(level)
        return True
    if root.handlers:
        return False
    handler = _CurrentStderrHandler()
    handler.name = _HANDLER_NAME
    handler.setFormatter(
        _JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
    return True
