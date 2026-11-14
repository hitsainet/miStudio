"""`GET /api/v1/version` names the app miWorkbench and reads the image's version from `MIWORKBENCH_VERSION`
(042 PD-10, TPD-26; FTASKS 12.18, 12.20).

`MISTUDIO_VERSION` was renamed OUTRIGHT at P3 (setter and reader in one image): the Dockerfile sets the new name and
the route reads only it, so an old variable alone is not this image's version.
"""
from __future__ import annotations

import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
OLD = "_".join(("MISTUDIO", "VERSION"))


async def test_version_reports_miworkbench_and_reads_new_var(client, monkeypatch):
    monkeypatch.delenv(OLD, raising=False)
    monkeypatch.setenv("MIWORKBENCH_VERSION", "9.9.9+cut")
    body = (await client.get("/api/v1/version")).json()
    assert body == {"version": "9.9.9+cut", "app": "miWorkbench"}

    monkeypatch.delenv("MIWORKBENCH_VERSION")
    monkeypatch.setenv(OLD, "0.0.1+old")
    assert (await client.get("/api/v1/version")).json()["version"] != "0.0.1+old"

    dockerfile = (BACKEND / "Dockerfile").read_text()
    assert re.search(r"^ENV MIWORKBENCH_VERSION=\$\{APP_VERSION\}$", dockerfile, re.M)
    assert OLD not in dockerfile
