"""The two deferred torch CVEs are deferred BECAUSE their APIs are unused.

CVE-2025-3000 (`torch.jit.script`) and CVE-2025-3001 (`torch.lstm_cell`) are
open Dependabot alerts against `torch==2.9.1`. They are deliberately NOT
patched: the fixes require torch 2.10/2.13, which is a coupled CUDA/triton pin
swap across the whole ML stack.

That deferral rests on ONE fact — this codebase calls neither API, so the
vulnerable code paths are unreachable regardless of the pin. Verified by grep
on 2026-07-18 and again on 2026-07-21.

A fact nothing enforces is a fact that quietly stops being true. A future
feature adding `torch.jit.script` for inference speed would silently convert a
reasoned deferral into an unpatched vulnerability, and the Dependabot alert
would look exactly the same either way.

So the deferral is now a TEST. If it fails, the choice is: patch torch, or
rewrite the feature that introduced the call — not ignore the alert.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: CVE → the API whose absence justifies deferring it.
DEFERRED_CVE_APIS = {
    "CVE-2025-3000": (
        r"\btorch\.jit\.(script|trace|load)\b|\bjit\.script\(",
        "torch.jit.script/trace/load — memory corruption via TorchScript",
    ),
    "CVE-2025-3001": (
        r"\btorch\.lstm_cell\b|\bnn\.LSTMCell\b|\bLSTMCell\(",
        "torch.lstm_cell / nn.LSTMCell — memory corruption",
    ),
}


def _python_sources():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]


class TestDeferredCVEApisStayUnused:
    def test_the_scan_is_not_vacuous(self):
        """An empty file list would make every assertion below pass."""
        files = _python_sources()
        assert len(files) > 100, (
            f"only found {len(files)} source files under {SRC} — the layout "
            "changed and this guard is checking nothing"
        )

    @pytest.mark.parametrize("cve", sorted(DEFERRED_CVE_APIS))
    def test_the_vulnerable_api_is_not_called(self, cve):
        pattern, description = DEFERRED_CVE_APIS[cve]
        rx = re.compile(pattern)
        offenders = []
        for path in _python_sources():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                # Comments and docstring prose may NAME the API (this file's
                # own docstring does); only executable references count.
                if stripped.startswith("#"):
                    continue
                if rx.search(line):
                    offenders.append(f"{path.relative_to(SRC)}:{i}: {stripped[:90]}")
        assert not offenders, (
            f"{cve} is an OPEN, UNPATCHED Dependabot alert deferred solely "
            f"because this codebase does not call {description}. It now does:\n  "
            + "\n  ".join(offenders)
            + "\n\nEither patch torch (2.10+/2.13+, a coupled CUDA/triton pin "
            "swap) or avoid the API. Do not simply silence this test — the "
            "deferral's only justification is the absence it asserts."
        )

    def test_the_torch_pin_is_the_one_the_deferral_assumes(self):
        """If torch is upgraded past the patched versions, this whole file is
        obsolete and should be deleted rather than left as noise."""
        req = (Path(__file__).resolve().parents[2] / "requirements.txt")
        if not req.exists():
            pytest.skip(f"no requirements.txt at {req}")
        m = re.search(r"^torch==(\d+)\.(\d+)", req.read_text(), re.M)
        assert m, "torch is not pinned in requirements.txt"
        major, minor = int(m.group(1)), int(m.group(2))
        if (major, minor) >= (2, 13):
            pytest.fail(
                f"torch is pinned at {major}.{minor}, which patches both "
                "deferred CVEs. Delete this test file and close the alerts."
            )
