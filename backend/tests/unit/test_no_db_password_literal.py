"""No database password lives in a test file (042 FR-042.40; R16; FTASKS 5.8, control K-5.15).

The workstation's database password was a literal in ten test modules (`tests/support/test_databases.py`
and nine migration tests), as the fallback when `DATABASE_URL_SYNC` was unset; R16 rotates it, and a
rotated password still in the tree is a leak waiting for the mirror. Tests now read their database only
from the environment (`tests/support/test_databases.sync_test_url()`).

The scan looks for a credential in every form a test could carry it (memory
`leak-tests-must-look-for-the-encoded-form`):
* the known credentials themselves, in RAW TEXT (comments included), base64 and URL-encoded;
* a URL whose userinfo carries a password, in any string constant, including folded `+` concatenations
  and the constant parts of f-strings — unless the password is a placeholder of at most
  `PLACEHOLDER_MAX` characters (`u:p@h`, `user:pw@…`: fixtures of URL-handling tests), a redaction
  marker (`<redacted>`, `***`) or an f-string hole;
* `PGPASSWORD=<literal>` in a string, and a `password=` keyword or `"password"` dict key with a literal.

Exemptions name ONE line (file, line, count) with a reason, and each must still match a finding, so an
exemption cannot outlive its site. Discovery asserts it scanned the files it should.
"""
from __future__ import annotations

import ast
import base64
import re
import urllib.parse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ROOTS = (REPO / "backend" / "tests", REPO / "rule_runner" / "tests")

#: Built, never written whole, so this file does not carry them either.
KNOWN = ("dev" + "password", "test" + "password")
PLACEHOLDER_MAX = 4
URL_WITH_PASSWORD = re.compile(r"\b[a-z][a-z0-9+.-]*://(?P<user>[^\s/:@]+):(?P<pw>[^\s/@]+)@")
PGPASSWORD = re.compile(r"\bPGPASSWORD\s*=\s*(?P<pw>[A-Za-z0-9][^\s'\"$}{]*)")
#: A redaction marker, an f-string hole or a WHOLE shell variable reference (`${SYNC_TOKEN}`, `$TOKEN`: the
#: value is read from the environment, e.g. the mirror remote in sync-to-clean.yml) is not a password. A literal
#: with a variable inside it (`abc${X}`) is still one.
MARKER = re.compile(r"^(<[^>]*>|\*+|\{[A-Za-z0-9_]*\}|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*)$")

#: (file relative to the repo, line) -> (count, reason). One line each, never a whole file.
EXEMPT: dict[tuple[str, int], tuple[int, str]] = {
    ("backend/tests/unit/test_error_responses_carry_no_exception_text.py", 17): (
        1, "a PLANTED fake secret: the test proves error responses never echo it"),
}


def _forms(word: str) -> set[str]:
    raw = word.encode()
    return {word, base64.b64encode(raw).decode().rstrip("="), urllib.parse.quote(word, safe="")}


def _fold(node: ast.AST) -> str | None:
    """A string constant, a `+` concatenation of foldable parts, or an f-string with `{}` holes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    return None


def scan_source(text: str, rel: str) -> list[tuple[str, int, str]]:
    """Findings `(file, line, what)` in one Python source."""
    out: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for word in KNOWN:
            for form in _forms(word):
                if form in line:
                    out.append((rel, lineno, "a known credential (or its encoded form)"))
    tree = ast.parse(text)
    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        folded = _fold(node)
        if folded is not None:
            for m in URL_WITH_PASSWORD.finditer(folded):
                if len(m["pw"]) > PLACEHOLDER_MAX and not MARKER.match(m["pw"]):
                    seen.add((node.lineno, f"a URL carrying a password ({m['user']}:…)"))
            for m in PGPASSWORD.finditer(folded):
                seen.add((node.lineno, "PGPASSWORD set to a literal"))
        if isinstance(node, ast.keyword) and node.arg and "password" in node.arg.lower():
            value = _fold(node.value)
            if value:
                seen.add((node.value.lineno, f"keyword {node.arg}= with a literal"))
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                key = _fold(k) if k is not None else None
                if key and "password" in key.lower() and _fold(v):
                    seen.add((v.lineno, f"dict key {key!r} with a literal"))
    out.extend((rel, line, what) for line, what in sorted(seen))
    return out


def _files() -> list[Path]:
    return sorted(p for root in ROOTS for p in root.rglob("*.py"))


def findings() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for path in _files():
        out.extend(scan_source(path.read_text(encoding="utf-8"), str(path.relative_to(REPO))))
    return out


def test_no_url_with_password_under_backend_tests():
    files = _files()
    assert len(files) > 500, f"the scan found only {len(files)} test files; it must not pass on nothing"
    assert any(p.name == "test_databases.py" for p in files), "tests/support/test_databases.py was not scanned"
    found = findings()
    unexempted = []
    counts: dict[tuple[str, int], int] = {}
    for rel, line, what in found:
        if (rel, line) in EXEMPT:
            counts[(rel, line)] = counts.get((rel, line), 0) + 1
        else:
            unexempted.append(f"{rel}:{line}: {what}")
    assert not unexempted, ("a database credential in a test file (042 FR-042.40); read the URL from the "
                            "environment through tests/support/test_databases.sync_test_url():\n"
                            + "\n".join(unexempted))
    for key, (count, reason) in EXEMPT.items():
        assert counts.get(key, 0) == count, (
            f"exemption {key} ({reason}) expects {count} finding(s) and matched {counts.get(key, 0)}: "
            "remove or correct it")


def _j(*parts: str) -> str:
    """Join at runtime, so a planted sample is never a literal of this file."""
    return "".join(parts)


S = _j("s3cr3t", "value")


@pytest.mark.parametrize("source, what", [
    (_j('U = "postgresql://postgres:', KNOWN[0], '@localhost:5432/x"\n'), "known credential"),
    (_j('U = "postgresql://postgres:', S, '@localhost:5432/x"\n'), "URL carrying a password"),
    (_j('U = "postgresql://postgres:" + "', S, '" + "@db/x"\n'), "URL carrying a password"),
    (_j('U = f"postgresql://{user}:', S, '@{host}/x"\n'), "URL carrying a password"),
    (_j('R = "https://x-access-token:', S, '${TAIL}@github.com/x.git"\n'), "URL carrying a password"),
    (_j('R = "https://x-access-token:${HEAD}', S, '@github.com/x.git"\n'), "URL carrying a password"),
    (_j('E = "PGPAS', 'SWORD=', S, ' psql -h db"\n'), "PGPASSWORD"),
    (_j('c = connect(host="db", password="', S, '")\n'), "keyword password="),
    (_j('c = {"password": "', S, '"}\n'), "dict key"),
    ("# the old one was " + KNOWN[1] + "\n", "known credential"),
    ('B = "' + base64.b64encode(KNOWN[0].encode()).decode() + '"\n', "known credential"),
])
def test_the_scanner_finds_each_form(source, what):
    found = scan_source(source, "planted.py")
    assert found and any(what.split()[0].lower() in f[2].lower() for f in found), (source, found)


@pytest.mark.parametrize("source", [
    'U = "postgresql://u:p@h:5432/db"\n',
    'U = "postgresql://user:pw@localhost:1/db"\n',
    'U = "http://svc:<redacted>@backend:8000"\n',
    'U = os.environ["DATABASE_URL_SYNC"]\n',
    'U = f"postgresql://{user}:{password}@{host}:{port}/{name}"\n',
    'R = \'REMOTE="https://x-access-token:${SYNC_TOKEN}@github.com/${MIRROR}.git"\'\n',
])
def test_placeholders_and_environment_reads_pass(source):
    assert scan_source(source, "ok.py") == []
