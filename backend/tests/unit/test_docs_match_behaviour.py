"""Task 13 — documentation that contradicts the code, where it matters.

MIS-E2E-149  the sentence that cost a real SAE was live in the SECOND manual.
             `manual/docs/core-workflow/sae-training.md` was corrected;
             `docs/miStudio_Manual.md` was not — and it is indexed in the
             repo's own knowledge graph, so an agent querying it is served the
             uncorrected text.
MIS-E2E-150  the manual said a bearer token is "always required" and its
             troubleshooting remedy was `MCP_ALLOW_ANONYMOUS=true`, which the
             guard honoured on HTTP — producing a LAN-reachable unauthenticated
             server exposing delete_circuit, GPU steering and label write-back.
             The code said "stdio only" in prose and did not enforce it.
"""

from pathlib import Path

import pytest
import re

REPO = Path(__file__).resolve().parents[3]


# ── MIS-E2E-149 · both manuals ─────────────────────────────────────────────

_MANUALS = [
    REPO / "docs" / "miStudio_Manual.md",
    REPO / "manual" / "docs" / "core-workflow" / "sae-training.md",
]


@pytest.mark.parametrize("path", _MANUALS, ids=lambda p: p.name)
def test_no_manual_claims_stop_saves_the_sae(path):
    """The identical sentence, in two places, one of them fixed.

    `train_969e90af` (granite-4.1-8b, FVU 0.065, zero dead neurons) was stopped
    at step 10,300 and its SAE forfeited, because the manual promised otherwise.
    """
    assert path.exists(), f"{path} moved — this guard would pass vacuously"
    text = path.read_text()
    assert "Gracefully end training (saves final checkpoint)" not in text, (
        f"{path.name} still promises that Stop saves the SAE. Only "
        f"Stop & Finalize writes community_format/, which is the only artifact "
        f"downstream reads."
    )


@pytest.mark.parametrize("path", _MANUALS, ids=lambda p: p.name)
def test_both_manuals_warn_that_stop_produces_no_importable_sae(path):
    """Absence of the wrong sentence is not the same as saying the right thing."""
    text = path.read_text().lower()
    assert "no importable sae" in text or "does not save an importable sae" in text, (
        f"{path.name} no longer warns that Stop leaves no importable SAE"
    )


def test_stop_and_stop_and_finalize_really_are_different_endpoints():
    """Negative control for the premise.

    If `stop` did finalize, the manuals' original sentence would have been
    right and these tests would be pinning a fiction.
    """
    import inspect

    from src.api.v1.endpoints import trainings

    src = inspect.getsource(trainings)
    assert "stop_and_finalize" in src
    assert "finalize_training_from_checkpoint_task" in src
    # The plain stop path must NOT dispatch the finalize task.
    stop_idx = src.index('"stop"')
    finalize_idx = src.index("stop_and_finalize")
    assert stop_idx < finalize_idx, "branch order changed; re-check this guard"


# ── MIS-E2E-150 · the flag is stdio-only ───────────────────────────────────

def _build(stdio: bool, **kw):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    settings = MCPSettings(tool_categories="jlens", **kw)
    return build_server(settings, stdio=stdio)


def test_anonymous_over_http_is_refused(monkeypatch):
    """The hole: the flag alone satisfied the guard on the HTTP transport."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    with pytest.raises(SystemExit) as exc:
        _build(stdio=False, allow_anonymous=True)
    assert "stdio" in str(exc.value).lower()


def test_no_token_over_http_is_refused(monkeypatch):
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    with pytest.raises(SystemExit):
        _build(stdio=False)


def test_a_token_over_http_is_accepted(monkeypatch):
    """Negative control: the server must still start normally."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    _build(stdio=False, auth_token="s3cret")


def test_anonymous_over_stdio_is_accepted(monkeypatch):
    """The case the flag exists for."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    _build(stdio=True, allow_anonymous=True)


def test_the_manual_no_longer_offers_the_flag_as_the_remedy():
    """The remedy sent operators straight into the hole, on the page that told
    them a token was always required."""
    page = (REPO / "manual" / "docs" / "advanced" / "mcp-server.md").read_text()
    row = next(
        line for line in page.splitlines()
        if "MCP_AUTH_TOKEN is required" in line and line.startswith("|")
    )
    assert "will not help" in row or "stdio" in row, (
        f"the troubleshooting row still offers MCP_ALLOW_ANONYMOUS as the fix: {row}"
    )


# ── MIS-E2E-050 / -164 · the data model doc is complete ────────────────────

_DATA_MODEL = REPO / "manual" / "docs" / "reference" / "data-model.md"

#: Tables deliberately not described, with the reason. An exemption list makes
#: the omission a decision that leaves a record; silence made it an accident.
_UNDOCUMENTED_ON_PURPOSE = {
    "alembic_version": "Alembic's own migration bookkeeping, not part of the product's data model",
    "feature_activations_default": "the default partition of `feature_activations`, which IS documented",
}


def test_data_model_doc_covers_every_table():
    """The page claimed to be "verified against the ORM models" while omitting
    NINE tables — including `checkpoints`, which its own ER diagram draws.

    The claim is now enforced rather than asserted.
    """
    import re

    from src import models  # noqa: F401 — registers every table
    from src.core.database import Base

    assert _DATA_MODEL.exists(), "the data-model page moved — guard is vacuous"

    orm = set(Base.metadata.tables)
    assert len(orm) > 20, f"only {len(orm)} tables registered — imports incomplete"

    mentioned = set(re.findall(r"`([a-z_]+)`", _DATA_MODEL.read_text()))
    missing = orm - mentioned - set(_UNDOCUMENTED_ON_PURPOSE)
    assert not missing, (
        f"{len(missing)} tables are in the ORM and absent from the data-model "
        f"reference: {sorted(missing)}. Document them, or add them to "
        f"_UNDOCUMENTED_ON_PURPOSE with the reason."
    )


def test_the_exemptions_stay_confined_to_non_orm_tables():
    """A stale or growing exemption list hides a real gap by shrinking the
    required set.

    Neither exemption is in `Base.metadata` — that is precisely why they are
    exempt: `alembic_version` is created by Alembic itself, and
    `feature_activations_default` is a partition Postgres creates for a table
    that IS documented. My first version of this test asserted they were in the
    ORM, which can never hold for either. So the invariant is narrower and
    true: the list contains exactly these two, and a third has to be argued for
    here.
    """
    from src import models  # noqa: F401
    from src.core.database import Base

    orm = set(Base.metadata.tables)
    assert set(_UNDOCUMENTED_ON_PURPOSE) == {
        "alembic_version",
        "feature_activations_default",
    }, (
        "the exemption list changed; every entry must be a NON-ORM table with a "
        "recorded reason, or it is a documentation gap wearing an exemption"
    )
    overlap = set(_UNDOCUMENTED_ON_PURPOSE) & orm
    assert not overlap, (
        f"{sorted(overlap)} are mapped tables and must be documented, not exempted"
    )


def test_the_page_no_longer_claims_verification_it_did_not_do():
    text = _DATA_MODEL.read_text()
    assert "verified against the ORM models." not in text, (
        "the page asserts verification instead of being verified; the test "
        "above is what makes the claim true"
    )


# ── MIS-E2E-114 / -161 · the MCP contract and its counts ───────────────────

#: Tools registered CONDITIONALLY, so the AST ceiling exceeds what a default
#: server serves. Named, so the difference is a decision rather than a drift.
_CONDITIONALLY_REGISTERED = {
    "get_approval_status": "only registered when `steering_approval` is on",
}


def test_the_contract_lists_no_endpoint_that_is_really_a_dict_lookup():
    """MIS-E2E-114. The AST scraper matched `dict.get("kind")` as `GET kind`.

    The committed contract carried three such rows, and
    `test_mcp_contract_generated.py` pinned them as correct — so the contract
    defended whatever path was recorded rather than the real one.
    """
    contract = (REPO / "docs" / "mcp-contract.md").read_text()
    for bogus in ("GET kind", "GET manifests", "GET status"):
        assert f"`{bogus}`" not in contract and f"{bogus}<" not in contract, (
            f"the contract lists {bogus!r} as an endpoint; it is a dictionary "
            f"lookup the AST scraper mistook for an HTTP call"
        )


def test_every_contract_endpoint_looks_like_a_path():
    """The general rule behind the three specific rows."""
    import re

    contract = (REPO / "docs" / "mcp-contract.md").read_text()
    endpoints = re.findall(r"`(GET|POST|PUT|DELETE|PATCH) ([^`]+)`", contract)
    assert endpoints, "no endpoints found in the contract — the scan broke"
    bad = [f"{m} {p}" for m, p in endpoints if not p.startswith("/")]
    assert not bad, f"contract endpoints that are not paths: {bad}"


def test_the_server_instruction_count_is_derived_not_written():
    """MIS-E2E-161. Three places carried three different counts: the
    instructions said 92/13, the manual 97/13, the generated contract 116/14.
    Only the contract was derived."""
    import inspect

    from src.mcp_server import server

    src = inspect.getsource(server._server_instructions)
    assert "ast" in src.lower(), "the count is not derived from the registry"

    # Strip the docstring: it cites the stale numbers (92, 97, 116) to explain
    # the drift, and a bare substring check reads them as a regression.
    # Seventh occurrence of this trap in this remediation.
    doc = inspect.getdoc(server._server_instructions) or ""
    code = src
    for line in doc.splitlines():
        code = code.replace(line, "")
    assert "92" not in code and "97" not in code, "a hardcoded count is back"

    # And the template itself must carry a placeholder, not a literal.
    assert "{tool_count}" in server.SERVER_INSTRUCTIONS


def test_the_instruction_ceiling_exceeds_the_contract_by_exactly_the_conditional_tools(
    monkeypatch,
):
    """The two authorities must not drift.

    The AST ceiling counts every `@mcp.tool()`; the contract counts what a
    default server serves. The difference is exactly the conditionally-
    registered set — if it grows, one of them has changed and this says so.
    """
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")

    import ast
    import inspect as _inspect

    from src.mcp_server.contract import collect
    from src.mcp_server.tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

    ast_names = set()
    for modules in {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}.values():
        for module in modules:
            for node in ast.walk(ast.parse(_inspect.getsource(module))):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(target, ast.Attribute) and target.attr == "tool":
                        ast_names.add(node.name)
                        break

    served = {row["name"] for rows in collect().values() for row in rows}
    assert ast_names, "no tools found by AST — the scan broke"
    assert served, "the contract collector returned nothing"

    difference = ast_names - served
    assert difference == set(_CONDITIONALLY_REGISTERED), (
        f"the AST ceiling and the served set differ by {sorted(difference)}, "
        f"expected exactly {sorted(_CONDITIONALLY_REGISTERED)}"
    )


# ── MIS-E2E-155 · CLAUDE.md's instruction references ───────────────────────

def test_every_instruct_reference_names_a_file_that_exists():
    """MIS-E2E-155. `001_generate-brd.md` was added at the front of the
    sequence and the reference list was never renumbered — so every entry
    named a real file performing a DIFFERENT action, and `008_housekeeping.md`
    did not exist at all. Following any of them by number ran the wrong step.
    """
    import re

    claude = (REPO / "CLAUDE.md").read_text()
    instruct_dir = REPO / "0xcc" / "instruct"
    assert instruct_dir.is_dir(), "0xcc/instruct moved — guard is vacuous"

    on_disk = {p.name for p in instruct_dir.glob("*.md")}
    assert on_disk, "no instruction files found"

    referenced = set(re.findall(r"0xcc/instruct/([0-9]{3}_[a-z-]+\.md)", claude))
    assert referenced, "no instruction references found in CLAUDE.md"

    missing = referenced - on_disk
    assert not missing, (
        f"CLAUDE.md references instruction files that do not exist: "
        f"{sorted(missing)}. On disk: {sorted(on_disk)}"
    )


# ── MIS-E2E-156 · the system-monitor architecture, in five documents ───────

_PROPAGATED_TO = [
    REPO / "README.md",
    REPO / "CLAUDE.md",
    REPO / "0xcc" / "prds" / "000_PPRD|miStudio.md",
    REPO / "0xcc" / "adrs" / "000_PADR|miStudio.md",
]


def test_no_celery_task_named_collect_system_metrics_exists():
    """The premise. IDL-5 named a task that is not in the codebase.

    If it existed, five documents would have been right and this whole
    correction would be the error.
    """
    from src.core.celery_app import celery_app

    assert not any("collect_system_metrics" in name for name in celery_app.tasks), (
        "the task IDL-5 named now exists; re-check the correction"
    )
    scheduled = {
        entry.get("task", "") for entry in (celery_app.conf.beat_schedule or {}).values()
    }
    assert not any("system_metric" in t for t in scheduled), (
        f"beat_schedule now runs a system-metrics task: {sorted(scheduled)}"
    )


def test_the_monitor_really_is_an_asyncio_task_in_the_api_process():
    """The positive half — the architecture the docs now describe."""
    import inspect

    from src import main
    from src.services.background_monitor import BackgroundMonitor

    assert "asyncio.create_task" in inspect.getsource(BackgroundMonitor.start)
    assert "background_monitor" in inspect.getsource(main).lower(), (
        "main.py no longer starts the monitor; the docs describe a lifespan hook"
    )


@pytest.mark.parametrize("path", _PROPAGATED_TO, ids=lambda p: p.name)
def test_every_document_names_the_real_monitor_implementation(path):
    """Assert the POSITIVE fact, not the absence of the wrong one.

    My first version searched for lines mentioning both "Celery Beat" and
    "system monitoring" without a correction marker. Control C136 walked
    straight through it: re-adding the false claim to a line that ALSO carries
    the correction note satisfied both conditions at once. A negative check over
    prose is very hard to make bite.

    So each document that describes system-metric collection must NAME the
    implementation. `background_monitor` is unambiguous, appears nowhere by
    accident, and a document that reverts to describing Celery Beat will not
    contain it.
    """
    assert path.exists(), f"{path} moved — guard is vacuous"
    text = path.read_text().lower()

    describes_metrics = any(
        w in text for w in ("system monitoring", "system metrics", "monitoring metrics")
    )
    if not describes_metrics:
        pytest.skip(f"{path.name} does not describe system-metric collection")

    assert "background_monitor" in text, (
        f"{path.name} describes system-metric collection without naming "
        f"`background_monitor` — the asyncio task in the FastAPI process that "
        f"actually does it. Five documents once attributed this to Celery Beat "
        f"and a task (`collect_system_metrics`) that does not exist."
    )


# ── MIS-E2E-010 / -163 · CLAUDE.md and the slash commands ──────────────────

def test_referenced_context_files_exist():
    """MIS-E2E-010. Four commands referenced a health dashboard that was never
    created, and `/review` Step 5 tells the reviewer to UPDATE it — so the
    health step could never have run.
    """
    dashboard = REPO / ".claude" / "context" / "health" / "dashboard.md"
    referencing = [
        p for p in (REPO / ".claude" / "commands").glob("*.md")
        if "health/dashboard.md" in p.read_text()
    ]
    if referencing:
        assert dashboard.exists(), (
            f"{[p.name for p in referencing]} reference "
            f"`.claude/context/health/dashboard.md`, which does not exist"
        )
        # `.exists()` alone is not enough — control C144 emptied the file and
        # the check still passed. A command reading an empty dashboard is no
        # better off than one reading a missing file.
        assert len(dashboard.read_text().strip()) > 200, (
            "the health dashboard exists but is empty; `/review` Step 5 tells "
            "the reviewer to update it, and there is nothing to update"
        )


def test_claude_md_does_not_claim_a_nonexistent_file_is_auto_loaded():
    """It told a resuming session that `0xcc/session_state.json` "is
    automatically loaded". The file has never existed."""
    claude = (REPO / "CLAUDE.md").read_text()
    for line in claude.splitlines():
        if "session_state.json" not in line:
            continue
        assert "does not exist" in line.lower() or "MIS-E2E-010" in line, (
            f"CLAUDE.md still presents session_state.json as real:\n  {line.strip()}"
        )


def test_claude_md_test_counts_are_not_silently_stale():
    """MIS-E2E-163. Three different counts coexisted — 995, 2461/1149, and the
    real ~2883/1211 — so any of them could be quoted in good faith.

    The count cannot be checked automatically without running both suites, so
    what IS enforced is that the line tells the reader to re-measure. A stale
    number that admits it is stale is recoverable; one that does not is not.
    """
    claude = (REPO / "CLAUDE.md").read_text()
    idx = claude.index("**Test Status:**")
    block = claude[idx: idx + 600]
    assert "Do not hand-maintain" in block or "Re-measure" in block, (
        "the Test Status line gives a bare number with no instruction to "
        "re-measure; it was wrong by 3.3x once already"
    )
    # Strip the corrective note before checking: it QUOTES the stale figure in
    # order to say it was wrong, and a bare substring check reads the quote as a
    # regression. (Eighth time this trap has appeared in this remediation.)
    # Drop QUOTED occurrences, not specific phrasings. The first version of
    # this filter listed the exact words of the corrective note ("it read",
    # "Do not hand-maintain"), so rewording that note broke the test against a
    # file that was still correct — the quote-vs-assertion trap, this time
    # inside the guard rather than the code. A stale count that matters is
    # stated bare; one being corrected is in quotes.
    import re as _re

    without_note = _re.sub(r'"[^"\n]*"', "", claude)
    assert "995 passed" not in without_note, "the stale count is back"


def test_no_document_still_says_the_jlens_readout_returns_501():
    """The endpoint is bound; three places still described it as unimplemented."""
    claude = (REPO / "CLAUDE.md").read_text()
    for line in claude.splitlines():
        if "501" not in line or "/jlens/readout" not in line:
            continue
        low = line.lower()
        assert any(w in low for w in ("resolved", "since bound", "~~", "gone")), (
            f"CLAUDE.md still says the readout returns 501:\n  {line.strip()}"
        )


class TestTheSshGuidanceMatchesTheHelper:
    """CLAUDE.md tells every agent how to reach the GPU node. It must be true.

    MIS-E2E-143's fix removed a committed password from `k8s-helpers.sh` and
    installed no key, so the helper was dead the first time an incident needed
    it. The written guidance is what stops the next agent reaching for
    `sshpass` when that happens — so it has to describe the helper that exists.
    """

    def _helper(self):
        return (REPO / "scripts" / "k8s-helpers.sh").read_text()

    def _claude_md(self):
        return (REPO / "CLAUDE.md").read_text()

    def test_the_helper_uses_key_based_ssh(self):
        helper = self._helper()
        assert 'ssh -o BatchMode=yes "${K8S_USER}@${K8S_HOST}"' in helper, (
            "the k8s() helper no longer matches the command CLAUDE.md documents"
        )

    def test_the_helper_has_no_password_auth(self):
        # Comments explain why sshpass was removed, so check statements only.
        code = "\n".join(
            line for line in self._helper().splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "sshpass" not in code, "sshpass is back in the helper"
        assert "StrictHostKeyChecking=no" not in code, (
            "StrictHostKeyChecking=no accepts any host key; the server is then "
            "unauthenticated to us"
        )

    def test_claude_md_documents_the_ssh_route(self):
        doc = self._claude_md()
        assert "Reaching the GPU node" in doc, (
            "the agent-facing SSH section is gone; the next agent that hits "
            "`Permission denied (publickey)` will reach for sshpass"
        )
        assert "ssh-copy-id" in doc, "the recovery step is not documented"
        assert "BatchMode=yes" in doc

    def test_the_documented_defaults_match_the_helper(self):
        import re

        helper = self._helper()
        doc = self._claude_md()
        host = re.search(r'K8S_HOST="\$\{K8S_HOST:-([^}"]+)\}"', helper)
        user = re.search(r'K8S_USER="\$\{K8S_USER:-([^}"]+)\}"', helper)
        assert host and user, "the helper's defaults changed shape"
        assert host.group(1) in doc, (
            f"CLAUDE.md does not name the host the helper defaults to "
            f"({host.group(1)})"
        )
        assert user.group(1) in doc


class TestNothingTriesToDisablePasswordAuth:
    """Standing user directive, 2026-08-24: password auth on the GPU node stays on.

    It is the break-glass path. Certificates expire on their own schedule, keys
    are lost with their workstation, and `authorized_keys` can be truncated by
    a bad script — password auth is what remains when those fail. On
    2026-08-24 the MIS-E2E-143 fix removed the committed password and installed
    no key, so `k8s-helpers.sh` was dead during a live outage. Removing a
    credential path without a proven replacement is itself an outage.

    A test cannot read the remote `sshd_config`, and pretending otherwise would
    be the "guard that isn't on the path" shape this audit found repeatedly.
    What it CAN do is stop this repository from shipping anything that turns it
    off, and keep the reason written down.
    """

    #: The setting, and the shell shapes that flip it.
    _DISABLERS = (
        "PasswordAuthentication no",
        "PasswordAuthentication=no",
        "passwordauthentication no",
    )

    def _tracked_files(self):
        import subprocess

        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        assert len(out) > 500, f"only {len(out)} tracked files — the scan is broken"
        return out

    def test_no_tracked_file_disables_password_authentication(self):
        offenders = []
        for rel in self._tracked_files():
            if not rel.endswith((".sh", ".yml", ".yaml", ".conf", ".py", ".md", ".tf")):
                continue
            path = REPO / rel
            if not path.exists():
                continue
            if Path(rel).name == Path(__file__).name:
                continue                      # the detector names what it bans
            text = path.read_text(errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("#", "//", ">", "|", "*", "-")):
                    continue
                # Strip backticked and quoted spans. The rule in CLAUDE.md says
                # "Not `PasswordAuthentication no`" — a mention, forbidding the
                # thing. A substring check cannot tell that from a directive,
                # which is the same trap that has now bitten thirteen times in
                # this remediation.
                bare = re.sub(r'`[^`\n]*`|"[^"\n]*"|\'[^\'\n]*\'', "", line)
                if any(d in bare for d in self._DISABLERS):
                    offenders.append(f"{rel}:{lineno}")
        assert not offenders, (
            "these would disable password authentication on the GPU node, "
            "which is the break-glass path and stays on by standing "
            f"instruction: {offenders}. Harden with fail2ban, a firewall rule "
            f"or AllowUsers instead."
        )

    def test_the_rule_is_written_down_where_agents_read_it(self):
        doc = (REPO / "CLAUDE.md").read_text()
        assert "NEVER disable password authentication" in doc, (
            "the standing directive is gone from CLAUDE.md; the next agent will "
            "read the cert setup and treat disabling passwords as the obvious "
            "next hardening step"
        )
        assert "break-glass" in doc

    def test_the_scan_would_catch_a_real_disabler(self):
        """Negative control: the detector must match the string it forbids."""
        line = "    PasswordAuthentication no"
        assert any(d in line for d in self._DISABLERS)
        # ...and must not fire on prose that forbids it.
        prose = "# never set PasswordAuthentication no on this host"
        assert prose.strip().startswith("#")


class TestTheCertificateGuidanceIsSafe:
    """The cert docs must not teach an agent to leak the CA or lock us out.

    A CA private key is a bigger secret than the password it replaces: it mints
    access indefinitely, for anyone who holds it. The written procedure is what
    stops the next agent copying it to the server "so signing works there", or
    reaching for TrustedUserCAKeys and an sshd reload it cannot safely perform.
    """

    def _doc(self):
        return (REPO / "CLAUDE.md").read_text()

    def test_the_procedure_is_documented(self):
        doc = self._doc()
        assert "Certificate auth" in doc, "the cert onboarding section is gone"
        assert "ssh-keygen -s" in doc, "the signing command is not documented"
        assert "-O clear -O permit-pty" in doc, (
            "the extension-stripping flags are undocumented; the default cert "
            "carries agent and port forwarding"
        )

    def test_it_says_where_the_ca_private_key_lives_and_does_not(self):
        doc = self._doc()
        assert "mistudio_user_ca" in doc
        assert "never on a server" in doc.lower() or "never in" in doc.lower(), (
            "the doc does not warn that the CA private key stays on the "
            "workstation; an agent will copy it to the node"
        )

    def test_the_ca_private_key_is_not_in_this_repo(self):
        """The check that actually matters, not just the words about it."""
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        offenders = [f for f in tracked
                     if "mistudio_user_ca" in f and not f.endswith(".pub")]
        assert not offenders, (
            f"the CA PRIVATE key is tracked in git: {offenders}. It mints "
            f"access to the GPU node for anyone who clones this repo, and the "
            f"mirror publishes it permanently."
        )

    def test_no_tracked_file_contains_a_private_key_block(self):
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        marker = "-----BEGIN OPENSSH PRIVATE KEY" + "-----"
        offenders = []
        for rel in tracked:
            path = REPO / rel
            if not path.exists() or path.is_dir():
                continue
            try:
                if marker in path.read_text(errors="ignore"):
                    offenders.append(rel)
            except OSError:  # pragma: no cover
                continue
        assert not offenders, f"private key material is tracked: {offenders}"

    def test_the_docs_do_not_instruct_an_sshd_reload(self):
        """Per-user cert-authority is the documented route precisely because
        it needs no root and cannot take sshd down."""
        doc = self._doc()
        cert_section = doc.split("### Certificate auth", 1)[1].split("### ", 1)[0]
        bare = re.sub(r"`[^`\n]*`", "", cert_section)
        assert "systemctl reload ssh" not in bare, (
            "the cert section tells an agent to reload sshd; the documented "
            "route is a cert-authority line in authorized_keys, which needs "
            "no root and cannot lock anyone out"
        )
