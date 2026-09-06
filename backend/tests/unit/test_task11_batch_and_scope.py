"""Task 11, second batch — silent no-ops and cross-boundary writes.

MIS-E2E-059  `job_batch_size` bound only inside `if template:` and read
             unconditionally at three sites: the explicitly-supported
             "no template" path died with UnboundLocalError.
MIS-E2E-060  per-job `max_tokens` was overwritten by the template's, so a
             control the UI exposes did nothing.
MIS-E2E-058  a deliberate cancellation was reported as a failure — under a
             comment asserting the opposite.
MIS-E2E-066  batch extraction dispatched on the loop index, so a skipped first
             SAE dispatched NOTHING and a skipped middle one stranded the tail.
MIS-E2E-109  NLP analysis wrote across extraction boundaries.
MIS-E2E-108  template import overwrote protected system templates.
"""

import ast
import inspect

import pytest


def _fn_source(module, name: str) -> str:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found in {module.__name__}")


# ── MIS-E2E-059 · the unbound local ────────────────────────────────────────

def test_job_batch_size_is_bound_on_every_path():
    """The no-template path is supported and used to raise UnboundLocalError."""
    from src.services import labeling_service

    body = _fn_source(labeling_service.LabelingService, "label_features_for_extraction")
    tree = ast.parse(body)

    assigns, reads = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "job_batch_size":
            (assigns if isinstance(node.ctx, ast.Store) else reads).append(node.lineno)
    assert assigns and reads, "job_batch_size no longer appears — scan broke"

    # Every read must come after an assignment that is NOT nested in a branch
    # the read can skip. Approximated by column: a top-of-function binding sits
    # at a shallower indent than one inside `if template:`.
    lines = body.splitlines()
    unconditional = [
        ln for ln in assigns
        if len(lines[ln - 1]) - len(lines[ln - 1].lstrip()) <= 12
    ]
    assert unconditional, (
        "job_batch_size is only assigned inside a conditional branch, so the "
        "no-template path reaches its reads unbound"
    )
    assert min(unconditional) < min(reads)


# ── MIS-E2E-060 · the control that did nothing ─────────────────────────────

def test_the_jobs_max_tokens_wins_over_the_templates():
    """`max_tokens` is a per-job setting in the API and the UI.

    It was then unconditionally replaced by the template's (default 50), so a
    user raising it had the value accepted and every description still
    truncated. `max_examples` already had this precedence right.
    """
    from src.services import labeling_service

    src = inspect.getsource(labeling_service)
    assert "if labeling_job.max_tokens:" in src, (
        "the template still overwrites the job's max_tokens unconditionally"
    )
    # Both sites, not one — the assignment appeared twice.
    assert src.count("if labeling_job.max_tokens:") == 2, (
        "only one of the two max_tokens sites was fixed"
    )


# ── MIS-E2E-058 · cancellation is not failure ──────────────────────────────

def test_cancellation_is_handled_before_the_generic_handler():
    from src.services import labeling_service

    # Compare handlers WITHIN THE SAME try, via AST. String offsets found an
    # inner `except Exception` several hundred lines earlier and compared
    # against the wrong thing entirely.
    body = _fn_source(labeling_service.LabelingService, "label_features_for_extraction")
    tree = ast.parse(body)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        names = []
        for h in node.handlers:
            if h.type is None:
                names.append("bare")
            else:
                names.append(ast.unparse(h.type))
        if any("_LabelingCancelled" in n for n in names):
            cancelled_at = next(
                i for i, n in enumerate(names) if "_LabelingCancelled" in n
            )
            generic = [i for i, n in enumerate(names) if n in ("Exception", "bare")]
            assert not generic or cancelled_at < min(generic), (
                f"handlers run in order {names}; the generic one catches "
                f"_LabelingCancelled first and sets status=FAILED, so a user's "
                f"own cancel is reported as a failure"
            )
            return
    raise AssertionError("no try/except handles _LabelingCancelled")


def test_the_cancellation_handler_writes_CANCELLED_not_FAILED():
    from src.services import labeling_service

    src = inspect.getsource(labeling_service)
    idx = src.index("except LabelingService._LabelingCancelled:")
    block = src[idx: idx + 1400]
    assert "LabelingStatus.CANCELLED.value" in block
    assert "LabelingStatus.FAILED.value" not in block


# ── MIS-E2E-066 · the batch dispatch ───────────────────────────────────────

def test_the_first_created_job_is_dispatched_not_loop_position_one():
    """A skipped first SAE meant no job had position 1, so NOTHING ran."""
    from src.services import extraction_service

    src = inspect.getsource(extraction_service)
    assert "if not created_jobs:" in src, (
        "dispatch still keys on the enumerate index, so a batch whose first "
        "SAE is skipped silently does nothing"
    )
    assert "if position == 1:" not in src


def test_the_batch_advances_past_a_gap():
    """A skipped middle SAE leaves a gap; `position + 1` found nothing and the
    tail sat QUEUED until the 3-hour reaper blamed a crashed worker."""
    from src.workers import nlp_analysis_tasks

    body = _fn_source(nlp_analysis_tasks, "_start_next_batch_job")
    assert "batch_position > current_position" in body, (
        "the advance demands an exact position + 1, so any gap strands the "
        "rest of the batch"
    )
    assert "order_by" in body, "without ordering, 'the next one' is arbitrary"


# ── MIS-E2E-109 · the extraction boundary ──────────────────────────────────

def test_nlp_analysis_binds_feature_ids_to_the_path_extraction():
    """The ids branch dropped the scope the no-ids branch applies."""
    from src.workers import nlp_analysis_tasks

    body = _fn_source(nlp_analysis_tasks, "analyze_features_nlp_task")
    tree = ast.parse(body)

    # Find the `if feature_ids:` branch and require the scope inside it.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "feature_ids"
        ):
            # The FILTER COMPARISON, not the identifier. My first version
            # searched for "extraction_job_id" anywhere in the branch and
            # passed against a version with no filter at all, because the log
            # message inside it names the variable. Control C112 caught that.
            # `node.body`, NOT `node` — walking the If includes its `orelse`,
            # so this found the OTHER branch's filter and passed against a
            # body with none. Two rounds of control C112 to notice.
            comparisons = [
                ast.unparse(cmp_node)
                for stmt in node.body
                for cmp_node in ast.walk(stmt)
                if isinstance(cmp_node, ast.Compare)
            ]
            assert any(
                "Feature.extraction_job_id" in c and "extraction_job_id" in c
                for c in comparisons
            ), (
                "features are selected by id with no extraction filter — a "
                "caller can overwrite another extraction's curated analysis "
                "while the progress counters land on this one. Comparisons "
                f"found: {comparisons}"
            )
            return
    raise AssertionError("the feature_ids branch was not found — scan broke")


def test_the_no_ids_branch_is_still_scoped():
    """Negative control: the branch that was already right must stay right."""
    from src.workers import nlp_analysis_tasks

    body = _fn_source(nlp_analysis_tasks, "analyze_features_nlp_task")
    assert body.count("Feature.extraction_job_id == extraction_job_id") >= 2


# ── MIS-E2E-108 · the protected template ───────────────────────────────────

def test_import_refuses_to_overwrite_a_system_template():
    """`update` and `delete` both refuse one; import had neither guard.

    It matched on NAME alone and replaced the prompt body — then promoted the
    row to default, so every later bulk-labeling run executed the imported
    instructions while the UI still showed the template as protected.
    """
    from src.services import labeling_prompt_template_service as svc

    # Scope to the function, not a fixed-size window — my first version used a
    # 2200-char slice and the explanatory comment alone pushed the write past
    # its end.
    body = _fn_source(svc.LabelingPromptTemplateService, "import_templates")
    assert "existing_template.is_system" in body, (
        "the import overwrite branch does not check is_system"
    )
    guard = body.index("existing_template.is_system")
    write = body.index("existing_template.system_message =")
    assert guard < write, "the guard runs after the overwrite"


def test_import_cannot_grant_is_system():
    """The inverse: an import must not promote itself to protected either."""
    from src.services import labeling_prompt_template_service as svc

    src = inspect.getsource(svc)
    assert 'existing_template.is_system = ' not in src
    assert 'template_data.get("is_system")' not in src


def test_the_sibling_guards_still_exist():
    """Negative control. This fix copies the rule from update/delete; if those
    lost it, the reference the fix is modelled on would be gone."""
    from src.services import labeling_prompt_template_service as svc

    src = inspect.getsource(svc)
    assert src.count("is_system") >= 3, (
        "the update/delete system-template guards appear to be missing"
    )
