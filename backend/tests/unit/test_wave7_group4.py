"""MIS-E2E-068, -111, -118: a fabricated number, a leaked URL, a noisy shutdown."""

import ast
import asyncio
import logging
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def code_only(path: Path) -> str:
    """Source with comments and docstrings removed.

    Every check below states what the code must NOT contain, and every fix in
    this repo explains itself by quoting the line it replaced. A substring
    check cannot tell a quotation from a statement — that trap has now
    accounted for ten findings-in-fixes across this remediation. Tokenizing is
    the reliable way to ask what the file actually *does*.
    """
    import io
    import tokenize

    out = []
    prev_type = tokenize.INDENT
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.COMMENT:
                continue
            # A string that is the whole statement is a docstring.
            if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                prev_type = tok.type
                continue
            out.append(tok.string)
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                prev_type = tok.type
    return " ".join(out)


class TestProgressReportsWhatExists:
    """MIS-E2E-068(1): a count of features that had not been written."""

    def test_features_extracted_is_not_derived_from_the_progress_bar(self):
        text = code_only(SRC / "services" / "extraction_service.py")
        assert "int ( latent_dim * progress )" not in text, (
            "features_extracted is computed from the sampling progress bar, "
            "not from any row. No feature record exists until the commit phase "
            "that follows this loop, so the UI showed a rising count of "
            "features that did not exist."
        )

    def test_the_sampling_emit_reports_zero_written(self):
        text = (SRC / "services" / "extraction_service.py").read_text()
        assert '"features_extracted": 0,' in text


class TestTheContextWindowComesFromTheModel:
    """MIS-E2E-068(2): a hardcoded 2048 that silently truncated."""

    def test_no_hardcoded_window_in_the_tokenize_call(self):
        text = code_only(SRC / "services" / "steering_service.py")
        assert "max_length = 2048 - params . max_new_tokens" not in text, (
            "the prompt window is a constant again; on a larger model it "
            "discards prompt the model could read, and at max_new_tokens=2048 "
            "the budget is zero so the prompt is cut to nothing"
        )

    def test_it_reads_max_position_embeddings(self):
        text = (SRC / "services" / "steering_service.py").read_text()
        assert "max_position_embeddings" in text

    def test_an_over_long_prompt_is_refused_not_truncated(self):
        """The behavioural intent: refuse, do not silently shorten."""
        text = (SRC / "services" / "steering_service.py").read_text()
        assert "Shorten the prompt or lower max_new_tokens" in text, (
            "a truncated prompt produces a confident answer to a question the "
            "user did not ask; it has to be an error, not a quiet edit"
        )


class TestTheSteeringHookDoesNotLogPerForwardPass:
    """MIS-E2E-068(3): logger.info inside the hook, plus two comprehensions."""

    def test_the_hook_log_is_debug_level(self):
        text = (SRC / "services" / "steering_service.py").read_text()
        assert "logger . info ( f\"[Steering Hook] FIRED" not in code_only(
            SRC / "services" / "steering_service.py"
        ), (
            "the per-forward-pass log is back at INFO"
        )
        assert '[Steering Hook] FIRED' in text, "the diagnostic is gone entirely"

    def test_it_is_guarded_so_the_fstring_is_not_built(self):
        text = (SRC / "services" / "steering_service.py").read_text()
        assert "logger.isEnabledFor(logging.DEBUG)" in text, (
            "without the guard the f-string and both list comprehensions are "
            "evaluated on every forward pass even when DEBUG is off"
        )


class TestTheEnhancedLabelingResponseWithholdsTheEndpoint:
    """MIS-E2E-111: the sibling schema withholds it; this one listed every column."""

    def test_endpoint_is_not_a_field(self):
        from src.schemas.enhanced_labeling import EnhancedLabelingJobResponse

        assert "endpoint" not in EnhancedLabelingJobResponse.model_fields, (
            "the configured LLM server URL is published to the browser again"
        )

    def test_the_useful_fields_are_still_there(self):
        """Withholding one field must not gut the response."""
        from src.schemas.enhanced_labeling import EnhancedLabelingJobResponse

        fields = EnhancedLabelingJobResponse.model_fields
        for expected in ("status", "model", "workers", "examples_total",
                         "examples_completed", "celery_task_id"):
            assert expected in fields, f"{expected} was dropped too"

    def test_the_sibling_still_withholds_it_too(self):
        """Both paths expose the same job shape; they must agree."""
        from src.schemas.labeling import LabelingStatusResponse

        assert "endpoint" not in LabelingStatusResponse.model_fields


class TestAGracefulShutdownIsQuiet:
    """MIS-E2E-118: `.exception()` on a cancelled task re-raises."""

    def test_the_callback_returns_early_for_a_cancelled_task(self):
        from src.mcp_server.health_gate import _retrieve_exception

        async def forever():
            await asyncio.sleep(3600)

        async def run():
            task = asyncio.get_running_loop().create_task(forever())
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # The defect: this raised CancelledError inside the callback,
            # which asyncio logs as `ERROR asyncio: Exception in callback`.
            _retrieve_exception(task)

        asyncio.run(run())

    def test_a_real_exception_is_still_retrieved(self):
        """Swallowing everything would reintroduce the warning it prevents."""
        from src.mcp_server.health_gate import _retrieve_exception

        async def boom():
            raise RuntimeError("probe failed")

        async def run():
            task = asyncio.get_running_loop().create_task(boom())
            try:
                await task
            except RuntimeError:
                pass
            _retrieve_exception(task)
            assert task.exception() is not None

        asyncio.run(run())

    def test_the_lambda_is_gone(self):
        text = (SRC / "mcp_server" / "health_gate.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "add_done_callback"):
                arg = node.args[0]
                assert not isinstance(arg, ast.Lambda), (
                    "add_done_callback got a lambda again; `lambda t: t.exception()` "
                    "re-raises CancelledError on every graceful shutdown"
                )
