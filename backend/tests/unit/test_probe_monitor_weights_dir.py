"""Which directory a probe run loads its model weights from (032 FR-5).

⚠ WHY THIS FILE EXISTS. The first Stage 1 acceptance run died at model load with

    OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
    '/data/models/quantized/m_40e78d80_FP16'

`_load_model_for_run` read `model_row.quantized_path or model_row.file_path`, and
`quantized_path` is set by `model_tasks` for EVERY format other than FP32 while nothing
ever writes that directory — quantization is applied at load time by bitsandbytes, from
the raw checkpoint. So the column is a phantom on every row on this estate, and
`from_pretrained` turned the missing directory into a hub-repo-id complaint that names
neither the model nor the real problem.

Every other consumer — `extraction_service`, `analysis_service`,
`circuit_capture_service`, `circuit_attribution_service` — reads `file_path` alone, and
`logit_lens_service` checks `quantized_path` for existence first. The decision is now a
pure function so it can be tested without a GPU, a model or a database, and the call is
asserted by walking the AST rather than by searching the source text: a substring search
matches the comment that describes the call, which this repo has shipped five times.

MUTATION CONTROLS (each verified to fail this file):
  M130  back to `quantized_path or file_path`        → the fallback test
  M131  `resolved.exists()` → `True`                 → the fallback test
  M132  the refusal replaced by `return value`       → the refusal test
  M133  the call in `_load_model_for_run` removed    → the AST wiring test
"""
import ast
import inspect

import pytest

from src.services import probe_monitor_run
from src.services.probe_monitor_run import resolve_weights_dir


@pytest.fixture()
def raw(tmp_path):
    """A directory standing in for the HuggingFace cache root `file_path` points at."""
    directory = tmp_path / "raw" / "m_test"
    (directory / "models--org--name" / "snapshots" / "abc").mkdir(parents=True)
    return directory


class TestThePhantomQuantizedPathIsSkipped:
    def test_a_quantized_path_that_does_not_exist_falls_back_to_file_path(self, raw, tmp_path):
        """The exact shape of the row that broke Stage 1: FP16, so `quantized_path` is
        set, and the directory was never written."""
        chosen = resolve_weights_dir(
            "m_test",
            file_path=str(raw),
            quantized_path=str(tmp_path / "quantized" / "m_test_FP16"),
        )
        assert chosen == str(raw)

    def test_a_quantized_path_that_exists_wins(self, raw, tmp_path):
        """The `logit_lens_service` precedent: a real quantized checkpoint on disk is
        what the served model would read, so it is preferred when present."""
        quantized = tmp_path / "quantized" / "m_test_Q4"
        quantized.mkdir(parents=True)
        chosen = resolve_weights_dir(
            "m_test", file_path=str(raw), quantized_path=str(quantized)
        )
        assert chosen == str(quantized)

    def test_file_path_alone_is_enough(self, raw):
        assert resolve_weights_dir("m_test", file_path=str(raw), quantized_path=None) == str(raw)


class TestARefusalNamesWhatItLookedAt:
    """The point of the refusal is that the next reader is not debugging a hub error."""

    def test_neither_path_exists(self, tmp_path):
        with pytest.raises(ValueError) as caught:
            resolve_weights_dir(
                "m_test",
                file_path=str(tmp_path / "gone"),
                quantized_path=str(tmp_path / "also-gone"),
            )
        message = str(caught.value)
        assert "m_test" in message
        assert "file_path" in message and "quantized_path" in message
        assert "re-download" in message

    def test_both_unset(self):
        with pytest.raises(ValueError) as caught:
            resolve_weights_dir("m_test", file_path=None, quantized_path=None)
        assert "unset" in str(caught.value)

    def test_it_never_returns_a_path_that_does_not_exist(self, tmp_path):
        """The defect in one sentence: a non-existent path handed to `from_pretrained`
        becomes a repo id, not a missing-file error."""
        with pytest.raises(ValueError):
            resolve_weights_dir("m_test", file_path=str(tmp_path / "nope"), quantized_path=None)


class TestTheLoaderActuallyCallsIt:
    """A pure function nothing calls is the failure mode this repo has shipped three
    times. Asserted by AST, over the function's own body, so a mention in a docstring
    or a comment cannot satisfy it."""

    def _calls_in(self, function):
        tree = ast.parse(inspect.getsource(function).lstrip())
        return {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_load_model_for_run_calls_resolve_weights_dir(self):
        assert "resolve_weights_dir" in self._calls_in(probe_monitor_run._load_model_for_run)

    def test_the_ast_walk_can_tell_a_call_from_a_mention(self):
        """The negative control for the walk itself: a function that only NAMES the
        helper in a string and a comment must not register as calling it."""

        def decoy():
            # resolve_weights_dir(1, 2, 3)
            return "resolve_weights_dir"

        assert "resolve_weights_dir" not in self._calls_in(decoy)

    def test_the_loader_does_not_read_quantized_path_directly_any_more(self):
        """The `or` expression is what broke: if it comes back, the helper is bypassed
        and every test above passes while the run still dies."""
        source = inspect.getsource(probe_monitor_run._load_model_for_run)
        tree = ast.parse(source.lstrip())
        bare_or = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and any(
                isinstance(value, ast.Attribute) and value.attr == "quantized_path"
                for value in node.values
            )
        ]
        assert not bare_or, "the `quantized_path or file_path` expression is back"
