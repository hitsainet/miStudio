"""MIS-E2E-087, -104, -045, -039, -096: five defects that go quiet rather than loud."""

import ast
import re
from pathlib import Path

import pytest
import torch.nn as nn

SRC = Path(__file__).resolve().parents[2] / "src"
VERSIONS = Path(__file__).resolve().parents[2] / "alembic" / "versions"


class TestLayerDiscoveryHonoursPatternOrder:
    """MIS-E2E-087: `dir()` is alphabetical, so the preference list was discarded."""

    def test_the_preferred_norm_wins_over_an_alphabetically_earlier_one(self):
        from src.ml.layer_discovery import LAYER_NORM_PATTERNS, _find_matching_attr

        class Llamaish(nn.Module):
            def __init__(self):
                super().__init__()
                # `input_layernorm` sorts before `post_attention_layernorm`,
                # which is exactly how the wrong one used to win.
                self.input_layernorm = nn.LayerNorm(4)
                self.post_attention_layernorm = nn.LayerNorm(4)

        got = _find_matching_attr(Llamaish(), LAYER_NORM_PATTERNS)
        assert got == "post_attention_layernorm", (
            f"got {got!r}. LAYER_NORM_PATTERNS lists post-attention norms first "
            f"because that is the residual-stream point; returning the "
            f"pre-attention norm hooks the wrong tensor."
        )

    def test_the_fixture_would_expose_the_old_behaviour(self):
        """Guard the guard: the trap only exists if the names sort that way."""
        assert "input_layernorm" < "post_attention_layernorm"

    def test_it_agrees_with_is_transformer_layer(self):
        """The two used to disagree about the same model."""
        from src.ml.layer_discovery import LAYER_NORM_PATTERNS, _find_matching_attr

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(4)
                self.post_attention_layernorm = nn.LayerNorm(4)
                self.self_attn = nn.Linear(4, 4)
                self.mlp = nn.Linear(4, 4)

        picked = _find_matching_attr(Layer(), LAYER_NORM_PATTERNS)
        earliest = next(p for p in LAYER_NORM_PATTERNS
                        if hasattr(Layer(), p))
        assert picked == earliest

    def test_a_missing_preferred_norm_falls_through(self):
        from src.ml.layer_discovery import LAYER_NORM_PATTERNS, _find_matching_attr

        class OnlyPre(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(4)

        assert _find_matching_attr(OnlyPre(), LAYER_NORM_PATTERNS) == "input_layernorm"


class TestTheCeleryIdExistsBeforeTheTaskDoes:
    """MIS-E2E-104: `delay()` then write the id leaves a window with no record."""

    def test_the_endpoint_does_not_dispatch_before_persisting(self):
        source = (SRC / "api" / "v1" / "endpoints" / "trainings.py").read_text()
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_training")

        persist_line = dispatch_line = None
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", "")
            if name == "start_training":
                persist_line = node.lineno
            elif name in ("delay", "apply_async"):
                dispatch_line = node.lineno

        assert persist_line and dispatch_line, "could not find both calls"
        assert persist_line < dispatch_line, (
            "the task is dispatched before its id is persisted; a failure in "
            "that window leaves a GPU training running with nothing recording "
            "which Celery task it is, so no janitor can revoke it"
        )

    def test_the_id_is_generated_not_read_back(self):
        source = (SRC / "api" / "v1" / "endpoints" / "trainings.py").read_text()
        assert "apply_async" in source and "task_id=task_id" in source, (
            "`delay()` returns the id only AFTER queueing; a caller-supplied "
            "`task_id` is what lets it be persisted first"
        )


class TestSeedMigrationsDoNotClobberUserTemplates:
    """MIS-E2E-045: matched on name alone, with a `pass` downgrade."""

    SEEDS = ("n1o2p3q4r5s6_improve_default_labeling_template.py",
             "o2p3q4r5s6t7_fix_nextgen_labeling_template.py")

    @pytest.mark.parametrize("filename", SEEDS)
    def test_the_update_is_guarded_on_is_system(self, filename):
        text = (VERSIONS / filename).read_text()
        assert "UPDATE labeling_prompt_templates" in text, "the seed UPDATE is gone"
        assert "is_system = true" in text, (
            f"{filename} overwrites templates matched on name alone. A user "
            f"template sharing that name loses its edits, and downgrade() is a "
            f"no-op so they do not come back."
        )


class TestADowngradeThatCannotIdentifyItsRowsRefuses:
    """MIS-E2E-039: it deleted every tokenization for the matched datasets."""

    MIGRATION = "2e1feb9cc451_migrate_existing_tokenizations_to_new_.py"

    def test_the_downgrade_no_longer_deletes(self):
        text = (VERSIONS / self.MIGRATION).read_text()
        tree = ast.parse(text)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "downgrade")
        # Strip the FUNCTION's docstring: the refusal explains itself by
        # quoting the DELETE it replaced, and a substring check cannot tell a
        # quotation from a statement. Parsing to the function was not enough —
        # the docstring lives inside it. (Ninth time in this remediation.)
        statements = [n for n in fn.body
                      if not (isinstance(n, ast.Expr)
                              and isinstance(n.value, ast.Constant)
                              and isinstance(n.value.value, str))]
        body = "\n".join(ast.get_source_segment(text, n) or "" for n in statements)
        assert "DELETE FROM dataset_tokenizations" not in body, (
            "the downgrade still deletes rows it cannot prove it created"
        )

    def test_the_downgrade_refuses_loudly(self):
        text = (VERSIONS / self.MIGRATION).read_text()
        tree = ast.parse(text)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "downgrade")
        raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        assert raises, (
            "a downgrade that cannot identify its own rows must refuse, not "
            "silently do nothing and not guess"
        )


class TestLivenessAndDiagnostics:
    """MIS-E2E-096."""

    def test_every_jlens_fit_progress_report_carries_a_heartbeat(self):
        path = SRC / "workers" / "jlens_fit_tasks.py"
        tree = ast.parse(path.read_text())
        bare = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "update_state"):
                continue
            for kw in node.keywords:
                if kw.arg != "meta":
                    continue
                # meta must be built by beat(...), not a bare dict literal.
                if not (isinstance(kw.value, ast.Call)
                        and getattr(kw.value.func, "id", "") == "beat"):
                    bare.append(node.lineno)
        assert not bare, (
            f"update_state at lines {bare} passes meta without beat(). A meta "
            f"dict REPLACES the previous one, so this erases the liveness "
            f"timestamp and the janitor reaps a fit that is still running."
        )

    def test_the_stuck_job_message_reports_the_status_it_was_stuck_in(self):
        path = SRC / "workers" / "cleanup_stuck_enhanced_labeling.py"
        text = path.read_text()
        assert 'f"Job stuck in {job.status}' not in text, (
            "the message interpolates job.status AFTER it is set to FAILED, so "
            "every diagnostic reads 'stuck in FAILED' and the real status is lost"
        )
        assert "stuck_in = job.status" in text, "the original status is not captured"
        assert 'f"Job stuck in {stuck_in}' in text
