"""A batch's next extraction must not wait on the previous job's NLP.

OBSERVED IN PRODUCTION
----------------------
A 3-SAE batch: job 1 finished extracting at 15:56 and jobs 2 and 3 sat QUEUED
with celery_task_id NULL — never dispatched. The chain only advanced from the
NLP-completion path, and job 1's NLP was still running.

Measured NLP rate: 0.72 features/sec.
  32,759 features  -> ~12.6 hours before job 2 could start
  3 SAEs           -> ~38 hours for the batch

NLP is analysis of features ALREADY written to the database. The next
extraction does not depend on it, so it must not gate it.

IDEMPOTENCE MATTERS NOW: several paths call the chain (extraction complete,
extraction failed, NLP complete, NLP-queue failure). Without a claim, two of
them would both find the same QUEUED row and dispatch the SAME extraction
twice — wasting hours of GPU and corrupting the job's feature rows.

MUTATION CONTROLS:
  * gate the chain call on auto_nlp again -> the release tests fail
  * drop `celery_task_id.is_(None)` from the claim -> the double-dispatch test fails
"""

import inspect

import pytest


class TestChainIsReleasedOnExtractionNotNlp:
    def test_the_chain_call_precedes_the_nlp_queue(self):
        """Ordering is the fix: release the batch, THEN queue NLP."""
        from src.services import extraction_service

        src = inspect.getsource(extraction_service)
        marker = "RELEASE THE BATCH FIRST"
        assert marker in src, "the batch release comment/step is gone"

        completed_block = src.split("event=\"extraction:completed\"")[1]
        chain_at = completed_block.find("_start_next_batch_job")
        nlp_at = completed_block.find("analyze_features_nlp_task.delay")
        assert chain_at != -1, "the completion path no longer starts the next job"
        assert nlp_at != -1, "the completion path no longer queues NLP"
        assert chain_at < nlp_at, (
            "NLP is queued before the batch is released — the next extraction "
            "can again end up waiting on a multi-hour NLP pass"
        )

    def test_no_auto_nlp_condition_guards_the_chain_call(self):
        """The release must be unconditional w.r.t. NLP.

        Checked by walking the AST rather than matching text: the earlier
        substring assertion (`"if auto_nlp:" not in ...`) survived a mutation
        that wrote `if not auto_nlp and extraction_job.batch_id:` on the chain
        line itself — the exact defect being fixed.
        """
        import ast

        from src.services import extraction_service

        tree = ast.parse(inspect.getsource(extraction_service))

        offenders = []

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.guards = []

            def visit_If(self, node):
                names = {
                    n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
                }
                self.guards.append(names)
                for child in node.body:
                    self.visit(child)
                self.guards.pop()
                # orelse is NOT guarded by the same condition being true, but a
                # chain call in `else: # auto_nlp false` is equally a gate.
                self.guards.append(names)
                for child in node.orelse:
                    self.visit(child)
                self.guards.pop()

            def visit_Call(self, node):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "_start_next_batch_job":
                    for names in self.guards:
                        if "auto_nlp" in names:
                            offenders.append(node.lineno)
                self.generic_visit(node)

        Walk().visit(tree)

        assert not offenders, (
            f"_start_next_batch_job at line(s) {offenders} is guarded by an "
            "auto_nlp condition — the next extraction waits on NLP again"
        )

class TestChainClaimIsIdempotent:
    def test_the_claim_requires_an_undispatched_job(self):
        """celery_task_id IS NULL is the claim flag."""
        from src.workers import nlp_analysis_tasks

        src = inspect.getsource(nlp_analysis_tasks._start_next_batch_job)
        assert "celery_task_id.is_(None)" in src, (
            "the claim no longer excludes already-dispatched jobs — two callers "
            "would dispatch the same extraction twice"
        )
        assert "with_for_update" in src, "the claim is not row-locked"

    def test_double_invocation_dispatches_only_once(self):
        """Simulates two callers racing to advance the same batch."""
        from src.workers import nlp_analysis_tasks as nat

        class Job:
            def __init__(self):
                self.id = "extr_002"
                self.batch_id = "batch_1"
                self.batch_position = 2
                self.batch_total = 3
                self.external_sae_id = "sae_x"
                self.config = {}
                self.status = "queued"
                self.celery_task_id = None

        nxt = Job()

        class Q:
            def __init__(self, rows): self._rows = rows
            def filter(self, *a): return self
            def with_for_update(self, **kw): return self
            # MIS-E2E-066: the real query now orders by `batch_position`, because
            # it takes the NEXT queued job rather than demanding an exact
            # `position + 1` — a skipped SAE leaves a gap and stranded the tail.
            # A fake that cannot express the real query cannot test it.
            def order_by(self, *a): return self
            def first(self):
                # Mirror the real predicate: only an undispatched job qualifies,
                # lowest position first.
                candidates = [r for r in self._rows if r.celery_task_id is None]
                candidates.sort(key=lambda r: getattr(r, "batch_position", 0))
                return candidates[0] if candidates else None

        class DB:
            def __init__(self, rows): self._rows = rows; self.commits = 0
            def query(self, *a): return Q(self._rows)
            def commit(self): self.commits += 1

        db = DB([nxt])
        current = Job(); current.batch_position = 1

        dispatched = []

        class FakeTask:
            def apply_async(self, args=None, **kw):
                dispatched.append(args)
                return type("R", (), {"id": f"task_{len(dispatched)}"})()

        import src.workers.extraction_tasks as et
        original = et.extract_features_from_sae_task
        et.extract_features_from_sae_task = FakeTask()
        try:
            nat._start_next_batch_job(db, current)
            nat._start_next_batch_job(db, current)   # racing second caller
        finally:
            et.extract_features_from_sae_task = original

        assert len(dispatched) == 1, (
            f"the same extraction was dispatched {len(dispatched)} times — "
            "hours of duplicated GPU work"
        )
        assert nxt.celery_task_id == "task_1"
