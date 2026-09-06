"""The J-space endpoints must be reachable, not merely importable.

HOUSE RULE (CLAUDE.md): a capability is not shipped until a test FAILS when its
wiring is removed. This repo's cautionary case is the 16 `millm_circuit_*` MCP
tools — fully implemented, unit-tested and documented while never registered
with the server. Every test passed by importing the module directly, so the
suite was green and the docs said shipped while no caller could reach the
feature.

Asserting `from ... import jlens` succeeds would reproduce exactly that failure.
This asserts membership in the ASSEMBLED api_router instead.

NOTE on the accessor: `api_router.routes` holds `_IncludedRouter` wrappers in
this FastAPI version, not expanded routes, and mounting into a fresh app does
not expand them either — a naive `app.include_router(...)` check returns zero
paths for EVERY endpoint module, which reads as "jlens is broken" when nothing
is. Reach the sub-router through `original_router`.

MUTATION CONTROLS:
  * delete the include_router(jlens.router) line -> registration test fails
  * rename a route path                          -> path test fails
  * change POST to GET                           -> method test fails
"""

from pathlib import Path

import pytest

from src.api.v1.endpoints import jlens
from src.api.v1.router import api_router

#: Every J-space POST surface a user or an agent can reach.
#:
#: `/jlens/interventions` was ABSENT from this set for the whole arc that built
#: it. Deleting its `@router.post` decorator left the suite green: the only
#: mention of the path anywhere in the tests was an assertion on a MagicMock
#: client's URL string in the MCP file, which proves what a CALLER sends and
#: nothing about what the server serves — two views sharing one blind spot,
#: which is the anti-pattern this file exists to prevent.
EXPECTED = {
    "/jlens/readout",
    "/jlens/probe",
    "/jlens/interventions",
    "/jlens/token-check",
    "/jlens/acquire",
    "/jlens/acquire/preview",
    "/jlens/publish",
}


def _reachable_paths() -> set:
    """Every path reachable through the assembled api_router."""
    paths = set()
    for included in api_router.routes:
        origin = getattr(included, "original_router", None)
        if origin is None:
            continue
        for route in getattr(origin, "routes", []):
            path = getattr(route, "path", None)
            if path:
                paths.add(path)
    return paths


class TestEndpointsAreReachable:
    def test_the_jlens_router_is_registered(self):
        """Membership in the assembled router, not importability."""
        registered = any(
            getattr(inc, "original_router", None) is jlens.router
            for inc in api_router.routes
        )
        assert registered, (
            "jlens.router is not registered in api_router. The module imports "
            "fine and its unit tests pass — and no caller can reach it. This "
            "is the exact shape of the unregistered-MCP-tools defect."
        )

    def test_both_paths_are_reachable(self):
        missing = EXPECTED - _reachable_paths()
        assert not missing, f"unreachable jlens paths: {sorted(missing)}"

    def test_the_accessor_sees_other_modules_too(self):
        """Guards the test itself.

        If `_reachable_paths` silently returned nothing, every assertion above
        would pass vacuously on an empty set difference. Prove the accessor
        actually resolves routes.
        """
        paths = _reachable_paths()
        assert len(paths) > 50, (
            f"only {len(paths)} paths resolved — the accessor is broken, so the "
            "reachability assertions above prove nothing"
        )

    @pytest.mark.parametrize("path", sorted(EXPECTED))
    def test_paths_accept_post(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "POST" in route.methods, (
                    f"{path} does not accept POST; a readout carries a prompt "
                    "body and cannot be a GET"
                )
                return
        pytest.fail(f"{path} not defined on the jlens router")


class TestBoundBackendStillFailsLoudly:
    """Both backends are now bound, so the old 501 assertion no longer applies.

    It is replaced rather than deleted: the hazard it guarded did not go away
    when the endpoint was implemented, it MOVED. An unbound endpoint could
    fabricate an empty result; a bound one can return an empty result for a
    task that FAILED, which is the same lie with a 200 on it.
    """

    def test_the_polling_routes_exist_alongside_the_submitting_ones(self):
        """Queue-and-poll needs both halves. A POST that returns a task id
        nobody can poll is a capability with no way to collect its result."""
        paths = _reachable_paths()
        for path in ("/jlens/readout/{task_id}", "/jlens/probe/{task_id}"):
            assert path in paths, f"unreachable poll route: {path}"

    @pytest.mark.parametrize(
        "path", ["/jlens/readout/{task_id}", "/jlens/probe/{task_id}"]
    )
    def test_poll_routes_accept_get(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "GET" in route.methods
                return
        pytest.fail(f"{path} not defined on the jlens router")

    def test_a_failed_task_carries_its_reason_and_no_results(self):
        """The failure path must be distinguishable from an empty success.

        MUTATION CONTROL: return `scores=[]` on FAILURE instead of None and
        this fails — which is exactly the shape that would let a caller read a
        crashed probe as "this direction scores nowhere".
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        failed = MagicMock()
        failed.state = "FAILURE"
        failed.info = RuntimeError("expected scalar type BFloat16 but found Half")

        with patch("celery.result.AsyncResult", return_value=failed):
            result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                jlens.probe_result("t-1")
            )

        assert result.status == "FAILURE"
        assert result.scores is None, (
            "a failed probe returned a score list; an empty list is "
            "indistinguishable from a real probe that found nothing"
        )
        assert "BFloat16" in (result.error or ""), "the failure reason was dropped"

    def test_a_successful_probe_records_which_mode_produced_it(self):
        """Probe and full-ranking scores can disagree (BR-008), so a result
        that does not say which mode it used cannot be compared with one that
        does."""
        import asyncio
        from unittest.mock import MagicMock, patch

        done = MagicMock()
        done.state = "SUCCESS"
        done.result = {
            "scores": [{"layer": 3, "position": 0, "token": " Paris", "score": 1.5}],
            "mode": "probe",
            "lens_type": "LOGIT_LENS",
        }

        with patch("celery.result.AsyncResult", return_value=done):
            result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                jlens.probe_result("t-2")
            )

        assert result.mode == "probe"
        assert result.lens_type == "LOGIT_LENS"
        assert result.scores and result.scores[0].token == " Paris"


# ---------------------------------------------------------------------------
# Band-report computation must be REACHABLE
#
# `compute_band_report`, `save_band_report`, `decide_gate` and `save_gate` were
# fully implemented and unit-tested with ZERO production callers — verified by
# grep before this was written. The suite was green while no user or agent could
# produce a band report at all, so the panel's band rendering was permanently
# unreachable and `classify_behaviour` returned UNKNOWN forever. Same shape as
# the 16 MCP tools this repo once shipped registered with nothing.
#
# MUTATION CONTROLS (each must turn this section red):
#   * delete the POST /jlens/band-report route      -> "compute route" fails
#   * delete the POST /jlens/gate route             -> "gate route" fails
#   * drop jlens_band_tasks from celery `include`   -> "task is registered" fails
#   * make the endpoint stop calling .delay         -> "endpoint queues" fails
# ---------------------------------------------------------------------------


BAND_PATHS = {"/jlens/band-report", "/jlens/gate"}


class TestBandReportIsReachable:
    def test_the_compute_and_gate_routes_are_registered(self):
        missing = BAND_PATHS - _reachable_paths()
        assert not missing, (
            f"unreachable band paths: {sorted(missing)}. The band service was "
            "implemented and tested with no caller at all — a route is what "
            "makes it exist for anyone"
        )

    @pytest.mark.parametrize("path", sorted(BAND_PATHS))
    def test_band_paths_accept_post(self, path):
        for route in jlens.router.routes:
            if getattr(route, "path", None) == path:
                assert "POST" in route.methods
                return
        pytest.fail(f"{path} not defined on the jlens router")

    def test_the_band_tasks_are_registered_with_celery(self):
        """A task the worker never imports is a task nothing can run.

        Asserts the TASK NAME in the live registry, not that the module
        imports: `task_routes` globs match the task name, so a short or
        unregistered name lands on the default queue silently.
        """
        from src.core.celery_app import celery_app

        for name in (
            "src.workers.jlens_band_tasks.compute_band_report",
            "src.workers.jlens_band_tasks.record_gate",
        ):
            assert name in celery_app.tasks, (
                f"{name} is not in the live Celery registry; the endpoint would "
                "queue a task no worker can execute"
            )

    def test_the_band_tasks_route_to_a_worker_that_exists(self):
        """Routed to `extraction`, where the GPU worker actually listens."""
        from src.core.celery_app import celery_app

        routes = celery_app.conf.task_routes
        assert routes.get("src.workers.jlens_band_tasks.*", {}).get("queue") == (
            "extraction"
        ), "band tasks are not routed to the extraction queue"

    @pytest.mark.asyncio
    async def test_the_compute_endpoint_queues_with_the_arguments_it_was_given(self):
        """Payload AND call count — "was called" passes against wrong arguments."""
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.jlens import BandReportRequest

        db = MagicMock()
        db.execute = _async_returning(_scalar(object()))

        with patch(
            "src.workers.jlens_band_tasks.compute_band_report_task"
        ) as task:
            task.delay.return_value = MagicMock(id="t-band")
            accepted = await jlens.compute_band_report(
                BandReportRequest(
                    model_id="m_1",
                    prompts=["a", "b"],
                    control_seed=1234,
                    layers=[3, 4],
                    use_artifact=True,
                ),
                db=db,
            )

        assert task.delay.call_count == 1
        sent = task.delay.call_args.kwargs
        assert sent["model_id"] == "m_1"
        assert sent["prompts"] == ["a", "b"]
        # The seed must survive the trip: the autocorrelation null is drawn from
        # it, and a report whose control cannot be reproduced is not evidence.
        assert sent["control_seed"] == 1234
        assert sent["layers"] == [3, 4]
        assert accepted.task_id == "t-band"

    def test_no_band_boundary_can_be_SUPPLIED_through_the_api(self):
        """BR-002 by construction, not by discipline.

        Bands come from the model's own kurtosis profile or they do not exist
        for it. A request field that accepted boundaries would make porting the
        published Sonnet-4.5 numbers a one-line change.
        """
        from src.api.v1.endpoints.jlens import BandReportRequest

        fields = set(BandReportRequest.model_fields)
        for forbidden in ("boundaries", "workspace_start", "motor_start", "bands"):
            assert forbidden not in fields, (
                f"BandReportRequest accepts {forbidden!r}; boundaries measured "
                "on another model must be impossible to supply, not merely "
                "discouraged (BR-002)"
            )


def _scalar(value):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _async_returning(value):
    async def _call(*_args, **_kwargs):
        return value

    return _call


class TestRestoreSupersededIsReachable:
    """The recovery route for a lens that was published over.

    A service method nobody can call is the shape this repo has shipped before:
    16 MCP tools fully implemented, unit-tested, documented and never
    registered. `restore_superseded` exists to spare anyone a shell rename
    inside the pod, and it only does that if it is reachable over HTTP.

    MUTATION CONTROLS:
      * remove the @router.post decorator      -> the path test fails
      * change the path or the method to GET   -> the method test fails
    """

    PATH = "/jlens/artifacts/{slug}/restore-superseded"

    def test_the_restore_path_is_reachable_through_the_assembled_router(self):
        assert self.PATH in _reachable_paths(), (
            "restore-superseded is not reachable through api_router, so the "
            "only way to recover a displaced artifact is a shell rename inside "
            "the pod — which is what it was written to replace"
        )

    def test_it_accepts_POST_and_not_GET(self):
        """A restore MUTATES. A GET that swaps directories is a trap for any
        crawler, prefetcher or retry."""
        methods = set()
        for included in api_router.routes:
            origin = getattr(included, "original_router", None)
            if origin is None:
                continue
            for route in getattr(origin, "routes", []):
                if getattr(route, "path", None) == self.PATH:
                    methods |= set(getattr(route, "methods", set()))
        assert "POST" in methods, f"restore route methods: {methods or 'none'}"
        assert "GET" not in methods, (
            "a directory swap must not be reachable by GET"
        )


class TestCausalEvidenceIsReachable:
    """The read surface a serving runtime would use.

    Evidence written to disk that nothing can fetch is the same defect as a
    service method with no route: real, tested, and unreachable. This is the
    route miLLM will call once a lens has been pulled down from HuggingFace.

    MUTATION CONTROLS:
      * remove the @router.get decorator -> the path test fails
      * change GET to POST               -> the method test fails
    """

    PATH = "/jlens/artifacts/{slug}/interventions"

    def test_the_causal_path_is_reachable_through_the_assembled_router(self):
        assert self.PATH in _reachable_paths(), (
            "causal evidence is written beside the lens but nothing can read "
            "it back over HTTP"
        )

    def test_it_is_a_GET(self):
        """Reading demonstrated behaviour has no side effects."""
        methods = set()
        for included in api_router.routes:
            origin = getattr(included, "original_router", None)
            if origin is None:
                continue
            for route in getattr(origin, "routes", []):
                if getattr(route, "path", None) == self.PATH:
                    methods |= set(getattr(route, "methods", set()))
        assert "GET" in methods, f"causal route methods: {methods or 'none'}"


class TestAMalformedSwapIsRefusedSYNCHRONOUSLY:
    """Knowable at request time, so it must not cost a queue slot.

    The worker already refuses a swap with one token — but only after this
    endpoint has returned 202 with a task id. The caller is told the request was
    accepted, the job takes a slot on a single-GPU queue, and the refusal
    arrives a minute later behind a poll.

    OBSERVED IN PRODUCTION while probing what had deployed: a swap with one
    token came back `{"task_id": ...}` and failed a minute later. It also made
    the deploy probe read "not landed" for a guard that had landed, because the
    probe looked at the HTTP response and the guard lived a layer deeper.

    MUTATION CONTROLS:
      * drop the validator                     -> "refused before queueing" fails
      * allow target_token == direction_token  -> "the SAME token twice" fails
      * let dynamic_topk_ablation through      -> "unimplemented" fails
    """

    def _request(self, **over):
        from src.api.v1.endpoints.jlens import InterventionRequest

        body = dict(
            model_id="m_1",
            prompt="hello",
            primitive="coordinate_swap",
            layers=[9],
            direction_token=" dog",
        )
        body.update(over)
        return InterventionRequest(**body)

    def test_a_swap_with_one_token_is_refused_BEFORE_queueing(self):
        with pytest.raises(ValueError, match="TWO different tokens"):
            self._request()

    def test_a_swap_with_the_SAME_token_twice_is_refused(self):
        with pytest.raises(ValueError, match="TWO different tokens"):
            self._request(target_token=" dog")

    def test_a_swap_with_two_DIFFERENT_tokens_is_accepted(self):
        """The guard must not block the thing it exists to enable."""
        req = self._request(target_token=" cat")
        assert req.target_token == " cat"

    def test_an_unimplemented_primitive_is_refused_here_too(self):
        with pytest.raises(ValueError, match="not implemented"):
            self._request(primitive="dynamic_topk_ablation", target_token=" cat")

    def test_the_other_primitives_are_untouched(self):
        for primitive in ("additive", "projective_ablation"):
            assert self._request(primitive=primitive).primitive == primitive


class TestAnUnknownPrimitiveIsRefusedAtTheDoor:
    """A typo used to cost a slot on the single-GPU queue.

    `primitive` was a free `str`, so "aditive" passed the schema, returned 202
    with a task id, queued behind a possible 45-minute fit, and failed before its
    first progress report — landing in exactly the "queued 0%" state, reported
    with the janitor's prose about bookkeeping rather than the enum the worker
    had built.

    MUTATION CONTROL: widen `primitive` back to `str` and this fails.
    """

    def test_a_typo_is_refused(self):
        from src.api.v1.endpoints.jlens import InterventionRequest

        with pytest.raises(ValueError):
            InterventionRequest(
                model_id="m_1", prompt="x", primitive="aditive", layers=[1],
                direction_token=" dog",
            )

    def test_every_real_primitive_is_still_accepted_by_the_schema(self):
        """The enum must not exclude a primitive the system implements."""
        from src.api.v1.endpoints.jlens import InterventionRequest
        from src.services.jlens_intervention import Primitive

        for p in Primitive:
            kwargs = dict(
                model_id="m_1", prompt="x", primitive=p.value, layers=[1],
                direction_token=" dog",
            )
            if p is Primitive.COORDINATE_SWAP:
                kwargs["target_token"] = " cat"
            if p is Primitive.DYNAMIC_TOPK_ABLATION:
                # Accepted by the enum, refused by the validator with a reason —
                # a different thing from being unspellable.
                with pytest.raises(ValueError, match="not implemented"):
                    InterventionRequest(**kwargs)
                continue
            assert InterventionRequest(**kwargs).primitive == p.value

class TestTheRequestGUARDSAreRealAndNotDecorative:
    """Every rule added to `InterventionRequest` shipped with no test.

    Each one refuses a request that would otherwise take a slot on a single-GPU
    queue — behind a possible 45-minute fit — and fail, or worse, succeed while
    measuring something other than what the recipe records.

    MUTATION CONTROLS (each must redden the named test):
      * drop the duplicate-layers check  -> "repeated layers"
      * drop the layer-count cap         -> "too many layers"
      * drop the per-prompt length bound -> "an overlong trial prompt"
      * drop the empty-prompt check      -> "a blank trial prompt"
      * drop the positions check         -> "repeated positions"
    """

    @staticmethod
    def _body(**over):
        body = dict(
            model_id="m_1",
            prompt="hello",
            primitive="additive",
            layers=[0, 1],
            direction_token=" Paris",
        )
        body.update(over)
        return body

    def _refused(self, **over):
        from pydantic import ValidationError

        from src.api.v1.endpoints.jlens import InterventionRequest

        with pytest.raises(ValidationError) as exc:
            InterventionRequest(**self._body(**over))
        return str(exc.value)

    def test_the_BASELINE_request_is_accepted(self):
        """Or every assertion below passes for the wrong reason."""
        from src.api.v1.endpoints.jlens import InterventionRequest

        assert InterventionRequest(**self._body()).layers == [0, 1]

    def test_REPEATED_LAYERS_are_refused(self):
        """Each entry registers its own hook, and each hook perturbs the output
        of the one before — so [9,9,9] at strength 1.0 applies 3.0 and records
        the recipe as 1.0. Reproducing that recipe cannot reproduce the result.
        """
        assert "more than once" in self._refused(layers=[9, 9, 9])

    def test_TOO_MANY_LAYERS_are_refused(self):
        from src.api.v1.endpoints.jlens import MAX_INTERVENED_LAYERS

        msg = self._refused(layers=list(range(MAX_INTERVENED_LAYERS + 1)))
        assert str(MAX_INTERVENED_LAYERS) in msg
        # AND THE BOUND ITSELF IS ACCEPTED, so the comparison is not off by one.
        from src.api.v1.endpoints.jlens import InterventionRequest

        assert InterventionRequest(
            **self._body(layers=list(range(MAX_INTERVENED_LAYERS)))
        )

    def test_an_OVERLONG_trial_prompt_is_refused(self):
        """`prompt` was capped at 8000 and `prompts` entries at nothing, so
        512 x 400 000 characters passed validation from one POST."""
        from src.api.v1.endpoints.jlens import MAX_PROMPT_CHARS

        msg = self._refused(prompts=["ok", "x" * (MAX_PROMPT_CHARS + 1)])
        assert "prompts[1]" in msg
        assert str(MAX_PROMPT_CHARS) in msg

    def test_a_prompt_AT_the_bound_is_accepted(self):
        from src.api.v1.endpoints.jlens import (
            MAX_PROMPT_CHARS,
            InterventionRequest,
        )

        assert InterventionRequest(**self._body(prompts=["x" * MAX_PROMPT_CHARS]))

    def test_a_BLANK_trial_prompt_is_refused(self):
        """A whitespace-only trial scores three forward passes over nothing and
        contributes its result to the rate as though it were an observation."""
        assert "prompts[0]" in self._refused(prompts=["   ", "real"])

    def test_REPEATED_POSITIONS_are_refused(self):
        """The hook loops over them and writes into the tensor it is reading."""
        assert "positions repeat" in self._refused(positions=[-1, -1])

class TestATypedTokenCanBeCheckedWithoutLoadingWeights:
    """A direction is `W_U[id]`, so ANY single token has one.

    Restricting the UI to tokens the readout surfaced was a limit the server
    never had — and the interesting swap targets are exactly the ones NOT on
    screen yet ("does ' Rome' arrive if I put ' Paris' where it was?").

    But whether a string is one token belongs to the model's vocabulary, not to
    the string. The worker refuses a multi-token direction correctly, and only
    after a 202 and a slot on a single-GPU queue that may sit behind a
    45-minute fit.

    MUTATION CONTROLS:
      * return usable=True for n != 1     -> "two tokens are refused" fails
      * drop the leading-space hint       -> "suggests the leading space" fails
      * load the model instead of the tokenizer -> "loads NO weights" fails
    """

    @staticmethod
    def _check(tokens, encoder):
        """Drive the endpoint with a stub tokenizer and a stub model row."""
        import asyncio
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.jlens import TokenCheckRequest, token_check

        tok = MagicMock()
        tok.encode = lambda s, add_special_tokens=False: encoder(s)

        db = MagicMock()
        res = MagicMock()
        res.scalar_one_or_none.return_value = object()

        async def execute(_q):
            return res

        db.execute = execute

        with patch(
            "src.services.jlens_model_registry.tokenizer_for", return_value=tok
        ) as loader, patch(
            "src.services.jlens_model_registry.load_for_readout"
        ) as weights:
            out = asyncio.run(
                token_check(
                    TokenCheckRequest(model_id="m_1", tokens=list(tokens)), db=db
                )
            )
        return out, loader, weights

    def test_a_SINGLE_token_is_usable(self):
        out, _l, _w = self._check([" Rome"], lambda s: [4874])
        assert out[0].usable is True
        assert out[0].ids == [4874]
        assert out[0].n_tokens == 1

    def test_TWO_tokens_are_refused_with_the_ids_shown(self):
        """"[4874, 883]" is what makes "this is two tokens" concrete."""
        out, _l, _w = self._check(["Rome"], lambda s: [4874, 883] if s == "Rome" else [1])
        assert out[0].usable is False
        assert out[0].ids == [4874, 883]
        assert "SINGLE token" in out[0].detail

    def test_it_SUGGESTS_THE_LEADING_SPACE_when_that_is_the_fix(self):
        """The cause almost every time, and invisible in a text box."""
        out, _l, _w = self._check(
            ["Rome"], lambda s: [4874] if s == " Rome" else [4874, 883]
        )
        assert "leading space" in out[0].detail, out[0].detail

    def test_it_does_NOT_suggest_a_space_that_does_not_help(self):
        """Otherwise the hint is noise on every failure and stops being read."""
        out, _l, _w = self._check(["zzz"], lambda s: [1, 2, 3])
        assert "leading space" not in out[0].detail

    def test_a_token_that_encodes_to_NOTHING_is_refused(self):
        out, _l, _w = self._check([" "], lambda s: [])
        assert out[0].usable is False
        assert out[0].n_tokens == 0

    def test_it_loads_NO_WEIGHTS(self):
        """A single-token check must not cost a model load on a single-GPU box.

        MUTATION CONTROL: call `load_for_readout` here and this fails.
        """
        _out, loader, weights = self._check([" Rome"], lambda s: [1])
        assert loader.called, "the tokenizer was never consulted"
        assert not weights.called, (
            "the check loaded the model's WEIGHTS; a tokenizer is a few "
            "megabytes of JSON and this runs on every keystroke's blur"
        )

class TestTheTokenizerIsFOUNDTheSameWayTheWeightsAre:
    """Every check above stubs `tokenizer_for`, so none exercised THIS.

    The first version read `name` — the DISPLAY name, "LFM2.5-1.2B-Instruct" —
    where `load_for_readout` reads `repo_id`, "LiquidAI/LFM2.5-1.2B-Instruct".
    `from_pretrained` then looked for a snapshot that does not exist and the
    endpoint 500'd on every real call, with a green suite: the tests mocked the
    one function whose job was to get this right.

    Two views of the same fact drifted because each derived it separately, which
    is why `locate_weights` is now the single definition.

    MUTATION CONTROLS:
      * read `name` instead of `repo_id`  -> "uses the repo id" fails
      * return the row's id on a missing repo_id -> "refuses without one" fails
      * drop the exists() check           -> "refuses when not downloaded" fails
    """

    @staticmethod
    def _row(**over):
        import types

        base = dict(
            id="m_1",
            repo_id="LiquidAI/LFM2.5-1.2B-Instruct",
            name="LFM2.5-1.2B-Instruct",
            file_path="/models/raw/m_1",
        )
        base.update(over)
        return types.SimpleNamespace(**base)

    def test_it_uses_the_REPO_ID_not_the_display_name(self, tmp_path):
        from unittest.mock import patch

        from src.services.jlens_model_registry import locate_weights

        with patch("src.services.jlens_model_registry.settings") as st:
            st.resolve_data_path.return_value = tmp_path
            repo_id, resolved = locate_weights(self._row())

        assert repo_id == "LiquidAI/LFM2.5-1.2B-Instruct", (
            "the display name was used; from_pretrained cannot resolve it and "
            "the endpoint 500s on every real call"
        )
        assert "/" in repo_id, "a repo id has an owner; a display name does not"
        assert resolved == tmp_path

    def test_it_REFUSES_a_row_without_a_repo_id(self):
        from src.services.jlens_model_registry import (
            ModelNotAvailable,
            locate_weights,
        )

        with pytest.raises(ModelNotAvailable, match="no repo_id"):
            locate_weights(self._row(repo_id=None))

    def test_it_REFUSES_when_the_weights_are_not_downloaded(self, tmp_path):
        from unittest.mock import patch

        from src.services.jlens_model_registry import (
            ModelNotAvailable,
            locate_weights,
        )

        with patch("src.services.jlens_model_registry.settings") as st:
            st.resolve_data_path.return_value = tmp_path / "absent"
            with pytest.raises(ModelNotAvailable, match="not downloaded"):
                locate_weights(self._row())

    def test_load_for_readout_reads_THE_SAME_FIELD(self):
        """The two derivations drifted once; this is what notices next time.

        Asserted against the SOURCE of the other path deliberately: the point is
        that both read `repo_id`, and a test that called `load_for_readout`
        would need a model on disk.
        """
        import inspect

        from src.services import jlens_model_registry as reg

        src = inspect.getsource(reg.load_for_readout)
        assert 'getattr(model_record, "repo_id"' in src
        # AND NOT the display name, which is the mistake being pinned.
        assert 'getattr(model_record, "name"' not in src

    def test_a_tokenizer_failure_is_ModelNotAvailable_not_a_bare_OSError(
        self, tmp_path
    ):
        """The endpoint turns that into a 409; an OSError escaped as a 500.

        MUTATION CONTROL: remove the try/except around `from_pretrained` and
        this fails with OSError.
        """
        from unittest.mock import patch

        from src.services.jlens_model_registry import (
            ModelNotAvailable,
            tokenizer_for,
        )

        with patch("src.services.jlens_model_registry.settings") as st, patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=OSError("no snapshot"),
        ):
            st.resolve_data_path.return_value = tmp_path
            with pytest.raises(ModelNotAvailable, match="no snapshot"):
                tokenizer_for(self._row(repo_id="org/never-cached"))

class TestAcquisitionIsWiredEndToEnd:
    """Registration in one list is not registration.

    `celery_app` holds FIVE ENUMERATED route entries and FIVE ENUMERATED
    autodiscover entries for J-space modules — there is no `jlens_*` glob. Get
    the task name right and miss the autodiscover entry, and the worker never
    imports the module: the task is absent from the registry and `.delay()`
    publishes a message nothing will ever consume. Get the autodiscover entry
    right and miss the route, and it runs on the default queue, which is
    `datasets` — no GPU, so the semantic check fails and nothing publishes.

    MUTATION CONTROLS:
      * drop the autodiscover entry -> "registered with celery"
      * drop the task_routes entry  -> "routes to extraction"
      * shorten the task name       -> both
    """

    TASK = "src.workers.jlens_acquire_tasks.acquire_jlens_artifact"

    def test_the_acquire_task_is_REGISTERED_with_celery(self):
        """From the live registry, not from an import in this test file.

        Importing the module here would make the assertion pass regardless of
        whether the WORKER ever imports it — which is the whole failure mode.
        """
        from src.core.celery_app import celery_app

        assert self.TASK in celery_app.tasks, (
            "the acquire task is not in the registry; a worker started from "
            "this configuration would never consume its messages"
        )

    def test_the_module_is_in_the_AUTODISCOVER_list(self):
        """The half that the registry check cannot distinguish on its own.

        This test file imports plenty; a task can be present because something
        else pulled the module in. Autodiscovery is what guarantees the WORKER
        does.
        """
        import inspect

        from src.core import celery_app as module

        source = inspect.getsource(module)
        assert '"src.workers.jlens_acquire_tasks"' in source, (
            "jlens_acquire_tasks is missing from autodiscover_tasks; the worker "
            "will not import it"
        )

    def test_the_acquire_task_ROUTES_TO_EXTRACTION(self):
        """The single-GPU queue, because the semantic check needs the model."""
        import fnmatch

        from src.core.celery_app import celery_app

        queue = None
        for pattern, config in (celery_app.conf.task_routes or {}).items():
            if fnmatch.fnmatch(self.TASK, pattern):
                queue = config.get("queue")
        assert queue == "extraction", (
            f"the acquire task routes to {queue!r}; the default queue has no GPU "
            "and the semantic check would fail there"
        )

    def test_it_owns_its_own_failure(self):
        """Auto-enrolled by the glob in test_task_heartbeat, asserted here too
        because a task that dies mid-download otherwise sits at 'running'."""
        from src.core.celery_app import celery_app
        from src.workers.jlens_progress import OWNERSHIP_MARKER

        task = celery_app.tasks[self.TASK]
        assert getattr(task.run, OWNERSHIP_MARKER, False) is True

    def test_ACQUIRE_is_a_declared_task_type(self):
        from src.workers import jlens_progress

        assert jlens_progress.ACQUIRE == "jlens_acquire"


class TestTheEndpointRefusesWhatIsKnowableWITHOUTTheGPU:
    """Three refusals that must happen before `.delay`.

    A 202 puts the job on the single-GPU queue, possibly behind a 45-minute
    fit, before it can discover it was doomed. Everything below is knowable
    from the request and the database.

    MUTATION CONTROLS:
      * move the weights check into the worker -> "model is not downloaded"
      * drop the disk guard                    -> "free disk"
      * skip the model lookup                  -> "unknown model"
    """

    @staticmethod
    def _call(record, monkeypatch, **over):
        import asyncio
        from unittest.mock import MagicMock, patch

        from src.api.v1.endpoints.jlens import AcquireRequest, acquire_artifact

        db = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = record

        async def execute(_q):
            return result

        db.execute = execute

        body = dict(
            model_id="m_1",
            repo_id="org/lenses",
            path_in_repo="a/b_jacobian_lens.pt",
        )
        body.update(over)

        task = MagicMock()
        task.delay.return_value = MagicMock(id="t-1")
        task.apply_async.return_value = MagicMock(id="t-1")
        with patch(
            "src.workers.jlens_acquire_tasks.acquire_jlens_artifact", task
        ), patch("src.workers.jlens_progress.open_row") as open_row:
            try:
                out = asyncio.run(
                    acquire_artifact(AcquireRequest(**body), db=db)
                )
            except Exception as exc:  # noqa: BLE001 - the refusal is the result
                return None, exc, task, open_row
        return out, None, task, open_row

    def test_an_UNKNOWN_MODEL_is_404_and_queues_nothing(self, monkeypatch):
        _out, exc, task, _row = self._call(None, monkeypatch)
        assert exc is not None and getattr(exc, "status_code", None) == 404
        assert task.delay.call_count == 0
        assert task.apply_async.call_count == 0

    def test_a_model_whose_WEIGHTS_ARE_ABSENT_is_409_and_queues_nothing(
        self, monkeypatch
    ):
        """A lens is unusable without its weights, and validating one MEANS
        reading out through it. Learning that after a 265 MB download is the
        expensive way.

        MUTATION CONTROL: move `locate_weights` into the worker and this fails.
        """
        import types
        from unittest.mock import patch

        from src.services.jlens_model_registry import ModelNotAvailable

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path=None)
        with patch(
            "src.services.jlens_model_registry.locate_weights",
            side_effect=ModelNotAvailable("org/m is not downloaded locally"),
        ):
            _out, exc, task, _row = self._call(record, monkeypatch)
        assert exc is not None and getattr(exc, "status_code", None) == 409
        assert "not downloaded" in str(getattr(exc, "detail", ""))
        assert task.delay.call_count == 0, "a doomed job took a GPU slot"
        assert task.apply_async.call_count == 0, "a doomed job took a GPU slot"

    def test_INSUFFICIENT_DISK_is_507_and_queues_nothing(self, monkeypatch):
        """No download path in this project checked disk. The data volume also
        holds every model, dataset and checkpoint."""
        import types
        from unittest.mock import patch

        from src.services.jlens_acquire_service import AcquisitionRefused

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch(
            "src.services.jlens_acquire_service.check_free_space",
            side_effect=AcquisitionRefused("0.2 GiB free"),
        ):
            _out, exc, task, _row = self._call(record, monkeypatch)
        assert exc is not None and getattr(exc, "status_code", None) == 507
        assert task.delay.call_count == 0
        assert task.apply_async.call_count == 0

    def test_a_GOOD_request_queues_with_the_arguments_it_was_given(
        self, monkeypatch
    ):
        """"Was called" passes against a call sending the wrong arguments."""
        import types
        from unittest.mock import patch

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch("src.services.jlens_acquire_service.check_free_space"):
            out, exc, task, open_row = self._call(
                record, monkeypatch, revision="abc123", allow_coverage_loss=True
            )
        assert exc is None, exc
        # QUEUED VIA apply_async, because the endpoint overrides `kwargsrepr` to
        # keep the HuggingFace token out of the message headers — Celery renders
        # that repr into every `task-sent` event on `celeryev`.
        assert task.apply_async.call_count == 1
        assert task.delay.call_count == 0
        sent = task.apply_async.call_args.kwargs["kwargs"]
        assert sent["repo_id"] == "org/lenses"
        assert sent["path_in_repo"] == "a/b_jacobian_lens.pt"
        assert sent["revision"] == "abc123"
        assert sent["allow_coverage_loss"] is True
        assert sent["allow_quality_regression"] is False
        assert out.task_id == "t-1"

    def test_OCCUPIED_STAGING_is_409_and_downloads_nothing(self, monkeypatch):
        """Found on hardware: leftover debris from an interrupted fit refused an
        acquisition — correctly, it was a converged 549-prompt artifact — but
        only after the worker had downloaded 265 MB, because `stage_from_file`
        runs after the fetch. Whether staging is occupied is a `stat` away.

        MUTATION CONTROL: move the check back into the worker and this fails.
        """
        import types
        from unittest.mock import MagicMock, patch

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        service = MagicMock()
        service.staging_dir.return_value = MagicMock(
            is_dir=lambda: True, name="m.staging"
        )
        service._ref_for.return_value = object()

        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch("src.services.jlens_acquire_service.check_free_space"), patch(
            "src.services.jlens_artifact_service.JLensArtifactService",
            return_value=service,
        ):
            _out, exc, task, _row = self._call(record, monkeypatch)
        assert exc is not None and getattr(exc, "status_code", None) == 409
        assert "replace_staged" in str(getattr(exc, "detail", ""))
        assert task.apply_async.call_count == 0, "265 MB would have been spent"

    def test_replace_staged_SKIPS_that_refusal(self, monkeypatch):
        """Or the flag the message names would not work."""
        import types
        from unittest.mock import MagicMock, patch

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        service = MagicMock()
        service.staging_dir.return_value = MagicMock(is_dir=lambda: True)
        service._ref_for.return_value = object()

        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch("src.services.jlens_acquire_service.check_free_space"), patch(
            "src.services.jlens_artifact_service.JLensArtifactService",
            return_value=service,
        ):
            _out, exc, task, _row = self._call(
                record, monkeypatch, replace_staged=True
            )
        assert exc is None, exc
        assert task.apply_async.call_count == 1

    def test_the_endpoint_OPENS_A_TASK_QUEUE_ROW(self, monkeypatch):
        """Without it the job never appears in Running Work, and a download that
        stalls is indistinguishable from one that was never queued.

        MUTATION CONTROL: drop the `open_row` call and this fails.
        """
        import types
        from unittest.mock import patch

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch("src.services.jlens_acquire_service.check_free_space"):
            _out, _exc, _task, open_row = self._call(record, monkeypatch)
        assert open_row.call_count == 1
        args = open_row.call_args.args
        assert args[0] == "jlens_acquire", args

    def test_the_TOKEN_does_not_reach_the_message_headers(self, monkeypatch):
        """Celery renders `kwargsrepr` into the message headers, and
        `task_send_sent_event` is on — so every `task-sent` event published to
        `celeryev` carries it, readable by Flower or any monitoring consumer.
        The BODY still carries the real value, so the worker is unaffected.

        MUTATION CONTROL: queue with `.delay(...)` and this fails.
        """
        import types
        from unittest.mock import patch

        record = types.SimpleNamespace(id="m_1", repo_id="org/m", file_path="/x")
        with patch(
            "src.services.jlens_model_registry.locate_weights",
            return_value=("org/m", "/x"),
        ), patch("src.services.jlens_acquire_service.check_free_space"):
            _out, exc, task, _row = self._call(
                record, monkeypatch, access_token="hf_SECRET_TOKEN"
            )
        assert exc is None, exc
        call = task.apply_async.call_args.kwargs
        assert call["kwargs"]["access_token"] == "hf_SECRET_TOKEN", (
            "the worker must still receive the real token"
        )
        assert "hf_SECRET_TOKEN" not in call["kwargsrepr"], (
            "the token is rendered into the message headers and the task-sent event"
        )

