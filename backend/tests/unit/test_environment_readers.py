"""One wiring test per allow-listed key (Feature 036; ENH-003 §4 "Wiring, per allow-listed key").

Each test sets an override the way a refresh point does (``runtime_settings.apply``), then drives the
REAL reader — the route handler, the task body, the watchdog, the monitor loop, the buffer — and asserts
the value it used. With every override absent the environment's value is in force, so each test fails
when its reader is reverted to ``settings.<key>`` (the mutation controls in
``0xcc/reviews/036_implementation_controls_2026-10-20.md``).

``test_every_runtime_key_has_a_reader_test`` pins the set: a key added to ``RUNTIME_KEYS`` without a
test here is red.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.core import config, runtime_settings
from src.core.config import settings

#: key -> the test(s) below that drive its reader.
READER_TESTS = {
    "mcp_approval_ttl_hours": "test_mcp_approval_ttl_hours_reaches_expire_stale",
    "steering_timeout_seconds": "test_steering_timeout_reaches_the_status_route and the watchdog test",
    "steering_hazard_prior_threshold": "test_hazard_threshold_and_constants_reach_the_multi_layer_route",
    "steering_cluster_constants_json": "test_cluster_constants_reach_the_single_layer_route (+ multi-layer)",
    "labeling_batch_size": "test_labeling_batch_wiring.py (every test drives the override)",
    "auto_cleanup_after_download": "test_auto_cleanup_reaches_the_download_task",
    "ollama_url": "test_ollama_url_reaches_the_model_listing_route",
    "neuronpedia_local_url": "test_neuronpedia_local_url_reaches_status_and_push",
    "neuronpedia_local_admin_user_id": "test_neuronpedia_admin_user_reaches_ensure_admin_user",
    "system_monitor_interval_seconds": "test_monitor_interval_is_read_every_tick",
    "log_level": "test_log_level_override_moves_the_root_logger (+ refresh-point tests)",
    "activation_read_threads": "test_rolling_activation_buffer.py::test_the_buffer_reads_the_runtime_settings",
    "activation_prefetch": "test_rolling_activation_buffer.py::test_the_buffer_reads_the_runtime_settings",
    # Feature 040 S1 (FR-7): each cap is read through `Limits.from_settings()` / the list route.
    "behaviour_max_sub_questions": "test_behaviour_detector_definition.py::test_each_cap_is_the_runtime_setting[sub_questions]",
    "behaviour_max_success_criteria": "test_behaviour_detector_definition.py::test_each_cap_is_the_runtime_setting[success_criteria]",
    "behaviour_max_held_out_columns": "test_behaviour_detector_definition.py::test_each_cap_is_the_runtime_setting[held_out_columns] (+ test_over_cap_through_the_route)",
    "behaviour_max_models": "test_behaviour_detector_definition.py::test_each_cap_is_the_runtime_setting[models]",
    "rule_set_max_bytes": "test_rule_set_store.py::test_an_oversized_set_is_refused_by_the_runtime_cap",
    "behaviour_list_page_default": "test_behaviour_catalogue_paging.py::test_the_default_page_is_the_runtime_setting",
    "behaviour_list_page_max": "test_behaviour_catalogue_paging.py::test_the_default_page_is_the_runtime_setting",
    # Feature 040 S2: read through `BreakdownSettings.from_settings` / `ColumnLimits` / the report routes.
    "calibration_floor_negatives": "test_probe_breakdown.py::test_the_memo_key_moves_with_every_setting_read (+ test_detector_success_report.py::test_the_floor_is_the_runtime_setting)",
    # ENH-006 D82: read by `recompute_rung` through `effective()`.
    "everyday_text_fpr_factor": "test_everyday_text_rung.py::TestRecompute::test_the_two_knobs_are_the_runtime_settings",
    "everyday_text_min_rows": "test_everyday_text_rung.py::TestRecompute::test_the_two_knobs_are_the_runtime_settings",
    # ENH-006 D84: read by `recompute_rung` through `effective()`.
    "nonsense_text_fpr_factor": "test_rung2_tighten.py::TestRecompute::test_the_three_knobs_are_the_runtime_settings",
    "nonsense_text_min_rows": "test_rung2_tighten.py::TestRecompute::test_the_three_knobs_are_the_runtime_settings",
    "rung2_min_recall": "test_rung2_tighten.py::TestRecompute::test_the_three_knobs_are_the_runtime_settings",
    "breakdown_min_subgroup_rows": "test_probe_breakdown.py::test_the_memo_key_moves_with_every_setting_read",
    "breakdown_page_default": "test_probe_breakdown.py::test_the_memo_key_moves_with_every_setting_read",
    "breakdown_page_max": "test_probe_breakdown.py::test_the_memo_key_moves_with_every_setting_read",
    "breakdown_max_filters": "test_probe_breakdown.py::test_the_memo_key_moves_with_every_setting_read",
    "group_column_max_values": "test_view_columns.py::test_a_row_id_column_is_refused_too_many_values",
    "view_max_group_columns": "test_view_columns.py::test_the_column_count_cap_is_the_runtime_setting",
    "report_list_page_default": "test_detector_success_report.py::test_report_pages_are_the_runtime_settings",
    "report_list_page_max": "test_detector_success_report.py::test_report_pages_are_the_runtime_settings",
    "success_report_max_bytes": "test_detector_success_report.py::test_an_oversized_report_is_refused",
    # Feature 040 S3: read through `rule_set_store.engine_limits` / the rules-evaluation request and reaper.
    "rule_set_max_signals": "test_rules_settings_readers.py::test_the_rule_set_caps_are_the_runtime_settings[rule_set_max_signals-signals]",
    "rule_set_max_cases": "test_rules_settings_readers.py::test_the_rule_set_caps_are_the_runtime_settings[rule_set_max_cases-cases]",
    "rule_text_max_code_points": "test_rule_sets_api.py::test_the_caps_refuse_naming_the_setting",
    "rule_pattern_max_mem_bytes": "test_rule_sets_api.py::test_the_engine_limits_are_the_runtime_settings",
    "rule_unknown_share_max": "test_rules_settings_readers.py::test_the_unknown_share_and_body_cap_are_read_at_request",
    "rule_signals_max_messages": "test_rule_sets_api.py::test_the_caps_refuse_naming_the_setting",
    "rules_evaluation_max_rows": "test_rules_settings_readers.py::test_the_row_cap_is_the_runtime_setting",
    "rules_evaluation_max_bytes": "test_rules_settings_readers.py::test_the_unknown_share_and_body_cap_are_read_at_request",
    "rule_verdict_cache_max_bytes": "test_cleanup_stuck_rules_evaluations.py::test_pruning_by_age_and_size_spares_entries_in_use",
    "rule_verdict_cache_max_age_days": "test_cleanup_stuck_rules_evaluations.py::test_pruning_by_age_and_size_spares_entries_in_use",
    "rules_evaluation_stale_after_seconds": "test_cleanup_stuck_rules_evaluations.py::test_a_silent_running_row_fails_worker_lost",
    # Feature 040 S4: read through `code_rule_runner.enabled` / `code_limits` by the routes and the rules task.
    "code_rules_enabled": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_call_fuel": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner (+ test_code_rule_evaluation_wiring.py::test_a_code_rule_limit_change_moves_the_engine_digest)",
    "code_rule_load_fuel": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_prelude_fuel": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_memory_bytes": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_input_code_points": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_output_bytes": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_wall_ms": "test_code_rules_api.py::test_the_switch_and_the_limits_reach_the_runner",
    "code_rule_batch_texts": "test_code_rule_evaluation_wiring.py::test_a_bulk_evaluation_sends_each_code_rule_once_per_chunk",
    # Feature 041 (FR-27): each read through `effective()` by its route or service.
    "blend_family_page_size": "test_blend_settings.py::test_family_page_one_over_refused_naming_setting",
    "blend_max_members": "test_blend_refusals.py::test_cap_exceeded_members_and_candidates (+ test_blend_settings.py)",
    "blend_max_candidates": "test_blend_refusals.py::test_cap_exceeded_members_and_candidates (+ test_blend_settings.py)",
    "blend_max_concurrent": "test_blend_start.py::test_third_start_over_cap_of_two_refused",
    "blend_folds_default": "test_blend_settings.py::test_folds_are_the_runtime_settings",
    "blend_folds_min": "test_blend_settings.py::test_folds_are_the_runtime_settings",
    "blend_folds_max": "test_blend_settings.py::test_folds_are_the_runtime_settings",
    "blend_cv_min_rows_per_class": "test_blend_settings.py::test_the_report_records_the_runtime_settings",
    "blend_min_aligned_fraction": "test_blend_settings.py::test_the_aligned_fraction_is_the_runtime_setting",
    "blend_report_page_size": "test_blend_reports_routes.py::test_report_page_size_is_the_runtime_setting",
    "blend_scores_page_limit": "test_blend_reports_routes.py::test_scores_page_and_limit",
    "blend_evaluation_time_limit_s": "test_blend_janitor.py::test_queued_past_time_limit_fails",
    "probe_score_batch_cap": "test_family_scoring.py::test_single_and_family_caps_read_one_setting",
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    runtime_settings.reset()
    yield
    runtime_settings.reset()


def test_every_runtime_key_has_a_reader_test():
    assert set(READER_TESTS) == set(config.RUNTIME_KEYS)


# ── mcp_approval_ttl_hours ───────────────────────────────────────────────────────────────────


async def test_mcp_approval_ttl_hours_reaches_expire_stale(async_session):
    from src.api.v1.endpoints.mcp_approvals import expire_stale
    from src.models.agent_approval import AgentApprovalRequest

    assert settings.mcp_approval_ttl_hours > 2  # the environment's value would NOT expire it
    row = AgentApprovalRequest(
        tool_name="millm_load_model", payload={"model": "m"}, status="pending",
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    async_session.add(row)
    await async_session.commit()

    runtime_settings.apply({"mcp_approval_ttl_hours": 1})
    expired = await expire_stale(async_session)
    await async_session.commit()
    await async_session.refresh(row)
    assert expired == [row.id]
    assert row.status == "expired" and row.reason == "expired: pending for more than 1 h"


# ── steering_timeout_seconds ─────────────────────────────────────────────────────────────────


async def test_steering_timeout_reaches_the_status_route(monkeypatch):
    from src.api.v1.endpoints import steering as steering_ep

    monkeypatch.setattr(
        steering_ep, "get_steering_service",
        lambda: SimpleNamespace(_loaded_models={}, _loaded_saes={}),
    )
    runtime_settings.apply({"steering_timeout_seconds": 333})
    status = await steering_ep.get_steering_status()
    assert status["timeout_seconds"] == 333
    runtime_settings.apply({"steering_timeout_seconds": 444})
    assert (await steering_ep.get_steering_status())["timeout_seconds"] == 444


def test_the_watchdog_rereads_the_timeout_at_every_generation(monkeypatch):
    from src.services import steering_service

    monkeypatch.setattr(steering_service, "_generation_watchdog", None)
    monkeypatch.setattr(steering_service.GenerationWatchdog, "start", lambda self: None)
    runtime_settings.apply({"steering_timeout_seconds": 200})
    watchdog = steering_service.get_generation_watchdog()
    watchdog.start_generation()
    assert watchdog.hard_timeout == 230
    watchdog.end_generation()

    # The singleton is built once per process; the next generation still sees a change.
    runtime_settings.apply({"steering_timeout_seconds": 300})
    assert steering_service.get_generation_watchdog() is watchdog
    watchdog.start_generation()
    assert watchdog.hard_timeout == 330


# ── steering_cluster_constants_json, steering_hazard_prior_threshold ────────────────────────


def _sae(layer):
    return SimpleNamespace(
        status="ready", local_path="no/such/sae", hook_type="residual", layer=layer,
        n_features=100, d_model=8, architecture="standard",
    )


@pytest.fixture
def allocation(monkeypatch):
    """The allocation route with its SAE rows stubbed (no weights on disk → approximate G=1), and
    spies on the two calls whose arguments carry the runtime settings."""
    from src.api.v1.endpoints import steering as steering_ep

    async def get_sae(db, sae_id):
        return _sae(int(sae_id.rsplit("_", 1)[1]))

    monkeypatch.setattr(steering_ep.SAEManagerService, "get_sae", staticmethod(get_sae))
    seen = {"constants": [], "thresholds": []}
    real_resolve = steering_ep.resolve_constants
    real_detect = steering_ep.steering_hazards.detect_hazards

    def resolve(settings_json, sae_id):
        seen["constants"].append(settings_json)
        return real_resolve(settings_json, sae_id)

    def detect(*args, **kwargs):
        seen["thresholds"].append(kwargs.get("prior_threshold"))
        return real_detect(*args, **kwargs)

    monkeypatch.setattr(steering_ep, "resolve_constants", resolve)
    monkeypatch.setattr(steering_ep.steering_hazards, "detect_hazards", detect)
    return steering_ep, seen


OVERRIDE_JSON = '{"default": {"a": 7.5}}'


async def test_cluster_constants_reach_the_single_layer_route(allocation):
    steering_ep, seen = allocation
    from src.schemas.steering import ClusterAllocationRequest

    runtime_settings.apply({"steering_cluster_constants_json": OVERRIDE_JSON})
    request = ClusterAllocationRequest(
        sae_id="sae_3", members=[{"feature_idx": 1, "layer": 3}, {"feature_idx": 2, "layer": 3}]
    )
    response = await steering_ep.compute_cluster_strength_allocation(request, db=None)
    assert seen["constants"] == [OVERRIDE_JSON]
    assert response.constants_used["a"] == 7.5


async def test_hazard_threshold_and_constants_reach_the_multi_layer_route(allocation):
    steering_ep, seen = allocation
    from src.schemas.steering import ClusterAllocationRequest

    runtime_settings.apply({
        "steering_cluster_constants_json": OVERRIDE_JSON,
        "steering_hazard_prior_threshold": 0.125,
    })
    request = ClusterAllocationRequest(
        sae_id="sae_3",
        members=[
            {"feature_idx": 1, "layer": 3, "sae_id": "sae_3"},
            {"feature_idx": 2, "layer": 5, "sae_id": "sae_5"},
        ],
    )
    response = await steering_ep.compute_cluster_strength_allocation(request, db=None)
    # Two per-layer resolutions and the shared one: every site reads the override.
    assert seen["constants"] == [OVERRIDE_JSON] * 3
    assert seen["thresholds"] == [0.125]
    assert {layer.constants_used["a"] for layer in response.layers.values()} == {7.5}


# ── auto_cleanup_after_download ──────────────────────────────────────────────────────────────


@pytest.fixture
def download_env(async_engine, tmp_path, monkeypatch):
    """The real download task body (the test_dataset_task_revision.py harness)."""
    from src.core import cancellation
    from src.workers import dataset_tasks

    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(dataset_tasks, "emit_dataset_progress", lambda *a, **k: None)
    monkeypatch.setattr(cancellation, "CancelWatchdog", lambda *a, **k: contextlib.nullcontext())

    def fake_load_dataset(repo_id, **kwargs):
        from datasets import Dataset as HFDataset

        return HFDataset.from_dict({"text": ["a", "b"]})

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return dataset_tasks


def _download(dataset_tasks):
    from src.core.database import SyncSessionLocal
    from src.models.dataset import Dataset, DatasetStatus

    with SyncSessionLocal() as db:
        row = Dataset(id=uuid.uuid4(), name="ds", source="HuggingFace", hf_repo_id="org/ds",
                      status=DatasetStatus.DOWNLOADING, extra_metadata={})
        db.add(row)
        db.commit()
        dataset_id = str(row.id)
    downloads = settings.datasets_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    (downloads / "chunk").write_text("partial")
    dataset_tasks.download_dataset_task(dataset_id=dataset_id, repo_id="org/ds", split=None, config=None)
    return downloads


async def test_auto_cleanup_reaches_the_download_task(download_env):
    assert settings.auto_cleanup_after_download is True  # the environment's value cleans up
    runtime_settings.apply({"auto_cleanup_after_download": False})
    assert _download(download_env).exists(), "the override said keep the download cache"
    runtime_settings.apply({})
    assert not _download(download_env).exists()


# ── ollama_url ───────────────────────────────────────────────────────────────────────────────


async def test_ollama_url_reaches_the_model_listing_route(client, monkeypatch):
    import httpx

    asked = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **k):
            asked.append(url)
            return httpx.Response(200, json={"data": [{"id": "m1"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    runtime_settings.apply({"ollama_url": "http://llm-override.test:9999"})
    response = await client.get("/api/v1/labeling/models/available")
    assert response.status_code == 200, response.text
    assert response.json()["models"][0]["name"] == "m1"
    assert asked == ["http://llm-override.test:9999/v1/models"]


# ── neuronpedia_local_url, neuronpedia_local_admin_user_id ─────────────────────────────────


async def test_neuronpedia_local_url_reaches_status_and_push(monkeypatch):
    from src.api.v1.endpoints import neuronpedia
    from src.services.neuronpedia_local_service import NeuronpediaLocalPushService

    monkeypatch.setattr(neuronpedia.settings, "neuronpedia_local_db_url", None)
    runtime_settings.apply({"neuronpedia_local_url": "http://np-override.test"})
    status = await neuronpedia.get_local_neuronpedia_status()
    assert status["public_url"] == "http://np-override.test"
    assert (NeuronpediaLocalPushService.public_source_url("m", "s")
            == "http://np-override.test/m/s")
    runtime_settings.apply({"neuronpedia_local_url": None})
    assert NeuronpediaLocalPushService.public_source_url("m", "s") is None


def test_the_push_builds_its_link_through_the_helper():
    """The push body is a 300-line database writer; its link line must CALL the helper, once
    (memory: mutation-survivors-cluster-on-callers)."""
    import ast
    import inspect
    import textwrap

    from src.services.neuronpedia_local_service import NeuronpediaLocalPushService

    tree = ast.parse(textwrap.dedent(inspect.getsource(NeuronpediaLocalPushService.push_sae_to_local)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "public_source_url"]
    assert len(calls) == 1
    reads = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "neuronpedia_local_url"]
    assert reads == [], "the push reads the URL itself instead of through the helper"


async def test_neuronpedia_admin_user_reaches_ensure_admin_user():
    from src.services.neuronpedia_local_service import NeuronpediaLocalClient

    seen = []

    class Conn:
        async def fetchrow(self, sql, user_id):
            seen.append(user_id)
            return {"id": user_id}

    class Pool:
        @contextlib.asynccontextmanager
        async def acquire(self):
            yield Conn()

    client = NeuronpediaLocalClient.__new__(NeuronpediaLocalClient)
    client._pool = Pool()
    runtime_settings.apply({"neuronpedia_local_admin_user_id": "np-user-override"})
    assert await client.ensure_admin_user() == "np-user-override"
    assert seen == ["np-user-override"]


# ── system_monitor_interval_seconds ──────────────────────────────────────────────────────────


async def test_monitor_interval_is_read_every_tick(monkeypatch):
    from src.services import background_monitor, gpu_monitor_service, system_monitor_service

    monkeypatch.setattr(background_monitor, "_background_monitor", None)
    monitor = background_monitor.get_background_monitor()
    monkeypatch.setattr(system_monitor_service, "get_system_monitor_service", lambda: None)
    monkeypatch.setattr(gpu_monitor_service, "get_gpu_monitor_service", lambda: None)

    async def collect(self, system_service, gpu_service):
        return None

    monkeypatch.setattr(background_monitor.BackgroundMonitor, "_collect_and_emit_metrics", collect)
    slept = []
    plan = iter([{"system_monitor_interval_seconds": 9}, None])

    async def fake_sleep(seconds):
        slept.append(seconds)
        nxt = next(plan, None)
        if nxt is None:
            monitor._running = False
        else:
            runtime_settings.apply(nxt)

    monkeypatch.setattr(background_monitor.asyncio, "sleep", fake_sleep)
    runtime_settings.apply({"system_monitor_interval_seconds": 7})
    monitor._running = True
    await monitor._monitor_loop()
    assert slept == [7, 9]


# ── log_level ────────────────────────────────────────────────────────────────────────────────


def test_log_level_override_moves_the_root_logger():
    root = logging.getLogger()
    before = root.level
    runtime_settings.apply({"log_level": "ERROR"})
    assert root.level == logging.ERROR
    runtime_settings.apply({})
    assert root.level == logging.getLevelName(settings.log_level)
    root.setLevel(before)


def test_no_override_leaves_the_process_logging_alone():
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    runtime_settings.apply({"labeling_batch_size": 4})
    assert root.level == logging.DEBUG
