"""Which pooling rule fits a training view, and a run that omits it (ENH-006 G39).

The rule (`probe_rule_recommendation.recommend`) is pure and tested at each threshold. The wiring goes through
the REAL routes over real `datasets` splits: registration reads the span column into `evidence_spans`, the
recommendation route reads that, `POST /runs` refuses a run omitting a recommended rule (and records an
acknowledged one), the forecast carries the recommendation and the refusal it would give, and the MCP tools
reach all of it. The run's post-training diagnostics are proven in `test_probe_rule_diagnostics_pipeline.py`.

Controls are recorded in `0xcc/reviews/g39_rule_recommendation_2026-11-09.md`.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from src.core.config import settings
from src.models.probe_monitor import ProbeMonitorDataset, ProbeMonitorRun
from src.services import probe_rule_recommendation as R
from tests.support.detector_fixtures import dataset_row, model_row, write_split

API = "/api/v1/probe-monitors"


# ── a word-level tokenizer: one token per whitespace-separated word ───────────────────────────


@pytest.fixture(scope="module")
def word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<unk>", "<s>"] + [f"w{i}" for i in range(64)] + ["ssn", "num"]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>")


def local_rows(n=60, evidence_words=6, hard_every=0, filler=40):
    """Rows of `filler` words; a positive carries `evidence_words` words of evidence in the middle, with
    its span; every `hard_every`-th negative carries the same surface form, also with a span."""
    texts, labels, spans = [], [], []
    for i in range(n):
        positive = i % 2 == 0
        hard = (not positive) and hard_every and (i // 2) % hard_every == 0
        before = " ".join(f"w{(i + k) % 64}" for k in range(filler // 2 + (i % 7)))
        after = " ".join(f"w{(i * 3 + k) % 64}" for k in range(filler // 2))
        if positive or hard:
            evidence = " ".join(["num"] * evidence_words)
            text = f"{before} {evidence} {after}"
            start = len(before) + 1
            spans.append(json.dumps([[start, start + len(evidence)]]))
        else:
            text = f"{before} {after}"
            spans.append("[]")
        texts.append(text)
        labels.append("1" if positive else "0")
    return {"text": texts, "label": labels, "spans": spans}


def whole_rows(n=60):
    texts = [" ".join(f"w{(i + k) % 64}" for k in range(10 + i % 5)) for i in range(n)]
    return {"text": texts, "label": [str((i + 1) % 2) for i in range(n)],
            "spans": [json.dumps([[0, len(t)]]) if i % 2 == 0 else None for i, t in enumerate(texts)]}


# ── the pure rule ──────────────────────────────────────────────────────────────────────────────


class TestReadingACell:
    @pytest.mark.parametrize("cell", [None, "", "  ", "[]", "null", []])
    def test_no_span(self, cell):
        assert R.parse_span_cell(cell) is None

    def test_the_rule_set_output_text_and_a_native_list(self):
        assert R.parse_span_cell("[[17,28],[40,51]]") == [(17, 28), (40, 51)]
        assert R.parse_span_cell([[3, 9]]) == [(3, 9)]

    @pytest.mark.parametrize("cell", ["[[1,2", "{}", "[[1]]", '[["a",2]]', "[[true,2]]", "7"])
    def test_anything_else_is_an_error_never_no_span(self, cell):
        with pytest.raises(ValueError):
            R.parse_span_cell(cell)

    def test_covered_counts_an_overlap_once(self):
        assert R.covered([(0, 10), (5, 12), (20, 22)]) == 14


class TestTheProfile:
    def test_numbers_of_a_local_view(self):
        rows = [(i, 1 if i % 2 == 0 else 0, "a" * 100, "[[10,20]]" if i % 2 == 0 else ("[[0,5]]" if i % 4 == 1 else None))
                for i in range(40)]
        p = R.profile_spans("s", rows)
        assert (p["positives"], p["positives_with_span"], p["negatives"], p["negatives_with_span"]) == (20, 20, 20, 10)
        assert p["ratio"]["median"] == 0.1 and p["span_chars"]["median"] == 10.0
        assert p["position"]["median"] == 0.1 and p["length_spread"] == 1.0

    def test_an_offset_outside_its_text_is_refused_naming_the_row(self):
        with pytest.raises(R.SpanColumnInvalid) as refused:
            R.profile_spans("s", [(0, 1, "abc", "[[0,2]]"), (7, 1, "abc", "[[1,4]]")])
        assert refused.value.code == "span_out_of_range" and refused.value.detail["row"] == 7

    def test_a_span_on_a_conversation_is_refused(self):
        with pytest.raises(R.SpanColumnInvalid) as refused:
            R.profile_spans("s", [(3, 1, None, "[[0,2]]")])
        assert refused.value.code == "span_needs_text"

    def test_an_unreadable_cell_is_refused(self):
        with pytest.raises(R.SpanColumnInvalid) as refused:
            R.profile_spans("s", [(4, 0, "abcdef", "[[0,")])
        assert refused.value.code == "span_unreadable" and refused.value.detail["row"] == 4


def profile(ratio, *, with_span=40, positives=40, negatives=40, hard=0, spread=1.0, span_chars=11.0):
    return {"column": "s", "positives": positives, "positives_with_span": with_span, "negatives": negatives,
            "negatives_with_span": hard, "ratio": {"median": ratio, "p75": ratio, "p90": ratio},
            "span_chars": {"median": span_chars, "p75": span_chars, "p90": span_chars},
            "position": {"p10": 0.1, "median": 0.5, "p90": 0.9}, "row_chars": {"p10": 100, "median": 200, "p90": 300},
            "length_spread": spread}


class TestTheRule:
    def test_no_profile_recommends_nothing_and_says_so(self):
        out = R.recommend(None)
        assert out["recorded"] is False and out["rules"] == [] and "no evidence span is recorded" in out["reason"]
        assert R.rule_choice_refusal(out, ["mean"], None) is None

    @pytest.mark.parametrize("ratio", [0.01, R.LOCAL_RATIO])
    def test_local_evidence_gets_max_and_rolling_mean_max(self, ratio):
        out = R.recommend(profile(ratio))
        assert (out["shape"], out["rules"]) == ("local", ["max", "rolling_mean_max"])

    @pytest.mark.parametrize("ratio", [R.WHOLE_TEXT_RATIO, 1.0])
    def test_whole_text_evidence_gets_mean(self, ratio):
        out = R.recommend(profile(ratio))
        assert (out["shape"], out["rules"]) == ("whole_text", ["mean"])

    @pytest.mark.parametrize("ratio", [R.LOCAL_RATIO + 0.001, R.WHOLE_TEXT_RATIO - 0.001])
    def test_between_gets_both_shapes(self, ratio):
        assert R.recommend(profile(ratio))["rules"] == ["mean", "max"]

    def test_too_few_rows_with_a_span_recommends_nothing(self):
        out = R.recommend(profile(0.01, with_span=R.MIN_SPAN_ROWS - 1, positives=R.MIN_SPAN_ROWS - 1))
        assert out["shape"] == "unmeasured" and out["rules"] == []
        assert R.recommend(profile(0.01, with_span=R.MIN_SPAN_ROWS, positives=R.MIN_SPAN_ROWS))["rules"]

    def test_spans_on_too_few_of_the_positives_recommend_nothing(self):
        assert R.recommend(profile(0.01, with_span=40, positives=201))["rules"] == []
        assert R.recommend(profile(0.01, with_span=40, positives=200))["rules"]

    def test_the_window_is_the_75th_percentile_in_tokens_clamped(self):
        assert R.recommend(profile(0.01), span_tokens={"n": 9, "median": 5, "p75": 6.2, "p90": 9})["rolling_windows"] == [7]
        assert R.recommend(profile(0.01), span_tokens={"n": 9, "median": 1, "p75": 1, "p90": 1})["rolling_windows"] == [R.MIN_WINDOW]
        assert R.recommend(profile(0.01), span_tokens={"n": 9, "median": 1, "p75": 900, "p90": 1})["rolling_windows"] == [R.MAX_WINDOW]

    def test_without_a_tokenizer_the_window_is_not_guessed(self):
        out = R.recommend(profile(0.01))
        assert out["rolling_windows"] is None and any("model_id" in c for c in out["cautions"])

    def test_surface_only_negatives_add_a_wider_window(self):
        tokens = {"n": 9, "median": 6, "p75": 6, "p90": 6}
        hard = int(40 * R.HARD_NEGATIVE_SHARE)
        assert R.recommend(profile(0.01, hard=hard), span_tokens=tokens)["rolling_windows"] == [6, 6 * R.CONTEXT_WINDOW_FACTOR]
        assert R.recommend(profile(0.01, hard=hard - 1), span_tokens=tokens)["rolling_windows"] == [6]

    def test_a_wide_length_spread_is_a_caution_not_a_refusal(self):
        out = R.recommend(profile(0.01, spread=R.LENGTH_SPREAD_CAUTION))
        assert any("length band" in c for c in out["cautions"])
        assert not any("length band" in c for c in R.recommend(profile(0.01, spread=3.9))["cautions"])

    def test_the_evidence_and_thresholds_travel_with_it(self):
        out = R.recommend(profile(0.05, hard=4))
        assert out["evidence"]["evidence_share_median"] == 0.05
        assert out["evidence"]["surface_only_negatives"] == 4 and out["thresholds"]["local_ratio"] == R.LOCAL_RATIO


class TestTheRefusal:
    def test_the_default_rules_omit_rolling_mean_max_on_a_local_view(self):
        refusal = R.rule_choice_refusal(R.recommend(profile(0.01)), ["mean", "max", "attention"], None)
        assert refusal["code"] == "rule_choice_omits_recommendation" and refusal["missing"] == ["rolling_mean_max"]
        assert "acknowledge_rule_choice" in refusal["message"]

    def test_an_acknowledged_omission_passes_and_a_blank_one_does_not(self):
        rec = R.recommend(profile(0.01))
        assert R.rule_choice_refusal(rec, ["mean"], "baseline only, the stack trains rolling elsewhere") is None
        assert R.rule_choice_refusal(rec, ["mean"], "   ") is not None

    def test_nothing_omitted_nothing_refused(self):
        assert R.rule_choice_refusal(R.recommend(profile(0.01)), ["rolling_mean_max", "max"], None) is None


class TestAfterTraining:
    def test_constant_floor(self):
        floor = R.constant_floor([17.65] * 5 + [18.0, 19.0, 20.0], [1, 0, 0, 0, 0, 1, 1, 0])
        assert (floor["rows_at_min"], floor["rows"], floor["floored"], floor["positives_at_min"]) == (5, 8, True, 1)
        assert R.constant_floor([1.0, 2.0, 3.0])["floored"] is False

    def test_dilution(self):
        assert R.dilution([2, 2, 4], [8, 8, 8]) == {"n": 3, "p10": 0.25, "median": 0.25, "p90": 0.45}


# ── the routes, over real splits ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    return tmp_path


async def _registered(client, async_session, data_dir, columns, *, span_column="spans", role="train"):
    ds = await dataset_row(async_session, write_split(data_dir, columns))
    await async_session.commit()
    body = {"name": f"view-{ds.id.hex[:6]}", "dataset_id": str(ds.id), "input_column": "text", "label_column": "label",
            "label_mapping": {"1": "positive", "0": "negative"}, "role": role}
    if span_column is not None:
        body["span_column"] = span_column
    made = await client.post(f"{API}/datasets", json=body)
    return made


class TestRegistration:
    async def test_the_span_column_is_read_into_a_profile_of_numbers(self, client, async_session, data_dir):
        columns = local_rows(hard_every=2)
        made = await _registered(client, async_session, data_dir, columns)
        assert made.status_code == 201, made.text
        view = made.json()
        assert view["span_column"] == "spans"
        p = view["evidence_spans"]
        assert (p["positives"], p["positives_with_span"], p["negatives"], p["negatives_with_span"]) == (30, 30, 30, 15)
        assert p["span_chars"]["median"] == float(len(" ".join(["num"] * 6)))
        assert 0 < p["ratio"]["median"] < R.LOCAL_RATIO
        assert "num" not in json.dumps(p)  # numbers only, never text

    async def test_without_a_span_column_nothing_is_read(self, client, async_session, data_dir):
        view = (await _registered(client, async_session, data_dir, local_rows(), span_column=None)).json()
        assert view["span_column"] is None and view["evidence_spans"] is None

    async def test_an_absent_column_is_refused(self, client, async_session, data_dir):
        made = await _registered(client, async_session, data_dir, local_rows(), span_column="nope")
        assert made.status_code == 422 and made.json()["detail"]["code"] == "column_absent"
        assert (await async_session.execute(select(func.count()).select_from(ProbeMonitorDataset))).scalar_one() == 0

    async def test_an_offset_past_its_row_is_refused(self, client, async_session, data_dir):
        columns = local_rows()
        columns["spans"][4] = json.dumps([[0, len(columns["text"][4]) + 1]])
        made = await _registered(client, async_session, data_dir, columns)
        assert made.status_code == 422
        assert (made.json()["detail"]["code"], made.json()["detail"]["row"]) == ("span_out_of_range", 4)


class TestTheRecommendationRoute:
    async def test_a_view_without_spans_gets_nothing_and_the_reason(self, client, async_session, data_dir):
        view = (await _registered(client, async_session, data_dir, local_rows(), span_column=None)).json()
        out = (await client.get(f"{API}/datasets/{view['id']}/rule-recommendation")).json()
        assert out["rules"] == [] and out["recorded"] is False and "no evidence span" in out["reason"]

    async def test_a_local_view_gets_max_and_rolling_and_a_window_in_the_models_tokens(
            self, client, async_session, data_dir, monkeypatch, word_tokenizer):
        from src.services import probe_monitor_run as run_module

        loaded = []
        monkeypatch.setattr(run_module, "_load_tokenizer_for_run", lambda row: loaded.append(row.model_id) or word_tokenizer)
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows(hard_every=2))).json()
        bare = (await client.get(f"{API}/datasets/{view['id']}/rule-recommendation")).json()
        assert bare["rules"] == ["max", "rolling_mean_max"] and bare["rolling_windows"] is None and loaded == []
        sized = (await client.get(f"{API}/datasets/{view['id']}/rule-recommendation", params={"model_id": model.id})).json()
        assert loaded == [model.id]
        assert sized["evidence"]["span_tokens"] == {"n": 30, "median": 6.0, "p75": 6.0, "p90": 6.0}
        assert sized["rolling_windows"] == [6, 24]  # half the negatives carry the surface form

    async def test_a_whole_text_view_gets_mean(self, client, async_session, data_dir):
        view = (await _registered(client, async_session, data_dir, whole_rows())).json()
        out = (await client.get(f"{API}/datasets/{view['id']}/rule-recommendation")).json()
        assert (out["shape"], out["rules"]) == ("whole_text", ["mean"])

    async def test_unknown_view_and_model_are_404(self, client, async_session, data_dir):
        assert (await client.get(f"{API}/datasets/pmd_nope/rule-recommendation")).status_code == 404
        view = (await _registered(client, async_session, data_dir, local_rows())).json()
        assert (await client.get(f"{API}/datasets/{view['id']}/rule-recommendation",
                                 params={"model_id": "m_nope"})).status_code == 404

    def test_the_route_is_served(self):
        from src.main import app

        assert "get" in app.openapi()["paths"][f"{API}/datasets/{{dataset_id}}/rule-recommendation"]


@pytest.fixture
def queued(monkeypatch):
    from src.api.v1.endpoints import probe_monitors as ep

    calls = []
    monkeypatch.setattr(ep, "gpu_delay", lambda task, gpu_request=None: (
        lambda **kw: calls.append(kw) or SimpleNamespace(id=f"t_{len(calls)}")))
    return calls


async def _runs(async_session):
    return (await async_session.execute(select(ProbeMonitorRun))).scalars().all()


class TestSubmittingARun:
    async def test_the_default_rules_on_a_local_view_are_refused_with_the_evidence(
            self, client, async_session, data_dir, queued):
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows())).json()
        r = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": view["id"]})
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["code"] == "rule_choice_omits_recommendation" and detail["missing"] == ["rolling_mean_max"]
        assert detail["recommendation"]["evidence"]["positives_with_span"] == 30
        assert queued == [] and await _runs(async_session) == []

    async def test_an_acknowledged_omission_runs_and_the_reason_is_on_the_run(
            self, client, async_session, data_dir, queued):
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows())).json()
        reason = "a mean baseline for the stack; rolling_mean_max trains in the next run"
        r = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": view["id"],
                                                   "config": {"acknowledge_rule_choice": reason}})
        assert r.status_code == 202, r.text
        (run,) = await _runs(async_session)
        assert run.config["acknowledge_rule_choice"] == reason
        record = run.config["rule_recommendation"]
        assert (record["rules"], record["missing"], record["acknowledged"]) == (
            ["max", "rolling_mean_max"], ["rolling_mean_max"], True)
        assert len(queued) == 1

    async def test_a_run_training_the_recommended_rules_is_not_refused(self, client, async_session, data_dir, queued):
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows())).json()
        r = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": view["id"],
                                                   "config": {"rules": ["max", "rolling_mean_max"]}})
        assert r.status_code == 202, r.text
        (run,) = await _runs(async_session)
        assert run.config["rule_recommendation"]["missing"] == [] and "acknowledge_rule_choice" in run.config

    async def test_a_view_without_spans_is_never_refused(self, client, async_session, data_dir, queued):
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows(), span_column=None)).json()
        r = await client.post(f"{API}/runs", json={"model_id": model.id, "train_dataset_id": view["id"],
                                                   "config": {"rules": ["mean"]}})
        assert r.status_code == 202, r.text
        (run,) = await _runs(async_session)
        assert "rule_recommendation" not in run.config


class TestTheForecast:
    async def test_it_carries_the_recommendation_and_the_refusal_the_run_would_get(
            self, client, async_session, data_dir, monkeypatch, word_tokenizer):
        from src.services import probe_monitor_run as run_module
        from src.services import probe_run_memory as memory

        monkeypatch.setattr(run_module, "_load_tokenizer_for_run", lambda row: word_tokenizer)
        monkeypatch.setattr(memory, "forecast_for_request", lambda *a, **k: {"checked": True, "fits": True})
        model = await model_row(async_session)
        await async_session.commit()
        view = (await _registered(client, async_session, data_dir, local_rows())).json()
        body = (await client.post(f"{API}/runs/forecast", json={"model_id": model.id, "train_dataset_id": view["id"]})).json()
        assert body["fits"] is True
        rec = body["rule_recommendation"]
        assert rec["rules"] == ["max", "rolling_mean_max"] and rec["rolling_windows"] == [6]
        assert rec["refusal"]["missing"] == ["rolling_mean_max"]
        ok = (await client.post(f"{API}/runs/forecast", json={
            "model_id": model.id, "train_dataset_id": view["id"],
            "config": {"rules": ["max", "rolling_mean_max"], "rolling_windows": [6]}})).json()
        assert ok["rule_recommendation"]["refusal"] is None


# ── the MCP surface ────────────────────────────────────────────────────────────────────────────


def _server(categories="probe_monitors"):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    return build_server(MCPSettings(tool_categories=categories, allow_anonymous=True, _env_file=None), stdio=True)


class TestTheTools:
    def test_recommend_probe_rules_is_on_the_server_the_deployment_builds(self):
        from tests.unit.test_reachability import TestTheDEPLOYMENTEnablesWhatTheCodeRegisters as T

        mcp, _ = _server(",".join(sorted(T()._deployed_categories())))
        assert "recommend_probe_rules" in {t.name for t in asyncio.run(mcp.list_tools())}

    def test_the_howto_and_the_instructions_name_it(self):
        from src.mcp_server.tools import howto

        assert "`recommend_probe_rules`" in howto.TOPICS["behaviour_detectors"]
        mcp, _ = _server()
        assert "recommend_probe_rules" in mcp.instructions


class _Studio:
    def __init__(self, forecast):
        self.forecast, self.calls = forecast, []

    async def post(self, path, json_body=None, **params):
        self.calls.append((path, json_body))
        if path == "/probe-monitors/runs/forecast":
            return self.forecast
        if path == "/mcp/approvals":
            return {"id": "apr_1", "status": "pending"}
        return {"id": "pmr_9", "task_id": "t_9", "status": "queued"}


def _start(studio, **extra):
    from mcp.server.mcpserver import MCPServer

    from src.mcp_server.config import MCPSettings
    from src.mcp_server.tools import probe_monitors

    mcp = MCPServer("t")
    probe_monitors.register(mcp, studio, MCPSettings(auth_token="x" * 32, probe_gpu_approval=True))
    arguments = {"model_id": "m_1", "train_dataset_id": "pmd_1", **extra}
    return asyncio.run(mcp._tool_manager._tools["start_probe_training_run"].fn(**arguments))  # noqa: SLF001


class TestStartingThroughTheTool:
    REFUSAL = {"code": "rule_choice_omits_recommendation", "message": "omits rolling_mean_max",
               "missing": ["rolling_mean_max"], "recommendation": {}}

    def test_a_run_the_route_would_refuse_is_refused_before_an_approval_is_filed(self):
        rec = {"rules": ["max", "rolling_mean_max"], "refusal": self.REFUSAL}
        studio = _Studio({"checked": True, "fits": True, "rule_recommendation": rec})
        out = _start(studio)
        assert (out["refused"], out["reason"], out["missing"]) == (
            "rule_choice_omits_recommendation", "omits rolling_mean_max", ["rolling_mean_max"])
        assert out["rule_recommendation"] == rec
        assert [c[0] for c in studio.calls] == ["/probe-monitors/runs/forecast"]

    def test_no_refusal_goes_on_to_the_approval(self):
        studio = _Studio({"checked": True, "fits": True, "rule_recommendation": {"rules": [], "refusal": None}})
        assert _start(studio)["status"] == "pending_approval"
        assert [c[0] for c in studio.calls] == ["/probe-monitors/runs/forecast", "/mcp/approvals"]
