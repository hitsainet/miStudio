"""The `last_user` window and per-window length bands (operator, 2026-10-04).

`prompt` is everything before the reply, so on a client that resends the conversation a
high-stakes earlier turn kept firing on every later one, and a long system prompt or a retrieved
document diluted or triggered it. `last_user` reads the newest user message only. And a window's
score drifts with its OWN length distribution, so each window now carries its own length bands,
which must move with its bar or a later re-cut is a silent no-op for that window.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.services import probe_monitor_run
from src.services.probe_monitor_capture import _row_mask
from src.services.probe_monitor_metrics import length_band_decisions
from src.services.probe_monitor_render import CONTRACT_WINDOW_SCOPES, RenderedExample


def _conversation(roles):
    """One token per role entry, one message per entry, so spans are easy to read."""
    return RenderedExample(
        input_ids=list(range(2, 2 + len(roles))),
        token_roles=list(roles),
        token_message=list(range(len(roles))),
        text="",
    )


class TestTheLastUserScope:
    def test_it_reads_the_newest_user_message_only(self):
        row = _conversation(["system", "user", "assistant", "user", "assistant"])
        assert row.scored_mask("last_user") == [False, False, False, True, False]

    def test_it_is_not_the_user_scope(self):
        """`user` is every user turn — the history a multi-turn client resends."""
        row = _conversation(["user", "assistant", "user"])
        assert row.scored_mask("user") == [True, False, True]
        assert row.scored_mask("last_user") == [False, False, True]

    def test_a_row_with_no_user_turn_scores_nothing(self):
        assert _conversation(["system", "assistant"]).scored_mask("last_user") == [False, False]

    def test_an_unreliable_row_scores_nothing_rather_than_everything(self):
        """Its span is unknown; scoring the whole row would mix conversations into the bar."""
        row = _conversation(["user", "assistant"])
        row.role_mask_reliable = False
        assert _row_mask(row, "last_user") == [False, False]
        assert _row_mask(row, "input") == [True, True], "existing scopes keep their fallback"

    def test_it_is_a_contract_window(self):
        assert CONTRACT_WINDOW_SCOPES["last_user"] == "last_user"


def _persist(tmp_path: Path, name: str, values) -> str:
    path = tmp_path / f"{name}.npy"
    np.save(path, np.asarray(values, dtype=np.float32))
    return str(path)


def _window_probe(tmp_path, *, with_lengths=True):
    rng = np.random.default_rng(3)
    scores = [float(v) for v in rng.normal(0.0, 1.0, 800).astype(np.float32)]
    lengths = [int(v) for v in rng.integers(5, 400, 800)]
    entry = {
        "threshold": 2.0, "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": 800,
        "scope": "last_user",
        "scores_path": _persist(tmp_path, "scores", scores),
        "lengths_path": _persist(tmp_path, "lengths", lengths) if with_lengths else None,
        "length_bands": length_band_decisions(scores, lengths, target_fpr=0.01, global_threshold=2.0),
    }
    probe = SimpleNamespace(
        id="pm_x", window_decisions={"last_user": entry}, threshold_source="calibration_set",
    )
    return probe, scores, lengths


class TestAWindowsBandsMoveWithItsBar:
    def test_a_recut_recuts_the_windows_own_bands(self, tmp_path):
        from src.services.probe_recalibration import recut_windows

        probe, scores, lengths = _window_probe(tmp_path)
        decisions, unrecuttable = recut_windows(probe, target_fpr=0.05)
        assert unrecuttable == []
        moved = decisions["last_user"]
        assert moved["target_fpr"] == 0.05
        assert moved["length_bands"] == length_band_decisions(
            scores, lengths, target_fpr=0.05, global_threshold=moved["threshold"]
        )
        assert moved["length_bands"] != probe.window_decisions["last_user"]["length_bands"], (
            "the bands did not move with the bar — the window would serve its old bands"
        )

    def test_bands_without_their_lengths_make_the_window_unrecuttable(self, tmp_path):
        from src.services.probe_recalibration import recut_windows

        probe, _scores, _lengths = _window_probe(tmp_path, with_lengths=False)
        decisions, unrecuttable = recut_windows(probe, target_fpr=0.05)
        assert unrecuttable == ["last_user"] and "last_user" not in decisions

    def test_the_gpu_arm_never_keeps_a_window_it_could_not_place(self):
        """A kept band table was cut over a span the current code may no longer produce."""
        previous = {"prompt": {"scores_path": "p.npy", "lengths_path": "l.npy", "length_bands": [{}]}}
        table, dropped = probe_monitor_run.rescored_window_table("pm_x", previous, {"all": {}})
        assert dropped == ["prompt"] and "prompt" not in table


class TestTheStageCutsBandsPerWindow:
    def test_each_window_gets_bands_from_its_own_negatives_and_lengths(self, tmp_path):
        from src.services.probe_monitor_run import CalibrationScores, _calibrate_windows

        rng = np.random.default_rng(5)
        scores_by_scope = {}
        for scope in CONTRACT_WINDOW_SCOPES.values():
            negatives = [float(v) for v in rng.normal(0.0, 1.0, 600).astype(np.float32)]
            lengths = [int(v) for v in rng.integers(5, 900, 600)]
            scores_by_scope[scope] = CalibrationScores(negatives=negatives, n_rows=600, lengths=lengths)
        context = SimpleNamespace(target_fpr=0.01, scope="all", artifact_dir=tmp_path)
        probe = SimpleNamespace(id="pm_x", calibration_lengths_path=None)
        decisions = _calibrate_windows(probe, context, "calibration_set", scores_by_scope)
        for window, scope in CONTRACT_WINDOW_SCOPES.items():
            got = decisions[window]
            source = scores_by_scope[scope]
            assert got["length_bands"] == length_band_decisions(
                source.negatives, source.lengths, target_fpr=0.01, global_threshold=got["threshold"]
            )
            assert [int(v) for v in np.load(got["lengths_path"])] == source.lengths


class TestTheExportAndThePushCarryOnlyTheContract:
    def test_the_definition_exports_a_windows_bands_and_no_bookkeeping(self):
        from src.services.probe_definition_builder import _window_decisions

        band = {"min_tokens": 0, "max_tokens": None, "threshold": 3.0, "threshold_source": "band",
                "target_fpr": 0.01, "realised_fpr": 0.01, "n_negatives": 500}
        probe = SimpleNamespace(window_decisions={"last_user": {
            "threshold": 3.0, "scope": "last_user", "scores_path": "/data/x.npy",
            "lengths_path": "/data/y.npy", "length_bands": [band],
        }})
        out = _window_decisions(probe)
        dumped = out["last_user"].model_dump(exclude_none=True)
        assert dumped["length_bands"][0]["threshold"] == 3.0
        assert "scores_path" not in dumped and "lengths_path" not in dumped

    def test_the_millm_push_strips_bookkeeping(self):
        from src.mcp_server.tools.millm_probes import contract_windows

        pushed = contract_windows({
            "prompt": {"threshold": 1.0, "scores_path": "/data/a.npy", "lengths_path": "/data/b.npy",
                       "length_bands": [{"min_tokens": 0}]},
            "response": {"threshold": None},
        })
        assert pushed == {"prompt": {"threshold": 1.0, "length_bands": [{"min_tokens": 0}]}}

    def test_the_push_tool_uses_the_filter(self):
        import ast
        import inspect

        from src.mcp_server.tools import millm_probes

        tree = ast.parse(inspect.getsource(millm_probes))
        uses = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "contract_windows"
        ]
        assert len(uses) == 1


# ── Review round 1, H1/H2: the span starts at the message's own header, pinned in BOTH repos ──

import json as _json
import os as _os

_CASES_PATH = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "last-user-span-cases.json"
_MILLM_CASES = Path(_os.environ.get("MILLM_REPO", "/home/x-sean/app/miLLM")) / "docs" / "schemas" / "last-user-span-cases.json"
_CASES = _json.loads(_CASES_PATH.read_text())


def _tokenizer(prepend_bos: bool, template: str | None = None, case: dict | None = None):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = (case or {}).get("vocab") or _CASES["vocab"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    if (case or {}).get("pretokenizer") == "metaspace_first":
        from tokenizers import Regex

        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="first", split=True),
            pre_tokenizers.Split(Regex("\u2581?<\\|[a-z]+\\|>"), behavior="isolated"),
        ])
    if prepend_bos:
        tok.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", unk_token="[UNK]")
    fast.chat_template = template or _CASES["template"]
    return fast


@pytest.mark.parametrize("case", _CASES["cases"], ids=lambda c: c["name"])
def test_the_calibrated_span_matches_the_shared_cases(case):
    """The span miStudio calibrates `last_user` over — the shared file miLLM's server is held to."""
    from src.services.probe_monitor_render import render_messages

    tok = _tokenizer(case["prepend_bos"], case.get("template"), case)
    row = render_messages(tok, case["messages"])
    mask = row.scored_mask("last_user")
    span = [tok.convert_ids_to_tokens(i) for i, keep in zip(row.input_ids, mask) if keep]
    assert (span or None) == case["expected_span"]


def test_the_case_file_is_identical_in_millm():
    """⚠ TWO RESOLVERS, ONE SPAN. If miLLM's copy differs, the bar and the served window drift."""
    if not _MILLM_CASES.exists():
        if _os.environ.get("MISTUDIO_REQUIRE_CROSS_REPO_CHECKS") == "1":
            pytest.fail(f"miLLM's copy of the span cases is missing at {_MILLM_CASES}")
        pytest.skip("miLLM checkout not present")
    assert _MILLM_CASES.read_bytes() == _CASES_PATH.read_bytes()


def test_a_newest_message_cut_at_the_front_is_left_out():
    """Review round 1 (L1): a partial turn is not calibrated as though it were whole."""
    from src.services.probe_monitor_render import render_messages

    tok = _tokenizer(False)
    messages = [{"role": "user", "content": "virus spreads fast"}]
    full = render_messages(tok, messages)
    span_start = full.last_user_start
    cut = render_messages(tok, messages, max_length=len(full.input_ids) - span_start - 1)
    assert cut.truncated and cut.last_user_start is None
    assert not any(cut.scored_mask("last_user"))


def test_a_window_too_thin_for_its_budget_places_no_bar(tmp_path):
    """Review round 1 (M2): leaving empty rows out makes a thin window possible; a bar cut from
    fewer than 1/target negatives would fire on nothing and block every later re-cut."""
    from src.services.probe_monitor_run import CalibrationScores, _calibrate_windows

    rng = np.random.default_rng(9)
    scores = {
        scope: CalibrationScores(
            negatives=[float(v) for v in rng.normal(size=60 if scope == "last_user" else 300)],
            n_rows=300,
            lengths=[10] * (60 if scope == "last_user" else 300),
        )
        for scope in CONTRACT_WINDOW_SCOPES.values()
    }
    context = SimpleNamespace(target_fpr=0.01, scope="all", artifact_dir=tmp_path)
    decisions = _calibrate_windows(
        SimpleNamespace(id="pm_x", calibration_lengths_path=None), context, "calibration_set", scores
    )
    assert "last_user" not in decisions, "60 negatives cannot place a 1% bar"
    assert {"all", "prompt", "response"} <= set(decisions)


def test_a_cut_in_the_earlier_history_keeps_the_newest_turn_whole():
    """Review round 2: front truncation usually removes OLD turns. The newest message is intact
    then, and must be scored exactly as it would be untruncated — not dropped as partial."""
    from src.services.probe_monitor_render import render_messages

    tok = _tokenizer(False)
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "virus spreads fast"},
    ]
    full = render_messages(tok, messages)
    whole = [i for i, keep in zip(full.input_ids, full.scored_mask("last_user")) if keep]
    # Cut two tokens into the first message: history is partial, the newest turn is not.
    cut = render_messages(tok, messages, max_length=len(full.input_ids) - 2)
    assert cut.truncated and cut.last_user_start is not None
    kept = [i for i, keep in zip(cut.input_ids, cut.scored_mask("last_user")) if keep]
    assert kept == whole and [tok.convert_ids_to_tokens(i) for i in kept] == [
        "<|user|>", "virus", "spreads", "fast", "<|end|>"
    ]


def test_a_stale_cached_preamble_is_recomputed_not_refused():
    """Review round 3 (M1): a template that stamps today's date into its preamble makes a prefix
    cached yesterday match nothing today. One recompute before refusing recovers it."""
    from src.services import probe_monitor_render as render

    tok = _tokenizer(False)
    messages = [{"role": "user", "content": "virus spreads fast"}]
    good = render.render_messages(tok, messages)
    prefix, start = render.first_user_header(tok)
    # Yesterday's preamble: same length, one token different.
    stale = list(prefix)
    stale[1] = tok.convert_tokens_to_ids("brief")
    render._header_cache[tok][f"first:{render.template_hash(tok)}"] = (stale, start)
    again = render.render_messages(tok, messages)
    assert again.last_user_start == good.last_user_start is not None
    assert render.first_user_header(tok) == (prefix, start), "the recompute was not kept"
