"""Attention capture — the `cap_2f3d68cc0956` failure and its neighbours.

A capture of granite-4.1-8b with attention enabled died with the bare message
`tuple index out of range`, ninety seconds in, at

    _append_attention(at_w, out.attentions[L], attn_cfg, doc_base, lengths)

Two independent defects met there.

`out.attentions` was `()`, not `None`. transformers 5 removed `output_attentions`
from the decoder forward; the replacement collects attention modules' outputs via
hooks and keeps them only when they are not None — and SDPA's kernel returns
`(attn_output, None)` always. So nothing was collected, the `is not None` guard
passed, and `()[34]` raised. Attention capture could not work at all on this
stack, and would have written an empty sidecar even if the index had been safe.

And `[L]` indexed by absolute decoder-layer number into a tuple holding one entry
per attention module that RAN. granite is dense so those coincide; LFM2 has
attention on 6 layers of 16, where they do not.

NO MOCKING of the thing under test — real `nn.Module`s throughout, matching the
style of `test_forward_hooks.py`. CPU only, no HF download, no GPU.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.ml.forward_hooks import HookManager, HookType
from src.ml.layer_discovery import attention_layer_indices, find_attention_module
from src.services import circuit_capture_service as cap


# ── a hybrid fixture: attention on layers 1 and 4 only ────────────────────

class FakeAttention(nn.Module):
    """Returns `(hidden, weights)` like every transformers attention module.

    `emit_weights=False` reproduces SDPA, which returns None in that slot.
    """

    def __init__(self, hidden: int, heads: int, peak_key: int, emit_weights=True):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)
        self.heads = heads
        self.peak_key = peak_key
        self.emit_weights = emit_weights

    def forward(self, x):
        b, s, h = x.shape
        out = self.o_proj(x)
        if not self.emit_weights:
            return out, None
        # A distinguishable pattern: all mass on `peak_key`, so a test can tell
        # WHICH layer's weights it received.
        w = torch.zeros(b, self.heads, s, s)
        w[:, :, :, min(self.peak_key, s - 1)] = 1.0
        return out, w


class FakeConvMixer(nn.Module):
    """An LFM2-style sequence mixer: no attention, returns a bare tensor."""

    def __init__(self, hidden: int):
        super().__init__()
        self.conv = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.conv(x)


class FakeLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, attention: bool, peak_key=0,
                 emit_weights=True):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = nn.Linear(hidden, hidden)
        if attention:
            self.self_attn = FakeAttention(hidden, heads, peak_key, emit_weights)
        else:
            self.conv = FakeConvMixer(hidden)

    def forward(self, x):
        mixer = getattr(self, "self_attn", None) or self.conv
        out = mixer(self.input_layernorm(x))
        if isinstance(out, tuple):
            out = out[0]
        x = x + out
        return x + self.mlp(self.post_attention_layernorm(x))


class FakeHybridLM(nn.Module):
    """6 layers, attention on {1, 4} — so layer 4 is attention-index 1."""

    HIDDEN, HEADS = 8, 2

    def __init__(self, emit_weights=True):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "fakehybrid"})()
        self.embed = nn.Embedding(32, self.HIDDEN)
        self.layers = nn.ModuleList([
            FakeLayer(self.HIDDEN, self.HEADS, attention=(i in (1, 4)),
                      # layer 1 peaks on the LAST key, layer 4 on the FIRST
                      peak_key=(0 if i == 4 else 99),
                      emit_weights=emit_weights)
            for i in range(6)
        ])
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        # Mimics transformers 5 under SDPA: the field exists and is EMPTY.
        return type("Out", (), {"attentions": (), "logits": x})()


class _RecordingWriter:
    """Stands in for AttnTopKWriter, capturing what would hit disk."""

    def __init__(self):
        self.rows = []
        self.count = 0

    def append(self, doc_ids, t_q, heads, keys, mass):
        self.rows.append((np.asarray(doc_ids), np.asarray(t_q),
                          np.asarray(heads), np.asarray(keys), np.asarray(mass)))
        self.count += len(np.asarray(t_q))


class IdentitySAE(nn.Module):
    """encode/decode that keep the capture path honest without a real SAE."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def encode(self, x):
        return torch.relu(x)

    def decode(self, z):
        return z


def _drive(model, attn_layers, *, top_k=2, heads=None, seq=5, docs=2):
    """Run `_capture_batch` for real and return the attention writer."""
    hidden = FakeHybridLM.HIDDEN
    saes = {L: IdentitySAE(hidden) for L in (1, 4)}
    layer_indices = sorted(saes)
    at = {L: _RecordingWriter() for L in layer_indices}

    class _EvW:
        count = 0

        def append(self, *a):
            pass

    writers = {L: (_EvW(), _EvW(), at[L]) for L in layer_indices}
    input_ids = torch.randint(0, 32, (docs, seq))
    mask = torch.ones(docs, seq, dtype=torch.long)
    attn_cfg = {"layers": list(attn_layers), "top_k": top_k, "heads": heads}

    cap._capture_batch(
        model, saes, layer_indices, writers, input_ids, mask, 0, [seq] * docs,
        epsilon_by_layer={L: 0.0 for L in layer_indices},
        floor_by_layer={L: 0.0 for L in layer_indices},
        probe_max={}, attn_cfg=attn_cfg,
    )
    return at


class TestTheOriginalCrash:
    def test_an_EMPTY_attentions_tuple_no_longer_ends_the_capture(self):
        """The exact production failure, without a GPU.

        The model returns `attentions=()` just as transformers 5 does under
        SDPA. The old code indexed it by absolute layer and raised
        `IndexError: tuple index out of range`; weights now come from hooks, so
        the empty field is simply irrelevant.
        """
        model = FakeHybridLM()
        assert model(input_ids=torch.zeros(1, 3, dtype=torch.long)).attentions == ()

        at = _drive(model, attn_layers=[4])

        assert at[4].count > 0, "no attention rows were captured at all"

    def test_it_reads_the_requested_ABSOLUTE_layer_not_a_tuple_position(self):
        """The hybrid case, which a dense fixture cannot detect.

        Attention lives on layers 1 and 4, so layer 4 is position 1 of the
        attention subsequence. Layer 4 puts all its mass on key 0; layer 1 puts
        it on the last key. Reading by position would return layer 1's tensor.
        """
        at = _drive(FakeHybridLM(), attn_layers=[4], seq=5)

        keys = np.concatenate([r[3] for r in at[4].rows])
        assert keys.size, "no rows captured"
        assert set(keys.tolist()) == {0}, (
            f"expected layer 4's key-0 peak, got keys {sorted(set(keys.tolist()))} "
            "— this is layer 1's pattern, i.e. positional indexing"
        )

    def test_SDPA_style_None_weights_is_REFUSED_not_silently_empty(self):
        """The condition that produced the bug must be loud.

        An empty sidecar looks like a completed capture; downstream it reads as
        "this model has no attention structure" rather than "we never recorded
        any".
        """
        model = FakeHybridLM(emit_weights=False)

        with pytest.raises(HookManager.AttentionWeightsUnavailable, match="eager"):
            _drive(model, attn_layers=[4])


class TestWhatGetsWritten:
    def test_future_keys_and_zero_mass_rows_are_dropped(self):
        """topk over the full square returns masked positions for early queries.

        A key the model could not see is not a weak edge; recording it with mass
        0.0 puts a position in the evidence that never participated.
        """
        at = _drive(FakeHybridLM(), attn_layers=[4], top_k=3, seq=5)

        for _doc, t_q, _hd, keys, mass in at[4].rows:
            assert (keys <= t_q).all(), "a key from the future was recorded"
            assert (mass > 0).all(), "a zero-mass row was recorded as an edge"

    def test_an_out_of_range_head_is_refused(self):
        with pytest.raises(cap.CaptureConfigError, match="out of range"):
            _drive(FakeHybridLM(), attn_layers=[4], heads=[0, 99])

    def test_a_layer_with_no_attention_module_is_refused(self):
        """Layer 2 is a conv mixer. Asking it for probabilities must not
        proceed with two of three layers quietly attached."""
        with pytest.raises(cap.CaptureConfigError, match="no attention module"):
            _drive(FakeHybridLM(), attn_layers=[2])


class TestTheStrictAttentionResolver:
    """`ATTENTION_PATTERNS` includes "conv" for LFM2's mixer, which has no
    probabilities. The weights hook must not accept one."""

    def test_it_finds_real_attention(self):
        layer = FakeLayer(8, 2, attention=True)
        assert find_attention_module(layer) is not None

    def test_it_REJECTS_a_conv_mixer(self):
        layer = FakeLayer(8, 2, attention=False)
        assert find_attention_module(layer) is None

    def test_attention_layers_are_identified_PER_LAYER(self):
        """`discover_transformer_structure` reads layer 0 and generalises,
        which is exactly wrong for a hybrid."""
        assert attention_layer_indices(FakeHybridLM().layers) == [1, 4]


class TestTheWeightsHookItself:
    def test_it_captures_index_1_not_index_0(self):
        """`create_hook` takes `output[0]` — the hidden states — under a comment
        naming the weights as the thing it drops. The shapes differ, so a flip
        is detectable."""
        model = FakeHybridLM()
        with HookManager(model) as hm:
            hm.register_hooks([4], [HookType.ATTENTION_WEIGHTS], "fakehybrid")
            model(input_ids=torch.randint(0, 32, (2, 5)))
            got = hm.activations["layer_4_attention_weights"][-1]

        assert got.shape == (2, FakeHybridLM.HEADS, 5, 5), (
            f"expected [b, heads, q, k], got {tuple(got.shape)} — index 0 is "
            "[b, seq, hidden]"
        )

    def test_a_second_register_call_reports_its_own_count(self):
        """`register_hooks` only raises when NOTHING is registered, so adding
        attention hooks after residual ones can attach zero and stay silent."""
        model = FakeHybridLM()
        with HookManager(model) as hm:
            hm.register_hooks([1, 4], [HookType.RESIDUAL], "fakehybrid")
            before = len(hm.hook_names)
            hm.register_hooks([1, 4], [HookType.ATTENTION_WEIGHTS], "fakehybrid")
            assert len(hm.hook_names) - before == 2


# ── the eager wiring: attention probabilities exist only under eager ──────

from unittest.mock import MagicMock, patch  # noqa: E402

from src.ml import model_loader  # noqa: E402


def _loader_stubs(model_obj):
    """Patch out everything `load_model_from_hf` touches except the model call."""
    cfg = MagicMock()
    cfg.model_type = "fakehybrid"
    cfg.num_hidden_layers = 6
    return (
        patch.object(model_loader.AutoConfig, "from_pretrained", return_value=cfg),
        patch.object(model_loader.AutoTokenizer, "from_pretrained",
                     return_value=MagicMock()),
        patch.object(model_loader, "extract_architecture_config", return_value={}),
    )


class TestTheLoaderCarriesTheAttentionBackend:
    def _call(self, from_pretrained, **kw):
        model = MagicMock()
        model.parameters.return_value = [torch.zeros(1)]
        model.config._attn_implementation = kw.get("attn_implementation") or "sdpa"
        from_pretrained.return_value = model
        a, b, c = _loader_stubs(model)
        with a, b, c, patch.object(model_loader.AutoModelForCausalLM,
                                   "from_pretrained", from_pretrained):
            return model_loader.load_model_from_hf("vendor/m", **kw)

    def test_eager_is_forwarded_when_asked_for(self):
        fp = MagicMock()
        _m, _t, _c, meta = self._call(fp, attn_implementation="eager")

        assert fp.call_args.kwargs["attn_implementation"] == "eager"
        assert meta["attn_implementation"] == "eager", (
            "the caller cannot verify the request took without this"
        )

    def test_a_DEFAULT_caller_sees_an_unchanged_call(self):
        """Training, extraction, steering and jlens must be untouched."""
        fp = MagicMock()
        self._call(fp)

        assert "attn_implementation" not in fp.call_args.kwargs

    def test_it_survives_the_OOM_FALLBACK(self):
        """The line that is always forgotten.

        The fallback re-invokes `load_model_from_hf` with an explicit kwarg
        list. Omitting `attn_implementation` there means an OOM retry quietly
        returns an SDPA model to a caller that asked for eager — and the only
        symptom is a capture that records nothing.
        """
        calls = []

        def fp(repo, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            model = MagicMock()
            model.parameters.return_value = [torch.zeros(1)]
            model.config._attn_implementation = kw.get("attn_implementation")
            return model

        a, b, c = _loader_stubs(None)
        with a, b, c, patch.object(model_loader.AutoModelForCausalLM,
                                   "from_pretrained", side_effect=fp):
            model_loader.load_model_from_hf(
                "vendor/m", attn_implementation="eager",
                quant_format=model_loader.QuantizationFormat.Q8)

        assert len(calls) >= 2, "the fallback did not run; the test proves nothing"
        assert calls[-1].get("attn_implementation") == "eager", (
            f"the fallback dropped the attention backend: {calls[-1]}"
        )


class TestSubmitTimeRefusals:
    """422 at the door. Every one of these used to be found on the GPU."""

    def _cfg(self, **over):
        cfg = {
            "dataset_id": "ds_1",
            "model_id": "m_1",
            "layers": [{"layer": 4, "sae_id": "sae_a"}],
            "attention_capture": {"layers": [4], "top_k": 4, "heads": None},
        }
        cfg["attention_capture"].update(over.pop("attention_capture", {}))
        cfg.update(over)
        return cfg

    def _validate(self, cfg, arch_config):
        model = MagicMock()
        model.architecture_config = arch_config
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = model
        capture_layers = {e["layer"] for e in cfg["layers"]}
        cap._validate_attention_config(db, cfg, "m_1", capture_layers)

    def test_a_valid_request_PASSES(self):
        """Positive control — a guard that refuses everything is not a guard."""
        self._validate(self._cfg(), {"num_hidden_layers": 40,
                                     "num_attention_heads": 32})

    def test_an_attention_layer_outside_the_capture_layers_is_refused(self):
        """It silently produced no sidecar: only capture layers get a writer."""
        with pytest.raises(cap.CaptureConfigError, match="not among"):
            self._validate(self._cfg(attention_capture={"layers": [9]}),
                           {"num_hidden_layers": 40})

    def test_a_layer_beyond_the_model_depth_is_refused(self):
        with pytest.raises(cap.CaptureConfigError, match="out of range"):
            self._validate(self._cfg(layers=[{"layer": 99, "sae_id": "s"}],
                                     attention_capture={"layers": [99]}),
                           {"num_hidden_layers": 40})

    def test_a_head_beyond_the_head_count_is_refused(self):
        with pytest.raises(cap.CaptureConfigError, match="heads"):
            self._validate(self._cfg(attention_capture={"heads": [0, 64]}),
                           {"num_hidden_layers": 40, "num_attention_heads": 32})

    def test_a_NON_ATTENTION_layer_is_refused_when_the_config_says_so(self):
        """KNOWN-BAD only. `layer_types` is authoritative when present."""
        types = ["conv"] * 6
        types[1] = "full_attention"
        with pytest.raises(cap.CaptureConfigError, match="not attention layers"):
            self._validate(self._cfg(), {"num_hidden_layers": 6,
                                         "layer_types": types})

    def test_a_model_WITHOUT_layer_types_is_not_guessed_about(self):
        """Dense models have no such key; refusing them would be a guess."""
        self._validate(self._cfg(), {"num_hidden_layers": 40})

    def test_attention_enabled_with_no_layers_is_refused(self):
        with pytest.raises(cap.CaptureConfigError, match="no layers"):
            self._validate(self._cfg(attention_capture={"layers": []}),
                           {"num_hidden_layers": 40})


class TestTheCaptureACTUALLYAsksForEager:
    """Reachability: a capability is not shipped until a test fails when its
    wiring is removed.

    Everything else here can be correct while `run_capture` forgets to request
    eager — in which case attention capture is exactly as broken as it was, and
    every other test still passes.
    """

    def _drive_to_the_load(self, attention: bool):
        """Run `run_capture` far enough to see the loader call, then let it go."""
        manifest = {
            "model_id": "m_1",
            "corpus": {"tokenization_id": "tok_1", "sample_cap": 4},
            "layers": [{"layer": 4, "sae_id": "sae_a", "epsilon": 0.1,
                        "theta_floor": 0.01}],
        }
        if attention:
            manifest["attention_capture"] = {"layers": [4], "top_k": 4}

        # ONE mock plays the run, the Model row and the tokenization row —
        # `db.query(X).filter(...).first()` returns the same object for each,
        # so it just needs every attribute all three paths read.
        row = MagicMock(manifest=manifest, tokenized_path="datasets/tok_1",
                        file_path=None, quantization="FP16")
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = row

        captured = {}

        def fake_load(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here — the kwargs are what we came for")

        # Both are LOCAL imports inside `run_capture`, so they must be patched
        # where they are defined, not on the service module.
        ds = MagicMock()
        ds.__len__ = lambda self: 4
        ds.select = lambda r: []
        with patch("datasets.load_from_disk", return_value=ds), \
             patch("src.ml.model_loader.load_model_from_hf", side_effect=fake_load):
            with pytest.raises(Exception):
                cap.CircuitCaptureService.run_capture(db, "cap_1", confirmed=True)
        return captured

    def test_attention_capture_requests_EAGER(self):
        kwargs = self._drive_to_the_load(attention=True)

        assert kwargs, "the loader was never reached; this test proves nothing"
        assert kwargs.get("attn_implementation") == "eager", (
            "the capture did not ask for eager attention, so SDPA would return "
            "None weights and the sidecar would be empty"
        )

    def test_a_capture_WITHOUT_attention_does_not_pay_for_eager(self):
        """Eager is materially slower; the default path must be untouched."""
        kwargs = self._drive_to_the_load(attention=False)

        assert kwargs, "the loader was never reached; this test proves nothing"
        assert kwargs.get("attn_implementation") is None


class TestTheResolvedBackendIsREPORTEDNotAssumed:
    def test_metadata_reflects_the_MODEL_not_the_request(self):
        """The capture refuses on this value, so it must come from the model.

        transformers can decline a request, and an OOM fallback can lose it.
        A metadata field that just echoes the argument would make the capture's
        guard vacuous — it would be checking its own input.
        """
        model = MagicMock()
        model.parameters.return_value = [torch.zeros(1)]
        model.config._attn_implementation = "sdpa"      # the model DECLINED
        fp = MagicMock(return_value=model)
        a, b, c = _loader_stubs(model)
        with a, b, c, patch.object(model_loader.AutoModelForCausalLM,
                                   "from_pretrained", fp):
            *_, meta = model_loader.load_model_from_hf(
                "vendor/m", attn_implementation="eager")

        assert meta["attn_implementation"] == "sdpa", (
            "metadata echoed the request instead of reading the model, so the "
            "capture's eager check can never fire"
        )


# ── the estimate: 20 billion events / 242 GB from 32 documents ────────────

class _FakeDataset:
    def __init__(self, n=8, seq=6):
        self._rows = [{"input_ids": list(range(seq))} for _ in range(n)]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, sl):
        rows = self._rows[sl]
        return {"input_ids": [r["input_ids"] for r in rows]}


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class TestTheEstimateMeasuresWhatWillBeWritten:
    """The confirm dialog's numbers have to describe the capture.

    They did not. The probe counted `z > 0` — every strictly positive encoder
    output, no threshold — while the capture writes only
    `z > clamp(eps * probe_max, floor)`. On granite-4.1-8b the unthresholded
    count came to 60% density against the 0.48% the SAE recorded in training,
    and the estimate read 20,192,354,625 events / 242 GB.
    """

    def test_the_threshold_rule_is_SHARED_by_probe_and_capture(self):
        """One definition, so the two cannot drift apart again."""
        z = torch.tensor([[0.0, 1.0, 5.0]])
        pm = torch.tensor([0.0, 10.0, 10.0])

        thr = cap._event_threshold(z, pm, eps=0.1, floor=0.01)

        assert torch.allclose(thr, torch.tensor([0.01, 1.0, 1.0]))

    def test_a_higher_epsilon_estimates_FEWER_events(self):
        """The property `z > 0` cannot have: sensitivity to the threshold.

        Under the old count the two runs below were identical, because epsilon
        never entered the estimate at all.
        """
        model = FakeHybridLM()
        saes = {1: IdentitySAE(FakeHybridLM.HIDDEN)}
        args = (model, _FakeTokenizer(), _FakeDataset(), saes, [1], "cpu")

        _pm_lo, lo_events, _t = cap._probe(
            *args, epsilon_by_layer={1: 0.01}, floor_by_layer={1: 0.0})
        _pm_hi, hi_events, _t = cap._probe(
            *args, epsilon_by_layer={1: 0.95}, floor_by_layer={1: 0.0})

        assert hi_events < lo_events, (
            f"epsilon does not affect the estimate ({hi_events} vs {lo_events}) "
            "— it is not counting what the capture will write"
        )


class TestTheSizeCeilingSeesTheSidecar:
    """Attention rows are buffered in host RAM until finalize().

    A default request is ~131M rows ≈ 1.5 GiB per attention layer. The running
    guard counted only `ev.count`, which was harmless for exactly as long as
    the sidecar could never contain anything.
    """

    def test_attention_rows_count_toward_the_buffered_total(self):
        """Calls the SHIPPED function, not a copy of its expression.

        The first version of this test grepped `run_capture`'s source for the
        expression, and stayed green against a mutation that kept the text and
        multiplied the result by zero. A scrape fails open.
        """
        writers = {4: (MagicMock(count=10), MagicMock(), MagicMock(count=1_000_000))}

        assert cap._buffered_rows(writers) == 1_000_010, (
            "the sidecar is invisible to the size guard, so an attention "
            "capture can OOM the worker with no ceiling in its way"
        )

    def test_a_capture_with_no_sidecar_still_counts_its_events(self):
        """Control: the non-attention path must be unchanged."""
        writers = {4: (MagicMock(count=10), MagicMock(), None)}

        assert cap._buffered_rows(writers) == 10

    def test_run_capture_actually_CALLS_the_helper(self):
        """A correct helper nothing calls is not a fix.

        Checked against the compiled code object's referenced names rather than
        the source text: an inline `sum(ev.count for ...)` that reproduces the
        old behaviour drops the reference, and this notices. It proves the call
        is present, not that it is reached at runtime — driving the batch loop
        needs a whole store on disk — so it is deliberately paired with the
        behavioural test above rather than replacing it.
        """
        names = cap.CircuitCaptureService.run_capture.__code__.co_names

        assert "_buffered_rows" in names, (
            "run_capture computes the buffered total itself again; the ceiling "
            "and the helper can now disagree"
        )
