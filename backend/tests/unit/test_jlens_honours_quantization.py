"""The J-lens loader must honour the quantization the model row asks for.

FOUND BY RUNNING A FIT, 2026-09-05. `_load` passed `dtype="auto"` and no
`quantization_config`, so a row configured Q8 was silently ignored for every
fit and readout. On gemma-4-12B that is not a fidelity question but a hard
stop: ~12.3B bf16 parameters are ~24.6 GB against a 23.56 GB card, and the fit
OOM'd during a forward pass with the model already resident.

THE PRECISION COMMENT ABOVE IT IS STILL RIGHT, AND IS NOT THIS. That one
guards against FORCING fp16 onto a bf16 checkpoint, which leaves the model
internally mixed and dies with "expected scalar type BFloat16 but found Half"
— observed on gemma-2-2b-it. bitsandbytes is a different mechanism: it swaps
the linear layers for quantized ones and leaves the rest in the checkpoint's
own dtype, which `dtype="auto"` still selects. Both properties have to hold at
once, so both are tested here.
"""

import inspect

import pytest

from src.models.model import QuantizationFormat


def _load_source() -> str:
    """The nested `_load` function ONLY — not the whole module.

    Three mutations survived against a whole-module search in this very file,
    every one of them because the string being asserted also appears in the
    FALLBACK loader twenty lines below the primary path. A test that cannot
    tell the path it is about from the path beside it is not about anything.
    """
    import ast
    import textwrap

    from src.services import jlens_model_registry

    module_src = inspect.getsource(jlens_model_registry)
    tree = ast.parse(module_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load":
            seg = ast.get_source_segment(module_src, node)
            if seg:
                return textwrap.dedent(seg)
    raise AssertionError("_load no longer exists in jlens_model_registry")


class TestTheRowsQuantizationReachesFromPretrained:
    def test_a_quantization_config_is_passed(self):
        src = _load_source()
        assert "quantization_config=quant_config" in src, (
            "the loader ignores the model row's quantization, so a 12B model "
            "loads at native dtype and cannot fit on a 24 GB card"
        )

    def test_it_is_derived_from_the_model_row(self):
        """Trace the ASSIGNMENT, not the presence of a string.

        `_load` contains the fallback loader too, and that reads the same
        `getattr(model_record, "quantization", None)`. So a mutation setting
        `quant_name = None` on the PRIMARY path left the string present via the
        fallback and survived twice — once against the whole module, once
        against `_load`'s own source. The question is what the variable that
        reaches `from_pretrained` is actually assigned from.
        """
        import ast

        tree = ast.parse(_load_source())

        # what does the primary from_pretrained get for quantization_config?
        passed = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "from_pretrained"
                    and getattr(getattr(node.func, "value", None), "id", "")
                    == "AutoModelForCausalLM"):
                for kw in node.keywords:
                    if kw.arg == "quantization_config":
                        passed = getattr(kw.value, "id", None)
        assert passed, (
            "the primary load passes no quantization_config, so the model "
            "row's setting is ignored"
        )

        # and where does THAT name come from?
        sources = [
            ast.unparse(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == passed for t in n.targets)
        ]
        assert sources, f"{passed} is never assigned"
        assert any("get_quantization_config" in src for src in sources), (
            f"{passed} is not derived from get_quantization_config; it is "
            f"assigned from {sources}"
        )

        # ...and the value fed to it must come off the model row.
        row_reads = [
            ast.unparse(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "quant_name" for t in n.targets)
        ]
        assert row_reads, "quant_name is never assigned"
        assert all("model_record" in r for r in row_reads), (
            f"quant_name is not read from the model row; assigned from "
            f"{row_reads} — setting Q8 on the model would change nothing"
        )

    @pytest.mark.parametrize(
        "fmt,quantized",
        [
            (QuantizationFormat.Q8, True),
            (QuantizationFormat.Q4, True),
            (QuantizationFormat.FP16, False),
            (QuantizationFormat.FP32, False),
        ],
    )
    def test_only_the_quantized_formats_produce_a_config(self, fmt, quantized):
        """FP16/FP32 must still load at native dtype — the whole point of the
        precision fix this sits beside."""
        from src.ml.model_loader import get_quantization_config

        cfg = get_quantization_config(fmt)
        assert (cfg is not None) is quantized, (
            f"{fmt} produced {cfg!r}; a config here would force a precision "
            f"the checkpoint may not use, and None would ignore the setting"
        )

    def test_the_native_dtype_selection_survives(self):
        """`dtype="auto"` must remain on the ACTUAL CALL.

        Replacing it with a forced dtype is the bug the surrounding comment
        records: gemma-2-2b-it died with "expected scalar type BFloat16 but
        found Half".

        Read out of the AST. The first version of this asserted
        `'dtype="auto"' in source` over the WHOLE MODULE, which another
        occurrence satisfied — so a mutation forcing fp16 on the real call
        survived. The companion check missed it too, because it looked for the
        literal `torch.float16` and the mutation spelled it
        `__import__("torch").float16`. Two string matches, one defect, both
        blind: the assertion has to be about the argument, not about text
        appearing somewhere in a file.
        """
        import ast

        tree = ast.parse(_load_source())
        loads = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "from_pretrained"
            and getattr(getattr(n.func, "value", None), "id", "")
            == "AutoModelForCausalLM"
        ]
        assert loads, "the model is no longer loaded via AutoModelForCausalLM"
        for call in loads:
            dtype = next(
                (kw.value for kw in call.keywords if kw.arg == "dtype"), None
            )
            assert dtype is not None, "the load no longer specifies a dtype"
            assert isinstance(dtype, ast.Constant) and dtype.value == "auto", (
                f"the load forces dtype={ast.unparse(dtype)}; on a bf16 "
                f"checkpoint that leaves the model internally mixed and the "
                f"forward pass dies before any readout arithmetic"
            )

    def test_an_unrecognised_value_falls_back_rather_than_raising(self):
        """A bad row must not make the model unloadable — it should load at
        native dtype and say so."""
        src = _load_source()
        assert "except ValueError:" in src
        assert "Unrecognised quantization" in src
