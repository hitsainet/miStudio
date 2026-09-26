"""The extraction loader must load the quantization it was asked for.

Reported 2026-08-23: a Q4 gemma-4-12B — 6 GB on disk — OOM'd a 23.56 GiB card,
dying while placing the last fragment of the model, before any activation was
read.

`_load_model` accepted `quantization: QuantizationFormat`, documented it,
logged it and wrote it into the extraction metadata, then called
`from_pretrained` with a hardcoded `torch_dtype=torch.float16` and no
`quantization_config`. 12B params at fp16 is 22.4 GB of weights. The setting
was recorded everywhere and applied nowhere.

`get_quantization_config()` had exactly one caller — `ml/model_loader.py`, the
OTHER load path. Two loaders, one honouring the format.

The second half matters as much: the GPU preflight computes from the REQUESTED
format, so for Q4 it calculated 8.7 GB and said FITS, then the loader pulled
22.4 GB. The preflight did not fail — it was bypassed, because it and the
loader disagreed about what was being loaded. `test_the_preflight_and_the_loader_agree`
is the invariant that was actually broken.
"""

import ast
import inspect
from pathlib import Path

import pytest

from src.models.model import QuantizationFormat


class TestTheConfigReachesFromPretrained:
    def test_load_model_passes_quantization_config(self):
        from src.services.activation_service import ActivationService

        source = inspect.getsource(ActivationService._load_model)
        tree = ast.parse(inspect.cleandoc(source))

        call = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "from_pretrained"),
            None,
        )
        assert call is not None, "the from_pretrained call is gone"
        kwargs = {k.arg: k.value for k in call.keywords}
        assert "quantization_config" in kwargs, (
            "from_pretrained is called without quantization_config, so the "
            "`quantization` argument this function accepts is ignored and "
            "every extraction loads fp16 — 22.4 GB for a 12B model"
        )
        # Presence is not enough: `quantization_config=None` keeps the kwarg
        # and restores the defect. Control C230 passed against exactly that
        # until this checked the VALUE.
        value = kwargs["quantization_config"]
        assert isinstance(value, ast.Name), (
            f"quantization_config is passed a literal "
            f"({ast.dump(value)[:60]}…) rather than the computed config; the "
            f"requested format is still ignored"
        )

    def test_the_argument_is_actually_used_not_just_accepted(self):
        """It was referenced only in the signature and the docstring."""
        from src.services.activation_service import ActivationService

        tree = ast.parse(inspect.cleandoc(inspect.getsource(ActivationService._load_model)))
        fn = tree.body[0]
        body_without_docstring = [
            n for n in fn.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
        ]
        uses = [
            n for stmt in body_without_docstring for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and n.id == "quantization"
        ]
        assert uses, (
            "`quantization` is accepted and never read in the body — recorded "
            "in metadata, applied to nothing"
        )

    def test_get_quantization_config_has_more_than_one_caller(self):
        """It had exactly one, in the other load path."""
        src_root = Path(__file__).resolve().parents[2] / "src"
        callers = set()
        for path in src_root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "get_quantization_config"):
                    callers.add(path.name)
        assert len(callers) >= 2, (
            f"only {sorted(callers)} applies quantization; the other load path "
            f"silently ignores it"
        )


class TestTheArithmeticThatCausedTheOom:
    """Numbers, so the regression is recognisable rather than abstract."""

    PARAMS = 12e9

    def test_fp16_does_not_fit_the_card(self):
        from src.services.resource_config import _BYTES_PER_PARAM

        gb = self.PARAMS * _BYTES_PER_PARAM["FP16"] / 1024 ** 3
        assert gb > 21.0, f"fp16 12B is {gb:.1f} GB — the premise changed"

    def test_q4_leaves_room_to_work(self):
        from src.services.resource_config import _BYTES_PER_PARAM

        gb = self.PARAMS * _BYTES_PER_PARAM["Q4"] / 1024 ** 3
        assert gb < 8.0, f"Q4 12B is {gb:.1f} GB"


class TestThePreflightAndTheLoaderAgree:
    """The invariant that was actually broken.

    The preflight sizes the job from the requested format. If the loader loads
    something else, the preflight is not wrong — it is measuring a different
    model than the one about to be loaded, and it will wave through exactly the
    job that OOMs.
    """

    def test_both_read_the_same_quantization_argument(self):
        from src.services import activation_service
        from src.services.resource_config import preflight_gpu_capacity

        assert "quantization" in inspect.signature(preflight_gpu_capacity).parameters
        assert "quantization" in inspect.signature(
            activation_service.ActivationService._load_model
        ).parameters

    @pytest.mark.parametrize("fmt", [QuantizationFormat.Q4, QuantizationFormat.Q8])
    def test_a_quantized_request_produces_a_bitsandbytes_config(self, fmt):
        from src.ml.model_loader import get_quantization_config

        cfg = get_quantization_config(fmt)
        assert cfg is not None, f"{fmt.value} yields no config, so it loads fp16"
        assert getattr(cfg, "load_in_4bit", False) or getattr(cfg, "load_in_8bit", False)

    def test_fp16_deliberately_yields_no_config(self):
        """None is correct here — `torch_dtype` carries it."""
        from src.ml.model_loader import get_quantization_config

        assert get_quantization_config(QuantizationFormat.FP16) is None
