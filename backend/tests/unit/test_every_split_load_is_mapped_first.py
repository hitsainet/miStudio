"""Every split load is mapped before a weight is read, not only the shared loader's.

Multi-GPU Phase 2, review round 2 (placement + loader), 2026-09-14.

Round 1 (bd38ae13) found that transformers 5.15.1 holds back the largest layer's
size on the LOWEST-index card when it maps a "sequential" split, for a CPU
put-back a GPU-only budget never makes. A split whose budgets held the model
then spilled a module to "disk" and was refused only after `from_pretrained`
had read every weight. On the node the lowest index is the 12 GB 3080 Ti. The
fix, `ml/split_load.plan_split_load`, maps the split with transformers' own
inference first. Round 1 wired it into `load_model_from_hf` and the local judge.

Two more split loads call `from_pretrained` themselves, and neither was wired:

* the J-lens registry's `load_for_readout`, behind every J-lens job that splits
  (readout, probe, band report, intervention, acquisition);
* `SteeringService.load_model`, behind steering and its cleanup paths.

Both still loaded with the placement's raw budget, so they spilled and were refused
after loading, exactly as the shared loader did before round 1. The J-lens registry
also sent an out-of-memory load to its fallback loader, which loaded the model AGAIN.
For an FP32 row with a bf16 checkpoint, that second load was at twice the size of the
first. That is the retry-at-a-larger-format f6124e22 removed from the shared loader.

The loads here run transformers' REAL map inference over a meta skeleton built
from the kwargs each loader passes (`_transformers_skeleton` +
`_get_device_map`), with the small card at index 0 as on the node. The AST guard
at the end makes the next direct split load fail until it is mapped first.

Red on c8fccbcd (before the fix): all five behavioural tests and the AST guard,
which names jlens_model_registry._load and SteeringService.load_model.

MUTATION CONTROLS (review round 2, 2026-09-14; scratchpad p2-r2-place/mutate.py,
each applied alone, this module run, source restored and checked by sha256):
  P1  J-lens: the plan's budget is not passed (max_memory=max_memory)
        -> test_a_jlens_split_its_budgets_hold_loads_on_its_gpus
  P2  J-lens: the refusal is swallowed (planned = None)
        -> test_a_jlens_split_its_budgets_hold..., test_a_jlens_split_the_cards_cannot_hold_is_refused_before_loading
  P3  J-lens: the planner call removed -> both J-lens split tests + the AST guard
  P4  J-lens: an out-of-memory load goes to the fallback loader again
        -> test_a_jlens_load_that_runs_out_of_memory_is_not_loaded_again
  P5  J-lens: the failed load's cards are not released
        -> test_a_jlens_load_that_runs_out_of_memory_is_not_loaded_again
  P6  steering: the plan's budget is not passed -> test_a_steering_split_its_budgets_hold_loads_on_its_gpus
  P7  steering: the planner call removed -> both steering split tests + the AST guard
  P8  steering: the refusal is swallowed -> test_a_steering_split_the_cards_cannot_hold_is_refused_before_loading
  P9  the guard no longer recognises from_pretrained -> test_the_guard_finds_the_split_loads_it_is_guarding
All 9 went red. Round 1's recorded controls on the lines this changed, re-run the same way:
  R-M2  J-lens primary load without max_memory= (test_jlens_split_gpu M2)      -> red (3 tests)
  R-M3  J-lens fallback without max_memory= (M3)                               -> red
  R-M10 steering split load without max_memory= (test_steering_gpu_placement M10) -> red (4 tests)
  R-N1  steering no longer re-reads a split's budget (R1-N1)                   -> red (2 tests)
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import math
import pathlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers.integrations.accelerate import _get_device_map

from src.services.gpu_placement import SHARD_RESERVE_MB, Placement
from tests.unit.test_jlens_split_gpu import _Loaded, _record
from tests.unit.test_split_load_maps_onto_the_gpus import (
    _cards,
    _config,
    _mapped_mib,
    _place,
    _sizes_mib,
    _transformers_skeleton,
)
from tests.unit.test_steering_gpu_placement import (  # noqa: F401 - `cuda` is a fixture
    CUDA0,
    CUDA1,
    RTX,
    TI,
    FakeModel,
    _service,
    cuda,
)

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"


def _holding_budgets(sizes: dict) -> tuple[float, float]:
    """Budgets that hold the model module for module ONLY IF the hold-back is returned:
    the 3080 Ti the embedding and three layers, the 3090 the rest."""
    budget0 = sizes["model.embed_tokens"] + 3 * sizes["model.layers.0"] + 1
    return budget0, sizes[""] - (budget0 - 1) + 1


def _short_budgets(sizes: dict) -> tuple[float, float]:
    """Budgets one and a half layers short of the model."""
    budget0 = sizes["model.embed_tokens"] + 3 * sizes["model.layers.0"] + 1
    return budget0, sizes[""] - (budget0 - 1) - 1.5 * sizes["model.layers.0"]


def _assert_on_its_gpus_within_budget(device_map: dict, sizes: dict, placement: Placement) -> None:
    assert not {d for d in device_map.values() if not isinstance(d, int)}, device_map
    mapped = _mapped_mib(device_map, sizes)
    for index, budget in placement.max_memory_mb.items():
        assert mapped.get(index, 0.0) <= budget, (index, mapped, placement.max_memory_mb)


# ── J-lens ───────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _jlens_hub(tmp_path, config, *, out_of_memory=False):
    """The registry's real `load_for_readout`; `from_pretrained` computes transformers' real map."""
    from src.services import jlens_model_registry as registry

    seen = SimpleNamespace(primary=[], fallback=[], maps=[], emptied=[])

    def primary(repo_id, **kwargs):
        seen.primary.append(kwargs)
        if out_of_memory:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        dtype = config.dtype if kwargs["dtype"] == "auto" else kwargs["dtype"]
        model, quantizer = _transformers_skeleton(
            config, dtype, kwargs.get("quantization_config"), kwargs["device_map"]
        )
        max_memory = kwargs.get("max_memory")
        device_map = _get_device_map(
            model, kwargs["device_map"], dict(max_memory) if max_memory else None, quantizer
        )
        seen.maps.append(device_map)
        return _Loaded(device_map)

    def fallback(**kwargs):
        seen.fallback.append(kwargs)
        return _Loaded({"": 1}), object(), None, {}

    with patch.object(registry, "_CACHE", registry._SingleEntryCache()), patch.object(
        type(registry.settings), "resolve_data_path", lambda self, raw: tmp_path
    ), patch("transformers.AutoModelForCausalLM.from_pretrained", primary), patch(
        "transformers.AutoConfig.from_pretrained", lambda *a, **k: copy.deepcopy(config)
    ), patch("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()), patch(
        "src.ml.model_loader.load_model_from_hf", fallback
    ), patch(
        "src.ml.layer_discovery.discover_transformer_structure",
        lambda model: SimpleNamespace(num_layers=2),
    ), patch("src.services.analysis_service.resolve_snapshot_dir", lambda *a: None), patch(
        "torch.cuda.is_available", lambda: True
    ), patch(
        "src.ml.model_devices.empty_cache_on",
        lambda devices: seen.emptied.append(sorted(str(d) for d in devices)),
    ):
        yield registry, seen


def _bf16_config():
    config = _config()
    config.dtype = torch.float16
    return config


def test_a_jlens_split_its_budgets_hold_loads_on_its_gpus(tmp_path):
    config = _bf16_config()
    sizes = _sizes_mib(config)
    placement = _place(_cards(*_holding_budgets(sizes)), required_mb=math.floor(sizes[""]))
    assert placement.is_shard, "the fixture must be a model no single card holds"

    with _jlens_hub(tmp_path, config) as (registry, seen):
        loaded = registry.load_for_readout(_record(), placement=placement)

    assert loaded is not None and seen.fallback == []
    assert len(seen.primary) == 1
    _assert_on_its_gpus_within_budget(seen.maps[0], sizes, placement)


def test_a_jlens_split_the_cards_cannot_hold_is_refused_before_loading(tmp_path):
    config = _bf16_config()
    sizes = _sizes_mib(config)
    budget0, budget1 = _short_budgets(sizes)
    placement = _place(_cards(budget0, budget1), required_mb=math.floor(budget0 + budget1 - 2))
    assert placement.is_shard

    with _jlens_hub(tmp_path, config) as (registry, seen):
        with pytest.raises(registry.ModelNotAvailable, match="does not fit on the GPUs") as refused:
            registry.load_for_readout(_record(), placement=placement)

    assert seen.primary == [], "the refusal came after from_pretrained had started reading weights"
    assert seen.fallback == [], "the refusal was taken for a failed load and loaded again"
    assert "disk" in str(refused.value) and "cuda:0" in str(refused.value)


def test_a_jlens_load_that_runs_out_of_memory_is_not_loaded_again(tmp_path):
    """The fallback loader forces the ROW's format: FP32 on a bf16 checkpoint is twice
    what just ran out. An out-of-memory load gives its cards back and says so."""
    from src.ml.model_loader import OutOfMemoryError

    config = _bf16_config()
    one_card = Placement(card=RTX, device=CUDA1)

    with _jlens_hub(tmp_path, config, out_of_memory=True) as (registry, seen):
        with pytest.raises(OutOfMemoryError, match="out of memory") as raised:
            registry.load_for_readout(_record(quantization="FP32"), placement=one_card)
        assert registry.loaded_model_key() is None

    assert len(seen.primary) == 1
    assert seen.fallback == [], "an out-of-memory load was loaded again through the fallback"
    assert seen.emptied == [["cuda:1"]]
    assert "org/model" in str(raised.value)


# ── Steering ─────────────────────────────────────────────────────────────────


@pytest.fixture
def steering_hub(monkeypatch, cuda):  # noqa: F811 - the imported fixture
    """`SteeringService.load_model` with transformers' real map computed from its kwargs."""
    import transformers

    from src.services import resource_config
    from src.services import steering_service as steering_module

    state = SimpleNamespace(config=_config(), loads=[], maps=[], free_mb={0: 0, 1: 0})

    def from_pretrained(path, **kwargs):
        model, quantizer = _transformers_skeleton(
            state.config, kwargs["torch_dtype"], kwargs.get("quantization_config"), kwargs["device_map"]
        )
        max_memory = kwargs.get("max_memory")
        device_map = _get_device_map(
            model, kwargs["device_map"], dict(max_memory) if max_memory else None, quantizer
        )
        state.loads.append(kwargs)
        state.maps.append(device_map)
        devices = [
            torch.device("cuda", target) if isinstance(target, int) else torch.device("meta")
            for target in device_map.values()
        ]
        return FakeModel(devices[0], devices=devices, hf_device_map=device_map)

    def mem_get_info(device=None):
        index = device.index if isinstance(device, torch.device) else device
        return state.free_mb[index] * 1024**2, 24_576 * 1024**2

    monkeypatch.setattr(steering_module.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(
        steering_module.AutoTokenizer, "from_pretrained",
        lambda *a, **k: SimpleNamespace(pad_token="<pad>", eos_token="</s>"),
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: copy.deepcopy(state.config))
    monkeypatch.setattr(resource_config, "preflight_gpu_capacity", lambda **kwargs: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
    return state


def _steering_split(state, budget0: float, budget1: float) -> Placement:
    """The 3090 first (most free), each card's free memory its budget plus the reserve."""
    state.free_mb = {0: math.ceil(budget0) + SHARD_RESERVE_MB, 1: math.ceil(budget1) + SHARD_RESERVE_MB}
    return Placement(
        card=RTX, device=CUDA1, cards=(RTX, TI), devices=(CUDA1, CUDA0),
        max_memory_mb={1: math.ceil(budget1), 0: math.ceil(budget0)},
    )


def test_a_steering_split_its_budgets_hold_loads_on_its_gpus(monkeypatch, steering_hub):
    sizes = _sizes_mib(steering_hub.config)
    placement = _steering_split(steering_hub, *_holding_budgets(sizes))
    svc = _service(monkeypatch)

    asyncio.run(svc.load_model("org/model", placement=placement))

    assert len(steering_hub.loads) == 1
    _assert_on_its_gpus_within_budget(steering_hub.maps[0], sizes, placement)


def test_a_steering_split_the_cards_cannot_hold_is_refused_before_loading(monkeypatch, steering_hub):
    sizes = _sizes_mib(steering_hub.config)
    placement = _steering_split(steering_hub, *_short_budgets(sizes))
    svc = _service(monkeypatch)

    with pytest.raises(RuntimeError, match="does not fit on the GPUs") as refused:
        asyncio.run(svc.load_model("org/model", placement=placement))

    assert steering_hub.loads == [], "the refusal came after from_pretrained had started reading weights"
    assert "disk" in str(refused.value)
    assert "org/model" not in svc._loaded_models


# ── The guard ───────────────────────────────────────────────────────────────


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _is_model_from_pretrained(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "from_pretrained"
        and ast.unparse(func.value).split(".")[-1] == "AutoModelForCausalLM"
    )


def _own_nodes(func: ast.AST):
    """The nodes of a function body, not of the functions nested in it."""
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _functions():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield path.relative_to(SRC), node


def _planners(functions) -> set[str]:
    """`plan_split_load` and every function in src/ that calls it."""
    names = {"plan_split_load"}
    for _path, func in functions:
        if any(isinstance(n, ast.Call) and _called_name(n) == "plan_split_load" for n in _own_nodes(func)):
            names.add(func.name)
    return names


def _names_max_memory(func) -> bool:
    for node in _own_nodes(func):
        if isinstance(node, ast.keyword) and node.arg == "max_memory":
            return True
        if isinstance(node, ast.Name) and node.id == "max_memory":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "max_memory":
            return True
        if isinstance(node, ast.Constant) and node.value == "max_memory":
            return True
    return False


def unplanned_split_loads(planners: set[str]) -> list[str]:
    """Functions that load a model with a split budget and call no planner before it."""
    offenders = []
    for path, func in _functions():
        loads = [n for n in _own_nodes(func) if isinstance(n, ast.Call) and _is_model_from_pretrained(n)]
        if not loads or not _names_max_memory(func):
            continue
        plans = [
            n.lineno for n in _own_nodes(func) if isinstance(n, ast.Call) and _called_name(n) in planners
        ]
        first_load = min(n.lineno for n in loads)
        if not plans or min(plans) > first_load:
            offenders.append(f"{path}:{func.name}")
    return sorted(offenders)


def test_every_split_from_pretrained_is_mapped_before_it_loads():
    """A function that hands `from_pretrained` a split budget calls the planner first.

    Asserted on CALLS found by walking the AST, never on text: a comment naming the
    planner satisfies nothing."""
    functions = list(_functions())
    assert unplanned_split_loads(_planners(functions)) == []


def test_the_guard_finds_the_split_loads_it_is_guarding():
    """Not vacuous: with no planner, it names every split load in src/."""
    offenders = unplanned_split_loads(set())
    assert "ml/model_loader.py:load_model_from_hf" in offenders
    assert "services/jlens_model_registry.py:_load" in offenders
    assert "services/steering_service.py:load_model" in offenders
    assert "services/local_labeling_service.py:load_model" in offenders
