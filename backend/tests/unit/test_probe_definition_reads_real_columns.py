"""Every ORM field the export reads must exist on the class it reads it from.

⚠ WHY THIS FILE EXISTS, AND WHY IT IS AN AST WALK RATHER THAN MORE TESTS.

033's builder was written against hand-built stand-ins — objects carrying whatever attribute the
code asked for. Fixtures that agree with the code by construction cannot disagree with it, so the
ORM was never consulted, and **five** field names were wrong at once in one file:

  1. `model_row.hf_repo_id`                  — the column is `repo_id`; AttributeError, every
                                               build crashed
  2. `getattr(model_row, "hf_revision")`     — no such column; the tier never fired
  3. `getattr(model_row, "metadata_")`       — no such column; that tier never fired either, so
                                               `resolve_model_revision`'s two "recorded" sources
                                               were both fiction
  4. `getattr(sae_row, "normalize_activations")` — lives at
                                               `sae_metadata["training_hyperparameters"]`; the
                                               default fired on every SAE and happened to be
                                               RIGHT for the one it was found on
  5. `getattr(view, "hf_id")`                — no such field; a fallback that could never fire

Only the first announced itself. Two silently disabled a code path, one silently guessed a value
that changes what a consumer computes, and one was decoration. Writing four more unit tests would
have pinned the four names I happened to think of; walking the module's attribute accesses pins
every name, including the ones added next year.

**The rule this file enforces:** a name read off an ORM row must be a column, relationship or
descriptor of that class. A `getattr` with a default is held to the same standard as a bare access
— a default that always fires is not a fallback, it is a fabricated value wearing a fallback's
clothes, and that is the shape of defect 4.

`ALLOWED_ABSENT` is the escape hatch, and it is deliberately awkward: an entry needs a reason, so
adding one is a decision rather than a reflex.
"""

import ast
import collections
import pathlib
from typing import Dict, Set, Tuple

import pytest
from sqlalchemy import inspect as sqla_inspect

from src.models.dataset import Dataset
from src.models.external_sae import ExternalSAE
from src.models.model import Model
from src.models.probe_monitor import ProbeMonitor, ProbeMonitorDataset, ProbeMonitorRun

#: Variable name in the source → the ORM class it is bound to. The builder names its rows
#: consistently, which is what makes this check possible; a new name reading an ORM row belongs
#: here, and a name NOT here is simply not checked.
#: It is PER MODULE, and it has to be. `dataset` is a `Dataset` row in the builder and a plain
#: manifest dict in the publisher, so one global map would report `dataset.get` as a missing column
#: — a false positive, and a guard that cries wolf gets widened until it says nothing.
BOUND_TO_BY_MODULE: Dict[str, Dict[str, type]] = {
    "src/services/probe_definition_builder.py": {
        "model_row": Model,
        "sae_row": ExternalSAE,
        "view": ProbeMonitorDataset,
        "dataset": Dataset,
        "probe": ProbeMonitor,
        "run": ProbeMonitorRun,
    },
    "src/services/probe_definition_publisher.py": {
        "probe": ProbeMonitor,
    },
}
BOUND_TO = BOUND_TO_BY_MODULE["src/services/probe_definition_builder.py"]

#: (variable, attribute) → why it is legitimately absent from the class.
ALLOWED_ABSENT: Dict[Tuple[str, str], str] = {}

MODULES = (
    "src/services/probe_definition_builder.py",
    "src/services/probe_definition_publisher.py",
)

# ⚠ WHY ONLY THESE TWO, when the same bug class could live anywhere.
#
# The 032 services were scanned the same way before this file was scoped, and every hit was a name
# collision rather than a defect: `model.parameters()` and `model.fit(...)` are a torch module and
# an sklearn estimator in `probe_monitor_capture`/`_trainer`, and in the endpoints `run` is a
# `ProbeMonitorJudgeRun` about as often as a `ProbeMonitorRun` — so `run.parse_failures` and
# `run.prompt_version` are correct reads of a class this map does not name.
#
# Widening the map to cover those modules means adding an ALLOWED_ABSENT entry per collision, and a
# guard whose escape list grows faster than its coverage is one that gets widened until it says
# nothing. The scan is cheap to re-run by hand against any module; this file pins the two where the
# bug class actually bit, and the bindings are unambiguous.


def _known_names(cls: type) -> Set[str]:
    mapper = sqla_inspect(cls)
    return (
        set(mapper.columns.keys())
        | set(mapper.relationships.keys())
        | set(mapper.all_orm_descriptors.keys())
        | set(dir(cls))
    )


def _reads(module_path: str) -> Dict[str, Set[Tuple[str, str]]]:
    """Every `<name>.<attr>` and `getattr(<name>, "attr", …)` for the names we track."""
    bound = BOUND_TO_BY_MODULE[module_path]
    tree = ast.parse(pathlib.Path(module_path).read_text())
    found: Dict[str, Set[Tuple[str, str]]] = collections.defaultdict(set)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in bound
        ):
            found[node.value.id].add(("bare", node.attr))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in bound
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            found[node.args[0].id].add(("getattr", node.args[1].value))
    return found


class TestTheExportReadsColumnsThatExist:
    @pytest.mark.parametrize("module_path", MODULES)
    def test_every_orm_read_names_a_real_field(self, module_path: str) -> None:
        missing = []
        bound = BOUND_TO_BY_MODULE[module_path]
        for variable, reads in _reads(module_path).items():
            known = _known_names(bound[variable])
            for kind, attribute in sorted(reads):
                if attribute in known:
                    continue
                if (variable, attribute) in ALLOWED_ABSENT:
                    continue
                missing.append(
                    f"{module_path}: {variable}.{attribute} ({kind}) is not a field of "
                    f"{bound[variable].__name__}"
                )
        assert not missing, (
            "these reads name fields the ORM does not have. A bare access raises; a getattr with "
            "a default silently fabricates a value. Either fix the name, or add it to "
            "ALLOWED_ABSENT with a reason:\n  " + "\n  ".join(missing)
        )

    def test_the_walk_actually_finds_something(self) -> None:
        """A scrape that matches nothing passes — this file must not be able to fail open.

        This repo has shipped source-scrape guards that asserted nothing because their pattern
        stopped matching after a refactor. The check above is only worth its runtime while it is
        still looking at reads.
        """
        reads = _reads("src/services/probe_definition_builder.py")
        assert "model_row" in reads and "sae_row" in reads and "probe" in reads
        assert sum(len(v) for v in reads.values()) > 30

    def test_the_five_names_that_were_wrong_stay_gone(self) -> None:
        """Named individually as well, because a regression here is a silent one.

        `ALLOWED_ABSENT` could be widened in a future round to get a build moving; these five are
        the ones that cost a day, so they are pinned by name rather than by policy.
        """
        reads = _reads("src/services/probe_definition_builder.py")
        assert ("bare", "hf_repo_id") not in reads["model_row"]
        assert ("getattr", "hf_revision") not in reads["model_row"]
        assert ("getattr", "metadata_") not in reads["model_row"]
        assert ("getattr", "normalize_activations") not in reads["sae_row"]
        assert ("getattr", "hf_id") not in reads["view"]

    def test_the_model_column_it_uses_now_is_the_real_one(self) -> None:
        assert "repo_id" in _known_names(Model)
        assert "hf_repo_id" not in set(sqla_inspect(Model).columns.keys())
        # And the dataset's, which was the first of the five to be found.
        assert "hf_repo_id" in set(sqla_inspect(Dataset).columns.keys())
        assert "dataset_id" in set(sqla_inspect(ProbeMonitorDataset).columns.keys())
        assert "dataset" not in set(sqla_inspect(ProbeMonitorDataset).relationships.keys())
