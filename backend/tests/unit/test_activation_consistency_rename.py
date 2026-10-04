"""The "interpretability score" is activation consistency, and is presented as such.

THE DEFECT. `calculate_interpretability_score` is
`0.7 * consistency + 0.3 * sparsity` over BINARISED activation patterns. It
measures how similar a feature's activation SHAPE is across its top examples
and never inspects which tokens fire. Measured on the 16k L11 extraction, it
correlates at r = -0.50 with evidence diversity, monotonically:

    distinct prime tokens   mean score   cleared the "interpretable" bar
    1 (all one token)          0.714           98.8%
    21-25 (all distinct)       0.578           92.0%

So the headline "96% interpretable" was inflated by the least useful features,
and the same number went to Neuronpedia as each explanation's `scoreV1` — the
column that means how well an explanation predicts activations.

The formula and the database column are unchanged: the column is NOT NULL and
has live consumers, and inventing a replacement "interpretability" metric would
swap one unvalidated number for another. What changes is every claim about it.

MUTATION CONTROLS (each alone; suite must go red):
  R1  turn interpretability_score into an `alias` of activation_consistency
        -> test_both_names_serialise  (an alias renames the field ON OUTPUT,
           silently dropping the old wire field for every existing consumer)
  R2  restore `"score": feature.interpretability_score` in the export
        -> test_neuronpedia_gets_no_explanation_score
  R3  restore the "interpretable_count" statistic
        -> test_extraction_statistics_make_no_interpretability_claim
  R4  pass score=interpretability_score to the local Neuronpedia writer
        -> test_the_local_writer_sends_no_explanation_score
"""

import ast
import inspect
import textwrap
from datetime import datetime, timezone
from types import SimpleNamespace

from src.schemas.feature import FeatureResponse, FeatureStatistics
from src.services import extraction_service, neuronpedia_local_service
from src.services.neuronpedia_export_service import NeuronpediaExportService


def _feature_response():
    now = datetime.now(timezone.utc)
    return FeatureResponse(
        id="feat_1", neuron_index=7, name="x", description=None, label_source="auto",
        activation_frequency=0.4, interpretability_score=0.714, max_activation=5.0,
        mean_activation=2.0, is_favorite=False, created_at=now, updated_at=now,
        training_id="t", extraction_job_id="e",
    )


def test_both_names_serialise():
    """The honest name is ADDED; the deprecated one is not dropped."""
    body = _feature_response().model_dump()

    assert body["activation_consistency"] == 0.714
    assert body["interpretability_score"] == 0.714, "the old wire field was dropped"


def test_the_schema_publishes_the_new_field():
    """It must appear in the published schema, not only in a Python dump."""
    props = FeatureResponse.model_json_schema(mode="serialization")["properties"]

    assert "activation_consistency" in props
    assert "interpretability_score" in props


def test_feature_statistics_make_no_interpretability_claim():
    assert "interpretable_percentage" not in FeatureStatistics.model_fields


def test_neuronpedia_gets_no_explanation_score():
    """scoreV1 means explanation quality. We compute none, so we send none."""
    feature = SimpleNamespace(
        neuron_index=7, name="a label", label_source="auto",
        interpretability_score=0.714, labeled_at=None,
    )

    exported = NeuronpediaExportService()._generate_explanations_json([feature])

    blob = str(exported)
    assert "0.714" not in blob, "activation consistency leaked into an explanation score"


def test_neuronpedia_feature_json_names_it_honestly():
    feature = SimpleNamespace(
        neuron_index=7, activation_frequency=0.4, max_activation=5.0,
        mean_activation=2.0, interpretability_score=0.714,
    )

    stats = NeuronpediaExportService()._generate_feature_json(feature, None, [])["statistics"]

    assert stats["activation_consistency"] == 0.714
    assert "interpretability_score" not in stats


def _dict_keys(tree):
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys |= {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return keys


def test_extraction_statistics_make_no_interpretability_claim():
    tree = ast.parse(textwrap.dedent(inspect.getsource(extraction_service.ExtractionService)))
    keys = _dict_keys(tree)

    assert "interpretable_count" not in keys
    assert "avg_interpretability" not in keys
    assert "avg_activation_consistency" in keys


def test_the_local_writer_sends_no_explanation_score():
    """No call to create_explanation may pass the consistency score as `score`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(neuronpedia_local_service)))
    offending = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_explanation"):
            for kw in node.keywords:
                if kw.arg == "score" and "interpretability_score" in ast.unparse(kw.value):
                    offending.append(node.lineno)
    # The writer is still called — this guard must not pass by the call vanishing.
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "create_explanation"]
    assert calls, "create_explanation is no longer called; this guard would be vacuous"
    assert not offending, f"interpretability_score passed as an explanation score at {offending}"
