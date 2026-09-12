"""017 remediation: the statistical-estimate ablation must DISCLOSE it is not
causal (method field + docstring), and the MCP tool must point at the real
validation tier."""

import inspect


def test_ablation_response_has_method_field():
    from src.schemas.feature import AblationResponse
    fields = AblationResponse.model_fields
    assert "method" in fields
    assert fields["method"].default == "statistical_estimate"


def test_calculate_ablation_docstring_disclaims_causality():
    from src.services.analysis_service import AnalysisService
    doc = inspect.getdoc(AnalysisService.calculate_ablation) or ""
    assert "STATISTICAL ESTIMATE" in doc
    assert "NO model inference" in doc or "no inference" in doc.lower()
    assert "rung 2" in doc.lower() or "017" in doc


def test_mcp_ablation_docstring_points_to_validation():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "src" / "mcp_server"
           / "tools" / "features.py").read_text()
    # the get_feature_ablation docstring must relabel + redirect
    assert "statistical_estimate" in src or "STATISTICAL-ESTIMATE" in src
    assert "validate_circuit_edges" in src
