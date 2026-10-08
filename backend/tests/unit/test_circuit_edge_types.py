"""Pins for the edge-type classifier (018 Task 4.1, BR-020/021, A.9)."""

import json
from pathlib import Path

from src.services.circuit_edge_type_service import (
    classify_edge,
    label_similarity,
    token_identity_overlap,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "edge_type_audit.json"


class TestSignals:
    def test_token_overlap_jaccard(self):
        assert token_identity_overlap(["A", "b"], ["b", "c"]) == 1 / 3
        assert token_identity_overlap([], ["x"]) == 0.0

    def test_label_similarity_degrades_honestly(self):
        sim, method = label_similarity("fear_response", "fear_response")
        assert sim == 1.0 and method == "token_set"
        assert label_similarity(None, "x") == (0.0, "absent")


class TestClassifier:
    def test_two_of_three_rule(self):
        r = classify_edge(weight_prior=0.95,
                          up_top_tokens=["fear", "afraid"], down_top_tokens=["fear", "afraid"],
                          up_label="a", down_label="b")
        assert r["type"] == "persistence"
        assert r["signals"]["votes"]["weight_prior"] and r["signals"]["votes"]["token_overlap"]

    def test_low_prior_high_association_is_computed_not_penalized(self):
        r = classify_edge(weight_prior=0.05,
                          up_top_tokens=["died"], down_top_tokens=["condolences"],
                          up_label="death_mention", down_label="condolence_expression")
        assert r["type"] == "computed"
        assert r["signals"]["distinctness"] > 0.5  # ranking-facing value stays high

    def test_attention_mediated_reserved_for_head_evidence(self):
        r = classify_edge(weight_prior=0.99, up_top_tokens=["x"], down_top_tokens=["x"],
                          up_label="x", down_label="x", mediating_heads=[1])
        assert r["type"] == "attention_mediated"

    def test_signals_fully_disclosed(self):
        r = classify_edge(weight_prior=0.5, up_top_tokens=["a"], down_top_tokens=["b"],
                          up_label=None, down_label=None)
        s = r["signals"]
        assert {"weight_prior", "token_overlap", "label_similarity", "label_method",
                "thresholds", "votes", "echo_confidence", "distinctness"} <= set(s)


class TestAuditFixture:
    """The BR-021 regression gates: >=90% persistence recall, <=10% computed misclass."""

    def test_gates(self):
        cases = json.loads(FIXTURE.read_text())["cases"]
        persistence = [c for c in cases if c["expected"] == "persistence"]
        computed = [c for c in cases if c["expected"] == "computed"]

        def run(c):
            return classify_edge(
                weight_prior=c["weight_prior"],
                up_top_tokens=c["up_top_tokens"], down_top_tokens=c["down_top_tokens"],
                up_label=c["up_label"], down_label=c["down_label"],
                mediating_heads=c.get("mediating_heads"),
            )["type"]

        p_recall = sum(run(c) == "persistence" for c in persistence) / len(persistence)
        c_misclass = sum(run(c) != "computed" for c in computed) / len(computed)
        assert p_recall >= 0.9, f"persistence recall {p_recall:.0%}"
        assert c_misclass <= 0.1, f"computed misclassification {c_misclass:.0%}"

        for c in cases:
            if c["expected"] == "attention_mediated":
                assert run(c) == "attention_mediated"


class TestDistinctnessPins:
    """R2-T3: pin the R1-fixed ranking semantics so the exact regression
    (lone strong signal de-ranking a computed edge) can never silently return."""

    def test_lone_strong_prior_keeps_computed_distinctness_at_one(self):
        r = classify_edge(weight_prior=0.99, up_top_tokens=["alpha"],
                          down_top_tokens=["omega"], up_label="a", down_label="b")
        assert r["type"] == "computed"
        assert r["signals"]["distinctness"] == 1.0

    def test_persistence_vote_counts_are_distinguishable(self):
        two = classify_edge(weight_prior=0.95, up_top_tokens=["x"], down_top_tokens=["y"],
                            up_label="same_thing", down_label="same_thing")
        three = classify_edge(weight_prior=0.95, up_top_tokens=["x"], down_top_tokens=["x"],
                              up_label="same_thing", down_label="same_thing")
        assert two["type"] == three["type"] == "persistence"
        assert three["signals"]["echo_confidence"] > two["signals"]["echo_confidence"]
