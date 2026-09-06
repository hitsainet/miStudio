"""Cluster-level circuit edges must reach the hazard analysis.

BR-016 makes a cluster a first-class circuit member (`member_kind: cluster_ref`),
so a circuit's edges can be cluster-level. `detect_hazards` keys on
`(layer, feature_idx, …)` and a cluster endpoint has no `feature_idx`, so every
one of those edges was skipped.

The failure was silent and backwards. An edge that reached rung 2 carries a
MEASURED effect size; the fallback is a weight-prior heuristic explicitly
labelled `heuristic`. So steering a cluster-membered circuit discarded precisely
the strongest evidence available and reported an empty hazard list — which reads
as "safe", not as "not analysed".
"""

import pytest

from src.services import steering_hazards
from src.services.steering_hazards import expand_cluster_edges


def _cluster_edge(up_pid="cp_up", down_pid="cp_down", rung=2, es=0.8):
    return {
        "up": {"layer": 4, "cluster_profile_id": up_pid, "feature_idx": None},
        "down": {"layer": 9, "cluster_profile_id": down_pid, "feature_idx": None},
        "rung": rung,
        "effect_size": es,
    }


class TestAClusterEdgeBecomesTheFeaturePairsItStandsFor:
    def test_it_expands_to_the_cartesian_product(self):
        members = {"cp_up": [11, 12], "cp_down": [21, 22, 23]}
        edges, unresolved = expand_cluster_edges([_cluster_edge()], members.get)

        assert unresolved == []
        assert len(edges) == 6, "2 upstream x 3 downstream feature pairs"
        assert {(e["up"]["feature_idx"], e["down"]["feature_idx"]) for e in edges} == {
            (11, 21), (11, 22), (11, 23), (12, 21), (12, 22), (12, 23)
        }

    def test_each_pair_INHERITS_the_edge_evidence(self):
        """A supernode edge's rung and effect size apply to what it covers."""
        edges, _ = expand_cluster_edges(
            [_cluster_edge(rung=2, es=0.8)], {"cp_up": [11], "cp_down": [21]}.get
        )
        assert edges[0]["rung"] == 2
        assert edges[0]["effect_size"] == 0.8
        assert edges[0]["expanded_from_cluster_edge"] is True, (
            "a consumer must be able to tell an inherited effect size from one "
            "measured on this feature pair directly"
        )

    def test_a_feature_level_edge_is_passed_through_UNTOUCHED(self):
        plain = {"up": {"layer": 4, "feature_idx": 11},
                 "down": {"layer": 9, "feature_idx": 21}, "rung": 2}
        edges, unresolved = expand_cluster_edges([plain], {}.get)
        assert edges == [plain] and unresolved == []
        assert "expanded_from_cluster_edge" not in edges[0]

    def test_a_MIXED_edge_expands_only_the_cluster_side(self):
        mixed = {"up": {"layer": 4, "cluster_profile_id": "cp_up", "feature_idx": None},
                 "down": {"layer": 9, "feature_idx": 21}, "rung": 2}
        edges, _ = expand_cluster_edges([mixed], {"cp_up": [11, 12]}.get)
        assert {(e["up"]["feature_idx"], e["down"]["feature_idx"]) for e in edges} == {
            (11, 21), (12, 21)
        }


class TestUnresolvableProfilesAreReportedNotDropped:
    """"No hazards" and "not analysed" are different claims."""

    def test_a_missing_profile_is_REPORTED(self):
        edges, unresolved = expand_cluster_edges([_cluster_edge()], {}.get)

        assert edges == []
        assert len(unresolved) == 1
        assert "could not be checked" in unresolved[0]["reason"]

    def test_an_EMPTY_profile_is_reported_too(self):
        edges, unresolved = expand_cluster_edges(
            [_cluster_edge()], {"cp_up": [], "cp_down": [21]}.get
        )
        assert edges == [] and len(unresolved) == 1

    def test_a_raising_resolver_does_not_take_the_allocation_down(self):
        def boom(_pid):
            raise RuntimeError("db gone")

        edges, unresolved = expand_cluster_edges([_cluster_edge()], boom)
        assert edges == [] and len(unresolved) == 1


class TestExpansionIsBoundedByWhatIsBeingSteered:
    def test_keep_filters_to_the_reachable_pairs(self):
        """Two 20-feature clusters are 400 edges, of which few are reachable.

        `detect_hazards` only ever looks up pairs drawn from the steered list,
        so filtering to it is exactly equivalent and keeps the product small.
        """
        members = {"cp_up": list(range(20)), "cp_down": list(range(100, 120))}
        edges, _ = expand_cluster_edges(
            [_cluster_edge()], members.get, keep={(4, 3), (9, 107)}
        )
        assert len(edges) == 1
        assert (edges[0]["up"]["feature_idx"], edges[0]["down"]["feature_idx"]) == (3, 107)

    def test_without_keep_nothing_is_filtered(self):
        members = {"cp_up": [1, 2], "cp_down": [3, 4]}
        edges, _ = expand_cluster_edges([_cluster_edge()], members.get, keep=None)
        assert len(edges) == 4


class TestTheExpandedEdgesActuallyPRODUCEHazards:
    """End to end: the point is not the expansion, it is the warning."""

    def test_a_rung2_cluster_edge_now_yields_a_hazard(self):
        steered = [
            {"layer": 4, "feature_idx": 11, "strength": 1.0},
            {"layer": 9, "feature_idx": 21, "strength": 1.0},
        ]
        raw = [_cluster_edge(rung=2, es=0.8)]

        before = steering_hazards.detect_hazards(steered, circuit_edges=raw)
        assert before == [], "precondition: the unexpanded edge is unusable"

        expanded, _ = expand_cluster_edges(raw, {"cp_up": [11], "cp_down": [21]}.get)
        after = steering_hazards.detect_hazards(steered, circuit_edges=expanded)

        assert after, (
            "a causally-validated cluster-level edge produced no hazard — the "
            "strongest evidence available is the evidence being discarded"
        )


class TestAnInheritedEffectSizeSaysSo:
    """A cluster-scale ES must not render as a per-pair measurement.

    `A_C(t) = max_k a_{l,i_k}(t)` (Appendix A.4), so a cluster edge's effect
    size was measured on a signal that at any token is ONE member's activation.
    Resolving the edge to feature membership at steering time is what A.4
    prescribes — but the number belongs to the cluster pair, and printing it as
    `validated:ES=0.800` to three decimals claims it was measured here.

    This module already separates measured from heuristic. This is the third
    case: measured-HERE from inherited.
    """

    def _hazard_from(self, edges):
        steered = [
            {"layer": 4, "feature_idx": 11, "strength": 1.0},
            {"layer": 9, "feature_idx": 21, "strength": 1.0},
        ]
        h = steering_hazards.detect_hazards(steered, circuit_edges=edges)
        assert h, "precondition: a hazard was produced"
        return h[0]

    def test_an_expanded_cluster_edge_is_FLAGGED(self):
        expanded, _ = expand_cluster_edges(
            [_cluster_edge(rung=2, es=0.8)], {"cp_up": [11], "cp_down": [21]}.get
        )
        h = self._hazard_from(expanded)

        assert h.inherited_from_cluster_edge is True
        assert "inherited" in h.evidence
        assert "not measured on this feature pair" in h.evidence

    def test_a_DIRECTLY_measured_edge_is_not_flagged(self):
        """The control. Without it the flag could be hardcoded True."""
        direct = [{
            "up": {"layer": 4, "feature_idx": 11},
            "down": {"layer": 9, "feature_idx": 21},
            "rung": 2, "effect_size": 0.8,
        }]
        h = self._hazard_from(direct)

        assert h.inherited_from_cluster_edge is False
        assert h.evidence == "validated:ES=0.800"
        assert "inherited" not in h.evidence

    def test_the_two_are_DISTINGUISHABLE_in_the_serialised_form(self):
        """The caller reads to_dict(), so the flag has to survive it."""
        expanded, _ = expand_cluster_edges(
            [_cluster_edge(rung=2, es=0.8)], {"cp_up": [11], "cp_down": [21]}.get
        )
        inherited = self._hazard_from(expanded).to_dict()
        direct = self._hazard_from([{
            "up": {"layer": 4, "feature_idx": 11},
            "down": {"layer": 9, "feature_idx": 21},
            "rung": 2, "effect_size": 0.8,
        }]).to_dict()

        assert inherited["inherited_from_cluster_edge"] is True
        assert direct["inherited_from_cluster_edge"] is False
        assert inherited["evidence"] != direct["evidence"]

    def test_the_ES_ITSELF_is_NOT_apportioned(self):
        """The tempting fix is worse than the problem.

        Dividing the cluster ES by member count would look numerically modest
        while being exactly as unmeasured — and it would SUPPRESS warnings,
        which is the dangerous direction. A.4's answer to "which member pairs
        carry it" is to measure them, never to apportion.
        """
        members = {"cp_up": [11, 12, 13, 14], "cp_down": [21, 22, 23, 24]}
        expanded, _ = expand_cluster_edges([_cluster_edge(rung=2, es=0.8)], members.get)

        assert all(e["effect_size"] == 0.8 for e in expanded), (
            "the effect size was scaled by membership — that is an unmeasured "
            "assumption that quietly reduces warnings"
        )
