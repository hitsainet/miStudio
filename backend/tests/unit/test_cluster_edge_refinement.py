"""A.4 refinement: which member pairs carry a cluster-level edge.

A supernode's activation is `A_C(t) = max_k a_{l,i_k}(t)`, so a cluster edge's
effect size was measured on a signal that at any token is ONE member's
activation. The edge says the two clusters are linked; it does not say which
members do the linking, and its number cannot be apportioned to member pairs by
any arithmetic — apportioning would look modest while being exactly as
unmeasured, and would quietly shrink a hazard.

Appendix A.4's answer is to MEASURE: the A.3 statistics restricted to the two
clusters' members. These tests pin that it runs the same statistics as a
discovery pass, over the right candidate set, and that it never inherits a rung.
"""

import numpy as np
import pytest

from src.services import circuit_discovery_service as discovery
from src.services.circuit_discovery_service import DiscoveryConfigError


class TestItRefusesWhatItCannotMeasure:
    def test_a_backwards_edge_is_REFUSED(self):
        """Refinement runs upstream→downstream; A.3 is not symmetric in intent."""
        with pytest.raises(DiscoveryConfigError, match="upstream"):
            discovery.refine_cluster_edge(
                None, "cap_1",
                {"layer": 9, "members": [{"feature_idx": 1}]},
                {"layer": 4, "members": [{"feature_idx": 2}]},
            )

    def test_a_SAME_LAYER_edge_is_refused(self):
        with pytest.raises(DiscoveryConfigError, match="upstream"):
            discovery.refine_cluster_edge(
                None, "cap_1",
                {"layer": 4, "members": [{"feature_idx": 1}]},
                {"layer": 4, "members": [{"feature_idx": 2}]},
            )

    def test_an_EMPTY_membership_is_refused_not_silently_empty(self):
        """A cluster with no resolvable members cannot be refined.

        Returning an empty result would read as 'no member pair carries this
        edge', which is a claim. Refusing says we could not look.
        """
        with pytest.raises(DiscoveryConfigError, match="no members"):
            discovery.refine_cluster_edge(
                None, "cap_1",
                {"layer": 4, "members": []},
                {"layer": 9, "members": [{"feature_idx": 2}]},
            )


class TestMemberUnitsAreNotCappedOrTrimmed:
    """`_feature_units` trims to `max_units_per_layer` by support. Neither
    applies to a membership the reviewer explicitly asked about."""

    def _reader(self, per_feature):
        class R:
            layer = 4
            feature_ids = list(per_feature)

            def feature_events(self, fid):
                n = per_feature.get(int(fid), 0)
                return np.array(
                    [(0, i, 1.0) for i in range(n)],
                    dtype=[("doc_id", "u4"), ("token_pos", "u2"), ("act", "f4")],
                )
        return R()

    def test_every_member_above_support_is_kept(self):
        per = {i: 50 for i in range(40)}
        units = discovery._member_units(
            self._reader(per), 4, list(per), np.array([0], dtype=np.uint32), s_min=20)
        assert len(units) == 40, "a membership must not be trimmed by a unit cap"

    def test_a_member_BELOW_support_is_excluded(self):
        per = {1: 50, 2: 3}
        units = discovery._member_units(
            self._reader(per), 4, [1, 2], np.array([0], dtype=np.uint32), s_min=20)
        assert [u["feature_idx"] for u in units] == [1]


class TestTheEvidenceIsNotInflated:
    """The refinement produces association statistics, and says so."""

    def test_a_measured_pair_is_MARKED_as_measured_here(self, monkeypatch):
        out = _run_fake_refinement(monkeypatch)
        assert out["member_pairs"], "precondition: a pair survived"
        assert all(p["measured_on_this_pair"] is True for p in out["member_pairs"])
        assert all("null_threshold_n_ud" in p for p in out["member_pairs"])

    def test_no_rung_is_inherited_from_the_cluster_edge(self, monkeypatch):
        out = _run_fake_refinement(monkeypatch)
        for p in out["member_pairs"]:
            assert "rung" not in p, (
                "a refinement pair carries a rung — A.3 statistics are "
                "association; rung 2 is earned by an intervention (A.5)"
            )
        assert "intervention" in out["evidence_note"]

    def test_excluded_members_are_NAMED_on_BOTH_sides(self, monkeypatch):
        """"This member does not carry it" and "this member was never tested"
        are different answers, and an absent row reads as the first.

        BOTH SIDES ASSERTED. The first version of this test checked only the
        upstream exclusion, and a mutation that hid every downstream one left it
        green — the exact "fixed one representative, never generalized" shape
        this repo has hit repeatedly.
        """
        out = _run_fake_refinement(
            monkeypatch,
            up_support={1: 50, 2: 2},          # 2 is below the floor
            down_support={100: 50, 101: 3},    # 101 is below the floor
        )
        excluded = {(e["layer"], e["feature_idx"]) for e in out["excluded_members"]}
        assert (4, 2) in excluded, "an untested UPSTREAM member was not named"
        assert (9, 101) in excluded, "an untested DOWNSTREAM member was not named"
        assert all("support" in e["reason"] for e in out["excluded_members"])

    def test_a_capped_null_pass_is_DISCLOSED(self, monkeypatch):
        out = _run_fake_refinement(
            monkeypatch, up_support={i: 50 for i in range(1, 6)},
            down_support={i: 50 for i in range(100, 105)}, max_null_tested=3)
        assert out["null_capped"] is True, (
            "a truncated drill-down that looked complete would answer 'which "
            "members carry this edge' with a silent subset"
        )
        assert out["params"]["max_null_tested"] == 3


def _run_fake_refinement(monkeypatch, up_support=None, down_support=None,
                         max_null_tested=200):
    """Drive the real function against an in-memory capture."""
    up_support = up_support or {1: 50}
    down_support = down_support or {100: 50}
    per_layer = {4: up_support, 9: down_support}

    class R:
        def __init__(self, layer):
            self.layer = layer
            self._per = per_layer[layer]
            self.feature_ids = list(self._per)

        def feature_events(self, fid):
            n = self._per.get(int(fid), 0)
            return np.array([(0, i, 1.0) for i in range(n)],
                            dtype=[("doc_id", "u4"), ("token_pos", "u2"), ("act", "f4")])

    monkeypatch.setattr(discovery, "open_capture", lambda db, cid: discovery.OpenedCapture(
        manifest={}, store_dir="/nope",
        doc_lengths={0: 200}, heldout=np.array([], dtype=np.uint32),
        discovery_docs=np.array([0], dtype=np.uint32),
        n_tokens_discovery=200, sae_by_layer={}))
    monkeypatch.setattr(discovery, "layer_files_exist", lambda d, L: True)
    monkeypatch.setattr(discovery, "EventReader", lambda d, L: R(L))

    return discovery.refine_cluster_edge(
        None, "cap_1",
        {"layer": 4, "cluster_profile_id": "cp_up",
         "members": [{"feature_idx": i} for i in up_support]},
        {"layer": 9, "cluster_profile_id": "cp_down",
         "members": [{"feature_idx": i} for i in down_support]},
                # 50, not 3: an empirical p floors at 1/(K+1), so with K=3 the floor
        # is 0.25 and BH at q=0.05 can never keep anything — every pair would
        # be filtered for a reason that has nothing to do with the code.
        s_min=20, null_shuffles=50, max_null_tested=max_null_tested,
    )


class TestTheRouteIsReachable:
    """A capability is not shipped until a test fails when its wiring is removed.

    Asserted against the SERVED schema, not `router.routes`: this build wraps
    included routers in `_IncludedRouter` objects that carry no `.path`, so
    reading `.routes` reports zero routes for the whole API and would make this
    guard pass by finding nothing to disagree with.
    """

    def test_the_refinement_route_is_in_the_live_schema(self):
        from src.main import app

        paths = app.openapi()["paths"]
        assert [p for p in paths if "circuit-discovery" in p], (
            "probe is blind — no circuit-discovery route found at all, so a "
            "missing refinement route would be indistinguishable from a broken "
            "probe")
        assert "/api/v1/circuit-discovery/refine-cluster-edge" in paths
        assert "post" in paths["/api/v1/circuit-discovery/refine-cluster-edge"]
