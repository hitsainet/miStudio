"""The behaviour checklist's step labels and links name the product as it is called now (042 FR-042.31;
FTASKS 15.4).

Each step carries a `link` the frontend turns into a tab jump, or, for the one step that happens in the
other app, into the hint "in miServer" (`BehaviourChecklist.tsx` compares `link.app` with `'miserver'`;
`behaviours.test.tsx` renders that hint from the same value). The value is this app's own answer, not a
frozen identifier: it was `millm` until P4, and a backend still sending the old value would silently
drop the hint, so the pair is pinned on both sides.
"""
from __future__ import annotations

from src.services import behaviour_checklist as bc
from tests.support import brand_surfaces as bs


def test_one_step_happens_in_miserver_and_says_so():
    in_other_app = [(sid, label, link) for sid, label, link in bc.STEPS if "app" in link]
    assert in_other_app == [("armed", "Armed in miServer", {"app": "miserver"})]


def test_every_step_links_to_exactly_one_place_and_no_label_names_an_old_brand():
    assert len(bc.STEPS) >= 8
    for sid, label, link in bc.STEPS:
        assert sorted(link) in (["tab"], ["app"]), (sid, link)
        assert not bs.STEM.search(label), (sid, label)
        assert not bs.STEM.search(str(link)), (sid, link)
