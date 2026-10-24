"""The comparator behind `scripts/verify_capture_store.py` (016 FTASKS 6.1).

The live check re-derives a capture store's events and error norms from an
independent forward pass. Its verdict is only as good as `compare_doc`, so the
comparator is pinned here on synthetic stores built with the store's own dtypes:
an exact store passes, and each way a store can be wrong is reported under its
own count. A comparator that passed everything would make the live PASS
meaningless.
"""

import numpy as np

from scripts.verify_capture_store import compare_doc, doc_passes
from src.services.circuit_capture_store import ERRNORM_DTYPE, EVENT_DTYPE


def _case():
    rng = np.random.default_rng(0)
    z = rng.uniform(0.0, 1.0, size=(6, 5)).astype(np.float32)
    z[z < 0.6] = 0.0                       # sparse codes
    thresh = np.full(5, 0.3, dtype=np.float32)
    err = rng.uniform(0.5, 2.0, size=6).astype(np.float32)
    pos, feat = np.nonzero(z > thresh[None, :])
    events = np.empty(len(pos), dtype=EVENT_DTYPE)
    events["doc_id"], events["token_pos"], events["feature_idx"] = 7, pos, feat
    events["act"] = z[pos, feat]
    norms = np.empty(6, dtype=ERRNORM_DTYPE)
    norms["doc_id"], norms["token_pos"], norms["norm"] = 7, np.arange(6), err
    return events, norms, z, err, thresh


def test_an_exact_store_passes():
    row = compare_doc(*_case())
    assert row["events"] > 0 and row["norm_tokens"] == 6
    assert doc_passes(row), row


def test_a_wrong_activation_is_an_act_mismatch():
    events, norms, z, err, thresh = _case()
    events["act"][0] = events["act"][0] * 1.5
    row = compare_doc(events, norms, z, err, thresh)
    assert row["act_mismatch"] == 1 and not doc_passes(row)


def test_a_shifted_position_is_caught():
    """The positions are the Tier-2.5 join key; an off-by-one must not pass."""
    events, norms, z, err, thresh = _case()
    events["token_pos"] = (events["token_pos"] + 1) % 6
    row = compare_doc(events, norms, z, err, thresh)
    assert row["missing"] > 0 and not doc_passes(row)


def test_a_dropped_event_is_missing_and_an_invented_one_is_extra():
    events, norms, z, err, thresh = _case()
    row = compare_doc(events[1:], norms, z, err, thresh)
    assert row["missing"] == 1
    zero = np.argwhere(z == 0.0)[0]
    invented = np.empty(1, dtype=EVENT_DTYPE)
    invented["doc_id"], invented["token_pos"], invented["feature_idx"] = 7, zero[0], zero[1]
    invented["act"] = 0.0
    row = compare_doc(np.concatenate([events, invented]), norms, z, err, thresh)
    assert row["extra"] == 1 and not doc_passes(row)


def test_error_norms_missing_or_wrong_are_reported():
    events, norms, z, err, thresh = _case()
    row = compare_doc(events, norms[:-1], z, err, thresh)
    assert row["norm_missing"] == 1
    norms["norm"][2] = norms["norm"][2] + 1.0
    row = compare_doc(events, norms, z, err, thresh)
    assert row["norm_mismatch"] == 1 and not doc_passes(row)


def test_fp16_storage_is_within_tolerance_and_borderline_never_fails():
    events, norms, z, err, thresh = _case()
    events["act"] = events["act"].astype(np.float16)   # what the store really holds
    norms["norm"] = norms["norm"].astype(np.float16)
    # An unstored pair the recomputation puts a hair above threshold: borderline,
    # decided by fp16 and batch numerics, so reported and not failed.
    z = z.copy()
    t, f = np.argwhere(z == 0.0)[0]
    z[t, f] = thresh[f] * 1.001
    row = compare_doc(events, norms, z, err, thresh)
    assert row["borderline"] >= 1
    assert doc_passes(row), row
