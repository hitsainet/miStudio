"""Every calibration states its size; a small one is flagged (Feature 040 S2 FR-19; 040-P4; FTASKS 9.6).

The floor is the behaviour's own `calibration_floor` when its definition sets one, else the runtime
setting `calibration_floor_negatives` (2,000); its source is recorded. A probe calibrated before S2 has no
recorded count: it is read from the stored array's length and labelled so; no array → null, never guessed.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.services.calibration_size import calibration_size


def probe(n=None, path=None, target=0.01):
    return SimpleNamespace(calibration_n_negatives=n, calibration_scores_path=path, target_fpr=target,
                           calibration_dataset_id="pmd_c", threshold_source="calibration_set")


def test_540_negatives_are_few_2000_are_not():
    few = calibration_size(probe(540), None, 2000)
    assert (few["few_negatives"], few["floor"], few["floor_source"]) == (True, 2000, "setting")
    assert few["sentence"] == "1% of 540 = 5 rows" and few["allowed_false_positives"] == 5
    enough = calibration_size(probe(2000), None, 2000)
    assert enough["few_negatives"] is False                       # exactly the floor is enough (>=)
    assert enough["sentence"] == "1% of 2,000 = 20 rows"


def test_a_behaviour_floor_overrides_the_setting():
    size = calibration_size(probe(2000), 5000, 2000)
    assert (size["few_negatives"], size["floor"], size["floor_source"]) == (True, 5000, "behaviour")


def test_a_pre_s2_probe_reads_its_count_from_the_array(tmp_path):
    path = tmp_path / "neg.npy"
    np.save(path, np.zeros(1234, dtype=np.float32))
    size = calibration_size(probe(None, str(path)), None, 2000)
    assert (size["n_negatives"], size["n_negatives_source"], size["few_negatives"]) == (1234, "array_length", True)
    nothing = calibration_size(probe(None, None), None, 2000)
    assert (nothing["n_negatives"], nothing["n_negatives_source"], nothing["few_negatives"], nothing["sentence"]) == (
        None, None, None, None)


def test_one_row_reads_singular():
    assert calibration_size(probe(100), None, 50)["sentence"] == "1% of 100 = 1 row"
