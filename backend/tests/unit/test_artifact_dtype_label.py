"""An artifact's precision is described as recorded, inferred or unknown — and never written."""

from datetime import datetime, timedelta, timezone

from src.services.artifact_dtype import INFERABLE_KINDS, PRECISION_RECORDED_SINCE, describe_dtype

BEFORE = PRECISION_RECORDED_SINCE - timedelta(days=30)
AFTER = PRECISION_RECORDED_SINCE + timedelta(hours=1)


def test_a_recorded_value_is_reported_as_recorded():
    assert describe_dtype("bfloat16", "extraction", AFTER) == {"value": "bfloat16", "source": "recorded", "note": None}


def test_null_before_the_cutover_on_a_float16_kind_is_inferred_and_says_so():
    for kind in ("extraction", "training", "probe_run", "sae"):
        label = describe_dtype(None, kind, BEFORE)
        assert label["value"] == "float16" and label["source"] == "inferred"
        assert "inferred" in label["note"]


def test_null_after_the_cutover_is_unknown_not_inferred():
    """⚠ Review round 1, M4: a queued training, a failed extraction, a probe run that died before
    recording — all NULL after the change, and all loaded bfloat16. Inferring float16 would be false."""
    assert describe_dtype(None, "extraction", AFTER)["source"] == "unknown"
    assert describe_dtype(None, "training", AFTER)["source"] == "unknown"


def test_an_on_the_fly_training_is_never_inferred():
    """Those loaded through the shared loader, which mapped an FP32 row to float32."""
    assert describe_dtype(None, "training", BEFORE, on_the_fly=True)["source"] == "unknown"


def test_no_timestamp_is_unknown():
    assert describe_dtype(None, "extraction", None)["source"] == "unknown"


def test_iso_strings_and_naive_datetimes_are_read():
    assert describe_dtype(None, "extraction", BEFORE.isoformat())["source"] == "inferred"
    assert describe_dtype(None, "extraction", BEFORE.replace(tzinfo=None))["source"] == "inferred"


def test_jlens_is_never_inferred():
    """⚠ The J-lens readout loaded with dtype="auto" — the checkpoint's own precision — so a NULL
    there is no evidence of float16."""
    assert "jlens" not in INFERABLE_KINDS
    assert describe_dtype(None, "jlens", BEFORE) == {"value": None, "source": "unknown", "note": None}


def test_the_cutover_precedes_the_first_commit_of_the_change():
    """⚠ It must not postdate anything the new code made. c0349451 was committed 12:32 UTC on
    2026-10-03; a cutover after that would label a new-code NULL as inferred float16."""
    assert PRECISION_RECORDED_SINCE <= datetime(2026, 10, 3, 12, 32, tzinfo=timezone.utc)


def test_the_module_never_writes():
    import ast
    import inspect

    from src.services import artifact_dtype

    tree = ast.parse(inspect.getsource(artifact_dtype))
    imported = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any(m.endswith(("database", "models", "db")) or ".models" in m for m in imported)
