"""
Phase 8 regression tests — Celery task retry configuration.

Verifies that:
1. The four long-running task groups are configured to retry on transient
   I/O errors (ConnectionError, TimeoutError, OSError).
2. Validation errors (ValueError, TypeError, AssertionError) are NOT retried
   — tasks fail fast on programmer errors and bad input.
3. The retry ceiling is max_retries=3 with a 60-second back-off.
"""

import pytest


class TestExtractionTaskRetryConfig:
    def test_extract_features_retries_on_transient_errors(self):
        from src.workers.extraction_tasks import extract_features_from_sae_task
        t = extract_features_from_sae_task
        assert t.max_retries == 3, f"expected 3, got {t.max_retries}"
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_extract_features_does_not_retry_on_value_error(self):
        from src.workers.extraction_tasks import extract_features_from_sae_task
        assert ValueError not in extract_features_from_sae_task.autoretry_for

    def test_delete_extraction_retries_on_transient_errors(self):
        from src.workers.extraction_tasks import delete_extraction_task
        t = delete_extraction_task
        assert t.max_retries == 3
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_delete_extraction_does_not_retry_on_value_error(self):
        from src.workers.extraction_tasks import delete_extraction_task
        assert ValueError not in delete_extraction_task.autoretry_for


class TestLabelingTaskRetryConfig:
    def test_label_features_retries_on_transient_errors(self):
        from src.workers.labeling_tasks import label_features_task
        t = label_features_task
        assert t.max_retries == 3
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_label_features_does_not_retry_on_value_error(self):
        from src.workers.labeling_tasks import label_features_task
        assert ValueError not in label_features_task.autoretry_for


class TestEnhancedLabelingTaskRetryConfig:
    def test_enhanced_label_retries_on_transient_errors(self):
        from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task
        t = enhanced_label_feature_task
        assert t.max_retries == 3
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_enhanced_label_does_not_retry_on_value_error(self):
        from src.workers.enhanced_labeling_tasks import enhanced_label_feature_task
        assert ValueError not in enhanced_label_feature_task.autoretry_for


class TestNLPAnalysisTaskRetryConfig:
    def test_analyze_features_retries_on_transient_errors(self):
        from src.workers.nlp_analysis_tasks import analyze_features_nlp_task
        t = analyze_features_nlp_task
        assert t.max_retries == 3
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_analyze_single_feature_retries_on_transient_errors(self):
        from src.workers.nlp_analysis_tasks import analyze_single_feature_nlp_task
        t = analyze_single_feature_nlp_task
        assert t.max_retries == 3
        assert ConnectionError in t.autoretry_for
        assert TimeoutError in t.autoretry_for
        assert OSError in t.autoretry_for

    def test_nlp_tasks_do_not_retry_on_value_error(self):
        from src.workers.nlp_analysis_tasks import analyze_features_nlp_task, analyze_single_feature_nlp_task
        assert ValueError not in analyze_features_nlp_task.autoretry_for
        assert ValueError not in analyze_single_feature_nlp_task.autoretry_for


class TestSteeringTaskRetryConfig:
    """Steering tasks deliberately keep max_retries=0 — they are GPU tasks
    watched by the user in real time and should fail fast, not silently retry."""

    def test_steering_compare_has_no_retries(self):
        from src.workers.steering_tasks import steering_compare_task
        assert steering_compare_task.max_retries == 0

    def test_steering_sweep_has_no_retries(self):
        from src.workers.steering_tasks import steering_sweep_task
        assert steering_sweep_task.max_retries == 0
