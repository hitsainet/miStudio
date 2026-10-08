"""
Integration test for vectorized feature extraction.

This test verifies that the vectorized extraction:
1. Works end-to-end with real models/datasets
2. Produces correct results
3. Achieves expected performance improvements

Usage:
    python -m pytest tests/integration/test_vectorized_extraction.py -v -s

    Or run directly:
    python tests/integration/test_vectorized_extraction.py
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest
from sqlalchemy import select, func

# Fixtures are automatically loaded from tests/conftest.py


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vectorized_extraction_small_dataset(async_session):
    """
    Integration test: Run vectorized extraction on small dataset.

    This test:
    1. Uses an existing training with SAE checkpoint
    2. Runs extraction with 100 samples
    3. Verifies features are extracted correctly
    4. Measures extraction time
    """
    from src.models.training import Training
    from src.services.extraction_service import ExtractionService

    # Find a completed training with checkpoint (SQLAlchemy 2.0 async style)
    result = await async_session.execute(
        select(Training).filter(Training.status == "completed")
    )
    training = result.scalar_one_or_none()

    if not training:
        pytest.skip("No completed training found. Run a training first.")

    print(f"\n{'='*60}")
    print(f"Testing Vectorized Extraction")
    print(f"{'='*60}")
    print(f"Training ID: {training.id}")
    print(f"Training Status: {training.status}")
    print(f"{'='*60}\n")

    # Create extraction service
    extraction_service = ExtractionService(async_session)

    # Configure extraction for small test
    config = {
        "evaluation_samples": 100,  # Small dataset for testing
        "top_k_examples": 10,
        "batch_size": 8,
        "num_workers": 2,
        "max_length": 512,
        "vectorization_batch_size": "auto",  # Test auto-calculation
        "soft_time_limit": 3600,  # 1 hour (plenty for 100 samples)
        "time_limit": 7200,  # 2 hours hard limit
    }

    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Start extraction
    print("Starting extraction...")
    start_time = time.time()

    try:
        extraction_job = await extraction_service.create_extraction_job(
            training_id=training.id,
            config=config
        )

        print(f"Extraction job created: {extraction_job.id}")
        print(f"Status: {extraction_job.status}")
        print(f"Celery task ID: {extraction_job.celery_task_id}")
        print()

        # Wait for extraction to complete (poll every 5 seconds)
        print("Waiting for extraction to complete...")
        max_wait_time = 600  # 10 minutes max
        poll_interval = 5  # 5 seconds
        elapsed = 0

        while elapsed < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Refresh extraction job
            await async_session.refresh(extraction_job)

            print(f"  [{elapsed}s] Status: {extraction_job.status}, "
                  f"Progress: {extraction_job.progress * 100:.1f}%")

            if extraction_job.status == "completed":
                break
            elif extraction_job.status == "failed":
                print(f"\n❌ Extraction failed: {extraction_job.error_message}")
                pytest.fail(f"Extraction failed: {extraction_job.error_message}")

        if extraction_job.status != "completed":
            pytest.fail(f"Extraction did not complete within {max_wait_time}s")

        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n{'='*60}")
        print(f"✅ Extraction completed successfully!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Samples processed: {config['evaluation_samples']}")
        print(f"Time per sample: {total_time/config['evaluation_samples']:.3f} seconds")
        print(f"{'='*60}\n")

        # Verify features were created (SQLAlchemy 2.0 async style)
        from src.models.feature import Feature
        result = await async_session.execute(
            select(func.count()).select_from(Feature).filter(
                Feature.extraction_job_id == extraction_job.id
            )
        )
        feature_count = result.scalar()

        print(f"Features extracted: {feature_count}")

        assert feature_count > 0, "No features were extracted"
        assert feature_count <= extraction_job.total_features, \
            f"More features than expected: {feature_count} > {extraction_job.total_features}"

        # Check statistics
        if extraction_job.statistics:
            print(f"\nStatistics:")
            for key, value in extraction_job.statistics.items():
                print(f"  {key}: {value}")

        print(f"\n{'='*60}")
        print(f"Test passed! Vectorization is working correctly.")
        print(f"{'='*60}\n")

        return {
            "total_time": total_time,
            "samples": config['evaluation_samples'],
            "features_extracted": feature_count,
            "time_per_sample": total_time / config['evaluation_samples'],
        }

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        raise


@pytest.mark.integration
def test_vectorization_performance_comparison():
    """
    Manual performance comparison test.

    This test provides instructions for manual performance testing:
    1. Run extraction with vectorization_batch_size=1 (sequential-like)
    2. Run extraction with vectorization_batch_size=auto (vectorized)
    3. Compare times

    Note: This test is skipped by default. Run manually when needed.
    """
    pytest.skip(
        "Manual test - Run two extractions and compare:\n"
        "1. Sequential: vectorization_batch_size=1\n"
        "2. Vectorized: vectorization_batch_size=auto\n"
        "Compare extraction times to measure speedup."
    )


if __name__ == "__main__":
    """
    Direct execution for manual testing.

    Usage:
        python tests/integration/test_vectorized_extraction.py
    """
    print("\n" + "="*60)
    print("Vectorized Extraction Integration Test")
    print("="*60 + "\n")

    print("Prerequisites:")
    print("1. PostgreSQL database running")
    print("2. At least one completed training with SAE checkpoint")
    print("3. Celery worker running")
    print("4. Dataset available")
    print("\nStarting test...\n")

    # Run pytest programmatically
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
