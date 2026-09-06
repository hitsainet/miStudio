#!/usr/bin/env python3
"""
Manual integration test for vectorized extraction.

This script tests the vectorized extraction by:
1. Finding a completed training
2. Running extraction on 100 samples
3. Measuring performance
4. Verifying results

Usage:
    python test_vectorization_manual.py
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.core.config import settings
from src.models.training import Training
from src.models.feature import Feature
from src.services.extraction_service import ExtractionService
from src.core.database import get_sync_db


def _check_completed_training_exists():
    """Check if a completed training exists in the database."""
    try:
        db = next(get_sync_db())
        training = db.query(Training).filter(
            Training.status == "completed"
        ).first()
        db.close()
        return training is not None
    except Exception:
        return False


# Skip test if no completed training exists
pytestmark = pytest.mark.skipif(
    not _check_completed_training_exists(),
    reason="No completed training available for vectorization test"
)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_vectorized_extraction():
    """Run vectorized extraction integration test."""

    print("\n" + "="*70)
    print("VECTORIZED EXTRACTION INTEGRATION TEST")
    print("="*70 + "\n")

    # Get database session
    db = next(get_sync_db())

    try:
        # Find a completed training
        print("Finding completed training...")
        training = db.query(Training).filter(
            Training.status == "completed"
        ).first()

        if not training:
            print("❌ No completed training found.")
            print("   Please run a training first, then retry this test.")
            return False

        print(f"✅ Found training: {training.id}")
        print(f"   Status: {training.status}")
        print()

        # Create extraction service
        extraction_service = ExtractionService(db)

        # Configure extraction for small test
        config = {
            "evaluation_samples": 100,  # Small for testing
            "top_k_examples": 10,
            "batch_size": 8,
            "num_workers": 2,
            "max_length": 512,
            "vectorization_batch_size": "auto",  # Use vectorization
            "soft_time_limit": 3600,  # 1 hour
            "time_limit": 7200,  # 2 hours
        }

        print("Extraction Configuration:")
        print("-" * 70)
        for key, value in config.items():
            print(f"  {key:30s}: {value}")
        print("-" * 70 + "\n")

        # Start extraction
        print("Starting extraction job...")
        start_time = time.time()

        extraction_job = await extraction_service.create_extraction_job(
            training_id=training.id,
            config=config
        )

        print(f"✅ Extraction job created: {extraction_job.id}")
        print(f"   Celery task ID: {extraction_job.celery_task_id}")
        print(f"   Initial status: {extraction_job.status}")
        print()

        # Wait for extraction to complete
        print("Monitoring extraction progress...")
        print("-" * 70)

        max_wait_time = 600  # 10 minutes
        poll_interval = 5  # 5 seconds
        elapsed = 0
        last_progress = 0

        while elapsed < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            # Refresh from database
            db.expire(extraction_job)
            db.refresh(extraction_job)

            progress_pct = extraction_job.progress * 100 if extraction_job.progress else 0

            # Only print if progress changed significantly
            if abs(progress_pct - last_progress) >= 5 or extraction_job.status != "extracting":
                timestamp = time.strftime("%H:%M:%S")
                print(f"  [{timestamp}] Elapsed: {elapsed:3d}s | "
                      f"Status: {extraction_job.status:12s} | "
                      f"Progress: {progress_pct:5.1f}%")
                last_progress = progress_pct

            if extraction_job.status == "completed":
                break
            elif extraction_job.status == "failed":
                print(f"\n❌ Extraction failed!")
                print(f"   Error: {extraction_job.error_message}")
                return False

        print("-" * 70 + "\n")

        if extraction_job.status != "completed":
            print(f"❌ Extraction did not complete within {max_wait_time}s")
            print(f"   Final status: {extraction_job.status}")
            return False

        end_time = time.time()
        total_time = end_time - start_time

        # Verify features were created
        feature_count = db.query(Feature).filter(
            Feature.extraction_job_id == extraction_job.id
        ).count()

        # Print results
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"✅ Extraction completed successfully!")
        print()
        print(f"Performance:")
        print(f"  Total time:        {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"  Samples processed: {config['evaluation_samples']}")
        print(f"  Time per sample:   {total_time/config['evaluation_samples']:.3f} seconds")
        print()
        print(f"Features:")
        print(f"  Features extracted: {feature_count}")
        print(f"  Total features:     {extraction_job.total_features}")
        print()

        if extraction_job.statistics:
            print(f"Statistics:")
            for key, value in extraction_job.statistics.items():
                if isinstance(value, float):
                    print(f"  {key:30s}: {value:.4f}")
                else:
                    print(f"  {key:30s}: {value}")
            print()

        # Performance analysis
        print("Performance Analysis:")
        print(f"  Expected time (sequential):    ~{config['evaluation_samples'] * 1.0:.1f}s (at 1s/sample)")
        print(f"  Actual time (vectorized):      {total_time:.1f}s")
        print(f"  Speedup factor:                {(config['evaluation_samples'] * 1.0) / total_time:.1f}x")
        print()

        # Validation
        print("Validation:")
        if feature_count > 0:
            print(f"  ✅ Features extracted successfully")
        else:
            print(f"  ⚠️  No features extracted (might be expected for sparse activations)")

        if feature_count <= extraction_job.total_features:
            print(f"  ✅ Feature count within expected range")
        else:
            print(f"  ❌ More features than expected")

        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


def main():
    """Main entry point."""
    print("\nStarting vectorized extraction integration test...")
    print("Make sure:")
    print("  1. PostgreSQL is running")
    print("  2. Celery worker is running")
    print("  3. At least one training is completed")
    print()

    # Run async test
    success = asyncio.run(test_vectorized_extraction())

    if success:
        print("✅ Integration test PASSED")
        sys.exit(0)
    else:
        print("❌ Integration test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
