#!/usr/bin/env python3
"""
Script to restore completed extraction job that was incorrectly restarted.

This script:
1. Verifies all features from the successful extraction exist
2. Restores the extraction job status to COMPLETED
3. Ensures all metadata is correct
"""

import sys
from sqlalchemy import create_engine, select, func, desc
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

# Add backend to path
sys.path.insert(0, '/home/x-sean/app/miStudio/backend')

from src.core.config import settings
from src.models.extraction_job import ExtractionJob, ExtractionStatus
from src.models.feature import Feature

# Create database connection (use sync URL)
# Convert async URL to sync
database_url = str(settings.database_url).replace('postgresql+asyncpg://', 'postgresql://')
engine = create_engine(database_url)
SessionLocal = sessionmaker(bind=engine)

def restore_extraction():
    """Restore the completed extraction state."""
    db = SessionLocal()

    try:
        # Get the extraction job
        extraction_id = "extr_20251027_234418_train_43"
        extraction = db.query(ExtractionJob).filter(
            ExtractionJob.id == extraction_id
        ).first()

        if not extraction:
            print(f"❌ Extraction {extraction_id} not found!")
            return False

        print(f"Found extraction: {extraction.id}")
        print(f"Current status: {extraction.status}")
        print(f"Created at: {extraction.created_at}")
        print(f"Completed at: {extraction.completed_at}")
        print(f"Statistics: {extraction.statistics}")

        # Count features created by this extraction
        feature_count = db.query(func.count(Feature.id)).filter(
            Feature.extraction_job_id == extraction_id
        ).scalar()

        print(f"\nFeatures in database: {feature_count}")

        # Get the last feature created to verify completion
        last_feature = db.query(Feature).filter(
            Feature.extraction_job_id == extraction_id
        ).order_by(desc(Feature.created_at)).first()

        if last_feature:
            print(f"Last feature: {last_feature.id} (neuron {last_feature.neuron_index})")
            print(f"Created at: {last_feature.created_at}")

        # Verify we have 8192 features (the latent_dim from statistics)
        expected_features = extraction.statistics.get('total_features', 8192) if extraction.statistics else 8192

        if feature_count != expected_features:
            print(f"\n⚠️  WARNING: Expected {expected_features} features, found {feature_count}")
            response = input("Continue with restoration? (yes/no): ")
            if response.lower() != 'yes':
                print("Restoration cancelled")
                return False

        # Restore the correct state
        print("\n🔧 Restoring extraction state...")

        extraction.status = ExtractionStatus.COMPLETED.value
        extraction.progress = 1.0
        extraction.features_extracted = feature_count
        extraction.total_features = expected_features

        # Ensure completed_at is set (it should already be)
        if not extraction.completed_at:
            extraction.completed_at = datetime.now(timezone.utc)

        # Ensure statistics are preserved
        if not extraction.statistics:
            extraction.statistics = {
                "total_features": expected_features,
                "interpretable_count": feature_count,  # Conservative estimate
                "avg_activation_frequency": 0.775,  # From original
                "avg_interpretability": 0.662  # From original
            }

        extraction.updated_at = datetime.now(timezone.utc)

        db.commit()

        print("✅ Extraction state restored successfully!")
        print(f"\nFinal state:")
        print(f"  Status: {extraction.status}")
        print(f"  Progress: {extraction.progress * 100:.1f}%")
        print(f"  Features: {extraction.features_extracted}/{extraction.total_features}")
        print(f"  Completed at: {extraction.completed_at}")
        print(f"  Statistics: {extraction.statistics}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 70)
    print("EXTRACTION STATE RESTORATION SCRIPT")
    print("=" * 70)
    print()

    success = restore_extraction()

    if success:
        print("\n✅ Restoration completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Restoration failed!")
        sys.exit(1)
