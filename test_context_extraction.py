#!/usr/bin/env python3
"""
Quick test script to validate context window extraction functionality.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.services.extraction_service import ExtractionService

def test_context_extraction():
    """Test extraction with custom context window settings."""
    # Database connection
    DATABASE_URL = "postgresql://postgres:devpassword@localhost:5432/mistudio"
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        extraction_service = ExtractionService(session)

        # Test configuration with custom context windows
        test_config = {
            "evaluation_samples": 100,  # Small sample for quick test
            "top_k_examples": 10,
            "context_prefix_tokens": 5,  # 5 tokens before
            "context_suffix_tokens": 3   # 3 tokens after
        }

        training_id = "train_79086bb3"

        print(f"Starting test extraction for training {training_id}")
        print(f"Config: {test_config}")
        print("=" * 60)

        try:
            # Create extraction job first (this is what the API does)
            from src.models.extraction_job import ExtractionJob, ExtractionStatus
            from datetime import datetime

            extraction_id = f"extr_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{training_id.split('_')[-1]}"

            extraction_job = ExtractionJob(
                id=extraction_id,
                training_id=training_id,
                config=test_config,
                status=ExtractionStatus.QUEUED,
                filter_special=True,
                filter_single_char=True,
                filter_punctuation=True,
                filter_numbers=True,
                filter_fragments=True,
                filter_stop_words=False
            )
            session.add(extraction_job)
            session.commit()

            print(f"Created extraction job: {extraction_id}")

            # Now run extraction
            statistics = extraction_service.extract_features_for_training(
                training_id=training_id,
                config=test_config
            )

            print("\n" + "=" * 60)
            print("Extraction completed successfully!")
            print(f"Statistics: {statistics}")

            # Query a sample feature_activation to verify structure
            from src.models.extraction_job import ExtractionJob
            from src.models.feature_activation import FeatureActivation
            from sqlalchemy import desc

            # Get the latest extraction job
            extraction_job = session.query(ExtractionJob).filter(
                ExtractionJob.training_id == training_id
            ).order_by(desc(ExtractionJob.created_at)).first()

            if extraction_job:
                print(f"\nExtraction Job ID: {extraction_job.id}")

                # Get a sample activation
                sample_activation = session.query(FeatureActivation).filter(
                    FeatureActivation.feature_id.like(f"feat_{extraction_job.id}_%")
                ).first()

                if sample_activation:
                    print(f"\nSample activation structure:")
                    print(f"  Tokens keys: {list(sample_activation.tokens.keys()) if isinstance(sample_activation.tokens, dict) else 'array'}")
                    print(f"  Sample tokens: {sample_activation.tokens if isinstance(sample_activation.tokens, list) else 'dict'}")
                    print(f"  Activations keys: {list(sample_activation.activations.keys()) if isinstance(sample_activation.activations, dict) else 'array'}")
                    print(f"\n  Has context window fields: {isinstance(sample_activation.tokens, list)}")

        except Exception as e:
            print(f"\nExtraction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True

if __name__ == "__main__":
    success = test_context_extraction()
    sys.exit(0 if success else 1)
