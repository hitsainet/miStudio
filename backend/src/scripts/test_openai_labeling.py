#!/usr/bin/env python3
"""
Test script for OpenAI API labeling without full extraction.

This script:
1. Queries existing features from a completed extraction
2. Aggregates token statistics for a small subset (default: 10 features)
3. Calls OpenAI API to generate labels
4. Prints results

Usage:
    python test_openai_labeling.py --extraction-id extr_20251107_020805_train_a0 --num-features 10
"""

import asyncio
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.core.config import settings
from src.models.feature import Feature
from src.models.feature_activation import FeatureActivation
from src.services.openai_labeling_service import OpenAILabelingService


def aggregate_token_stats_for_feature(db: Session, feature_id: str) -> dict:
    """Aggregate token statistics for a single feature."""
    activations = db.query(FeatureActivation).filter(
        FeatureActivation.feature_id == feature_id
    ).all()

    token_stats = defaultdict(lambda: {"count": 0, "total_activation": 0.0, "max_activation": 0.0})

    for activation in activations:
        tokens = activation.tokens
        activations_list = activation.activations

        for token, act in zip(tokens, activations_list):
            token_stats[token]["count"] += 1
            token_stats[token]["total_activation"] += act
            token_stats[token]["max_activation"] = max(
                token_stats[token]["max_activation"], act
            )

    return dict(token_stats)


async def test_openai_labeling(extraction_id: str, num_features: int = 10, api_key: str = None):
    """Test OpenAI labeling on a subset of features."""

    print("=" * 80)
    print("OpenAI API Labeling Test")
    print("=" * 80)
    print(f"Extraction ID: {extraction_id}")
    print(f"Features to test: {num_features}")
    print()

    # Create database connection
    engine = create_engine(str(settings.database_url).replace('+asyncpg', ''))
    db = Session(engine)

    try:
        # Query features with label_source='llm' (placeholder labels)
        print("Step 1: Querying features from database...")
        features = db.query(Feature).filter(
            Feature.extraction_job_id == extraction_id,
            Feature.label_source == "llm"
        ).limit(num_features).all()

        if not features:
            print(f"❌ No features found with label_source='llm' for extraction {extraction_id}")
            print("\nTrying to find ANY features for this extraction...")
            features = db.query(Feature).filter(
                Feature.extraction_job_id == extraction_id
            ).limit(num_features).all()

            if not features:
                print(f"❌ No features found at all for extraction {extraction_id}")
                return

        print(f"✓ Found {len(features)} features")
        print()

        # Aggregate token statistics
        print("Step 2: Aggregating token statistics...")
        features_token_stats = []
        neuron_indices = []

        for i, feature in enumerate(features, 1):
            print(f"  - Feature {i}/{len(features)}: {feature.id} (neuron {feature.neuron_index})")
            token_stats = aggregate_token_stats_for_feature(db, feature.id)
            features_token_stats.append(token_stats)
            neuron_indices.append(feature.neuron_index)

            # Show sample token stats for first feature
            if i == 1 and token_stats:
                print(f"    Sample tokens: {list(token_stats.keys())[:5]}")
                print(f"    Total unique tokens: {len(token_stats)}")

        print(f"✓ Aggregated token stats for {len(features)} features")
        print()

        # Initialize OpenAI service
        print("Step 3: Initializing OpenAI service...")
        if not api_key:
            # Try to get from config
            print("  - No API key provided, checking configuration...")
            # For now, we'll require explicit API key
            print("  ⚠️  Please provide OpenAI API key via --api-key argument")
            return

        service = OpenAILabelingService(api_key=api_key, model="gpt-4o-mini")
        print(f"✓ OpenAI service initialized (model: gpt-4o-mini)")
        print()

        # Generate labels
        print("Step 4: Calling OpenAI API to generate labels...")
        print("  (This may take a few seconds per batch)")

        labels = await service.batch_generate_labels(
            features_token_stats=features_token_stats,
            neuron_indices=neuron_indices,
            progress_callback=lambda i, total: print(f"  - Batch {i}/{total} completed"),
            batch_size=5  # Small batches for testing
        )

        print(f"✓ Generated {len(labels)} labels")
        print()

        # Display results
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        for feature, label in zip(features, labels):
            print(f"Neuron {feature.neuron_index:5d}: {label}")

        print()
        print("=" * 80)
        print("✅ OpenAI API integration test SUCCESSFUL!")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OpenAI API labeling")
    parser.add_argument(
        "--extraction-id",
        type=str,
        default="extr_20251107_020805_train_a0",
        help="Extraction job ID to use for testing"
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features to test (default: 10)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (required)"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("❌ Error: --api-key is required")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} --api-key YOUR_API_KEY")
        print(f"  python {sys.argv[0]} --api-key YOUR_API_KEY --num-features 5")
        sys.exit(1)

    asyncio.run(test_openai_labeling(
        extraction_id=args.extraction_id,
        num_features=args.num_features,
        api_key=args.api_key
    ))
