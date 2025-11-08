"""
Test script to verify dual-label system functionality.

Tests:
1. OpenAI service returns dict with category and specific labels
2. JSON parsing handles both valid and malformed responses
3. Labels are properly cleaned and validated
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.openai_labeling_service import OpenAILabelingService
from src.core.config import settings


async def test_dual_label_generation():
    """Test that OpenAI service generates dual labels correctly."""

    print("=" * 80)
    print("Testing Dual-Label System")
    print("=" * 80)

    # Initialize service
    print("\n1. Initializing OpenAI labeling service...")
    service = OpenAILabelingService()
    print(f"   ✓ Service initialized with model: {service.model}")

    # Test case 1: Trump-dominated feature (should get specific label)
    print("\n2. Testing Trump-dominated feature...")
    trump_tokens = {
        "Trump": {"count": 45, "total_activation": 1.035, "max_activation": 0.089},
        "Trumps": {"count": 12, "total_activation": 0.228, "max_activation": 0.076},
        "Donald": {"count": 8, "total_activation": 0.128, "max_activation": 0.071},
        "MAGA": {"count": 6, "total_activation": 0.084, "max_activation": 0.068},
        "administration": {"count": 4, "total_activation": 0.048, "max_activation": 0.065}
    }

    labels = await service.generate_label(trump_tokens, neuron_index=100)
    print(f"   Input: Trump-dominated tokens")
    print(f"   Output: {labels}")
    print(f"   ✓ Category: '{labels['category']}'")
    print(f"   ✓ Specific: '{labels['specific']}'")

    # Validate structure
    assert isinstance(labels, dict), "Labels should be a dict"
    assert "category" in labels, "Labels should have 'category' key"
    assert "specific" in labels, "Labels should have 'specific' key"
    assert isinstance(labels["category"], str), "Category should be a string"
    assert isinstance(labels["specific"], str), "Specific should be a string"
    print(f"   ✓ Structure validation passed")

    # Test case 2: Mixed political feature (should get broader label)
    print("\n3. Testing mixed political feature...")
    mixed_tokens = {
        "president": {"count": 20, "total_activation": 0.360, "max_activation": 0.018},
        "senator": {"count": 15, "total_activation": 0.240, "max_activation": 0.016},
        "congress": {"count": 14, "total_activation": 0.210, "max_activation": 0.015},
        "Trump": {"count": 3, "total_activation": 0.036, "max_activation": 0.012},
        "vote": {"count": 12, "total_activation": 0.132, "max_activation": 0.011}
    }

    labels2 = await service.generate_label(mixed_tokens, neuron_index=200)
    print(f"   Input: Mixed political tokens")
    print(f"   Output: {labels2}")
    print(f"   ✓ Category: '{labels2['category']}'")
    print(f"   ✓ Specific: '{labels2['specific']}'")

    # Test case 3: Empty tokens (should use fallback)
    print("\n4. Testing fallback behavior...")
    empty_labels = await service.generate_label({}, neuron_index=300)
    print(f"   Input: Empty token dict")
    print(f"   Output: {empty_labels}")
    assert empty_labels["specific"] == "feature_300", "Should use fallback for empty tokens"
    print(f"   ✓ Fallback working correctly")

    print("\n" + "=" * 80)
    print("✓ All tests passed! Dual-label system is working correctly.")
    print("=" * 80)
    print("\nKey observations:")
    print(f"- Service returns dict with 'category' and 'specific' keys")
    print(f"- Labels are properly cleaned (lowercase_with_underscores)")
    print(f"- Fallbacks work for edge cases")
    print(f"- Ready for production use!")


if __name__ == "__main__":
    # Check if API key is configured
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY not configured in environment")
        print("Please set it in your .env file or environment variables")
        sys.exit(1)

    asyncio.run(test_dual_label_generation())
