"""
Unit test for dual-label parsing logic (no API calls required).

Tests the _parse_dual_label method with various response formats.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.openai_labeling_service import OpenAILabelingService


def test_parse_dual_label():
    """Test parsing of dual-label JSON responses."""

    print("=" * 80)
    print("Testing Dual-Label Parsing Logic")
    print("=" * 80)

    # Initialize service (we won't make API calls, just test parsing)
    service = OpenAILabelingService(api_key="dummy_key_for_testing")

    # Test case 1: Valid JSON response
    print("\n1. Testing valid JSON response...")
    valid_json = '{"category": "political_terms", "specific": "trump_mentions"}'
    result = service._parse_dual_label(valid_json, "fallback_123")
    print(f"   Input: {valid_json}")
    print(f"   Output: {result}")
    assert result["category"] == "political_terms", f"Expected 'political_terms', got '{result['category']}'"
    assert result["specific"] == "trump_mentions", f"Expected 'trump_mentions', got '{result['specific']}'"
    print(f"   ✓ Valid JSON parsing works")

    # Test case 2: JSON with extra whitespace
    print("\n2. Testing JSON with whitespace...")
    whitespace_json = '  {"category": "function_words", "specific": "negative_contractions"}  '
    result = service._parse_dual_label(whitespace_json, "fallback_123")
    print(f"   Input: {whitespace_json}")
    print(f"   Output: {result}")
    assert result["category"] == "function_words"
    assert result["specific"] == "negative_contractions"
    print(f"   ✓ Whitespace handling works")

    # Test case 3: Malformed JSON (should use regex fallback)
    print("\n3. Testing malformed JSON with regex fallback...")
    malformed = 'The labels are: category: "code_keywords", specific: "python_syntax"'
    result = service._parse_dual_label(malformed, "fallback_456")
    print(f"   Input: {malformed}")
    print(f"   Output: {result}")
    assert result["category"] == "code_keywords", f"Expected 'code_keywords', got '{result['category']}'"
    assert result["specific"] == "python_syntax", f"Expected 'python_syntax', got '{result['specific']}'"
    print(f"   ✓ Regex fallback works")

    # Test case 4: Completely unparseable (should use fallbacks)
    print("\n4. Testing completely unparseable response...")
    unparseable = "This is not a valid response at all"
    result = service._parse_dual_label(unparseable, "fallback_789")
    print(f"   Input: {unparseable}")
    print(f"   Output: {result}")
    assert result["category"] == "uncategorized"
    assert result["specific"] == "fallback_789"
    print(f"   ✓ Fallback handling works")

    # Test case 5: Missing specific field (should use fallback)
    print("\n5. Testing JSON with missing 'specific' field...")
    missing_specific = '{"category": "names"}'
    result = service._parse_dual_label(missing_specific, "fallback_999")
    print(f"   Input: {missing_specific}")
    print(f"   Output: {result}")
    assert result["category"] == "names"
    assert result["specific"] == "fallback_999"
    print(f"   ✓ Missing field handling works")

    # Test case 6: Test label cleaning (uppercase, spaces, special chars)
    print("\n6. Testing label cleaning...")
    messy_json = '{"category": "Political Terms!!!", "specific": "Trump-Related Mentions"}'
    result = service._parse_dual_label(messy_json, "fallback_000")
    print(f"   Input: {messy_json}")
    print(f"   Output: {result}")
    assert result["category"] == "political_terms"
    assert result["specific"] == "trump_related_mentions"
    print(f"   ✓ Label cleaning works (lowercase, underscores, no special chars)")

    print("\n" + "=" * 80)
    print("✓ All parsing tests passed!")
    print("=" * 80)
    print("\nKey features verified:")
    print("- ✓ Valid JSON parsing")
    print("- ✓ Whitespace handling")
    print("- ✓ Regex fallback for malformed JSON")
    print("- ✓ Complete fallback for unparseable responses")
    print("- ✓ Missing field handling")
    print("- ✓ Label cleaning (lowercase_with_underscores)")
    print("\nDual-label system is ready for production!")


if __name__ == "__main__":
    test_parse_dual_label()
