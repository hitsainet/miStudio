#!/usr/bin/env python3
"""
Quick test script to demonstrate text cleaning functionality.

Run with: python test_text_cleaning.py
"""

from src.utils.text_cleaning import get_standard_cleaner, get_aggressive_cleaner, get_minimal_cleaner


def test_cleaning():
    """Test text cleaning with various examples."""

    # Test examples with common junk
    test_texts = [
        "This is normal text.",
        "<html><body>HTML text with tags</body></html>",
        "Text with &nbsp; HTML &lt;entities&gt;",
        "Text with URL https://example.com/path",
        "Email: user@example.com in text",
        "Control\x00chars\x01\x02\x03\x04",
        "Excessive!!!!!!! punctuation??????",
        "    Too    much    whitespace    ",
        "<?xml version='1.0'?><doc>XML content</doc>",
        "Short",
        "",
    ]

    print("=" * 80)
    print("TEXT CLEANING DEMONSTRATION")
    print("=" * 80)

    # Test with different cleaners
    for cleaner_name, cleaner in [
        ("Standard", get_standard_cleaner()),
        ("Aggressive", get_aggressive_cleaner()),
        ("Minimal", get_minimal_cleaner()),
    ]:
        print(f"\n{cleaner_name} Cleaner:")
        print("-" * 80)

        for i, text in enumerate(test_texts, 1):
            cleaned = cleaner.clean(text)
            print(f"{i}. Input:  {repr(text[:60])}")
            print(f"   Output: {repr(cleaned[:60]) if cleaned else 'FILTERED OUT'}")
            print()


if __name__ == "__main__":
    test_cleaning()
