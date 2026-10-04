"""
Integration tests for tokenization preview endpoint.

Tests the /api/v1/datasets/tokenize-preview endpoint functionality.
"""

import pytest
from httpx import AsyncClient


class TestTokenizePreview:
    """Test suite for tokenization preview endpoint."""

    @pytest.mark.asyncio
    async def test_tokenize_preview_success(self, client: AsyncClient):
        """Test successful tokenization preview with default settings."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello, world! This is a test.",
                "max_length": 512,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "tokens" in data
        assert "attention_mask" in data
        assert "token_count" in data
        assert "sequence_length" in data
        assert "special_token_count" in data

        # Verify token structure
        assert len(data["tokens"]) > 0
        first_token = data["tokens"][0]
        assert "id" in first_token
        assert "text" in first_token
        assert "type" in first_token
        assert "position" in first_token
        assert first_token["type"] in ["special", "regular"]

        # Verify counts
        assert data["token_count"] > 0
        assert data["sequence_length"] > 0
        assert data["special_token_count"] >= 0

        # Verify attention mask length matches token count
        assert len(data["attention_mask"]) == data["token_count"]

    @pytest.mark.asyncio
    async def test_tokenize_preview_without_attention_mask(self, client: AsyncClient):
        """Test tokenization preview without requesting attention mask."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello, world!",
                "max_length": 512,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify attention mask is None when not requested
        assert data["attention_mask"] is None

    @pytest.mark.asyncio
    async def test_tokenize_preview_without_special_tokens(self, client: AsyncClient):
        """Test tokenization preview without adding special tokens."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello, world!",
                "max_length": 512,
                "padding": "do_not_pad",
                "truncation": "longest_first",
                "add_special_tokens": False,
                "return_attention_mask": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify special token count is 0 when not adding special tokens
        assert data["special_token_count"] == 0

    @pytest.mark.asyncio
    async def test_tokenize_preview_padding_strategies(self, client: AsyncClient):
        """Test different padding strategies."""
        text = "Short text"

        # Test max_length padding (should pad to max_length)
        response_max = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": text,
                "max_length": 20,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response_max.status_code == 200
        data_max = response_max.json()
        assert data_max["token_count"] == 20  # Padded to max_length

        # Test do_not_pad (should not pad)
        response_no_pad = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": text,
                "max_length": 20,
                "padding": "do_not_pad",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response_no_pad.status_code == 200
        data_no_pad = response_no_pad.json()
        assert data_no_pad["token_count"] < 20  # Not padded

    @pytest.mark.asyncio
    async def test_tokenize_preview_truncation_strategies(self, client: AsyncClient):
        """Test different truncation strategies with long text."""
        # Create a long text that exceeds max_length (but stays under 1000 char limit)
        # Using 180 words (~900 chars) to test truncation to 50 tokens
        long_text = " ".join(["word"] * 180)

        # Test longest_first truncation (default)
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": long_text,
                "max_length": 50,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["token_count"] == 50  # Truncated to max_length

    @pytest.mark.asyncio
    async def test_tokenize_preview_invalid_tokenizer(self, client: AsyncClient):
        """Test tokenization preview with invalid tokenizer name."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "invalid-tokenizer-name-that-does-not-exist",
                "text": "Hello, world!",
                "max_length": 512,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )

        # Should return 400 for invalid tokenizer
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "tokenizer" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_tokenize_preview_empty_text(self, client: AsyncClient):
        """Test tokenization preview with empty text."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "",
                "max_length": 512,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )

        # Should fail validation (min_length=1)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_tokenize_preview_text_too_long(self, client: AsyncClient):
        """Test tokenization preview with text exceeding 1000 character limit."""
        long_text = "a" * 1001  # 1001 characters

        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": long_text,
                "max_length": 512,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )

        # Should fail validation (max_length=1000)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_tokenize_preview_different_tokenizers(self, client: AsyncClient):
        """Test tokenization preview with different tokenizer types."""
        text = "Hello, world!"
        tokenizers = ["gpt2", "bert-base-uncased"]

        for tokenizer_name in tokenizers:
            response = await client.post(
                "/api/v1/datasets/tokenize-preview",
                json={
                    "tokenizer_name": tokenizer_name,
                    "text": text,
                    "max_length": 512,
                    "padding": "max_length",
                    "truncation": "longest_first",
                    "add_special_tokens": True,
                    "return_attention_mask": True,
                },
            )

            assert response.status_code == 200, f"Failed for tokenizer: {tokenizer_name}"
            data = response.json()

            # Verify basic structure
            assert data["token_count"] > 0
            assert len(data["tokens"]) > 0

    @pytest.mark.asyncio
    async def test_tokenize_preview_special_token_detection(self, client: AsyncClient):
        """Test that special tokens are correctly identified."""
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello",
                "max_length": 20,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Find special tokens
        special_tokens = [t for t in data["tokens"] if t["type"] == "special"]
        regular_tokens = [t for t in data["tokens"] if t["type"] == "regular"]

        # Should have some special tokens (at least BOS/EOS for GPT-2)
        assert len(special_tokens) > 0
        # Should have some regular tokens
        assert len(regular_tokens) > 0

        # Verify special token count matches
        assert len(special_tokens) == data["special_token_count"]

    @pytest.mark.asyncio
    async def test_tokenize_preview_max_length_validation(self, client: AsyncClient):
        """Test max_length validation (1-8192 range)."""
        # Test valid max_length
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello",
                "max_length": 1024,
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response.status_code == 200

        # Test max_length too high
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello",
                "max_length": 10000,  # Exceeds 8192 limit
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response.status_code == 422

        # Test max_length too low
        response = await client.post(
            "/api/v1/datasets/tokenize-preview",
            json={
                "tokenizer_name": "gpt2",
                "text": "Hello",
                "max_length": 0,  # Below 1 minimum
                "padding": "max_length",
                "truncation": "longest_first",
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
        assert response.status_code == 422
