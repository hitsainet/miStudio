"""
Unit tests for hf_utils module.

Tests HuggingFace utility functions including repository validation,
metadata retrieval, and URL generation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.utils.hf_utils import (
    validate_repo_id,
    check_repo_exists,
    get_repo_info,
    get_repo_card,
    parse_repo_id,
    get_repo_url,
)


class TestValidateRepoId:
    """Test validate_repo_id function."""

    def test_validate_repo_id_valid_formats(self):
        """Test valid repository ID formats."""
        valid_ids = [
            "roneneldan/TinyStories",
            "username/dataset-name",
            "user123/data_set-v2",
            "org/dataset.v1",
            "a/b",  # Minimum length
            "user-name/dataset-name_v2.1",  # Hyphens in username, not underscores
        ]

        for repo_id in valid_ids:
            assert validate_repo_id(repo_id) is True, f"Should accept {repo_id}"

    def test_validate_repo_id_invalid_formats(self):
        """Test invalid repository ID formats."""
        invalid_ids = [
            "invalid",  # No slash
            "invalid/",  # No dataset name
            "/dataset",  # No username
            "user name/dataset",  # Space in username
            "user/dataset name",  # Space in dataset name
            "-username/dataset",  # Username starts with hyphen
            "username-/dataset",  # Username ends with hyphen
            "",  # Empty string
            "/",  # Just slash
            "user//dataset",  # Double slash
            "user/dataset/extra",  # Too many parts
        ]

        for repo_id in invalid_ids:
            assert validate_repo_id(repo_id) is False, f"Should reject {repo_id}"

    def test_validate_repo_id_none(self):
        """Test None input."""
        assert validate_repo_id(None) is False

    def test_validate_repo_id_non_string(self):
        """Test non-string input."""
        assert validate_repo_id(123) is False
        assert validate_repo_id([]) is False
        assert validate_repo_id({}) is False

    def test_validate_repo_id_special_characters(self):
        """Test handling of special characters."""
        # Valid special characters
        assert validate_repo_id("user/dataset-name_v1.0") is True

        # Invalid special characters
        assert validate_repo_id("user/dataset@v1") is False
        assert validate_repo_id("user/dataset#v1") is False
        assert validate_repo_id("user/dataset$v1") is False


@pytest.mark.asyncio
class TestCheckRepoExists:
    """Test check_repo_exists function."""

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_valid_repo(self, mock_load_builder):
        """Test checking existence of valid repository."""
        mock_builder = Mock()
        mock_load_builder.return_value = mock_builder

        exists, error = await check_repo_exists("roneneldan/TinyStories")

        assert exists is True
        assert error is None
        mock_load_builder.assert_called_once_with("roneneldan/TinyStories", token=None)

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_with_token(self, mock_load_builder):
        """Test checking repo with authentication token."""
        mock_builder = Mock()
        mock_load_builder.return_value = mock_builder

        exists, error = await check_repo_exists("private/dataset", token="hf_token123")

        assert exists is True
        assert error is None
        mock_load_builder.assert_called_once_with("private/dataset", token="hf_token123")

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_not_found(self, mock_load_builder):
        """Test checking nonexistent repository."""
        mock_load_builder.side_effect = FileNotFoundError("Dataset not found")

        exists, error = await check_repo_exists("user/nonexistent")

        assert exists is False
        assert error == "Repository not found"

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_gated_repo(self, mock_load_builder):
        """Test checking gated repository without token."""
        mock_load_builder.side_effect = PermissionError("Gated dataset")

        exists, error = await check_repo_exists("gated/dataset")

        assert exists is False
        assert error == "Repository is gated and requires authentication"

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_invalid_repo(self, mock_load_builder):
        """Test checking invalid repository."""
        mock_load_builder.side_effect = ValueError("Invalid dataset format")

        exists, error = await check_repo_exists("user/invalid-dataset")

        assert exists is False
        assert "Invalid repository" in error

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_check_repo_exists_network_error(self, mock_load_builder):
        """Test handling network errors."""
        mock_load_builder.side_effect = Exception("Network timeout")

        exists, error = await check_repo_exists("user/dataset")

        assert exists is False
        assert "Error accessing repository" in error

    async def test_check_repo_exists_invalid_format(self):
        """Test checking repo with invalid ID format."""
        exists, error = await check_repo_exists("invalid")

        assert exists is False
        assert error == "Invalid repository ID format"


@pytest.mark.asyncio
class TestGetRepoInfo:
    """Test get_repo_info function."""

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_get_repo_info_complete_metadata(self, mock_load_builder):
        """Test getting complete repository metadata."""
        mock_info = Mock()
        mock_info.description = "Test dataset description"
        mock_info.homepage = "https://example.com"
        mock_info.license = "MIT"
        mock_info.citation = "Citation text"
        mock_info.splits = {
            "train": Mock(num_examples=10000),
            "validation": Mock(num_examples=1000),
            "test": Mock(num_examples=500),
        }
        mock_info.features = "{'text': Value(dtype='string')}"
        mock_info.dataset_size = 1073741824  # 1 GB
        mock_info.download_size = 536870912  # 512 MB

        mock_builder = Mock()
        mock_builder.info = mock_info
        mock_load_builder.return_value = mock_builder

        info = await get_repo_info("user/dataset")

        assert info is not None
        assert info["repo_id"] == "user/dataset"
        assert info["description"] == "Test dataset description"
        assert info["homepage"] == "https://example.com"
        assert info["license"] == "MIT"
        assert info["citation"] == "Citation text"
        assert set(info["splits"]) == {"train", "validation", "test"}
        assert info["num_rows"]["train"] == 10000
        assert info["num_rows"]["validation"] == 1000
        assert info["num_rows"]["test"] == 500
        assert info["dataset_size"] == 1073741824
        assert info["download_size"] == 536870912

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_get_repo_info_minimal_metadata(self, mock_load_builder):
        """Test getting minimal repository metadata."""
        mock_info = Mock()
        mock_info.description = None
        mock_info.homepage = None
        mock_info.license = None
        mock_info.citation = None
        mock_info.splits = None
        mock_info.features = None
        mock_info.dataset_size = None
        mock_info.download_size = None

        mock_builder = Mock()
        mock_builder.info = mock_info
        mock_load_builder.return_value = mock_builder

        info = await get_repo_info("user/dataset")

        assert info is not None
        assert info["repo_id"] == "user/dataset"
        assert info["description"] == ""
        assert info["homepage"] == ""
        assert info["license"] == ""
        assert info["citation"] == ""
        assert info["splits"] == []
        assert info["features"] == ""
        assert info["num_rows"] == {}
        assert info["dataset_size"] == 0
        assert info["download_size"] == 0

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_get_repo_info_with_token(self, mock_load_builder):
        """Test getting repo info with authentication token."""
        mock_info = Mock()
        mock_info.description = "Test"
        mock_info.homepage = ""
        mock_info.license = ""
        mock_info.citation = ""
        mock_info.splits = None
        mock_info.features = None
        mock_info.dataset_size = 0
        mock_info.download_size = 0

        mock_builder = Mock()
        mock_builder.info = mock_info
        mock_load_builder.return_value = mock_builder

        info = await get_repo_info("private/dataset", token="hf_token123")

        assert info is not None
        mock_load_builder.assert_called_once_with("private/dataset", token="hf_token123")

    @patch('src.utils.hf_utils.load_dataset_builder')
    async def test_get_repo_info_error(self, mock_load_builder):
        """Test handling errors when fetching repo info."""
        mock_load_builder.side_effect = Exception("Network error")

        info = await get_repo_info("user/dataset")

        assert info is None

    async def test_get_repo_info_invalid_format(self):
        """Test getting info for invalid repo ID."""
        info = await get_repo_info("invalid")

        assert info is None


@pytest.mark.asyncio
class TestGetRepoCard:
    """Test get_repo_card function."""

    @patch('src.utils.hf_utils.httpx.AsyncClient')
    async def test_get_repo_card_success(self, mock_client_class):
        """Test successfully fetching repository README."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# Dataset Card\n\nThis is a test dataset."

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        card = await get_repo_card("user/dataset")

        assert card == "# Dataset Card\n\nThis is a test dataset."

    @patch('src.utils.hf_utils.httpx.AsyncClient')
    async def test_get_repo_card_with_token(self, mock_client_class):
        """Test fetching README with authentication token."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "# Private Dataset"

        mock_get = AsyncMock(return_value=mock_response)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = mock_get
        mock_client_class.return_value = mock_client

        card = await get_repo_card("private/dataset", token="hf_token123")

        assert card == "# Private Dataset"

        # Verify Authorization header was included
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer hf_token123"

    @patch('src.utils.hf_utils.httpx.AsyncClient')
    async def test_get_repo_card_not_found(self, mock_client_class):
        """Test fetching README when not found."""
        mock_response = Mock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        card = await get_repo_card("user/dataset")

        assert card is None

    @patch('src.utils.hf_utils.httpx.AsyncClient')
    async def test_get_repo_card_network_error(self, mock_client_class):
        """Test handling network errors."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Timeout"))
        mock_client_class.return_value = mock_client

        card = await get_repo_card("user/dataset")

        assert card is None

    async def test_get_repo_card_invalid_format(self):
        """Test fetching card for invalid repo ID."""
        card = await get_repo_card("invalid")

        assert card is None


class TestParseRepoId:
    """Test parse_repo_id function."""

    def test_parse_repo_id_valid(self):
        """Test parsing valid repository IDs."""
        username, dataset = parse_repo_id("roneneldan/TinyStories")

        assert username == "roneneldan"
        assert dataset == "TinyStories"

    def test_parse_repo_id_with_hyphens_underscores(self):
        """Test parsing repo ID with hyphens and underscores."""
        username, dataset = parse_repo_id("user-name/dataset_name-v2")

        assert username == "user-name"
        assert dataset == "dataset_name-v2"

    def test_parse_repo_id_invalid_format(self):
        """Test parsing invalid repo ID."""
        username, dataset = parse_repo_id("invalid")

        assert username is None
        assert dataset is None

    def test_parse_repo_id_empty_string(self):
        """Test parsing empty string."""
        username, dataset = parse_repo_id("")

        assert username is None
        assert dataset is None

    def test_parse_repo_id_only_slash(self):
        """Test parsing repo ID with only slash."""
        username, dataset = parse_repo_id("/")

        assert username is None
        assert dataset is None


class TestGetRepoUrl:
    """Test get_repo_url function."""

    def test_get_repo_url_valid(self):
        """Test generating URL for valid repo ID."""
        url = get_repo_url("roneneldan/TinyStories")

        assert url == "https://huggingface.co/datasets/roneneldan/TinyStories"

    def test_get_repo_url_with_special_chars(self):
        """Test URL generation with special characters."""
        url = get_repo_url("user-name/dataset-name_v2.1")

        assert url == "https://huggingface.co/datasets/user-name/dataset-name_v2.1"

    def test_get_repo_url_invalid(self):
        """Test URL generation for invalid repo ID."""
        url = get_repo_url("invalid")

        assert url is None

    def test_get_repo_url_empty_string(self):
        """Test URL generation for empty string."""
        url = get_repo_url("")

        assert url is None
