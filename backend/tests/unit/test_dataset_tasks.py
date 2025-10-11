"""
Unit tests for Celery dataset tasks.

This module tests the background tasks for downloading and tokenizing datasets,
including progress tracking, error handling, and database updates.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from uuid import uuid4
from pathlib import Path

from datasets import Dataset as HFDataset, DatasetDict
from sqlalchemy.ext.asyncio import AsyncSession

from src.workers.dataset_tasks import (
    download_dataset_task,
    tokenize_dataset_task,
    DatasetTask,
)
from src.models.dataset import Dataset, DatasetStatus
from src.schemas.dataset import DatasetUpdate


class TestDatasetTaskBase:
    """Tests for DatasetTask base class."""

    @pytest.mark.asyncio
    async def test_get_session(self):
        """Test async session creation."""
        task = DatasetTask()
        session = await task.get_session()

        assert session is not None
        assert isinstance(session, AsyncSession)

        # Verify session is reused
        session2 = await task.get_session()
        assert session is session2

        await task.close_session()

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test session cleanup."""
        task = DatasetTask()
        session = await task.get_session()

        await task.close_session()

        assert task._session is None

    @patch('httpx.Client')
    def test_emit_progress(self, mock_client_class):
        """Test WebSocket progress emission via HTTP."""
        task = DatasetTask()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        dataset_id = str(uuid4())
        task.emit_progress(
            dataset_id,
            "progress",
            {"progress": 50.0, "message": "Test message"}
        )

        # Verify HTTP POST was called
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        assert call_args[1]['json']['channel'] == f"datasets/{dataset_id}/progress"
        assert call_args[1]['json']['event'] == "progress"
        assert call_args[1]['json']['data']['progress'] == 50.0

    @patch('httpx.Client')
    def test_emit_progress_failure(self, mock_client_class):
        """Test WebSocket emission handles errors gracefully."""
        task = DatasetTask()
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Network error")
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Should not raise exception
        task.emit_progress(
            str(uuid4()),
            "progress",
            {"progress": 50.0}
        )

    @pytest.mark.asyncio
    async def test_update_dataset_status(self):
        """Test dataset status update in database."""
        task = DatasetTask()
        dataset_id = uuid4()

        with patch('src.workers.dataset_tasks.DatasetService') as mock_service:
            mock_service.update_dataset = AsyncMock()

            await task.update_dataset_status(
                dataset_id,
                DatasetStatus.PROCESSING,
                progress=50.0,
                error_message="Test error"
            )

            # Verify update was called with correct parameters
            mock_service.update_dataset.assert_called_once()
            call_args = mock_service.update_dataset.call_args

            updates = call_args[0][2]
            assert updates.status == DatasetStatus.PROCESSING.value
            assert updates.progress == 50.0
            assert updates.error_message == "Test error"


class TestDownloadDatasetTask:
    """Tests for download_dataset_task."""

    @patch('src.workers.dataset_tasks.load_dataset')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_download_dataset_success(
        self,
        mock_client_class,
        mock_service,
        mock_load_dataset
    ):
        """Test successful dataset download."""
        # Setup mocks
        dataset_id = str(uuid4())
        repo_id = "test/dataset"

        # Mock HuggingFace dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.size_in_bytes = 5000000
        mock_dataset.save_to_disk = Mock()
        mock_load_dataset.return_value = mock_dataset

        # Mock HTTP client for WebSocket
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Mock database service
        mock_service.update_dataset = AsyncMock()
        mock_service.get_dataset = AsyncMock()

        # Execute task directly
        result = download_dataset_task(dataset_id, repo_id)

        # Verify results
        assert result['status'] == 'ready'
        assert result['dataset_id'] == dataset_id
        assert result['num_samples'] == 1000
        assert result['size_bytes'] == 5000000

        # Verify dataset was loaded with correct parameters
        mock_load_dataset.assert_called_once()
        call_args = mock_load_dataset.call_args
        assert call_args[0][0] == repo_id

        # Verify dataset was saved
        mock_dataset.save_to_disk.assert_called_once()

        # Verify progress was emitted multiple times
        assert mock_client.post.call_count >= 3  # start, progress, complete

    @patch('src.workers.dataset_tasks.load_dataset')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_download_dataset_with_config_and_split(
        self,
        mock_client_class,
        mock_service,
        mock_load_dataset
    ):
        """Test dataset download with specific config and split."""
        dataset_id = str(uuid4())
        repo_id = "test/dataset"
        config = "default"
        split = "train"
        access_token = "test_token"

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=500)
        mock_dataset.size_in_bytes = 2000000
        mock_dataset.save_to_disk = Mock()
        mock_load_dataset.return_value = mock_dataset

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Mock database
        mock_service.update_dataset = AsyncMock()

        # Execute task directly
        result = download_dataset_task(dataset_id, repo_id, access_token, split, config)

        # Verify load_dataset was called with all parameters
        mock_load_dataset.assert_called_once()
        call_args = mock_load_dataset.call_args
        assert call_args[1]['name'] == config
        assert call_args[1]['split'] == split
        assert call_args[1]['token'] == access_token

    @patch('src.workers.dataset_tasks.load_dataset')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_download_dataset_failure(
        self,
        mock_client_class,
        mock_service,
        mock_load_dataset
    ):
        """Test dataset download error handling."""
        dataset_id = str(uuid4())
        repo_id = "test/nonexistent"

        # Mock load_dataset to raise exception
        mock_load_dataset.side_effect = Exception("Dataset not found")

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Mock database
        mock_service.update_dataset = AsyncMock()

        # Execute task and expect exception
        with pytest.raises(Exception) as exc_info:
            download_dataset_task(dataset_id, repo_id)

        assert "Dataset not found" in str(exc_info.value)

        # Verify error was emitted
        error_calls = [
            call for call in mock_client.post.call_args_list
            if 'error' in str(call)
        ]
        assert len(error_calls) > 0

    @patch('src.workers.dataset_tasks.load_dataset')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_download_dataset_database_commit_failure(
        self,
        mock_client_class,
        mock_service,
        mock_load_dataset
    ):
        """Test handling of database commit failure."""
        dataset_id = str(uuid4())
        repo_id = "test/dataset"

        # Mock successful download
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.size_in_bytes = 5000000
        mock_dataset.save_to_disk = Mock()
        mock_load_dataset.return_value = mock_dataset

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Mock database update to fail
        mock_service.update_dataset = AsyncMock(
            side_effect=Exception("Database error")
        )

        # Execute task and expect exception
        with pytest.raises(Exception) as exc_info:
            download_dataset_task(dataset_id, repo_id)

        assert "Database error" in str(exc_info.value)


class TestTokenizeDatasetTask:
    """Tests for tokenize_dataset_task."""

    @patch('src.workers.dataset_tasks.TokenizationService')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_tokenize_dataset_success(
        self,
        mock_client_class,
        mock_service,
        mock_tokenization_service
    ):
        """Test successful dataset tokenization."""
        dataset_id = str(uuid4())
        tokenizer_name = "gpt2"
        max_length = 512
        stride = 0

        # Mock dataset in database
        mock_dataset_obj = Mock()
        mock_dataset_obj.raw_path = "/data/test_dataset"
        mock_service.get_dataset = AsyncMock(return_value=mock_dataset_obj)
        mock_service.update_dataset = AsyncMock()

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenization_service.load_tokenizer.return_value = mock_tokenizer

        # Mock HuggingFace dataset
        mock_hf_dataset = Mock()
        mock_hf_dataset.__len__ = Mock(return_value=1000)
        mock_tokenization_service.load_dataset_from_disk.return_value = mock_hf_dataset

        # Mock schema analysis
        mock_schema = {
            'recommended_column': 'text',
            'text_columns': ['text'],
            'column_info': {'text': 'string'},
            'all_columns': ['text', 'label'],
            'is_multi_column': False
        }
        mock_tokenization_service.analyze_dataset_schema.return_value = mock_schema

        # Mock tokenized dataset
        mock_tokenized = Mock()
        mock_tokenization_service.tokenize_dataset.return_value = mock_tokenized

        # Mock statistics
        mock_stats = {
            'num_tokens': 250000,
            'avg_seq_length': 250.5,
            'min_seq_length': 10,
            'max_seq_length': 512
        }
        mock_tokenization_service.calculate_statistics.return_value = mock_stats
        mock_tokenization_service.save_tokenized_dataset.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Execute task directly
        result = tokenize_dataset_task(str(dataset_id), tokenizer_name, max_length, stride)

        # Verify results
        assert result['status'] == 'ready'
        assert result['dataset_id'] == str(dataset_id)
        assert result['statistics'] == mock_stats

        # Verify tokenizer was loaded
        mock_tokenization_service.load_tokenizer.assert_called_once_with(tokenizer_name)

        # Verify dataset was tokenized
        mock_tokenization_service.tokenize_dataset.assert_called_once()

        # Verify statistics were calculated
        mock_tokenization_service.calculate_statistics.assert_called_once()

        # Verify progress was emitted
        assert mock_client.post.call_count >= 5

    @patch('src.workers.dataset_tasks.TokenizationService')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_tokenize_dataset_with_dataset_dict(
        self,
        mock_client_class,
        mock_service,
        mock_tokenization_service
    ):
        """Test tokenization with DatasetDict (multi-split dataset)."""
        dataset_id = str(uuid4())

        # Mock dataset in database
        mock_dataset_obj = Mock()
        mock_dataset_obj.raw_path = "/data/test_dataset"
        mock_service.get_dataset = AsyncMock(return_value=mock_dataset_obj)
        mock_service.update_dataset = AsyncMock()

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenization_service.load_tokenizer.return_value = mock_tokenizer

        # Mock DatasetDict with multiple splits
        mock_train_split = Mock()
        mock_train_split.__len__ = Mock(return_value=800)
        mock_dataset_dict = DatasetDict({'train': mock_train_split, 'test': Mock()})
        mock_tokenization_service.load_dataset_from_disk.return_value = mock_dataset_dict

        # Mock schema analysis
        mock_schema = {
            'recommended_column': 'text',
            'text_columns': ['text'],
            'column_info': {'text': 'string'},
            'all_columns': ['text'],
            'is_multi_column': False
        }
        mock_tokenization_service.analyze_dataset_schema.return_value = mock_schema

        # Mock tokenization
        mock_tokenized = Mock()
        mock_tokenization_service.tokenize_dataset.return_value = mock_tokenized

        # Mock statistics
        mock_stats = {'num_tokens': 200000, 'avg_seq_length': 250.0, 'min_seq_length': 10, 'max_seq_length': 512}
        mock_tokenization_service.calculate_statistics.return_value = mock_stats
        mock_tokenization_service.save_tokenized_dataset.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Execute task directly
        result = tokenize_dataset_task(str(dataset_id), "gpt2", 512, 0)

        # Verify train split was used
        assert result['status'] == 'ready'

    @patch('src.workers.dataset_tasks.TokenizationService')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_tokenize_dataset_no_text_column(
        self,
        mock_client_class,
        mock_service,
        mock_tokenization_service
    ):
        """Test tokenization failure when no text column found."""
        dataset_id = str(uuid4())

        # Mock dataset in database
        mock_dataset_obj = Mock()
        mock_dataset_obj.raw_path = "/data/test_dataset"
        mock_service.get_dataset = AsyncMock(return_value=mock_dataset_obj)
        mock_service.update_dataset = AsyncMock()

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenization_service.load_tokenizer.return_value = mock_tokenizer

        # Mock dataset
        mock_hf_dataset = Mock()
        mock_tokenization_service.load_dataset_from_disk.return_value = mock_hf_dataset

        # Mock schema analysis with no text column
        mock_schema = {
            'recommended_column': None,
            'text_columns': [],
            'column_info': {'label': 'int64'},
            'all_columns': ['label'],
            'is_multi_column': False
        }
        mock_tokenization_service.analyze_dataset_schema.return_value = mock_schema

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Execute task and expect error
        with pytest.raises(Exception) as exc_info:
            tokenize_dataset_task(str(dataset_id), "gpt2", 512, 0)

        assert "No suitable text column found" in str(exc_info.value)

    @patch('src.workers.dataset_tasks.TokenizationService')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_tokenize_dataset_not_found(
        self,
        mock_client_class,
        mock_service,
        mock_tokenization_service
    ):
        """Test tokenization when dataset not found in database."""
        dataset_id = str(uuid4())

        # Mock dataset not found
        mock_service.get_dataset = AsyncMock(return_value=None)
        mock_service.update_dataset = AsyncMock()

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Execute task and expect error
        with pytest.raises(Exception) as exc_info:
            tokenize_dataset_task(str(dataset_id), "gpt2", 512, 0)

        assert "not found" in str(exc_info.value).lower()

    @patch('src.workers.dataset_tasks.TokenizationService')
    @patch('src.workers.dataset_tasks.DatasetService')
    @patch('httpx.Client')
    def test_tokenize_dataset_statistics_failure(
        self,
        mock_client_class,
        mock_service,
        mock_tokenization_service
    ):
        """Test handling of statistics calculation failure."""
        dataset_id = str(uuid4())

        # Mock dataset in database
        mock_dataset_obj = Mock()
        mock_dataset_obj.raw_path = "/data/test_dataset"
        mock_service.get_dataset = AsyncMock(return_value=mock_dataset_obj)
        mock_service.update_dataset = AsyncMock()

        # Mock successful tokenization
        mock_tokenizer = Mock()
        mock_tokenization_service.load_tokenizer.return_value = mock_tokenizer

        mock_hf_dataset = Mock()
        mock_hf_dataset.__len__ = Mock(return_value=1000)
        mock_tokenization_service.load_dataset_from_disk.return_value = mock_hf_dataset

        mock_schema = {
            'recommended_column': 'text',
            'text_columns': ['text'],
            'column_info': {'text': 'string'},
            'all_columns': ['text'],
            'is_multi_column': False
        }
        mock_tokenization_service.analyze_dataset_schema.return_value = mock_schema

        mock_tokenized = Mock()
        mock_tokenization_service.tokenize_dataset.return_value = mock_tokenized

        # Mock statistics calculation to fail
        mock_tokenization_service.calculate_statistics.side_effect = ValueError(
            "No valid tokenized samples found"
        )

        # Mock HTTP client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client_class.return_value = mock_client

        # Execute task and expect error
        with pytest.raises(Exception) as exc_info:
            tokenize_dataset_task(str(dataset_id), "gpt2", 512, 0)

        assert "No valid tokenized samples found" in str(exc_info.value)
