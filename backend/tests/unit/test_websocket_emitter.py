"""
Unit tests for the shared WebSocket emitter utility.

Tests the websocket_emitter module which provides standardized WebSocket
emission functionality for Celery workers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from src.workers.websocket_emitter import (
    emit_progress,
    emit_dataset_progress,
    emit_model_progress,
    emit_extraction_progress,
)


class TestEmitProgress:
    """Test the core emit_progress function."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emit_progress_success(self, mock_settings, mock_client_class):
        """Test successful progress emission."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Execute
        result = emit_progress(
            channel="test/channel",
            event="progress",
            data={"progress": 50.0, "status": "processing"}
        )

        # Assert
        assert result is True
        mock_client.__enter__.return_value.post.assert_called_once_with(
            "http://localhost:8000/api/internal/ws/emit",
            json={
                "channel": "test/channel",
                "event": "progress",
                "data": {"progress": 50.0, "status": "processing"}
            },
            timeout=1.0
        )

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emit_progress_non_200_status(self, mock_settings, mock_client_class):
        """Test progress emission with non-200 status code."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Execute
        result = emit_progress(
            channel="test/channel",
            event="error",
            data={"status": "failed"}
        )

        # Assert
        assert result is False

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emit_progress_timeout(self, mock_settings, mock_client_class):
        """Test progress emission with timeout exception."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value = mock_client

        # Execute
        result = emit_progress(
            channel="test/channel",
            event="progress",
            data={"progress": 50.0}
        )

        # Assert
        assert result is False

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emit_progress_generic_exception(self, mock_settings, mock_client_class):
        """Test progress emission with generic exception."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        # Execute
        result = emit_progress(
            channel="test/channel",
            event="progress",
            data={"progress": 50.0}
        )

        # Assert
        assert result is False

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_emit_progress_custom_timeout(self, mock_settings, mock_client_class):
        """Test progress emission with custom timeout value."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Execute
        result = emit_progress(
            channel="test/channel",
            event="progress",
            data={"progress": 50.0},
            timeout=5.0
        )

        # Assert
        assert result is True
        mock_client.__enter__.return_value.post.assert_called_once()
        call_args = mock_client.__enter__.return_value.post.call_args
        assert call_args[1]["timeout"] == 5.0


class TestEmitDatasetProgress:
    """Test the emit_dataset_progress convenience function."""

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_dataset_progress_constructs_correct_channel(self, mock_emit):
        """Test that dataset channel is correctly constructed."""
        # Setup
        mock_emit.return_value = True
        dataset_id = "abc-123"
        event = "progress"
        data = {
            "dataset_id": dataset_id,
            "progress": 75.0,
            "status": "processing",
            "message": "Processing dataset..."
        }

        # Execute
        result = emit_dataset_progress(dataset_id, event, data)

        # Assert
        assert result is True
        # Event names are prefixed with entity type (e.g., 'dataset:progress')
        mock_emit.assert_called_once_with(
            f"datasets/{dataset_id}/progress",
            f"dataset:{event}",
            data
        )

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_dataset_progress_all_event_types(self, mock_emit):
        """Test dataset progress with different event types."""
        # Setup
        mock_emit.return_value = True
        dataset_id = "test-dataset"

        event_types = ["progress", "completed", "error"]

        for event in event_types:
            mock_emit.reset_mock()

            # Execute
            result = emit_dataset_progress(
                dataset_id,
                event,
                {"dataset_id": dataset_id, "status": event}
            )

            # Assert
            assert result is True
            mock_emit.assert_called_once()


class TestEmitModelProgress:
    """Test the emit_model_progress convenience function."""

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_model_progress_constructs_correct_channel(self, mock_emit):
        """Test that model channel is correctly constructed."""
        # Setup
        mock_emit.return_value = True
        model_id = "m_xyz-789"
        event = "progress"
        data = {
            "type": "model_progress",
            "model_id": model_id,
            "progress": 60.0,
            "status": "downloading",
            "message": "Downloaded 3GB / 5GB"
        }

        # Execute
        result = emit_model_progress(model_id, event, data)

        # Assert
        assert result is True
        # Event names are prefixed with entity type (e.g., 'model:progress')
        mock_emit.assert_called_once_with(
            f"models/{model_id}/progress",
            f"model:{event}",
            data
        )

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_model_progress_all_statuses(self, mock_emit):
        """Test model progress with different statuses."""
        # Setup
        mock_emit.return_value = True
        model_id = "m_test-model"

        statuses = ["downloading", "loading", "quantizing", "ready", "error"]

        for status in statuses:
            mock_emit.reset_mock()

            # Execute
            result = emit_model_progress(
                model_id,
                "progress",
                {
                    "model_id": model_id,
                    "progress": 50.0,
                    "status": status,
                    "message": f"Status: {status}"
                }
            )

            # Assert
            assert result is True
            mock_emit.assert_called_once()


class TestEmitExtractionProgress:
    """Test the emit_extraction_progress convenience function."""

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_extraction_progress_constructs_correct_payload(self, mock_emit):
        """Test that extraction progress payload is correctly structured."""
        # Setup
        mock_emit.return_value = True
        model_id = "m_model-123"
        extraction_id = "ext_20250112_153045"
        progress = 45.0
        status = "extracting"
        message = "Processing batch 4/10"

        # Execute
        result = emit_extraction_progress(
            model_id,
            extraction_id,
            progress,
            status,
            message
        )

        # Assert
        assert result is True
        # Event names are prefixed with entity type (e.g., 'extraction:progress')
        mock_emit.assert_called_once_with(
            f"models/{model_id}/extraction",
            "extraction:progress",
            {
                "type": "extraction_progress",
                "model_id": model_id,
                "extraction_id": extraction_id,
                "progress": progress,
                "status": status,
                "message": message,
            }
        )

    @patch("src.workers.websocket_emitter.emit_progress")
    def test_emit_extraction_progress_all_statuses(self, mock_emit):
        """Test extraction progress with different statuses."""
        # Setup
        mock_emit.return_value = True
        model_id = "m_test"
        extraction_id = "ext_test"

        statuses = ["starting", "loading", "extracting", "processing", "saving", "complete", "error"]

        for idx, status in enumerate(statuses):
            mock_emit.reset_mock()
            progress = (idx + 1) * 10.0

            # Execute
            result = emit_extraction_progress(
                model_id,
                extraction_id,
                progress,
                status,
                f"Status: {status}"
            )

            # Assert
            assert result is True
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[0]
            call_data = call_args[2]
            assert call_data["status"] == status
            assert call_data["progress"] == progress


class TestIntegration:
    """Integration tests for WebSocket emitter."""

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_full_dataset_workflow(self, mock_settings, mock_client_class):
        """Test full dataset download workflow with multiple emissions."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        dataset_id = "test-dataset"

        # Execute workflow
        # 1. Start download
        result1 = emit_dataset_progress(
            dataset_id,
            "progress",
            {"dataset_id": dataset_id, "progress": 0.0, "status": "downloading", "message": "Starting..."}
        )

        # 2. Progress update
        result2 = emit_dataset_progress(
            dataset_id,
            "progress",
            {"dataset_id": dataset_id, "progress": 50.0, "status": "downloading", "message": "Downloading..."}
        )

        # 3. Completion
        result3 = emit_dataset_progress(
            dataset_id,
            "completed",
            {"dataset_id": dataset_id, "progress": 100.0, "status": "ready", "message": "Complete"}
        )

        # Assert
        assert all([result1, result2, result3])
        assert mock_client.__enter__.return_value.post.call_count == 3

    @patch("src.workers.websocket_emitter.httpx.Client")
    @patch("src.workers.websocket_emitter.settings")
    def test_full_model_workflow(self, mock_settings, mock_client_class):
        """Test full model download workflow with multiple emissions."""
        # Setup
        mock_settings.websocket_emit_url = "http://localhost:8000/api/internal/ws/emit"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        model_id = "m_test-model"

        # Execute workflow
        # 1. Start download
        result1 = emit_model_progress(
            model_id,
            "progress",
            {"model_id": model_id, "progress": 0.0, "status": "downloading", "message": "Starting..."}
        )

        # 2. Download progress
        result2 = emit_model_progress(
            model_id,
            "progress",
            {"model_id": model_id, "progress": 50.0, "status": "downloading", "message": "Downloaded 2.5GB / 5GB"}
        )

        # 3. Loading
        result3 = emit_model_progress(
            model_id,
            "progress",
            {"model_id": model_id, "progress": 95.0, "status": "loading", "message": "Loading model..."}
        )

        # 4. Ready
        result4 = emit_model_progress(
            model_id,
            "progress",
            {"model_id": model_id, "progress": 100.0, "status": "ready", "message": "Model ready"}
        )

        # Assert
        assert all([result1, result2, result3, result4])
        assert mock_client.__enter__.return_value.post.call_count == 4
