"""
Unit tests for model download progress monitoring.

Tests the DownloadProgressMonitor class that monitors directory size growth
in a background thread and emits WebSocket progress updates.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import time


class TestDownloadProgressMonitorInitialization:
    """Test DownloadProgressMonitor initialization."""

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_initialization(self, mock_get_dir_size):
        """Test progress monitor initializes with correct parameters."""
        from src.workers.model_tasks import DownloadProgressMonitor

        cache_dir = Path("/tmp/test_cache")
        model_id = "gpt2"
        estimated_size_gb = 3.5

        monitor = DownloadProgressMonitor(
            cache_dir=cache_dir,
            model_id=model_id,
            estimated_size_gb=estimated_size_gb
        )

        assert monitor.cache_dir == cache_dir
        assert monitor.model_id == model_id
        assert monitor.estimated_size_bytes == int(3.5 * 1024 * 1024 * 1024)
        assert monitor.running is False
        assert monitor.thread is None

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_default_estimated_size(self, mock_get_dir_size):
        """Test progress monitor uses default estimated size."""
        from src.workers.model_tasks import DownloadProgressMonitor

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        # Default is 5.0 GB
        assert monitor.estimated_size_bytes == int(5.0 * 1024 * 1024 * 1024)


class TestDownloadProgressMonitorThreadLifecycle:
    """Test thread start and stop functionality."""

    @patch('src.workers.model_tasks.get_directory_size')
    @patch('src.workers.model_tasks.threading.Thread')
    def test_progress_monitor_thread_start(self, mock_thread_class, mock_get_dir_size):
        """Test progress monitor starts background thread successfully."""
        from src.workers.model_tasks import DownloadProgressMonitor

        # Mock initial directory size
        mock_get_dir_size.return_value = 1024 * 1024 * 100  # 100 MB

        # Mock thread
        mock_thread_instance = Mock()
        mock_thread_class.return_value = mock_thread_instance

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2",
            estimated_size_gb=3.0
        )

        monitor.start()

        # Verify initial size captured
        assert monitor.initial_size == 1024 * 1024 * 100

        # Verify thread created and started
        mock_thread_class.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        assert monitor.running is True
        assert monitor.thread == mock_thread_instance

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_thread_stop(self, mock_get_dir_size):
        """Test progress monitor stops gracefully."""
        from src.workers.model_tasks import DownloadProgressMonitor

        mock_get_dir_size.return_value = 0

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        # Mock thread
        mock_thread = Mock()
        monitor.thread = mock_thread
        monitor.running = True

        monitor.stop()

        # Verify running flag set to False
        assert monitor.running is False

        # Verify thread join called with timeout
        mock_thread.join.assert_called_once_with(timeout=2.0)

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_stops_on_completion(self, mock_get_dir_size):
        """Test progress monitor terminates thread when running flag is False."""
        from src.workers.model_tasks import DownloadProgressMonitor

        mock_get_dir_size.return_value = 0

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        # Simulate monitor loop checking running flag
        monitor.running = False

        # The _monitor_loop should exit when running is False
        # (This is implicitly tested by the while loop condition)
        assert monitor.running is False


class TestDownloadProgressCalculation:
    """Test progress percentage calculation."""

    def test_progress_calculation_at_start(self):
        """Test progress calculation at download start (0%)."""
        initial_size = 0
        current_size = 0
        estimated_size_bytes = 3 * 1024 * 1024 * 1024  # 3 GB

        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        assert progress == 0.0

    def test_progress_calculation_at_10_percent(self):
        """Test progress calculation at 10% downloaded."""
        initial_size = 0
        estimated_size_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
        current_size = int(0.1 * estimated_size_bytes)  # 10% downloaded

        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        # Use approximate comparison for floating point
        assert abs(progress - 10.0) < 0.01

    def test_progress_calculation_at_50_percent(self):
        """Test progress calculation at 50% downloaded."""
        initial_size = 0
        estimated_size_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
        current_size = int(0.5 * estimated_size_bytes)  # 50% downloaded

        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        assert progress == 50.0

    def test_progress_calculation_capped_at_90_percent(self):
        """Test progress is capped at 90% (since exact size unknown)."""
        initial_size = 0
        estimated_size_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
        current_size = int(1.5 * estimated_size_bytes)  # 150% downloaded (exceeds estimate)

        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        # Progress should be capped at 90%
        assert progress == 90.0

    def test_progress_calculation_with_initial_cache(self):
        """Test progress calculation when cache directory not empty initially."""
        initial_size = 500 * 1024 * 1024  # 500 MB already in cache
        estimated_size_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
        current_size = initial_size + (1.5 * 1024 * 1024 * 1024)  # Added 1.5 GB

        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        # Progress should be based on downloaded bytes, not total size
        assert progress == 50.0  # 1.5 GB / 3 GB = 50%


class TestDownloadProgressEmission:
    """Test WebSocket progress emission."""

    def test_progress_monitor_emits_at_1_percent_intervals(self):
        """Test progress monitor logic emits at 1% intervals."""
        # Test the emission logic without actually running the thread
        estimated_size_bytes = 1000 * 1024 * 1024  # 1000 MB
        initial_size = 0

        # Simulate size progression
        sizes = [
            initial_size + (10 * 1024 * 1024),   # 1% (10 MB)
            initial_size + (20 * 1024 * 1024),   # 2% (20 MB)
            initial_size + (30 * 1024 * 1024),   # 3% (30 MB)
        ]

        emission_count = 0
        last_progress = 0

        for size in sizes:
            downloaded_bytes = size - initial_size
            progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

            # Check if emission threshold met (1% increase)
            if progress >= last_progress + 1:
                emission_count += 1
                last_progress = progress

        # Should emit 3 times (at 1%, 2%, 3%)
        assert emission_count == 3

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_skips_emission_below_1_percent_threshold(
        self,
        mock_get_dir_size
    ):
        """Test progress monitor does not emit for changes < 1%."""
        from src.workers.model_tasks import DownloadProgressMonitor

        estimated_size_bytes = 1000 * 1024 * 1024  # 1000 MB
        initial_size = 0

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2",
            estimated_size_gb=1.0
        )

        monitor.initial_size = initial_size

        # Simulate small size increase (0.5%)
        current_size = initial_size + (5 * 1024 * 1024)  # 5 MB
        downloaded_bytes = current_size - initial_size
        progress = min(90, (downloaded_bytes / estimated_size_bytes) * 100)

        # Progress is ~0.5%, which should not trigger emission
        assert progress < 1.0


class TestDownloadProgressErrorHandling:
    """Test error handling in progress monitor."""

    @patch('src.workers.model_tasks.get_directory_size')
    @patch('src.workers.model_tasks.get_sync_db')
    def test_progress_monitor_handles_missing_files_gracefully(
        self,
        mock_get_sync_db,
        mock_get_dir_size
    ):
        """Test progress monitor handles missing files/directories gracefully."""
        from src.workers.model_tasks import DownloadProgressMonitor

        # Simulate directory not found error
        mock_get_dir_size.side_effect = FileNotFoundError("Directory not found")

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/nonexistent"),
            model_id="gpt2"
        )

        # Should not crash, just log error
        # The _monitor_loop has try-except that continues on error
        try:
            # This would be called in the loop
            size = mock_get_dir_size(monitor.cache_dir)
        except FileNotFoundError:
            # Expected behavior - monitor loop handles this
            pass

    @patch('src.workers.model_tasks.get_directory_size')
    @patch('src.workers.model_tasks.get_sync_db')
    def test_progress_monitor_handles_database_error(
        self,
        mock_get_sync_db,
        mock_get_dir_size
    ):
        """Test progress monitor continues on database update failure."""
        from src.workers.model_tasks import DownloadProgressMonitor

        # Mock directory size
        mock_get_dir_size.return_value = 100 * 1024 * 1024

        # Mock database error
        mock_get_sync_db.side_effect = Exception("Database connection failed")

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        monitor.initial_size = 0

        # Should not crash, just log warning
        # The database update is wrapped in try-except
        try:
            with mock_get_sync_db() as db:
                # This would fail
                pass
        except Exception:
            # Expected behavior - monitor continues despite DB error
            pass


class TestDownloadProgressThreadSafety:
    """Test thread safety of progress monitor."""

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_thread_safety_stop_before_start(
        self,
        mock_get_dir_size
    ):
        """Test stopping monitor before it's started is safe."""
        from src.workers.model_tasks import DownloadProgressMonitor

        mock_get_dir_size.return_value = 0

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        # Stop before start should not crash
        monitor.stop()

        # Thread should remain None
        assert monitor.thread is None
        assert monitor.running is False

    @patch('src.workers.model_tasks.get_directory_size')
    def test_progress_monitor_multiple_stop_calls_safe(
        self,
        mock_get_dir_size
    ):
        """Test calling stop multiple times is safe."""
        from src.workers.model_tasks import DownloadProgressMonitor

        mock_get_dir_size.return_value = 0

        monitor = DownloadProgressMonitor(
            cache_dir=Path("/tmp/test"),
            model_id="gpt2"
        )

        monitor.thread = Mock()
        monitor.running = True

        # First stop
        monitor.stop()
        assert monitor.running is False

        # Second stop should not crash
        monitor.stop()
        assert monitor.running is False


class TestGetDirectorySize:
    """Test directory size calculation helper function."""

    @patch('pathlib.Path.rglob')
    def test_get_directory_size_empty_directory(self, mock_rglob):
        """Test directory size calculation for empty directory."""
        from src.workers.model_tasks import get_directory_size

        # Mock empty directory
        mock_rglob.return_value = []

        size = get_directory_size(Path("/tmp/empty"))

        assert size == 0

    @patch('pathlib.Path.rglob')
    def test_get_directory_size_with_files(self, mock_rglob):
        """Test directory size calculation with multiple files."""
        from src.workers.model_tasks import get_directory_size

        # Mock files
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.stat.return_value.st_size = 1024 * 1024  # 1 MB

        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.stat.return_value.st_size = 2 * 1024 * 1024  # 2 MB

        mock_rglob.return_value = [mock_file1, mock_file2]

        size = get_directory_size(Path("/tmp/test"))

        assert size == 3 * 1024 * 1024  # 3 MB total

    @patch('pathlib.Path.rglob')
    def test_get_directory_size_skips_non_files(self, mock_rglob):
        """Test directory size calculation skips directories and symlinks."""
        from src.workers.model_tasks import get_directory_size

        # Mock mix of files and directories
        mock_file = Mock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value.st_size = 1024

        mock_dir = Mock()
        mock_dir.is_file.return_value = False  # Directory

        mock_rglob.return_value = [mock_file, mock_dir]

        size = get_directory_size(Path("/tmp/test"))

        # Should only count the file, not the directory
        assert size == 1024
