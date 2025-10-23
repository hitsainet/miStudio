"""
Unit tests for file_utils module.

Tests file utility functions including directory management,
size calculations, and path safety checks.
"""

import pytest
from pathlib import Path
from src.utils.file_utils import (
    ensure_dir,
    get_directory_size,
    delete_directory,
    format_size,
    get_file_extension,
    is_safe_path,
)


class TestEnsureDir:
    """Test ensure_dir function."""

    def test_ensure_dir_creates_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_with_existing_directory(self, tmp_path):
        """Test with existing directory (should not raise error)."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = ensure_dir(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_ensure_dir_creates_nested_directories(self, tmp_path):
        """Test creating nested directory structure."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensure_dir(nested_dir)

        assert nested_dir.exists()
        assert (tmp_path / "level1").exists()
        assert (tmp_path / "level1" / "level2").exists()
        assert result == nested_dir

    def test_ensure_dir_with_string_path(self, tmp_path):
        """Test ensure_dir with string path."""
        new_dir_str = str(tmp_path / "string_path")

        result = ensure_dir(new_dir_str)

        assert Path(new_dir_str).exists()
        assert isinstance(result, Path)


class TestGetDirectorySize:
    """Test get_directory_size function."""

    def test_get_directory_size_empty_directory(self, tmp_path):
        """Test getting size of empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        size = get_directory_size(empty_dir)

        assert size == 0

    def test_get_directory_size_with_single_file(self, tmp_path):
        """Test directory with single file."""
        dir_with_file = tmp_path / "with_file"
        dir_with_file.mkdir()

        file_path = dir_with_file / "test.txt"
        content = "Hello, World!"
        file_path.write_text(content)

        size = get_directory_size(dir_with_file)

        assert size == len(content.encode())

    def test_get_directory_size_with_multiple_files(self, tmp_path):
        """Test directory with multiple files."""
        dir_path = tmp_path / "multiple_files"
        dir_path.mkdir()

        file1 = dir_path / "file1.txt"
        file2 = dir_path / "file2.txt"
        file3 = dir_path / "file3.txt"

        content1 = "File 1 content"
        content2 = "File 2 longer content"
        content3 = "File 3"

        file1.write_text(content1)
        file2.write_text(content2)
        file3.write_text(content3)

        size = get_directory_size(dir_path)

        expected_size = (
            len(content1.encode()) +
            len(content2.encode()) +
            len(content3.encode())
        )
        assert size == expected_size

    def test_get_directory_size_with_nested_directories(self, tmp_path):
        """Test directory with nested subdirectories."""
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        subdir1 = root_dir / "subdir1"
        subdir2 = root_dir / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()

        (root_dir / "file1.txt").write_text("Content 1")
        (subdir1 / "file2.txt").write_text("Content 2")
        (subdir2 / "file3.txt").write_text("Content 3")

        size = get_directory_size(root_dir)

        expected_size = (
            len("Content 1".encode()) +
            len("Content 2".encode()) +
            len("Content 3".encode())
        )
        assert size == expected_size

    def test_get_directory_size_nonexistent_path(self, tmp_path):
        """Test getting size of nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        size = get_directory_size(nonexistent)

        assert size == 0

    def test_get_directory_size_single_file_path(self, tmp_path):
        """Test getting size when path is a single file."""
        file_path = tmp_path / "single_file.txt"
        content = "File content"
        file_path.write_text(content)

        size = get_directory_size(file_path)

        assert size == len(content.encode())

    def test_get_directory_size_with_binary_files(self, tmp_path):
        """Test directory size with binary files."""
        dir_path = tmp_path / "binary_files"
        dir_path.mkdir()

        binary_file = dir_path / "data.bin"
        binary_data = b'\x00\x01\x02\x03\x04\x05'
        binary_file.write_bytes(binary_data)

        size = get_directory_size(dir_path)

        assert size == len(binary_data)


class TestDeleteDirectory:
    """Test delete_directory function."""

    def test_delete_directory_removes_directory(self, tmp_path):
        """Test deleting a directory."""
        dir_to_delete = tmp_path / "to_delete"
        dir_to_delete.mkdir()
        assert dir_to_delete.exists()

        result = delete_directory(dir_to_delete)

        assert not dir_to_delete.exists()
        assert result is True

    def test_delete_directory_with_files(self, tmp_path):
        """Test deleting directory with files."""
        dir_path = tmp_path / "with_files"
        dir_path.mkdir()

        (dir_path / "file1.txt").write_text("Content 1")
        (dir_path / "file2.txt").write_text("Content 2")

        result = delete_directory(dir_path)

        assert not dir_path.exists()
        assert result is True

    def test_delete_directory_with_nested_structure(self, tmp_path):
        """Test deleting directory with nested subdirectories."""
        root_dir = tmp_path / "root"
        root_dir.mkdir()

        subdir1 = root_dir / "subdir1"
        subdir2 = root_dir / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()

        (root_dir / "file.txt").write_text("Content")
        (subdir1 / "file.txt").write_text("Content")
        (subdir2 / "file.txt").write_text("Content")

        result = delete_directory(root_dir)

        assert not root_dir.exists()
        assert result is True

    def test_delete_directory_nonexistent_with_missing_ok_true(self, tmp_path):
        """Test deleting nonexistent directory with missing_ok=True."""
        nonexistent = tmp_path / "nonexistent"

        result = delete_directory(nonexistent, missing_ok=True)

        assert result is False

    def test_delete_directory_nonexistent_with_missing_ok_false(self, tmp_path):
        """Test deleting nonexistent directory with missing_ok=False."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError) as exc_info:
            delete_directory(nonexistent, missing_ok=False)

        assert "Directory not found" in str(exc_info.value)

    def test_delete_directory_single_file(self, tmp_path):
        """Test deleting a single file (not directory)."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("Content")
        assert file_path.exists()

        result = delete_directory(file_path)

        assert not file_path.exists()
        assert result is True


class TestFormatSize:
    """Test format_size function."""

    def test_format_size_bytes(self):
        """Test formatting sizes in bytes."""
        assert format_size(0) == "0 B"
        assert format_size(512) == "512 B"
        assert format_size(1000) == "1000 B"

    def test_format_size_kilobytes(self):
        """Test formatting sizes in kilobytes."""
        assert format_size(1024) == "1.00 KB"
        assert format_size(1536) == "1.50 KB"
        assert format_size(10240) == "10.0 KB"
        assert format_size(102400) == "100 KB"

    def test_format_size_megabytes(self):
        """Test formatting sizes in megabytes."""
        assert format_size(1048576) == "1.00 MB"  # 1 MB
        assert format_size(5242880) == "5.00 MB"  # 5 MB
        assert format_size(104857600) == "100 MB"  # 100 MB

    def test_format_size_gigabytes(self):
        """Test formatting sizes in gigabytes."""
        assert format_size(1073741824) == "1.00 GB"  # 1 GB
        assert format_size(5368709120) == "5.00 GB"  # 5 GB
        assert format_size(107374182400) == "100 GB"  # 100 GB

    def test_format_size_terabytes(self):
        """Test formatting sizes in terabytes."""
        assert format_size(1099511627776) == "1.00 TB"  # 1 TB
        assert format_size(10995116277760) == "10.0 TB"  # 10 TB

    def test_format_size_petabytes(self):
        """Test formatting sizes in petabytes."""
        assert format_size(1125899906842624) == "1.00 PB"  # 1 PB

    def test_format_size_none(self):
        """Test formatting None size."""
        assert format_size(None) == "Unknown"

    def test_format_size_negative(self):
        """Test formatting negative size."""
        assert format_size(-100) == "Unknown"

    def test_format_size_precision(self):
        """Test formatting precision for different ranges."""
        # Small values (< 10): 2 decimal places
        assert format_size(1536) == "1.50 KB"
        assert format_size(2560) == "2.50 KB"

        # Medium values (10-100): 1 decimal place
        assert format_size(15360) == "15.0 KB"
        assert format_size(25600) == "25.0 KB"

        # Large values (>= 100): 0 decimal places
        assert format_size(153600) == "150 KB"
        assert format_size(256000) == "250 KB"


class TestGetFileExtension:
    """Test get_file_extension function."""

    def test_get_file_extension_common_types(self):
        """Test getting extension for common file types."""
        assert get_file_extension("file.txt") == "txt"
        assert get_file_extension("image.png") == "png"
        assert get_file_extension("data.json") == "json"
        assert get_file_extension("document.pdf") == "pdf"

    def test_get_file_extension_multiple_dots(self):
        """Test file with multiple dots in name."""
        assert get_file_extension("archive.tar.gz") == "gz"
        assert get_file_extension("data.backup.json") == "json"

    def test_get_file_extension_no_extension(self):
        """Test file without extension."""
        assert get_file_extension("README") == ""
        assert get_file_extension("Makefile") == ""

    def test_get_file_extension_with_path(self):
        """Test getting extension from full path."""
        assert get_file_extension("/path/to/file.txt") == "txt"
        assert get_file_extension("relative/path/data.csv") == "csv"

    def test_get_file_extension_hidden_file(self):
        """Test hidden file (starts with dot)."""
        # Hidden files without second extension have no extension
        assert get_file_extension(".gitignore") == ""
        assert get_file_extension(".env") == ""

    def test_get_file_extension_uppercase(self):
        """Test extension case handling."""
        assert get_file_extension("FILE.TXT") == "TXT"
        assert get_file_extension("Image.PNG") == "PNG"


class TestIsSafePath:
    """Test is_safe_path function."""

    def test_is_safe_path_valid_subdirectory(self, tmp_path):
        """Test valid path within base directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        target = base_dir / "subdir" / "file.txt"

        assert is_safe_path(base_dir, target) is True

    def test_is_safe_path_same_directory(self, tmp_path):
        """Test path is same as base directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        assert is_safe_path(base_dir, base_dir) is True

    def test_is_safe_path_directory_traversal_attack(self, tmp_path):
        """Test directory traversal attempt (../)."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Try to access parent directory
        target = base_dir / ".." / "outside.txt"

        assert is_safe_path(base_dir, target) is False

    def test_is_safe_path_absolute_path_outside_base(self, tmp_path):
        """Test absolute path outside base directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()

        target = outside_dir / "file.txt"

        assert is_safe_path(base_dir, target) is False

    def test_is_safe_path_nested_traversal(self, tmp_path):
        """Test nested directory traversal."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        # Try multiple levels of traversal
        target = base_dir / "sub" / ".." / ".." / ".." / "outside.txt"

        assert is_safe_path(base_dir, target) is False

    def test_is_safe_path_with_string_paths(self, tmp_path):
        """Test with string paths instead of Path objects."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        target = str(base_dir / "subdir" / "file.txt")

        assert is_safe_path(str(base_dir), target) is True

    def test_is_safe_path_symlink_outside_base(self, tmp_path):
        """Test symlink pointing outside base directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("Secret data")

        # Create symlink inside base pointing outside
        symlink = base_dir / "link"
        try:
            symlink.symlink_to(outside_file)

            # Symlink resolves to outside base, should be unsafe
            assert is_safe_path(base_dir, symlink) is False
        except OSError:
            # Skip test if symlinks not supported (e.g., Windows without admin)
            pytest.skip("Symlinks not supported on this system")
