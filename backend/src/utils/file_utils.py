"""
File utility functions for dataset file management.

This module provides utilities for file operations including directory
management, size calculations, and cleanup operations.
"""

import shutil
from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_directory_size(path: str | Path) -> int:
    """
    Calculate total size of a directory by walking the file tree.

    Args:
        path: Directory path to calculate size for

    Returns:
        Total size in bytes

    Raises:
        OSError: If directory cannot be accessed
    """
    dir_path = Path(path)

    if not dir_path.exists():
        return 0

    if dir_path.is_file():
        return dir_path.stat().st_size

    total_size = 0
    for item in dir_path.rglob('*'):
        if item.is_file():
            try:
                total_size += item.stat().st_size
            except (OSError, PermissionError):
                # Skip files we can't access
                continue

    return total_size


def delete_directory(path: str | Path, missing_ok: bool = True) -> bool:
    """
    Delete a directory and all its contents.

    Args:
        path: Directory path to delete
        missing_ok: If True, don't raise error if path doesn't exist

    Returns:
        True if deleted, False if path didn't exist and missing_ok=True

    Raises:
        OSError: If directory cannot be deleted and missing_ok=False
    """
    dir_path = Path(path)

    if not dir_path.exists():
        if missing_ok:
            return False
        raise FileNotFoundError(f"Directory not found: {path}")

    if dir_path.is_file():
        dir_path.unlink()
        return True

    shutil.rmtree(dir_path)
    return True


def format_size(size_bytes: Optional[int]) -> str:
    """
    Format byte size to human-readable string with unit conversion.

    Args:
        size_bytes: Size in bytes, or None

    Returns:
        Formatted size string (e.g., "1.5 GB", "256 MB")

    Examples:
        >>> format_size(1536)
        "1.5 KB"
        >>> format_size(1073741824)
        "1.0 GB"
        >>> format_size(None)
        "Unknown"
    """
    if size_bytes is None or size_bytes < 0:
        return "Unknown"

    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    # Format with appropriate precision
    if size >= 100:
        return f"{size:.0f} {units[unit_index]}"
    elif size >= 10:
        return f"{size:.1f} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


def get_file_extension(path: str | Path) -> str:
    """
    Get file extension from path.

    Args:
        path: File path

    Returns:
        File extension without dot (e.g., "json", "txt")
    """
    return Path(path).suffix.lstrip('.')


def is_safe_path(base_dir: str | Path, target_path: str | Path) -> bool:
    """
    Check if target_path is within base_dir (prevent directory traversal).

    Args:
        base_dir: Base directory path
        target_path: Target path to check

    Returns:
        True if target_path is within base_dir, False otherwise
    """
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()

    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False
