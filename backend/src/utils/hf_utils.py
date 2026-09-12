"""
HuggingFace utility functions for dataset validation and metadata retrieval.

This module provides utilities for interacting with the HuggingFace Hub,
including repository validation and metadata fetching.
"""

import re
from typing import Optional, Dict, Any

import httpx
from datasets import load_dataset_builder


# HuggingFace Hub API base URL
HF_HUB_URL = "https://huggingface.co"
HF_API_URL = "https://huggingface.co/api"


def validate_repo_id(repo_id: str) -> bool:
    """
    Validate HuggingFace repository ID format.

    Args:
        repo_id: Repository ID to validate (e.g., "username/dataset-name")

    Returns:
        True if valid format, False otherwise

    Examples:
        >>> validate_repo_id("roneneldan/TinyStories")
        True
        >>> validate_repo_id("invalid")
        False
    """
    if not repo_id or not isinstance(repo_id, str):
        return False

    # Format: username/dataset-name
    # Username: 1-39 alphanumeric or hyphens, cannot start/end with hyphen
    # Dataset: 1-96 alphanumeric, hyphens, underscores, dots
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,37}[a-zA-Z0-9])?/[a-zA-Z0-9._-]{1,96}$'

    return bool(re.match(pattern, repo_id))


async def check_repo_exists(
    repo_id: str,
    token: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Check if HuggingFace repository exists and is accessible.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: Optional HuggingFace access token for gated datasets

    Returns:
        Tuple of (exists: bool, error_message: Optional[str])

    Examples:
        >>> await check_repo_exists("roneneldan/TinyStories")
        (True, None)
        >>> await check_repo_exists("nonexistent/dataset")
        (False, "Repository not found")
    """
    if not validate_repo_id(repo_id):
        return False, "Invalid repository ID format"

    try:
        # Use datasets library to check if dataset exists
        # This is more reliable than HTTP requests as it handles authentication
        builder = load_dataset_builder(repo_id, token=token)
        return True, None

    except FileNotFoundError:
        return False, "Repository not found"

    except PermissionError:
        return False, "Repository is gated and requires authentication"

    except ValueError as e:
        return False, f"Invalid repository: {str(e)}"

    except Exception as e:
        return False, f"Error accessing repository: {str(e)}"


async def get_repo_info(
    repo_id: str,
    token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get repository metadata from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: Optional HuggingFace access token for gated datasets

    Returns:
        Dictionary with repository metadata, or None if not found

    Metadata includes:
        - description: Dataset description
        - splits: Available splits (train, validation, test)
        - features: Dataset features/columns
        - num_rows: Approximate number of rows per split
        - dataset_size: Approximate size in bytes

    Examples:
        >>> info = await get_repo_info("roneneldan/TinyStories")
        >>> info['splits']
        ['train', 'validation']
    """
    if not validate_repo_id(repo_id):
        return None

    try:
        # Load dataset builder without downloading data
        builder = load_dataset_builder(repo_id, token=token)

        # Extract metadata
        info = {
            "repo_id": repo_id,
            "description": builder.info.description or "",
            "homepage": builder.info.homepage or "",
            "license": builder.info.license or "",
            "citation": builder.info.citation or "",
            "splits": list(builder.info.splits.keys()) if builder.info.splits else [],
            "features": str(builder.info.features) if builder.info.features else "",
            "num_rows": {
                split: builder.info.splits[split].num_examples
                for split in builder.info.splits
            } if builder.info.splits else {},
            "dataset_size": builder.info.dataset_size or 0,
            "download_size": builder.info.download_size or 0,
        }

        return info

    except Exception as e:
        print(f"Error fetching repository info for {repo_id}: {e}")
        return None


async def get_repo_card(
    repo_id: str,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Get repository README/dataset card content.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")
        token: Optional HuggingFace access token

    Returns:
        README content as markdown string, or None if not found
    """
    if not validate_repo_id(repo_id):
        return None

    try:
        url = f"{HF_HUB_URL}/datasets/{repo_id}/raw/main/README.md"
        headers = {}

        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=10.0)

            if response.status_code == 200:
                return response.text

            return None

    except Exception as e:
        print(f"Error fetching dataset card for {repo_id}: {e}")
        return None


def parse_repo_id(repo_id: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse repository ID into username and dataset name.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")

    Returns:
        Tuple of (username, dataset_name), or (None, None) if invalid

    Examples:
        >>> parse_repo_id("roneneldan/TinyStories")
        ("roneneldan", "TinyStories")
        >>> parse_repo_id("invalid")
        (None, None)
    """
    if not validate_repo_id(repo_id):
        return None, None

    parts = repo_id.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return None, None


def get_repo_url(repo_id: str) -> Optional[str]:
    """
    Get full URL to HuggingFace repository page.

    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")

    Returns:
        Full URL, or None if invalid repo_id

    Examples:
        >>> get_repo_url("roneneldan/TinyStories")
        "https://huggingface.co/datasets/roneneldan/TinyStories"
    """
    if not validate_repo_id(repo_id):
        return None

    return f"{HF_HUB_URL}/datasets/{repo_id}"
