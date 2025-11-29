"""
HuggingFace SAE service layer.

This module contains business logic for downloading and uploading SAEs
from/to HuggingFace Hub. It handles repository preview, file download
with progress tracking, and upload operations.
"""

import logging
import os
import re
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

from ..core.config import settings
from ..schemas.sae import HFFileInfo, HFRepoPreviewResponse

logger = logging.getLogger(__name__)

# Thread pool for running blocking HF operations
_executor = ThreadPoolExecutor(max_workers=2)


class HuggingFaceSAEService:
    """Service class for HuggingFace SAE operations."""

    # Common SAE file patterns
    SAE_FILE_PATTERNS = [
        "*.safetensors",
        "sae_weights.pt",
        "sae_weights.safetensors",
        "cfg.json",
        "config.json",
        "sparsity_*.json",
        "params.npz",  # Gemma Scope format
    ]

    # Known SAE repository patterns (model name hints)
    MODEL_NAME_PATTERNS = {
        "gemma-scope": "gemma",
        "gpt2": "gpt2",
        "llama": "llama",
        "mistral": "mistral",
        "pythia": "pythia",
    }

    @staticmethod
    async def preview_repository(
        repo_id: str,
        access_token: Optional[str] = None
    ) -> HFRepoPreviewResponse:
        """
        Preview a HuggingFace repository to discover available SAEs.

        Args:
            repo_id: HuggingFace repository ID (e.g., "google/gemma-scope-2b-pt-res")
            access_token: Optional access token for private repos

        Returns:
            Repository preview with file list and detected SAE paths
        """
        # Use provided token, or settings token, but treat empty/whitespace/"None" as None
        token = access_token or settings.hf_token
        if not token or not token.strip() or token.strip().lower() == "none":
            token = None

        def _preview() -> HFRepoPreviewResponse:
            api = HfApi(token=token)

            try:
                # Get repository info
                repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
            except RepositoryNotFoundError:
                # Try as dataset
                try:
                    repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
                except RepositoryNotFoundError:
                    raise ValueError(f"Repository not found: {repo_id}")

            # List all files
            try:
                files = list_repo_files(repo_id=repo_id, token=token)
            except Exception as e:
                logger.warning(f"Error listing files: {e}")
                files = []

            # Analyze files
            file_infos: List[HFFileInfo] = []
            sae_paths: List[str] = []
            total_size = 0

            for file_path in files:
                # Check if this looks like an SAE file
                is_sae = HuggingFaceSAEService._is_sae_file(file_path)

                file_info = HFFileInfo(
                    filepath=file_path,
                    size_bytes=None,  # Would need additional API call per file
                    is_sae=is_sae
                )
                file_infos.append(file_info)

                # Track SAE directories/files
                if is_sae:
                    # Get parent directory for grouped SAEs
                    parent = str(Path(file_path).parent)
                    if parent and parent != "." and parent not in sae_paths:
                        sae_paths.append(parent)

            # Detect model name from repo
            model_name = HuggingFaceSAEService._detect_model_name(repo_id)

            # Filter SAE files for the sae_files list and sort by layer, then L0 value
            sae_files = [f for f in file_infos if f.is_sae]
            sae_files.sort(key=lambda f: HuggingFaceSAEService._extract_sort_key(f.filepath))

            # Sort SAE paths by layer number and L0 as well
            sae_paths.sort(key=HuggingFaceSAEService._extract_sort_key)

            return HFRepoPreviewResponse(
                repo_id=repo_id,
                repo_type="model",
                description=None,  # Would need additional API call
                files=file_infos,
                sae_files=sae_files,
                sae_paths=sae_paths if sae_paths else ["."],
                model_name=model_name,
                total_size_bytes=total_size if total_size > 0 else None
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _preview)

    @staticmethod
    def _extract_sort_key(path: str) -> tuple[int, int, int]:
        """Extract layer, dimension, and L0 value from a path for natural sorting.

        Handles paths like:
        - layer_1/width_65k/average_l0_71/...
        - layer_10/width_16k/average_l0_139/...
        - gemma-scope-2b-pt-res/layer_5/width_16k/average_l0_82/...

        Returns a tuple (layer_number, dimension, l0_value) for sorting.
        Uses 999999 as default for missing values to put them at the end.
        """
        # Extract layer number
        layer_match = re.search(r'layer[_/](\d+)', path, re.IGNORECASE)
        layer_num = int(layer_match.group(1)) if layer_match else 999999

        # Extract dimension/width (e.g., width_16k -> 16000, width_65k -> 65000)
        width_match = re.search(r'width[_/](\d+)k', path, re.IGNORECASE)
        dimension = int(width_match.group(1)) * 1000 if width_match else 999999

        # Extract L0 value (e.g., average_l0_139, canonical_l0_71)
        l0_match = re.search(r'_l0[_/](\d+)', path, re.IGNORECASE)
        l0_value = int(l0_match.group(1)) if l0_match else 999999

        return (layer_num, dimension, l0_value)

    @staticmethod
    def _is_sae_file(file_path: str) -> bool:
        """Check if a file path looks like an SAE file."""
        path = Path(file_path)
        name = path.name.lower()

        # Check for Gemma Scope params.npz format
        if name == "params.npz":
            return True

        # Check for SAE-specific files
        sae_indicators = [
            "sae_weights",
            "cfg.json",
            "sparsity",
            "decoder",
            "encoder",
            "W_enc",
            "W_dec",
            "b_enc",
            "b_dec",
        ]

        for indicator in sae_indicators:
            if indicator.lower() in name:
                return True

        # Check for safetensors in SAE-looking directories
        if path.suffix == ".safetensors":
            parent = str(path.parent).lower()
            if any(x in parent for x in ["layer", "width", "canonical", "res", "mlp", "attn"]):
                return True

        # Check for npz files in Gemma Scope style directories
        if path.suffix == ".npz":
            parent = str(path.parent).lower()
            if any(x in parent for x in ["layer", "width", "average_l0"]):
                return True

        return False

    @staticmethod
    def _detect_model_name(repo_id: str) -> Optional[str]:
        """Try to detect the target model name from repo ID."""
        repo_lower = repo_id.lower()

        for pattern, model_name in HuggingFaceSAEService.MODEL_NAME_PATTERNS.items():
            if pattern in repo_lower:
                return model_name

        return None

    @staticmethod
    async def download_sae(
        repo_id: str,
        filepath: str,
        local_dir: Path,
        revision: Optional[str] = None,
        access_token: Optional[str] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Download SAE files from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            filepath: Path within repository (can be file or directory)
            local_dir: Local directory to download to
            revision: Git revision/branch (defaults to main)
            access_token: Optional access token
            progress_callback: Optional callback(progress_percent, bytes_downloaded, total_bytes)

        Returns:
            Dict with download info including local_path, file_size, metadata
        """
        # Use provided token, or settings token, but treat empty/whitespace/"None" as None
        token = access_token or settings.hf_token
        if not token or not token.strip() or token.strip().lower() == "none":
            token = None
        local_dir.mkdir(parents=True, exist_ok=True)

        def _download() -> Dict[str, Any]:
            try:
                # Try to determine if filepath is a directory or file
                files = list_repo_files(repo_id=repo_id, token=token, revision=revision)
                matching_files = [f for f in files if f.startswith(filepath)]

                if len(matching_files) == 0:
                    raise EntryNotFoundError(f"Path not found: {filepath}")

                elif len(matching_files) == 1 and matching_files[0] == filepath:
                    # Single file download
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filepath,
                        local_dir=str(local_dir),
                        revision=revision,
                        token=token,
                        local_dir_use_symlinks=False
                    )
                    file_size = os.path.getsize(local_path)

                    return {
                        "local_path": local_path,
                        "file_size_bytes": file_size,
                        "files_downloaded": [filepath],
                        "is_directory": False
                    }

                else:
                    # Directory download (multiple files)
                    # Filter to files within the requested path
                    allow_patterns = [f"{filepath}/*"] if not filepath.endswith("*") else [filepath]

                    local_path = snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(local_dir),
                        revision=revision,
                        token=token,
                        allow_patterns=allow_patterns,
                        local_dir_use_symlinks=False
                    )

                    # Calculate total size
                    total_size = 0
                    downloaded_files = []
                    for f in matching_files:
                        file_path = local_dir / f
                        if file_path.exists():
                            total_size += file_path.stat().st_size
                            downloaded_files.append(f)

                    return {
                        "local_path": str(local_dir / filepath),
                        "file_size_bytes": total_size,
                        "files_downloaded": downloaded_files,
                        "is_directory": True
                    }

            except EntryNotFoundError:
                raise ValueError(f"Path not found in repository: {filepath}")
            except RepositoryNotFoundError:
                raise ValueError(f"Repository not found: {repo_id}")
            except Exception as e:
                logger.error(f"Error downloading SAE: {e}")
                raise

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _download)

    @staticmethod
    async def upload_sae(
        local_path: Path,
        repo_id: str,
        filepath: str,
        access_token: str,
        create_repo: bool = False,
        private: bool = False,
        commit_message: str = "Upload SAE"
    ) -> Dict[str, Any]:
        """
        Upload SAE files to HuggingFace.

        Args:
            local_path: Local path to SAE files
            repo_id: Target HuggingFace repository ID
            filepath: Target path within repository
            access_token: HuggingFace access token with write permissions
            create_repo: Create repository if it doesn't exist
            private: Make repository private (only if creating)
            commit_message: Commit message for the upload

        Returns:
            Dict with upload info including url and commit_hash
        """
        def _upload() -> Dict[str, Any]:
            api = HfApi(token=access_token)

            # Create repo if requested
            if create_repo:
                try:
                    api.create_repo(
                        repo_id=repo_id,
                        repo_type="model",
                        private=private,
                        exist_ok=True
                    )
                except Exception as e:
                    logger.warning(f"Error creating repo (may already exist): {e}")

            # Upload files
            if local_path.is_dir():
                # Upload directory
                api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    path_in_repo=filepath,
                    commit_message=commit_message
                )
            else:
                # Upload single file
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=f"{filepath}/{local_path.name}" if filepath else local_path.name,
                    repo_id=repo_id,
                    commit_message=commit_message
                )

            # Get the URL
            url = f"https://huggingface.co/{repo_id}/tree/main/{filepath}"

            return {
                "repo_id": repo_id,
                "filepath": filepath,
                "url": url,
                "commit_hash": None  # Would need to get from API response
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _upload)

    @staticmethod
    async def check_token_permissions(access_token: str) -> Dict[str, bool]:
        """
        Check what permissions an access token has.

        Args:
            access_token: HuggingFace access token to check

        Returns:
            Dict with permission flags (read, write, etc.)
        """
        def _check() -> Dict[str, bool]:
            api = HfApi(token=access_token)

            try:
                # Get user info - if this works, token is valid
                whoami = api.whoami()
                return {
                    "valid": True,
                    "username": whoami.get("name", "unknown"),
                    "read": True,
                    "write": True  # Assume write if token is valid (can't easily check)
                }
            except Exception as e:
                logger.warning(f"Token validation failed: {e}")
                return {
                    "valid": False,
                    "username": None,
                    "read": False,
                    "write": False
                }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, _check)

    @staticmethod
    def get_sae_storage_path(sae_id: str) -> Path:
        """
        Get the local storage path for an SAE.

        Args:
            sae_id: SAE identifier

        Returns:
            Path to SAE storage directory
        """
        return settings.data_dir / "saes" / sae_id

    @staticmethod
    def ensure_sae_storage_dir() -> Path:
        """
        Ensure the SAE storage directory exists.

        Returns:
            Path to SAE storage directory
        """
        sae_dir = settings.data_dir / "saes"
        sae_dir.mkdir(parents=True, exist_ok=True)
        return sae_dir
