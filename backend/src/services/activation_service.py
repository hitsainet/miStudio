"""
Service for extracting and managing model activations.

This service handles the orchestration of activation extraction from transformer models,
including dataset loading, hook registration, batch processing, and statistics calculation.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime

import torch
import numpy as np
from datasets import load_from_disk

from ..ml.forward_hooks import HookManager, HookType
from ..ml.model_loader import get_quantization_config, load_model_from_hf
from ..models.model import QuantizationFormat
from ..core.config import settings
from . import activation_mask

logger = logging.getLogger(__name__)

# Ids this system generates: `extr_20260726_174056_sae_sae_d1a4_002`. No
# separators, no dots — so a traversal cannot be a valid id by construction.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,255}")


def _available_memory_bytes() -> Optional[int]:
    """Memory this process can still use, or None when it cannot be read.

    The smaller of the kernel's MemAvailable and the cgroup v2 headroom. Inside
    a container /proc/meminfo reports the HOST, so a pod with a memory limit
    would otherwise size its thread pool against memory it may not touch.
    """
    available: Optional[int] = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError):
        available = None

    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            headroom = max(0, int(limit) - used)
            available = headroom if available is None else min(available, headroom)
    except (OSError, ValueError):
        pass

    return available


class ActivationExtractionError(Exception):
    """Exception raised during activation extraction."""
    pass


#: The sequence length the old hardcoded cap assumed. Used only to derive a
#: default token budget, so that removing the cap keeps the SAME memory envelope
#: at 512 and shrinks the micro-batch proportionally beyond it.
REFERENCE_SEQ_LEN = 512

#: Guards the once-per-process budget warning below.
_BUDGET_WARNED = set()


def _dataset_looks_packed(dataset, probe: int = 32) -> bool:
    """Heuristic: are these rows packed blocks rather than documents?

    Packing produces rows that are entirely real tokens except the final block,
    so an all-ones attention mask across a sample of rows is the signature.
    A heuristic rather than a lookup because the extraction service is handed a
    PATH, not a tokenization row — plumbing the flag through every caller is a
    larger change than this warning is worth, and being wrong here costs only a
    spurious log line.
    """
    try:
        if "attention_mask" not in dataset.column_names:
            return False
        n = min(probe, len(dataset))
        if n == 0:
            return False
        masks = dataset.select(range(n))["attention_mask"]
        return all(all(m) for m in masks)
    except Exception:  # noqa: BLE001 - a heuristic must never break extraction
        return False


def micro_batch_size_for_length(
    micro_batch_size: int,
    seq_len: int,
    token_budget: Optional[int] = None,
) -> int:
    """How many sequences of `seq_len` may share one forward pass.

    Activation memory scales with rows x seq_len, so a fixed row count means a
    2048-token window costs 4x what a 512-token one did. The previous code
    "solved" that by truncating every sequence to 512 and never saying so. This
    keeps the token budget fixed instead, which is the quantity VRAM actually
    tracks, and lets the window be whatever the tokenization says.

    Never returns 0 — a single sequence always gets a forward pass, because
    refusing to process a long document is worse than a slow one.
    """
    if seq_len <= 0:
        return max(1, micro_batch_size)
    if token_budget is None:
        token_budget = max(1, micro_batch_size) * REFERENCE_SEQ_LEN

    # ATTENTION MEMORY IS THE OTHER TERM, and the budget alone does not see it.
    #
    # The budget models the captured tensor (rows x seq_len). Under an eager
    # attention path the peak is rows x heads x seq_len^2, so at a CONSTANT
    # rows x seq_len the true peak still grows LINEARLY in seq_len — 16x at 8192
    # versus 512. The old hardcoded cap hid this by making long windows
    # impossible; removing it means the budget has to shrink faster than
    # tokens-per-forward once sequences get long.
    #
    # Scaled against the reference length rather than capped, so the envelope at
    # 512 is unchanged and 8192 gets the 16x reduction its attention cost
    # implies.
    length_penalty = max(1, seq_len // REFERENCE_SEQ_LEN)
    effective_budget = max(1, token_budget // length_penalty)
    rows = max(1, min(micro_batch_size, effective_budget // seq_len))

    # THE FLOOR AT 1 MEANS THIS IS NOT A BOUND FOR A LONG ENOUGH SEQUENCE.
    #
    # At the schema maximum of 8192 with the default budget (8 x 512 = 4096),
    # this returns 1 and still spends 8192 tokens in one forward — twice the
    # budget. Refusing instead would mean refusing to process a long document,
    # which is worse. So it is reported, not hidden.
    #
    # A second, larger caveat: the budget models the CAPTURED tensor
    # (rows x seq_len). Peak attention memory under an eager path is
    # rows x heads x seq_len^2, which at constant rows x seq_len grows LINEARLY
    # in seq_len — so at 8192 the true peak is understated by roughly another
    # 16x versus 512. The old hardcoded cap hid this by making it impossible.
    if rows * seq_len > effective_budget and not _BUDGET_WARNED:
        # Once per process. The condition is CONSTANT across a run, and this is
        # called once per batch — at max_samples 2x10^5 and batch_size 8 that is
        # ~25,000 identical WARNING lines per extraction.
        _BUDGET_WARNED.add(True)
        logger.warning(
            "A single %d-token sequence exceeds the %d-token budget; processing "
            "it anyway at micro-batch 1. Peak attention memory also grows with "
            "seq_len beyond what this budget models.",
            seq_len, effective_budget,
        )
    return rows


class ActivationService:
    """
    Service for extracting activations from transformer models.

    This service coordinates the extraction process including:
    - Loading models and datasets
    - Registering forward hooks
    - Running batched inference
    - Saving activations to disk
    - Computing statistics
    """

    def __init__(self):
        """Initialize the ActivationService."""
        self.activations_dir = settings.data_dir / "activations"
        self.activations_dir.mkdir(parents=True, exist_ok=True)

    def _log_gpu_memory(self, stage: str, gpu_id: int = 0) -> None:
        """
        Log current GPU memory usage.

        Args:
            stage: Description of the current stage (e.g., "before_load", "after_extraction")
            gpu_id: GPU device ID to check memory for
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)    # GB
            logger.info(f"[GPU {gpu_id} Memory - {stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    #: Reserved GB left on the card that still counts as a successful release.
    #: A CUDA context plus allocator bookkeeping is a few hundred MB; anything
    #: approaching a model's footprint is a pool that never came back.
    RESERVED_FLOOR_GB = 1.0

    @staticmethod
    def _gpu_memory(gpu_id: int = 0) -> dict:
        """Allocated and reserved GB, or zeros when there is no CUDA device."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0}
        return {
            "allocated": torch.cuda.memory_allocated(gpu_id) / (1024 ** 3),
            "reserved": torch.cuda.memory_reserved(gpu_id) / (1024 ** 3),
        }

    def _release_gpu_memory(self, gpu_id: int = 0) -> dict:
        """Hand the allocator's pool back when there is no model object to drop.

        Used on the paths where a model reference was never obtained -- a load
        that raised, or a job whose model was already cleaned. Collecting and
        emptying the cache costs milliseconds and is the difference between the
        next extraction starting on a clear card or on a full one.
        """
        import gc

        before = self._gpu_memory(gpu_id)

        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(gpu_id)
            except Exception as e:                      # pragma: no cover
                logger.warning(f"Could not empty the CUDA cache on {gpu_id}: {e}")

        return self._report_cleanup(gpu_id, before)

    def _report_cleanup(self, gpu_id: int, before: dict) -> dict:
        """Compare the card before and after, and say which one happened."""
        after = self._gpu_memory(gpu_id)
        released = before["reserved"] - after["reserved"]

        if after["reserved"] > self.RESERVED_FLOOR_GB:
            logger.warning(
                f"GPU {gpu_id} did not give the memory back: reserved "
                f"{before['reserved']:.2f} -> {after['reserved']:.2f} GB "
                f"(released {released:.2f} GB) with {after['allocated']:.2f} GB "
                "still allocated. That pool is unavailable to anything else on "
                "the card until this process exits."
            )
        else:
            logger.info(
                f"Model unloaded from GPU {gpu_id}: released "
                f"{released:.2f} GB, {after['reserved']:.2f} GB reserved"
            )

        return {"before": before, "after": after, "released": released}

    def _cleanup_model(self, model: torch.nn.Module, gpu_id: int = 0) -> dict:
        """
        Explicitly clean up model from GPU memory.

        This ensures GPU memory is freed immediately rather than waiting
        for Python's garbage collector. Critical for sequential extraction jobs.

        Args:
            model: PyTorch model to clean up
            gpu_id: GPU device ID that the model was loaded on
        """
        import gc

        before = self._gpu_memory(gpu_id)

        try:
            # Log memory before cleanup
            self._log_gpu_memory("before_cleanup", gpu_id)

            # Synchronize CUDA to ensure all operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize(gpu_id)

            # AGGRESSIVE CLEANUP: Delete all model parameters and buffers explicitly
            # This is more reliable than model.cpu() when using device_map
            try:
                # Clear all parameters
                for param in model.parameters():
                    param.data = torch.empty(0)
                    if param.grad is not None:
                        param.grad = None

                # Clear all buffers
                for buffer in model.buffers():
                    buffer.data = torch.empty(0)
            except Exception as e:
                logger.warning(f"Error clearing model parameters/buffers: {e}")

            # Try to move model to CPU (may not work with device_map, but worth trying)
            try:
                model.cpu()
            except Exception as e:
                logger.warning(f"model.cpu() failed (expected with device_map): {e}")

            # Delete all module attributes that might hold tensor references
            try:
                for name, child in list(model.named_children()):
                    delattr(model, name)
            except Exception as e:
                logger.warning(f"Error deleting model children: {e}")

            # Delete model reference
            del model

            # Multiple rounds of garbage collection to ensure cleanup
            for _ in range(3):
                gc.collect()

            # Empty CUDA cache on the specific GPU
            if torch.cuda.is_available():
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    # Also synchronize again after cache clear
                    torch.cuda.synchronize(gpu_id)

            # Log memory after cleanup
            self._log_gpu_memory("after_cleanup", gpu_id)

            # Report what actually happened, not that the code ran.
            #
            # This used to log success unconditionally. On 2026-08-25 three
            # consecutive extractions logged "Model cleaned up" with 5.38 GB,
            # 9.32 GB and 7.16 GB still allocated, and a fourth left a 6.99 GB
            # reserved pool that never came back -- nvidia-smi showed 7.3 GB
            # held by a worker with no live tensors, nine hours later. A claim
            # nothing measures cannot surface its own failure.
            return self._report_cleanup(gpu_id, before)

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
            # Still try to clear cache even if other cleanup failed
            if torch.cuda.is_available():
                try:
                    gc.collect()
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # A failed cleanup is precisely when the card's state matters, so
            # measure here too rather than returning None and saying nothing.
            return self._report_cleanup(gpu_id, before)

    def extract_activations(
        self,
        model_id: str,
        model_path: str,
        architecture: str,
        quantization: QuantizationFormat,
        dataset_path: str,
        layer_indices: List[int],
        hook_types: List[str],
        max_samples: int,
        batch_size: int = 8,
        micro_batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        micro_batch_token_budget: Optional[int] = None,
        extraction_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        gpu_id: int = 0,
        statistics_progress_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Extract activations from a model using a dataset.

        Args:
            model_id: Model database ID
            model_path: Path to model files
            architecture: Model architecture (llama, gpt2, etc.)
            quantization: Quantization format
            dataset_path: Path to tokenized dataset
            layer_indices: List of layer indices to extract from
            hook_types: List of hook types ('residual', 'mlp', 'attention')
            max_samples: Maximum number of samples to process
            batch_size: Batch size for processing
            micro_batch_size: GPU micro-batch size for memory efficiency (defaults to batch_size)
            extraction_id: Optional extraction ID (generated if not provided)
            progress_callback: Optional callback function(samples_processed, total_samples)
            gpu_id: GPU device ID to use for extraction (default: 0)
            statistics_progress_callback: Optional callback
                function(layer_index, n_layers, chunks_done, n_chunks), called
                on this thread throughout the statistics phase. It is the
                phase's heartbeat and cancellation checkpoint; without it the
                row goes silent after the last sample.

        Returns:
            Dictionary with extraction metadata including output_path and statistics

        Raises:
            ActivationExtractionError: If extraction fails
        """
        if extraction_id is None:
            extraction_id = f"ext_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Default micro_batch_size to batch_size if not specified
        if micro_batch_size is None:
            micro_batch_size = batch_size
            logger.info(f"micro_batch_size not specified, defaulting to batch_size={batch_size}")

        logger.info(
            f"Starting activation extraction: {extraction_id} "
            f"(model={model_id}, layers={layer_indices}, hooks={hook_types}, "
            f"max_samples={max_samples}, batch_size={batch_size}, micro_batch_size={micro_batch_size})"
        )

        # Create output directory
        output_dir = self._extraction_dir(extraction_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = None  # Initialize to None for cleanup in finally block
        try:
            # Log GPU memory before loading model
            self._log_gpu_memory("before_load", gpu_id)

            # Load model
            logger.info(f"Loading model from {model_path} to GPU {gpu_id}")
            model, tokenizer = self._load_model(model_path, quantization, gpu_id=gpu_id)
            model.eval()  # Set to evaluation mode

            # Log GPU memory after loading model
            self._log_gpu_memory("after_load", gpu_id)

            # Load dataset
            logger.info(f"Loading dataset from {dataset_path}")
            dataset = self._load_dataset(dataset_path, max_samples)

            # PACKED ROWS ARE NOT DOCUMENTS.
            #
            # `pack_token_blocks`' own docstring says any consumer assuming
            # row == document must be checked before packing is enabled. This is
            # one: `max_samples` means BLOCKS here, not documents, so 10,000
            # against a packed Bloomberg corpus is ~30x the real tokens and
            # compute of 10,000 unpacked rows, under an unchanged label. Said
            # out loud rather than left for someone to discover from a bill.
            if _dataset_looks_packed(dataset):
                logger.warning(
                    "This tokenization looks PACKED (every row is full width). "
                    "max_samples=%s therefore selects %s BLOCKS, not documents — "
                    "each block is several concatenated documents, so this is "
                    "substantially more text than the same number of unpacked "
                    "rows.", max_samples, max_samples,
                )

            # Convert hook type strings to enums
            hook_type_enums = [HookType(ht) for ht in hook_types]

            # Extract activations
            logger.info(f"Extracting activations with hooks: {hook_types}")
            created_at_timestamp = datetime.now().isoformat()
            activations = self._run_extraction(
                model,
                tokenizer,
                dataset,
                architecture,
                layer_indices,
                hook_type_enums,
                batch_size,
                micro_batch_size,
                max_seq_length,
                micro_batch_token_budget,
                progress_callback,
                output_dir=output_dir,
                extraction_id=extraction_id,
                model_id=model_id,
                quantization=quantization,
                dataset_path=dataset_path,
                created_at=created_at_timestamp,
            )

            # Log GPU memory after extraction
            self._log_gpu_memory("after_extraction", gpu_id)

            # CRITICAL: Clean up GPU memory immediately after extraction completes
            # Model and hooks are no longer needed for saving/statistics phases
            # This frees ~8-10 GB of GPU memory that would otherwise sit idle
            logger.info(f"Cleaning up GPU memory after extraction (model no longer needed)")
            if model is not None:
                self._cleanup_model(model, gpu_id)
                model = None  # Mark as cleaned up to avoid double cleanup in finally block

            # Save activations to disk
            logger.info(f"Saving activations to {output_dir}")
            saved_files = self._save_activations(output_dir, activations)

            # Calculate statistics
            logger.info("Calculating activation statistics")
            statistics = self._calculate_statistics(
                activations, on_progress=statistics_progress_callback
            )

            # Save metadata
            metadata = {
                "extraction_id": extraction_id,
                "model_id": model_id,
                "architecture": architecture,
                "quantization": quantization.value,
                "dataset_path": dataset_path,
                "layer_indices": layer_indices,
                "hook_types": hook_types,
                "max_samples": max_samples,
                "batch_size": batch_size,
                "num_samples_processed": len(dataset),
                "status": "completed",
                "created_at": created_at_timestamp,
                "completed_at": datetime.now().isoformat(),
                # The width of what was actually saved. Mask recovery needs it:
                # `dataset_path` alone is not an identity, and the path used to
                # omit `max_length`, so a 512 and a 2048 tokenization of the same
                # source occupied one directory. Without this, a directory that
                # was later overwritten still row-matches and its mask is trimmed
                # and returned as authoritative — wrong by CONTENT, not by row.
                "seq_len": (
                    int(next(iter(activations.values())).shape[1])
                    if activations else None
                ),
                "saved_files": saved_files,
                "statistics": statistics,
                # WHERE in each layer these activations were read. Extractions
                # before 2026-09-12 recorded "residual" and meant the output of
                # a normalisation module (LFM2's ffn_norm, Llama's
                # post_attention_layernorm) — a pre-MLP, RMS-normalised signal.
                # miLLM applies SAEs at the decoder layer's OUTPUT. An SAE is
                # only valid at the point it was trained on, so the point is
                # recorded rather than implied by a hook-type name.
                "hook_point": "resid_post",
            }

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Extraction complete: {extraction_id}")

            return {
                "extraction_id": extraction_id,
                "output_path": str(output_dir),
                "num_samples": len(dataset),
                "saved_files": saved_files,
                "statistics": statistics,
                "metadata_path": str(metadata_path),
            }

        except Exception as e:
            logger.exception(f"Activation extraction failed: {e}")

            # An OOM's traceback pins every frame of the load that failed, and
            # those frames hold exactly the tensors that need releasing -- so
            # carrying the chain up the stack keeps several GB alive past the
            # cleanup below. The full trace is already in the log above, and
            # the message survives, which is what classify_extraction_error
            # matches on to schedule the smaller-batch retry.
            if "out of memory" in str(e).lower():
                e.__traceback__ = None
                raise ActivationExtractionError(
                    f"Extraction failed: {str(e)}"
                ) from None

            raise ActivationExtractionError(f"Extraction failed: {str(e)}") from e

        finally:
            # CRITICAL: Always clean up GPU memory, even if extraction failed
            if model is not None:
                logger.info(f"Cleaning up model for extraction {extraction_id} on GPU {gpu_id}")
                self._cleanup_model(model, gpu_id)
            else:
                # A load that died part-way still left its weights on the card.
                #
                # `model` is bound only when _load_model RETURNS, so an OOM
                # during loading arrives here with several GB resident and the
                # name still None. This branch used to log "nothing to clean
                # up" and return, which is how a failed extraction kept
                # everything it had allocated. The next one then OOMs against a
                # nearly full card -- 2026-08-25, a 26-second failure reporting
                # "total capacity of 23.56 GiB of which 21.75 MiB is free".
                logger.info(
                    f"No model handle for extraction {extraction_id}; releasing "
                    f"whatever a failed load left on GPU {gpu_id}"
                )
                self._release_gpu_memory(gpu_id)

    def _load_model(
        self,
        model_path: str,
        quantization: QuantizationFormat,
        gpu_id: int = 0
    ) -> tuple[torch.nn.Module, Any]:
        """
        Load model from disk, handling HuggingFace cache structure.

        Args:
            model_path: Path to model files (may contain HF cache structure)
            quantization: Quantization format
            gpu_id: GPU device ID to use (default: 0)

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import glob

        model_path_obj = Path(model_path)

        # Check if model_path uses HuggingFace cache structure
        # Look for: model_path/models--{org}--{model}/snapshots/{hash}/
        models_dirs = list(model_path_obj.glob("models--*"))

        if models_dirs:
            # HuggingFace cache structure detected
            logger.info(f"Detected HuggingFace cache structure at {model_path}")
            models_dir = models_dirs[0]  # Should only be one

            # Find the snapshot directory (there should be exactly one)
            snapshot_dirs = list((models_dir / "snapshots").glob("*"))
            if not snapshot_dirs:
                raise ActivationExtractionError(
                    f"No snapshots found in HuggingFace cache at {models_dir}/snapshots"
                )

            actual_model_path = str(snapshot_dirs[0])
            logger.info(f"Using snapshot path: {actual_model_path}")
        else:
            # Direct model files (flat structure)
            actual_model_path = model_path
            logger.info(f"Using direct model path: {actual_model_path}")

        # Check CUDA availability
        if not torch.cuda.is_available():
            raise ActivationExtractionError("CUDA is not available. GPU is required for activation extraction.")

        # Validate GPU ID
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            raise ActivationExtractionError(
                f"GPU {gpu_id} not available. System has {num_gpus} GPU(s) (indices 0-{num_gpus-1})."
            )

        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"Loading model to device: {device} (CUDA device: {torch.cuda.get_device_name(gpu_id)})")

        # APPLY THE REQUESTED QUANTIZATION.
        #
        # This function took `quantization`, documented it, logged it and wrote
        # it into the extraction metadata — and never passed it to
        # `from_pretrained`. Every extraction loaded fp16 regardless. A Q4
        # gemma-4-12B is 6 GB on disk and 24 GB in fp16, so it did not fit on a
        # 23.56 GiB card: the OOM reported 2026-08-23 died placing the last
        # fragment of the model, before a single activation was read.
        #
        # `get_quantization_config` already returns exactly what
        # `from_pretrained` wants, and had precisely ONE caller — in
        # `ml/model_loader.py`, the OTHER load path. Two paths, one of them
        # honouring the setting.
        #
        # `torch_dtype` stays: with `load_in_4bit` it sets the dtype of the
        # modules bitsandbytes does NOT quantize (norms, embeddings, the LM
        # head), and it is what `bnb_4bit_compute_dtype` matches. Hidden states
        # — what the hooks capture — are fp16 either way, which is the property
        # the SAE encode path depends on.
        quantization_config = get_quantization_config(quantization)

        # IMPORTANT: Use explicit cuda:N instead of device_map="auto" to force GPU placement
        # device_map="auto" with accelerate can offload to CPU if it thinks there's not enough memory
        model = AutoModelForCausalLM.from_pretrained(
            actual_model_path,
            device_map={"": device},  # Force all layers to specified GPU
            torch_dtype=torch.float16,
            quantization_config=quantization_config,  # None for FP16/FP32
            low_cpu_mem_usage=True,  # Minimize CPU memory during loading
        )

        logger.info(
            "Model loaded successfully. Device: %s, dtype: %s, quantization: %s "
            "(config applied: %s)",
            model.device, model.dtype, quantization.value,
            quantization_config is not None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path)

        return model, tokenizer

    def _load_dataset(self, dataset_path: str, max_samples: int) -> Any:
        """
        Load dataset from disk.

        Args:
            dataset_path: Path to tokenized dataset
            max_samples: Maximum number of samples to load

        Returns:
            Dataset object
        """
        dataset = load_from_disk(dataset_path)

        # Limit to max_samples
        if max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset

    def _write_incremental_metadata(
        self,
        output_dir: Path,
        extraction_id: str,
        model_id: str,
        architecture: str,
        quantization: QuantizationFormat,
        dataset_path: str,
        layer_indices: List[int],
        hook_types: List[str],
        max_samples: int,
        batch_size: int,
        num_samples_processed: int,
        created_at: str,
    ) -> None:
        """
        Write incremental metadata file during extraction.

        Uses atomic write (temp file + rename) to ensure consistency.
        This allows inspection of partial results if extraction crashes.

        Args:
            output_dir: Output directory
            extraction_id: Extraction ID
            model_id: Model ID
            architecture: Model architecture
            quantization: Quantization format
            dataset_path: Dataset path
            layer_indices: Layer indices
            hook_types: Hook types
            max_samples: Maximum samples
            batch_size: Batch size
            num_samples_processed: Number of samples processed so far
            created_at: Creation timestamp
        """
        import tempfile
        import os

        metadata = {
            "extraction_id": extraction_id,
            "model_id": model_id,
            "architecture": architecture,
            "quantization": quantization.value,
            "dataset_path": dataset_path,
            "layer_indices": layer_indices,
            "hook_types": hook_types,
            "max_samples": max_samples,
            "batch_size": batch_size,
            "num_samples_processed": num_samples_processed,
            "status": "in_progress",
            "created_at": created_at,
            "last_updated": datetime.now().isoformat(),
        }

        # Atomic write: write to temp file, then rename
        metadata_path = output_dir / "metadata.json"
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=output_dir,
            delete=False,
            suffix='.tmp'
        ) as temp_file:
            json.dump(metadata, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_name = temp_file.name

        # Rename temp file to final name (atomic operation)
        os.replace(temp_name, metadata_path)
        logger.debug(f"Updated incremental metadata: {num_samples_processed}/{max_samples} samples")

    def _run_extraction(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        dataset: Any,
        architecture: str,
        layer_indices: List[int],
        hook_types: List[HookType],
        batch_size: int,
        micro_batch_size: int,
        max_seq_length: Optional[int] = None,
        micro_batch_token_budget: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        output_dir: Optional[Path] = None,
        extraction_id: Optional[str] = None,
        model_id: Optional[str] = None,
        quantization: Optional[QuantizationFormat] = None,
        dataset_path: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run activation extraction with hooks using micro-batched inference.

        This method uses a two-level batching strategy:
        1. Outer loop: Process dataset in chunks of batch_size (for progress reporting)
        2. Inner loop: Split each batch into micro-batches of micro_batch_size (for GPU memory)

        Args:
            model: PyTorch model
            tokenizer: Model tokenizer
            dataset: Dataset to process
            architecture: Model architecture
            layer_indices: Layers to hook
            hook_types: Types of hooks to register
            batch_size: Logical batch size for progress tracking (1, 8, 16, 32, 64, 128, 256, 512)
            micro_batch_size: GPU micro-batch size for memory efficiency (must be <= batch_size)
            progress_callback: Optional callback function(samples_processed, total_samples)
            output_dir: Optional output directory for incremental metadata
            extraction_id: Optional extraction ID for metadata
            model_id: Optional model ID for metadata
            quantization: Optional quantization format for metadata
            dataset_path: Optional dataset path for metadata
            created_at: Optional creation timestamp for metadata

        Returns:
            Dictionary mapping layer names to activation arrays
        """
        # Create hook manager
        with HookManager(model) as hook_manager:
            # Register hooks
            hook_manager.register_hooks(layer_indices, hook_types, architecture)

            # Get model's vocabulary size for validation.
            #
            # NOT `model.config.vocab_size`: unified and multimodal configs keep
            # the text fields on a sub-config, and this line killed two real
            # extraction jobs on gemma-4-12B-it with
            # "'Gemma4UnifiedConfig' object has no attribute 'vocab_size'".
            from ..ml.layer_discovery import resolve_vocab_size

            vocab_size = resolve_vocab_size(model)
            if vocab_size is None:
                # Fail loudly rather than validating token ids against a guess.
                raise ValueError(
                    "Could not determine this model's vocabulary size from its "
                    "config or its embedding table, so token ids cannot be "
                    "range-checked. Report the model architecture."
                )
            logger.info(f"Model vocabulary size: {vocab_size}")
            logger.info(f"Processing {len(dataset)} samples with batch_size={batch_size}")

            # Get pad token ID (use eos_token if pad_token not available)
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
                logger.info(f"Using eos_token_id {pad_token_id} as pad_token_id")

            samples_processed = 0

            # Track accumulated activation file paths for each layer
            accumulated_files = {}  # layer_name -> list of temp file paths
            #: Real (pre-padding) token count per sample, in extraction order.
            #: Written out as `attention_mask.npy` so training can exclude PAD
            #: positions without having to re-derive them from the tokenized
            #: dataset. See services/activation_mask.py.
            sample_token_lengths = []
            #: Samples whose real tokens are NOT a right-hand prefix. The
            #: sidecar cannot describe those with a length alone.
            non_prefix_masks = []

            # Process dataset in batches
            with torch.no_grad():
                for batch_start in range(0, len(dataset), batch_size):
                    batch_end = min(batch_start + batch_size, len(dataset))
                    batch_samples = dataset[batch_start:batch_end]

                    # Extract input_ids from batch
                    # NOTE: dataset[start:end] returns a dict with keys as column names
                    # batch_samples = {"input_ids": [[...], [...]], "attention_mask": [[...], [...]]}
                    batch_input_ids = []
                    #: The tokenization's OWN attention mask, when it has one.
                    #
                    # THIS IS THE CORRECTION FOR THE WORST DEFECT IN THIS ARC.
                    # `len(input_ids)` is NOT the real length: rows arrive from
                    # Arrow already padded to `max_length` (the schema default),
                    # so every length was `max_length` and the sidecar mask this
                    # code writes was ALL-TRUE on every extraction that can
                    # complete — a mask that is confidently wrong, which
                    # activation_mask.py's own docstring calls worse than none.
                    # The truth was one column away the whole time.
                    batch_source_masks = []

                    # Check if batch_samples is a dict (HuggingFace dataset format)
                    if isinstance(batch_samples, dict) and 'input_ids' in batch_samples:
                        # batch_samples['input_ids'] is a list of token ID lists
                        for input_ids in batch_samples['input_ids']:
                            # Convert to list if needed
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            elif hasattr(input_ids, 'tolist'):
                                input_ids = input_ids.tolist()

                            batch_input_ids.append(input_ids)

                        for mask in (batch_samples.get('attention_mask') or []):
                            if isinstance(mask, torch.Tensor) or hasattr(mask, 'tolist'):
                                mask = mask.tolist()
                            batch_source_masks.append(list(mask))
                    else:
                        # Fallback for other formats (single samples, non-HF datasets)
                        # Iterate over samples
                        if isinstance(batch_samples, list):
                            samples_to_iterate = batch_samples
                        else:
                            samples_to_iterate = [batch_samples]

                        for sample in samples_to_iterate:
                            if isinstance(sample, dict):
                                input_ids = sample.get("input_ids")
                            else:
                                input_ids = sample

                            # Convert to list if needed
                            if isinstance(input_ids, torch.Tensor):
                                input_ids = input_ids.tolist()
                            elif hasattr(input_ids, 'tolist'):
                                input_ids = input_ids.tolist()

                            batch_input_ids.append(input_ids)

                    # SEQUENCE LENGTH COMES FROM THE TOKENIZATION, NOT FROM HERE.
                    #
                    # This used to be `MAX_SEQ_LENGTH = 512`, hardcoded, "to
                    # prevent GPU OOM". It meant tokenizing at 1024 or 2048 and
                    # then extracting silently discarded everything past token
                    # 512 — reported only at logger.debug, so invisible. The
                    # operator got a 2048-token tokenization and 512 tokens of
                    # activations, with nothing to show the difference.
                    #
                    # Memory is bounded by the TOKEN BUDGET below instead, which
                    # shrinks the micro-batch as sequences lengthen. That is the
                    # quantity that actually drives VRAM; sequence length alone
                    # never was.
                    cleaned_batch_input_ids = []
                    cleaned_batch_masks = []
                    truncated_in_batch = 0
                    for idx, input_ids in enumerate(batch_input_ids):
                        # Convert to tensor for validation
                        ids_tensor = torch.tensor(input_ids)
                        source_mask = (
                            batch_source_masks[idx]
                            if idx < len(batch_source_masks) else None
                        )

                        # Truncate only if an explicit cap was configured.
                        if max_seq_length is not None and len(ids_tensor) > max_seq_length:
                            truncated_in_batch += 1
                            ids_tensor = ids_tensor[:max_seq_length]
                            if source_mask is not None:
                                # In LOCKSTEP, or the mask describes positions
                                # that are no longer there.
                                source_mask = source_mask[:max_seq_length]
                        cleaned_batch_masks.append(source_mask)

                        # Clamp to valid vocabulary range
                        max_token = ids_tensor.max().item()
                        min_token = ids_tensor.min().item()

                        if max_token >= vocab_size or min_token < 0:
                            if max_token >= vocab_size:
                                logger.warning(
                                    f"Sample {batch_start + idx} contains token ID {max_token} "
                                    f"(vocab_size={vocab_size}). Clamping to valid range."
                                )
                            if min_token < 0:
                                logger.warning(
                                    f"Sample {batch_start + idx} contains negative token ID {min_token}. "
                                    f"Clamping to valid range."
                                )
                            ids_tensor = torch.clamp(ids_tensor, 0, vocab_size - 1)

                        cleaned_batch_input_ids.append(ids_tensor.tolist())

                    if truncated_in_batch:
                        logger.warning(
                            "Truncated %d/%d samples in batch %d to max_seq_length=%d; "
                            "those tokens are not in the extraction",
                            truncated_in_batch, len(batch_input_ids), batch_start,
                            max_seq_length,
                        )

                    max_length = max(len(ids) for ids in cleaned_batch_input_ids)

                    # ONE implementation, shared with the on-the-fly training
                    # path and unit-tested. Writing this inline twice is how the
                    # second copy came to reproduce the exact defect the first
                    # one had just fixed.
                    (
                        padded_input_ids,
                        attention_masks,
                        batch_real_lengths,
                        batch_non_prefix,
                        missing_source_mask,
                    ) = activation_mask.build_padded_batch(
                        cleaned_batch_input_ids,
                        cleaned_batch_masks,
                        max_length,
                        pad_token_id,
                    )
                    sample_token_lengths.extend(batch_real_lengths)
                    non_prefix_masks.extend(batch_start + k for k in batch_non_prefix)

                    if missing_source_mask:
                        # NEVER SILENT. Without the column we cannot tell text
                        # from padding, and everything downstream would treat the
                        # whole row as real — the defect this arc exists to remove.
                        logger.warning(
                            "%d/%d samples in batch %d have no attention_mask "
                            "column; assuming every position is a real token. "
                            "Re-tokenize with return_attention_mask=True.",
                            missing_source_mask, len(cleaned_batch_input_ids),
                            batch_start,
                        )

                    # MICRO-BATCHING: split the batch for GPU memory efficiency.
                    micro_batch_activations = {}  # Accumulate across micro-batches

                    # The real VRAM driver is (rows x seq_len), not seq_len. A
                    # longer window therefore buys a proportionally smaller
                    # micro-batch rather than a silent truncation.
                    effective_micro_batch = micro_batch_size_for_length(
                        micro_batch_size, max_length, micro_batch_token_budget
                    )
                    if effective_micro_batch < micro_batch_size:
                        # The budget is None unless a caller sets it, and "%d"
                        # against None raises inside logging — so the one line
                        # telling the operator their batch shrank was replaced by
                        # a logging traceback, exactly when it mattered.
                        budget = (
                            micro_batch_token_budget
                            if micro_batch_token_budget is not None
                            else micro_batch_size * REFERENCE_SEQ_LEN
                        )
                        logger.info(
                            "Sequence length %d: micro-batch %d -> %d to stay within "
                            "the %d-token budget",
                            max_length, micro_batch_size, effective_micro_batch,
                            budget,
                        )

                    for micro_batch_start in range(0, len(padded_input_ids), effective_micro_batch):
                        micro_batch_end = min(micro_batch_start + effective_micro_batch, len(padded_input_ids))

                        # Get micro-batch slices
                        micro_input_ids = padded_input_ids[micro_batch_start:micro_batch_end]
                        micro_attention_masks = attention_masks[micro_batch_start:micro_batch_end]

                        logger.debug(
                            f"Processing micro-batch {micro_batch_start}-{micro_batch_end} "
                            f"of batch {batch_start}-{batch_end} "
                            f"(micro_batch_size={micro_batch_size})"
                        )

                        # Convert to tensors and move to device
                        input_ids_tensor = torch.tensor(micro_input_ids, dtype=torch.long).to(model.device)
                        attention_mask_tensor = torch.tensor(micro_attention_masks, dtype=torch.long).to(model.device)

                        # Run forward pass (hooks will capture activations for this micro-batch)
                        _ = model(input_ids_tensor, attention_mask=attention_mask_tensor)

                        # Get activations from this micro-batch
                        current_micro_activations = hook_manager.get_activations_as_numpy()

                        # Accumulate activations from this micro-batch
                        for layer_name, activation_array in current_micro_activations.items():
                            if layer_name not in micro_batch_activations:
                                micro_batch_activations[layer_name] = []
                            micro_batch_activations[layer_name].append(activation_array)

                        # Clear hooks to free GPU memory before next micro-batch
                        hook_manager.clear_activations()

                    # Concatenate all micro-batch activations into single batch
                    # This happens in CPU/RAM, not GPU
                    batch_activations = {}
                    for layer_name, activation_list in micro_batch_activations.items():
                        batch_activations[layer_name] = np.concatenate(activation_list, axis=0)

                    # CRITICAL FIX: Save batch activations to disk immediately to prevent memory accumulation

                    # Save each layer's batch to a temporary file
                    for layer_name, activation_array in batch_activations.items():
                        # Initialize list for this layer if first batch
                        if layer_name not in accumulated_files:
                            accumulated_files[layer_name] = []

                        # Save batch activation to temporary file
                        import tempfile
                        import os
                        # Create temp file and close it immediately before numpy writes
                        # This avoids numpy memory-mapping issues with open file handles
                        temp_file = tempfile.NamedTemporaryFile(
                            dir=output_dir,
                            delete=False,
                            suffix=f'_{layer_name}_batch{len(accumulated_files[layer_name])}.npy'
                        )
                        temp_file_path = temp_file.name
                        temp_file.close()  # Close the file handle before numpy writes

                        # Now numpy can write to the closed file without memory-mapping issues
                        np.save(temp_file_path, activation_array)

                        accumulated_files[layer_name].append(temp_file_path)
                        logger.debug(f"Saved batch activation for {layer_name}: {activation_array.shape}")

                    # Note: Activations already cleared after each micro-batch for memory efficiency

                    samples_processed = batch_end

                    # Log progress every 10 samples or every batch (whichever is more frequent)
                    if samples_processed % 10 == 0 or samples_processed == batch_end:
                        logger.info(f"Processed {samples_processed}/{len(dataset)} samples")

                        # Call progress callback if provided
                        if progress_callback:
                            try:
                                progress_callback(samples_processed, len(dataset))
                            except Exception as e:
                                logger.warning(f"Progress callback failed: {e}")

                    # Write incremental metadata every 50 samples
                    if samples_processed % 50 == 0 and output_dir and extraction_id and model_id and quantization and dataset_path and created_at:
                        try:
                            hook_type_strs = [ht.value for ht in hook_types]
                            self._write_incremental_metadata(
                                output_dir=output_dir,
                                extraction_id=extraction_id,
                                model_id=model_id,
                                architecture=architecture,
                                quantization=quantization,
                                dataset_path=dataset_path,
                                layer_indices=layer_indices,
                                hook_types=hook_type_strs,
                                max_samples=len(dataset),
                                batch_size=batch_size,
                                num_samples_processed=samples_processed,
                                created_at=created_at,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to write incremental metadata: {e}")

            # Concatenate all batch files for each layer to create final activations
            # Use memory-efficient chunked concatenation to avoid OOM during large extractions
            logger.info("Concatenating batch activations into final arrays (memory-efficient mode)")
            activations = {}
            for layer_name, temp_files in accumulated_files.items():
                if not temp_files:
                    continue

                logger.info(f"Processing {len(temp_files)} batch files for {layer_name}")

                # Strategy: Use numpy.memmap for zero-copy concatenation
                # 1. Determine final shape by loading first file
                # 2. Pre-allocate output file with final shape
                # 3. Copy batches directly into output file
                # 4. Clean up temp files as we go

                # Load first file to get shape and dtype
                first_array = np.load(temp_files[0])
                sample_shape = first_array.shape[1:]  # Shape without batch dimension
                dtype = first_array.dtype

                # Calculate total samples across all batches
                total_samples = 0
                batch_sizes = []
                for temp_file_path in temp_files:
                    arr = np.load(temp_file_path, mmap_mode='r')  # Memory-mapped read
                    batch_sizes.append(arr.shape[0])
                    total_samples += arr.shape[0]

                # Create output file path
                final_shape = (total_samples,) + sample_shape
                output_file = output_dir / f"{layer_name}_temp_concat.npy"

                logger.info(f"Creating memory-mapped output for {layer_name}: shape={final_shape}, dtype={dtype}")

                # Pre-allocate output array as memory-mapped file
                final_array = np.lib.format.open_memmap(
                    str(output_file),
                    mode='w+',
                    dtype=dtype,
                    shape=final_shape
                )

                # Copy batches into final array in chunks
                current_idx = 0
                for i, (temp_file_path, batch_size) in enumerate(zip(temp_files, batch_sizes)):
                    # Load batch as memory-mapped array (no memory allocation)
                    batch_array = np.load(temp_file_path, mmap_mode='r')

                    # Copy directly into output file
                    final_array[current_idx:current_idx + batch_size] = batch_array
                    current_idx += batch_size

                    # Force flush to disk every 10 batches to free memory
                    if i % 10 == 0:
                        final_array.flush()

                    # Immediately delete temp file to free disk space
                    try:
                        Path(temp_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

                # Final flush
                final_array.flush()

                # Load final array (still memory-mapped, not in RAM)
                activations[layer_name] = final_array

                logger.info(f"Concatenated {len(temp_files)} batches for {layer_name}: final shape={final_array.shape}")

                # Clean up any remaining temp files
                for temp_file_path in temp_files:
                    try:
                        if Path(temp_file_path).exists():
                            Path(temp_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

        logger.info(f"Extracted activations for {len(activations)} layers")

        if non_prefix_masks:
            logger.warning(
                "%d samples are NOT right-padded (e.g. index %s). The mask "
                "sidecar records a token COUNT, which attributes those tokens to "
                "the wrong positions. Training will fall back to the "
                "tokenization's own attention_mask, which is correct.",
                len(non_prefix_masks), non_prefix_masks[:3],
            )
        else:
            self._write_attention_mask(output_dir, activations, sample_token_lengths)

        return activations

    @staticmethod
    def _write_attention_mask(output_dir, activations, sample_token_lengths) -> None:
        """Record which saved positions are real tokens rather than padding.

        Without this, training cannot tell text from padding and samples both.
        Refuses rather than guesses whenever the recorded lengths do not line up
        with what was actually saved — a mask that is confidently wrong is worse
        than none, because the caller warns loudly on a missing one.
        """
        from .activation_mask import MASK_FILENAME

        if not sample_token_lengths or not activations:
            logger.warning(
                "No attention mask written for %s: nothing to describe", output_dir
            )
            return

        first = next(iter(activations.values()))
        if first.ndim != 3:
            logger.warning(
                "No attention mask written: activations are %d-D, expected 3", first.ndim
            )
            return

        num_samples, seq_len = first.shape[0], first.shape[1]
        if len(sample_token_lengths) != num_samples:
            logger.warning(
                "No attention mask written: recorded %d sample lengths but saved "
                "%d samples", len(sample_token_lengths), num_samples,
            )
            return

        lengths = np.asarray(sample_token_lengths, dtype=np.int64)
        mask = np.arange(seq_len)[None, :] < np.clip(lengths, 0, seq_len)[:, None]
        np.save(output_dir / MASK_FILENAME, mask)
        logger.info(
            "Wrote %s: %s/%s positions are real tokens (%.1f%%)",
            MASK_FILENAME, f"{int(mask.sum()):,}", f"{mask.size:,}",
            mask.sum() / max(mask.size, 1) * 100,
        )

    def _save_activations(
        self,
        output_dir: Path,
        activations: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Save activations to disk as .npy files.

        Handles both regular numpy arrays and memory-mapped arrays.
        For memory-mapped arrays created during concatenation, renames the file
        instead of copying to avoid memory overhead.

        Args:
            output_dir: Directory to save files
            activations: Dictionary of activations

        Returns:
            List of saved file paths
        """
        saved_files = []

        for layer_name, activation_array in activations.items():
            # Create final filename
            filename = f"{layer_name}.npy"
            filepath = output_dir / filename

            # Check if this is a memory-mapped array from concatenation
            temp_concat_file = output_dir / f"{layer_name}_temp_concat.npy"

            if temp_concat_file.exists() and isinstance(activation_array, np.memmap):
                # This is a memory-mapped array - just rename the file
                # First, ensure all data is flushed to disk
                if hasattr(activation_array, 'flush'):
                    activation_array.flush()

                # Delete reference to allow file rename
                del activation_array

                # Rename temp file to final filename
                temp_concat_file.rename(filepath)
                logger.debug(
                    f"Renamed memory-mapped file for {layer_name} (zero-copy save)"
                )
            else:
                # Regular array - save normally
                np.save(filepath, activation_array)
                logger.debug(
                    f"Saved {layer_name}: shape={activation_array.shape}, "
                    f"dtype={activation_array.dtype}, size={activation_array.nbytes / 1024 / 1024:.2f}MB"
                )

            saved_files.append(filename)

        return saved_files

    #: Values below this magnitude count as "near zero" for the sparsity figure.
    NEAR_ZERO_THRESHOLD = 0.01

    @staticmethod
    def _finalise_statistics(
        sum_raw: float,
        sum_abs: float,
        sum_sq: float,
        min_val: float,
        max_val: float,
        count_near_zero: int,
        total_elements: int,
    ) -> Dict[str, Any]:
        """Turn accumulators into statistics, or into None when they are unusable.

        A statistic that could not be computed is reported as None. It used to
        be reported as 0.0, which is indistinguishable from a real measurement:
        a 73 GB extraction of gemma-4 layers 44 and 46 stored
        `mean_magnitude: 0.0`, `std_activation: 0.0` beside a perfectly valid
        `max_activation: 29.86` (2026-08-27). Zero is a plausible-looking
        answer and therefore the worst possible way to report a failure.
        """
        if total_elements <= 0:
            return {
                "mean_magnitude": None, "std_activation": None,
                "min_activation": None, "max_activation": None,
                "sparsity_percent": None,
            }

        mean_raw = sum_raw / total_elements
        mean_magnitude = sum_abs / total_elements

        # Variance of the RAW values: E[x^2] - E[x]^2. This used to subtract
        # E[|x|]^2, which is not a variance -- it mixes raw squares with mean
        # magnitude and understates the spread of a zero-centred distribution.
        variance = (sum_sq / total_elements) - (mean_raw ** 2)
        std_activation = float(np.sqrt(variance)) if variance >= 0 else None

        def _clean(value):
            if value is None:
                return None
            value = float(value)
            return None if (np.isinf(value) or np.isnan(value)) else value

        return {
            "mean_magnitude": _clean(mean_magnitude),
            "std_activation": _clean(std_activation),
            "min_activation": _clean(min_val),
            "max_activation": _clean(max_val),
            "sparsity_percent": _clean((count_near_zero / total_elements) * 100),
        }

    #: Arrays larger than this are reduced in chunks, on a thread pool.
    CHUNKED_STATISTICS_THRESHOLD_BYTES = 1024 ** 3

    #: Ceiling on statistics threads. Measured on the 16-core GPU node
    #: (2026-09-12) with the per-chunk work below: 3.8x at 4 threads, 4.6x at
    #: 8, 6.5x at 12, results bit-identical. Beyond that the disk is the limit —
    #: 399 MB/s with one reader, 544 MB/s with four — not the CPU.
    MAX_STATISTICS_WORKERS = 12

    @classmethod
    def statistics_worker_count(
        cls,
        cpu_count: Optional[int],
        mem_available_bytes: Optional[int],
        chunk_nbytes: int,
    ) -> int:
        """How many chunks to reduce at once.

        Each in-flight chunk allocates about 1.5x its own size on the heap (the
        `abs` copy plus the near-zero mask), so budget 3x for headroom. Two
        cores are left for the API and beat processes sharing the pod. Unknown
        memory falls back to a small pool rather than guessing large — an OOM
        kill here loses the whole extraction, not just the statistics.
        """
        by_cpu = max(1, (cpu_count or 1) - 2)
        if mem_available_bytes is None or chunk_nbytes <= 0:
            by_memory = 4
        else:
            by_memory = max(1, int(mem_available_bytes // (3 * chunk_nbytes)))
        return max(1, min(cls.MAX_STATISTICS_WORKERS, by_cpu, by_memory))

    @staticmethod
    def _chunk_accumulators(chunk: np.ndarray, near_zero: float) -> tuple:
        """One chunk's exact contribution: (sum, sum|x|, sum x^2, min, max, near-zero count, size).

        Every accumulation is float64. Reducing a float16 array in its own
        dtype overflows: one chunk is 100 x 512 x 3840 = 196,608,000 elements,
        and a float16 accumulator saturates at 65,504, so `abs_chunk.sum()`
        returned `inf` on the very first chunk.

        Pure and independent of every other chunk, which is what makes the
        thread pool exact rather than approximate: sums add, extremes compose,
        counts add. NumPy releases the GIL inside each of these reductions, so
        the threads genuinely run concurrently.
        """
        flat = np.ascontiguousarray(chunk).reshape(-1)
        abs_chunk = np.abs(flat)
        return (
            float(flat.sum(dtype=np.float64)),
            float(abs_chunk.sum(dtype=np.float64)),
            # einsum keeps the float64 accumulation without materialising a
            # squared copy of the chunk.
            float(np.einsum("i,i->", flat, flat, dtype=np.float64)),
            # The TRUE extremes, not the extremes of |x|. min(|x|) is 0 for any
            # real activation tensor, so the old figure said nothing while
            # being labelled "Min Activation".
            float(flat.min()) if flat.size else float("inf"),
            float(flat.max()) if flat.size else float("-inf"),
            int(np.count_nonzero(abs_chunk < near_zero)),
            int(flat.size),
        )

    def _chunked_statistics(
        self,
        activation_array: np.ndarray,
        chunk_size: int = 100,
        workers: Optional[int] = None,
        on_chunk: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Statistics over an array too large to reduce in one pass.

        WHY IT IS PARALLEL. This ran on one core. A 10,000-sample extraction at
        2,048 tokens x 2,048 dims is ~42 billion float16 values per layer, and
        on 2026-09-12 each layer took 17 minutes — 85 minutes of statistics
        after a 52-minute GPU pass, on a 16-core node with 15 cores idle.

        WHY IT REPORTS. `on_chunk(done, total)` runs on THIS thread after each
        chunk is merged. It is how the caller heartbeats the database and polls
        for cancellation. Without it the row goes silent for the whole phase,
        and the stuck-extraction janitor was one sweep away from failing that
        live job over valid output. If `on_chunk` raises, queued chunks are
        abandoned; at most `workers` chunks already in flight finish first.
        """
        n_samples = activation_array.shape[0]
        starts = range(0, n_samples, chunk_size)
        n_chunks = len(starts)
        chunk_nbytes = activation_array[: min(chunk_size, n_samples)].nbytes if n_samples else 0
        if workers is None:
            workers = self.statistics_worker_count(
                os.cpu_count(), _available_memory_bytes(), chunk_nbytes
            )
        near_zero = self.NEAR_ZERO_THRESHOLD

        def reduce(start: int) -> tuple:
            return self._chunk_accumulators(
                activation_array[start : min(start + chunk_size, n_samples)], near_zero
            )

        sum_raw = sum_abs = sum_sq = 0.0
        max_val = float("-inf")
        min_val = float("inf")
        count_near_zero = 0
        total_elements = 0

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="activation-stats")
        done = 0
        try:
            # Explicit futures, not `pool.map`: map's result generator also
            # cancels pending work when it is dropped, which left two
            # abandonment mechanisms and made `cancel_futures` below untestable.
            # One mechanism, and a test that fails without it.
            futures = [pool.submit(reduce, start) for start in starts]
            for future in futures:
                c_raw, c_abs, c_sq, c_min, c_max, c_near_zero, c_size = future.result()
                sum_raw += c_raw
                sum_abs += c_abs
                sum_sq += c_sq
                min_val = min(min_val, c_min)
                max_val = max(max_val, c_max)
                count_near_zero += c_near_zero
                total_elements += c_size
                done += 1
                if on_chunk is not None:
                    on_chunk(done, n_chunks)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        return self._finalise_statistics(
            sum_raw, sum_abs, sum_sq, min_val, max_val,
            count_near_zero, total_elements,
        )

    def _direct_statistics(self, activation_array: np.ndarray) -> Dict[str, Any]:
        """Statistics over an array small enough to reduce in one pass."""
        flat = np.ascontiguousarray(activation_array).reshape(-1)
        abs_all = np.abs(flat)

        return self._finalise_statistics(
            sum_raw=float(flat.sum(dtype=np.float64)),
            sum_abs=float(abs_all.sum(dtype=np.float64)),
            sum_sq=float(np.einsum("i,i->", flat, flat, dtype=np.float64)),
            min_val=float(flat.min()) if flat.size else float("inf"),
            max_val=float(flat.max()) if flat.size else float("-inf"),
            count_near_zero=int((abs_all < self.NEAR_ZERO_THRESHOLD).sum()),
            total_elements=int(flat.size),
        )

    def _calculate_statistics(
        self,
        activations: Dict[str, np.ndarray],
        on_progress: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for extracted activations using chunked processing.

        For large arrays, uses memory-mapped files and chunked computation on a
        thread pool, to avoid loading the entire array into memory and to use
        more than one core.

        Args:
            activations: Dictionary of activation arrays
            on_progress: Optional callback function(layer_index, n_layers,
                chunks_done, n_chunks), called on this thread. The caller's
                heartbeat and cancellation checkpoint for this phase.

        Returns:
            Dictionary mapping layer names to statistics dictionaries
        """
        statistics = {}
        n_layers = len(activations)

        for layer_index, (layer_name, activation_array) in enumerate(activations.items()):
            array_size_gb = activation_array.nbytes / (1024 ** 3)

            if activation_array.nbytes > self.CHUNKED_STATISTICS_THRESHOLD_BYTES:
                logger.info(f"Large array detected ({array_size_gb:.2f} GB), using chunked statistics calculation")

                def on_chunk(done: int, total: int, _layer: int = layer_index) -> None:
                    if on_progress is not None:
                        on_progress(_layer, n_layers, done, total)

                started = datetime.now()
                stats = self._chunked_statistics(activation_array, on_chunk=on_chunk)
                logger.info(
                    f"Statistics for {layer_name} ({layer_index + 1}/{n_layers}) took "
                    f"{(datetime.now() - started).total_seconds():.0f}s"
                )
            else:
                stats = self._direct_statistics(activation_array)
                if on_progress is not None:
                    on_progress(layer_index, n_layers, 1, 1)

            mean_magnitude = stats["mean_magnitude"]
            max_val = stats["max_activation"]
            min_val = stats["min_activation"]
            std_activation = stats["std_activation"]
            sparsity = stats["sparsity_percent"]


            statistics[layer_name] = {
                "shape": list(activation_array.shape),
                "mean_magnitude": mean_magnitude,
                "max_activation": max_val,
                "min_activation": min_val,
                "std_activation": std_activation,
                "sparsity_percent": sparsity,
                "size_mb": float(activation_array.nbytes / 1024 / 1024),
            }

        return statistics

    def get_extraction_info(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a completed extraction.

        Args:
            extraction_id: Extraction ID

        Returns:
            Metadata dictionary or None if not found
        """
        # Read primitive — same guard as the delete (MIS-E2E-070).
        metadata_path = self._extraction_dir(extraction_id) / "metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_extractions(self) -> List[Dict[str, Any]]:
        """
        List all completed extractions.

        Returns:
            List of extraction metadata dictionaries
        """
        extractions = []

        for extraction_dir in self.activations_dir.iterdir():
            if extraction_dir.is_dir():
                metadata_path = extraction_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        extractions.append(json.load(f))

        # Sort by creation time (newest first)
        extractions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return extractions

    def _extraction_dir(self, extraction_id: str) -> Path:
        """Resolve an extraction's directory from an id that may be hostile.

        THREE LAYERS, because these paths are read from and deleted
        (MIS-E2E-070). The joins used to be a bare
        `self.activations_dir / extraction_id`: `Path.__truediv__` does not
        normalise `..`, and `exists()`, `open()` and `rmtree()` all resolve it
        through the OS. `{"extraction_ids": ["../../../../etc"]}` reached
        `rmtree("/data/activations/../../../../etc")`.

        1. SHAPE — an extraction id is generated by this system and never
           contains a separator or a dot, so a traversal cannot be a valid id.
        2. CONTAINMENT — `resolve_user_path` normalises as a STRING (it never
           touches the filesystem with raw input, so there is no symlink probe
           before the check) and allow-lists against the trusted roots. It is
           the correct resolver and had ONE production caller in the whole
           codebase; `resolve_data_path`, used nearly everywhere, performs no
           containment check at all.
        3. ROOT CHECK — cheap, and independent of both.

        Raises:
            ValueError: the id is not a valid extraction id, or resolves
                outside the activations root.
        """
        if not _SAFE_ID.fullmatch(extraction_id or ""):
            logger.error("rejecting extraction id %r — invalid shape", extraction_id)
            raise ValueError(f"invalid extraction id: {extraction_id!r}")

        # `resolve_user_path` expects a path RELATIVE to a trusted root — it
        # strips a leading `/data/` or `/` and re-joins under `data_dir`.
        # Handing it an already-absolute path makes it produce a doubled path
        # that then fails its own containment check, rejecting valid ids.
        # (Caught by this fix's own test run, not in review.)
        root = Path(self.activations_dir).resolve()
        try:
            rel = root.relative_to(Path(settings.data_dir).resolve()) / extraction_id
        except ValueError:
            # activations_dir is not under data_dir in this configuration;
            # layers 1 and 3 still apply.
            rel = None

        if rel is not None:
            try:
                resolved = settings.resolve_user_path(rel)
            except ValueError as e:
                logger.error("rejecting extraction id %r — %s", extraction_id, e)
                raise
        else:
            resolved = root / extraction_id

        if root not in resolved.resolve().parents:
            logger.error("rejecting %s — resolves outside %s", resolved, root)
            raise ValueError(f"invalid extraction id: {extraction_id!r}")

        return resolved

    def delete_extraction(self, extraction_id: str) -> bool:
        """
        Delete an extraction and all its files.

        Args:
            extraction_id: Extraction ID to delete

        Returns:
            True if deleted successfully, False if not found
        """
        extraction_dir = self._extraction_dir(extraction_id)

        if not extraction_dir.exists():
            return False

        import shutil
        shutil.rmtree(extraction_dir)

        logger.info(f"Deleted extraction: {extraction_id}")
        return True
