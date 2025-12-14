"""
LLM Inference Wrapper

Uses HuggingFace transformers with memory optimization techniques:
- 8-bit/4-bit quantization via bitsandbytes
- Automatic device mapping via accelerate
- Memory-efficient attention (when available)

Provides a clean interface for model loading, text generation, and memory management.
"""

import os
import logging
import asyncio
from typing import Optional, AsyncGenerator, List, Dict, Any
from threading import Lock
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .config import settings
from .models import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    StreamChoice,
    DeltaMessage,
    Usage,
)

# Known model sizes (approximate GB in VRAM with fp16/int8)
# Used for pre-flight memory checks before loading
MODEL_SIZE_ESTIMATES = {
    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.5,
    # Gemma models
    "google/gemma-3-1b-it": 2.5,
    "google/gemma-3-4b-it": 5.0,
    "google/gemma-3-12b-it": 22.0,  # TOO LARGE for 24GB GPU!
    "google/gemma-3-27b-it": 50.0,  # TOO LARGE
    "google/gemma-2-2b-it": 3.0,
    "google/gemma-2-9b-it": 12.0,
    "google/gemma-2-27b-it": 50.0,
    # Phi models
    "microsoft/Phi-3-mini-4k-instruct": 4.0,
    "microsoft/phi-2": 3.0,
    # Qwen models
    "Qwen/Qwen2-0.5B-Instruct": 1.0,
    "Qwen/Qwen2-1.5B-Instruct": 2.0,
    "Qwen/Qwen2-7B-Instruct": 10.0,
    "Qwen/Qwen2.5-3B-Instruct": 4.0,
    "Qwen/Qwen2.5-7B-Instruct": 10.0,
    # Llama models
    "meta-llama/Llama-3.2-1B-Instruct": 2.0,
    "meta-llama/Llama-3.2-3B-Instruct": 4.0,
    "meta-llama/Llama-3.1-8B-Instruct": 12.0,
    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.2": 10.0,
}


class ModelTooLargeError(Exception):
    """Raised when a model is too large for available GPU memory."""
    pass


class InferenceTimeoutError(Exception):
    """Raised when inference times out."""
    pass

logger = logging.getLogger(__name__)


class LLMInference:
    """
    LLM inference wrapper with model management and generation capabilities.

    Provides:
    - Lazy model loading with quantization support
    - Memory-efficient inference via bitsandbytes
    - Streaming and non-streaming generation
    - Token counting for usage statistics
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_id: Optional[str] = None
        self._lock = Lock()
        self._available_models: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1)  # For timeout handling

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU available: {gpu_name} ({self.gpu_total_memory_gb:.1f} GB)")
        else:
            self.device = torch.device("cpu")
            self.gpu_total_memory_gb = 0
            logger.warning("No GPU available, using CPU (will be slow)")

        # Ensure cache directories exist
        os.makedirs(settings.model_cache_dir, exist_ok=True)

    def estimate_model_size(self, model_id: str) -> float:
        """
        Estimate the VRAM requirement for a model.

        Uses known model sizes or estimates based on model name patterns.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Estimated VRAM requirement in GB
        """
        # Check known models first
        if model_id in MODEL_SIZE_ESTIMATES:
            return MODEL_SIZE_ESTIMATES[model_id]

        # Estimate based on model name patterns
        model_lower = model_id.lower()

        # Extract size indicators from model name
        if "27b" in model_lower:
            return 50.0
        elif "12b" in model_lower or "13b" in model_lower:
            return 22.0
        elif "8b" in model_lower or "9b" in model_lower:
            return 12.0
        elif "7b" in model_lower:
            return 10.0
        elif "4b" in model_lower or "5b" in model_lower:
            return 5.0
        elif "3b" in model_lower:
            return 4.0
        elif "2b" in model_lower:
            return 3.0
        elif "1b" in model_lower or "1.5b" in model_lower:
            return 2.0
        elif "0.5b" in model_lower or "500m" in model_lower:
            return 1.0

        # Default estimate for unknown models (conservative)
        logger.warning(f"Unknown model size for {model_id}, using conservative estimate of 10GB")
        return 10.0

    def check_model_fits_in_memory(self, model_id: str) -> tuple[bool, str]:
        """
        Check if a model will fit in available GPU memory.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Tuple of (fits: bool, message: str)
        """
        if not self.gpu_available:
            return True, "Running on CPU, no GPU memory limits"

        estimated_size = self.estimate_model_size(model_id)
        max_allowed = settings.max_model_memory_gb

        # Calculate available memory (need headroom for inference)
        available = self.gpu_total_memory_gb - 4.0  # Reserve 4GB for inference buffers

        if estimated_size > max_allowed:
            return False, (
                f"Model {model_id} requires ~{estimated_size:.1f}GB VRAM, "
                f"but max allowed is {max_allowed:.1f}GB. "
                f"Use a smaller model like google/gemma-3-4b-it or Qwen/Qwen2-7B-Instruct."
            )

        if estimated_size > available:
            return False, (
                f"Model {model_id} requires ~{estimated_size:.1f}GB VRAM, "
                f"but only {available:.1f}GB is available (total: {self.gpu_total_memory_gb:.1f}GB, "
                f"reserved for inference: 4.0GB). "
                f"Use a smaller model."
            )

        return True, f"Model {model_id} (~{estimated_size:.1f}GB) fits in available memory ({available:.1f}GB)"

    @property
    def model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self._model is not None

    @property
    def current_model(self) -> Optional[str]:
        """Get the currently loaded model ID"""
        return self._current_model_id

    def _apply_chat_template(self, messages: List[ChatMessage]) -> str:
        """
        Apply chat template to convert messages to model input format.
        Uses the tokenizer's built-in chat template if available.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Convert to the format expected by tokenizer
        formatted_messages = [
            {"role": msg.role, "content": msg.content or ""}
            for msg in messages
        ]

        # Use tokenizer's chat template if available
        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                return self._tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")

        # Fallback: simple format for models without chat template
        prompt = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
        return prompt

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer"""
        if self._tokenizer is None:
            # Rough estimate: ~4 chars per token
            return len(text) // 4
        return len(self._tokenizer.encode(text))

    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a model for inference with memory optimization.

        Args:
            model_id: HuggingFace model ID or local path
            force_reload: Force reload even if model is already loaded

        Returns:
            True if model loaded successfully

        Raises:
            ModelTooLargeError: If model is too large for GPU memory
        """
        with self._lock:
            if self._current_model_id == model_id and not force_reload:
                logger.info(f"Model {model_id} already loaded")
                return True

            # Pre-flight memory check
            fits, message = self.check_model_fits_in_memory(model_id)
            if not fits:
                logger.error(f"Model memory check failed: {message}")
                raise ModelTooLargeError(message)
            logger.info(f"Memory check passed: {message}")

            logger.info(f"Loading model: {model_id}")
            start_time = time.time()

            try:
                # Unload current model to free memory
                await self.unload_model()

                # Load tokenizer
                logger.info(f"Loading tokenizer for {model_id}")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=settings.model_cache_dir,
                    trust_remote_code=True,
                    token=settings.hf_token,
                )

                # Set pad token if not set
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                # Configure quantization for memory efficiency
                quantization_config = None
                is_gemma3 = "gemma-3" in model_id.lower() or "gemma3" in model_id.lower()

                if self.gpu_available and not is_gemma3:
                    try:
                        # Use 8-bit quantization for non-Gemma 3 models
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                        )
                        logger.info("Using 8-bit quantization for memory efficiency")
                    except Exception as e:
                        logger.warning(f"Quantization not available: {e}")
                elif is_gemma3:
                    # Gemma 3 models have quantization issues - use fp16 without quantization
                    logger.info("Gemma 3 detected - using fp16 without quantization")

                # Load model with optimizations
                logger.info(f"Loading model {model_id} with memory optimizations")

                model_kwargs = {
                    "cache_dir": settings.model_cache_dir,
                    "trust_remote_code": True,
                    "token": settings.hf_token,
                    "low_cpu_mem_usage": True,
                }

                if self.gpu_available:
                    model_kwargs["device_map"] = "auto"
                    # Gemma 3 requires bfloat16, other models use float16
                    if is_gemma3:
                        model_kwargs["torch_dtype"] = torch.bfloat16
                    else:
                        model_kwargs["torch_dtype"] = torch.float16
                    if quantization_config:
                        model_kwargs["quantization_config"] = quantization_config
                else:
                    model_kwargs["torch_dtype"] = torch.float32

                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **model_kwargs
                )

                self._current_model_id = model_id

                # Track available models
                self._available_models[model_id] = {
                    "id": model_id,
                    "loaded_at": time.time(),
                }

                load_time = time.time() - start_time
                logger.info(f"Model {model_id} loaded in {load_time:.2f}s")

                # Log memory usage
                if self.gpu_available:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

                return True

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                self._model = None
                self._tokenizer = None
                self._current_model_id = None
                raise

    async def unload_model(self):
        """Unload the current model to free memory"""
        if self._model is not None:
            logger.info(f"Unloading model: {self._current_model_id}")
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._current_model_id = None

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion (non-streaming).

        Args:
            request: Chat completion request

        Returns:
            ChatCompletionResponse with generated text
        """
        # Ensure model is loaded
        if not self.model_loaded:
            await self.load_model(request.model)
        elif self._current_model_id != request.model:
            await self.load_model(request.model)

        # Apply chat template
        prompt = self._apply_chat_template(request.messages)
        prompt_tokens = self._count_tokens(prompt)

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_context_length,
        )

        # Move inputs to device
        if self.gpu_available:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        max_tokens = request.max_tokens or settings.default_max_tokens
        temperature = request.temperature if request.temperature is not None else settings.default_temperature
        top_p = request.top_p if request.top_p is not None else settings.default_top_p

        # Calculate repetition_penalty from frequency_penalty and presence_penalty
        # OpenAI penalties are -2 to 2, we convert to HuggingFace repetition_penalty (typically 1.0 to 1.5)
        # Base value of 1.0 means no penalty, higher values penalize repetition
        freq_penalty = request.frequency_penalty or 0
        pres_penalty = request.presence_penalty or 0
        # Combine penalties: average them and scale to repetition_penalty range
        combined_penalty = (freq_penalty + pres_penalty) / 2
        repetition_penalty = 1.0 + (combined_penalty * 0.25)  # Scale to 1.0-1.5 range
        repetition_penalty = max(1.0, min(1.5, repetition_penalty))  # Clamp to safe range

        # Handle stop sequences
        eos_token_ids = [self._tokenizer.eos_token_id]
        if request.stop:
            stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
            for seq in stop_sequences:
                # Convert stop sequence to token IDs
                seq_tokens = self._tokenizer.encode(seq, add_special_tokens=False)
                if seq_tokens:
                    # Add each token as a potential stop token
                    eos_token_ids.extend(seq_tokens)
            # Remove duplicates while preserving order
            eos_token_ids = list(dict.fromkeys(eos_token_ids))

        # Handle seed for reproducibility
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if self.gpu_available:
                torch.cuda.manual_seed_all(request.seed)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos_token_ids,
                repetition_penalty=repetition_penalty,
            )

        # Decode output (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completion_tokens = len(generated_tokens)

        # Determine finish reason
        finish_reason = "stop"
        if completion_tokens >= max_tokens:
            finish_reason = "length"

        # Build response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=self._current_model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=generated_text.strip(),
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            system_fingerprint=f"llm-{self._current_model_id}",
        )

        return response

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Generate a chat completion with streaming.

        Args:
            request: Chat completion request

        Yields:
            ChatCompletionChunk objects for streaming
        """
        # Ensure model is loaded
        if not self.model_loaded:
            await self.load_model(request.model)
        elif self._current_model_id != request.model:
            await self.load_model(request.model)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Apply chat template
        prompt = self._apply_chat_template(request.messages)

        # Tokenize input
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_context_length,
        )

        if self.gpu_available:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        max_tokens = request.max_tokens or settings.default_max_tokens
        temperature = request.temperature if request.temperature is not None else settings.default_temperature

        # Send initial chunk with role
        yield ChatCompletionChunk(
            id=completion_id,
            model=self._current_model_id,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None,
                )
            ],
        )

        # Generate token by token
        generated_tokens = 0
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self._model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Check for EOS
                if next_token.item() == self._tokenizer.eos_token_id:
                    yield ChatCompletionChunk(
                        id=completion_id,
                        model=self._current_model_id,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    break

                # Decode token
                token_text = self._tokenizer.decode(next_token[0], skip_special_tokens=True)

                # Yield chunk
                yield ChatCompletionChunk(
                    id=completion_id,
                    model=self._current_model_id,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=token_text),
                            finish_reason=None,
                        )
                    ],
                )

                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens += 1

                # Allow other async operations
                await asyncio.sleep(0)

        # Final chunk if we hit max tokens
        if generated_tokens >= max_tokens:
            yield ChatCompletionChunk(
                id=completion_id,
                model=self._current_model_id,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason="length",
                    )
                ],
            )

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        stats = {
            "gpu_available": self.gpu_available,
        }

        if self.gpu_available:
            stats["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            stats["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            stats["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

        return stats

    def list_available_models(self) -> List[str]:
        """List models that have been loaded or are available"""
        return list(self._available_models.keys())


# Singleton instance
inference_engine = LLMInference()
