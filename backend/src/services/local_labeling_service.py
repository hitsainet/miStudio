"""
Local model-based feature labeling service.

This service uses a local instruction-tuned LLM (Phi-3-mini) to generate
semantic labels for SAE features based on their activation patterns.
Provides zero-cost, privacy-preserving alternative to API-based labeling.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalLabelingService:
    """
    Service for generating feature labels using local instruction-tuned models.

    Uses Microsoft Phi-3-mini-4k-instruct for semantic analysis of activation patterns.
    Model is loaded on-demand and unloaded after batch processing to save memory.

    Memory Management:
    - Model loaded only during labeling phase (after feature extraction)
    - 4-bit quantization reduces memory footprint to ~2GB
    - Batch processing amortizes model loading overhead
    - Explicit unload frees GPU memory for other tasks
    """

    # Model configuration
    DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    ALTERNATIVE_MODELS = {
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "llama": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen": "Qwen/Qwen2.5-3B-Instruct"
    }

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize local labeling service.

        Args:
            model_name: Model identifier (phi3, llama, qwen) or full HF path
        """
        self.model = None
        self.tokenizer = None

        # Resolve model name
        if model_name is None or model_name == "phi3":
            self.model_name = self.DEFAULT_MODEL
        elif model_name in self.ALTERNATIVE_MODELS:
            self.model_name = self.ALTERNATIVE_MODELS[model_name]
        else:
            self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load_model(self):
        """
        Load the labeling model into memory.

        Uses 4-bit quantization for memory efficiency (~2GB VRAM).
        Model is cached in HuggingFace cache after first download.
        """
        if self.is_loaded:
            logger.debug("Labeling model already loaded")
            return

        logger.info(f"Loading labeling model: {self.model_name}")
        logger.info(f"Target device: {self.device}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
                load_in_4bit=True if self.device == "cuda" else False,
                bnb_4bit_compute_dtype=torch.float16 if self.device == "cuda" else None
            )

            self.is_loaded = True

            # Log memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                logger.info(f"Labeling model loaded. GPU memory: {memory_allocated:.2f}GB")
            else:
                logger.info("Labeling model loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load labeling model: {e}", exc_info=True)
            raise

    def unload_model(self):
        """
        Unload the model to free memory.

        Should be called after batch processing is complete.
        """
        if not self.is_loaded:
            return

        logger.info("Unloading labeling model")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            logger.info(f"Labeling model unloaded. GPU memory: {memory_allocated:.2f}GB")

    def generate_label(
        self,
        token_stats: Dict[str, Dict[str, float]],
        top_k: int = 50,
        neuron_index: Optional[int] = None
    ) -> str:
        """
        Generate semantic label for a feature based on token statistics.

        Args:
            token_stats: Dict mapping token to {count, total_activation, max_activation}
            top_k: Number of top tokens to include in prompt
            neuron_index: Optional neuron index for fallback naming

        Returns:
            Single word or short phrase representing the feature concept
        """
        if not token_stats:
            logger.warning("Empty token stats, using fallback label")
            return f"feature_{neuron_index}" if neuron_index is not None else "empty_feature"

        # Sort tokens by total activation strength
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["total_activation"],
            reverse=True
        )[:top_k]

        # Filter out junk tokens (stopwords, punctuation, digits, artifacts)
        filtered_tokens = self._filter_junk_tokens(sorted_tokens)

        if not filtered_tokens:
            logger.warning(f"All tokens filtered as junk for neuron {neuron_index}, using fallback label")
            return f"feature_{neuron_index}" if neuron_index is not None else "filtered_feature"

        # Build prompt with filtered token frequency table
        prompt = self._build_prompt(filtered_tokens)

        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()

        try:
            # Format prompt for chat model
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Respond with single-word or 2-3 word labels only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Apply chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,  # Allow 2-3 words
                    temperature=0.3,  # Low temperature for consistency
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode and extract label
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Clean and validate
            label = self._clean_label(response)

            logger.debug(f"Generated label: '{label}' from response: '{response[:50]}'")
            return label

        except Exception as e:
            logger.error(f"Error generating label: {e}", exc_info=True)
            fallback = f"feature_{neuron_index}" if neuron_index is not None else "error_feature"
            return fallback

    def _build_prompt(self, sorted_tokens: List[tuple]) -> str:
        """
        Build analysis prompt with token frequency table.

        Args:
            sorted_tokens: List of (token, stats_dict) tuples sorted by activation

        Returns:
            Formatted prompt string
        """
        prompt = """Analyze this sparse autoencoder neuron's activation pattern.

Top tokens that activate this neuron:

TOKEN                | COUNT | AVG_ACT | MAX_ACT
---------------------|-------|---------|--------
"""

        for token, stats in sorted_tokens[:50]:  # Limit to top 50
            avg_act = stats["total_activation"] / stats["count"]

            # Escape and truncate token for display
            token_display = repr(token)[:20].ljust(20)

            prompt += f"{token_display} | {stats['count']:5} | {avg_act:7.2f} | {stats['max_activation']:7.2f}\n"

        prompt += """
What single concept does this neuron represent?

Respond with ONLY one word or short phrase (max 3 words). Examples:
- "determiners" (the, a, an)
- "negation" (not, never, no)
- "plural_nouns" (words ending in -s)
- "past_tense" (was, had, did)
- "code_keywords" (def, class, import)

Concept:"""

        return prompt

    def _filter_junk_tokens(self, sorted_tokens: List[tuple]) -> List[tuple]:
        """
        Filter out junk tokens (punctuation, artifacts, stopwords, digits).

        Removes:
        - Punctuation-only tokens
        - Tokenization artifacts (##, Ġ, etc.)
        - High-frequency stopwords (the, a, and, etc.)
        - Single characters (except $ and %)
        - Pure digit tokens (0-9, 10, 2023, etc.)
        - Whitespace-only tokens
        - Junk characters with no cognitive meaning

        Args:
            sorted_tokens: List of (token, stats_dict) tuples

        Returns:
            Filtered list of (token, stats_dict) tuples
        """
        import string

        # Define stopwords (common function words with little semantic value)
        stopwords = {
            # Articles, determiners, conjunctions
            'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'yet', 'so',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
            'under', 'over', 'off', 'down', 'near', 'onto', 'upon',
            # Common verbs (to be, to have, modals)
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having',
            'do', 'does', 'did', 'doing', 'done',
            'will', 'would', 'should', 'can', 'could', 'may', 'might', 'must', 'shall',
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them', 'their', 'theirs',
            'my', 'mine', 'your', 'yours', 'his', 'her', 'hers', 'its', 'our', 'ours',
            'me', 'him', 'us', 'themselves', 'myself', 'yourself', 'himself', 'herself', 'itself',
            # Demonstratives & interrogatives
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
            'when', 'where', 'why', 'how',
            # Common adverbs & adjectives
            'as', 'if', 'than', 'so', 'just', 'very', 'too', 'also', 'only', 'own', 'same',
            'such', 'no', 'not', 'more', 'most', 'less', 'least', 'other', 'some', 'any',
            'each', 'every', 'all', 'both', 'few', 'many', 'much', 'several', 'another',
            'even', 'while', 'out', 'there', 'here', 'now', 'then', 'still', 'again',
            # Common verbs (action)
            'get', 'got', 'getting', 'make', 'made', 'making', 'go', 'going', 'went', 'gone',
            'take', 'took', 'taken', 'taking', 'see', 'saw', 'seen', 'seeing', 'come', 'came', 'coming',
            'give', 'gave', 'given', 'giving', 'use', 'used', 'using', 'find', 'found', 'finding',
            'tell', 'told', 'telling', 'ask', 'asked', 'asking', 'work', 'worked', 'working',
            'seem', 'seemed', 'seeming', 'feel', 'felt', 'feeling', 'try', 'tried', 'trying',
            'leave', 'left', 'leaving', 'call', 'called', 'calling', 'put', 'putting'
        }

        # Define punctuation set
        punctuation_set = set(string.punctuation)

        filtered = []
        for token, stats in sorted_tokens:
            # Strip spaces for analysis
            token_stripped = token.strip()

            # Skip empty or whitespace-only tokens
            if not token_stripped or token_stripped.isspace():
                continue

            # Skip pure punctuation (but keep if mixed with other chars)
            if all(c in punctuation_set or c.isspace() for c in token_stripped):
                continue

            # Skip tokenization artifacts
            # - WordPiece markers: ##word, word##
            # - Special whitespace: Ġ (GPT-2 style), ▁ (SentencePiece alone)
            # - BPE markers: </w>, <w>
            if ('##' in token_stripped or
                'Ġ' in token_stripped or
                token_stripped == '▁' or  # SentencePiece marker alone
                token_stripped.startswith(('</w>', '<w>')) or
                token_stripped.endswith(('</w>', '<w>'))):
                continue

            # Skip single characters (except meaningful ones like $ and %)
            # Handle both regular tokens and SentencePiece tokens (▁X)
            token_without_marker = token_stripped.lstrip().lstrip('▁')
            if len(token_without_marker) == 1 and token_without_marker not in {'$', '%', '€', '£', '¥'}:
                continue

            # Skip pure digit tokens (0, 1, 10, 2023, etc.)
            # Remove leading space/SentencePiece marker and check if rest is all digits
            token_no_marker = token_stripped.lstrip().lstrip('▁')
            if token_no_marker.isdigit():
                continue

            # Skip stopwords (case-insensitive, removing leading space/SentencePiece marker)
            # Handle both regular spaces and SentencePiece marker (▁)
            token_lower = token_stripped.lstrip().lstrip('▁').lower()
            if token_lower in stopwords:
                continue

            # Keep this token!
            filtered.append((token, stats))

        return filtered

    def _clean_label(self, response: str) -> str:
        """
        Clean and validate model response.

        Converts model output to standardized feature label format.

        Args:
            response: Raw model output

        Returns:
            Cleaned label (lowercase_with_underscores)
        """
        # Remove quotes, extra whitespace
        label = response.strip().strip('"\'').strip()

        # Take first line if multiline
        label = label.split('\n')[0]

        # Remove common prefixes
        for prefix in ["concept:", "label:", "answer:"]:
            if label.lower().startswith(prefix):
                label = label[len(prefix):].strip()

        # Convert to lowercase with underscores
        label = label.lower().replace(' ', '_').replace('-', '_')

        # Remove special characters except underscore
        label = ''.join(c for c in label if c.isalnum() or c == '_')

        # Remove leading/trailing underscores
        label = label.strip('_')

        # Collapse multiple underscores
        while '__' in label:
            label = label.replace('__', '_')

        # Truncate if too long
        if len(label) > 30:
            label = label[:30]

        # Fallback if empty
        if not label or label == '_':
            label = "unknown_feature"

        return label

    def batch_generate_labels(
        self,
        features_token_stats: List[Dict[str, Dict[str, float]]],
        neuron_indices: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Generate labels for multiple features efficiently.

        Loads model once and processes all features before unloading.

        Args:
            features_token_stats: List of token stats dicts, one per feature
            neuron_indices: Optional list of neuron indices for fallback naming
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of labels in same order as input
        """
        logger.info(f"Starting batch label generation for {len(features_token_stats)} features")

        # Load model once for entire batch
        self.load_model()

        labels = []
        try:
            for i, token_stats in enumerate(features_token_stats):
                neuron_index = neuron_indices[i] if neuron_indices else None
                label = self.generate_label(token_stats, neuron_index=neuron_index)
                labels.append(label)

                # Progress updates
                if (i + 1) % 100 == 0 or i == len(features_token_stats) - 1:
                    logger.info(f"Labeled {i + 1}/{len(features_token_stats)} features")
                    if progress_callback:
                        progress_callback(i + 1, len(features_token_stats))

        finally:
            # Always unload model to free memory
            self.unload_model()

        logger.info(f"Batch labeling complete. Generated {len(labels)} labels")
        return labels
