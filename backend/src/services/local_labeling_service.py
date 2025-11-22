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
        examples: List[Dict[str, Any]],
        neuron_index: Optional[int] = None,
        feature_id: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate semantic label for a feature based on context examples.

        This is the new context-based labeling method that uses full activation examples
        with prefix/prime/suffix tokens instead of aggregated token statistics.

        Args:
            examples: List of top-K activation example dicts with keys:
                - prefix_tokens: List[str] - Tokens before prime
                - prime_token: str - The token with maximum activation
                - suffix_tokens: List[str] - Tokens after prime
                - max_activation: float - Peak activation value
            neuron_index: Optional neuron index for fallback naming
            feature_id: Optional feature ID for fallback naming

        Returns:
            Dict with {"category": "...", "specific": "...", "description": "..."}
        """
        fallback_label = f"feature_{feature_id or neuron_index or 'unknown'}"

        if not examples:
            logger.warning("Empty examples, using fallback label")
            return {"category": "empty_features", "specific": fallback_label, "description": ""}

        # Build prompt with context examples
        prompt = self._build_prompt_from_examples(examples, feature_id=feature_id)

        # Ensure model is loaded
        if not self.is_loaded:
            self.load_model()

        try:
            # Format prompt for chat model
            system_message = """You analyze sparse autoencoder (SAE) features using full-context activation examples. Your ONLY job is to infer the single underlying conceptual meaning shared by the most strongly-activating tokens, taking into account both the highlighted token(s) and their surrounding context.

You are given short text spans. In each span, the token(s) where the feature activates most strongly are wrapped in double angle brackets, like <<this>>. Use all of the examples and their context to infer a single latent direction: a 1–2 word human concept that would be useful for steering model behavior.

You must NOT:
- describe grammar, syntax, token types, or surface patterns
- list the example tokens back
- say "this feature detects words like..."
- label the feature with only a grammatical category
- describe frequency, morphology, or implementation details

If ANY coherent conceptual theme exists, use category 'semantic'.
If no coherent theme exists, use category 'system' and concept 'noise_feature'.

You must return ONLY a valid JSON object in this structure:
{
  "specific": "one_or_two_word_concept",
  "category": "semantic_or_other",
  "description": "One sentence describing the real conceptual meaning represented by this feature."
}

Rules:
- JSON only
- No markdown
- No notes
- No code fences
- No text before or after the JSON
- Double quotes only"""

            messages = [
                {
                    "role": "system",
                    "content": system_message
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
                max_length=3072  # Increased for context examples
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Allow for JSON response
                    temperature=0.2,  # Low temperature for consistency
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

            # Parse JSON response
            labels = self._parse_dual_label(response, fallback_label)

            logger.debug(f"Generated labels: category='{labels['category']}', specific='{labels['specific']}'")
            return labels

        except Exception as e:
            logger.error(f"Error generating label: {e}", exc_info=True)
            return {"category": "error_feature", "specific": fallback_label, "description": ""}

    def _build_prompt_from_examples(
        self,
        examples: List[Dict[str, Any]],
        feature_id: Optional[str] = None
    ) -> str:
        """
        Build user prompt from context-based activation examples.

        Args:
            examples: List of activation example dicts with prefix/prime/suffix tokens
            feature_id: Optional feature ID for context

        Returns:
            Formatted prompt string with examples
        """
        feature_label = feature_id or "this feature"

        prompt = f"""Analyze sparse autoencoder feature {feature_label}.
You are given some of the highest-activating examples for this feature. In each example, the main activating token(s) are wrapped in << >>.

Use ALL of the examples, including their surrounding context, to infer the smallest semantic concept that explains why these tokens activate the same feature.

Each example is formatted as:
  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

Examples:

"""

        # Format each example
        for i, ex in enumerate(examples[:10], 1):  # Use first 10 examples
            prefix = ' '.join(ex.get('prefix_tokens', []))
            prime = ex.get('prime_token', '')
            suffix = ' '.join(ex.get('suffix_tokens', []))
            activation = ex.get('max_activation', 0.0)

            # Truncate very long contexts
            if len(prefix) > 100:
                prefix = '...' + prefix[-97:]
            if len(suffix) > 100:
                suffix = suffix[:97] + '...'

            prompt += f"Example {i} (activation: {activation:.2f}): {prefix} <<{prime}>> {suffix}\n"

        prompt += """
Instructions:
- Focus on what the highlighted tokens have in common when interpreted IN CONTEXT.
- Ignore purely syntactic or tokenization details.
- Prefer semantic, conceptual, or functional interpretations (e.g., 'legal_procedure', 'feminist_politics', 'scientific_uncertainty').
- If you cannot find a coherent concept, treat this as a noise feature.

Return ONLY this exact JSON object:
{
  "specific": "concept",
  "category": "semantic_or_other",
  "description": "One sentence describing the conceptual meaning."
}"""

        return prompt

    def _parse_dual_label(self, response: str, fallback_label: str) -> Dict[str, str]:
        """
        Parse JSON response containing category, specific, and description.

        Args:
            response: Raw model output (expected JSON format)
            fallback_label: Fallback label if parsing fails

        Returns:
            Dict with {"category": "...", "specific": "...", "description": "..."}
        """
        import json
        import re

        # Try to extract JSON from response
        try:
            # Remove markdown code fences if present
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Extract content between code fences
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_clean, re.DOTALL)
                if match:
                    response_clean = match.group(1).strip()
                else:
                    # Remove just the fence markers
                    response_clean = response_clean.replace('```json', '').replace('```', '').strip()

            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_clean, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                labels = json.loads(json_str)

                # Validate required fields
                if 'specific' in labels and 'category' in labels:
                    # Clean the labels
                    specific = str(labels['specific']).strip().lower().replace(' ', '_').replace('-', '_')
                    category = str(labels['category']).strip().lower().replace(' ', '_').replace('-', '_')
                    description = str(labels.get('description', '')).strip()

                    # Remove special characters
                    specific = ''.join(c for c in specific if c.isalnum() or c == '_').strip('_')
                    category = ''.join(c for c in category if c.isalnum() or c == '_').strip('_')

                    # Collapse multiple underscores
                    while '__' in specific:
                        specific = specific.replace('__', '_')
                    while '__' in category:
                        category = category.replace('__', '_')

                    # Truncate if too long
                    if len(specific) > 50:
                        specific = specific[:50]
                    if len(category) > 30:
                        category = category[:30]

                    # Fallback if empty
                    if not specific or specific == '_':
                        specific = fallback_label
                    if not category or category == '_':
                        category = "semantic"

                    return {
                        "category": category,
                        "specific": specific,
                        "description": description
                    }

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse JSON label: {e}")
            logger.debug(f"Response was: {response[:200]}")

        # Fallback: use simple label extraction
        logger.warning("Using fallback label parsing")
        return {
            "category": "semantic",
            "specific": fallback_label,
            "description": ""
        }

    def batch_generate_labels(
        self,
        features_examples: List[List[Dict[str, Any]]],
        neuron_indices: Optional[List[int]] = None,
        feature_ids: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, str]]:
        """
        Generate labels for multiple features efficiently using context examples.

        Loads model once and processes all features before unloading.

        Args:
            features_examples: List of example lists, one per feature
            neuron_indices: Optional list of neuron indices for fallback naming
            feature_ids: Optional list of feature IDs for fallback naming
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of label dicts ({"category": "...", "specific": "...", "description": "..."})
            in same order as input
        """
        logger.info(f"Starting batch label generation for {len(features_examples)} features")

        # Load model once for entire batch
        self.load_model()

        labels = []
        try:
            for i, examples in enumerate(features_examples):
                neuron_index = neuron_indices[i] if neuron_indices else None
                feature_id = feature_ids[i] if feature_ids else None
                label = self.generate_label(
                    examples=examples,
                    neuron_index=neuron_index,
                    feature_id=feature_id
                )
                labels.append(label)

                # Progress updates
                if (i + 1) % 100 == 0 or i == len(features_examples) - 1:
                    logger.info(f"Labeled {i + 1}/{len(features_examples)} features")
                    if progress_callback:
                        progress_callback(i + 1, len(features_examples))

        finally:
            # Always unload model to free memory
            self.unload_model()

        logger.info(f"Batch labeling complete. Generated {len(labels)} labels")
        return labels
