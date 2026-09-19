"""
Local model-based feature labeling service.

This service uses a local instruction-tuned LLM (Phi-3-mini) to generate
semantic labels for SAE features based on their activation patterns.
Provides zero-cost, privacy-preserving alternative to API-based labeling.

Enhanced with NLP analysis that provides statistical summaries of all 100
activation examples to help the LLM make better labeling decisions.
"""

import gc

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from src.ml.model_devices import cuda_devices, detach_dispatch_hooks, input_device, off_gpu_modules
from src.services.nlp_analysis_service import NLPAnalysisService

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

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        device: "torch.device | str",
        device_map: Optional[str] = None,
        max_memory: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize local labeling service.

        Args:
            model_name: Model identifier (phi3, llama, qwen) or full HF path
            device: The device to run on. REQUIRED: the worker chooses it with
                ``gpu_placement.place_job``. This used to pick ``"cuda"`` itself,
                which is whatever card CUDA calls current — card 0, the 12 GB
                3080 Ti, once a second GPU was added.
            device_map: The placement's ``device_map`` (``Placement.device_map``),
                passed through as it is. None puts every weight on ``device``.
            max_memory: A SPLIT placement's per-card budget
                (``Placement.max_memory``), for a judge no single card can hold.

        Raises:
            ValueError: a split budget without the split's ``device_map``; the
                budget would be ignored and the judge loaded onto one card.
        """
        if max_memory and device_map is None:
            raise ValueError("A split's max_memory needs the split's device_map (Placement.device_map).")
        self.model = None
        self.tokenizer = None
        self.model_name = self.resolve_model_name(model_name)
        self.device = torch.device(device)
        self.device_map = device_map
        self.max_memory = dict(max_memory) if max_memory else None
        #: The cards the loaded judge holds tensors on, for memory and cleanup.
        self._devices: List[torch.device] = []
        self.is_loaded = False

    @classmethod
    def resolve_model_name(cls, model_name: Optional[str]) -> str:
        """The HuggingFace id that an alias (phi3, llama, qwen) or a full path names."""
        if model_name is None or model_name == "phi3":
            return cls.DEFAULT_MODEL
        return cls.ALTERNATIVE_MODELS.get(model_name, model_name)

    @property
    def is_split(self) -> bool:
        return self.max_memory is not None

    def _cards_in_use(self) -> List[torch.device]:
        """Every card the judge uses: what the load recorded, else its one device when that is a GPU."""
        if self._devices:
            return list(self._devices)
        return [self.device] if self.device.type == "cuda" else []

    def _input_device(self) -> torch.device:
        """Where a prompt goes.

        The judge's one device, or on a split the embedding's card — which is not
        ``self.device`` (the placement's first card) whenever accelerate put the
        embedding elsewhere, and ``generate`` then fails in the embedding lookup.
        """
        return input_device(self.model) if self.is_split else self.device

    def _plan_split(self):
        """The split's budget, verified against transformers' own map; None when it cannot be mapped here.

        The judge loads through ``from_pretrained`` directly, not the shared
        loader, so it maps its own split. See ``ml/split_load.py``: transformers
        holds back the largest layer's size on the lowest-index card, so a split
        whose budgets held the judge still spilled to disk.

        Raises:
            RuntimeError: transformers would map part of the judge off the split's GPUs.
        """
        from transformers import BitsAndBytesConfig

        from src.ml.split_load import SplitDoesNotFit, plan_split_load

        if self.device_map != "sequential":
            return None
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - no config on disk: the load maps the split itself
            logger.warning(
                "No local config for judge %s, so its split is mapped only by the load: %s",
                self.model_name, exc,
            )
            return None
        try:
            return plan_split_load(
                config,
                max_memory=self.max_memory,
                dtype=torch.float16,
                # What `load_in_4bit=True` with an fp16 compute dtype builds.
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                ),
                trust_remote_code=True,
                model_name=f"Labeling model {self.model_name}",
            )
        except SplitDoesNotFit as refusal:
            raise RuntimeError(str(refusal)) from refusal

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

            on_gpu = self.device.type == "cuda"

            # Load model with 4-bit quantization. Every weight goes to the ONE
            # device the job was placed on, or — for a judge no single card can
            # hold — across the split's cards within their budgets, by the
            # placement's own device_map. bitsandbytes also reads the current
            # device, which place_job has made the first.
            load_kwargs: Dict[str, Any] = dict(
                torch_dtype=torch.float16 if on_gpu else torch.float32,
                device_map=self.device_map if self.device_map is not None else {"": self.device},
                trust_remote_code=True,
                load_in_4bit=on_gpu,
                bnb_4bit_compute_dtype=torch.float16 if on_gpu else None
            )
            if self.is_split:
                # Mapped before loading, with transformers' own inference: a
                # split its budgets hold gets back the room transformers holds
                # on the lowest-index card, and one that still spills is refused
                # here rather than after the judge's weights are read.
                planned = self._plan_split() if on_gpu else None
                load_kwargs["max_memory"] = dict(planned.max_memory if planned else self.max_memory)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            if self.is_split:
                # GPUs ONLY (operator decision 3). The budget has no "cpu" key,
                # but accelerate keeps "disk" as a last resort, so a judge the
                # cards cannot hold still loads — and then reads layers from disk
                # for every token of every label. Refuse it instead.
                offloaded = off_gpu_modules(self.model)
                if offloaded:
                    self._drop_model()
                    name, target = next(iter(offloaded.items()))
                    raise RuntimeError(
                        f"Labeling model {self.model_name} does not fit on the GPUs it was split "
                        f"across: {len(offloaded)} module(s) would run from "
                        f"{'/'.join(sorted(set(offloaded.values())))} (for example {name} on "
                        f"{target}). Local labeling runs on GPUs only; free memory on those cards "
                        f"or choose a smaller judge."
                    )
                self._devices = cuda_devices(self.model)
            else:
                self._devices = [self.device] if on_gpu else []

            self.is_loaded = True

            # Log memory usage — of every card the model is on, not card 0.
            if on_gpu:
                memory_allocated = sum(
                    torch.cuda.memory_allocated(d) for d in self._devices
                ) / (1024**3)
                logger.info(
                    f"Labeling model loaded on {self._devices}. GPU memory: {memory_allocated:.2f}GB"
                )
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

        devices = self._cards_in_use()
        self._drop_model()

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        # Clear the cache of EVERY card the model was on: a split judge's
        # layers sit on each of them.
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        if devices:
            memory_allocated = sum(torch.cuda.memory_allocated(d) for d in devices) / (1024**3)
            logger.info(
                f"Labeling model unloaded from {devices}. GPU memory: {memory_allocated:.2f}GB"
            )

    def _drop_model(self) -> None:
        """Release the model object; a split model loses its dispatch hooks first."""
        if self.model is not None:
            if self.is_split and isinstance(self.model, torch.nn.Module):
                detach_dispatch_hooks(self.model)
            del self.model
            self.model = None
            gc.collect()
        self._devices = []

    def generate_label(
        self,
        examples: List[Dict[str, Any]],
        neuron_index: Optional[int] = None,
        feature_id: Optional[str] = None,
        all_examples: Optional[List[Dict[str, Any]]] = None,
        nlp_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate semantic label for a feature based on context examples.

        This is the new context-based labeling method that uses full activation examples
        with prefix/prime/suffix tokens instead of aggregated token statistics.

        Enhanced with NLP analysis that provides statistical patterns from ALL examples
        (not just the top 10 displayed) to give the LLM better context for labeling.

        Args:
            examples: List of top-K activation example dicts with keys:
                - prefix_tokens: List[str] - Tokens before prime
                - prime_token: str - The token with maximum activation
                - suffix_tokens: List[str] - Tokens after prime
                - max_activation: float - Peak activation value
            neuron_index: Optional neuron index for fallback naming
            feature_id: Optional feature ID for fallback naming
            all_examples: Optional full list of all examples (for NLP analysis)
            nlp_analysis: Optional pre-computed NLP analysis results

        Returns:
            Dict with {"category": "...", "specific": "...", "description": "..."}
        """
        fallback_label = f"feature_{feature_id or neuron_index or 'unknown'}"

        if not examples:
            logger.warning("Empty examples, using fallback label")
            # `error` is what marks this a FAILURE. `category` is the judge's
            # semantic answer and cannot carry that: overloading it is what made
            # a crash indistinguishable from a verdict downstream.
            return {
                "category": "empty_features",
                "specific": fallback_label,
                "description": "",
                "error": "no activating examples were retrieved for this feature",
            }

        # Compute NLP analysis if not provided and we have all examples
        analysis_summary = None
        if nlp_analysis:
            analysis_summary = nlp_analysis.get("summary_for_prompt", "")
        elif all_examples and len(all_examples) > len(examples):
            try:
                nlp_service = NLPAnalysisService()
                analysis_result = nlp_service.analyze_feature(all_examples, feature_id or "unknown")
                analysis_summary = analysis_result.get("summary_for_prompt", "")
                logger.info(f"Computed NLP analysis for feature {feature_id} with {len(all_examples)} examples")
            except Exception as e:
                logger.warning(f"Failed to compute NLP analysis: {e}")

        # Build prompt with context examples and analysis
        prompt = self._build_prompt_from_examples(examples, feature_id=feature_id, analysis_summary=analysis_summary)

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
            ).to(self._input_device())

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
            # The exception text used to stop here, at a log line. It is the
            # only account of why a feature has no label, so it travels with
            # the result.
            return {
                "category": "error_feature",
                "specific": fallback_label,
                "description": "",
                "error": f"{type(e).__name__}: {e}",
            }

    def _build_prompt_from_examples(
        self,
        examples: List[Dict[str, Any]],
        feature_id: Optional[str] = None,
        analysis_summary: Optional[str] = None
    ) -> str:
        """
        Build user prompt from context-based activation examples.

        Args:
            examples: List of activation example dicts with prefix/prime/suffix tokens
            feature_id: Optional feature ID for context
            analysis_summary: Optional NLP analysis summary to include

        Returns:
            Formatted prompt string with examples
        """
        feature_label = feature_id or "this feature"

        # Start with analysis summary if available
        analysis_section = ""
        if analysis_summary:
            analysis_section = f"""
## STATISTICAL ANALYSIS OF ALL EXAMPLES:

{analysis_summary}

"""

        prompt = f"""Analyze sparse autoencoder feature {feature_label}.
{analysis_section}You are given some of the highest-activating examples for this feature. In each example, the main activating token(s) are wrapped in << >>.

Use the statistical analysis above (if provided) along with ALL of the examples, including their surrounding context, to infer the smallest semantic concept that explains why these tokens activate the same feature.

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
        #
        # THIS PATH WROTE `category="semantic"` WITH A PLACEHOLDER NAME. A parse
        # failure was therefore indistinguishable from a real semantic verdict
        # by ANY test — not even the `error_feature` string the other failure
        # paths at least left behind. It is a failure and now says so.
        logger.warning("Using fallback label parsing")
        return {
            "category": "uncategorized",
            "specific": fallback_label,
            "description": "",
            "error": "the judge's response could not be parsed as a label",
        }

    def batch_generate_labels(
        self,
        features_examples: List[List[Dict[str, Any]]],
        neuron_indices: Optional[List[int]] = None,
        feature_ids: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
        all_features_examples: Optional[List[List[Dict[str, Any]]]] = None,
        nlp_analyses: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate labels for multiple features efficiently using context examples.

        Loads model once and processes all features before unloading.

        Args:
            features_examples: List of example lists (top 10), one per feature
            neuron_indices: Optional list of neuron indices for fallback naming
            feature_ids: Optional list of feature IDs for fallback naming
            progress_callback: Optional callback(current, total) for progress updates
            all_features_examples: Optional full example lists (all 100) for NLP analysis
            nlp_analyses: Optional pre-computed NLP analyses, one per feature

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
                all_examples = all_features_examples[i] if all_features_examples else None
                nlp_analysis = nlp_analyses[i] if nlp_analyses else None

                label = self.generate_label(
                    examples=examples,
                    neuron_index=neuron_index,
                    feature_id=feature_id,
                    all_examples=all_examples,
                    nlp_analysis=nlp_analysis
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


def local_judge_required_mb(model_name: Optional[str]) -> Optional[float]:
    """A local judge's 4-bit footprint in MB, from a config already on disk; None when unknown.

    ``place_job`` can split a judge no single card holds only when it knows the
    judge's size — without one, Auto takes the freest card and the load then fails
    there. Read with ``local_files_only``: placement must not wait on the network,
    and a judge never downloaded has no size to give, so it is placed on one card
    exactly as before.
    """
    from src.ml.model_loader import QuantizationFormat, estimate_model_memory, estimate_parameter_count

    name = LocalLabelingService.resolve_model_name(model_name)
    try:
        config = AutoConfig.from_pretrained(name, trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - no config on disk is a normal state
        logger.info("No local config for judge %s, so it is placed without a size: %s", name, exc)
        return None
    params = estimate_parameter_count(config)
    if not params:
        return None
    return estimate_model_memory(params, QuantizationFormat.Q4) / (1024**2)
