"""
OpenAI API-based feature labeling service.

This service uses OpenAI's GPT models to generate semantic labels
for SAE features. Provides high-quality, fast alternative to local models.

Updated for OpenAI Python library v1.0+
"""

from openai import AsyncOpenAI, OpenAIError, RateLimitError, AuthenticationError
from typing import List, Dict, Any, Optional
import logging
import asyncio
from src.core.config import settings

logger = logging.getLogger(__name__)


class OpenAILabelingService:
    """
    Service for generating feature labels using OpenAI API.

    Uses GPT-4o-mini for cost-effective, high-quality labeling.
    Provides fastest labeling option with excellent semantic understanding.

    Cost Analysis:
    - GPT-4o-mini: $0.150/1M input tokens, $0.600/1M output tokens
    - Per feature: ~500 input + ~5 output tokens = ~$0.0001 per feature
    - 16,384 features: ~$1.64 total cost
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    ALTERNATIVE_MODELS = {
        "gpt4-mini": "gpt-4o-mini",
        "gpt4": "gpt-4-turbo-preview",
        "gpt35": "gpt-3.5-turbo"
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI labeling service.

        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            model: Model identifier or full model name
        """
        # Set API key
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Resolve model name
        if model is None or model == "gpt4-mini":
            self.model = self.DEFAULT_MODEL
        elif model in self.ALTERNATIVE_MODELS:
            self.model = self.ALTERNATIVE_MODELS[model]
        else:
            self.model = model

        logger.info(f"Initialized OpenAI labeling service with model: {self.model}")

    async def generate_label(
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

        if not sorted_tokens:
            return f"feature_{neuron_index}" if neuron_index is not None else "no_activations"

        # Build prompt with token frequency table
        prompt = self._build_prompt(sorted_tokens)

        try:
            # Call OpenAI API (new v1+ syntax)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Respond with single-word or 2-3 word labels only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=15,  # Allow 2-3 words
                top_p=0.9
            )

            # Extract label from response (new v1+ syntax)
            label_text = response.choices[0].message.content.strip()

            # Clean and validate
            label = self._clean_label(label_text)

            logger.debug(f"Generated label: '{label}' from GPT response: '{label_text[:50]}'")
            return label

        except RateLimitError:
            logger.warning("OpenAI rate limit reached, using fallback")
            return f"feature_{neuron_index}" if neuron_index is not None else "rate_limited"

        except AuthenticationError:
            logger.error("OpenAI authentication failed - check API key")
            raise

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
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

    def _clean_label(self, response: str) -> str:
        """
        Clean and validate model response.

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

    async def batch_generate_labels(
        self,
        features_token_stats: List[Dict[str, Dict[str, float]]],
        neuron_indices: Optional[List[int]] = None,
        progress_callback: Optional[callable] = None,
        batch_size: int = 10
    ) -> List[str]:
        """
        Generate labels for multiple features with concurrent API calls.

        Args:
            features_token_stats: List of token stats dicts, one per feature
            neuron_indices: Optional list of neuron indices for fallback naming
            progress_callback: Optional callback(current, total) for progress updates
            batch_size: Number of concurrent API calls

        Returns:
            List of labels in same order as input
        """
        logger.info(f"Starting batch label generation for {len(features_token_stats)} features")

        labels = []
        total = len(features_token_stats)

        # Process in batches to respect rate limits
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_stats = features_token_stats[i:batch_end]
            batch_indices = neuron_indices[i:batch_end] if neuron_indices else [None] * len(batch_stats)

            # Create concurrent tasks
            tasks = [
                self.generate_label(stats, neuron_index=idx)
                for stats, idx in zip(batch_stats, batch_indices)
            ]

            # Execute batch concurrently
            batch_labels = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            for j, label in enumerate(batch_labels):
                if isinstance(label, Exception):
                    logger.error(f"Error generating label for feature {i+j}: {label}")
                    fallback_idx = batch_indices[j]
                    label = f"feature_{fallback_idx}" if fallback_idx is not None else "error_feature"

                labels.append(label)

            # Progress updates
            completed = min(batch_end, total)
            logger.info(f"Labeled {completed}/{total} features")
            if progress_callback:
                progress_callback(completed, total)

            # Small delay between batches to avoid rate limits
            if batch_end < total:
                await asyncio.sleep(0.5)

        logger.info(f"Batch labeling complete. Generated {len(labels)} labels")
        return labels
