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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_message: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 50,
        top_p: float = 0.9
    ):
        """
        Initialize OpenAI labeling service.

        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            model: Model identifier or full model name
            base_url: Optional base URL for OpenAI-compatible endpoints (e.g., Ollama, vLLM)
            system_message: Custom system message (overrides default)
            user_prompt_template: Custom user prompt template (overrides default, must contain {tokens_table})
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response (10-500)
            top_p: Nucleus sampling parameter (0.0-1.0)
        """
        # Set API key (not required for OpenAI-compatible endpoints)
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        if not self.api_key and not base_url:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize async client with optional base_url for OpenAI-compatible endpoints
        client_kwargs = {"api_key": self.api_key or "not-needed"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)

        # Resolve model name
        if model is None or model == "gpt4-mini":
            self.model = self.DEFAULT_MODEL
        elif model in self.ALTERNATIVE_MODELS:
            self.model = self.ALTERNATIVE_MODELS[model]
        else:
            self.model = model

        # Store prompt template configuration
        self.system_message = system_message
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        logger.info(f"Initialized OpenAI labeling service with model: {self.model}")
        logger.info(f"  Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Top P: {self.top_p}")
        if system_message:
            logger.info(f"  Using custom system message (length: {len(system_message)} chars)")
        if user_prompt_template:
            logger.info(f"  Using custom user prompt template (length: {len(user_prompt_template)} chars)")

    async def generate_label(
        self,
        token_stats: Dict[str, Dict[str, float]],
        top_k: int = 50,
        neuron_index: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Generate semantic label for a feature based on token statistics.

        Args:
            token_stats: Dict mapping token to {count, total_activation, max_activation}
            top_k: Number of top tokens to include in prompt
            neuron_index: Optional neuron index for fallback naming

        Returns:
            Dict with {"category": "broad_label", "specific": "precise_label"}
        """
        fallback_label = f"feature_{neuron_index}" if neuron_index is not None else "empty_feature"

        if not token_stats:
            logger.warning("Empty token stats, using fallback label")
            return {"category": "empty_features", "specific": fallback_label}

        # Sort tokens by total activation strength
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["total_activation"],
            reverse=True
        )[:top_k]

        if not sorted_tokens:
            return {"category": "no_activations", "specific": fallback_label}

        # Build prompt with token frequency table
        if self.user_prompt_template:
            # Use custom template
            prompt = self._build_prompt_from_template(sorted_tokens, neuron_index)
        else:
            # Use default prompt
            prompt = self._build_prompt(sorted_tokens)

        try:
            # Prepare system message (use custom or default)
            system_message = self.system_message or "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Provide both category and specific labels in JSON format."

            # Log API call details for debugging
            logger.info(f"🔍 OpenAI API Call Debug Info:")
            logger.info(f"  - Base URL: {self.client.base_url}")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Neuron Index: {neuron_index}")
            logger.info(f"  - Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Top P: {self.top_p}")
            logger.info(f"  - Prompt length: {len(prompt)} chars, Top tokens: {len(sorted_tokens)}")
            logger.info(f"\n📝 SYSTEM MESSAGE:\n{system_message}")
            logger.info(f"\n📝 USER PROMPT:\n{prompt}")

            # Call OpenAI API (new v1+ syntax)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )

            # Extract label from response (new v1+ syntax)
            label_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            logger.info(f"✅ API Response received (length: {len(label_text)} chars):")
            logger.info(f"📤 FULL RESPONSE:\n{label_text}")

            # Parse JSON response
            labels = self._parse_dual_label(label_text, fallback_label)

            logger.debug(f"Generated labels: category='{labels['category']}', specific='{labels['specific']}' from GPT response")
            return labels

        except RateLimitError as e:
            logger.warning(f"⚠️ OpenAI rate limit reached: {e}")
            return {"category": "rate_limited", "specific": fallback_label}

        except AuthenticationError as e:
            logger.error(f"❌ OpenAI authentication failed: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Error calling OpenAI API:")
            logger.error(f"   Base URL: {self.client.base_url}")
            logger.error(f"   Model: {self.model}")
            logger.error(f"   Error Type: {type(e).__name__}")
            logger.error(f"   Error Message: {e}", exc_info=True)
            return {"category": "error_feature", "specific": fallback_label}

    def _build_prompt(self, sorted_tokens: List[tuple]) -> str:
        """
        Build analysis prompt with token frequency table.

        Uses contrastive examples to encourage maximum specificity in labels.

        Args:
            sorted_tokens: List of (token, stats_dict) tuples sorted by activation

        Returns:
            Formatted prompt string
        """
        prompt = """You are labeling a sparse autoencoder feature. Provide BOTH a high-level category AND a specific interpretation.

INSTRUCTIONS:
Provide two labels:
1. CATEGORY: A broad, high-level grouping (for filtering/organizing)
2. SPECIFIC: The most precise interpretation possible (for understanding mechanism)

EXAMPLES:

Tokens: Trump, Trumps, Donald, MAGA, administration
→ category: "political_terms"
→ specific: "trump_mentions"

Tokens: Biden, Joe, Bidens, President, administration
→ category: "political_terms"
→ specific: "biden_administration"

Tokens: COVID, coronavirus, pandemic, vaccine, quarantine
→ category: "health_topics"
→ specific: "covid_pandemic"

Tokens: Elizabeth, Lizzie, Liz, Beth, Betty
→ category: "names"
→ specific: "elizabeth_variations"

Tokens: def, class, import, return, function
→ category: "code_keywords"
→ specific: "python_syntax"

Tokens: don, didn, wouldn, couldn, shouldn
→ category: "function_words"
→ specific: "negative_contractions"

Tokens: president, senator, congress, vote, bill
→ category: "political_terms"
→ specific: "political_institutions"

TOP TOKENS FOR THIS FEATURE:
"""

        for token, stats in sorted_tokens[:30]:  # Show top 30 for better context
            avg_act = stats["total_activation"] / stats["count"]
            token_display = repr(token)[:20].ljust(20)
            prompt += f"{token_display} | count={stats['count']:4d} | avg={avg_act:6.3f} | max={stats['max_activation']:6.3f}\n"

        prompt += """
DECISION TREE FOR SPECIFIC LABEL:
1. Is ONE entity/person dominant (70%+ tokens)? → Name it specifically
2. Is there a NARROW domain (60%+ tokens)? → Name the narrow domain
3. Is there a SPECIFIC pattern? → Name the pattern
4. Otherwise → Use a precise descriptor

Respond in JSON format:
{"category": "broad_category", "specific": "precise_interpretation"}

Both labels must be lowercase_with_underscores (1-3 words max each).
"""

        return prompt

    def _build_prompt_from_template(self, sorted_tokens: List[tuple], neuron_index: Optional[int] = None) -> str:
        """
        Build analysis prompt using custom template.

        Substitutes {tokens_table} placeholder with formatted token data.

        Args:
            sorted_tokens: List of (token, stats_dict) tuples sorted by activation
            neuron_index: Optional neuron index for context

        Returns:
            Formatted prompt string from template
        """
        # Build token table
        tokens_table = ""
        for token, stats in sorted_tokens[:30]:  # Show top 30 for better context
            avg_act = stats["total_activation"] / stats["count"]
            token_display = repr(token)[:20].ljust(20)
            tokens_table += f"{token_display} | count={stats['count']:4d} | avg={avg_act:6.3f} | max={stats['max_activation']:6.3f}\n"

        # Substitute placeholders in template
        prompt = self.user_prompt_template.replace("{tokens_table}", tokens_table)

        # Add optional placeholders if they exist in template
        if "{neuron_index}" in prompt:
            prompt = prompt.replace("{neuron_index}", str(neuron_index) if neuron_index is not None else "unknown")
        if "{layer_name}" in prompt:
            prompt = prompt.replace("{layer_name}", "unknown")  # Can be extended with actual layer info

        return prompt

    def _parse_dual_label(self, response: str, fallback_label: str) -> Dict[str, str]:
        """
        Parse JSON response containing category and specific labels.

        Args:
            response: Raw JSON response from GPT
            fallback_label: Fallback if parsing fails

        Returns:
            Dict with cleaned {"category": "...", "specific": "..."}
        """
        import json
        import re

        try:
            # Try to parse JSON
            data = json.loads(response)

            # Extract and clean both labels
            category = self._clean_label(data.get("category", "uncategorized"))
            specific = self._clean_label(data.get("specific", fallback_label))

            return {"category": category, "specific": specific}

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse dual label from response: {response[:100]}, error: {e}")

            # Try to extract from plain text if JSON parsing fails
            # Look for patterns like: category: "X", specific: "Y"
            category_match = re.search(r'category["\s:]+([a-z_]+)', response.lower())
            specific_match = re.search(r'specific["\s:]+([a-z_]+)', response.lower())

            category = self._clean_label(category_match.group(1)) if category_match else "uncategorized"
            specific = self._clean_label(specific_match.group(1)) if specific_match else fallback_label

            return {"category": category, "specific": specific}

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
    ) -> List[Dict[str, str]]:
        """
        Generate labels for multiple features with concurrent API calls.

        Args:
            features_token_stats: List of token stats dicts, one per feature
            neuron_indices: Optional list of neuron indices for fallback naming
            progress_callback: Optional callback(current, total) for progress updates
            batch_size: Number of concurrent API calls

        Returns:
            List of label dicts with {"category": "...", "specific": "..."} in same order as input
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
                    fallback_name = f"feature_{fallback_idx}" if fallback_idx is not None else "error_feature"
                    label = {"category": "error_feature", "specific": fallback_name}

                labels.append(label)

                # Log sample labels (every 100th label + first 5)
                feature_num = i + j + 1
                neuron_idx = batch_indices[j]
                if feature_num <= 5 or feature_num % 100 == 0:
                    logger.info(f"✨ Sample label #{feature_num}: neuron_{neuron_idx} = category:'{label['category']}', specific:'{label['specific']}'")

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
