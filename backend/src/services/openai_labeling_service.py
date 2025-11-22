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
        top_p: float = 0.9,
        filter_special: bool = True,
        filter_single_char: bool = True,
        filter_punctuation: bool = True,
        filter_numbers: bool = True,
        filter_fragments: bool = True,
        filter_stop_words: bool = False,
        save_requests_for_testing: bool = False,
        labeling_job_id: Optional[str] = None
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
            filter_special: Filter special tokens (<s>, </s>, etc.) from token analysis
            filter_single_char: Filter single character tokens from token analysis
            filter_punctuation: Filter pure punctuation tokens from token analysis
            filter_numbers: Filter pure numeric tokens from token analysis
            filter_fragments: Filter word fragments (BPE subwords) from token analysis
            filter_stop_words: Filter common stop words from token analysis
            save_requests_for_testing: Save API requests to tmp_api/ for testing and debugging
            labeling_job_id: Labeling job ID for organizing saved requests
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

        # Store token filtering configuration
        self.filter_special = filter_special
        self.filter_single_char = filter_single_char
        self.filter_punctuation = filter_punctuation
        self.filter_numbers = filter_numbers
        self.filter_fragments = filter_fragments
        self.filter_stop_words = filter_stop_words

        # Store debugging configuration
        self.save_requests_for_testing = save_requests_for_testing
        self.labeling_job_id = labeling_job_id
        self._request_dir = None  # Cached directory path for saved requests (created once per job)

        logger.info(f"Initialized OpenAI labeling service with model: {self.model}")
        logger.info(f"  Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Top P: {self.top_p}")
        if system_message:
            logger.info(f"  Using custom system message (length: {len(system_message)} chars)")
        if user_prompt_template:
            logger.info(f"  Using custom user prompt template (length: {len(user_prompt_template)} chars)")
            logger.info(f"  Token Filtering: special={filter_special}, fragments={filter_fragments}, stop_words={filter_stop_words}")

    def _save_request_for_testing(
        self,
        request_payload: Dict[str, Any],
        neuron_index: Optional[int] = None
    ) -> None:
        """
        Save API request to file for testing in Postman or cURL.

        Creates three files in tmp_api/{datetime}_{job_id}/
        - JSON file: Ready to import into Postman
        - Shell script: cURL command ready to execute
        - Postman collection: Import into Postman app

        Args:
            request_payload: The request payload dict
            neuron_index: Optional neuron index for filename
        """
        import json
        import os
        from pathlib import Path
        from datetime import datetime

        try:
            # Get application root (parent of backend/)
            app_root = Path(__file__).parent.parent.parent.parent

            # Create base tmp_api directory
            tmp_api_dir = app_root / "tmp_api"
            tmp_api_dir.mkdir(exist_ok=True)

            # Create subfolder ONCE per labeling job (reuse for all neurons)
            # Format: YYYYMMDD_HHMMSS_{job_id}
            if self._request_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                job_id_str = self.labeling_job_id or "unknown_job"
                folder_name = f"{timestamp}_{job_id_str}"
                request_dir = tmp_api_dir / folder_name
                request_dir.mkdir(exist_ok=True)
                # Cache for reuse across all neurons in this job
                self._request_dir = request_dir
                self._request_timestamp = timestamp
                self._request_folder_name = folder_name
                logger.info(f"📁 Created API request folder for labeling job: tmp_api/{folder_name}/")
            else:
                # Reuse cached values
                request_dir = self._request_dir
                timestamp = self._request_timestamp
                folder_name = self._request_folder_name

            # Create filename with folder name prefix for self-identification
            # Format: {folder_name}_neuron_{idx}.json (ties file to labeling job and SAE)
            neuron_str = f"neuron_{neuron_index}" if neuron_index is not None else "request"
            base_filename = request_dir / f"{folder_name}_{neuron_str}"

            # Determine endpoint URL
            base_url = str(self.client.base_url).rstrip('/')
            endpoint_url = f"{base_url}/chat/completions"

            # 1. Save JSON payload for Postman
            json_file = f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(request_payload, f, indent=2)

            # 2. Create cURL command
            curl_file = f"{base_filename}.sh"
            headers = []
            if self.api_key and self.api_key != "not-needed" and self.api_key != "dummy-key-not-required":
                headers.append(f"-H 'Authorization: Bearer {self.api_key}'")
            headers.append("-H 'Content-Type: application/json'")

            # Use just the filename (not full path) so script runs from within its folder
            filename_only = f"{folder_name}_{neuron_str}"

            curl_command = f"""#!/bin/bash
# OpenAI API Request - Generated {timestamp}
# Labeling Job ID: {self.labeling_job_id or 'N/A'}
# Neuron Index: {neuron_index if neuron_index is not None else 'N/A'}
# Base URL: {base_url}
# Model: {self.model}
# Folder: {folder_name}

curl -X POST '{endpoint_url}' \\
  {' '.join(headers)} \\
  -d @{filename_only}.json

# Alternative: Inline JSON (if you want to modify the request directly)
# curl -X POST '{endpoint_url}' \\
#   {' '.join(headers)} \\
#   -d '{json.dumps(request_payload)}'
"""

            with open(curl_file, 'w') as f:
                f.write(curl_command)

            # Make shell script executable
            import os
            os.chmod(curl_file, 0o755)

            # 3. Create Postman-ready collection
            postman_file = f"{base_filename}_postman.json"
            postman_collection = {
                "info": {
                    "name": f"OpenAI Labeling Request - {timestamp}",
                    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
                },
                "item": [
                    {
                        "name": f"Label Feature (Neuron {neuron_index})",
                        "request": {
                            "method": "POST",
                            "header": [
                                {
                                    "key": "Content-Type",
                                    "value": "application/json"
                                }
                            ],
                            "body": {
                                "mode": "raw",
                                "raw": json.dumps(request_payload, indent=2)
                            },
                            "url": {
                                "raw": endpoint_url,
                                "protocol": "https" if "https" in endpoint_url else "http",
                                "host": [endpoint_url.split("://")[1].split("/")[0]],
                                "path": endpoint_url.split("://")[1].split("/")[1:]
                            }
                        }
                    }
                ]
            }

            # Add Authorization header if API key exists
            if self.api_key and self.api_key not in ["not-needed", "dummy-key-not-required"]:
                postman_collection["item"][0]["request"]["header"].append({
                    "key": "Authorization",
                    "value": f"Bearer {self.api_key}",
                    "type": "text"
                })

            with open(postman_file, 'w') as f:
                json.dump(postman_collection, f, indent=2)

            logger.info(f"💾 Saved API request for testing:")
            logger.info(f"   Folder: tmp_api/{folder_name}/")
            logger.info(f"   JSON Payload: {json_file}")
            logger.info(f"   cURL Script: {curl_file}")
            logger.info(f"   Postman Collection: {postman_file}")
            logger.info(f"")
            logger.info(f"   Quick test: bash {curl_file}")
            logger.info(f"   Or import {postman_file} into Postman")

        except Exception as e:
            logger.warning(f"Failed to save request for testing: {e}")
            # Don't fail the actual labeling if saving fails
            pass

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

        # Sort tokens by frequency (count) instead of activation strength
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )  # No limit here - show all tokens

        # DEBUG: Log instance configuration
        logger.info(f"🔧 OpenAILabelingService filter configuration:")
        logger.info(f"   self.filter_stop_words = {self.filter_stop_words}")
        logger.info(f"   self.filter_special = {self.filter_special}")
        logger.info(f"   self.filter_fragments = {self.filter_fragments}")

        # Filter out junk tokens based on user configuration
        filtered_tokens = self._filter_junk_tokens(
            sorted_tokens,
            filter_special=self.filter_special,
            filter_single_char=self.filter_single_char,
            filter_punctuation=self.filter_punctuation,
            filter_numbers=self.filter_numbers,
            filter_fragments=self.filter_fragments,
            filter_stop_words=self.filter_stop_words
        )

        if not filtered_tokens:
            logger.warning(f"All tokens filtered as junk for neuron {neuron_index}, using fallback label")
            return {"category": "filtered_junk", "specific": fallback_label}

        # Build prompt with token frequency table (using filtered tokens)
        if self.user_prompt_template:
            # Use custom template
            prompt = self._build_prompt_from_template(filtered_tokens, neuron_index)
        else:
            # Use default prompt
            prompt = self._build_prompt(filtered_tokens)

        try:
            # Prepare system message (use custom or default)
            system_message = self.system_message or "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Provide both category and specific labels in JSON format."

            # Prepare request payload
            request_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }

            # Log API call details for debugging
            logger.info(f"🔍 OpenAI API Call Debug Info:")
            logger.info(f"  - Base URL: {self.client.base_url}")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Neuron Index: {neuron_index}")
            logger.info(f"  - Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Top P: {self.top_p}")
            logger.info(f"  - Tokens: {len(sorted_tokens)} total, {len(filtered_tokens)} after filtering ({len(sorted_tokens) - len(filtered_tokens)} junk removed)")
            logger.info(f"  - Prompt length: {len(prompt)} chars")
            logger.info(f"\n📝 SYSTEM MESSAGE:\n{system_message}")
            logger.info(f"\n📝 USER PROMPT:\n{prompt}")

            # Save request to file for Postman/cURL testing (if enabled)
            if self.save_requests_for_testing:
                self._save_request_for_testing(request_payload, neuron_index)

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
        NOTE: Tokens are already filtered by the caller, no need to filter again.

        Args:
            sorted_tokens: List of (token, stats_dict) tuples sorted by activation (already filtered)
            neuron_index: Optional neuron index for context

        Returns:
            Formatted prompt string from template
        """
        # Build token frequency table with the already-filtered tokens
        tokens_table = ""
        for token, stats in sorted_tokens:  # Use sorted_tokens directly (already filtered)
            # Clean token for display (remove SentencePiece underscore prefix)
            display_token = token.replace('▁', ' ').strip()
            if not display_token:
                display_token = token

            # Format: 'token'                                    → count times
            token_str = f"'{display_token}'"
            padded_token = token_str.ljust(42)
            tokens_table += f"{padded_token} → {stats['count']} {'time' if stats['count'] == 1 else 'times'}\n"

        if not tokens_table:
            tokens_table = "(No tokens found after filtering)"

        # Substitute placeholders in template
        prompt = self.user_prompt_template.replace("{tokens_table}", tokens_table)

        # Add optional placeholders if they exist in template
        if "{neuron_index}" in prompt:
            prompt = prompt.replace("{neuron_index}", str(neuron_index) if neuron_index is not None else "unknown")
        if "{layer_name}" in prompt:
            prompt = prompt.replace("{layer_name}", "unknown")  # Can be extended with actual layer info

        return prompt

    def _format_examples_block(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        feature_id: str,
        logit_effects: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format activation examples into a prompt-ready text block using LabelingContextFormatter.

        Dispatches to the appropriate formatter method based on template_type:
        - mistudio_context: miStudio Internal format (prefix <<prime>> suffix)
        - anthropic_logit: Anthropic Style format (with logit effects section)
        - eleutherai_detection: EleutherAI Detection format (test examples for scoring)

        Args:
            examples: List of example dicts with keys:
                - prefix_tokens: List of tokens before prime
                - prime_token: The token with maximum activation
                - suffix_tokens: List of tokens after prime
                - max_activation: Peak activation value for this example
            template_config: Dict with template configuration:
                - template_type: 'mistudio_context', 'anthropic_logit', or 'eleutherai_detection'
                - prime_token_marker: Marker format like '<<>>'
                - include_prefix: Whether to include prefix tokens
                - include_suffix: Whether to include suffix tokens
                - include_logit_effects: Whether to include logit effects section
                - top_promoted_tokens_count: Number of promoted tokens to show
                - top_suppressed_tokens_count: Number of suppressed tokens to show
            feature_id: Feature identifier for context
            logit_effects: Optional dict with 'top_promoted' and 'top_suppressed' token lists

        Returns:
            Formatted examples block string ready for prompt insertion
        """
        from src.services.labeling_context_formatter import LabelingContextFormatter

        template_type = template_config.get('template_type', 'mistudio_context')

        # Dispatch to appropriate formatter based on template type
        if template_type == 'anthropic_logit':
            return LabelingContextFormatter.format_anthropic_logit(
                examples=examples,
                logit_effects=logit_effects or {},
                template_config=template_config,
                feature_id=feature_id
            )
        elif template_type == 'eleutherai_detection':
            return LabelingContextFormatter.format_eleutherai_detection(
                examples=examples,
                template_config=template_config
            )
        else:  # Default to mistudio_context
            return LabelingContextFormatter.format_mistudio_context(
                examples=examples,
                template_config=template_config,
                feature_id=feature_id
            )

    def _build_user_prompt(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        user_prompt_template: str,
        feature_id: str,
        logit_effects: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build user prompt from template by replacing {examples_block} placeholder.

        This method orchestrates the full prompt building workflow:
        1. Formats examples using _format_examples_block
        2. Replaces {examples_block} placeholder in template
        3. Replaces other optional placeholders (feature_id, logit tokens)

        Args:
            examples: List of activation example dicts
            template_config: Template configuration dict
            user_prompt_template: User prompt template string with placeholders
            feature_id: Feature identifier for context
            logit_effects: Optional logit effects data for Anthropic template

        Returns:
            Fully formatted user prompt ready for API call
        """
        # Format examples block using context formatter
        examples_block = self._format_examples_block(
            examples=examples,
            template_config=template_config,
            feature_id=feature_id,
            logit_effects=logit_effects
        )

        # Start with the template
        prompt = user_prompt_template

        # Replace examples_block placeholder
        if '{examples_block}' in prompt:
            prompt = prompt.replace('{examples_block}', examples_block)

        # Replace feature_id placeholder
        if '{feature_id}' in prompt:
            prompt = prompt.replace('{feature_id}', feature_id)

        # Replace logit effects placeholders (for Anthropic template)
        if logit_effects and template_config.get('include_logit_effects', False):
            if '{top_promoted_tokens}' in prompt:
                promoted = logit_effects.get('top_promoted', [])
                promoted_str = ', '.join(promoted) if promoted else '(none)'
                prompt = prompt.replace('{top_promoted_tokens}', promoted_str)

            if '{top_suppressed_tokens}' in prompt:
                suppressed = logit_effects.get('top_suppressed', [])
                suppressed_str = ', '.join(suppressed) if suppressed else '(none)'
                prompt = prompt.replace('{top_suppressed_tokens}', suppressed_str)

        return prompt

    def _parse_dual_label(self, response: str, fallback_label: str) -> Dict[str, str]:
        """
        Parse JSON response containing category, specific labels, and description.

        Supports two formats:
        1. {"category": "...", "specific": "...", "description": "..."}
        2. {"category": "...", "label": "...", "description": "..."} (Custom_V1 format)

        Args:
            response: Raw JSON response from GPT
            fallback_label: Fallback if parsing fails

        Returns:
            Dict with cleaned {"category": "...", "specific": "...", "description": "..."}
        """
        import json
        import re

        try:
            # Clean markdown code blocks if present (common with Ollama/local models)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                # Remove markdown code fence
                lines = cleaned_response.split('\n')
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's just ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_response = '\n'.join(lines).strip()

            # Try to parse JSON
            data = json.loads(cleaned_response)

            # Extract category
            category = self._clean_label(data.get("category", "uncategorized"))

            # Extract specific label (accept both "specific" and "label" keys for compatibility)
            specific = data.get("specific") or data.get("label")
            if specific:
                specific = self._clean_label(specific)
            else:
                specific = fallback_label

            # Extract description (optional, not cleaned)
            description = data.get("description", "")
            if description:
                description = description.strip()

            return {"category": category, "specific": specific, "description": description}

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse dual label from response: {response[:100]}, error: {e}")

            # Try to extract from plain text if JSON parsing fails
            # Look for patterns like: category: "X", specific: "Y" or label: "Y"
            category_match = re.search(r'category["\s:]+([a-z_]+)', response.lower())
            specific_match = re.search(r'(?:specific|label)["\s:]+([a-z_]+)', response.lower())
            description_match = re.search(r'description["\s:]+["\']([^"\']+)["\']', response, re.IGNORECASE)

            category = self._clean_label(category_match.group(1)) if category_match else "uncategorized"
            specific = self._clean_label(specific_match.group(1)) if specific_match else fallback_label
            description = description_match.group(1).strip() if description_match else ""

            return {"category": category, "specific": specific, "description": description}

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

    def _filter_junk_tokens(
        self,
        sorted_tokens: List[tuple],
        filter_special: bool = True,
        filter_single_char: bool = True,
        filter_punctuation: bool = True,
        filter_numbers: bool = True,
        filter_fragments: bool = True,
        filter_stop_words: bool = False
    ) -> List[tuple]:
        """
        Filter out junk tokens based on configuration flags.

        Each filter can be independently enabled/disabled:
        - filter_special: Remove tokenization artifacts (##, Ġ, ▁, etc.)
        - filter_single_char: Remove single character tokens (except $ and %)
        - filter_punctuation: Remove punctuation-only tokens
        - filter_numbers: Remove pure digit tokens (0-9, 10, 2023, etc.)
        - filter_fragments: Remove word fragments (BPE subwords like 'tion', 'ing')
        - filter_stop_words: Remove high-frequency stop words (the, a, and, etc.)

        Args:
            sorted_tokens: List of (token, stats_dict) tuples
            filter_special: Remove tokenization artifacts
            filter_single_char: Remove single character tokens
            filter_punctuation: Remove punctuation-only tokens
            filter_numbers: Remove pure digit tokens
            filter_fragments: Remove word fragments (BPE subwords)
            filter_stop_words: Remove common stop words

        Returns:
            Filtered list of (token, stats_dict) tuples
        """
        import re
        import string

        # DEBUG: Log filter configuration
        logger.info(f"🔍 _filter_junk_tokens called with:")
        logger.info(f"   filter_stop_words={filter_stop_words}")
        logger.info(f"   Total input tokens: {len(sorted_tokens)}")
        if sorted_tokens:
            logger.info(f"   First 5 tokens: {[token for token, _ in sorted_tokens[:5]]}")

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

            # Skip empty or whitespace-only tokens (always filter these)
            if not token_stripped or token_stripped.isspace():
                continue

            # Apply filters based on configuration flags
            skip_token = False

            # Filter 1: Pure punctuation
            if filter_punctuation:
                if all(c in punctuation_set or c.isspace() for c in token_stripped):
                    skip_token = True

            # Filter 2: Tokenization artifacts (special tokens)
            if filter_special and not skip_token:
                # - WordPiece markers: ##word, word##
                # - Special whitespace: Ġ (GPT-2 style), ▁ (SentencePiece alone)
                # - BPE markers: </w>, <w>
                if ('##' in token_stripped or
                    'Ġ' in token_stripped or
                    token_stripped == '▁' or  # SentencePiece marker alone
                    token_stripped.startswith(('</w>', '<w>')) or
                    token_stripped.endswith(('</w>', '<w>'))):
                    skip_token = True

            # Filter 3: Single characters (except meaningful ones)
            if filter_single_char and not skip_token:
                # Handle both regular tokens and SentencePiece tokens (▁X)
                token_without_marker = token_stripped.lstrip().lstrip('▁')
                if len(token_without_marker) == 1 and token_without_marker not in {'$', '%', '€', '£', '¥'}:
                    skip_token = True

            # Filter 4: Pure digit tokens
            if filter_numbers and not skip_token:
                # Remove leading space/SentencePiece marker and check if rest is all digits
                token_no_marker = token_stripped.lstrip().lstrip('▁')
                if token_no_marker.isdigit():
                    skip_token = True

            # Filter 5: Word fragments (BPE subwords)
            if filter_fragments and not skip_token:
                # Common BPE fragment patterns (subword pieces)
                # These typically appear as: 'tion', 'ing', 'ed', 'ly', etc.
                token_clean = token_stripped.lstrip().lstrip('▁').lower()
                # Fragment patterns: starts/ends with common morphemes, or very short without vowels
                fragment_patterns = {
                    'tion', 'sion', 'ment', 'ness', 'less', 'ful', 'able', 'ible',
                    'ing', 'ed', 'er', 'est', 'ly', 'al', 'ous', 'ive', 'ic'
                }
                if token_clean in fragment_patterns:
                    skip_token = True

            # Filter 6: Stop words (high-frequency function words)
            if filter_stop_words and not skip_token:
                # Case-insensitive, removing quotes, spaces, and SentencePiece markers
                # Tokens can appear as: '" and"' or ' and' or '▁and' or 'and'
                token_clean = token_stripped.strip().strip('"').strip("'").lstrip('▁').strip().lower()
                if token_clean in stopwords:
                    logger.info(f"   🚫 Filtering stop word: '{token}' (original: '{token}', cleaned: '{token_clean}')")
                    skip_token = True

            # Keep this token if it passed all enabled filters
            if not skip_token:
                filtered.append((token, stats))

        # DEBUG: Log filtering results
        logger.info(f"   ✅ Filtering complete: {len(filtered)} tokens kept, {len(sorted_tokens) - len(filtered)} filtered out")
        if filtered:
            logger.info(f"   First 5 filtered tokens: {[token for token, _ in filtered[:5]]}")

        return filtered

    async def generate_label_from_examples(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        user_prompt_template: str,
        system_message: str,
        feature_id: str,
        logit_effects: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate semantic label for a feature using context-based examples.

        This is the new context-based labeling method that uses full activation examples
        with prefix/prime/suffix tokens instead of aggregated token statistics.

        Args:
            examples: List of top-K activation example dicts with keys:
                - prefix_tokens: List[str] - Tokens before prime
                - prime_token: str - The token with maximum activation
                - suffix_tokens: List[str] - Tokens after prime
                - max_activation: float - Peak activation value
            template_config: Dict with template configuration (from LabelingPromptTemplate)
            user_prompt_template: User prompt template string with {examples_block} placeholder
            system_message: System message for the LLM
            feature_id: Feature identifier for context
            logit_effects: Optional dict with 'top_promoted' and 'top_suppressed' token lists

        Returns:
            Dict with {"category": "...", "specific": "...", "description": "..."}
        """
        fallback_label = f"feature_{feature_id}"

        if not examples:
            logger.warning(f"Empty examples for feature {feature_id}, using fallback label")
            return {"category": "empty_features", "specific": fallback_label, "description": ""}

        try:
            # Build user prompt using the new _build_user_prompt method
            user_prompt = self._build_user_prompt(
                examples=examples,
                template_config=template_config,
                user_prompt_template=user_prompt_template,
                feature_id=feature_id,
                logit_effects=logit_effects
            )

            # Prepare request payload
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p
            }

            # Log API call details
            logger.info(f"🔍 OpenAI API Call for feature {feature_id}:")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Examples: {len(examples)} activation examples")
            logger.info(f"  - Prompt length: {len(user_prompt)} chars")
            logger.debug(f"\n📝 SYSTEM MESSAGE:\n{system_message}")
            logger.debug(f"\n📝 USER PROMPT:\n{user_prompt}")

            # Save request for testing if enabled
            if self.save_requests_for_testing:
                self._save_request_for_testing(request_payload, neuron_index=None)

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=request_payload["messages"],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )

            # Extract and parse response
            label_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            logger.info(f"✅ API Response received for feature {feature_id} (length: {len(label_text)} chars)")
            logger.debug(f"📤 FULL RESPONSE:\n{label_text}")

            # Parse JSON response
            labels = self._parse_dual_label(label_text, fallback_label)
            logger.debug(f"Generated labels for {feature_id}: category='{labels['category']}', specific='{labels['specific']}'")
            return labels

        except RateLimitError as e:
            logger.warning(f"⚠️ OpenAI rate limit for feature {feature_id}: {e}")
            return {"category": "rate_limited", "specific": fallback_label, "description": ""}

        except AuthenticationError as e:
            logger.error(f"❌ OpenAI authentication failed for feature {feature_id}: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Error calling OpenAI API for feature {feature_id}:")
            logger.error(f"   Error Type: {type(e).__name__}")
            logger.error(f"   Error Message: {e}", exc_info=True)
            return {"category": "error_feature", "specific": fallback_label, "description": ""}

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
