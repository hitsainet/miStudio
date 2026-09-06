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
import re
from src.core.config import settings
from src.services.nlp_analysis_service import NLPAnalysisService

logger = logging.getLogger(__name__)

# Appended to every labeling system message, including custom templates.
#
# Generation cost is linear in tokens emitted. An unconstrained instruct model
# narrates before answering ("The provided examples appear to be a repeating
# sequence of tokens: ...") — measured at 205 output tokens / 11.7s against
# granite-4.1-8b, versus 99 tokens / 5.8s with this directive. The prose was
# also what broke the JSON parser, so an entire run silently produced
# category='uncategorized' with empty descriptions while progress advanced.
#
# Note miLLM ignores the OpenAI `response_format` parameter (verified: identical
# output with and without), so the instruction has to live in the prompt.
JSON_ONLY_DIRECTIVE = (
    "\n\nCRITICAL OUTPUT RULE: Respond with ONE JSON object and NOTHING else. "
    "No preamble, no reasoning, no explanation, no markdown fences. "
    "Start your reply with '{' and end it with '}'."
)


def _enforce_json_only(system_message: str) -> str:
    """Append the JSON-only directive unless it is already present."""
    if not system_message:
        return JSON_ONLY_DIRECTIVE.strip()
    if "CRITICAL OUTPUT RULE" in system_message:
        return system_message
    return system_message + JSON_ONLY_DIRECTIVE




def _clean_optional(value: Any) -> Optional[str]:
    """Normalise a self-assessment field, or None when the template omits it.

    Module-level on purpose: `_parse_dual_label` is borrowed by tests with a
    minimal stand-in object, and an instance method here would force every such
    caller to know about a helper that has nothing to do with the instance.

    Deliberately NOT run through `_clean_label`, which lowercases and strips to
    a snake_case identifier — that would turn "7/10" into "710".
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# The model's own reported evidence, made BINDING.
#
# `fit_count` ("N/10") and the `uninterpretable` category were both advisory:
# the template asked the model to refuse when fewer than half its examples fit,
# the parser read the number off the wire, and then nothing compared it to
# anything. Self-policing failed exactly where it mattered — on a run over 390
# features, 107 got the single label `proper_noun_entities` and 49 of 120
# declared-uninterpretable features still carried a confident name.
#
# These two functions are the enforcement the prompt could only request.
MIN_FIT_RATIO = 0.5
REFUSAL_LABEL = "uninterpretable"
# Values a model emits meaning "no label"; each became a literal feature name.
_NULL_LABELS = {"none", "null", "n/a", "na", "unknown", "", "uninterpretable"}


def _fit_ratio(fit_count: Optional[str]) -> Optional[float]:
    """Parse "N/10" to a ratio, or None when it cannot be read.

    None is NOT a refusal. Templates predating `fit_count` never emit it, and
    forcing their every label to uninterpretable would destroy working
    behaviour. Absent evidence leaves the model's own verdict standing; only
    evidence that CONTRADICTS the verdict overrides it.
    """
    if not fit_count:
        return None
    m = re.search(r'(\d+)\s*/\s*(\d+)', str(fit_count))
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    if den <= 0:
        return None
    return num / den


def _enforce_refusal(label: Dict[str, str]) -> Dict[str, str]:
    """Make a declared refusal, and a self-reported poor fit, actually stick.

    Two guards:

    1. If the model reports that fewer than half its examples fit, the label is
       not supported by the evidence the model itself cited. It becomes a
       refusal whatever the model chose to call it.
    2. If the category says uninterpretable, the NAME must say so too.
       Downstream — search, the feature browser, detection scoring — reads
       `specific`, not `category`. A row saying "uninterpretable /
       proper_noun_entities" surfaces everywhere as a confident claim, which
       launders a refusal into an assertion and is worse than either answer
       alone. One such row's own description read "without a unifying theme".

    The original verdict is preserved in `fit_count`/`confidence`, so nothing
    is lost - only the authoritative fields are corrected.
    """
    ratio = _fit_ratio(label.get("fit_count"))
    if ratio is not None and ratio < MIN_FIT_RATIO:
        label["category"] = REFUSAL_LABEL
        label["specific"] = REFUSAL_LABEL
        return label

    if (label.get("category") or "").strip().lower() == REFUSAL_LABEL:
        label["specific"] = REFUSAL_LABEL
        return label

    # A model that answers the "no label" question in words rather than by
    # category still must not have those words stored as a feature name.
    if (label.get("specific") or "").strip().lower() in _NULL_LABELS:
        label["category"] = REFUSAL_LABEL
        label["specific"] = REFUSAL_LABEL
    return label


class BatchUnsupportedError(RuntimeError):
    """The server did not serve the request as a batch.

    Raised when the X-miLLM-Batch capability header is missing or disagrees
    with the batch that was sent. Both miLLM request schemas are
    extra="ignore", so a server predating the extension ACCEPTS `extra_messages`
    and returns a single choice with no error at all — silently labeling one
    feature out of every eight and leaving the rest to be filled with fallback
    labels. This exception is what turns that silence into a serial fallback.
    """


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
        max_tokens: int = 300,
        top_p: float = 0.9,
        timeout: float = 120.0,
        filter_special: bool = True,
        filter_single_char: bool = True,
        filter_punctuation: bool = True,
        filter_numbers: bool = True,
        filter_fragments: bool = True,
        filter_stop_words: bool = False,
        save_requests_for_testing: bool = False,
        export_format: str = "both",
        labeling_job_id: Optional[str] = None,
        save_poor_quality_labels: bool = False,
        poor_quality_sample_rate: float = 1.0,
        save_requests_sample_rate: float = 1.0,
        chat_template_kwargs: Optional[dict] = None,
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
            timeout: API request timeout in seconds (default: 120.0, max: 600.0)
            filter_special: Filter special tokens (<s>, </s>, etc.) from token analysis
            filter_single_char: Filter single character tokens from token analysis
            filter_punctuation: Filter pure punctuation tokens from token analysis
            filter_numbers: Filter pure numeric tokens from token analysis
            filter_fragments: Filter word fragments (BPE subwords) from token analysis
            filter_stop_words: Filter common stop words from token analysis
            save_requests_for_testing: Save API requests to tmp_api/ for testing and debugging
            export_format: Format for saved requests: 'postman', 'curl', or 'both' (default: 'both')
            labeling_job_id: Labeling job ID for organizing saved requests
        """
        # Set API key (not required for OpenAI-compatible endpoints)
        self.api_key = api_key or getattr(settings, 'openai_api_key', None)
        if not self.api_key and not base_url:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize async client with optional base_url and timeout for OpenAI-compatible endpoints
        # Limit connection pool to avoid "Too many open files" errors
        import httpx
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        client_kwargs = {
            "api_key": self.api_key or "not-needed",
            "http_client": http_client,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self._http_client = http_client  # Keep reference for cleanup
        self.timeout = timeout
        self._api_semaphore = asyncio.Semaphore(10)  # Max 10 concurrent API calls

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

        # Chat-template variables forwarded to the server (miLLM extension).
        #
        # Defaults to disabling reasoning. Labeling wants a JSON object, not
        # deliberation: granite-4.2-8b's template sets enable_thinking=True
        # when undefined and its generation prompt ends with an OPEN <think>
        # tag, so the model resumes inside a reasoning block. The opening tag
        # is never in the completion, which means _strip_think() cannot see it
        # and the reasoning lands in the parsed label as untagged prose.
        #
        # Safe to send unconditionally: a template that does not reference the
        # variable ignores it (verified 2026-09-02 -- gemma-4-12B-it accepts it
        # with the rendered prompt byte-identical, granite-4.2-8b acts on it).
        # Pass {} to send nothing at all.
        self.chat_template_kwargs = (
            {"enable_thinking": False}
            if chat_template_kwargs is None
            else dict(chat_template_kwargs)
        )

        # Store token filtering configuration
        self.filter_special = filter_special
        self.filter_single_char = filter_single_char
        self.filter_punctuation = filter_punctuation
        self.filter_numbers = filter_numbers
        self.filter_fragments = filter_fragments
        self.filter_stop_words = filter_stop_words

        # Store debugging configuration
        self.save_requests_for_testing = save_requests_for_testing
        self.export_format = export_format
        self.labeling_job_id = labeling_job_id
        self._request_dir = None  # Cached directory path for saved requests (created once per job)

        # Store poor quality detection configuration
        self.save_poor_quality_labels = save_poor_quality_labels
        self.poor_quality_sample_rate = poor_quality_sample_rate

        # Store request sampling configuration
        self.save_requests_sample_rate = save_requests_sample_rate

        logger.info(f"Initialized OpenAI labeling service with model: {self.model}")
        logger.info(f"  Temperature: {self.temperature}, Max Tokens: {self.max_tokens}, Top P: {self.top_p}, Timeout: {self.timeout}s")
        if system_message:
            logger.info(f"  Using custom system message (length: {len(system_message)} chars)")
        if user_prompt_template:
            logger.info(f"  Using custom user prompt template (length: {len(user_prompt_template)} chars)")
            logger.info(f"  Token Filtering: special={filter_special}, fragments={filter_fragments}, stop_words={filter_stop_words}")

    def close(self):
        """Close the underlying HTTP client to release file descriptors."""
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If there's a running loop, schedule close as a task
                loop.create_task(self._http_client.aclose())
            except RuntimeError:
                # No running loop — run synchronously
                asyncio.run(self._http_client.aclose())
            logger.info("Closed OpenAI labeling service HTTP client")
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")

    async def _call_openai(self, messages: list, **kwargs) -> Any:
        """
        Call OpenAI API with automatic fallback for unsupported parameters
        and retry logic for transient connection errors.

        Some models (o-series reasoning models) reject temperature, top_p, etc.
        On BadRequestError mentioning 'unsupported', retry without those params.
        Uses a semaphore to limit concurrent connections and avoid fd exhaustion.
        Retries up to 3 times on APIConnectionError with exponential backoff.
        """
        from openai import BadRequestError, APIConnectionError

        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
            **kwargs,
        }
        if self.chat_template_kwargs:
            extra = dict(call_kwargs.get("extra_body") or {})
            extra.setdefault("chat_template_kwargs", self.chat_template_kwargs)
            call_kwargs["extra_body"] = extra

        # Ask for JSON natively when the server supports it. Generation cost is
        # linear in tokens emitted, and an unconstrained model narrates before
        # answering ("The provided examples all share a common pattern: ...") —
        # measured at ~204 tokens where ~40 would do, i.e. most of the latency
        # is spent producing prose that then breaks the parser.
        # Unsupported servers raise BadRequestError and the retry below drops it.
        call_kwargs.setdefault("response_format", {"type": "json_object"})

        max_retries = 3
        async with self._api_semaphore:
            for attempt in range(max_retries + 1):
                try:
                    return await self.client.chat.completions.create(**call_kwargs)
                except APIConnectionError as e:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning(
                            f"Connection error (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Connection error after {max_retries + 1} attempts: {e}")
                        raise
                except BadRequestError as e:
                    # Server does not implement response_format — drop it and
                    # retry once rather than failing the whole label.
                    if "response_format" in call_kwargs and (
                        "response_format" in str(e).lower()
                        or "unsupported" in str(e).lower()
                    ):
                        logger.info(
                            "Server rejected response_format; retrying without it"
                        )
                        call_kwargs.pop("response_format", None)
                        continue
                    error_msg = str(e).lower()
                    if "unsupported" not in error_msg:
                        raise

                    # Retry without temperature and top_p for reasoning models
                    logger.warning(
                        f"Model {self.model} rejected sampling params, retrying without temperature/top_p"
                    )
                    call_kwargs.pop("temperature", None)
                    call_kwargs.pop("top_p", None)
                    return await self.client.chat.completions.create(**call_kwargs)

    def _save_request_for_testing(
        self,
        request_payload: Dict[str, Any],
        neuron_index: Optional[int] = None
    ) -> None:
        """
        Save API request to file for testing in Postman or cURL.

        Creates files in tmp_api/{datetime}_{job_id}/ based on export_format:
        - JSON file: Request payload (always created for reference)
        - cURL file (*.curl.txt): cURL command (if export_format='curl' or 'both')
        - Postman collection (*_postman.json): Import into Postman (if export_format='postman' or 'both')

        Args:
            request_payload: The request payload dict
            neuron_index: Optional neuron index for filename
        """
        import json
        import os
        from pathlib import Path
        from datetime import datetime

        try:
            # Use settings.data_dir for writable volume (works in Docker/k8s)
            from src.core.config import settings
            tmp_api_dir = settings.data_dir / "tmp_api"
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
                logger.info(f"📁 Created API request folder for labeling job: {request_dir}/")
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

            # 1. Always save JSON payload (needed for all formats)
            json_file = f"{base_filename}.json"
            with open(json_file, 'w') as f:
                json.dump(request_payload, f, indent=2)

            files_created = [f"JSON: {json_file}"]

            # 2. Conditionally create cURL command file (text file, not shell script)
            if self.export_format in ["curl", "both"]:
                curl_file = f"{base_filename}.curl.txt"
                headers = []
                if self.api_key and self.api_key != "not-needed" and self.api_key != "dummy-key-not-required":
                    # NEVER write the real bearer token to disk — these files are
                    # debug artifacts that get checked into bug reports / shared.
                    # Replace with a placeholder the user substitutes manually.
                    headers.append("-H \"Authorization: Bearer $OPENAI_API_KEY\"")
                headers.append("-H 'Content-Type: application/json'")

                # Use just the filename (not full path) for portability
                filename_only = f"{folder_name}_{neuron_str}"

                curl_command = f"""# OpenAI API Request - Generated {timestamp}
# Labeling Job ID: {self.labeling_job_id or 'N/A'}
# Neuron Index: {neuron_index if neuron_index is not None else 'N/A'}
# Base URL: {base_url}
# Model: {self.model}
# Folder: {folder_name}

# cURL command (copy and paste into terminal from within the tmp_api/{folder_name}/ directory):
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

                files_created.append(f"cURL: {curl_file}")

            # 3. Conditionally create Postman collection
            if self.export_format in ["postman", "both"]:
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
                        # NEVER the real token — same rule as the curl branch above, which
                        # was hardened and this one was not (MIS-E2E-072). These are
                        # debug artifacts that get attached to bug reports, and
                        # export_format defaults to "both", so the DEFAULT path wrote
                        # the operator's key to disk once per feature labelled.
                        # {{OPENAI_API_KEY}} is a Postman variable the user fills in.
                        "value": "Bearer {{OPENAI_API_KEY}}",
                        "type": "text"
                    })

                with open(postman_file, 'w') as f:
                    json.dump(postman_collection, f, indent=2)

                files_created.append(f"Postman: {postman_file}")

            # Log files created based on export format
            logger.info(f"💾 Saved API request for testing (format: {self.export_format}):")
            logger.info(f"   Folder: {request_dir}/")
            for file_info in files_created:
                logger.info(f"   {file_info}")

        except Exception as e:
            logger.warning(f"Failed to save request for testing: {e}")
            # Don't fail the actual labeling if saving fails
            pass

    def _save_response_for_testing(
        self,
        response: Any,
        neuron_index: Optional[int] = None,
        elapsed_time: Optional[float] = None
    ) -> None:
        """
        Save API response to file for testing and debugging.

        Creates response file in tmp_api/{datetime}_{job_id}/:
        - Response JSON: Full API response with metadata

        Args:
            response: The API response object (ChatCompletion)
            neuron_index: Optional neuron index for filename
            elapsed_time: Optional API call elapsed time in seconds
        """
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            # Only save if we have a request directory (created by _save_request_for_testing)
            if self._request_dir is None:
                return

            request_dir = self._request_dir
            folder_name = self._request_folder_name
            neuron_str = f"neuron_{neuron_index}" if neuron_index is not None else "request"
            base_filename = request_dir / f"{folder_name}_{neuron_str}"

            # Save response
            response_file = f"{base_filename}_response.json"

            # Convert response to dict (handles both dict and object types)
            if hasattr(response, 'model_dump'):
                # Pydantic v2 model
                response_data = response.model_dump()
            elif hasattr(response, 'dict'):
                # Pydantic v1 model
                response_data = response.dict()
            elif isinstance(response, dict):
                response_data = response
            else:
                # Fallback: convert to dict using vars()
                response_data = vars(response) if hasattr(response, '__dict__') else str(response)

            # Build response metadata
            response_metadata = {
                "timestamp": datetime.now().isoformat(),
                "labeling_job_id": self.labeling_job_id or "N/A",
                "neuron_index": neuron_index if neuron_index is not None else "N/A",
                "model": self.model,
                "elapsed_time_seconds": round(elapsed_time, 3) if elapsed_time is not None else None,
                "response": response_data
            }

            with open(response_file, 'w') as f:
                json.dump(response_metadata, f, indent=2)

            logger.info(f"💾 Saved API response: {response_file}")

        except Exception as e:
            logger.warning(f"Failed to save response for testing: {e}")
            # Don't fail the actual labeling if saving fails
            pass

    def is_poor_quality_label(self, labels: Dict[str, str]) -> bool:
        """
        Detect if a label is poor quality (ineffective).

        Poor quality indicators:
        - Contains "uncategorized", "unknown", "unclear", "generic", "empty", "other"
        - Very short (< 3 characters)
        - Contains only generic words like "feature", "pattern", "text", "token"

        Args:
            labels: Dict with "category" and "specific" keys

        Returns:
            True if label is poor quality, False otherwise
        """
        # Extract labels (case-insensitive check)
        category = labels.get("category", "").lower()
        specific = labels.get("specific", "").lower()

        # List of poor quality keywords
        poor_quality_keywords = [
            "uncategorized",
            "unknown",
            "unclear",
            "generic",
            "empty",
            "other",
            "n/a",
            "none",
            "error",
            "fallback",
            "default",
        ]

        # Check if category or specific contains poor quality keywords
        for keyword in poor_quality_keywords:
            if keyword in category or keyword in specific:
                return True

        # Check if labels are too short (likely meaningless)
        if len(category) < 3 or len(specific) < 3:
            return True

        # Check if labels are too generic (only common words)
        generic_only_words = ["feature", "pattern", "text", "token", "word", "words"]
        if category in generic_only_words or specific in generic_only_words:
            return True

        return False

    def _save_poor_quality_debug(
        self,
        labels: Dict[str, str],
        token_stats: Dict[str, Dict[str, float]],
        neuron_index: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save debug files for poor quality labels.

        Creates three separate files in tmp_api/{datetime}_{job_id}/:
        - Postman collection: *_postman.json (import directly into Postman)
        - Response data: *_response.json (API response for analysis)
        - Metadata: *_metadata.json (labels, token stats, quality indicators)

        Args:
            labels: The poor quality labels that were generated
            token_stats: Token statistics used for labeling
            neuron_index: Optional neuron index for filename
            response_data: Optional API response data
            request_payload: Optional API request payload for debugging
        """
        import json
        import random
        from datetime import datetime
        from pathlib import Path

        try:
            # Only save if enabled
            if not self.save_poor_quality_labels:
                return

            # Apply sampling rate
            if random.random() > self.poor_quality_sample_rate:
                logger.debug(f"Skipping poor quality save for neuron {neuron_index} (sample rate: {self.poor_quality_sample_rate})")
                return

            # Create directory if it doesn't exist (use settings.data_dir for writable volume)
            if self._request_dir is None:
                from src.core.config import settings
                tmp_api_dir = settings.data_dir / "tmp_api"
                tmp_api_dir.mkdir(exist_ok=True)

                # Create subfolder with timestamp and job ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"{timestamp}_{self.labeling_job_id}" if self.labeling_job_id else timestamp
                request_dir = tmp_api_dir / folder_name
                request_dir.mkdir(parents=True, exist_ok=True)
                self._request_dir = request_dir
                self._request_folder_name = folder_name

            request_dir = self._request_dir
            folder_name = self._request_folder_name
            neuron_str = f"neuron_{neuron_index}" if neuron_index is not None else "request"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_filename = request_dir / f"{folder_name}_{neuron_str}_poor_quality_{timestamp_str}"

            # 1. Save Postman collection (if request_payload provided)
            if request_payload:
                postman_file = f"{base_filename}_postman.json"

                # Determine endpoint URL
                base_url = str(self.client.base_url).rstrip('/')
                endpoint_url = f"{base_url}/chat/completions"

                postman_collection = {
                    "info": {
                        "name": f"Poor Quality Label Debug - {timestamp_str}",
                        "description": f"Poor quality label for neuron {neuron_index if neuron_index is not None else 'N/A'}. Category: {labels.get('category', 'N/A')}, Specific: {labels.get('specific', 'N/A')}",
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
                        # NEVER the real token — same rule as the curl branch above, which
                        # was hardened and this one was not (MIS-E2E-072). These are
                        # debug artifacts that get attached to bug reports, and
                        # export_format defaults to "both", so the DEFAULT path wrote
                        # the operator's key to disk once per feature labelled.
                        # {{OPENAI_API_KEY}} is a Postman variable the user fills in.
                        "value": "Bearer {{OPENAI_API_KEY}}",
                        "type": "text"
                    })

                with open(postman_file, 'w') as f:
                    json.dump(postman_collection, f, indent=2)

                logger.info(f"💾 Saved Postman collection: {postman_file}")

            # 2. Save response data (if response_data provided)
            if response_data:
                response_file = f"{base_filename}_response.json"
                with open(response_file, 'w') as f:
                    json.dump(response_data, f, indent=2)
                logger.info(f"💾 Saved response data: {response_file}")

            # 3. Save metadata (labels, token stats, quality indicators)
            metadata_file = f"{base_filename}_metadata.json"
            debug_metadata = {
                "timestamp": datetime.now().isoformat(),
                "labeling_job_id": self.labeling_job_id or "N/A",
                "neuron_index": neuron_index if neuron_index is not None else "N/A",
                "model": self.model,
                "labels": labels,
                "quality_issue": "Poor quality label detected",
                "token_stats_count": len(token_stats),
                "top_tokens": [
                    {
                        "token": token,
                        "count": stats["count"],
                        "avg_activation": stats["total_activation"] / stats["count"],
                        "max_activation": stats["max_activation"]
                    }
                    for token, stats in sorted(
                        token_stats.items(),
                        key=lambda x: x[1]["count"],
                        reverse=True
                    )[:20]  # Top 20 tokens
                ]
            }

            with open(metadata_file, 'w') as f:
                json.dump(debug_metadata, f, indent=2)

            logger.info(f"💾 Saved metadata: {metadata_file}")
            logger.info(f"📦 Poor quality debug files saved with base name: {base_filename.name}")

        except Exception as e:
            logger.warning(f"Failed to save poor quality debug files: {e}")
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
            system_message = _enforce_json_only(
                self.system_message
                or "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Provide both category and specific labels in JSON format."
            )

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
                "max_completion_tokens": self.max_tokens,
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

            # Save request to file for Postman/cURL testing (if enabled and sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_request_for_testing(request_payload, neuron_index)
                else:
                    logger.debug(f"Skipping request save due to sample rate: {self.save_requests_sample_rate}")

            # Call OpenAI API (with automatic fallback for reasoning models)
            import time
            start_time = time.time()
            response = await self._call_openai(
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            elapsed_time = time.time() - start_time

            # Save response to file for testing (if enabled and sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_response_for_testing(response, neuron_index, elapsed_time)

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
        logit_effects: Optional[Dict[str, Any]] = None,
        negative_examples: Optional[List[Dict[str, Any]]] = None
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
            negative_examples: Optional list of low-activation examples for contrastive learning

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
                feature_id=feature_id,
                negative_examples=negative_examples
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
                feature_id=feature_id,
                negative_examples=negative_examples
            )

    def _build_user_prompt(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        user_prompt_template: str,
        feature_id: str,
        logit_effects: Optional[Dict[str, Any]] = None,
        negative_examples: Optional[List[Dict[str, Any]]] = None,
        analysis_summary: Optional[str] = None
    ) -> str:
        """
        Build user prompt from template by replacing {examples_block} placeholder.

        This method orchestrates the full prompt building workflow:
        1. Formats examples using _format_examples_block
        2. Replaces {examples_block} placeholder in template
        3. Replaces other optional placeholders (feature_id, logit tokens, analysis)

        Args:
            examples: List of activation example dicts
            template_config: Template configuration dict
            user_prompt_template: User prompt template string with placeholders
            feature_id: Feature identifier for context
            logit_effects: Optional logit effects data for Anthropic template
            negative_examples: Optional list of low-activation examples for contrastive learning
            analysis_summary: Optional NLP analysis summary to include

        Returns:
            Fully formatted user prompt ready for API call
        """
        # Format examples block using context formatter
        examples_block = self._format_examples_block(
            examples=examples,
            template_config=template_config,
            feature_id=feature_id,
            logit_effects=logit_effects,
            negative_examples=negative_examples
        )

        # Start with the template (fall back to default if None)
        prompt = user_prompt_template or (
            "Analyze the following activation examples for this SAE feature and provide a semantic label.\n\n"
            "{analysis_block}"
            "{examples_block}"
        )

        # Insert NLP analysis summary before examples_block if available
        # This provides statistical context about ALL examples, not just the displayed ones
        if analysis_summary:
            analysis_section = f"""## STATISTICAL ANALYSIS OF ALL EXAMPLES:

{analysis_summary}

"""
            # If template has {analysis_block} placeholder, use it
            if '{analysis_block}' in prompt:
                prompt = prompt.replace('{analysis_block}', analysis_section)
            # Otherwise, prepend to examples_block
            else:
                examples_block = analysis_section + examples_block

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

        # Clean up unused placeholders
        if '{analysis_block}' in prompt:
            prompt = prompt.replace('{analysis_block}', '')

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
            # Strip thinking from reasoning models. Three shapes, not one:
            #
            #  a) <think>…</think>{answer}   - the model emitted both tags
            #  b) <think>…                   - truncated by max_tokens, no close
            #  c) …reasoning…</think>{answer} - NO OPENING TAG
            #
            # (c) is the one this missed, and it is not exotic: LFM2.5-2.6B's
            # chat template appends "<|im_start|>assistant\n<think>" whenever
            # add_generation_prompt is set, which every chat-completions server
            # does. The opener therefore lives in the PROMPT and is never echoed,
            # so the reply begins with bare reasoning and carries only the
            # CLOSING tag. A pattern anchored on <think> matched nothing, the
            # answer after </think> was discarded, and a capable model looked
            # like it had returned prose. Take everything after the LAST closing
            # tag when there is no opener.
            cleaned_response = response.strip()
            think_pattern = re.compile(r'<think>.*?</think>\s*', re.DOTALL)
            cleaned_response = think_pattern.sub('', cleaned_response).strip()
            if '</think>' in cleaned_response and '<think>' not in cleaned_response:
                cleaned_response = cleaned_response.rsplit('</think>', 1)[1].strip()
            # Handle unclosed <think> tag (response truncated before </think>)
            if cleaned_response.startswith('<think>'):
                cleaned_response = ''  # Entire response was thinking - will fall through to fallback

            # Clean markdown code blocks if present (common with Ollama/local models)
            if cleaned_response.startswith("```"):
                # Remove markdown code fence
                lines = cleaned_response.split('\n')
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's just ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned_response = '\n'.join(lines).strip()

            # Try to parse JSON (use raw_decode to handle extra text after JSON)
            try:
                data = json.loads(cleaned_response)
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                try:
                    data, _ = decoder.raw_decode(cleaned_response)
                except json.JSONDecodeError:
                    # raw_decode only works when JSON starts at position 0, so a
                    # model that narrates BEFORE answering ("The provided examples
                    # all share a common pattern: ... {json}") fails here even
                    # though the JSON is present and valid.
                    #
                    # Reasoning-style models do this constantly — it is how a
                    # whole labeling run silently produced category='uncategorized'
                    # with empty descriptions while the progress counter advanced.
                    # Scan forward to each '{' and take the first object that
                    # decodes.
                    data = None
                    for idx, ch in enumerate(cleaned_response):
                        if ch != '{':
                            continue
                        try:
                            data, _ = decoder.raw_decode(cleaned_response[idx:])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(data, dict):
                            break
                        data = None
                    if data is None:
                        raise

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

            # Carry the model's SELF-ASSESSMENT through.
            #
            # Templates have long asked for `fit_count` ("N/10", how many of the
            # ten examples the hypothesis explains) and `confidence`, and both
            # were parsed off the wire and dropped here. They are the only signal
            # in the pipeline for "the model refused / was unsure", which is
            # exactly what a labelling run needs to be validated against — a
            # confident wrong label and a hedged right one were indistinguishable
            # downstream. Measured on gemma-4-12B-it these vary meaningfully with
            # the evidence (0/10 on incoherent features, 10/10 where the ten
            # prime tokens were identical), so they carry real information.
            #
            # Optional by design: templates that do not ask for them yield None,
            # and no caller may assume they are present.
            return _enforce_refusal({
                "category": category,
                "specific": specific,
                "description": description,
                "fit_count": _clean_optional(data.get("fit_count")),
                "confidence": _clean_optional(data.get("confidence")),
            })

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

            fit_match = re.search(r'fit_count["\s:]+["\']?(\d+\s*/\s*\d+)', response, re.IGNORECASE)
            conf_match = re.search(r'confidence["\s:]+["\']?(high|medium|low)', response, re.IGNORECASE)

            return _enforce_refusal({
                "category": category,
                "specific": specific,
                "description": description,
                "fit_count": fit_match.group(1).replace(" ", "") if fit_match else None,
                "confidence": conf_match.group(1).lower() if conf_match else None,
            })

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

        # Truncate if too long (database limit: 500 characters)
        if len(label) > 500:
            label = label[:500]

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

    def _resolve_user_prompt(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        user_prompt_template: str,
        feature_id: str,
        logit_effects: Optional[Dict[str, Any]] = None,
        all_examples: Optional[List[Dict[str, Any]]] = None,
        nlp_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Resolve NLP analysis and render the user prompt for one feature.

        Shared by the serial and batched paths so a batched label is built from
        byte-identical inputs to the serial one. Duplicating this was the
        obvious alternative and the wrong one: the two would drift, and the
        difference would show up as an unexplained quality gap between batch
        sizes rather than as an error.
        """
        include_nlp = template_config.get('include_nlp_analysis', False)
        analysis_summary = None
        if include_nlp:
            if nlp_analysis:
                analysis_summary = nlp_analysis.get("summary_for_prompt", "")
            elif all_examples and len(all_examples) > len(examples):
                try:
                    nlp_service = NLPAnalysisService()
                    analysis_result = nlp_service.analyze_feature(
                        all_examples, feature_id or "unknown"
                    )
                    analysis_summary = analysis_result.get("summary_for_prompt", "")
                except Exception as e:
                    logger.warning(
                        f"Failed to compute NLP analysis for feature {feature_id}: {e}"
                    )
        return self._build_user_prompt(
            examples=examples,
            template_config=template_config,
            user_prompt_template=user_prompt_template,
            feature_id=feature_id,
            logit_effects=logit_effects,
            analysis_summary=analysis_summary,
        )

    # miLLM serves a batch in ONE forward pass, reading the model weights once
    # instead of once per feature: measured 5.59x aggregate throughput at batch
    # 8 on gemma-4-12B-it. Eight is the shipped default because it leaves 4.8 GB
    # of headroom on a 24 GB card; 12 reaches 7.31x with 1.7 GB left and 16 OOMs.
    BATCH_SIZE = 8

    async def generate_labels_from_examples_batched(
        self,
        requests: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Label many features per round trip, falling back to serial on any doubt.

        `requests` is a list of kwargs dicts for generate_label_from_examples.
        Returns one label dict per request, IN INPUT ORDER, always the same
        length as `requests` — a caller can zip it against its own feature list
        without checking.

        NOT for labeling trials. Batch composition changes greedy output under
        int8 quantisation (measured: a prompt that is longest in its batch, and
        so unpadded, still differs between batch sizes; each size is itself
        deterministic). Quality is unaffected — 5 of 8 labels identical, the
        rest differing only in wording — but a trial is supposed to vary the
        template and nothing else, so trials must run serially or hold both the
        batch size and the panel order fixed.
        """
        if not requests:
            return []

        size = batch_size or self.BATCH_SIZE
        results: List[Optional[Dict[str, str]]] = [None] * len(requests)

        for start in range(0, len(requests), size):
            chunk = requests[start:start + size]
            if len(chunk) == 1:
                results[start] = await self.generate_label_from_examples(**chunk[0])
                continue
            try:
                labels = await self._generate_chunk_batched(chunk)
            except Exception as e:
                # Granularity is the thing being protected here. One batched
                # request is one failure domain: a timeout or a 5xx loses all N,
                # where the serial path loses exactly one feature. So on ANY
                # batch failure, re-run the chunk serially rather than writing
                # N error labels — the run degrades in speed, not in coverage.
                logger.warning(
                    f"Batched labeling failed for {len(chunk)} features "
                    f"({type(e).__name__}: {e}); falling back to serial"
                )
                labels = []
                for req in chunk:
                    labels.append(await self.generate_label_from_examples(**req))

            for offset, label in enumerate(labels):
                results[start + offset] = label

        # No None may survive: the caller zips this against its features.
        return [
            r if r is not None
            else {"category": "error_feature",
                  "specific": f"feature_{i}", "description": ""}
            for i, r in enumerate(results)
        ]

    async def _generate_chunk_batched(
        self, chunk: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """One batched round trip. Raises rather than returning partial results."""
        message_sets = []
        for req in chunk:
            user_prompt = self._resolve_user_prompt(
                examples=req["examples"],
                template_config=req["template_config"],
                user_prompt_template=req["user_prompt_template"],
                feature_id=req["feature_id"],
                logit_effects=req.get("logit_effects"),
                all_examples=req.get("all_examples"),
                nlp_analysis=req.get("nlp_analysis"),
            )
            system_message = req.get("system_message") or (
                "You are an expert in mechanistic interpretability analyzing "
                "sparse autoencoder features. You will be given multiple "
                "activation examples. You MUST synthesize across ALL of them — "
                "do not describe or name specific tokens from individual "
                "examples. Find the shared concept that explains why all "
                "examples activate the same feature. Provide category, "
                "specific label, and description in JSON format."
            )
            message_sets.append([
                {"role": "system", "content": _enforce_json_only(system_message)},
                {"role": "user", "content": user_prompt},
            ])

        response, advertised = await self._call_openai_batched(message_sets)

        # Capability check BEFORE trusting the response. A server without the
        # extension answers a batch of 8 with one choice and no error.
        # ONE guard, not two. A missing header leaves `advertised` None, and
        # None != len(...) is always true, so a separate `is None` branch would
        # look like independent protection while being unreachable as a
        # distinct behaviour — it only changed the message. It is a message
        # variant here, which is what it always was.
        if advertised != len(message_sets):
            raise BatchUnsupportedError(
                "server sent no X-miLLM-Batch header; batching unsupported"
                if advertised is None else
                f"server served {advertised} of {len(message_sets)} conversations"
            )

        choices = list(response.choices or [])
        if len(choices) != len(chunk):
            raise BatchUnsupportedError(
                f"expected {len(chunk)} choices, got {len(choices)}"
            )

        # Demux on `index`, never on wire order — the OpenAI schema does not
        # promise choices arrive sorted, and a silent mis-ordering would attach
        # every label to the wrong feature while looking entirely healthy.
        indices = sorted(c.index for c in choices)
        if indices != list(range(len(chunk))):
            raise BatchUnsupportedError(f"non-contiguous choice indices: {indices}")
        ordered = sorted(choices, key=lambda c: c.index)

        labels = []
        for req, choice in zip(chunk, ordered):
            text = (choice.message.content or "").strip()
            labels.append(
                self._parse_dual_label(text, f"feature_{req['feature_id']}")
            )
        return labels

    async def _call_openai_batched(self, message_sets: List[list]):
        """Send N conversations as one request; return (response, batch_header).

        Uses with_raw_response because the capability signal is a HEADER — the
        parsed body of an unsupported server's reply is indistinguishable from
        a successful batch of one.
        """
        call_kwargs = {
            "model": self.model,
            "messages": message_sets[0],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
            "extra_body": {"extra_messages": message_sets[1:]},
        }
        if self.chat_template_kwargs:
            call_kwargs["extra_body"]["chat_template_kwargs"] = (
                self.chat_template_kwargs
            )
        call_kwargs.setdefault("response_format", {"type": "json_object"})

        async with self._api_semaphore:
            raw = await self.client.chat.completions.with_raw_response.create(
                **call_kwargs
            )
        advertised = None
        try:
            header = raw.headers.get("X-miLLM-Batch")
            if header is not None:
                advertised = int(header)
        except (TypeError, ValueError):
            advertised = None
        return raw.parse(), advertised

    async def generate_label_from_examples(
        self,
        examples: List[Dict[str, Any]],
        template_config: Dict[str, Any],
        user_prompt_template: str,
        system_message: str,
        feature_id: str,
        neuron_index: Optional[int] = None,
        logit_effects: Optional[Dict[str, Any]] = None,
        all_examples: Optional[List[Dict[str, Any]]] = None,
        nlp_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate semantic label for a feature using context-based examples.

        This is the new context-based labeling method that uses full activation examples
        with prefix/prime/suffix tokens instead of aggregated token statistics.

        Enhanced with NLP analysis that provides statistical patterns from ALL examples
        (not just the top K displayed) to give the LLM better context for labeling.

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
            all_examples: Optional full list of all examples (for NLP analysis)
            nlp_analysis: Optional pre-computed NLP analysis results

        Returns:
            Dict with {"category": "...", "specific": "...", "description": "..."}
        """
        fallback_label = f"feature_{feature_id}"

        if not examples:
            logger.warning(f"Empty examples for feature {feature_id}, using fallback label")
            return {"category": "empty_features", "specific": fallback_label, "description": ""}

        # Compute NLP analysis only if enabled in template config
        include_nlp = template_config.get('include_nlp_analysis', False)
        analysis_summary = None
        if include_nlp:
            if nlp_analysis:
                analysis_summary = nlp_analysis.get("summary_for_prompt", "")
            elif all_examples and len(all_examples) > len(examples):
                try:
                    nlp_service = NLPAnalysisService()
                    analysis_result = nlp_service.analyze_feature(all_examples, feature_id or "unknown")
                    analysis_summary = analysis_result.get("summary_for_prompt", "")
                    logger.debug(f"Computed NLP analysis for feature {feature_id} with {len(all_examples)} examples")
                except Exception as e:
                    logger.warning(f"Failed to compute NLP analysis for feature {feature_id}: {e}")

        # Fall back to defaults when no template is selected
        effective_system_message = system_message or (
            "You are an expert in mechanistic interpretability analyzing sparse autoencoder features. "
            "You will be given multiple activation examples. You MUST synthesize across ALL of them — "
            "do not describe or name specific tokens from individual examples. "
            "Find the shared concept that explains why all examples activate the same feature. "
            "Provide category, specific label, and description in JSON format."
        )

        try:
            # Build user prompt using the new _build_user_prompt method
            user_prompt = self._build_user_prompt(
                examples=examples,
                template_config=template_config,
                user_prompt_template=user_prompt_template,
                feature_id=feature_id,
                logit_effects=logit_effects,
                analysis_summary=analysis_summary
            )

            # Prepare request payload
            request_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _enforce_json_only(effective_system_message)},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
                "top_p": self.top_p
            }

            # Log API call details
            logger.info(f"🔍 OpenAI API Call for feature {feature_id}:")
            logger.info(f"  - Model: {self.model}")
            logger.info(f"  - Examples: {len(examples)} activation examples")
            logger.info(f"  - Prompt length: {len(user_prompt)} chars")
            logger.debug(f"\n📝 SYSTEM MESSAGE:\n{system_message}")
            logger.debug(f"\n📝 USER PROMPT:\n{user_prompt}")

            # Save request to file for Postman/cURL testing (if enabled and sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_request_for_testing(request_payload, neuron_index)
                else:
                    logger.debug(f"Skipping request save due to sample rate: {self.save_requests_sample_rate}")

            # Call OpenAI API (with automatic fallback for reasoning models)
            import time
            start_time = time.time()
            response = await self._call_openai(
                messages=request_payload["messages"]
            )
            elapsed_time = time.time() - start_time

            # Save response to file for testing (if enabled and sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_response_for_testing(response, neuron_index, elapsed_time)

            # Extract and parse response
            label_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            logger.info(f"✅ API Response received for feature {feature_id} (length: {len(label_text)} chars)")
            logger.debug(f"📤 FULL RESPONSE:\n{label_text}")

            # Parse JSON response
            labels = self._parse_dual_label(label_text, fallback_label)
            logger.debug(f"Generated labels for {feature_id}: category='{labels['category']}', specific='{labels['specific']}'")

            # Check for poor quality labels and save debug info if needed
            if self.is_poor_quality_label(labels):
                logger.info(f"⚠️ Poor quality label detected for {feature_id}: category='{labels['category']}', specific='{labels['specific']}'")
                # Convert response to dict for saving
                response_data = None
                if hasattr(response, 'model_dump'):
                    response_data = response.model_dump()
                elif hasattr(response, 'dict'):
                    response_data = response.dict()
                # Note: We don't have token_stats in this method, so we'll pass an empty dict
                # The examples contain the actual context data
                self._save_poor_quality_debug(
                    labels=labels,
                    token_stats={},  # Not available in context-based labeling
                    neuron_index=neuron_index,
                    response_data=response_data,
                    request_payload=request_payload
                )

            return labels

        except RateLimitError as e:
            logger.warning(f"⚠️ OpenAI rate limit for feature {feature_id}: {e}")

            # Save debug files for troubleshooting rate limits (if sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_request_for_testing(request_payload, neuron_index=neuron_index)
                    # Note: No response to save for rate limit errors
                    logger.info(f"💾 Saved debug files for rate-limited feature (neuron_index={neuron_index})")

            return {"category": "rate_limited", "specific": fallback_label, "description": ""}

        except AuthenticationError as e:
            logger.error(f"❌ OpenAI authentication failed for feature {feature_id}: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Error calling OpenAI API for feature {feature_id}:")
            logger.error(f"   Error Type: {type(e).__name__}")
            logger.error(f"   Error Message: {e}", exc_info=True)

            # Save debug files for troubleshooting errors (if sampled)
            if self.save_requests_for_testing:
                import random
                if random.random() <= self.save_requests_sample_rate:
                    self._save_request_for_testing(request_payload, neuron_index=neuron_index)
                    # Try to save response if available (may not be if error was before API call)
                    if 'response' in locals():
                        self._save_response_for_testing(response, neuron_index=neuron_index, elapsed_time=elapsed_time)
                    logger.info(f"💾 Saved debug files for failed feature (neuron_index={neuron_index}, error={type(e).__name__})")

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
