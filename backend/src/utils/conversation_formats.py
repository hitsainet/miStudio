"""
Conversation format detection and text extraction utilities.

This module provides utilities for detecting and extracting text from various
conversation dataset formats commonly used in LLM training datasets.

Supported formats:
- OpenAI/ChatML: [{"role": "user/assistant/system", "content": "..."}]
- ShareGPT: [{"from": "human/gpt", "value": "..."}]
- Simple list: ["message1", "message2", ...]
- LMSYS: Same as OpenAI but typically in 'conversation' column
- Vicuna: [{"from": "human/gpt", "value": "..."}] (alias for ShareGPT)
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ConversationFormat(str, Enum):
    """Enumeration of supported conversation formats."""

    OPENAI = "openai"           # [{role: "user", content: "..."}, ...]
    SHAREGPT = "sharegpt"       # [{from: "human", value: "..."}, ...]
    SIMPLE_LIST = "simple_list" # ["message1", "message2", ...]
    ALPACA = "alpaca"           # {instruction: "...", input: "...", output: "..."}
    UNKNOWN = "unknown"         # Format not recognized
    NOT_CONVERSATION = "not_conversation"  # Not a conversation column


@dataclass
class ConversationColumnInfo:
    """Information about a detected conversation column."""

    column_name: str
    format: ConversationFormat
    confidence: float  # 0.0 to 1.0
    sample_roles: List[str]  # e.g., ["user", "assistant"] or ["human", "gpt"]
    avg_turns: float  # Average number of turns in conversations
    text_key: str  # Key used for text content (e.g., "content" or "value")
    role_key: str  # Key used for role (e.g., "role" or "from")
    description: str  # Human-readable description of the format
    extraction_hint: str  # Hint for how to extract text


@dataclass
class SchemaAnalysisResult:
    """Complete schema analysis result with conversation detection."""

    # Original schema info
    text_columns: List[str]
    column_info: Dict[str, str]
    all_columns: List[str]

    # Conversation detection results
    conversation_columns: List[ConversationColumnInfo]

    # Recommendations
    recommended_column: Optional[str]
    recommended_format: Optional[ConversationFormat]
    requires_preprocessing: bool
    preprocessing_hint: Optional[str]

    # User-facing messages
    warnings: List[str]
    suggestions: List[str]


def detect_conversation_format(
    sample_data: List[Any],
    column_name: str,
) -> ConversationColumnInfo:
    """
    Detect the conversation format from sample data.

    Args:
        sample_data: List of sample values from the column (typically 5-10 samples)
        column_name: Name of the column being analyzed

    Returns:
        ConversationColumnInfo with detected format details
    """
    if not sample_data:
        return ConversationColumnInfo(
            column_name=column_name,
            format=ConversationFormat.NOT_CONVERSATION,
            confidence=1.0,
            sample_roles=[],
            avg_turns=0,
            text_key="",
            role_key="",
            description="Empty column",
            extraction_hint="Column is empty",
        )

    # Check if it's a list/sequence column
    first_sample = sample_data[0]

    # Handle case where sample is not a list
    if not isinstance(first_sample, (list, tuple)):
        return ConversationColumnInfo(
            column_name=column_name,
            format=ConversationFormat.NOT_CONVERSATION,
            confidence=1.0,
            sample_roles=[],
            avg_turns=0,
            text_key="",
            role_key="",
            description="Not a list/sequence column",
            extraction_hint="Use as-is (string column)",
        )

    # Count format matches across samples
    format_counts = {
        ConversationFormat.OPENAI: 0,
        ConversationFormat.SHAREGPT: 0,
        ConversationFormat.SIMPLE_LIST: 0,
    }

    all_roles = set()
    total_turns = 0
    valid_samples = 0

    for sample in sample_data:
        if not isinstance(sample, (list, tuple)) or len(sample) == 0:
            continue

        valid_samples += 1
        total_turns += len(sample)

        # Check first element to determine format
        first_elem = sample[0]

        if isinstance(first_elem, str):
            # Simple list format: ["msg1", "msg2", ...]
            format_counts[ConversationFormat.SIMPLE_LIST] += 1

        elif isinstance(first_elem, dict):
            # Dictionary-based conversation format
            keys = set(first_elem.keys())

            # Check for OpenAI format: {role, content}
            if "role" in keys and "content" in keys:
                format_counts[ConversationFormat.OPENAI] += 1
                for turn in sample:
                    if isinstance(turn, dict) and "role" in turn:
                        all_roles.add(turn.get("role", ""))

            # Check for ShareGPT format: {from, value}
            elif "from" in keys and "value" in keys:
                format_counts[ConversationFormat.SHAREGPT] += 1
                for turn in sample:
                    if isinstance(turn, dict) and "from" in turn:
                        all_roles.add(turn.get("from", ""))

            # Check alternative ShareGPT keys: {human, gpt} as keys
            elif "human" in keys or "gpt" in keys:
                format_counts[ConversationFormat.SHAREGPT] += 1
                all_roles.update(keys & {"human", "gpt", "system"})

    if valid_samples == 0:
        return ConversationColumnInfo(
            column_name=column_name,
            format=ConversationFormat.UNKNOWN,
            confidence=0.0,
            sample_roles=[],
            avg_turns=0,
            text_key="",
            role_key="",
            description="No valid samples found",
            extraction_hint="Manual inspection required",
        )

    # Determine the winning format
    max_count = max(format_counts.values())
    confidence = max_count / valid_samples

    if format_counts[ConversationFormat.OPENAI] == max_count:
        detected_format = ConversationFormat.OPENAI
        text_key = "content"
        role_key = "role"
        description = "OpenAI/ChatML format: [{role, content}, ...]"
        extraction_hint = (
            "Extract text from 'content' field, optionally prefixing with role. "
            "Format: '<|role|>\\ncontent\\n' for each turn."
        )
    elif format_counts[ConversationFormat.SHAREGPT] == max_count:
        detected_format = ConversationFormat.SHAREGPT
        text_key = "value"
        role_key = "from"
        description = "ShareGPT/Vicuna format: [{from, value}, ...]"
        extraction_hint = (
            "Extract text from 'value' field, mapping roles: human->user, gpt->assistant. "
            "Format: '<|role|>\\nvalue\\n' for each turn."
        )
    elif format_counts[ConversationFormat.SIMPLE_LIST] == max_count:
        detected_format = ConversationFormat.SIMPLE_LIST
        text_key = ""
        role_key = ""
        description = "Simple list format: [str, str, ...]"
        extraction_hint = (
            "Join all strings with newlines or alternating role prefixes. "
            "Odd indices = user, even indices = assistant (if desired)."
        )
    else:
        detected_format = ConversationFormat.UNKNOWN
        text_key = ""
        role_key = ""
        description = "Unknown conversation format"
        extraction_hint = "Manual inspection required to determine extraction method."

    avg_turns = total_turns / valid_samples if valid_samples > 0 else 0

    return ConversationColumnInfo(
        column_name=column_name,
        format=detected_format,
        confidence=confidence,
        sample_roles=sorted(list(all_roles)),
        avg_turns=avg_turns,
        text_key=text_key,
        role_key=role_key,
        description=description,
        extraction_hint=extraction_hint,
    )


def extract_text_from_conversation(
    conversation: List[Any],
    format: ConversationFormat,
    include_roles: bool = True,
    role_template: str = "<|{role}|>\n{content}\n",
    join_separator: str = "\n",
    role_mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Extract text from a conversation in a specific format.

    Args:
        conversation: The conversation data (list of turns)
        format: The conversation format
        include_roles: Whether to include role prefixes in output
        role_template: Template for formatting each turn (uses {role} and {content})
        join_separator: Separator between turns
        role_mapping: Optional mapping to normalize roles (e.g., {"human": "user", "gpt": "assistant"})

    Returns:
        Extracted text as a single string
    """
    if not conversation:
        return ""

    # Default role mappings for normalization
    default_role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "ai": "assistant",
        "model": "assistant",
    }

    if role_mapping is None:
        role_mapping = default_role_mapping
    else:
        # Merge with defaults
        role_mapping = {**default_role_mapping, **role_mapping}

    extracted_parts = []

    for i, turn in enumerate(conversation):
        if format == ConversationFormat.SIMPLE_LIST:
            # Simple list: just strings
            if isinstance(turn, str):
                if include_roles:
                    # Alternate between user and assistant
                    role = "user" if i % 2 == 0 else "assistant"
                    extracted_parts.append(role_template.format(role=role, content=turn))
                else:
                    extracted_parts.append(turn)

        elif format == ConversationFormat.OPENAI:
            # OpenAI format: {role, content}
            if isinstance(turn, dict):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")

                # Normalize role
                role = role_mapping.get(role.lower(), role)

                if include_roles:
                    extracted_parts.append(role_template.format(role=role, content=content))
                else:
                    extracted_parts.append(content)

        elif format == ConversationFormat.SHAREGPT:
            # ShareGPT format: {from, value}
            if isinstance(turn, dict):
                role = turn.get("from", "unknown")
                content = turn.get("value", "")

                # Normalize role
                role = role_mapping.get(role.lower(), role)

                if include_roles:
                    extracted_parts.append(role_template.format(role=role, content=content))
                else:
                    extracted_parts.append(content)

        else:
            # Unknown format - try to extract any text we can find
            if isinstance(turn, str):
                extracted_parts.append(turn)
            elif isinstance(turn, dict):
                # Try common keys
                for key in ["content", "value", "text", "message"]:
                    if key in turn:
                        extracted_parts.append(str(turn[key]))
                        break

    return join_separator.join(extracted_parts)


def create_conversation_preprocessor(
    conversation_column: str,
    output_column: str = "text",
    format: ConversationFormat = ConversationFormat.OPENAI,
    include_roles: bool = True,
    role_template: str = "<|{role}|>\n{content}\n",
    join_separator: str = "\n",
    role_mapping: Optional[Dict[str, str]] = None,
):
    """
    Create a preprocessing function for HuggingFace dataset.map().

    Args:
        conversation_column: Name of the column containing conversation data
        output_column: Name of the output column for extracted text
        format: The conversation format
        include_roles: Whether to include role prefixes
        role_template: Template for each turn
        join_separator: Separator between turns
        role_mapping: Optional role normalization mapping

    Returns:
        A function suitable for dataset.map()
    """
    def preprocessor(examples):
        """Preprocess a batch of examples."""
        texts = []
        conversations = examples[conversation_column]

        for conv in conversations:
            text = extract_text_from_conversation(
                conversation=conv,
                format=format,
                include_roles=include_roles,
                role_template=role_template,
                join_separator=join_separator,
                role_mapping=role_mapping,
            )
            texts.append(text)

        return {output_column: texts}

    return preprocessor


def analyze_column_for_conversation(
    dataset,
    column_name: str,
    num_samples: int = 10,
) -> ConversationColumnInfo:
    """
    Analyze a specific column to detect if it contains conversation data.

    Args:
        dataset: HuggingFace dataset
        column_name: Name of column to analyze
        num_samples: Number of samples to inspect

    Returns:
        ConversationColumnInfo with detection results
    """
    try:
        # Get sample data from the column
        sample_indices = list(range(min(num_samples, len(dataset))))
        sample_data = [dataset[i][column_name] for i in sample_indices]

        return detect_conversation_format(sample_data, column_name)

    except Exception as e:
        logger.warning(f"Error analyzing column {column_name}: {e}")
        return ConversationColumnInfo(
            column_name=column_name,
            format=ConversationFormat.UNKNOWN,
            confidence=0.0,
            sample_roles=[],
            avg_turns=0,
            text_key="",
            role_key="",
            description=f"Error during analysis: {str(e)}",
            extraction_hint="Manual inspection required",
        )


def get_format_recommendations(
    schema_result: SchemaAnalysisResult,
) -> List[str]:
    """
    Generate user-friendly recommendations based on schema analysis.

    Args:
        schema_result: The schema analysis result

    Returns:
        List of recommendation strings for display to user
    """
    recommendations = []

    if schema_result.conversation_columns:
        # Found conversation columns
        best_conv = max(
            schema_result.conversation_columns,
            key=lambda x: x.confidence
        )

        recommendations.append(
            f"Detected conversation data in column '{best_conv.column_name}' "
            f"({best_conv.description}, {best_conv.confidence*100:.0f}% confidence)"
        )

        if best_conv.avg_turns > 0:
            recommendations.append(
                f"Average conversation length: {best_conv.avg_turns:.1f} turns"
            )

        if best_conv.sample_roles:
            recommendations.append(
                f"Detected roles: {', '.join(best_conv.sample_roles)}"
            )

        recommendations.append(f"Extraction hint: {best_conv.extraction_hint}")

        if schema_result.requires_preprocessing:
            recommendations.append(
                "**Action required**: This dataset needs preprocessing to extract "
                "text from conversations before tokenization. The system will "
                "automatically flatten conversations to text."
            )

    elif schema_result.text_columns:
        # Simple text columns found
        recommendations.append(
            f"Found {len(schema_result.text_columns)} text column(s): "
            f"{', '.join(schema_result.text_columns)}"
        )

        if schema_result.recommended_column:
            recommendations.append(
                f"Recommended column for tokenization: '{schema_result.recommended_column}'"
            )

    else:
        # No suitable columns found
        recommendations.append(
            "**Warning**: No suitable text or conversation columns detected. "
            "Available columns: " + ", ".join(schema_result.all_columns)
        )
        recommendations.append(
            "Please manually specify which column contains the text data."
        )

    return recommendations


# Convenience functions for common format detection patterns

def is_openai_format(sample: List[Dict]) -> bool:
    """Check if a sample matches OpenAI conversation format."""
    if not sample or not isinstance(sample, list):
        return False
    if not sample:
        return False
    first = sample[0]
    return isinstance(first, dict) and "role" in first and "content" in first


def is_sharegpt_format(sample: List[Dict]) -> bool:
    """Check if a sample matches ShareGPT conversation format."""
    if not sample or not isinstance(sample, list):
        return False
    if not sample:
        return False
    first = sample[0]
    return isinstance(first, dict) and "from" in first and "value" in first


def is_simple_list_format(sample: List) -> bool:
    """Check if a sample is a simple list of strings."""
    if not sample or not isinstance(sample, list):
        return False
    return all(isinstance(item, str) for item in sample)
