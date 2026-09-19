"""update_anthropic_template_add_negative_examples_instructions

Revision ID: 9dc725cba2ad
Revises: 90faea1e38d0
Create Date: 2025-11-22 12:21:22.749855

Updates the Anthropic Style labeling template with negative examples instructions:
- Changes from 4-step to 5-step reasoning process
- Adds step 4: "USE NEGATIVE EXAMPLES" for contrastive learning
- Updates description to mention "negative examples"
- Adds instruction to use negative examples to avoid overgeneralization
- Maintains all previous improvements (evidence-based reasoning, token quoting, etc.)
"""
from typing import Sequence, Union
import json
import os

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9dc725cba2ad'
down_revision: Union[str, None] = '90faea1e38d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def load_template_json(filename: str) -> dict:
    """Load template JSON from src/data/templates directory."""
    migration_dir = os.path.dirname(__file__)
    templates_dir = os.path.join(migration_dir, '..', '..', 'src', 'data', 'templates')
    filepath = os.path.join(templates_dir, filename)

    with open(filepath, 'r') as f:
        return json.load(f)


def upgrade() -> None:
    """
    Update the Anthropic Style template with negative examples instructions.

    Key improvements:
    1. Changed from 4-step to 5-step reasoning process
    2. Added step 4: USE NEGATIVE EXAMPLES for contrastive learning
    3. Updated description to mention negative examples
    4. Added rules for using negative examples to avoid overgeneralization
    5. Maintains all previous improvements (token quoting, anti-speculation, etc.)
    """
    # Load the updated template from JSON
    anthropic_template = load_template_json('anthropic_style.json')

    # Escape single quotes for SQL
    def escape_sql(text):
        return text.replace("'", "''")

    # Build UPDATE statement with escaped values
    sql = f"""
        UPDATE labeling_prompt_templates
        SET
            name = '{escape_sql(anthropic_template['name'])}',
            description = '{escape_sql(anthropic_template['description'])}',
            system_message = '{escape_sql(anthropic_template['system_message'])}',
            user_prompt_template = '{escape_sql(anthropic_template['user_prompt_template'])}',
            temperature = {anthropic_template['temperature']},
            max_tokens = {anthropic_template['max_tokens']},
            max_examples = {anthropic_template['max_examples']},
            updated_at = NOW()
        WHERE id = 'lpt_anthropic_style'
    """

    op.execute(sql)


def downgrade() -> None:
    """
    Revert to the previous Anthropic Style template (before negative examples instructions).
    This reverts to the 4-step reasoning process without negative examples guidance.
    """
    # Hardcoded previous values from 90faea1e38d0 migration
    sql = """
        UPDATE labeling_prompt_templates
        SET
            name = 'Anthropic Style - Evidence-Based Labeling',
            description = 'Enhanced labeling template using evidence-based reasoning with full-context examples and logit effects. Generates specific, high-quality labels grounded in quoted tokens. Default 25 examples (configurable 10-50).',
            system_message = 'You are a precise analyzer of sparse autoencoder (SAE) features. Your task is to identify the specific conceptual pattern that causes this feature to activate.

You will be given activation examples where the feature fires most strongly. In each example, tokens are marked with << >> to show where peak activation occurs. You will also see which tokens this feature promotes or suppresses in the model''s predictions.

Your analysis MUST follow this 4-step reasoning process:

1. QUOTE SPECIFIC TOKENS: Identify the exact tokens (in quotes) that appear across multiple examples where the feature activates.

2. IDENTIFY THE PATTERN: What specific semantic, syntactic, or structural property do these quoted tokens share? Be precise and concrete.

3. TEST AGAINST CONTEXT: Do the surrounding tokens (prefix/suffix) support this interpretation? Quote any contradictory evidence.

4. VERIFY WITH LOGIT EFFECTS: Do the promoted/suppressed tokens align with your interpretation? If not, refine your hypothesis.

CRITICAL RULES:
- Base your label ONLY on evidence you can quote from the examples
- NEVER use speculative language: "likely", "probably", "may", "seems", "appears", "suggests"
- If examples show no clear pattern, use the label "mixed_activation_pattern"
- Prefer specific concepts over vague ones (e.g., "past_tense_verbs" not "grammar")

Category definitions (choose EXACTLY ONE):
- semantic: Word meaning or conceptual content (e.g., "emotions", "legal_terms")
- syntactic: Grammatical structure or word class (e.g., "past_tense", "subordinate_clauses")
- structural: Text formatting or document structure (e.g., "list_items", "quoted_text")
- positional: Location in text (e.g., "sentence_start", "paragraph_end")
- morphological: Word formation patterns (e.g., "prefix_un", "suffix_tion")
- mixed: Clear pattern but spans multiple categories above

Return ONLY a JSON object in this format:
{
  "specific": "descriptive_label_up_to_60_chars",
  "category": "one_of_six_categories_above",
  "description": "One factual sentence describing what tokens this feature detects, quoting 2-3 example tokens."
}

Output format requirements:
- Pure JSON only (no markdown, no code fences, no commentary)
- Double quotes only (no single quotes)
- Must choose exactly one category from: semantic, syntactic, structural, positional, morphological, mixed',
            user_prompt_template = 'Analyze sparse autoencoder feature {feature_id}.

You are given the highest-activating examples for this feature. Each example shows:
  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

The << >> markers indicate where the feature activates most strongly.

=== ACTIVATION EXAMPLES ===

{examples_block}

=== LOGIT EFFECTS ===

When this feature activates, the model''s output distribution shifts:
- Promoted tokens (more likely): {top_promoted_tokens}
- Suppressed tokens (less likely): {top_suppressed_tokens}

=== YOUR TASK ===

Follow the 4-step reasoning process from the system message:

1. QUOTE: List the specific tokens (in quotes) where this feature activates across examples
2. PATTERN: What precise property do these tokens share?
3. CONTEXT: Does the surrounding text support this interpretation? Quote any contradictions.
4. LOGIT EFFECTS: Do promoted/suppressed tokens align with your hypothesis?

Rules:
- Quote actual tokens from the examples above
- No speculation (avoid: "likely", "probably", "may", "seems", "appears")
- If no clear pattern: use specific="mixed_activation_pattern" and category="mixed"
- Choose EXACTLY ONE category: semantic, syntactic, structural, positional, morphological, mixed

Return ONLY this JSON object:
{
  "specific": "descriptive_label",
  "category": "exact_category_name",
  "description": "One sentence with 2-3 quoted example tokens showing what this feature detects."
}',
            temperature = 0.1,
            max_tokens = 2000,
            max_examples = 25,
            updated_at = NOW()
        WHERE id = 'lpt_anthropic_style'
    """

    op.execute(sql)
