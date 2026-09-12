"""update_anthropic_template_improved_prompts

Revision ID: 90faea1e38d0
Revises: c458030cd014
Create Date: 2025-11-22 11:58:04.066860

Updates the Anthropic Style labeling template with improved prompts:
- Fixes category system (removes 'semantic_or_other', adds 6 specific categories)
- Adds 4-step reasoning structure for evidence-based labeling
- Requires quoting specific tokens from examples
- Prohibits speculative language ("likely", "probably", "may", etc.)
- Adds fallback label 'mixed_activation_pattern' for unclear features
- Increases default examples from 50 to 25 (more practical default)
- Increases max_tokens from 1500 to 2000 for detailed reasoning
- Decreases temperature from 0.2 to 0.1 for more consistent output
"""
from typing import Sequence, Union
import json
import os
from datetime import datetime

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '90faea1e38d0'
down_revision: Union[str, None] = 'c458030cd014'
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
    Update the Anthropic Style template with improved evidence-based prompts.

    Key improvements:
    1. Fixed category system with 6 specific categories
    2. 4-step reasoning process (QUOTE → PATTERN → CONTEXT → LOGIT EFFECTS)
    3. Mandatory token quoting requirement
    4. Anti-speculation rules
    5. Fallback label for unclear features
    6. Increased label length limit to 60 characters
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
    Revert to the original Anthropic Style template.
    """
    # Hardcoded original values from c458030cd014_seed_system_labeling_templates.py
    sql = """
        UPDATE labeling_prompt_templates
        SET
            name = 'Anthropic Style - Context + Logit Effects',
            description = 'Enhanced labeling template using full-context examples PLUS logit effects (promoted/suppressed tokens). Generates richer explanations for important features. K=50 examples, ~1,720 tokens.',
            system_message = 'You analyze sparse autoencoder (SAE) features using rich activation examples.
Your job is to infer the single underlying conceptual meaning represented by a feature.

You are given multiple short text spans for a single feature. In each span, the token(s) where this feature activates most strongly are wrapped in << >>. You are also given the feature''s activation strength on that span and the tokens whose logits it promotes or suppresses.

Use ALL the examples, their context, and the logit effects to infer ONE human concept that best describes this feature. Prefer semantic or functional concepts over surface-level descriptions.

Return ONLY a JSON object in this format:
{
  "specific": "one_or_two_word_concept",
  "category": "semantic_or_other",
  "description": "One sentence describing the conceptual meaning represented by this feature."
}

Rules:
- JSON only
- No markdown
- No code fences
- No commentary before or after the JSON
- Double quotes only',
            user_prompt_template = 'Analyze sparse autoencoder feature {feature_id}.
You are given some of the highest-activating examples for this feature, along with its logit effects.

Each example is formatted as:
  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

Examples:

{examples_block}

Logit effects for this feature:
- Top promoted tokens: {top_promoted_tokens}
- Top suppressed tokens: {top_suppressed_tokens}

Instructions:
- Focus on what the highlighted tokens represent IN CONTEXT.
- Use the promoted/suppressed tokens to refine your understanding of what the feature is doing to the model''s output distribution.
- Prefer semantic labels that could be used to steer or monitor the model.
- If no coherent pattern exists, use category = "system" and specific = "noise_feature".

Return ONLY this JSON object:
{
  "specific": "concept",
  "category": "semantic_or_other",
  "description": "One sentence describing the conceptual meaning."
}',
            temperature = 0.2,
            max_tokens = 1500,
            max_examples = 50,
            updated_at = NOW()
        WHERE id = 'lpt_anthropic_style'
    """

    op.execute(sql)
