"""update mistudio template with 15 token limit

Revision ID: 8b9c0d1e2f3a
Revises: 7a8b9c0d1e2f
Create Date: 2025-12-21 19:30:00.000000

"""
from typing import Sequence, Union
import json
import os

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b9c0d1e2f3a'
down_revision: Union[str, None] = '7a8b9c0d1e2f'
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
    """Update miStudio internal template with 15 token limit constraint."""
    template = load_template_json('mistudio_internal.json')
    
    # Update the existing template
    op.execute(
        sa.text("""
            UPDATE labeling_prompt_templates 
            SET 
                description = :description,
                system_message = :system_message,
                user_prompt_template = :user_prompt_template,
                updated_at = NOW()
            WHERE id = 'lpt_mistudio_internal'
        """).bindparams(
            description=template['description'],
            system_message=template['system_message'],
            user_prompt_template=template['user_prompt_template']
        )
    )


def downgrade() -> None:
    """Revert to previous template (without explicit token limit)."""
    # Previous version without explicit 15-token limit
    old_system_message = """You analyze sparse autoencoder (SAE) features using full-context activation examples. Your ONLY job is to infer the single underlying conceptual meaning shared by the most strongly-activating tokens, taking into account both the highlighted token(s) and their surrounding context.

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

    old_user_prompt = """Analyze sparse autoencoder feature {feature_id}.
You are given some of the highest-activating examples for this feature. In each example, the main activating token(s) are wrapped in << >>. Use ALL of the examples, including their surrounding context, to infer the smallest semantic concept that explains why these tokens activate the same feature.

Each example is formatted as:
  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

Examples:

{examples_block}

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

    old_description = "Default labeling template using full-context activation examples with << >> markers around prime tokens. Generates concise semantic labels based on context windows. K=10 examples, ~520 tokens."

    op.execute(
        sa.text("""
            UPDATE labeling_prompt_templates 
            SET 
                description = :description,
                system_message = :system_message,
                user_prompt_template = :user_prompt_template,
                updated_at = NOW()
            WHERE id = 'lpt_mistudio_internal'
        """).bindparams(
            description=old_description,
            system_message=old_system_message,
            user_prompt_template=old_user_prompt
        )
    )
