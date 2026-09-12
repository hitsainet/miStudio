"""seed_system_labeling_templates

Revision ID: c458030cd014
Revises: 77cc0d1b2f7d
Create Date: 2025-11-21 23:14:08.592610

"""
from typing import Sequence, Union
import json
import os
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c458030cd014'
down_revision: Union[str, None] = '77cc0d1b2f7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def load_template_json(filename: str) -> dict:
    """Load template JSON from src/data/templates directory."""
    # Get the directory where this migration file is located
    migration_dir = os.path.dirname(__file__)
    # Navigate to backend/src/data/templates
    templates_dir = os.path.join(migration_dir, '..', '..', 'src', 'data', 'templates')
    filepath = os.path.join(templates_dir, filename)

    with open(filepath, 'r') as f:
        return json.load(f)


def upgrade() -> None:
    """
    Seed the database with three system labeling prompt templates.

    Templates:
    1. miStudio Internal - Full-context labeling (K=10, default)
    2. Anthropic Style - Full-context + logit effects (K=50)
    3. EleutherAI Detection - Explanation scoring
    """
    # Load template data from JSON files
    mistudio_template = load_template_json('mistudio_internal.json')
    anthropic_template = load_template_json('anthropic_style.json')
    eleutherai_template = load_template_json('eleutherai_detection.json')

    # Prepare template records
    now = datetime.now()  # Use naive datetime to match database schema
    templates = [
        {
            'id': 'lpt_mistudio_internal',
            'name': mistudio_template['name'],
            'description': mistudio_template['description'],
            'system_message': mistudio_template['system_message'],
            'user_prompt_template': mistudio_template['user_prompt_template'],
            'temperature': mistudio_template['temperature'],
            'max_tokens': mistudio_template['max_tokens'],
            'top_p': mistudio_template['top_p'],
            'template_type': mistudio_template['template_type'],
            'max_examples': mistudio_template['max_examples'],
            'include_prefix': mistudio_template['include_prefix'],
            'include_suffix': mistudio_template['include_suffix'],
            'prime_token_marker': mistudio_template['prime_token_marker'],
            'include_logit_effects': mistudio_template['include_logit_effects'],
            'top_promoted_tokens_count': mistudio_template.get('top_promoted_tokens_count'),
            'top_suppressed_tokens_count': mistudio_template.get('top_suppressed_tokens_count'),
            'is_detection_template': mistudio_template['is_detection_template'],
            'is_default': mistudio_template['is_default'],
            'is_system': mistudio_template['is_system'],
            'created_at': now,
            'updated_at': now,
        },
        {
            'id': 'lpt_anthropic_style',
            'name': anthropic_template['name'],
            'description': anthropic_template['description'],
            'system_message': anthropic_template['system_message'],
            'user_prompt_template': anthropic_template['user_prompt_template'],
            'temperature': anthropic_template['temperature'],
            'max_tokens': anthropic_template['max_tokens'],
            'top_p': anthropic_template['top_p'],
            'template_type': anthropic_template['template_type'],
            'max_examples': anthropic_template['max_examples'],
            'include_prefix': anthropic_template['include_prefix'],
            'include_suffix': anthropic_template['include_suffix'],
            'prime_token_marker': anthropic_template['prime_token_marker'],
            'include_logit_effects': anthropic_template['include_logit_effects'],
            'top_promoted_tokens_count': anthropic_template.get('top_promoted_tokens_count'),
            'top_suppressed_tokens_count': anthropic_template.get('top_suppressed_tokens_count'),
            'is_detection_template': anthropic_template['is_detection_template'],
            'is_default': anthropic_template['is_default'],
            'is_system': anthropic_template['is_system'],
            'created_at': now,
            'updated_at': now,
        },
        {
            'id': 'lpt_eleutherai_detection',
            'name': eleutherai_template['name'],
            'description': eleutherai_template['description'],
            'system_message': eleutherai_template['system_message'],
            'user_prompt_template': eleutherai_template['user_prompt_template'],
            'temperature': eleutherai_template['temperature'],
            'max_tokens': eleutherai_template['max_tokens'],
            'top_p': eleutherai_template['top_p'],
            'template_type': eleutherai_template['template_type'],
            'max_examples': eleutherai_template['max_examples'],
            'include_prefix': eleutherai_template['include_prefix'],
            'include_suffix': eleutherai_template['include_suffix'],
            'prime_token_marker': eleutherai_template['prime_token_marker'],
            'include_logit_effects': eleutherai_template['include_logit_effects'],
            'top_promoted_tokens_count': eleutherai_template.get('top_promoted_tokens_count'),
            'top_suppressed_tokens_count': eleutherai_template.get('top_suppressed_tokens_count'),
            'is_detection_template': eleutherai_template['is_detection_template'],
            'is_default': eleutherai_template['is_default'],
            'is_system': eleutherai_template['is_system'],
            'created_at': now,
            'updated_at': now,
        },
    ]

    # Insert templates using bulk insert
    op.bulk_insert(
        sa.table(
            'labeling_prompt_templates',
            sa.column('id', sa.String),
            sa.column('name', sa.String),
            sa.column('description', sa.Text),
            sa.column('system_message', sa.Text),
            sa.column('user_prompt_template', sa.Text),
            sa.column('temperature', sa.Float),
            sa.column('max_tokens', sa.Integer),
            sa.column('top_p', sa.Float),
            sa.column('template_type', sa.String),
            sa.column('max_examples', sa.Integer),
            sa.column('include_prefix', sa.Boolean),
            sa.column('include_suffix', sa.Boolean),
            sa.column('prime_token_marker', sa.String),
            sa.column('include_logit_effects', sa.Boolean),
            sa.column('top_promoted_tokens_count', sa.Integer),
            sa.column('top_suppressed_tokens_count', sa.Integer),
            sa.column('is_detection_template', sa.Boolean),
            sa.column('is_default', sa.Boolean),
            sa.column('is_system', sa.Boolean),
            sa.column('created_at', sa.DateTime),
            sa.column('updated_at', sa.DateTime),
        ),
        templates
    )


def downgrade() -> None:
    """
    Remove the system labeling prompt templates.
    """
    op.execute(
        "DELETE FROM labeling_prompt_templates WHERE id IN ("
        "'lpt_mistudio_internal', 'lpt_anthropic_style', 'lpt_eleutherai_detection'"
        ")"
    )
