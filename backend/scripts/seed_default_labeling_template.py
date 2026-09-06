"""
Seed script for default labeling prompt template.

This script creates the default system template with the current OpenAI labeling prompt.
Run this after database migrations to ensure a default template exists.
"""

import asyncio
import sys
from pathlib import Path

# Add backend src to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.models.labeling_prompt_template import LabelingPromptTemplate


# Default system prompt template (current OpenAI labeling prompt)
DEFAULT_SYSTEM_MESSAGE = """You are an expert in mechanistic interpretability analyzing sparse autoencoder features. Provide both category and specific labels in JSON format."""

DEFAULT_USER_PROMPT_TEMPLATE = """Analyze this sparse autoencoder feature and provide:
1. A broad category (e.g., "syntactic", "semantic", "positional")
2. A specific label describing what the feature detects

Feature #{neuron_index} from layer {layer_name}
Top activating tokens (sorted by activation strength):
{tokens_table}

Provide your analysis in JSON format:
{{
  "category": "broad category here",
  "specific": "specific behavior here"
}}

Decision tree for specificity:
- If clear pattern → specific label
- If mixed signals → describe the combination
- If unclear → "mixed" or "unclear"

Guidelines:
- Category: Choose from syntactic, semantic, positional, structural, or mixed
- Specific: Be precise but concise (2-5 words)
- Focus on consistent patterns across top tokens
- Consider token context and activation strength
"""


async def seed_default_template():
    """Create the default labeling prompt template if it doesn't exist."""

    print("🌱 Seeding default labeling prompt template...")

    async with AsyncSessionLocal() as db:
        # Check if default template already exists
        result = await db.execute(
            select(LabelingPromptTemplate).where(
                LabelingPromptTemplate.is_system == True,
                LabelingPromptTemplate.is_default == True
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            print(f"✅ Default template already exists: {existing.name} (ID: {existing.id})")
            print("   Skipping seed operation.")
            return

        # Create default system template
        default_template = LabelingPromptTemplate(
            id="lpt_system_default_001",
            name="Default Labeling Prompt",
            description="System default prompt template for semantic feature labeling. Uses dual labeling approach (category + specific) with top activating tokens analysis.",
            system_message=DEFAULT_SYSTEM_MESSAGE,
            user_prompt_template=DEFAULT_USER_PROMPT_TEMPLATE,
            temperature=0.3,
            max_tokens=50,
            top_p=0.9,
            is_default=True,
            is_system=True,
            created_by=None
        )

        db.add(default_template)
        await db.commit()
        await db.refresh(default_template)

        print(f"✅ Created default template: {default_template.name}")
        print(f"   ID: {default_template.id}")
        print(f"   Temperature: {default_template.temperature}")
        print(f"   Max Tokens: {default_template.max_tokens}")
        print(f"   Top P: {default_template.top_p}")
        print("   Status: System template, set as default")


if __name__ == "__main__":
    print("=" * 60)
    print("Default Labeling Prompt Template Seed Script")
    print("=" * 60)
    print()

    try:
        asyncio.run(seed_default_template())
        print()
        print("✅ Seed operation completed successfully!")
        print("=" * 60)
    except Exception as e:
        print()
        print(f"❌ Error during seed operation: {str(e)}")
        print("=" * 60)
        sys.exit(1)
