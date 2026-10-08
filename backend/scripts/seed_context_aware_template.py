"""
Seed script: Context-Aware Labeling template.

This template shifts focus from the prime token to the full semantic context
of each activation example, producing labels that describe WHAT IS HAPPENING
across examples rather than WHAT TOKEN fired.

Run with:
  DATABASE_URL=... python scripts/seed_context_aware_template.py
"""

import asyncio
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from sqlalchemy import select
from src.core.database import AsyncSessionLocal
from src.models.labeling_prompt_template import LabelingPromptTemplate

TEMPLATE_ID = "lpt_context_aware_v1"

SYSTEM_MESSAGE = """\
You are an expert in mechanistic interpretability analyzing sparse autoencoder (SAE) features.

Your task: identify the semantic or functional PATTERN a feature has learned — what situation, role, or linguistic function do all its activation examples share?

The marked token is the trigger location. The feature's true identity is what the surrounding passages have in common — not the surface form of the token.

Process:
1. Read each example as a complete passage and understand what it means
2. Ask: what situation, role, or linguistic function is operating HERE?
3. Find the concept, function, or situation SHARED across nearly all examples
4. Name it specifically and precisely

Critical constraint: Do NOT use the words "context", "contextual", "trigger", "feature", or "pattern" in the specific field. These words describe the meta-task, not the feature — reaching for them means you have not found the real pattern yet.

If the examples are genuinely diverse with no coherent shared pattern (common for very high-frequency features), classify as noise: use category "noise" and specific "high_frequency_polysemantic".\
"""

USER_PROMPT_TEMPLATE = """\
Below are activation examples for SAE feature {feature_id}. The highest-activation token is marked with << >>.

{examples_block}

---

For each example, ask: what is the sentence ABOUT? What semantic domain, grammatical role, entity type, or discourse function is operating?

Now find the shared concept, role, or function across ALL examples. The marked token is a clue, not the answer — features routinely fire on different surface tokens that share a deeper semantic situation (e.g., all introduce a new entity, all mark a causal relation, all appear in obligation contexts).

If examples look completely unrelated with no coherent pattern, this is a polysemantic or noise feature — use the noise output.

Respond ONLY with this JSON — no text outside it:
{{"category": "syntactic|semantic|positional|discourse|entity|noise|mixed", "specific": "snake_case_label_max_5_words", "description": "Fires on: [precise one-sentence description of the shared semantic or functional pattern]."}}

Rules:
- specific: 2–5 words, snake_case, names the SEMANTIC PATTERN or FUNCTION — good examples: source_attribution_from, modal_epistemic_obligation, entity_introduction_definite, comparative_clause_than
- FORBIDDEN in specific: "context", "contextual", "trigger", "feature", "pattern" — if you reach for these, look harder for the actual pattern
- description: must start with "Fires on:" then precisely describe what passages share
- noise: if examples have no coherent pattern or activation frequency is very high, use category "noise" and specific "high_frequency_polysemantic"\
"""


async def seed():
    print("Seeding Context-Aware Labeling template...")

    async with AsyncSessionLocal() as db:
        existing = await db.get(LabelingPromptTemplate, TEMPLATE_ID)
        if existing:
            print(f"Template '{TEMPLATE_ID}' already exists — updating prompts.")
            existing.system_message = SYSTEM_MESSAGE
            existing.user_prompt_template = USER_PROMPT_TEMPLATE
            await db.commit()
            print("Updated.")
            return

        template = LabelingPromptTemplate(
            id=TEMPLATE_ID,
            name="Context-Aware Labeling (Semantic Pattern)",
            description=(
                "Focuses on full semantic context. Prohibits generic fallbacks (context, trigger, feature). "
                "Includes noise escape hatch for polysemantic features. "
                "meaning, function, or discourse role — rather than naming the surface token. "
                "Recommended for OpenAI GPT-4o / GPT-5 class models."
            ),
            system_message=SYSTEM_MESSAGE,
            user_prompt_template=USER_PROMPT_TEMPLATE,
            template_type="mistudio_context",  # uses {examples_block} with full context windows
            temperature=0.2,
            max_tokens=200,
            top_p=0.95,
            prime_token_marker="<< >>",
            include_prefix=True,
            include_suffix=True,
            include_negative_examples=True,
            num_negative_examples=3,
            include_logit_effects=False,
            include_nlp_analysis=False,
            is_default=False,
            is_system=True,  # visible to all users, can't be deleted
            created_by=None,
        )
        db.add(template)
        await db.commit()
        print(f"Created template: '{template.name}' (id={template.id})")
        print("Select it in Labeling → Templates when starting a bulk job.")


if __name__ == "__main__":
    asyncio.run(seed())
