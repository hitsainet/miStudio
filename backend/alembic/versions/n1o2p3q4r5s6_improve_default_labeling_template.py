"""improve default labeling template for synthesis across all examples

Revision ID: n1o2p3q4r5s6
Revises: m9n0o1p2q3r4
Create Date: 2026-03-28 22:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "n1o2p3q4r5s6"
down_revision = "m9n0o1p2q3r4"
branch_labels = None
depends_on = None

NEW_SYSTEM_MESSAGE = """You analyze sparse autoencoder (SAE) features using full-context activation examples. Your ONLY job is to infer the single underlying conceptual meaning shared across ALL of the provided examples — not just the first one or two.

You are given short text spans. In each span, the token(s) where the feature activates most strongly are wrapped in double angle brackets, like <<this>>. Use ALL of the examples and their context to infer a single latent direction: a concise human concept that would be useful for steering model behavior.

CRITICAL SYNTHESIS RULE: Before writing your answer, ask yourself: "What do the LAST examples have in common with the FIRST examples?" Your label and description must reflect a pattern present across ALL examples, not just the highest-activation ones.

LABEL CONSTRAINTS:
- The 'specific' label MUST be 1-5 words maximum (15 tokens max)
- Use lowercase_with_underscores format
- Be as specific as possible within this limit
- Examples: 'trump_mentions', 'legal_procedure', 'scientific_uncertainty', 'greeting_phrases'

You must NOT:
- describe grammar, syntax, token types, or surface patterns
- list the example tokens back
- say "this feature detects words like..."
- label the feature with only a grammatical category
- describe frequency, morphology, or implementation details
- use labels longer than 5 words
- name specific tokens from individual examples in the description

If ANY coherent conceptual theme exists across the examples, use category 'semantic'.
If no coherent theme exists, use category 'system' and concept 'noise_feature'.

You must return ONLY a valid JSON object in this structure:
{
  "specific": "concise_label_max_5_words",
  "category": "semantic_or_other",
  "description": "One sentence describing the shared conceptual meaning found across ALL examples, without naming specific tokens."
}

Rules:
- JSON only
- No markdown
- No notes
- No code fences
- No text before or after the JSON
- Double quotes only
- Label must be 1-5 words (15 tokens max)"""

NEW_USER_PROMPT = """Analyze sparse autoencoder feature {feature_id}.

You are given activation examples for this feature in RANDOM ORDER (not ranked by importance). Every example is equally important. In each example, the main activating token(s) are wrapped in << >>. Synthesize across ALL examples to infer the smallest semantic concept that explains why these tokens activate the same feature.

Each example is formatted as:
  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

Examples:

{examples_block}

Before answering, complete this reasoning:
- What does Example 1 have in common with the LAST example?
- Is there a single concept present in ALL examples (not just the first few)?

Instructions:
- Focus on what the highlighted tokens have in common when interpreted IN CONTEXT across ALL examples.
- Ignore purely syntactic or tokenization details.
- Prefer semantic, conceptual, or functional interpretations.
- If you cannot find a coherent concept shared across all examples, treat this as a noise feature.
- CRITICAL: Label must be 1-5 words maximum (15 tokens max), using lowercase_with_underscores.
- CRITICAL: Description must NOT name specific tokens from individual examples.

Good label examples: 'trump_mentions', 'legal_terms', 'scientific_uncertainty', 'greeting_phrases', 'negative_sentiment'
Bad label examples: 'words_that_appear_in_legal_documents_and_contracts' (too long)

Return ONLY this exact JSON object:
{
  "specific": "concise_label",
  "category": "semantic_or_other",
  "description": "One sentence describing the shared conceptual meaning across ALL examples, without naming specific tokens."
}"""


def upgrade() -> None:
    op.execute(
        sa.text(
            # MIS-E2E-045: `is_system` guard. Matching on name alone
            # overwrites a USER template that happens to share the name,
            # and `downgrade()` is a no-op, so those edits are gone for
            # good. Only the shipped system template is re-seeded.
            "UPDATE labeling_prompt_templates "
            "SET system_message = :sys, user_prompt_template = :usr "
            "WHERE name = 'miStudio Internal - Full Context' AND is_system = true"
        ).bindparams(sys=NEW_SYSTEM_MESSAGE, usr=NEW_USER_PROMPT)
    )


def downgrade() -> None:
    pass  # No rollback — template edits are user-adjustable
