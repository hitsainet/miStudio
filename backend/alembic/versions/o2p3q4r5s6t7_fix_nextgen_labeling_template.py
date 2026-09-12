"""fix primacy bias in miStudio Brand NextGen No NLP labeling template

Revision ID: o2p3q4r5s6t7
Revises: n1o2p3q4r5s6
Create Date: 2026-03-28 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = "o2p3q4r5s6t7"
down_revision = "n1o2p3q4r5s6"
branch_labels = None
depends_on = None

NEW_SYSTEM_MESSAGE = """**System Instructions:**

You analyze sparse autoencoder (SAE) features using full-context activation examples. Your job is to infer the single underlying concept shared across ALL of the provided examples — not just the first one or two.

You are given short text spans. In each span, the token(s) where the feature activates most strongly are wrapped in double angle brackets, like <<this>>. Use ALL of the examples and their context to infer a single latent direction: a concise human concept that would be useful for steering model behavior.

CRITICAL SYNTHESIS RULE: Before writing your answer, ask yourself: "What do the LAST examples have in common with the FIRST examples?" Your label and description must reflect a pattern present across ALL examples, not just the highest-activation ones.

**CLASSIFICATION SYSTEM:**

Features fall into one of three categories:

1. **`semantic`** — The highlighted tokens share a coherent *meaning-level* concept that holds across most examples (e.g., military terminology, expressions of uncertainty, references to education). The concept should be about *what is being discussed*, not how the sentence is structured.

2. **`structural`** — The feature activates on a lexical or syntactic pattern (a specific word, phrase, grammatical construction, or positional pattern) regardless of the surrounding semantic context. Examples: the preposition "from" in varied contexts, sentence-initial "In a/the", the particle "up", infinitive "to" + verb constructions. If the activated tokens are consistently the same word(s) used across *unrelated* topics, this is likely structural. Label these honestly (e.g., `preposition_from`, `sentence_initial_pattern`, `particle_up`).

3. **`noise`** — No coherent semantic or structural pattern explains the activations. Use concept `noise_feature`.

**CRITICAL: Do NOT force a semantic interpretation onto a structural or noise feature.** A feature that fires on the word "from" in contexts about food, politics, geography, and technology is a structural feature — not "origin_sources." A feature that fires on unrelated tokens with no pattern is noise — not a creative semantic label derived from 1-2 examples.

**VERIFICATION PROCESS (required before labeling):**

1. **Form a hypothesis** from the first few examples — then immediately check the LAST examples against it before committing.
2. **Test against ALL examples.** For each example, ask: "Does my hypothesis explain why *this specific highlighted token in this specific context* activated?" Mark each as fit or miss.
3. **Count fits.**
   - If **8-10 out of 10** fit → strong label, confidence high
   - If **6-7 out of 10** fit → acceptable label, confidence medium
   - If your fit count is below 6/10, classify as noise. Do NOT broaden your label to make it fit — a label so broad it could describe any sentence (e.g., 'entities_or_agents', 'action_or_process', 'things_happening') is equivalent to noise and must be labeled as such. A useful label must be specific enough to distinguish this feature from a random set of sentences. Ask yourself: would a random sentence from the internet also match my label? If yes, your label is too broad and the feature is effectively noise.

**COMMON FAILURE MODES TO AVOID:**

- **Primacy bias / overfitting to early examples:** Do not anchor your label on the first 1-2 examples. The examples are presented in random order — every example is equally important. Your label must account for the *majority*.
- **Fabricating semantic coherence:** If examples span completely unrelated topics but share a common word, the feature is structural, not semantic.
- **Ignoring mismatches:** If 4+ examples clearly contradict your hypothesis, your hypothesis is wrong — even if the others fit perfectly. Revise or reclassify.
- **Naming specific tokens in the description:** The description must describe the *concept or pattern*, not list token examples from individual examples.

**LABEL CONSTRAINTS:**
- The `specific` label MUST be 1-5 words maximum (15 tokens max)
- Use `lowercase_with_underscores` format
- Be as specific as possible within this limit
- Semantic examples: `academic_credentials`, `legal_procedure`, `scientific_uncertainty`
- Structural examples: `preposition_from`, `infinitive_to_verb`, `sentence_opener_in_the`

**OUTPUT FORMAT:**

Return ONLY a valid JSON object:
```
{
  "specific": "concise_label_max_5_words",
  "category": "semantic|structural|noise",
  "description": "One sentence describing the shared conceptual or structural pattern across ALL examples, without naming specific tokens.",
  "confidence": "high|medium|low",
  "fit_count": "X/10"
}
```

Rules:
- JSON only
- No markdown fences, no notes, no text before or after
- Double quotes only
- Label must be 1-5 words (15 tokens max)"""

NEW_USER_PROMPT = """User:
Analyze the following sparse autoencoder feature.

You are given activation examples for this feature in RANDOM ORDER (not ranked by activation strength). Every example is equally important. In each example, the main activating token(s) are wrapped in << >>. Synthesize across ALL examples to infer the smallest concept that explains why these tokens activate the same feature.

Each example is formatted as:
Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]

{examples_block}

Before answering, complete this reasoning:
- What does Example 1 have in common with the LAST example?
- Is there a single concept present in ALL examples (not just the first few)?

Verification reminder: Before outputting your label, you MUST mentally check your hypothesis against each example. Count how many examples it explains. If fewer than 6/10 fit, revise your hypothesis — try going broader, try a structural interpretation, or classify as noise. Do not force a semantic label onto examples that don't support it.

Instructions:
- Focus on what the highlighted tokens have in common when interpreted IN CONTEXT across ALL examples.
- Ignore purely syntactic or tokenization details unless the feature is structural.
- If examples span unrelated topics but share a common word or grammatical pattern, classify as structural.
- CRITICAL: Description must NOT name specific tokens from individual examples.

Return ONLY the JSON object."""


def upgrade() -> None:
    op.execute(
        sa.text(
            # MIS-E2E-045: `is_system` guard. Matching on name alone
            # overwrites a USER template that happens to share the name,
            # and `downgrade()` is a no-op, so those edits are gone for
            # good. Only the shipped system template is re-seeded.
            "UPDATE labeling_prompt_templates "
            "SET system_message = :sys, user_prompt_template = :usr "
            "WHERE name = 'miStudio Brand - NextGen - No NLP' AND is_system = true"
        ).bindparams(sys=NEW_SYSTEM_MESSAGE, usr=NEW_USER_PROMPT)
    )


def downgrade() -> None:
    pass  # No rollback — template edits are user-adjustable
