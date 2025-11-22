# miStudio Prompt Templates for SAE Feature Interpretation

This document collects the JSON Postman-style templates for:

1. **miStudio Internal Format** (full-context interpretation prompt)
2. **Anthropic-Style Format** (full-context + logit effects)
3. **EleutherAI-Style Detection Format** (explanation scoring)

Each section includes:
- A short description of when to use the template.
- A JSON example suitable for Postman / HTTP clients targeting the OpenAI `chat/completions` API.
- Notes on which fields are **static** and which must be **dynamically injected** by your application (e.g., Claude Code).

All JSON examples assume:
- You will replace `YOUR_OPENAI_API_KEY` with a real key at runtime.
- You will substitute runtime variables for `#{...}` placeholders.

---

## 1. miStudio Internal Format – Full-Context Feature Labeling

**Reference:** miStudio internal

**Purpose:**

Use this request to generate a concise human label for a single SAE feature using **full-context activation examples**, where the prime token(s) are wrapped in `<< >>`. This is your default, scalable labeling format.

**Dynamic fields (to inject):**

- `#{feature_id}` – your internal feature identifier (e.g., `18_00000`).
- `#{examples_block}` – multi-line text of examples, e.g.
  ```
  Example 1 (activation: 0.007): commercial and residential r ena issuance of <<sorts>> . The last of four phases of a massive
  Example 2 (activation: 0.007): talk about women and women ’s issues l <<ately>> , a nod to the emergence of contra
  ```

**JSON template:**

```json
{
  "info": {
    "name": "miStudio - Label Feature (Full Context)",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Label Feature (Full Context)",
      "request": {
        "method": "POST",
        "header": [
          { "key": "Content-Type", "value": "application/json" },
          { "key": "Authorization", "value": "Bearer YOUR_OPENAI_API_KEY" }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"model\": \"gpt-4.1-mini\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You analyze sparse autoencoder (SAE) features using full-context activation examples. Your ONLY job is to infer the single underlying conceptual meaning shared by the most strongly-activating tokens, taking into account both the highlighted token(s) and their surrounding context.\\n\\nYou are given short text spans. In each span, the token(s) where the feature activates most strongly are wrapped in double angle brackets, like <<this>>. Use all of the examples and their context to infer a single latent direction: a 1–2 word human concept that would be useful for steering model behavior.\\n\\nYou must NOT:\\n- describe grammar, syntax, token types, or surface patterns\\n- list the example tokens back\\n- say \\\"this feature detects words like...\\\"\\n- label the feature with only a grammatical category\\n- describe frequency, morphology, or implementation details\\n\\nIf ANY coherent conceptual theme exists, use category ‘semantic’.\\nIf no coherent theme exists, use category ‘system’ and concept ‘noise_feature’.\\n\\nYou must return ONLY a valid JSON object in this structure:\\n{\\n  \\\"specific\\\": \\\"one_or_two_word_concept\\\",\\n  \\\"category\\\": \\\"semantic_or_other\\\",\\n  \\\"description\\\": \\\"One sentence describing the real conceptual meaning represented by this feature.\\\"\\n}\\n\\nRules:\\n- JSON only\\n- No markdown\\n- No notes\\n- No code fences\\n- No text before or after the JSON\\n- Double quotes only\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Analyze sparse autoencoder feature #{feature_id}.\\nYou are given some of the highest-activating examples for this feature. In each example, the main activating token(s) are wrapped in << >>. Use ALL of the examples, including their surrounding context, to infer the smallest semantic concept that explains why these tokens activate the same feature.\\n\\nEach example is formatted as:\\n  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]\\n\\nExamples:\\n\\n#{examples_block}\\n\\nInstructions:\\n- Focus on what the highlighted tokens have in common when interpreted IN CONTEXT.\\n- Ignore purely syntactic or tokenization details.\\n- Prefer semantic, conceptual, or functional interpretations (e.g., ‘legal_procedure’, ‘feminist_politics’, ‘scientific_uncertainty’).\\n- If you cannot find a coherent concept, treat this as a noise feature.\\n\\nReturn ONLY this exact JSON object:\\n{\\n  \\\"specific\\\": \\\"concept\\\",\\n  \\\"category\\\": \\\"semantic_or_other\\\",\\n  \\\"description\\\": \\\"One sentence describing the conceptual meaning.\\\"\\n}\"\n    }\n  ],\n  \"temperature\": 0.2,\n  \"max_tokens\": 256,\n  \"top_p\": 0.9\n}"
        },
        "url": {
          "raw": "https://api.openai.com/v1/chat/completions",
          "protocol": "https",
          "host": ["api", "openai", "com"],
          "path": ["v1", "chat", "completions"]
        }
      }
    }
  ]
}
```

---

## 2. Anthropic-Style Format – Full Context + Logit Effects

**Reference:** Anthropic Auto-Interp style

**Purpose:**

Use this request when you want richer feature explanations that incorporate **logit effects** (promoted/suppressed tokens) in addition to context. This is more expensive than the miStudio baseline but yields higher-quality explanations for subtle or important features.

**Dynamic fields (to inject):**

- `#{feature_id}` – feature identifier.
- `#{examples_block}` – multi-line examples, each with activation value and `<< >>`-highlighted tokens.
- `#{top_promoted_tokens}` – JSON array or comma-separated list of most promoted tokens.
- `#{top_suppressed_tokens}` – JSON array or comma-separated list of most suppressed tokens.

**JSON template:**

```json
{
  "info": {
    "name": "miStudio - Label Feature (Anthropic Style)",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Label Feature (Anthropic-style, with logits)",
      "request": {
        "method": "POST",
        "header": [
          { "key": "Content-Type", "value": "application/json" },
          { "key": "Authorization", "value": "Bearer YOUR_OPENAI_API_KEY" }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"model\": \"gpt-4.1-mini\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You analyze sparse autoencoder (SAE) features using rich activation examples.\\nYour job is to infer the single underlying conceptual meaning represented by a feature.\\n\\nYou are given multiple short text spans for a single feature. In each span, the token(s) where this feature activates most strongly are wrapped in << >>. You are also given the feature's activation strength on that span and the tokens whose logits it promotes or suppresses.\\n\\nUse ALL the examples, their context, and the logit effects to infer ONE human concept that best describes this feature. Prefer semantic or functional concepts over surface-level descriptions.\\n\\nReturn ONLY a JSON object in this format:\\n{\\n  \\\"specific\\\": \\\"one_or_two_word_concept\\\",\\n  \\\"category\\\": \\\"semantic_or_other\\\",\\n  \\\"description\\\": \\\"One sentence describing the conceptual meaning represented by this feature.\\\"\\n}\\n\\nRules:\\n- JSON only\\n- No markdown\\n- No code fences\\n- No commentary before or after the JSON\\n- Double quotes only\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Analyze sparse autoencoder feature #{feature_id}.\\nYou are given some of the highest-activating examples for this feature, along with its logit effects.\\n\\nEach example is formatted as:\\n  Example N (activation: A_N): [prefix tokens] <<prime tokens>> [suffix tokens]\\n\\nExamples:\\n\\n#{examples_block}\\n\\nLogit effects for this feature:\\n- Top promoted tokens: #{top_promoted_tokens}\\n- Top suppressed tokens: #{top_suppressed_tokens}\\n\\nInstructions:\\n- Focus on what the highlighted tokens represent IN CONTEXT.\\n- Use the promoted/suppressed tokens to refine your understanding of what the feature is doing to the model's output distribution.\\n- Prefer semantic labels that could be used to steer or monitor the model.\\n- If no coherent pattern exists, use category = \\\"system\\\" and specific = \\\"noise_feature\\\".\\n\\nReturn ONLY this JSON object:\\n{\\n  \\\"specific\\\": \\\"concept\\\",\\n  \\\"category\\\": \\\"semantic_or_other\\\",\\n  \\\"description\\\": \\\"One sentence describing the conceptual meaning.\\\"\\n}\"\n    }\n  ],\n  \"temperature\": 0.2,\n  \"max_tokens\": 256,\n  \"top_p\": 0.9\n}"
        },
        "url": {
          "raw": "https://api.openai.com/v1/chat/completions",
          "protocol": "https",
          "host": ["api", "openai", "com"],
          "path": ["v1", "chat", "completions"]
        }
      }
    }
  ]
}
```

---

## 3. EleutherAI-Style Detection Format – Explanation Scoring

**Reference:** EleutherAI detection-style scoring

**Purpose:**

Use this request after you have an explanation (from miStudio or Anthropic prompts) to **evaluate** whether the explanation correctly describes the feature. This is a scalable, low-cost scoring method: the model must decide for each example whether the feature is present (1) or absent (0) given only the explanation.

**Dynamic fields (to inject):**

- `#{feature_explanation}` – the explanation text or JSON for the feature (you may stringify the JSON explanation or pass only the critical fields).
- `#{example_1} ... #{example_N}` – plain-text examples (usually a mix of activating and non-activating examples). Numbered in order.

**JSON template:**

```json
{
  "info": {
    "name": "miStudio - Score Feature (EleutherAI Detection)",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Score Feature (Detection)",
      "request": {
        "method": "POST",
        "header": [
          { "key": "Content-Type", "value": "application/json" },
          { "key": "Authorization", "value": "Bearer YOUR_OPENAI_API_KEY" }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"model\": \"gpt-4.1-mini\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      \"content\": \"You evaluate whether text examples contain a specific sparse autoencoder feature.\\nYou are given a feature explanation and a list of text examples.\\n\\nFor each example, you must decide whether the feature is present (1) or absent (0) based ONLY on the explanation.\\n\\nReturn ONLY a JSON array of 0s and 1s, one per example, in order.\\nExample output: [0, 1, 1, 0, 0]\"\n    },\n    {\n      \"role\": \"user\",\n      \"content\": \"Feature explanation:\\n#{feature_explanation}\\n\\nDecide for each example whether it contains this feature.\\n\\nExamples:\\n1. #{example_1}\\n2. #{example_2}\\n3. #{example_3}\\n...\\nN. #{example_N}\\n\\nReturn ONLY a JSON array of 0s and 1s in this form:\\n[0, 1, 1, 0, ...]\"\n    }\n  ],\n  \"temperature\": 0.0,\n  \"max_tokens\": 64,\n  \"top_p\": 1.0\n}"
        },
        "url": {
          "raw": "https://api.openai.com/v1/chat/completions",
          "protocol": "https",
          "host": ["api", "openai", "com"],
          "path": ["v1", "chat", "completions"]
        }
      }
    }
  ]
}
```

---

## 4. How to Use These Templates in Code (Claude Code / other orchestrators)

At a high level, your orchestration code should:

1. **Build dynamic content:**
   - Fill in `#{feature_id}`, `#{examples_block}`, logit token lists, and example lists from your SAE, dataset, and logit-lens computations.

2. **Select the template based on the task:**
   - Use **miStudio Internal** for default feature labeling.
   - Use **Anthropic-style** when you need higher-fidelity explanations or when logit effects are especially informative.
   - Use **EleutherAI detection** when you need an automated score of explanation quality.

3. **String-substitute into the JSON:**
   - Programmatically replace the `#{...}` placeholders with serialized values.
   - Send the resulting JSON to the OpenAI `chat/completions` endpoint.

4. **Parse responses:**
   - For labeling: parse the returned JSON object with `specific`, `category`, and `description` fields.
   - For detection: parse the returned JSON array of 0/1 and compute precision/recall or a single quality score.

These three formats together give you a complete path for:
- Generating feature labels (miStudio / Anthropic), and
- Evaluating them at scale (EleutherAI detection style)."}

