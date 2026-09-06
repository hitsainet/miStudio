---
sidebar_position: 8
title: "Prompt-Template Trials"
description: "Compare labeling prompt templates over a fixed panel of features, scored objectively, without overwriting any label"
---

# Prompt-Template Trials

Label quality is set almost entirely by the prompt template. Until now there was no way to change a
template and find out whether the change helped, short of relabeling an entire extraction and
reading the results by hand.

A **trial** runs one template over a fixed set of features — a *panel* — and records the labels it
produced **without writing any of them to your features**. Run a second trial with a different
template over the same panel, and the two are directly comparable.

:::tip A trial never touches your labels
This is the whole point. Running five template variants over a panel would otherwise overwrite your
real labels five times, and the fifth variant would be scored against features the first four had
already rewritten. Trial labels live only in the trial record.
:::

## Where templates come from

Trials test the templates you author under **Templates → Labeling Templates**. That panel is where
the variable under test is defined.

![Labeling Templates — the three templates a trial can choose between, with their sampling parameters](/img/miStudio_Templates_Panel-LabelingTemplates.png)

Each row shows the template's temperature, max tokens and top-p, and which one is the **Default**
(used when a trial names no template). The job counts show how much real labeling each has already
driven.

Opening **Create New** shows every field a trial captures:

![The labeling template editor — system message, user prompt, and sampling parameters](/img/miStudio_Templates_Panel-LabelingTemplateEditor.png)

A trial freezes a **complete copy** of all of it — the system message, the user prompt template, the
sampling parameters — not a reference. Templates are editable, so a run holding only an id would
silently re-describe itself the moment someone tuned the template mid-experiment, and two runs would
appear to differ by a change neither actually used.

## Choosing a panel

A panel is an explicit list of feature ids from one extraction, between 1 and 200 of them.

Panel identity is **content-addressed**: it is a hash of the extraction id plus the sorted feature
ids. Two trials with the same `panel_id` therefore provably covered the same features, and a
comparison across different panels is **refused** rather than reconciled.

Pick features that span the range of what your SAE actually produces. A panel made only of easy,
high-frequency features will rate every template highly and tell you nothing.

## Running a trial

From an agent over MCP:

```
run_labeling_trial(
  extraction_job_id = "extr_...",
  feature_ids       = ["feat_...", "feat_...", ...],
  prompt_template_id = "lpt_...",
  name              = "baseline"
)
```

Or over REST:

```bash
curl -X POST http://mistudio.hitsai.local/api/v1/labeling/trials \
  -H 'Content-Type: application/json' \
  -d '{"extraction_job_id":"extr_...","feature_ids":["feat_a","feat_b"],
       "prompt_template_id":"lpt_...","name":"baseline"}'
```

Both return a `trial_run_id`. Poll `get_labeling_trial` (or `GET /labeling/trials/{id}`) for the
result.

Run a second trial over the **same** `feature_ids` with a different template, then compare.

## Comparing two trials

```
compare_labeling_trials(run_a = "ltr_...", run_b = "ltr_...")
```

The comparison reports, per feature, whether the label and the category changed, plus the overall
change rates.

It refuses more often than it answers, deliberately:

| situation | result |
|---|---|
| the two runs used different panels | **409** — comparing them would produce a number that looks like a template difference and is not one |
| no overlapping features | no verdict, with a reason. Comparing nothing is not comparing |
| every overlapping feature errored in one arm | `inconclusive` — **never** `identical`. Failed labels stringify the same way and would otherwise read as perfect agreement |

## Scoring labels objectively

Change rates tell you *that* two templates differ, not *which is better*. For that, miStudio can
score a label by **detection**: given only the label, can a judge model pick out which passages
activate the feature? A good label scores well above chance; a vague one scores at chance.

Three properties make that number trustworthy, and each exists because its absence produced a
wrong answer:

**The ruler is pinned.** The scoring prompt is a fixed, versioned constant, never one of your
editable templates. The template under test varies; the instrument measuring it must not, or scores
taken on different days are not comparable. A score always records the ruler version that produced
it.

**Negatives come from other features, and the claim is bounded.** miStudio stores only each
feature's top-activating passages, so a feature's own weakest examples are still *positives* — using
them as negatives would punish a correct label. Negatives are drawn from other features instead.
There is no way to certify a passage as non-activating without re-running the model, so the only
claim made is that it falls below the feature's weakest stored activation. That threshold travels
with every score.

**A judge that fails its sanity check produces no score at all.** Before scoring, the judge must
detect a token it was explicitly told to look for, and must *not* score well when handed a
deliberately mismatched label. Failing either yields `judge_unreliable` — never a low score.

:::warning A weak judge is not a bad prompt
Reporting a weak judge's failure as a low score would send you rewriting prompts that were already
good. If you see `judge_unreliable`, point the scorer at a stronger model; the 1.2B model this
cluster serves is not able to grade itself.
:::

### Reading the score

Two numbers are reported per feature, and the gap between them is the interesting part:

- **`ba_hard`** — scored against negatives that contain the feature's own top token in a *different
  sense*. This is the headline: it separates "fires on the word *running*" from "fires on physical
  locomotion".
- **`ba_easy`** — scored against unrelated passages.

A label that only names the surface token aces the easy set and fails the hard one. Measured on a
worked example:

| label under test | overall | `ba_hard` | `ba_easy` |
|---|---|---|---|
| the concept ("locomotion on foot") | 1.00 | 1.00 | 1.00 |
| the surface token ("the word *running*") | 0.75 | **0.50** | 1.00 |
| an empty label | 0.50 | 0.50 | 0.50 |

`ba_easy − ba_hard` is how much of a label's apparent quality is just naming the token.

### How big a difference is real

A panel comparison reports a confidence interval **and** the smallest difference the panel could
have resolved. Roughly 30 features resolves about **6 balanced-accuracy points**. A three-point gap
between two templates on a thirty-feature panel is not a result, and the comparison says so rather
than declaring a winner.

## Keeping a trial honest

Two things are enforced that are easy to get wrong:

- **The example order is seeded per panel and feature**, so two trials present identical passages in
  identical positions. Without this the prompt differs between runs and the template is not the only
  variable.
- **Every batch shown to the judge contains both activating and non-activating passages**, while the
  order within a batch stays unpredictable. A single-class batch is answered uniformly by a
  *correct* judge and would be misread as a broken one; a predictable order lets a judge score well
  by guessing.

## What a trial does not do

- It does not write labels. To adopt a winning label, apply it explicitly — over MCP that is
  `update_feature_label`, which *does* persist and carries `mcp_agent` provenance.
- It does not test detection templates. A template flagged as a detection/scoring template is
  refused as a trial subject: it is the ruler, not the thing being measured.

## Related

- [Auto-Labeling](./auto-labeling.md) — the bulk labeling run a trial is optimizing

![Semantic Labeling — the bulk labeling panel a trial exists to improve](/img/miStudio_Labeling_Panel-SemanticLabeling.png)
- [Enhanced Per-Feature Labeling](./enhanced-labeling.md) — the deeper two-pass path
- [Labeling API reference](../reference/api/features-labeling.md)
