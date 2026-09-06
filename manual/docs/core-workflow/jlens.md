---
sidebar_position: 11
title: "J-Lens"
description: "Read what the model is poised to say at every layer and token position — the readout view, and what it does and does not prove"
---

# J-Lens — Reading What the Model Is Poised to Say

**J-Lens** answers one question by looking rather than by writing code: *at this point in
the prompt, at this depth in the model, what is the model poised to say?*

It sits immediately before **Steering** in the sidebar, because reading comes before
intervening.

## What a readout is

At every (token position, layer) pair, J-Lens projects the residual stream through the
model's own final normalisation and unembedding, and reports the top-ranked tokens. That
projection is the **logit lens**, and it needs no fitted artifact — it works on any model
you have loaded, from the moment you load it.

A second lens, the **Jacobian lens**, replaces the identity projection with a fitted
per-layer matrix. It recovers content earlier in the stack than the logit lens does. It
requires an artifact fitted for that specific model, and until one exists the Jacobian and
Diff modes are **visibly disabled with the reason stated**. J-Lens will never show you
logit-lens data under a Jacobian label.

## Using it

1. Pick a model and enter a prompt, then **Read out**.
2. The grid is **layer (rows) × token position (columns)**. The top cell of each column is
   the output end of the stack.
3. **Hover** any cell for its full top-k readout. **Click** a token in that list to **pin**
   it.
4. With tokens pinned, the grid becomes a **rank heatmap** over those tokens — stronger
   colour is a better rank — and a **rank-vs-layer chart** appears for the selected
   position. Lower is stronger, and gaps are layers where the token left the top-k
   entirely.
5. The **by-layer rail** on the right lists the readout at every layer for the selected
   position. Clicking a layer there selects it, so the whole panel is usable without a
   mouse.

Pins are dropped when you change model — they are token strings from a different
vocabulary, and carrying them across would draw empty lines that look like a measured
absence.

## What a readout does *not* prove

This is the part worth reading twice.

A readout is **evidence rung 0**. A concept appearing in one is **not a causal claim**: it
says the direction was present, not that the model used it. To raise the rung, run a
coordinate swap with a matched control — see the [evidence ladder](/concepts/evidence-ladder).

Three further limits are stated in the panel itself, because each is easy to get wrong:

- **Readouts are limited to concepts with single-token names.** A concept the vocabulary
  has no single token for cannot appear, however strongly the model represents it.
- **A readout that resists interpretation is not a null result.** It may be averaging
  noise, a multi-token concept, or genuine content nobody has named yet. Readouts in the
  earliest layers are *expected* to be uninterpretable and are marked **diffuse** rather
  than presented as content.
- **Absence of a signal is not evidence of absence.** Not finding a concept in a readout
  does not mean the underlying computation did not occur.

## Bands

Published work on this technique divides the stack into sensory / workspace / motor bands.
**miStudio draws no bands unless a band report has been computed for the model you are
looking at**, and says so when there is none.

This is deliberate and it is a product rule, not an oversight. The boundaries in the
literature were measured on a specific model. Applying them to a different model —
different depth, different architecture — would produce authoritative-looking shading that
means nothing. There is no default band value anywhere in the product for this reason.

## Provenance

The footer states what produced the readout. For the logit lens it states plainly that **no
artifact is involved** — the readout comes from the model's own unembedding and final norm.
When a Jacobian artifact is in use, the strip carries its identity and the full construction
recipe: target layer, attention-gradient treatment, position scope, aggregation, corpus,
prompt count, sequence length and dtype.

## Fitting a lens

The Jacobian lens needs an artifact fitted for **the exact weights you are reading**. Most
models have no pre-fitted lens, so fitting is the normal path rather than a fallback.

The **J-lens artifact** strip above the readout shows whether one exists for the selected
model, and lets you run the validation suite against it. Fitting itself is a GPU job —
queue it from an agent with `fit_jlens_artifact`, or via `POST /api/v1/jlens/fit`. It
refuses a corpus below **100 prompts**, because an under-fitted lens is indistinguishable
from a good one by inspection.

### Two verdicts, and they are not the same

Validation reports six checks and two verdicts:

- **Serviceable** — the four checks that bear on local correctness passed, so miStudio can
  read out with this artifact.
- **Cleared for handover** — all six passed, including the two that compare against a live
  external consumer. Those two cannot run without that consumer, so a freshly fitted
  artifact is normally *serviceable but not yet cleared*. That is not a fault.

An artifact fitted for a different model is **refused**, not adapted. A base model and its
instruction-tuned variant have different weights, and a lens from one applied to the other
produces a fluent, wrong readout — so the check is on identity, not on name similarity.

## Limits today

- No band report exists yet, so no band shading appears anywhere.
- **The first readout for a model takes about a minute.** A J-space readout needs the whole
  model resident for a forward pass, so the first request loads it; the panel shows
  "loading the model" while that happens. Subsequent readouts on the same model are fast.
- One model stays resident at a time. Switching models evicts the previous one, because a
  readout needs the whole model loaded and this workbench shares a GPU with serving.
- Readout cost grows with the number of positions, so very long prompts are refused rather
  than trickled in.
