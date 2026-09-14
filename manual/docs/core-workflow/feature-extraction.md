---
sidebar_position: 4
title: "Feature Extraction"
description: "Recording the evidence — extraction configuration, token filtering, and dead feature filtering"
---

# Feature Extraction — Recording the Evidence

After training (or downloading an external SAE), run an **Extraction Job** to scan your dataset and record which features activate on which tokens.

![Extraction Panel — Completed extraction jobs](/img/miStudio_Extraction_Panel-JobBrowser.jpg)

## Extraction Configuration

miStudio supports two extraction types: **Feature Extraction** (from a trained SAE) and **Activation Extraction** (raw activations for training).

### Feature Extraction Configuration

![Feature Extraction Job Configuration — SAE selection and parameters](/img/miStudio_Extraction_Panel-FeatureExtractionJobConfig_01.jpg)

![Feature Extraction Job Configuration — Token filtering and context](/img/miStudio_Extraction_Panel-FeatureExtractionJobConfig_02.jpg)

### Activation Extraction Configuration

![Activation Extraction Job Configuration](/img/miStudio_Extraction_Panel-ActivationExtractionJobConfig_01.jpg)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Evaluation Samples** | 10,000 | 100–1,000,000 | Dataset samples to scan. More = better coverage but slower. |
| **Top-K Examples** | 100 | 10–1,000 | Max-activating examples saved per feature. More = richer context for labeling. |
| **Batch Size** | Auto | 8–256 | Processing batch size. Auto-detected based on available VRAM. |

## Token Filtering

Control which tokens appear in activation examples. These filters affect both extraction and labeling:

| Filter | Default | Effect |
|--------|---------|--------|
| **Special Tokens** | ✅ On | Removes `<s>`, `</s>`, `<pad>`, etc. |
| **Single Characters** | ✅ On | Removes single-character tokens |
| **Punctuation** | ✅ On | Removes pure punctuation tokens |
| **Numbers** | ✅ On | Removes pure numeric tokens |
| **Fragments** | ✅ On | Blocklist of ~440 affixes — and it contains ordinary words |
| **Stop Words** | ❌ Off | Optionally removes "the", "and", "is", etc. |

:::warning A filter discards the whole example, not just the token

These settings are applied to the **prime token** — the single position where a
feature peaks in a document. When the prime token is filtered, the entire
(document, feature) pair is dropped: no example is stored, and the feature's
recorded activation frequency does not count that document.

So the filters do not merely tidy what you read. They decide which features
clear the dead-feature gate below, and therefore which features exist.

**Fragments** deserves particular care. It is a hand-written blocklist of ~440
entries, not a detector of BPE subwords, and it includes real words —
`land`, `field`, `like`, `some`, `one`, `intra`, `trans`. A worked case: feature
`eb48_00046` was labelled *"technical or specialized identifiers"*, and what it
actually detects is the seam of a word the tokenizer split. The blocklist had
removed every document whose peak landed on a *conventional* affix while letting
unusual splits through — `intra` and `trans` are both blocklisted, and
`intrat|um|oral` and `Aut|otrans|former` survived by landing on the odd piece.
The filter was selecting for the "technical" appearance that produced the label.

If a feature's label looks oddly specific, re-extract with **Fragments** off
before believing it.
:::

:::tip Filter Strategy
Keep most filters ON for cleaner labeling results. Only disable fragment filtering if you're specifically studying tokenization patterns. Enable stop word filtering when you want labels focused on content words.
:::

## Context Window

Each activation example includes surrounding context for interpretation:

| Setting | Default | Description |
|---------|---------|-------------|
| **Prefix Tokens** | 25 | Tokens shown before the activating token |
| **Suffix Tokens** | 25 | Tokens shown after the activating token |

The asymmetric window (25+25=50 tokens of context) is based on research showing this window size captures sufficient context for accurate labeling.

## Dead Feature Filtering

Features that activate too rarely are filtered:

| Setting | Default | Description |
|---------|---------|-------------|
| **Min Activation Frequency** | 0.001 (0.1%) | Features firing less than this rate are excluded as "dead" |

:::warning The rate this gate reads is measured *after* token filtering

`activation_frequency` counts only the samples where a feature's **peak landed
on a token the filters kept**. A feature can fire on 5% of your corpus and still
be cut, if its strongest activations sit on punctuation, on a numeral, or on one
of the ~440 entries in the word-fragment blocklist.

Being cut here is not "left unlabelled" — the feature gets no row at all, and
its stored examples are discarded with it.

Extractions run from 2026-09 onward also record `activation_frequency_true`, the
same count taken **before** filtering, and report `filter_suppressed_neurons` in
the extraction statistics: the number of features that cleared this threshold on
their true rate and were cut anyway. If that number is large for your corpus,
the filters — not the model — decided which features exist.
:::

## Browsing Extracted Features

Once extraction completes, browse the discovered features in the feature browser:

![Feature Browser — Browsing extracted features with labels and statistics](/img/miStudio_Extraction_Panel-FeatureBrowser.jpg)

Click any feature to view its activation examples, token context, and detailed statistics:

![Feature Details — Activation examples and token-level analysis](/img/miStudio_Extraction_Panel-FeatureDetails.jpg)
