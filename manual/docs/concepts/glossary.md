---
sidebar_position: 6
title: "Glossary"
description: "The words and abbreviations miStudio uses, in plain language"
---

# Glossary

The words and abbreviations miStudio uses, in plain language. The in-app **Help** screen links
here from its "Words used here" topic. Each entry is a short definition; the linked pages explain
the idea in full.

## Words

| Word | What it means |
|---|---|
| **Activation** | How strongly a feature (or a neuron) responds to one token. Zero means it did not fire. |
| **Circuit** | A set of features at different layers that work together, such as an earlier feature that drives a later one. See [Circuits](/core-workflow/circuits). |
| **Cluster** | A group of features that respond most strongly to the same word, split by the context the word appears in. See [Clusters](/core-workflow/feature-groups). |
| **Dead feature** | A feature that almost never fires on the text it was measured on. Extraction drops them. |
| **Evidence rung** | The grade of how strong the evidence for a circuit or one of its links is, from "mined" to "faithfulness-tested". See [The Evidence Ladder](/concepts/evidence-ladder). |
| **Extraction** | A job that runs the model over text and records, for each feature, the passages where it fires most strongly. See [Feature Extraction](/core-workflow/feature-extraction). |
| **Feature** | One direction a sparse autoencoder learned, which tends to stand for one idea. Features are what you label, group and steer. |
| **Hook point** | The place inside a layer where miStudio reads (or changes) the model's activations, for example the residual stream after the layer. |
| **J-Lens** | The screen that shows what the model is poised to say at each token and layer. See [J-Lens](/core-workflow/jlens). |
| **Jacobian lens** | A fitted, per-layer view used by J-Lens that recovers what the model is preparing to say earlier in the model than the logit lens does. It needs a lens fitted for that model. |
| **Label** | A short name and category for a feature, written by a language model from the feature's strongest passages. See [Auto-Labeling](/core-workflow/auto-labeling). |
| **Layer** | One step of the model's processing. A model has a stack of them; earlier layers tend to handle spelling and grammar, later ones meaning. |
| **Logit lens** | Reading a layer's activations through the model's own output step, to see which tokens they point towards. It needs no training. |
| **Probe** | A small detector that reads one layer and answers one question about the text, such as "is this sarcastic?". See [Probe Monitors](/core-workflow/probe-monitors). |
| **Quantization** | Storing a model's numbers in a smaller format so it fits in less memory, at a small cost in accuracy. |
| **Residual stream** | The running total each layer reads from and adds to. Most sparse autoencoders and probes read it. |
| **Sparse autoencoder** | A small network trained to rewrite one layer's activations as a few active features out of many. See [SAE Training](/core-workflow/sae-training). |
| **Steering** | Turning chosen features up or down while the model writes, to test what they do. See [Model Steering](/core-workflow/steering). |
| **Strength** | How far steering pushes a feature. Too high and the text falls apart. |
| **Template** | A saved set of settings, so an experiment can be repeated or shared. See [Templates](/advanced/templates). |
| **Token** | A piece of text the model reads, often part of a word. |
| **Tokenization** | Splitting text into a particular model's tokens. Each model family splits text differently, so a dataset is tokenized once per model. |

## Abbreviations

Every abbreviation the in-app Help uses is listed here; a test in the app's frontend fails if one
is missing.

| Abbreviation | Stands for | What it means |
|---|---|---|
| **AI** | artificial intelligence | Software that does work a person would otherwise do; here, the agents that call miStudio's tools and ask for your approval. See [Approvals](/advanced/approvals). |
| **API** | application programming interface | The web addresses the frontend, scripts and agents call to drive miStudio. |
| **AUROC** | area under the receiver operating characteristic curve | One number for how well a detector, such as a probe, separates two classes: 0.5 is chance and 1.0 is perfect. |
| **FPR** | false-positive rate | The share of negative examples a detector wrongly flags as positive; a probe's threshold is set to spend a chosen FPR, such as 1%. |
| **FVU** | fraction of variance unexplained | How much of a layer's activity a sparse autoencoder fails to reconstruct. Lower is better. |
| **JSON** | JavaScript Object Notation | A plain-text format for structured data, such as `{"answer": "yes"}`. Judge prompts usually ask the judge to reply in it. |
| **GPU** | graphics processing unit | The graphics card that runs the model, training and extraction. |
| **LLM** | large language model | A model that reads and writes text, such as the model you study or the one that writes labels. |
| **MCP** | Model Context Protocol | The protocol AI agents use to call miStudio's tools. See [MCP Server](/advanced/mcp-server). |
| **PIN** | personal identification number | The code that locks the API Keys, Storage and Environment tabs of Settings. A convenience lock, not a login. See [Settings Reference](/advanced/settings-reference#pin-protection). |
| **PII** | personally identifiable information | Data that identifies a person: a name, an email address, a phone number, an account number. One of the seeded judge prompts detects it. |
| **PMI** | pointwise mutual information | How much more often two features fire together than chance would predict. Circuit discovery uses it. |
| **SAE** | sparse autoencoder | See *Sparse autoencoder* above. |
| **TF-IDF** | term frequency–inverse document frequency | A way of weighting words by how distinctive they are, used to split clusters by context. |
