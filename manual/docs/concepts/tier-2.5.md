---
sidebar_position: 5
title: "Lag-0 and Tier-2.5"
description: "Why same-position discovery can't see attention-mediated circuits — and how miStudio is built to lift that limit without re-capturing"
---

# Lag-0 and Tier-2.5

## What "lag-0" means

Circuit discovery mines **same-token-position** co-activation: it asks whether an upstream feature and a downstream feature tend to fire on the *same* token. That is exactly right for **computed** structure — a feature at one layer influencing a feature at a higher layer *within a single position's residual stream*.

It is blind to the circuits people most want to name: **attention-mediated, cross-position** ones. When a feature fires on a name three sentences ago and drives a pronoun-resolution feature *here*, the two never share a token position — so a same-position count can't see the link. This is the **lag-0 limitation**, and miStudio discloses it on every discovery surface: the UI Discovery tab, every MCP tool description, and a `lag0_disclosure` line in every run report.

## Why the limit is liftable without re-capturing

miStudio's capture store and its portable circuit contract were built **position- and attention-aware from day one**:

- every activation event stores its **token position** as a first-class column;
- capture can record an **attention sidecar** (which key positions each query token attended to);
- the circuit contract reserves an `attention_mediated` edge type and nullable position/head fields.

Because those fields already exist (populated or explicitly null), enabling cross-position discovery is a matter of turning on new mining — **not** migrating any stored circuit or re-running any capture that already recorded attention.

## Tier-2.5: the designed fast-follow

**Tier-2.5** is the name for attention-mediated cross-position mining. Its design is fully specified (attention-weighted candidate generation, a shuffled-join null that tests the attention routing, frozen-attention attribution, and position-restricted causal validation). The *implementation* is a named fast-follow — deliberately deferred so the honest lag-0 tier ships first — and when it lands, it populates contract fields that already exist rather than changing the schema.

The point: the interpretability studio tells you today exactly what it can and can't discover, and it's engineered so that lifting the limit is future work, not a migration.
