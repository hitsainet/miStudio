---
sidebar_position: 4
title: "The Evidence Ladder"
description: "How miStudio grades what it knows about a circuit — from mined to faithfulness-tested — and why it never overclaims"
---

# The Evidence Ladder

miStudio can *discover* candidate circuits, but discovery is not proof. A cross-layer edge can be a strong statistical association and still not be a mechanism the model actually uses. So every circuit and every edge carries an **evidence rung** — a single, honest grade of how strong the evidence for it is — and the interface uses **server-rendered language** for each rung so nothing ever gets described more confidently than the evidence allows.

## The four rungs

| Rung | Name | What earned it |
|---|---|---|
| **0** | **mined** | Surfaced by statistical mining (PMI, support, null model, held-out replication). An association, disclosed as such. |
| **1** | **attribution-supported** | A gradient-attribution pass agrees in sign and magnitude. A second independent signal — still not causal. |
| **2** | **causally validated** | Intervention (suppress the upstream, measure the downstream) shows a real effect beyond a matched null (Feature 017). |
| **3** | **faithfulness-tested** | The circuit's members are shown necessary and sufficient for the behavior, not just individually implicated (Feature 017). |

A circuit's rung is the **minimum over its edges** — a circuit is only as validated as its weakest link, and a circuit with no edges sits at rung 0.

## Rules that keep it honest

- **The word "causal" never appears below rung 2.** Mined and attribution-supported edges are described as candidates and associations. Only intervention earns causal language, and the language comes from the server, not the client.
- **A failed test never demotes.** If an edge is tested and the effect doesn't hold, that `tested_and_failed` fact is recorded as history — it doesn't erase a rung the edge earned another way, and it's visible so you know what was tried.
- **Promotion is a badge, not a gate.** You can promote an unvalidated circuit into a steering profile — it just carries its rung visibly everywhere, so a rung-0 circuit is never mistaken for a proven one.

The ladder is why you can trust the Circuits panel: it tells you exactly how much is known, and it is structurally prevented from telling you more.
