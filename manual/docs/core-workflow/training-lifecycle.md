---
sidebar_position: 4
title: "Training Lifecycle & Checkpoints"
description: "Stopping a run without losing its SAE, finalizing from a checkpoint, and reclaiming checkpoint disk"
---

# Training Lifecycle & Checkpoints

A training run does not have to reach `total_steps` to be useful. This page
covers what each control actually does to your artifacts, how to keep the SAE
from a run you stop early, and how to reclaim the disk that checkpoints
accumulate.

## What a training produces

A run writes two different things, and the distinction matters:

| Artifact | Path | Who reads it |
|---|---|---|
| **Checkpoints** | `trainings/{id}/checkpoints/checkpoint_{step}/` | the training loop, to resume |
| **Community Standard export** | `trainings/{id}/community_format/` | **everything else** |

Feature extraction, SAE import, steering, circuit capture and Neuronpedia export
all read the **export**, never the checkpoints. A run with checkpoints but no
export has weights on disk that no other part of miStudio can open.

A multi-layer run writes **one checkpoint per layer per step**, all inside a
single `checkpoint_{step}/` directory. A three-layer run saving every 2,000
steps produces three files per save.

## The controls

| Control | Effect on the run | Effect on artifacts |
|---|---|---|
| **Pause** | suspends; GPU is freed | nothing written |
| **Resume** | continues from where it paused | resumes from the latest checkpoint |
| **Stop** | ends the run as `cancelled` | **no export is written** |
| **Stop & Finalize** | ends the run *and* exports | export written from the newest checkpoint |
| **Finalize** | (on an already-stopped run) | export written from the newest checkpoint |

:::warning Stop alone does not save an importable SAE
**Stop** ends the run and leaves its checkpoints in place, but it does **not**
write the Community Standard export. The SAE will not appear under
**Import to SAEs**. If you want to keep the model, use **Stop & Finalize** — or
use **Finalize** afterwards, which does the same thing for a run you already
stopped.
:::

## Stopping a run and keeping its SAE

The usual reason to stop early is that the run has converged: FVU has flattened
and further steps are buying little. See
[SAE Training](/core-workflow/sae-training) for reading those metrics.

Click **Stop & Finalize**. miStudio stops the run, rebuilds the SAEs from the
newest complete checkpoint, and writes `community_format/`. The work runs on the
CPU, so it does not queue behind GPU jobs, and it takes a few minutes for a
large multi-layer run.

When it finishes the card shows **Completed** with an amber
**Finalized early @ N** badge, and **Import to SAEs** becomes available.

### Why it says "Completed"

`Completed` is what unlocks the SAE import path, so a finalized run has to carry
it. But the run genuinely did not reach `total_steps`, so miStudio does **not**
pretend otherwise:

- the progress bar keeps its real value (a run stopped at 20% still shows 20%)
- `finalized_from_step` records the checkpoint step it was built from
- the amber badge states it plainly

If you see **Finalized early @ 10,000** on a run configured for 50,000 steps,
that SAE is the step-10,000 weights. That is usually exactly what you wanted —
but it should never be a surprise.

## Rescuing an already-stopped run

Runs stopped before this feature existed still have their checkpoints. Open the
run and click **Finalize**. Same result: the export is written and the SAE
becomes importable.

**Failed runs** also offer Finalize when they have checkpoints. Because a
crashed run's checkpoints may predate whatever went wrong, miStudio asks you to
confirm before building from them.

## Choosing a checkpoint

Finalize uses the newest **complete** checkpoint step by default.

"Complete" matters for multi-layer runs: if a run was terminated while saving,
the newest step can be missing some of its layers. Exporting that would give you
a partial SAE presented as a whole one, so miStudio skips it and falls back to
the newest step that has every layer, noting the skip in the logs.

## Checkpoint retention

Checkpoints accumulate. A 50,000-step run saving every 2,000 steps across three
layers writes 75 files; at ~1.1 GB each that is a large multiple of the exported
SAE you actually use.

Once a run has produced its export, its intermediate checkpoints are only useful
for resuming. Retention prunes them under a policy you control, in
**Settings → Storage**.

### The policy

| Setting | Default | Meaning |
|---|---|---|
| Enable scheduled pruning | **off** | run the daily sweep at all |
| Dry run | **on** | report what would be deleted, delete nothing |
| Always keep the best checkpoint | on | never delete the lowest-loss checkpoint |
| Keep most recent steps | 2 | how many newest steps to preserve |
| Minimum age (hours) | 24 | never prune anything younger |

### What is never deleted

Regardless of policy, pruning will not touch:

- the **best** (lowest-loss) checkpoint, while that setting is on
- the **newest** step, so a run stays resumable
- any checkpoint of a run that is pending, initializing, running or **paused**
- anything younger than the minimum age

Pruning also operates on **whole steps**. It never deletes some layers of a step
and leaves others, because a partial step cannot be loaded.

### Reviewing before deleting

Deletion is permanent, so the shipped defaults do nothing until you change them.

1. Open **Settings → Storage**
2. Pick a training and click **Preview** — a read-only report of the steps that
   would be pruned, the steps that would be kept, and the space it would free
3. If it looks right, use **Prune now** for that one training, or enable the
   scheduled sweep

Leave **Dry run** on until you have read a report you are happy with. Turning it
off is what makes the next run delete files.

:::tip Best ≠ last
For SAE training the lowest-loss checkpoint is frequently *not* the final one.
Keeping "last N" alone can discard your best weights, which is why
**Always keep the best checkpoint** is on by default.
:::

## Deleting a single checkpoint

Individual checkpoints can be deleted from the **Checkpoints** panel on a
training card. Deleting the **best** checkpoint asks for confirmation first.

## Troubleshooting

**"Import to SAEs" is missing on a stopped run.**
No export was written. Click **Finalize**.

**Finalize reports the step is incomplete.**
The newest checkpoint is missing layers, usually because the worker was
terminated mid-save. Finalize normally falls back automatically; if every step
is incomplete, the message lists the steps present on disk.

**A prune reported 0.00 GB freed.**
Either nothing was eligible — check the preview — or files could not be deleted
(permissions, a read-only mount). The run reports a `files_failed` count in that
case, and the rows are deliberately kept so a later prune can retry.

**A run shows "Finalized early" but I expected a full run.**
It was stopped or crashed before `total_steps`. The badge shows the step its
weights come from.

## See also

- [SAE Training](/core-workflow/sae-training) — configuration and metrics
- [SAE Management](/advanced/external-saes) — importing the export
- [Trainings API](/reference/api/trainings) — endpoints and parameters
