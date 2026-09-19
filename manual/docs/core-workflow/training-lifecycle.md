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
steps produces three weight files per save, plus one `training_state.pt`.

### What a checkpoint holds

| File | Contents |
|---|---|
| `layer_{i}_{hook}/checkpoint.safetensors` | that SAE's weights |
| `training_state.pt` | everything else a resume needs, for every SAE of the step |

`training_state.pt` holds the optimizer's moments, the learning-rate schedule's
position, the mixed-precision loss scale, the dead-latent statistics, the random
number generators and the position in the training data. It is what lets a
resumed run continue **exactly** where it stopped instead of restarting parts of
it. It loads as plain data only (never as executable pickled objects).

It is about **twice the size of the weights**, because Adam keeps two moments per
parameter. A 2,048 × 8,192 SAE is ~134 MB of weights and ~268 MB of state, so a
three-layer step is roughly 0.4 GB of weights plus 0.8 GB of state.

Checkpoints written before September 2026 have no `training_state.pt`. They still
resume, with the caveats under [Resuming](#resuming).

## The controls

| Control | Effect on the run | Effect on artifacts |
|---|---|---|
| **Pause** | suspends; GPU is freed | writes a checkpoint at the step it stops (one step later just after an out-of-memory retry), so a resume repeats nothing |
| **Resume** | continues after the newest complete checkpoint — from a **paused** run or a **failed** one | restores the optimizer, schedule and data position from it — see [Resuming](#resuming) |
| **Stop** | ends the run as `cancelled` | **no export is written** |
| **Stop & Finalize** | ends the run *and* exports | export written from the newest checkpoint |
| **Finalize** | (on an already-stopped run) | export written from the newest checkpoint; refused (`409`) when the export already holds the run's final weights |

**During the post-run evaluation.** A run is marked **Completed** as soon as its full-length export
is saved, before its evaluation starts. **Stop** and **Stop & Finalize** then cancel only the
evaluation (its record ends `cancelled`, naming the control) and finalize nothing. **Pause** responds
`409`. **Finalize** responds `409`, even with `force`, because it would replace the final weights
with a checkpoint's. The card shows no Stop button for a completed run; stop its evaluation with
`POST /api/v1/trainings/{id}/control` and `{"action": "stop"}`.

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

## Resuming

**Resume** continues a run from its **newest complete** checkpoint — not the best
one, which would discard every step trained after it. The run picks up at the
step after the checkpoint, with the optimizer, the learning-rate schedule
(including a decay in progress), the loss scale, the dead-latent statistics, the
random number generators and the position in the training data all restored.
Given the same configuration, it produces the same weights an uninterrupted run
would have.

**A crashed run is resumable too.** A killed worker, an out-of-memory kill or a
rebooted node leaves the run **Failed** with its checkpoints intact, and Resume
continues it exactly as it continues a pause. The card offers Resume once the
run has at least one checkpoint; resuming clears the failure message, so a
running row never carries the traceback of the crash it recovered from.

Two caveats worth knowing:

- **A crash takes up to 30 minutes to become resumable.** While the row still
  reads **Running**, nothing is executing but the system cannot yet tell a dead
  worker from a slow step, so Resume is refused. The stuck-job sweep marks the
  run Failed once it has been silent for thirty minutes, and Resume works from
  then on. (The GPU itself frees sooner — the card's lease expires ten minutes
  after the worker stops renewing it.)
- **Cancelled runs are not resumable.** A Stop is a decision, not an accident.
  Use **Finalize** to export the SAE a cancelled run reached.

Resume is the recovery that keeps training; the alternatives do not. **Finalize**
writes the SAE as it stood at the checkpoint and trains no further, and **Retry**
starts a new run at step 0. For a long run the difference is the whole point: a
crash at step 120,000 of 150,000 loses nothing under Resume.

A checkpoint written before `training_state.pt` existed still resumes from its
weights, and the training log says so with a **LEGACY CHECKPOINT** warning: the
optimizer restarts from zero moments and the learning-rate warmup restarts,
because that state was never saved. The weights, the decoder bias and JumpReLU
thresholds are kept as saved.

## Choosing a checkpoint

Finalize uses the newest **complete** checkpoint step by default.

"Complete" matters for multi-layer runs: if a run was terminated while saving,
the newest step can be missing some of its layers. Exporting that would give you
a partial SAE presented as a whole one, so miStudio skips it and falls back to
the newest step that has every layer, noting the skip in the logs.

## Checkpoint retention

Checkpoints accumulate. A 50,000-step run saving every 2,000 steps across three
layers writes 75 weight files; at ~1.1 GB each that is a large multiple of the
exported SAE you actually use. Each of those 25 steps also carries a
`training_state.pt` about twice the size of that step's weights, so a step's
whole footprint is roughly three times its weights. Previews and prune reports
count it, and it is deleted together with the step's last layer.

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

**A resumed run logged "LEGACY CHECKPOINT".**
The checkpoint predates `training_state.pt`, so only its weights could be
restored. The run continues from those weights, but the optimizer and the
learning-rate warmup start over. Any checkpoint the run writes from then on
includes the full state.

**Resume reports no complete checkpoint.**
Every step on record is missing at least one layer's file, usually because the
worker was terminated while saving. The run cannot be resumed; **Finalize** an
earlier complete step if one exists, or start the run again.

**A run shows "Finalized early" but I expected a full run.**
It was stopped or crashed before `total_steps`. The badge shows the step its
weights come from.

## See also

- [SAE Training](/core-workflow/sae-training) — configuration and metrics
- [SAE Management](/advanced/external-saes) — importing the export
- [Trainings API](/reference/api/trainings) — endpoints and parameters
