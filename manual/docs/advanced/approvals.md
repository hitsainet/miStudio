---
sidebar_position: 9
title: "Approvals"
description: "Agent requests that wait for your decision, the banner on every screen, and the decided history"
---

# Approvals

Some things an AI agent can ask miStudio to do hold a graphics card for a long time or run code you have not read.
Those requests wait for you. The **Approvals** page lists them, and a banner shows them on every screen.

## What waits for you

| Request | Why it waits |
|---|---|
| Train probes on a graphics card | Holds a graphics card for minutes to hours; every other job queues behind it |
| Evaluate a probe on a graphics card | The same |
| Re-score a probe's calibration on a graphics card | The same |
| Judge probe datasets with miLLM, Spot-check a behaviour with miLLM | Thousands of generations on the card that serves miLLM's users |
| Run operator code in the isolated runner | Code you should read before it runs |
| Steering (only when steering approval is switched on) | Loads a model per task |

Requests that change **miLLM** itself — loading, unloading or deleting a model, changing a model's probe limits,
measuring probe capacity, importing a rule set with operator code — are decided in **miLLM's own Approvals page**,
not here. An agent that asks through miStudio gets miLLM's answer and waits there.

## The banner

While any request waits, a banner sits at the top of every screen with the request cards and an **Open
Approvals** link. It shows the three that expire soonest and says how many more are waiting. **Collapse** reduces
it to one line for this browser session; a new request opens it again. It is not shown on the Approvals page,
which shows the same cards. If the queue cannot be read, the banner says so.

## A request card

Each card shows, in order: what is asked, who asked and when, when it expires, the facts of the request, any extra
detail (the probe run's model, card and datasets; operator code's source and case results with its current
approval state), every argument the agent sent (collapsed), and the request id.

- **Approve** carries the request out once, exactly as shown.
- **Reject** opens a reason field. The reason is optional; left blank, "Rejected by the operator." is recorded.
  A rejected request never runs.

A request you do not decide expires after 24 hours (`mcp_approval_ttl_hours` in Settings, Environment). Steering
requests do not expire. An agent can never approve or reject a request.

## History

History lists every decided request, the most recent decision first: the outcome, what was asked, who asked and
when, who decided and when, the reason and the result. It loads 200 at a time; **Show older** loads more.

| Outcome | Meaning |
|---|---|
| Approved and run | A steering request the backend started; the result names its task |
| Approved, waiting for the agent | Approved; the agent has not used it yet |
| Approved and used | The agent used the approval once |
| Approved (recorded) | Operator code approved for this hash; it stays approved until revoked |
| Rejected | Never ran |
| Expired | Nobody decided in time, or an approval was not used in time |

Times are shown in your browser's time zone; hover over one for the exact instant in Coordinated Universal Time.
The suite has no login, so the decider is always recorded as "operator".
