/**
 * Every tool a row of this app's approval queue can carry has a plain-words label (ENH-008 §2.1; ENH-009 042
 * FR-042.28; FTASKS 12.23, control K-12.41).
 *
 * The set is GENERATED from the backend, never typed here: `approvalTools.fixture.json` holds the live
 * `APPROVABLE_TOOLS` of `api/v1/endpoints/mcp_approvals.py` and the history tool names (the `millm_*` rows the
 * e008 migration drained, plus the legacy-identifier export's `history_tool` records). The backend test
 * `test_approval_tools_fixture.py` fails when the fixture drifts from those authorities.
 */
import { describe, it, expect } from 'vitest';
import fixture from './approvalTools.fixture.json';
import { ACTION_LABELS, actionLabel } from './approvalLabels';

describe('approvalLabels', () => {
  it('every approvable or history tool has a label', () => {
    const tools = [...fixture.approvable, ...fixture.history];
    expect(fixture.approvable.length).toBeGreaterThan(3);
    expect(fixture.history.length).toBeGreaterThan(0);
    const missing = tools.filter((t) => !(t in ACTION_LABELS));
    expect(missing).toEqual([]);
    for (const t of tools) expect(actionLabel(t)).not.toBe(t);
  });
});
