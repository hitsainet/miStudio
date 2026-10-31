/**
 * ⚠ A REFRESH USED TO LAND ON "RUNS" WHATEVER YOU WERE LOOKING AT.
 *
 * `activeTab` defaulted to `'runs'` and the store had `devtools` but no `persist`, so a reload
 * discarded a navigation choice for no reason. A control that removed the persistence left the
 * suite green, which is why this file exists.
 */
import { beforeEach, describe, expect, it } from 'vitest';

import { useProbeMonitorsStore } from '../probeMonitorsStore';

const KEY = 'probe-monitors';

describe('the probes sub-tab survives a reload', () => {
  beforeEach(() => window.localStorage.clear());

  it('writes the chosen tab to storage', () => {
    useProbeMonitorsStore.getState().setActiveTab('probes');
    const saved = JSON.parse(window.localStorage.getItem(KEY) ?? '{}');
    expect(saved.state?.activeTab).toBe('probes');
  });

  it('does NOT persist server state, which would render stale as current', () => {
    /*
     * `runs`, `probes`, `datasets`, `judgeRuns` and `report` all carry live status. A persisted
     * copy would show a run as `running` after it finished, or a probe that has been deleted —
     * stale data presented as current, which is worse than a spinner.
     */
    useProbeMonitorsStore.setState({
      activeTab: 'judge',
      runs: [{ id: 'pmr_x', status: 'running' }] as never,
      probes: [{ id: 'pm_x' }] as never,
    });
    const saved = JSON.parse(window.localStorage.getItem(KEY) ?? '{}');
    expect(saved.state?.activeTab).toBe('judge');
    expect(saved.state?.runs).toBeUndefined();
    expect(saved.state?.probes).toBeUndefined();
    expect(saved.state?.report).toBeUndefined();
  });
});
