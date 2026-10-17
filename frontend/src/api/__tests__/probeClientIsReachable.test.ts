/**
 * ⚠ EVERY METHOD ON THE PROBE API CLIENT MUST HAVE A CALLER.
 *
 * This file exists because five capabilities were written, tested and unreachable. An audit of
 * `probeMonitorsApi`'s methods against their callers found FOUR with none —
 *
 *     submitJudgeRun    cancelJudgeRun    evaluate    score
 *
 * — plus one route (`GET /probes/{id}/score/{task_id}`) with no client method at all. The damage
 * was not cosmetic: rung 3 needs a judge run and rungs 1-2 need `evaluate`, so **the evidence
 * ladder could not be climbed from the UI at all**, and `score` — pointing a probe at your own
 * text — is the single most direct thing anyone wants from a detector.
 *
 * This is the repo's reachability rule one layer out. The MCP version of this defect had 16 tools
 * registered nowhere; the backend version had a score endpoint with no way to read it back; this
 * is the frontend version. All three look identical from inside: green tests over a capability no
 * user can reach.
 *
 * ⚠ IT READS THE CLIENT, NOT A HAND-KEPT LIST. A list would have to be updated by the same person
 * who forgot to wire the method, which is exactly who cannot be relied on to update it.
 */
import { describe, expect, it } from 'vitest';

/*
 * ⚠ SOURCES COME FROM VITE'S GLOB, NOT `node:fs`. The first version imported `node:fs`,
 * `node:path` and used `__dirname`, which the test-file type-check has no types for — three new
 * errors against a down-only ratchet. `import.meta.glob(..., { query: '?raw', eager: true })` is
 * the same read with no node types and no `__dirname`.
 */
const RAW = import.meta.glob('../../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/**
 * Glob keys are relative to THIS file, so `src/api/probeMonitors.ts` arrives as
 * `../probeMonitors.ts` — an `endsWith('api/probeMonitors.ts')` matches nothing and the guard
 * then crashes on undefined instead of asserting. `mustFind` makes a miss say which pattern
 * failed, so a guard that stops finding its own inputs cannot be mistaken for a passing one.
 */
function mustFind(suffix: string): string {
  const key = Object.keys(RAW).find((k) => k.endsWith(suffix));
  if (!key) {
    throw new Error(
      `this guard could not find ${suffix} among ${Object.keys(RAW).length} globbed sources — ` +
        'fix the pattern rather than letting it assert nothing'
    );
  }
  return key;
}

const CLIENT_KEY = mustFind('probeMonitors.ts');
const STORE_KEY = mustFind('probeMonitorsStore.ts');

/**
 * Methods that legitimately have no caller, each with the reason.
 *
 * Deliberately awkward to add to: an entry is a claim that a capability is unreachable ON PURPOSE,
 * which is a decision, not a shrug.
 */
const ALLOWED_WITHOUT_CALLER: Record<string, string> = {
  getRun:
    'The list endpoint returns every field a single-run fetch would, and the panel polls the ' +
    'list while anything is live. A single-run fetch would be a second path to the same data.',
  getDefinition:
    'The built document reaches a user through `exportUrl` (the Download button), which serves ' +
    'the exact bytes that were built and whose digest the manifest pins. The panel reads the ' +
    "definition's metadata from `probe.definition_build`, which the list already carries. " +
    'Fetching and re-rendering the JSON here would be a third view of one document.',
};

function clientMethods(): string[] {
  const body = RAW[CLIENT_KEY].slice(RAW[CLIENT_KEY].indexOf('export const probeMonitorsApi'));
  return [...body.matchAll(/^ {2}([a-zA-Z][a-zA-Z0-9]*):\s*(?:async\s*)?\(/gm)].map((m) => m[1]);
}

/** Every source except the client, the store and the test files. */
function otherSources(...exclude: string[]): string[] {
  return Object.entries(RAW)
    .filter(([key]) => !exclude.includes(key) && !/__tests__|\.test\.tsx?$/.test(key))
    .map(([, text]) => text);
}

describe('the probe API client has no unreachable methods', () => {
  const methods = clientMethods();
  const sources = otherSources(CLIENT_KEY);

  it('finds the client methods at all — this guard must not fail open', () => {
    /*
     * A source scrape that matches nothing passes and asserts nothing. This repo has shipped that
     * three times, so the pattern is checked against known members before it is trusted.
     */
    expect(methods.length).toBeGreaterThan(12);
    for (const known of ['submitRun', 'listProbes', 'getReport', 'publishDefinition']) {
      expect(methods).toContain(known);
    }
  });

  it('the five that were unreachable are each called now', () => {
    for (const method of [
      'submitJudgeRun',
      'cancelJudgeRun',
      'evaluate',
      'score',
      'getScoreResult',
    ]) {
      expect(methods).toContain(method);
      const callers = sources.filter((text) => text.includes(`.${method}(`)).length;
      expect(callers, `${method} has no caller outside the client`).toBeGreaterThan(0);
    }
  });

  it('every method is called somewhere, or listed as deliberately unreachable with a reason', () => {
    const orphans = methods.filter(
      (method) =>
        !(method in ALLOWED_WITHOUT_CALLER) &&
        !sources.some((text) => text.includes(`.${method}(`))
    );
    expect(
      orphans,
      'these client methods have no caller. Wire them to the UI, or add them to ' +
        'ALLOWED_WITHOUT_CALLER with the reason they are unreachable on purpose'
    ).toEqual([]);
  });

  /*
   * ⚠ THE SECOND LINK, AND A MUTATION PROVED IT WAS MISSING.
   *
   * The checks above pin CLIENT ← STORE. A control that unwired the judge form's `onSubmit` in
   * the panel left every one of them green, because the store still called the client — the
   * capability was unreachable again and the guard could not see it.
   *
   * This repo has recorded that exact shape before: a reachability test retargeted at one link
   * "leaves a hole exactly the size of the refactor". So the chain is pinned at both ends —
   * client ← store, and store ← the components that drive it.
   */
  describe('and the store actions reach a component', () => {
    const storeSource = RAW[STORE_KEY];
    const componentSources = Object.entries(RAW)
      .filter(([key]) => key.includes('/components/') && !/__tests__|\.test\.tsx?$/.test(key))
      .map(([, text]) => text);

    /** The actions the interface declares, which is the store's own contract. */
    const actions = [
      ...storeSource
        .slice(0, storeSource.indexOf('export const useProbeMonitorsStore'))
        .matchAll(/^ {2}([a-zA-Z][a-zA-Z0-9]*):\s*\(/gm),
    ].map((m) => m[1]);

    it('finds the store actions at all', () => {
      expect(actions.length).toBeGreaterThan(10);
      for (const known of ['submitRun', 'submitJudgeRun', 'evaluateProbe', 'scoreProbe']) {
        expect(actions).toContain(known);
      }
    });

    it('every action the ladder depends on is CALLED by a component', () => {
      /*
       * ⚠ THE CALL FORM, NOT THE NAME. Matching the bare name passed against two controls that
       * unwired a handler, because the panel still DESTRUCTURES the action from the store even
       * when nothing calls it — `submitJudgeRun,` in the destructuring block satisfied a
       * `includes('submitJudgeRun')`. Same wrong-occurrence trap as matching a comment.
       */
      for (const action of ['submitJudgeRun', 'cancelJudgeRun', 'evaluateProbe', 'scoreProbe']) {
        const called = new RegExp(`\\b${action}\\s*\\(`);
        const drivers = componentSources.filter((text) => called.test(text)).length;
        expect(
          drivers,
          `${action} is in the store and no component CALLS it — the capability is unreachable`
        ).toBeGreaterThan(0);
      }
    });

    /*
     * ⚠ THE LIST ABOVE IS HAND-KEPT, AND A MUTATION CAUGHT WHAT THAT COSTS.
     *
     * This file's own header says the client check "READS THE CLIENT, NOT A HAND-KEPT LIST —
     * a list would have to be updated by the same person who forgot to wire the method, which is
     * exactly who cannot be relied on to update it". The second link was then written AS a list
     * of four names, so every store action added afterwards had no store←component guard at all.
     *
     * Found 2026-10-02 by a control that unwired the recalibration dial's `onRecalibrate`: all
     * six tests in this file stayed green, because the STORE still called the client and
     * `recalibrateProbe` was not one of the four names. The capability was unreachable and the
     * guard written to see that could not.
     *
     * So this derives the set instead, and an exemption is a sentence rather than an omission.
     */
    const ACTIONS_WITHOUT_A_COMPONENT: Record<string, string> = {
      applyRunProgress:
        'Driven by the WebSocket hook, not a component — it merges a live progress event into ' +
        'the run list so a socket update needs no refetch.',
      clearError:
        'Called by the panel through the shared error banner, which takes the setter as a prop ' +
        'rather than naming the action.',
    };

    it('EVERY store action is called by a component, or exempt with a reason', () => {
      const orphans = actions.filter((action) => {
        if (action in ACTIONS_WITHOUT_A_COMPONENT) return false;
        const called = new RegExp(`\\b${action}\\s*\\(`);
        return !componentSources.some((text) => called.test(text));
      });
      expect(
        orphans,
        'these store actions are declared and no component CALLS them, so the capability is ' +
          'unreachable from the UI. Wire them, or add them to ACTIONS_WITHOUT_A_COMPONENT with ' +
          'the reason — and note that DESTRUCTURING an action is not calling it'
      ).toEqual([]);
    });
  });

  it('every allowlist entry still exists on the client, so the list cannot rot', () => {
    for (const name of Object.keys(ALLOWED_WITHOUT_CALLER)) {
      expect(methods, `${name} is allowlisted but is no longer a client method`).toContain(name);
    }
  });
});
