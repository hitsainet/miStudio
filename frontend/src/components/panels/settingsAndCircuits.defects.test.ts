/**
 * MIS-E2E-130 and -131: seven UI defects, most of them silent.
 *
 * Asserted against source structure rather than rendered output. These panels
 * are 1,368 and 1,400+ lines with heavy store and network coupling, and a
 * rendering harness for them is its own piece of work (that gap is
 * MIS-E2E-016, recorded as debt). What each check pins is the specific line
 * whose absence produced the reported behaviour, and every one is confirmed by
 * a mutation that fails it.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';

const settings = readFileSync(join(__dirname, 'SettingsPanel.tsx'), 'utf-8');
const circuits = readFileSync(join(__dirname, 'CircuitsPanel.tsx'), 'utf-8');

describe('MIS-E2E-130 — Settings', () => {
  it('the Saved Endpoints list is filtered by key shape, not category', () => {
    // `openai_compatible_model` and `ollama_url` also carry category
    // 'endpoints'; a model NAME rendered beside real URLs with a delete button.
    expect(settings).toMatch(
      /getByCategory\('endpoints'\)\s*\.filter\(\(s\) => s\.key\.startsWith\('endpoint:'\)\)/
    );
  });

  it('real endpoints are still written with that key prefix', () => {
    // If handleAdd stops using the prefix, the filter above hides everything.
    expect(settings).toContain('const key = `endpoint:${trimmed}`;');
  });

  it('clearing the ollama url awaits and catches the delete', () => {
    const handler = settings.slice(settings.indexOf("remove('ollama_url')") - 400,
                                   settings.indexOf("remove('ollama_url')") + 400);
    expect(handler).toContain('await remove');
    expect(handler).toContain('catch');
    expect(settings).not.toMatch(/onClick=\{\(\) => \{ remove\('ollama_url'\); /);
  });

  it('the labeling fields seed once instead of on every settings change', () => {
    expect(settings).toContain('seededRef');
    expect(settings).toMatch(/if \(seededRef\.current \|\| settings\.length === 0\) return;/);
  });
});

describe('MIS-E2E-131 — Circuits', () => {
  it('mountedRef is set true on mount, not only false on cleanup', () => {
    // StrictMode mounts/unmounts/remounts, so a ref only ever set false was
    // already false by the time the real mount happened.
    expect(circuits).toMatch(/mountedRef\.current = true;/);
    expect(circuits).toMatch(/return \(\) => \{ mountedRef\.current = false; \};/);
  });

  it('the slice export appends the anchor before clicking it', () => {
    // Firefox ignores a click on a detached anchor — a silent no-op.
    const exportBlock = circuits.slice(
      circuits.indexOf('.slices.json'), circuits.indexOf('.slices.json') + 700
    );
    expect(exportBlock).toContain('document.body.appendChild(a)');
    expect(exportBlock).toContain('a.click()');
    expect(exportBlock).toContain('document.body.removeChild(a)');
  });

  it('the object URL is not revoked on the line after the click', () => {
    const exportBlock = circuits.slice(
      circuits.indexOf('.slices.json'), circuits.indexOf('.slices.json') + 700
    );
    expect(exportBlock).toMatch(/setTimeout\(\(\) => URL\.revokeObjectURL\(url\), 0\)/);
  });

  it('seed refs with an unparseable feature_idx are dropped, not sent as null', () => {
    // `6` with no colon yielded feature_idx: NaN, which JSON.stringify emits
    // as null — a malformed seed the user was never told about.
    expect(circuits).toMatch(/return !isNaN\(\(r as \{ feature_idx: number \}\)\.feature_idx\);/);
  });

  it('un-ticking Force clears a captureId that is no longer eligible', () => {
    const block = circuits.slice(
      circuits.indexOf('checked={force}'), circuits.indexOf('checked={force}') + 900
    );
    expect(block).toContain('setCaptureId(\'\')');
    expect(block).toContain('!c.stale');
  });
});
