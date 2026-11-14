/**
 * DR-25: browser storage keys stay frozen (042 FR-042.7; FTASKS 6.7). A renamed key would silently empty
 * every user's saved sidebar state and half-filled forms after the rebrand.
 *
 * The keys are not re-typed here: the test lets the REAL writers write (the persisted zustand store and the
 * `usePersistentState` hook) into jsdom's real `localStorage`, reads back the key each one used, and requires
 * the export's DR-25 `pattern_js` to match it. A value stored under that key by an earlier build is read
 * back by the hook. (The miLLM P0 review found its exercise reading `undefined` from a stubbed storage while
 * it looked green; `src/test/setup.ts` here stubs nothing, and the first assertion proves the store writes.)
 *
 * Control 6.21.17 alters the key in `stores/uiStore.ts` → red.
 */
import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { usePersistentState } from '../hooks/usePersistentState';
import { useUIStore } from '../stores/uiStore';
import { loadExport } from './legacyExport';

function dr25(): RegExp {
  const entry = loadExport().records.find((r) => r.section === 'entry' && r.id === 'DR-25');
  if (!entry) throw new Error('the export has no DR-25 entry');
  return new RegExp(entry.pattern_js as string, entry.js_flags as string);
}

function keys(): string[] {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) as string);
}

describe('frozen browser storage keys', () => {
  it('DR-25 storage keys are unchanged', () => {
    const frozen = dr25();
    window.localStorage.clear();

    // The persisted UI store writes its preferences under the frozen key.
    act(() => useUIStore.getState().toggleSidebar());
    const storeKeys = keys();
    expect(storeKeys).toHaveLength(1);
    const [storeKey] = storeKeys;
    expect(frozen.test(JSON.stringify(storeKey)), `the store wrote ${storeKey}`).toBe(true);
    expect(JSON.parse(window.localStorage.getItem(storeKey) as string).state.sidebar).toEqual(
      useUIStore.getState().sidebar,
    );

    // A draft field writes under the frozen prefix…
    const { result, unmount } = renderHook(() => usePersistentState('dr25-field', 'first'));
    act(() => result.current[1]('typed'));
    const draftKeys = keys().filter((k) => k !== storeKey);
    expect(draftKeys).toHaveLength(1);
    const [draftKey] = draftKeys;
    expect(draftKey.endsWith('dr25-field')).toBe(true);
    expect(frozen.test(JSON.stringify(draftKey)), `the hook wrote ${draftKey}`).toBe(true);
    unmount();

    // …and reads back a value an earlier build stored under the same key.
    window.localStorage.setItem(draftKey, JSON.stringify('saved before the rebrand'));
    const again = renderHook(() => usePersistentState('dr25-field', 'first'));
    expect(again.result.current[0]).toBe('saved before the rebrand');
    again.unmount();
  });
});
