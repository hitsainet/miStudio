/**
 * Form drafts survive a refresh — and a restored id that no longer exists does not.
 *
 * ⚠ WHY THE SECOND HALF MATTERS AS MUCH AS THE FIRST. The first version of this persisted every
 * field including `modelId`, `trainDatasetId` and `evalDatasetIds` — exactly what the hook's own
 * docstring says not to persist, because an id can stop existing. A select bound to a deleted
 * model renders blank, reads as "nothing chosen", and submits a dead id if nobody notices.
 */
import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  usePersistentState,
  useValidSelection,
  useValidSelections,
} from './usePersistentState';

beforeEach(() => window.localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('a draft survives a reload', () => {
  it('starts at the initial value when nothing was saved', () => {
    const { result } = renderHook(() => usePersistentState('k', 'initial'));
    expect(result.current[0]).toBe('initial');
  });

  it('restores what was there before — this is the whole point', () => {
    const first = renderHook(() => usePersistentState('k', ''));
    act(() => first.result.current[1]('half a configuration'));
    first.unmount();

    // A reload is a fresh mount against the same storage.
    const second = renderHook(() => usePersistentState('k', ''));
    expect(second.result.current[0]).toBe('half a configuration');
  });

  it('keeps arrays and objects, not only strings', () => {
    const first = renderHook(() => usePersistentState<string[]>('sets', []));
    act(() => first.result.current[1](['a', 'b']));
    first.unmount();
    expect(renderHook(() => usePersistentState<string[]>('sets', [])).result.current[0]).toEqual([
      'a',
      'b',
    ]);
  });

  it('clear() forgets the draft and returns to the initial value', () => {
    const { result } = renderHook(() => usePersistentState('k', 'initial'));
    act(() => result.current[1]('typed'));
    act(() => result.current[2]());
    expect(result.current[0]).toBe('initial');
  });

  it('re-reads when the KEY changes, rather than showing the previous key\'s draft', () => {
    /** A form keyed by probe id would otherwise show one probe's draft under another's name. */
    window.localStorage.setItem('mistudio.draft.a', JSON.stringify('A'));
    window.localStorage.setItem('mistudio.draft.b', JSON.stringify('B'));
    const { result, rerender } = renderHook(({ k }) => usePersistentState(k, ''), {
      initialProps: { k: 'a' },
    });
    expect(result.current[0]).toBe('A');
    rerender({ k: 'b' });
    expect(result.current[0]).toBe('B');
  });
});

describe('storage being unavailable is not an error', () => {
  it('falls back to in-memory state when reading throws', () => {
    /*
     * A private window with site data blocked throws on access. The field must still work — it
     * simply will not survive a reload.
     */
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const { result } = renderHook(() => usePersistentState('k', 'initial'));
    expect(result.current[0]).toBe('initial');
    act(() => result.current[1]('still typeable'));
    expect(result.current[0]).toBe('still typeable');
  });

  it('a full or blocked store does not break setting a value', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    const { result } = renderHook(() => usePersistentState('k', ''));
    act(() => result.current[1]('typed'));
    expect(result.current[0]).toBe('typed');
  });

  it('corrupt JSON reads as "no draft" rather than throwing', () => {
    window.localStorage.setItem('mistudio.draft.k', '{not json');
    expect(renderHook(() => usePersistentState('k', 'initial')).result.current[0]).toBe('initial');
  });
});

describe('a restored id that no longer exists is dropped', () => {
  it('clears a single selection once the options have arrived without it', () => {
    const setValue = vi.fn();
    renderHook(() => useValidSelection('m_deleted', setValue, ['m_a', 'm_b'], true));
    expect(setValue).toHaveBeenCalledWith('');
  });

  it('keeps a selection that still exists', () => {
    const setValue = vi.fn();
    renderHook(() => useValidSelection('m_a', setValue, ['m_a', 'm_b'], true));
    expect(setValue).not.toHaveBeenCalled();
  });

  it('WAITS for the options — before they load, everything looks absent', () => {
    /** Clearing on an empty list would wipe a valid draft every time the panel opened. */
    const setValue = vi.fn();
    renderHook(() => useValidSelection('m_a', setValue, [], false));
    expect(setValue).not.toHaveBeenCalled();
  });

  it('keeps the surviving ids of a multi-select and drops the rest', () => {
    const setValues = vi.fn();
    renderHook(() => useValidSelections(['a', 'gone', 'b'], setValues, ['a', 'b'], true));
    expect(setValues).toHaveBeenCalledWith(['a', 'b']);
  });

  it('leaves a fully-valid multi-select alone, so it does not loop', () => {
    const setValues = vi.fn();
    renderHook(() => useValidSelections(['a', 'b'], setValues, ['a', 'b'], true));
    expect(setValues).not.toHaveBeenCalled();
  });
});
