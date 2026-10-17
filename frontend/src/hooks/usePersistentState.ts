/**
 * `useState`, but the value survives a page refresh.
 *
 * ⚠ WHY. Refreshing the Probes panel dropped you back on the Runs sub-tab and emptied every form
 * on every sub-tab — a half-filled run configuration, a dataset view's column mapping, a judge
 * endpoint. Nothing about that state is expensive to keep and all of it is annoying to retype,
 * and a reload is not a request to discard work.
 *
 * ⚠ EVERY ACCESS IS GUARDED, because `localStorage` is not always there. It throws in a private
 * window with site data blocked, it can be full, and it does not exist during SSR or in some test
 * environments. A component must render correctly when it is unavailable, so a failure here falls
 * back to ordinary in-memory state rather than breaking the panel.
 *
 * ⚠ WHAT NOT TO PUT IN IT. This is per-viewer convenience, not state anything depends on. Do not
 * persist an id that can stop existing — `jlensStore` records the case: a saved GPU UUID can name
 * a card that has been removed, and restoring it aims work at hardware that is gone. Prefer
 * persisting what the user TYPED over what the app RESOLVED.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

const PREFIX = 'mistudio.draft.';

function read<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    // A blocked, full or absent store is not an error worth surfacing — it means "no draft".
    return fallback;
  }
}

export function usePersistentState<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => read(key, initial));

  // The key can change between renders (a form keyed by probe id); re-read rather than keep the
  // previous key's value, which would show one probe's draft under another's name.
  const previousKey = useRef(key);
  useEffect(() => {
    if (previousKey.current !== key) {
      previousKey.current = key;
      setValue(read(key, initial));
    }
    // `initial` is deliberately not a dependency: a caller passing an inline object or array
    // literal would otherwise reset the field on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    try {
      window.localStorage.setItem(PREFIX + key, JSON.stringify(value));
    } catch {
      // Out of quota or blocked: the field keeps working, it simply will not survive a reload.
    }
  }, [key, value]);

  const clear = useCallback(() => {
    try {
      window.localStorage.removeItem(PREFIX + key);
    } catch {
      /* nothing to clear */
    }
    setValue(initial);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return [value, setValue, clear] as const;
}

/**
 * Drop a restored selection that no longer exists.
 *
 * ⚠ THIS IS THE OTHER HALF OF PERSISTING A FORM, and the first version of that change shipped
 * without it — including the id fields, which the docstring above explicitly warns against. A
 * persisted `modelId` naming a model that has since been deleted leaves a select bound to a value
 * that is not among its options: it renders blank, reads as "nothing chosen", and submits a dead
 * id if the user does not notice.
 *
 * Runs after the options arrive, because on first render they are usually still loading and
 * everything would look absent.
 */
export function useValidSelection<T extends string>(
  value: T | '',
  setValue: (next: T | '') => void,
  available: readonly string[],
  ready: boolean
) {
  useEffect(() => {
    if (!ready || value === '') return;
    if (!available.includes(value)) setValue('');
  }, [ready, value, available, setValue]);
}

/** The same, for a multi-select: keeps the ids that still exist. */
export function useValidSelections(
  values: string[],
  setValues: (next: string[]) => void,
  available: readonly string[],
  ready: boolean
) {
  useEffect(() => {
    if (!ready || values.length === 0) return;
    const kept = values.filter((id) => available.includes(id));
    if (kept.length !== values.length) setValues(kept);
  }, [ready, values, available, setValues]);
}
