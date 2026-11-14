/**
 * Circuits list store (016 task 0.3 — suite tracker 4.22, 2026-10-16).
 *
 * Extracted from `CircuitsPanel`'s list tab, where the fetch and its state lived
 * in component `useState`. Two defects came out with it:
 *
 *  - **The list showed the first 50 circuits and never said there were more.**
 *    The tab called `circuitsApi.list()` with no parameters, so the server's
 *    default page (limit 50) was the whole list as far as anyone could tell —
 *    `total` arrived in every response and was discarded. The server has filtered
 *    and paged in SQL since 865f6355; nothing in the UI could ask it to.
 *  - **No filter controls.** `edge_type`, `min_rung` and `promoted` were
 *    reachable only from the REST API and MCP.
 *
 * A late response can no longer overwrite a newer one: each fetch takes a
 * sequence number and only the latest writes, so flipping a filter twice quickly
 * cannot leave the list showing the first filter's rows under the second's label.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { circuitsApi } from '../api/circuits';
import type { CircuitSummary, EdgeType } from '../types/circuits';

export interface CircuitListFilters {
  promoted?: boolean;
  min_rung?: number;
  edge_type?: EdgeType;
}

/** The server's own maximum (`limit: int = Query(50, ge=1, le=200)`). */
export const CIRCUITS_PAGE_SIZE = 50;

interface CircuitsState {
  rows: CircuitSummary[];
  total: number;
  limit: number;
  offset: number;
  filters: CircuitListFilters;
  loading: boolean;
  error: string | null;

  fetchCircuits: () => Promise<void>;
  setFilters: (patch: Partial<CircuitListFilters>) => Promise<void>;
  setPage: (offset: number) => Promise<void>;
  setPromoted: (id: string, promoted: boolean) => Promise<void>;
  removeCircuit: (id: string) => Promise<void>;
  importDefinition: (definition: unknown) => Promise<void>;
  setError: (error: string | null) => void;
}

let fetchSeq = 0;

function message(e: unknown): string {
  return String((e as Error)?.message ?? e);
}

/** Drop filter keys left `undefined` so they are not sent as empty strings. */
function definedFilters(filters: CircuitListFilters): CircuitListFilters {
  return Object.fromEntries(
    Object.entries(filters).filter(([, v]) => v !== undefined && v !== null && v !== ''),
  ) as CircuitListFilters;
}

export const useCircuitsStore = create<CircuitsState>()(
  devtools(
    (set, get) => ({
      rows: [],
      total: 0,
      limit: CIRCUITS_PAGE_SIZE,
      offset: 0,
      filters: {},
      loading: true,
      error: null,

      fetchCircuits: async () => {
        const seq = ++fetchSeq;
        const { filters, limit, offset } = get();
        set({ loading: true });
        try {
          const r = await circuitsApi.list({ ...definedFilters(filters), limit, offset });
          if (seq !== fetchSeq) return; // a newer fetch owns the list now
          set({ rows: r.circuits, total: r.total, error: null });
        } catch (e) {
          if (seq !== fetchSeq) return;
          set({ error: message(e) });
        } finally {
          if (seq === fetchSeq) set({ loading: false });
        }
      },

      setFilters: async (patch) => {
        // A new filter starts from the first page: offset 150 of a filter that
        // matches 12 circuits is an empty page that looks like "no results".
        set({ filters: { ...get().filters, ...patch }, offset: 0 });
        await get().fetchCircuits();
      },

      setPage: async (offset) => {
        set({ offset: Math.max(0, offset) });
        await get().fetchCircuits();
      },

      setPromoted: async (id, promoted) => {
        try {
          const updated = await circuitsApi.setPromoted(id, promoted);
          set({
            rows: get().rows.map((r) => (r.id === id ? { ...r, promoted: updated.promoted } : r)),
          });
          // Under a `promoted` filter the row no longer belongs on this page.
          if (get().filters.promoted !== undefined) await get().fetchCircuits();
        } catch (e) {
          set({ error: message(e) });
        }
      },

      removeCircuit: async (id) => {
        try {
          await circuitsApi.remove(id);
          // Removing the last row of a page must not strand the view on an empty page.
          const { offset, limit, rows } = get();
          if (rows.length === 1 && offset > 0) set({ offset: Math.max(0, offset - limit) });
          await get().fetchCircuits();
        } catch (e) {
          set({ error: message(e) });
        }
      },

      importDefinition: async (definition) => {
        try {
          await circuitsApi.importDefinition(definition);
          await get().fetchCircuits();
        } catch (e) {
          set({ error: `Import failed: ${message(e)}` });
        }
      },

      setError: (error) => set({ error }),
    }),
    { name: 'circuits' },
  ),
);
