/**
 * circuitsStore + the Circuits list tab (016 task 0.3, suite tracker 4.22).
 *
 * The tab used to fetch an unparameterised first page and drop `total`, so a 51st
 * circuit was invisible and nothing said so. These tests drive the REAL tab through
 * the REAL store with only the HTTP client mocked, and assert the parameters each
 * call SENT — a test that checked only "list was called" would pass against the
 * old no-argument call.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

vi.mock('../api/circuits', () => ({
  circuitsApi: {
    list: vi.fn(),
    get: vi.fn(),
    setPromoted: vi.fn(),
    remove: vi.fn(),
    importDefinition: vi.fn(),
    exportUrl: (id: string) => `/api/v1/circuits/${id}/export`,
    exportSlices: vi.fn(),
  },
}));

import { circuitsApi } from '../api/circuits';
import { useCircuitsStore, CIRCUITS_PAGE_SIZE } from './circuitsStore';
import { CircuitsListTab } from '../components/panels/CircuitsPanel';
import type { CircuitSummary } from '../types/circuits';

const list = circuitsApi.list as unknown as ReturnType<typeof vi.fn>;

function row(i: number, extra: Partial<CircuitSummary> = {}): CircuitSummary {
  return {
    id: `circ_${i}`,
    name: `Circuit ${i}`,
    rung: 0,
    rung_language: 'mined association',
    rung_next_step: null,
    promoted: false,
    member_count: 2,
    edge_count: 1,
    layers: [11, 12],
    ...extra,
  } as unknown as CircuitSummary;
}

function page(offset: number, n: number, total: number) {
  return { circuits: Array.from({ length: n }, (_, k) => row(offset + k)), total };
}

beforeEach(() => {
  vi.clearAllMocks();
  useCircuitsStore.setState({
    rows: [], total: 0, limit: CIRCUITS_PAGE_SIZE, offset: 0, filters: {}, loading: true, error: null,
  });
});

describe('circuitsStore', () => {
  it('sends limit and offset and keeps the total', async () => {
    list.mockResolvedValueOnce(page(0, 50, 73));
    await useCircuitsStore.getState().fetchCircuits();
    expect(list).toHaveBeenCalledTimes(1);
    expect(list).toHaveBeenCalledWith({ limit: CIRCUITS_PAGE_SIZE, offset: 0 });
    expect(useCircuitsStore.getState().total).toBe(73);
    expect(useCircuitsStore.getState().rows).toHaveLength(50);
  });

  it('a filter change resets to the first page and sends only set filters', async () => {
    useCircuitsStore.setState({ offset: 100 });
    list.mockResolvedValueOnce(page(0, 3, 3));
    await useCircuitsStore.getState().setFilters({ edge_type: 'persistence', min_rung: undefined });
    expect(list).toHaveBeenCalledWith({ edge_type: 'persistence', limit: CIRCUITS_PAGE_SIZE, offset: 0 });
    expect(useCircuitsStore.getState().offset).toBe(0);
  });

  it('a late response never overwrites a newer one', async () => {
    let resolveFirst: (v: unknown) => void = () => {};
    list.mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }));
    list.mockResolvedValueOnce({ circuits: [row(2)], total: 1 });
    const first = useCircuitsStore.getState().fetchCircuits();
    await useCircuitsStore.getState().setFilters({ edge_type: 'computed' });
    resolveFirst({ circuits: [row(1), row(9)], total: 2 });
    await first;
    expect(useCircuitsStore.getState().rows.map((r) => r.id)).toEqual(['circ_2']);
    expect(useCircuitsStore.getState().loading).toBe(false);
  });

  it('removing the last row of a later page steps back a page', async () => {
    useCircuitsStore.setState({ rows: [row(50)], total: 51, offset: 50 });
    (circuitsApi.remove as unknown as ReturnType<typeof vi.fn>).mockResolvedValueOnce({});
    list.mockResolvedValueOnce(page(0, 50, 50));
    await useCircuitsStore.getState().removeCircuit('circ_50');
    expect(list).toHaveBeenCalledWith({ limit: CIRCUITS_PAGE_SIZE, offset: 0 });
  });
});

describe('CircuitsListTab is driven by the store', () => {
  it('says how many circuits exist and pages to the next 50', async () => {
    list.mockResolvedValueOnce(page(0, 50, 73));
    render(<CircuitsListTab />);
    expect(await screen.findByText('1–50 of 73')).toBeInTheDocument();

    list.mockResolvedValueOnce(page(50, 23, 73));
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Next' })); });
    await waitFor(() => expect(screen.getByText('51–73 of 73')).toBeInTheDocument());
    expect(list).toHaveBeenLastCalledWith({ limit: CIRCUITS_PAGE_SIZE, offset: 50 });
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
  });

  it('the edge-type control filters on the server', async () => {
    list.mockResolvedValue(page(0, 2, 2));
    render(<CircuitsListTab />);
    await screen.findByText('1–2 of 2');
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Filter by edge type'), { target: { value: 'attention_mediated' } });
    });
    expect(list).toHaveBeenLastCalledWith({
      edge_type: 'attention_mediated', limit: CIRCUITS_PAGE_SIZE, offset: 0,
    });
  });

  it('an empty filtered page says the filters matched nothing, not "no circuits yet"', async () => {
    list.mockResolvedValueOnce(page(0, 2, 2));
    render(<CircuitsListTab />);
    await screen.findByText('1–2 of 2');
    list.mockResolvedValueOnce({ circuits: [], total: 0 });
    await act(async () => {
      fireEvent.change(screen.getByLabelText('Filter by minimum evidence rung'), { target: { value: '3' } });
    });
    expect(list).toHaveBeenLastCalledWith({ min_rung: 3, limit: CIRCUITS_PAGE_SIZE, offset: 0 });
    expect(await screen.findByText(/No circuits match these filters/)).toBeInTheDocument();
    expect(screen.queryByText('No circuits yet')).not.toBeInTheDocument();
  });
});
