/**
 * The Datasets tab: archive, delete permanently, and the row browser (operator decisions
 * 2026-10-21). Every action is asserted by the CALL it makes — method, arguments and count —
 * against the mocked API module, the same seam production uses.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import type {
  ProbeDataset,
  ProbeDatasetPurgePlan,
  ProbeDatasetRowsPage,
} from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  archiveDataset: vi.fn(),
  unarchiveDataset: vi.fn(),
  planDatasetPurge: vi.fn(),
  purgeDatasets: vi.fn(),
  datasetRows: vi.fn(),
  listRuns: vi.fn(),
}));

vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));
vi.mock('../ProbeDatasetForm', () => ({ ProbeDatasetForm: () => <div data-testid="form-stub" /> }));

import { DatasetsTab } from '../DatasetsTab';
import { activeViews, blockedReason } from '../datasetArchive';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

function view(id: string, name: string, extra: Partial<ProbeDataset> = {}): ProbeDataset {
  return {
    id,
    name,
    dataset_id: 'd-1',
    config: null,
    split: 'train',
    input_column: 'prompt',
    label_column: 'label',
    label_mapping: { yes: 'positive', no: 'negative' },
    keyword_filter: null,
    pair_column: null,
    role: 'eval',
    distribution: null,
    counts: { positive: 25, negative: 25, excluded: 3, filtered_out: 4, unparseable: 4 },
    created_at: '2026-10-20T12:00:00Z',
    archived_at: null,
    archived_by: null,
    archive_reason: null,
    ...extra,
  };
}

const ARCHIVED = { archived_at: '2026-10-21T14:05:00Z', archived_by: 'operator' };
const VIEWS = [
  view('pmd_live', 'PII test rows'),
  view('pmd_free', 'old e2e leftover', ARCHIVED),
  view('pmd_used', 'humor eval', { ...ARCHIVED, archived_by: 'agent:mcp', archive_reason: 'superseded' }),
  view('pmd_free2', 'another leftover', ARCHIVED),
];

function plan(id: string, extra: Partial<ProbeDatasetPurgePlan> = {}): ProbeDatasetPurgePlan {
  const v = VIEWS.find((x) => x.id === id);
  return {
    id,
    found: true,
    name: v?.name ?? id,
    archived: Boolean(v?.archived_at),
    purgeable: Boolean(v?.archived_at),
    not_archived: v?.archived_at ? null : { code: 'not_archived', message: 'not archived' },
    blockers: [],
    removes: { rows: { probe_monitor_datasets: 1 } },
    kept: [{ what: 'downloaded dataset', name: 'humor-download', dataset_id: 'd-1', other_views: 3, why: '' }],
    ...extra,
  };
}

const USED_BLOCKER = {
  code: 'run_uses_it',
  message: 'Probe run pmr_1 (completed) uses this view as its evaluation set.',
};

function plansFor(ids: string[]) {
  return ids.map((id) =>
    id === 'pmd_used' ? plan(id, { purgeable: false, blockers: [USED_BLOCKER] }) : plan(id)
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  api.planDatasetPurge.mockImplementation(async (ids: string[]) => plansFor(ids));
  api.listDatasets.mockResolvedValue(VIEWS);
  act(() => {
    useProbeMonitorsStore.setState({ datasets: VIEWS, error: null });
  });
});

async function showArchived(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId('datasets-show-archived'));
  await waitFor(() => expect(screen.getByTestId('datasets-purge-header')).toBeInTheDocument());
  await waitFor(() => expect(screen.getByTestId('dataset-blocked-pmd_used')).toBeInTheDocument());
}

describe('archive', () => {
  it('hides archived datasets until "Show archived" is ticked, then says who and when', async () => {
    const user = userEvent.setup();
    render(<DatasetsTab />);
    expect(screen.getByTestId('dataset-row-pmd_live')).toBeInTheDocument();
    expect(screen.queryByTestId('dataset-row-pmd_free')).not.toBeInTheDocument();
    expect(screen.getByText('Show archived (3)')).toBeInTheDocument();
    await showArchived(user);
    expect(screen.getByTestId('dataset-archived-note-pmd_used').textContent).toBe(
      'Archived Oct 21, 10:05 AM ET by an agent (mcp) · superseded'
    );
  });

  it('Archive sends exactly that id, once, and the row leaves the list', async () => {
    const user = userEvent.setup();
    api.archiveDataset.mockResolvedValue({ ...VIEWS[0], ...ARCHIVED });
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-archive-pmd_live'));
    expect(api.archiveDataset).toHaveBeenCalledTimes(1);
    expect(api.archiveDataset).toHaveBeenCalledWith('pmd_live', undefined);
    await waitFor(() => expect(screen.queryByTestId('dataset-row-pmd_live')).not.toBeInTheDocument());
  });

  it('Unarchive sends exactly that id, once', async () => {
    const user = userEvent.setup();
    api.unarchiveDataset.mockResolvedValue({ ...VIEWS[1], archived_at: null, archived_by: null });
    render(<DatasetsTab />);
    await showArchived(user);
    await user.click(screen.getByTestId('dataset-unarchive-pmd_free'));
    expect(api.unarchiveDataset).toHaveBeenCalledTimes(1);
    expect(api.unarchiveDataset).toHaveBeenCalledWith('pmd_free');
  });

  it('new-run pickers never see an archived dataset', () => {
    expect(activeViews(VIEWS).map((v) => v.id)).toEqual(['pmd_live']);
  });
});

describe('selecting archived datasets for permanent deletion', () => {
  it('asks the server which can be deleted, for exactly the archived rows on screen', async () => {
    const user = userEvent.setup();
    render(<DatasetsTab />);
    await showArchived(user);
    expect(api.planDatasetPurge).toHaveBeenCalledWith(['pmd_free', 'pmd_used', 'pmd_free2']);
    expect(screen.queryByTestId('dataset-select-pmd_live')).not.toBeInTheDocument();
  });

  it('a blocked dataset shows why, inline, and cannot be ticked', async () => {
    const user = userEvent.setup();
    render(<DatasetsTab />);
    await showArchived(user);
    expect(screen.getByTestId('dataset-blocked-pmd_used').textContent).toContain(USED_BLOCKER.message);
    expect(screen.getByTestId('dataset-select-pmd_used')).toBeDisabled();
    expect(screen.getByText('2 of 3 archived datasets can be deleted')).toBeInTheDocument();
  });

  it('Select all takes only the deletable ones; partial is indeterminate; Deselect all clears', async () => {
    const user = userEvent.setup();
    render(<DatasetsTab />);
    await showArchived(user);
    const header = screen.getByTestId('datasets-select-all') as HTMLInputElement;
    expect(screen.getByTestId('datasets-selected-count').textContent).toBe('0 selected');
    expect(screen.getByTestId('datasets-delete-selected')).toBeDisabled();

    await user.click(screen.getByTestId('dataset-select-pmd_free'));
    expect(screen.getByTestId('datasets-selected-count').textContent).toBe('1 selected');
    expect(header.indeterminate).toBe(true);
    expect(header.getAttribute('aria-checked')).toBe('mixed');
    expect(screen.getByTestId('datasets-select-all-label').textContent).toBe('Deselect all');

    await user.click(header); // partial → deselect all
    expect(screen.getByTestId('datasets-selected-count').textContent).toBe('0 selected');
    await user.click(header); // none → select all deletable
    expect(screen.getByTestId('datasets-selected-count').textContent).toBe('2 selected');
    expect(header.checked).toBe(true);
    expect(header.indeterminate).toBe(false);
    expect((screen.getByTestId('dataset-select-pmd_used') as HTMLInputElement).checked).toBe(false);
  });

  it('Delete permanently confirms first and then sends EXACTLY the selection, with the reason', async () => {
    const user = userEvent.setup();
    api.purgeDatasets.mockResolvedValue({
      results: [
        { outcome: 'purged', id: 'pmd_free', name: 'old e2e leftover' },
        { outcome: 'purged', id: 'pmd_free2', name: 'another leftover' },
      ],
      purged: 2,
      blocked: 0,
    });
    render(<DatasetsTab />);
    await showArchived(user);
    await user.click(screen.getByTestId('datasets-select-all'));
    await user.click(screen.getByTestId('datasets-delete-selected'));
    const dialog = await screen.findByTestId('dataset-purge-dialog');
    await within(dialog).findByTestId('dataset-purge-going-pmd_free');
    expect(within(dialog).getAllByText(/Keeps the download "humor-download"/)).toHaveLength(2);
    expect(api.purgeDatasets).not.toHaveBeenCalled();
    await user.type(within(dialog).getByTestId('dataset-purge-reason'), 'audit 2026-10-21');
    await user.click(within(dialog).getByTestId('dataset-purge-confirm'));
    expect(api.purgeDatasets).toHaveBeenCalledTimes(1);
    expect(api.purgeDatasets).toHaveBeenCalledWith(['pmd_free', 'pmd_free2'], 'audit 2026-10-21');
    expect(api.archiveDataset).not.toHaveBeenCalled();
    expect(await within(dialog).findByTestId('dataset-purge-outcome')).toBeInTheDocument();
  });

  it('a dataset that became blocked since it was ticked is skipped, not sent', async () => {
    const user = userEvent.setup();
    render(<DatasetsTab />);
    await showArchived(user);
    await user.click(screen.getByTestId('dataset-select-pmd_free'));
    api.planDatasetPurge.mockImplementationOnce(async (ids: string[]) =>
      ids.map((id) => plan(id, { purgeable: false, blockers: [USED_BLOCKER] }))
    );
    await user.click(screen.getByTestId('datasets-delete-selected'));
    const dialog = await screen.findByTestId('dataset-purge-dialog');
    await within(dialog).findByTestId('dataset-purge-skipped-pmd_free');
    expect(within(dialog).getByTestId('dataset-purge-confirm')).toBeDisabled();
  });
});

describe('the confirmation sends exactly what it lists as going', () => {
  it('a selection with one view blocked since it was ticked sends only the other', async () => {
    const user = userEvent.setup();
    api.purgeDatasets.mockResolvedValue({
      results: [{ outcome: 'purged', id: 'pmd_free2', name: 'another leftover' }],
      purged: 1,
      blocked: 0,
    });
    render(<DatasetsTab />);
    await showArchived(user);
    await user.click(screen.getByTestId('datasets-select-all'));
    api.planDatasetPurge.mockImplementationOnce(async (ids: string[]) =>
      ids.map((id) => (id === 'pmd_free' ? plan(id, { purgeable: false, blockers: [USED_BLOCKER] }) : plan(id)))
    );
    await user.click(screen.getByTestId('datasets-delete-selected'));
    const dialog = await screen.findByTestId('dataset-purge-dialog');
    await within(dialog).findByTestId('dataset-purge-skipped-pmd_free');
    expect(within(dialog).getByTestId('dataset-purge-confirm').textContent).toBe('Delete 1 dataset permanently');
    await user.click(within(dialog).getByTestId('dataset-purge-confirm'));
    expect(api.purgeDatasets).toHaveBeenCalledTimes(1);
    expect(api.purgeDatasets).toHaveBeenCalledWith(['pmd_free2'], undefined);
  });
});

describe('Delete… straight from a visible row (archive, then delete)', () => {
  it('archives the dataset and then deletes it, after the same confirmation', async () => {
    const user = userEvent.setup();
    api.archiveDataset.mockResolvedValue({ ...VIEWS[0], ...ARCHIVED });
    api.purgeDatasets.mockResolvedValue({
      results: [{ outcome: 'purged', id: 'pmd_live', name: 'PII test rows' }],
      purged: 1,
      blocked: 0,
    });
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-delete-pmd_live'));
    const dialog = await screen.findByTestId('dataset-purge-dialog');
    await within(dialog).findByTestId('dataset-purge-going-pmd_live');
    expect(within(dialog).getByTestId('dataset-purge-confirm').textContent).toBe(
      'Archive and delete 1 dataset permanently'
    );
    await user.click(within(dialog).getByTestId('dataset-purge-confirm'));
    expect(api.archiveDataset).toHaveBeenCalledTimes(1);
    expect(api.archiveDataset).toHaveBeenCalledWith('pmd_live', undefined);
    expect(api.purgeDatasets).toHaveBeenCalledTimes(1);
    expect(api.purgeDatasets).toHaveBeenCalledWith(['pmd_live'], undefined);
    expect(api.archiveDataset.mock.invocationCallOrder[0]).toBeLessThan(
      api.purgeDatasets.mock.invocationCallOrder[0]
    );
  });

  it('a dataset something still uses is not archived or deleted; the dialog says why', async () => {
    const user = userEvent.setup();
    api.planDatasetPurge.mockImplementation(async (ids: string[]) =>
      ids.map((id) => plan(id, { purgeable: false, blockers: [USED_BLOCKER] }))
    );
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-delete-pmd_live'));
    const dialog = await screen.findByTestId('dataset-purge-dialog');
    expect((await within(dialog).findByTestId('dataset-purge-skipped-pmd_live')).textContent).toContain(
      USED_BLOCKER.message
    );
    expect(within(dialog).getByTestId('dataset-purge-confirm')).toBeDisabled();
    expect(api.archiveDataset).not.toHaveBeenCalled();
  });

  it('"not archived" alone blocks the plain purge but not the archive-first path', () => {
    const p = plan('pmd_live');
    expect(blockedReason(p)).toBe('not archived');
    expect(blockedReason(p, true)).toBeNull();
  });
});

function rowsPage(extra: Partial<ProbeDatasetRowsPage> = {}): ProbeDatasetRowsPage {
  return {
    view_id: 'pmd_live',
    name: 'PII test rows',
    role: 'eval',
    dataset_name: 'pii-segments',
    split: 'train',
    input_column: 'prompt',
    label_column: 'label',
    pair_column: 'pair',
    other_columns: ['source', 'notes'],
    returned_columns: ['prompt', 'label', 'pair', 'source', 'notes'],
    total_rows: 60,
    matched: 60,
    offset: 0,
    limit: 50,
    label: null,
    search: null,
    counts: { positive: 25, negative: 25, excluded: 3, filtered_out: 4, unparseable: 4 },
    recount: null,
    counts_agree: null,
    rows: [
      {
        position: 0,
        outcome: 'positive',
        mapped_label: 'positive',
        raw_label: 'yes',
        input_kind: 'plain',
        values: { prompt: 'my card number 0', label: 'yes', pair: 'p0', source: 'src-0', notes: 'n' },
        truncated: [],
      },
      {
        position: 1,
        outcome: 'unparseable',
        mapped_label: null,
        raw_label: 'maybe',
        input_kind: null,
        values: { prompt: 'card maybe', label: 'maybe', pair: 'p0', source: 'src-1', notes: 'x'.repeat(400) },
        truncated: [],
      },
    ],
    ...extra,
  };
}

describe('browsing a dataset’s rows', () => {
  it('opens from the name, asks for the first page, and labels each row in plain words', async () => {
    const user = userEvent.setup();
    api.datasetRows.mockResolvedValue(rowsPage());
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-browse-pmd_live'));
    await screen.findByTestId('dataset-rows-table');
    expect(api.datasetRows).toHaveBeenCalledTimes(1);
    expect(api.datasetRows).toHaveBeenCalledWith('pmd_live', {
      offset: 0, limit: 50, label: null, search: null, columns: null,
    });
    expect(screen.getByTestId('dataset-row-outcome-0').textContent).toBe('Positive');
    expect(screen.getByTestId('dataset-row-outcome-1').textContent).toBe('Unreadable');
    expect(screen.getByText('raw: maybe')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-rows-range').textContent).toContain('Rows 1–2 of 60');
    expect(screen.getByTestId('dataset-rows-count-positive').textContent).toBe('Positive 25');
  });

  it('every control becomes the request: label, page size, columns, paging, search', async () => {
    const user = userEvent.setup();
    api.datasetRows.mockResolvedValue(rowsPage({ matched: 120, total_rows: 120 }));
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-browse-pmd_live'));
    await screen.findByTestId('dataset-rows-table');

    await user.click(screen.getByTestId('dataset-rows-count-excluded'));
    await waitFor(() =>
      expect(api.datasetRows).toHaveBeenLastCalledWith('pmd_live', expect.objectContaining({ label: 'excluded', offset: 0 }))
    );
    await user.selectOptions(screen.getByTestId('dataset-rows-page-size'), '100');
    await waitFor(() =>
      expect(api.datasetRows).toHaveBeenLastCalledWith('pmd_live', expect.objectContaining({ limit: 100 }))
    );
    await user.click(screen.getByTestId('dataset-rows-column-notes'));
    await waitFor(() =>
      expect(api.datasetRows).toHaveBeenLastCalledWith('pmd_live', expect.objectContaining({ columns: ['source'] }))
    );
    await user.click(screen.getByTestId('dataset-rows-next'));
    await waitFor(() =>
      expect(api.datasetRows).toHaveBeenLastCalledWith('pmd_live', expect.objectContaining({ offset: 100 }))
    );
    await user.type(screen.getByTestId('dataset-rows-search'), 'card{Enter}');
    await waitFor(() =>
      expect(api.datasetRows).toHaveBeenLastCalledWith(
        'pmd_live',
        expect.objectContaining({ search: 'card', offset: 0, label: 'excluded', limit: 100, columns: ['source'] })
      )
    );
  });

  it('long text is cut with a way to expand it', async () => {
    const user = userEvent.setup();
    api.datasetRows.mockResolvedValue(rowsPage());
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-browse-pmd_live'));
    const row = await screen.findByTestId('dataset-row-1');
    expect(row.textContent).not.toContain('x'.repeat(400));
    await user.click(within(row).getByText('more'));
    expect(row.textContent).toContain('x'.repeat(400));
  });

  it('says so when the download no longer matches the recorded counts', async () => {
    const user = userEvent.setup();
    api.datasetRows.mockResolvedValue(
      rowsPage({ counts_agree: false, recount: { positive: 3, negative: 0, excluded: 0, filtered_out: 0, unparseable: 0 } })
    );
    render(<DatasetsTab />);
    await user.click(screen.getByTestId('dataset-browse-pmd_live'));
    expect(await screen.findByTestId('dataset-rows-drift')).toBeInTheDocument();
  });
});
