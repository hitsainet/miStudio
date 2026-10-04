/**
 * A corpus sample must wrap in the Samples tab, not scroll it sideways.
 *
 * The sample `<pre>` renders raw upstream text, and this project's corpora are
 * full of runs with no space in them: codeparrot source lines, base64 blobs and
 * long URLs in the Pile, and — for a non-string column — a `JSON.stringify`
 * with no break opportunity of its own. `whitespace-pre-wrap` alone does not
 * help, because pre-wrap only wraps where an opportunity already exists.
 *
 * The tab body is `flex-1 overflow-y-auto`, and `overflow-y: auto` makes
 * `overflow-x` compute to `auto` too, so one such row gives the sample list its
 * own inner horizontal scrollbar.
 *
 * MUTATION CONTROL: recorded in the commit message.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { DatasetDetailModal } from './DatasetDetailModal';
import { DatasetStatus } from '../../types/dataset';
import type { Dataset } from '../../types/dataset';
import { expectWrapsLongRuns } from '../../test/expectWrapsLongRuns';

vi.mock('../common/StatusBadge', () => ({
  StatusBadge: ({ status }: { status: string }) => <span>{status}</span>,
}));
vi.mock('../../utils/formatters', () => ({
  formatFileSize: (b: number) => `${b} B`,
  formatDateTime: (d: string) => d,
}));
vi.mock('../../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
  WS_URL: '',
  WS_PATH: '/ws/socket.io',
}));

/** A real shape: a code line with no space in it, after a newline. */
const SAMPLE_TEXT =
  'import os\n' + 'os.environ["X"]=' + '"aGVsbG8gd29ybGQ='.repeat(20) + '"';

const dataset: Dataset = {
  id: 'ds_wrap',
  name: 'codeparrot-valid',
  source: 'huggingface',
  hf_repo_id: 'codeparrot/valid',
  status: DatasetStatus.READY,
  progress: 100,
  created_at: '2026-09-27T00:00:00Z',
  updated_at: '2026-09-27T00:00:00Z',
  num_samples: 1,
  size_bytes: 1024,
};

describe('a dataset sample wraps a space-free run', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // `globalThis`, not `global`: the test-file type-check has no @types/node,
    // and the ratchet counts a new error against this file.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          data: [{ index: 0, data: { content: SAMPLE_TEXT } }],
          pagination: null,
        }),
      }),
    );
  });

  it('breaks inside the run instead of scrolling the sample list sideways', async () => {
    render(<DatasetDetailModal dataset={dataset} onClose={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: 'Samples' }));

    // Located by the sample's own text, not by the class under assertion.
    const block = await waitFor(() =>
      screen.getByText(SAMPLE_TEXT, { normalizer: (t) => t }),
    );
    expect(block.tagName).toBe('PRE');
    expectWrapsLongRuns(block);
  });
});
