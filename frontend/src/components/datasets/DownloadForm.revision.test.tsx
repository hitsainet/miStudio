/**
 * Feature 035 FR-8: the optional revision field, and the commit the backend pinned.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DownloadForm } from './DownloadForm';

const COMMIT = 'fedcba9876543210fedcba9876543210fedcba98';

function fill(repo: string) {
  fireEvent.change(screen.getByLabelText('Repository ID'), { target: { value: repo } });
}

describe('DownloadForm revision', () => {
  it('has the labelled field and the helper text', () => {
    render(<DownloadForm onDownload={vi.fn()} />);
    expect(screen.getByLabelText('Revision (commit, branch or tag)')).toBeInTheDocument();
    expect(
      screen.getByText('Leave empty to pin the current head. A branch or tag is resolved to its commit.'),
    ).toBeInTheDocument();
  });

  it('passes a typed revision to onDownload', async () => {
    const onDownload = vi.fn().mockResolvedValue(undefined);
    render(<DownloadForm onDownload={onDownload} />);
    fill('org/ds');
    fireEvent.change(screen.getByLabelText('Revision (commit, branch or tag)'), { target: { value: ' v1.0 ' } });
    fireEvent.click(screen.getByText('Download'));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(1));
    expect(onDownload.mock.calls[0][4]).toBe('v1.0');
  });

  it('passes no revision when the field is blank', async () => {
    const onDownload = vi.fn().mockResolvedValue(undefined);
    render(<DownloadForm onDownload={onDownload} />);
    fill('org/ds');
    fireEvent.click(screen.getByText('Download'));
    await waitFor(() => expect(onDownload).toHaveBeenCalledTimes(1));
    expect(onDownload.mock.calls[0][4]).toBeUndefined();
  });

  it('shows the commit the backend resolved', async () => {
    const onDownload = vi.fn().mockResolvedValue({ id: 'x', hf_repo_id: 'org/ds', revision: COMMIT });
    render(<DownloadForm onDownload={onDownload} />);
    fill('org/ds');
    fireEvent.change(screen.getByLabelText('Revision (commit, branch or tag)'), { target: { value: 'main' } });
    fireEvent.click(screen.getByText('Download'));
    const pinned = await screen.findByTestId('pinned-commit');
    expect(pinned.textContent).toContain('org/ds pinned to commit');
    expect(screen.getByText(COMMIT.slice(0, 12))).toHaveAttribute('title', COMMIT);
  });
});
