/**
 * Feature 035 FR-5 / FR-8: the commit a dataset holds, or the honest absence of one.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RevisionBadge, REVISION_NOT_RECORDED } from './RevisionBadge';
import { DatasetCard } from './DatasetCard';
import { DatasetStatus } from '../../types/dataset';
import type { Dataset } from '../../types/dataset';

const COMMIT = '0123456789abcdef0123456789abcdef01234567';

function dataset(overrides: Partial<Dataset> = {}): Dataset {
  return {
    id: 'd1',
    name: 'models-under-pressure (training)',
    source: 'HuggingFace',
    hf_repo_id: 'Arrrlex/models-under-pressure',
    status: DatasetStatus.READY,
    created_at: '2026-10-18T00:00:00Z',
    updated_at: '2026-10-18T00:00:00Z',
    ...overrides,
  };
}

describe('RevisionBadge', () => {
  it('shows the first 12 characters, with the full commit in the tooltip', () => {
    render(<RevisionBadge revision={COMMIT} />);
    const code = screen.getByText(COMMIT.slice(0, 12));
    expect(code).toHaveAttribute('title', COMMIT);
    expect(screen.queryByText(COMMIT)).toBeNull();
  });

  it('copies the FULL commit', () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<RevisionBadge revision={COMMIT} />);
    fireEvent.click(screen.getByLabelText('Copy full commit'));
    expect(writeText).toHaveBeenCalledWith(COMMIT);
  });

  it.each([null, undefined])('reads "revision not recorded" for %s, never latest or head', (value) => {
    const { container } = render(<RevisionBadge revision={value} />);
    expect(screen.getByText(REVISION_NOT_RECORDED)).toBeInTheDocument();
    expect(container.textContent?.toLowerCase()).not.toMatch(/latest|head/);
  });
});

describe('DatasetCard shows the revision (FR-8)', () => {
  it('a pinned row shows its short commit', () => {
    render(<DatasetCard dataset={dataset({ revision: COMMIT })} />);
    expect(screen.getByText(COMMIT.slice(0, 12))).toBeInTheDocument();
  });

  it('a legacy row says the revision was not recorded', () => {
    const { container } = render(<DatasetCard dataset={dataset({ revision: null })} />);
    expect(screen.getByText(REVISION_NOT_RECORDED)).toBeInTheDocument();
    expect(container.textContent?.toLowerCase()).not.toMatch(/latest|\bhead\b/);
  });

  it('a local registration shows no Hub revision at all', () => {
    render(<DatasetCard dataset={dataset({ source: 'local', hf_repo_id: undefined, revision: null })} />);
    expect(screen.queryByTestId('revision-badge')).toBeNull();
  });
});
