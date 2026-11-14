/**
 * Feature 041 (FTASKS 14.1): family export and publish from the behaviour screen. The publish payload and its
 * call count are asserted; a blocked export lists every blocker; the ZIP is a link to the server's route.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '../../../api/client';

const api = vi.hoisted(() => ({
  exportIndex: vi.fn(),
  exportUrl: vi.fn((key: string, model?: string | null) => `/api/v1/probe-monitors/behaviours/${key}/family/export?model=${model}`),
  publishFamily: vi.fn(),
}));
vi.mock('../../../api/probeFamilies', () => ({ probeFamiliesApi: api }));

import { useProbeFamiliesStore } from '../../../stores/probeFamiliesStore';
import { FamilyExportActions } from '../FamilyExportActions';

beforeEach(() => {
  api.exportIndex.mockReset();
  api.publishFamily.mockReset();
  useProbeFamiliesStore.setState({ exportIndex: null, published: null, busy: false, refusal: null });
});

describe('FamilyExportActions', () => {
  it('publishes the family with exactly one POST carrying the model, repository and visibility', async () => {
    api.publishFamily.mockResolvedValue({ task_id: 't-1', members: ['pm_a', 'pm_b'] });
    render(<FamilyExportActions behaviourKey="k-x" model="m_1" />);
    await userEvent.click(screen.getByRole('button', { name: 'Publish the family' }));
    await userEvent.type(screen.getByTestId('probe-publish-repo'), 'owner/fam');
    await userEvent.click(screen.getByRole('button', { name: 'Publish' }));
    await waitFor(() => expect(api.publishFamily).toHaveBeenCalledTimes(1));
    expect(api.publishFamily).toHaveBeenCalledWith('k-x', { model: 'm_1', repo_id: 'owner/fam', private: true });
    expect(await screen.findByRole('status')).toHaveTextContent('Publishing 2 probe(s) in one commit');
  });

  it('carries a public choice through to the POST (the visibility is the dialog\'s, not a constant)', async () => {
    api.publishFamily.mockResolvedValue({ task_id: 't-2', members: ['pm_a'] });
    render(<FamilyExportActions behaviourKey="k-x" model="m_1" />);
    await userEvent.click(screen.getByRole('button', { name: 'Publish the family' }));
    await userEvent.type(screen.getByTestId('probe-publish-repo'), 'owner/open');
    await userEvent.click(screen.getByTestId('probe-publish-private'));
    await userEvent.click(screen.getByRole('button', { name: 'Publish' }));
    await waitFor(() => expect(api.publishFamily).toHaveBeenCalledTimes(1));
    expect(api.publishFamily).toHaveBeenCalledWith('k-x', { model: 'm_1', repo_id: 'owner/open', private: false });
  });

  it('lists every blocker of a blocked export', async () => {
    api.exportIndex.mockRejectedValue(new APIError(409, 'blocked', {
      code: 'family_export_blocked', message: '2 blocker(s) stop the family export',
      next: [{ tool: 'build_probe_definition', arguments: { probe_id: 'pm_a' }, why: 'build it' },
             { tool: 'build_probe_definition', arguments: { probe_id: 'pm_b' }, why: 'build that' }] }));
    render(<FamilyExportActions behaviourKey="k-x" model="m_1" />);
    await userEvent.click(screen.getByRole('button', { name: 'Check the family export' }));
    const blockers = await screen.findByTestId('family-blockers');
    expect(blockers).toHaveTextContent('2 blocker(s)');
    expect(blockers.querySelectorAll('li')).toHaveLength(2);
  });

  it('links the ZIP to the server route', () => {
    render(<FamilyExportActions behaviourKey="k-x" model="m_1" />);
    expect(screen.getByRole('link', { name: 'Download the family (ZIP)' })).toHaveAttribute(
      'href', '/api/v1/probe-monitors/behaviours/k-x/family/export?model=m_1');
  });
});
