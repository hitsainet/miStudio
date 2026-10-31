/**
 * Settings → API Keys → "Remove" PIN.
 *
 * The button called the generic `DELETE /settings/settings_pin_hash` (through
 * the settings store's `remove`). The backend refuses that key with 403 — on
 * purpose (MIS-E2E-055: a delete with no PIN check removed the gate) — so the
 * button showed "Failed to remove PIN" every time (found by the ENH-003
 * inventory, 2026-10-20). It now posts to `/settings/pin/remove` with the
 * current PIN, and sends none only while the bypass is active.
 *
 * NEGATIVE CONTROL (2026-10-20): restoring the old `remove('settings_pin_hash')`
 * handler fails every test that asserts the request.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { SettingsPanel } from './SettingsPanel';
import { useSettingsStore } from '../../stores/settingsStore';
import { fetchAPI, APIError } from '../../api/client';

vi.mock('../../stores/settingsStore');
vi.mock('../../stores/trainingsStore');
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return { ...actual, fetchAPI: vi.fn() };
});

const storeRemove = vi.fn().mockResolvedValue(undefined);

function setup(status: { configured: boolean; bypass_active: boolean }, removeResult?: unknown) {
  (useSettingsStore as never as { mockReturnValue: (v: unknown) => void }).mockReturnValue({
    settings: [],
    isLoading: false,
    fetchAll: vi.fn().mockResolvedValue(undefined),
    getByCategory: vi.fn().mockReturnValue([]),
    upsert: vi.fn().mockResolvedValue(undefined),
    remove: storeRemove,
  });
  vi.mocked(fetchAPI).mockImplementation(async (path: string) => {
    if (path === '/settings/pin/status') return status as never;
    if (path === '/settings/pin/verify') return { valid: true } as never;
    if (path === '/settings/pin/remove') {
      if (removeResult instanceof Error) throw removeResult;
      return (removeResult ?? { success: true, removed: true }) as never;
    }
    return [] as never;
  });
}

async function openApiKeysUnlocked(user: ReturnType<typeof userEvent.setup>, bypass: boolean) {
  render(<SettingsPanel />);
  await user.click(screen.getByRole('button', { name: 'API Keys' }));
  if (!bypass) {
    // The gate: enter the PIN to unlock the tab.
    await user.type(await screen.findByPlaceholderText('Enter PIN'), '1234');
    await user.click(screen.getByRole('button', { name: 'Unlock' }));
  }
  await screen.findByText('PIN is set');
}

function removeCalls() {
  return vi.mocked(fetchAPI).mock.calls.filter(([path]) => path === '/settings/pin/remove');
}

describe('Remove PIN', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // SettingsPanel resets scroll on mount; jsdom does not implement it.
    window.scrollTo = vi.fn() as never;
  });

  it('posts the current PIN to /settings/pin/remove, never the generic delete', async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: false });
    await openApiKeysUnlocked(user, false);

    await user.click(screen.getByRole('button', { name: 'Remove' }));
    await user.type(screen.getByLabelText('Current PIN'), '1234');
    await user.click(screen.getByRole('button', { name: 'Remove PIN' }));

    await screen.findByText('PIN removed');
    const calls = removeCalls();
    expect(calls).toHaveLength(1);
    const [, init] = calls[0];
    expect(init?.method).toBe('POST');
    expect(JSON.parse(String(init?.body))).toEqual({ current_pin: '1234' });
    expect(storeRemove).not.toHaveBeenCalled();
    expect(screen.queryByText('Failed to remove PIN')).not.toBeInTheDocument();
    // The section now offers to set a PIN.
    expect(screen.getByRole('button', { name: 'Set PIN' })).toBeInTheDocument();
  });

  it('will not send without the current PIN', async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: false });
    await openApiKeysUnlocked(user, false);

    await user.click(screen.getByRole('button', { name: 'Remove' }));
    expect(screen.getByRole('button', { name: 'Remove PIN' })).toBeDisabled();
    expect(removeCalls()).toHaveLength(0);
  });

  it("shows the backend's reason when the PIN is wrong, and keeps the PIN", async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: false }, new APIError(401, 'Current PIN is incorrect'));
    await openApiKeysUnlocked(user, false);

    await user.click(screen.getByRole('button', { name: 'Remove' }));
    await user.type(screen.getByLabelText('Current PIN'), '9999');
    await user.click(screen.getByRole('button', { name: 'Remove PIN' }));

    expect(await screen.findByText('Current PIN is incorrect')).toBeInTheDocument();
    expect(screen.queryByText('PIN removed')).not.toBeInTheDocument();
  });

  it('under the bypass, removes without asking for the current PIN', async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: true });
    await openApiKeysUnlocked(user, true);

    await user.click(screen.getByRole('button', { name: 'Remove' }));
    expect(screen.queryByLabelText('Current PIN')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Remove PIN' }));

    await screen.findByText('PIN removed');
    const calls = removeCalls();
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({});
  });

  it('Cancel returns to the PIN-is-set view without a request', async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: false });
    await openApiKeysUnlocked(user, false);

    await user.click(screen.getByRole('button', { name: 'Remove' }));
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.getByText('PIN is set')).toBeInTheDocument());
    expect(removeCalls()).toHaveLength(0);
  });
});
