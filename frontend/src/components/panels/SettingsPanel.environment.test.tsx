/**
 * The Environment tab is wired into Settings and sits behind the PIN (Feature 036; ENH-003 D8),
 * and the Endpoints tab's Ollama card reads the runtime setting instead of writing `app_settings`
 * (D4).
 *
 * Controls (0xcc/reviews/036_implementation_controls_2026-10-20.md):
 *   C-TAB   delete the `{ id: 'environment', … }` TABS entry        → the first test is red
 *   C-PIN   render `<EnvironmentTab …/>` without `<PinGate>`         → the locked test is red
 *   C-OLL   restore the Ollama card's `upsert({ key: 'ollama_url' })` → the Endpoints test is red
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { SettingsPanel } from './SettingsPanel';
import { useSettingsStore } from '../../stores/settingsStore';
import { fetchAPI } from '../../api/client';

vi.mock('../../stores/settingsStore');
vi.mock('../../stores/trainingsStore');
vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return { ...actual, fetchAPI: vi.fn() };
});

const upsert = vi.fn().mockResolvedValue(undefined);
const remove = vi.fn().mockResolvedValue(undefined);

const LISTING = {
  settings: {
    ollama_url: {
      label: 'Labeling model service address', env_name: 'OLLAMA_URL', description: 'd',
      group: 'runtime', value: 'http://llm-override:8001', env_value: 'http://localhost:11434',
      source: 'override', runtime_writable: true, effect: 'next request', processes: ['API'],
      restart: null, restart_warning: null, secret: false, is_set: null, updated_at: null,
      updated_by: null, reason: null, last_change: null,
    },
  },
  count: 1, runtime_keys: ['ollama_url'], overrides: { ollama_url: 'http://llm-override:8001' },
};

function setup(pin: { configured: boolean; bypass_active: boolean }) {
  (useSettingsStore as never as { mockReturnValue: (v: unknown) => void }).mockReturnValue({
    settings: [], isLoading: false, fetchAll: vi.fn().mockResolvedValue(undefined),
    getByCategory: vi.fn().mockReturnValue([]), upsert, remove,
  });
  vi.mocked(fetchAPI).mockImplementation(async (path: string) => {
    if (path === '/settings/pin/status') return pin as never;
    if (path === '/settings/pin/verify') return { valid: true } as never;
    if (path === '/environment') return LISTING as never;
    if (path.startsWith('/environment/history')) return { changes: [], count: 0 } as never;
    return [] as never;
  });
}

describe('Settings → Environment', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.scrollTo = vi.fn() as never;
  });

  it('is a tab of the Settings panel and renders the environment listing', async () => {
    const user = userEvent.setup();
    setup({ configured: false, bypass_active: false });
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'Environment' }));
    expect(await screen.findByTestId('environment-tab')).toBeInTheDocument();
    expect(await screen.findByTestId('env-row-ollama_url')).toBeInTheDocument();
  });

  it('stays locked behind the PIN until it is entered', async () => {
    const user = userEvent.setup();
    setup({ configured: true, bypass_active: false });
    render(<SettingsPanel />);
    await user.click(screen.getByRole('button', { name: 'Environment' }));
    expect(await screen.findByPlaceholderText('Enter PIN')).toBeInTheDocument();
    expect(screen.queryByTestId('environment-tab')).not.toBeInTheDocument();
    // Only the Endpoints tab's Ollama card has read the listing; the locked tab has not.
    expect(screen.queryByTestId('env-row-ollama_url')).not.toBeInTheDocument();
    expect(screen.queryByTestId('env-history-empty')).not.toBeInTheDocument();

    await user.type(screen.getByPlaceholderText('Enter PIN'), '1234');
    await user.click(screen.getByRole('button', { name: 'Unlock' }));
    expect(await screen.findByTestId('environment-tab')).toBeInTheDocument();
  });

  it("the Endpoints tab shows the address in force and never writes app_settings' ollama_url", async () => {
    const user = userEvent.setup();
    setup({ configured: false, bypass_active: false });
    render(<SettingsPanel />);
    expect(await screen.findByTestId('ollama-url-value')).toHaveTextContent('http://llm-override:8001');
    expect(screen.getByTestId('ollama-url-source')).toHaveTextContent('override');
    expect(upsert).not.toHaveBeenCalledWith(expect.objectContaining({ key: 'ollama_url' }));
    expect(remove).not.toHaveBeenCalledWith('ollama_url');

    await user.click(screen.getByRole('button', { name: 'Change in Environment' }));
    await waitFor(() => expect(screen.getByTestId('environment-tab')).toBeInTheDocument());
  });
});
