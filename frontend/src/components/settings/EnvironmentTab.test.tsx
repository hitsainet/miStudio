/**
 * Settings → Environment (Feature 036; ENH-003 §2 "Consistent UI", D12).
 *
 * The network is the only fake: `fetchAPI` records every call, so each assertion checks the
 * PAYLOAD and the CALL COUNT of what the tab sends (memory: mutation-survivors-cluster-on-callers).
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { EnvironmentTab } from './EnvironmentTab';
import { parseDraft, editorFor, showValue } from './environmentFormat';
import { fetchAPI, APIError } from '../../api/client';
import type { EnvironmentListing, EnvironmentRow } from '../../api/environment';

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>();
  return { ...actual, fetchAPI: vi.fn() };
});

function row(over: Partial<EnvironmentRow>): EnvironmentRow {
  return {
    label: 'L', env_name: 'X', description: 'd', group: 'runtime', value: null, env_value: null,
    source: 'env', runtime_writable: true, effect: null, processes: [], restart: null,
    restart_warning: null, secret: false, is_set: null, updated_at: null, updated_by: null,
    reason: null, last_change: null, ...over,
  };
}

function listing(overrides: Record<string, Partial<EnvironmentRow>> = {}): EnvironmentListing {
  const settings: Record<string, EnvironmentRow> = {
    labeling_batch_size: row({
      label: 'Labeling batch size', env_name: 'LABELING_BATCH_SIZE', value: 8, env_value: 8,
      effect: 'next labeling job', description: 'How many features share one request.',
    }),
    log_level: row({ label: 'Log level', env_name: 'LOG_LEVEL', value: 'INFO', env_value: 'INFO', effect: 'now' }),
    auto_cleanup_after_download: row({
      label: 'Clean up after dataset downloads', env_name: 'AUTO_CLEANUP_AFTER_DOWNLOAD',
      value: true, env_value: true, effect: 'next download',
    }),
    log_format: row({
      label: 'Log format', env_name: 'LOG_FORMAT', group: 'restart', value: 'json', env_value: 'json',
      runtime_writable: false, restart: 'mistudio-backend pod',
      restart_warning: 'Restarting the mistudio-backend pod stops every running job.',
    }),
    MCP_TOOL_CATEGORIES: row({
      label: 'Mcp tool categories', env_name: 'MCP_TOOL_CATEGORIES', group: 'restart',
      source: 'external', runtime_writable: false, restart: 'mistudio-mcp pod',
    }),
    hf_token: row({
      label: 'Hf token', env_name: 'HF_TOKEN', group: 'environment', runtime_writable: false,
      secret: true, is_set: true,
    }),
    openai_api_key: row({
      label: 'Openai api key', env_name: 'OPENAI_API_KEY', group: 'environment', runtime_writable: false,
      secret: true, is_set: false,
    }),
  };
  for (const [key, patch] of Object.entries(overrides)) settings[key] = { ...settings[key], ...patch };
  return {
    settings, count: Object.keys(settings).length,
    runtime_keys: ['labeling_batch_size', 'log_level', 'auto_cleanup_after_download'], overrides: {},
  };
}

let current: EnvironmentListing;
let failNextPut: Error | null = null;

function calls(method: string) {
  return vi.mocked(fetchAPI).mock.calls.filter(([, init]) => (init?.method ?? 'GET') === method);
}

beforeEach(() => {
  vi.clearAllMocks();
  current = listing();
  failNextPut = null;
  vi.mocked(fetchAPI).mockImplementation(async (path: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET';
    if (path === '/environment' && method === 'GET') return current as never;
    if (path.startsWith('/environment/history')) {
      return { changes: [{ id: 1, key: 'log_level', action: 'set', previous_value: 'INFO',
        previous_source: 'env', new_value: 'ERROR', new_source: 'override', actor: 'operator',
        reason: 'noisy', changed_at: '2026-10-20T12:00:00Z' }], count: 1 } as never;
    }
    if (method === 'PUT') {
      if (failNextPut) throw failNextPut;
      return { key: 'x', value: 1, previous: 0, source: 'override' } as never;
    }
    if (method === 'DELETE') return { key: 'x', value: 1, previous: 0, source: 'env' } as never;
    throw new Error(`unexpected ${method} ${path}`);
  });
});

describe('EnvironmentTab', () => {
  it('shows the three sections and the history', async () => {
    render(<EnvironmentTab />);
    expect(await screen.findByText('Changeable while miStudio runs')).toBeInTheDocument();
    expect(screen.getByText('Needs a restart')).toBeInTheDocument();
    expect(screen.getByText('From the environment')).toBeInTheDocument();
    expect(screen.getByTestId('env-runtime-count')).toHaveTextContent('3 of 7 settings');
    expect(screen.getByTestId('env-effect-labeling_batch_size')).toHaveTextContent('Takes effect: next labeling job');
    expect(screen.getByTestId('env-history')).toHaveTextContent('noisy');
    // Restart rows name the pod; an external one says where it is set and shows no value.
    expect(screen.getByTestId('env-fixed-log_format')).toHaveTextContent('restart the mistudio-backend pod');
    expect(screen.getByTestId('env-fixed-MCP_TOOL_CATEGORIES')).toHaveTextContent("Set in the mistudio-mcp pod's environment");
    // Secrets: set / not set only.
    expect(screen.getByTestId('env-secret-hf_token')).toHaveTextContent(/^set/);
    expect(screen.getByTestId('env-secret-openai_api_key')).toHaveTextContent(/^not set/);
  });

  it('saves a typed value with the reason: one PUT, exact payload, then re-reads', async () => {
    const user = userEvent.setup();
    render(<EnvironmentTab />);
    const input = await screen.findByTestId('env-input-labeling_batch_size');
    // Wait for the loaded value to land before clearing: under full-suite load the draft could be
    // filled AFTER the clear, and the typed '4' became '84' (seen 2026-10-21, e2e/views-archive).
    await waitFor(() => expect((input as HTMLInputElement).value).not.toBe(''));
    await user.clear(input);
    await user.type(input, '4');
    expect((input as HTMLInputElement).value).toBe('4');
    await user.type(screen.getByTestId('env-reason-labeling_batch_size'), 'judge OOM');
    const before = calls('GET').filter(([p]) => p === '/environment').length;
    await user.click(screen.getByTestId('env-save-labeling_batch_size'));

    await waitFor(() => expect(calls('PUT')).toHaveLength(1));
    const [path, init] = calls('PUT')[0];
    expect(path).toBe('/environment/labeling_batch_size');
    expect(JSON.parse(String(init?.body))).toEqual({ value: 4, reason: 'judge OOM' });
    await waitFor(() =>
      expect(calls('GET').filter(([p]) => p === '/environment').length).toBe(before + 1));
  });

  it('sends a boolean as a boolean and the log level from its closed list', async () => {
    const user = userEvent.setup();
    render(<EnvironmentTab />);
    await user.selectOptions(await screen.findByTestId('env-input-auto_cleanup_after_download'), 'false');
    await user.click(screen.getByTestId('env-save-auto_cleanup_after_download'));
    await user.selectOptions(screen.getByTestId('env-input-log_level'), 'CRITICAL');
    await user.click(screen.getByTestId('env-save-log_level'));
    await waitFor(() => expect(calls('PUT')).toHaveLength(2));
    expect(JSON.parse(String(calls('PUT')[0][1]?.body))).toEqual({ value: false });
    expect(JSON.parse(String(calls('PUT')[1][1]?.body))).toEqual({ value: 'CRITICAL' });
  });

  it("shows the backend's refusal on the row", async () => {
    const user = userEvent.setup();
    failNextPut = new APIError(422, 'invalid_value: labeling_batch_size: Input should be less than or equal to 32');
    render(<EnvironmentTab />);
    const input = await screen.findByTestId('env-input-labeling_batch_size');
    await user.clear(input);
    await user.type(input, '99');
    await user.click(screen.getByTestId('env-save-labeling_batch_size'));
    expect(await screen.findByTestId('env-problem-labeling_batch_size'))
      .toHaveTextContent('less than or equal to 32');
  });

  it('offers "Use the environment value" only for an override, and sends one DELETE', async () => {
    const user = userEvent.setup();
    current = listing({
      log_level: { source: 'override', value: 'ERROR',
        last_change: { actor: 'operator', action: 'set', at: '2026-10-20T12:00:00Z' } },
    });
    render(<EnvironmentTab />);
    const rowEl = await screen.findByTestId('env-row-log_level');
    expect(within(rowEl).getByTestId('env-source-log_level')).toHaveTextContent('override');
    expect(within(rowEl).getByTestId('env-last-log_level')).toHaveTextContent('Set by operator');
    expect(screen.queryByTestId('env-reset-labeling_batch_size')).not.toBeInTheDocument();
    await user.click(within(rowEl).getByTestId('env-reset-log_level'));
    await waitFor(() => expect(calls('DELETE')).toHaveLength(1));
    expect(calls('DELETE')[0][0]).toBe('/environment/log_level');
  });

  it('a secret link opens the API Keys tab', async () => {
    const user = userEvent.setup();
    const open = vi.fn();
    render(<EnvironmentTab onOpenApiKeys={open} />);
    const secret = await screen.findByTestId('env-secret-hf_token');
    await user.click(within(secret).getByRole('button', { name: 'Manage in API Keys' }));
    expect(open).toHaveBeenCalledTimes(1);
  });
});

describe('draft parsing', () => {
  it('types each editor', () => {
    expect(editorFor('log_level', 'INFO')).toBe('log-level');
    expect(editorFor('steering_cluster_constants_json', '{}')).toBe('json');
    expect(editorFor('x', true)).toBe('boolean');
    expect(editorFor('x', 3)).toBe('number');
    expect(editorFor('neuronpedia_local_url', null)).toBe('text');
    expect(parseDraft('number', 'abc')).toEqual({ error: 'Enter a number' });
    expect(parseDraft('text', '  ')).toEqual({ value: null });
    expect(showValue(null)).toBe('not set');
    expect(showValue(false)).toBe('off');
  });
});
