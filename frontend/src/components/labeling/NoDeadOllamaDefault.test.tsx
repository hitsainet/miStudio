/**
 * MIS-E2E-147 (item 3) — the OpenAI-compatible endpoint must not default to a dead path.
 *
 * It defaulted to '/ollama/v1'. The Ollama proxy behind that path was removed on
 * 2026-09-13; the request fell through to the SPA and returned 200 + index.html, which
 * the panel reported as "No models found on this endpoint" — a routing failure dressed
 * as an empty model list. The fix is no default, not a new nginx location for a service
 * that no longer exists.
 *
 * MUTATION CONTROL: restore useState('/ollama/v1') -> this test fails.
 */
import { describe, it, expect } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

import { StartLabelingButton } from './StartLabelingButton';
import { useSettingsStore } from '../../stores/settingsStore';
import { LabelingMethod } from '../../types/labeling';

describe('OpenAI-compatible endpoint default', () => {
  it('starts EMPTY when nothing is saved, never /ollama/v1', async () => {
    useSettingsStore.setState({ settings: [] });
    render(<StartLabelingButton extractionId="extr_1" />);
    fireEvent.click(await screen.findByText(/Label Features/i));
    const method = screen.getByDisplayValue(/OpenAI \(/i) as HTMLSelectElement;
    fireEvent.change(method, { target: { value: LabelingMethod.OPENAI_COMPATIBLE } });
    const endpoint = await screen.findByPlaceholderText(/host\/v1/i);
    expect((endpoint as HTMLInputElement).value).toBe('');
    expect(screen.queryByDisplayValue('/ollama/v1')).toBeNull();
  });
});
