/**
 * A prompt-template description the user clears is actually cleared — on UPDATE.
 *
 * `handleUpdate` sent `description.trim() || undefined`. `undefined` is dropped by
 * JSON.stringify, so the key never reached the backend, and
 * `prompt_template_service.py` updates with `model_dump(exclude_unset=True)`: a
 * MISSING key means "leave unchanged". Emptying the field therefore did nothing
 * and the old text came back, silently.
 *
 * `null` clears it (the column is `Text, nullable=True`) and is the convention
 * `TrainingTemplateForm` established for this defect — review round 2 lane R2-F,
 * control F5.
 *
 * `handleCreate` deliberately keeps `|| undefined`: on a create there is no stored
 * value to preserve, so omitting an empty optional is correct and sending `null`
 * would write an explicit empty where the column default would do. That asymmetry
 * is the point of the second test — a fix applied to both paths by pattern-matching
 * would be wrong.
 *
 * MUTATION CONTROLS (each applied alone, source restored by byte copy, sha256 verified):
 *   D4 update's `|| null` reverted to `|| undefined` -> RED, "clearing a description on update sends null"
 *   D5 create changed to `|| null`                   -> RED, "creating without a description omits the key"
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { PromptTemplatesPanel } from './PromptTemplatesPanel';
import { usePromptTemplatesStore } from '../../stores/promptTemplatesStore';
import * as api from '../../api/promptTemplates';
import type { PromptTemplate } from '../../types/promptTemplate';

vi.mock('../../api/promptTemplates');

const TEMPLATE: PromptTemplate = {
  id: 'pt_0001',
  name: 'Refusal probes',
  description: 'the old description',
  prompts: ['one', 'two'],
  tags: ['probe'],
  is_favorite: false,
  created_at: '2026-09-16T00:00:00Z',
  updated_at: '2026-09-16T00:00:00Z',
} as PromptTemplate;

beforeEach(() => {
  vi.mocked(api.getPromptTemplates).mockReset().mockResolvedValue({
    data: [TEMPLATE as any],
    pagination: { page: 1, limit: 50, total: 1, total_pages: 1, has_next: false, has_prev: false },
  } as any);
  vi.mocked(api.getFavoritePromptTemplates).mockReset().mockResolvedValue({ data: [] } as any);
  vi.mocked(api.updatePromptTemplate).mockReset().mockResolvedValue(TEMPLATE as any);
  vi.mocked(api.createPromptTemplate)
    .mockReset()
    .mockResolvedValue({ ...TEMPLATE, id: 'pt_new' } as any);
  usePromptTemplatesStore.setState({ templates: [], favorites: [], loading: false, error: null } as any);
});

describe('clearing a prompt-template description', () => {
  it('sends null on update, so the stored text is replaced', async () => {
    render(<PromptTemplatesPanel />);

    fireEvent.click(await screen.findByTitle(/Edit/i));
    fireEvent.change(screen.getByLabelText(/Description/i), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /Update Template|Save/ }));

    await waitFor(() => expect(api.updatePromptTemplate).toHaveBeenCalledTimes(1));
    const [, body] = vi.mocked(api.updatePromptTemplate).mock.calls[0];

    // `toBeNull`, not `toBeUndefined`: an absent key is exactly the defect.
    expect(body.description).toBeNull();
    expect('description' in body).toBe(true);
  });

  it('omits the key when creating without a description', async () => {
    render(<PromptTemplatesPanel />);

    // Exact: "New Template" opens the modal, "Create Template" submits it, and a
    // loose pattern matches both.
    fireEvent.click(await screen.findByRole('button', { name: /^New Template$/ }));
    fireEvent.change(screen.getByLabelText(/Name/i), { target: { value: 'fresh' } });
    // The "Prompts" label names a GROUP of dynamically added textareas, so it has no
    // single control to associate with and getByLabelText cannot reach them. Each
    // textarea carries its own placeholder, which is what actually identifies it.
    fireEvent.change(screen.getByPlaceholderText(/Prompt 1/), { target: { value: 'a prompt' } });
    fireEvent.click(screen.getByRole('button', { name: /^Create Template$/ }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledTimes(1));
    const [body] = vi.mocked(api.createPromptTemplate).mock.calls[0];

    // There is no stored value to preserve on a create, so an empty optional is
    // omitted rather than written as an explicit empty.
    expect(body.description).toBeUndefined();
  });
});
