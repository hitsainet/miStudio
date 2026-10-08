/**
 * Both preview blocks in the template preview modal must wrap.
 *
 * `system_message` and `user_prompt_template` are operator-authored and
 * unconstrained in line length: a rule with a long URL in it, a pasted
 * delimiter, or a long identifier has no space to break at, and both blocks had
 * `whitespace-pre-wrap` and nothing else — which only wraps at a break
 * opportunity that already exists.
 *
 * The preview modal is `max-h-[90vh] overflow-auto`, so an unbroken line scrolls
 * the modal sideways rather than being clipped, and the operator reading the
 * template loses the left edge of every other line while doing it.
 *
 * ⚠ TWO SITES, TWO ASSERTIONS. The system message and the user prompt are
 * separate elements with separately-written class lists; one test covering "the
 * preview modal" would pass with either half still broken.
 *
 * MUTATION CONTROLS: recorded in the commit message.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act, render, screen, fireEvent } from '@testing-library/react';
import type { LabelingPromptTemplate } from '../../types/labelingPromptTemplate';
import { expectWrapsLongRuns } from '../../test/expectWrapsLongRuns';

const SYSTEM_MESSAGE =
  'You label SAE features.\nRefer to ' + 'https://example.invalid/a/'.repeat(12);

/**
 * Keeps `{tokens_table}` out of it: the panel substitutes that placeholder with
 * a canned table, and the element's text would then not be what was authored.
 */
const USER_PROMPT =
  'Given {examples_block}, name the pattern.\nid=' + 'A'.repeat(250);

const template: LabelingPromptTemplate = {
  id: 'lpt_wrap',
  prompt_fingerprint: 'sha256:deadbeef',
  name: 'Wrapping fixture',
  description: 'fixture',
  system_message: SYSTEM_MESSAGE,
  user_prompt_template: USER_PROMPT,
  temperature: 0.3,
  max_tokens: 50,
  top_p: 0.9,
  include_nlp_analysis: false,
  is_default: false,
  is_system: false,
  created_at: '2026-09-27T00:00:00Z',
  updated_at: '2026-09-27T00:00:00Z',
};

const storeState = {
  templates: [template],
  defaultTemplate: null,
  loading: false,
  error: null,
  fetchTemplates: vi.fn(),
  fetchDefaultTemplate: vi.fn(),
  createTemplate: vi.fn(),
  updateTemplate: vi.fn(),
  deleteTemplate: vi.fn(),
  cloneTemplate: vi.fn(),
  setDefaultTemplate: vi.fn(),
  clearError: vi.fn(),
};

vi.mock('../../stores/labelingPromptTemplatesStore', () => ({
  useLabelingPromptTemplatesStore: () => storeState,
}));

vi.mock('../../api/labelingPromptTemplates', () => ({
  exportLabelingPromptTemplates: vi.fn(),
  importLabelingPromptTemplates: vi.fn(),
  getLabelingPromptTemplateUsageCount: vi.fn().mockResolvedValue({ usage_count: 0 }),
}));

import { LabelingPromptTemplatesPanel } from './LabelingPromptTemplatesPanel';

async function openPreview() {
  render(<LabelingPromptTemplatesPanel />);
  // The usage-count effect resolves after mount; flush it so its setState does
  // not land outside act() and print a warning over the real assertions.
  await act(async () => {});
  fireEvent.click(screen.getByTitle('Preview'));
}

describe('the template preview wraps space-free lines', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('breaks inside a long run in the system message', async () => {
    await openPreview();
    // Located by the authored text, so the class under assertion is not the selector.
    const block = screen.getByText(SYSTEM_MESSAGE, { normalizer: (t) => t });
    expect(block.tagName).toBe('PRE');
    expectWrapsLongRuns(block);
  });

  it('breaks inside a long run in the user prompt template', async () => {
    await openPreview();
    const block = screen.getByText(USER_PROMPT, { normalizer: (t) => t });
    expect(block.tagName).toBe('PRE');
    expectWrapsLongRuns(block);
  });
});
