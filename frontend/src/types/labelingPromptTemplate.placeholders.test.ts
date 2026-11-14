/**
 * OSD-16 — the placeholder rule the UI enforces must be the backend's rule.
 *
 * `requiredPlaceholder: '{tokens_table}'` named ONE placeholder while the backend
 * has always accepted either `{examples_block}` or `{tokens_table}`
 * (`labeling_prompt_template.py`), and nothing referenced the constant — dead and
 * wrong at the same time. The panel's own inline validation was correct, so this
 * never reached a user; it was a trap waiting for whoever used the constant.
 *
 * Now the constant lists both AND is what the panel reads, so the two cannot
 * drift apart. This test pins the pair, since a future single-value edit would
 * silently narrow what the UI accepts below what the API does.
 */
import { describe, it, expect } from 'vitest';
import { LabelingPromptTemplateConstraints } from './labelingPromptTemplate';

describe('the required-placeholder rule', () => {
  it('accepts either placeholder, matching the backend validator', () => {
    expect([...LabelingPromptTemplateConstraints.requiredPlaceholders].sort()).toEqual(
      ['{examples_block}', '{tokens_table}']
    );
  });

  it('is a list, not a single value — the old shape could only name one', () => {
    expect(Array.isArray(LabelingPromptTemplateConstraints.requiredPlaceholders)).toBe(true);
    expect(LabelingPromptTemplateConstraints.requiredPlaceholders.length).toBeGreaterThan(1);
  });

  it('no longer exposes the misleading single-placeholder key', () => {
    expect(
      (LabelingPromptTemplateConstraints as Record<string, unknown>).requiredPlaceholder
    ).toBeUndefined();
  });

  it('a template carrying either placeholder satisfies the rule', () => {
    const satisfies = (template: string) =>
      LabelingPromptTemplateConstraints.requiredPlaceholders.some((p) => template.includes(p));
    expect(satisfies('rank these: {examples_block}')).toBe(true);
    expect(satisfies('counts: {tokens_table}')).toBe(true);
    expect(satisfies('no placeholder at all')).toBe(false);
  });
});
