/**
 * The prompt staging box collapses so the results below it get the screen.
 *
 * Five prompts fill most of the viewport, which pushes the generated output
 * below the fold. Collapsing must hand that space over without discarding the
 * prompts, and must still say what is staged — a collapsed box that shows
 * nothing hides state the user is about to run against.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PromptListEditor } from './PromptListEditor';

const PROMPTS = ['The old lighthouse keeper retired', 'Three students failed', ''];

function setup(overrides = {}) {
  const props = {
    prompts: PROMPTS,
    onAddPrompt: vi.fn(),
    onRemovePrompt: vi.fn(),
    onUpdatePrompt: vi.fn(),
    onClearPrompts: vi.fn(),
    onReplacePromptWithMultiple: vi.fn(),
    ...overrides,
  };
  return { ...render(<PromptListEditor {...props} />), props };
}

const toggle = () => screen.getByRole('button', { name: /Prompts/ });

beforeEach(() => {
  localStorage.clear();
});

describe('PromptListEditor collapse', () => {
  it('starts expanded, with the textareas visible', () => {
    setup();
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getAllByRole('textbox').length).toBe(PROMPTS.length);
  });

  it('collapsing hides the editor without discarding the prompts', () => {
    setup();
    fireEvent.click(toggle());

    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
    // `hidden` keeps them mounted — their values are still what gets submitted.
    const list = document.getElementById('steering-prompt-list');
    expect(list).toHaveAttribute('hidden');
    expect(screen.getAllByRole('textbox', { hidden: true }).length).toBe(PROMPTS.length);
  });

  it('says what is staged while collapsed', () => {
    setup();
    fireEvent.click(toggle());
    // 2 of the 3 have content.
    const summary = screen.getByText(/2 prompts ready/).closest('button')!;
    expect(summary).toBeInTheDocument();
    // Scoped to the summary: the same text is also the hidden textarea's value.
    expect(summary).toHaveTextContent('The old lighthouse keeper retired');
  });

  it('the collapsed summary expands it again', () => {
    setup();
    fireEvent.click(toggle());
    fireEvent.click(screen.getByText(/2 prompts ready/));
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
  });

  it('remembers the choice across a remount', () => {
    const { unmount } = setup();
    fireEvent.click(toggle());
    unmount();

    setup();
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');
  });

  it('still works when localStorage throws', () => {
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('site data blocked');
    });
    const setSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('site data blocked');
    });

    setup();
    expect(toggle()).toHaveAttribute('aria-expanded', 'true');
    fireEvent.click(toggle());
    expect(toggle()).toHaveAttribute('aria-expanded', 'false');

    spy.mockRestore();
    setSpy.mockRestore();
  });

  it('handles an empty staging box', () => {
    setup({ prompts: ['', ''] });
    fireEvent.click(toggle());
    expect(screen.getByText(/No prompts yet/)).toBeInTheDocument();
  });
});
