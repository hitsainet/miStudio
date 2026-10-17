/**
 * The shared assertion for "this element renders text that can arrive as one
 * unbroken run, so it must wrap".
 *
 * `whitespace-pre-wrap` only wraps at a soft-wrap opportunity that ALREADY
 * exists, and a run with no spaces in it has none — so pre-wrap alone lets the
 * line run past the right edge of its container. `break-words`
 * (`overflow-wrap: break-word`) adds the opportunity, and only for a run that
 * cannot fit a line on its own.
 *
 * `break-all` is refused rather than merely unused: it breaks ordinary prose
 * mid-word, and prose is the normal content at every one of these sites.
 *
 * These are styling fixes, so the class list IS the behaviour — jsdom does no
 * layout, and there is nothing else to observe.
 *
 * The same rule is asserted for steering output in
 * `src/components/steering/OutputWrapping.test.tsx` (commit f407a79c); this
 * helper exists so the rule is written down once.
 */
import { expect } from 'vitest';

export function expectWrapsLongRuns(el: HTMLElement | null | undefined): void {
  expect(el).toBeTruthy();
  const className = (el as HTMLElement).className;
  expect(className).toContain('whitespace-pre-wrap');
  expect(className).toContain('break-words');
  expect(className).not.toContain('break-all');
}
