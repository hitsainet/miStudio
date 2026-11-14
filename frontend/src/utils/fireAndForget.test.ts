/**
 * Guards the helper that stops a fire-and-forget store call from becoming an
 * unhandled promise rejection.
 *
 * MUTATION CONTROLS:
 *   * drop the `.catch()`        -> the rejection test fails
 *   * drop the guard on undefined -> the undefined test throws
 */

import { describe, it, expect, vi } from 'vitest';
import { fireAndForget } from './fireAndForget';

describe('fireAndForget', () => {
  it('swallows a rejection so it never reaches the runtime', async () => {
    const seen: unknown[] = [];
    const onUnhandled = (r: unknown) => seen.push(r);
    process.on('unhandledRejection', onUnhandled);

    fireAndForget(Promise.reject(new Error('network down')));
    // Two turns: one for the rejection, one for Node to decide it is unhandled.
    await new Promise((r) => setTimeout(r, 0));
    await new Promise((r) => setTimeout(r, 0));

    process.off('unhandledRejection', onUnhandled);
    expect(seen).toEqual([]);
  });

  it('does not disturb a promise that resolves', async () => {
    const after = vi.fn();
    fireAndForget(Promise.resolve('ok').then(after));
    await new Promise((r) => setTimeout(r, 0));

    expect(after).toHaveBeenCalledWith('ok');
  });

  it('tolerates a caller that returns nothing', () => {
    // Some store actions are typed Promise<void> but a mock may return
    // undefined; the helper must not become the thing that throws.
    expect(() => fireAndForget(undefined)).not.toThrow();
    expect(() => fireAndForget({} as unknown as Promise<unknown>)).not.toThrow();
  });
});
