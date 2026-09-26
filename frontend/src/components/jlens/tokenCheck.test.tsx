/**
 * A token verdict belongs to ONE model's vocabulary.
 *
 * `usable`, the piece count and the ids are all answers about a specific
 * tokenizer, and the cache is keyed by STRING alone. The intervention card is
 * never remounted on a model change, so a verdict earned against one model kept
 * being shown — and acted on — for the next.
 */

import { act, renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { useTokenCheck } from './tokenCheck';

vi.mock('../../api/jlens', () => ({
  jlensApi: { checkTokens: vi.fn() },
}));
import { jlensApi } from '../../api/jlens';

const verdict = (over = {}) => ({
  token: ' Rome',
  ids: [4874],
  n_tokens: 1,
  usable: true,
  detail: 'One token — usable as a direction.',
  ...over,
});

beforeEach(() => vi.clearAllMocks());

describe('useTokenCheck', () => {
  it('caches by string so the same token is not re-checked', async () => {
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([verdict()]);
    const { result } = renderHook(() => useTokenCheck('m_1'));

    await act(async () => {
      await result.current.check(' Rome');
    });
    await act(async () => {
      await result.current.check(' Rome');
    });
    expect(jlensApi.checkTokens).toHaveBeenCalledTimes(1);
  });

  it('DROPS every cached verdict when the model changes', async () => {
    /**
     * ' Rome' is one token in one vocabulary and three in another. Keeping the
     * old verdict showed a green "one token" badge for a model where it was
     * false, suppressed the weaker-evidence warning, and could fire the
     * encodes-to-nothing block on a model where the string encodes fine.
     *
     * MUTATION CONTROL: remove the `useEffect` that clears `checks` on
     * `modelId` and this fails — the stale verdict survives the switch.
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([verdict()]);
    const { result, rerender } = renderHook(({ id }) => useTokenCheck(id), {
      initialProps: { id: 'm_1' },
    });

    await act(async () => {
      await result.current.check(' Rome');
    });
    expect(result.current.checks[' Rome']).toBeTruthy();

    rerender({ id: 'm_2' });
    await waitFor(() =>
      expect(result.current.checks[' Rome']).toBeUndefined(),
    );
  });

  it('re-checks against the NEW model after a switch', async () => {
    /**
     * Clearing is only half of it: the next check must actually reach the
     * server, or the badge simply disappears and never comes back.
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([verdict()]);
    const { result, rerender } = renderHook(({ id }) => useTokenCheck(id), {
      initialProps: { id: 'm_1' },
    });
    await act(async () => {
      await result.current.check(' Rome');
    });

    rerender({ id: 'm_2' });
    await waitFor(() => expect(result.current.checks[' Rome']).toBeUndefined());

    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      verdict({ n_tokens: 3, usable: false, detail: '3 tokens' }),
    ]);
    await act(async () => {
      await result.current.check(' Rome');
    });
    expect(jlensApi.checkTokens).toHaveBeenCalledTimes(2);
    expect(result.current.checks[' Rome']?.usable).toBe(false);
  });

  it('rejected() flags a NO verdict and stays quiet otherwise', async () => {
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      verdict({ usable: false, n_tokens: 0, detail: 'Encodes to nothing' }),
    ]);
    const { result } = renderHook(() => useTokenCheck('m_1'));
    await act(async () => {
      await result.current.check(' Rome');
    });
    expect(result.current.rejected(' Rome')?.n_tokens).toBe(0);
    expect(result.current.rejected(' never-checked')).toBeUndefined();
  });

  it('a failed check never throws and never caches', async () => {
    /** An unreachable endpoint must not strand the form. */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('offline'),
    );
    const { result } = renderHook(() => useTokenCheck('m_1'));
    await act(async () => {
      await result.current.check(' Rome');
    });
    expect(result.current.checks[' Rome']).toBeUndefined();
    expect(result.current.checking).toBe(false);
  });
});
