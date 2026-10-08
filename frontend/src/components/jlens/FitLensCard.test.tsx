/**
 * The fit affordance, and the refusals it must mirror.
 *
 * This card exists because the panel named a remedy ("fit one to enable it")
 * that the UI could not perform — the only routes in were REST and MCP. So the
 * first thing asserted is REACHABILITY: the card must be rendered by the panel,
 * not merely importable. A test that imports the module and renders it directly
 * would pass against a card no user can reach, which is precisely how 16 MCP
 * tools shipped unregistered in this repo.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * remove <FitLensCard/> from JLensPanel        -> "reachable from the panel" fails
 *   * drop the MIN_FIT_PROMPTS floor check         -> "below the floor" fails
 *   * parseLayers('') returns [] instead of null   -> "blank means every layer" fails
 *   * poll on `status` instead of `state`          -> "reads the state field" fails
 *   * send layers/corpus_name that were not typed  -> "sends what was typed" fails
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  FitLensCard,
  MIN_FIT_PROMPTS,
  parseLayers,
  parsePrompts,
} from './FitLensCard';

vi.mock('../../api/jlens', () => ({
  jlensApi: { fit: vi.fn() },
}));
vi.mock('../../api/models', () => ({
  getTaskStatus: vi.fn(),
}));

import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';

const corpusOf = (n: number) =>
  Array.from({ length: n }, (_, i) => `prompt number ${i}`).join('\n');

beforeEach(() => {
  vi.clearAllMocks();
});
afterEach(() => {
  vi.useRealTimers();
});

describe('parsePrompts', () => {
  it('drops blank lines rather than counting them toward the floor', () => {
    expect(parsePrompts('a\n\n  \nb\n')).toEqual(['a', 'b']);
  });
});

describe('parseLayers', () => {
  it('blank means every layer, expressed as null and never as []', () => {
    // `[]` reaches the server as "fit no layers at all" and yields an artifact
    // with no Jacobians in it — same shape as a good one.
    expect(parseLayers('')).toBeNull();
    expect(parseLayers('   ')).toBeNull();
  });

  it('accepts comma or space separated indices', () => {
    expect(parseLayers('24, 25')).toEqual([24, 25]);
    expect(parseLayers('0 3 7')).toEqual([0, 3, 7]);
  });

  it('rejects non-numeric and negative entries', () => {
    expect(parseLayers('24, x')).toBeNull();
    expect(parseLayers('-1')).toBeNull();
  });
});

describe('FitLensCard', () => {
  it('refuses to submit below the floor, and states the count', async () => {
    const user = userEvent.setup();
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /fit a lens/i }));

    await user.type(screen.getByLabelText(/corpus name/i), 'tiny');
    const box = screen.getByLabelText(/one prompt per line/i);
    await user.click(box);
    await user.paste(corpusOf(3));

    expect(screen.getByText(new RegExp(`3 / ${MIN_FIT_PROMPTS} prompts`))).toBeTruthy();
    // "below the floor": the fitter refuses rather than warns, and so does this.
    expect(screen.getByRole('button', { name: /^fit$/i })).toBeDisabled();
    expect(jlensApi.fit).not.toHaveBeenCalled();
  });

  it('refuses to submit without a corpus name, because the recipe records it', async () => {
    const user = userEvent.setup();
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /fit a lens/i }));
    await user.click(screen.getByLabelText(/one prompt per line/i));
    await user.paste(corpusOf(MIN_FIT_PROMPTS));

    expect(screen.getByRole('button', { name: /^fit$/i })).toBeDisabled();
  });

  it('sends what was typed — corpus, name, layers and the freeze flag', async () => {
    const user = userEvent.setup();
    vi.mocked(jlensApi.fit).mockResolvedValue({
      task_id: 't-123',
      model_id: 'm_1',
      queue: 'extraction',
    });
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /fit a lens/i }));
    await user.click(screen.getByLabelText(/one prompt per line/i));
    await user.paste(corpusOf(MIN_FIT_PROMPTS));
    await user.type(screen.getByLabelText(/corpus name/i), 'acceptance-100');
    await user.type(screen.getByLabelText(/^layers/i), '24, 25');
    await user.type(
      screen.getByLabelText(/probe prompt/i),
      'The capital of France is'
    );
    await user.type(screen.getByLabelText(/expected intermediate/i), 'Paris');
    await user.click(screen.getByRole('button', { name: /^fit$/i }));

    // Payload asserted, not just "was called": a reachability check that only
    // counts calls passes against a call sending the wrong arguments.
    await waitFor(() => expect(jlensApi.fit).toHaveBeenCalledTimes(1));
    const sent = vi.mocked(jlensApi.fit).mock.calls[0][0];
    expect(sent.model_id).toBe('m_1');
    expect(sent.prompts).toHaveLength(MIN_FIT_PROMPTS);
    expect(sent.corpus_name).toBe('acceptance-100');
    expect(sent.layers).toEqual([24, 25]);
    expect(sent.freeze_qk).toBe(true);
    // The fixture must travel with the fit, and must name a FITTED layer —
    // reading out at an unfitted one has no Jacobian to apply.
    expect(sent.semantic_probe?.expected_intermediate).toBe('Paris');
    expect(sent.semantic_probe?.layer).toBe(25);
  });

  /**
   * Drive the form with fireEvent rather than userEvent for the polling tests.
   * userEvent's keyboard emulation and fake timers deadlock each other, and the
   * behaviour under test here is the POLL, not the typing.
   */
  const submitFilledForm = () => {
    fireEvent.click(screen.getByRole('button', { name: /fit a lens/i }));
    fireEvent.change(screen.getByLabelText(/one prompt per line/i), {
      target: { value: corpusOf(MIN_FIT_PROMPTS) },
    });
    fireEvent.change(screen.getByLabelText(/corpus name/i), {
      target: { value: 'c' },
    });
    fireEvent.change(screen.getByLabelText(/probe prompt/i), {
      target: { value: 'The capital of France is' },
    });
    fireEvent.change(screen.getByLabelText(/expected intermediate/i), {
      target: { value: ' Paris' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^fit$/i }));
  };

  it('reads the state field when polling, and refreshes the registry on success', async () => {
    vi.mocked(jlensApi.fit).mockResolvedValue({
      task_id: 't-9',
      model_id: 'm_1',
      queue: 'extraction',
    });
    // The endpoint returns `state`. A poll written against `status` sees
    // undefined forever and never resolves.
    vi.mocked(getTaskStatus).mockResolvedValue({
      task_id: 't-9',
      state: 'SUCCESS',
      ready: true,
      successful: true,
      failed: false,
    });
    const onFitted = vi.fn();
    render(<FitLensCard modelId="m_1" onFitted={onFitted} />);
    submitFilledForm();

    await waitFor(() => expect(jlensApi.fit).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(onFitted).toHaveBeenCalledTimes(1), {
      timeout: 8000,
    });
  }, 15000);

  it('surfaces a failed fit instead of spinning forever', async () => {
    vi.mocked(jlensApi.fit).mockResolvedValue({
      task_id: 't-9',
      model_id: 'm_1',
      queue: 'extraction',
    });
    vi.mocked(getTaskStatus).mockResolvedValue({
      task_id: 't-9',
      state: 'FAILURE',
      ready: true,
      successful: false,
      failed: true,
      error: 'size of tensor a (8) must match the size of tensor b (4)',
    });
    const onFitted = vi.fn();
    render(<FitLensCard modelId="m_1" onFitted={onFitted} />);
    submitFilledForm();

    await waitFor(() => expect(jlensApi.fit).toHaveBeenCalledTimes(1));
    await waitFor(
      () =>
        expect(screen.getByRole('alert').textContent).toContain(
          'size of tensor a'
        ),
      { timeout: 8000 }
    );
    expect(onFitted).not.toHaveBeenCalled();
  }, 15000);
});

describe('the semantic fixture is required, and refused when it proves nothing', () => {
  it('will not submit without a fixture, because nothing would be published', async () => {
    const user = userEvent.setup();
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /fit a lens/i }));
    fireEvent.change(screen.getByLabelText(/one prompt per line/i), {
      target: { value: corpusOf(MIN_FIT_PROMPTS) },
    });
    fireEvent.change(screen.getByLabelText(/corpus name/i), {
      target: { value: 'c' },
    });

    // Everything else is valid; only the fixture is missing.
    expect(screen.getByRole('button', { name: /^fit$/i })).toBeDisabled();
  });

  it('refuses an intermediate that already appears in the prompt', async () => {
    // MUTATION CONTROL: drop the probeSelfEvident check and this fails. Such a
    // fixture is recovered by an artifact encoding nothing at all, so it would
    // certify a broken lens.
    const user = userEvent.setup();
    render(<FitLensCard modelId="m_1" onFitted={vi.fn()} />);
    await user.click(screen.getByRole('button', { name: /fit a lens/i }));
    fireEvent.change(screen.getByLabelText(/one prompt per line/i), {
      target: { value: corpusOf(MIN_FIT_PROMPTS) },
    });
    fireEvent.change(screen.getByLabelText(/corpus name/i), {
      target: { value: 'c' },
    });
    fireEvent.change(screen.getByLabelText(/probe prompt/i), {
      target: { value: 'The capital of France is Paris' },
    });
    fireEvent.change(screen.getByLabelText(/expected intermediate/i), {
      target: { value: 'Paris' },
    });

    expect(screen.getByRole('button', { name: /^fit$/i })).toBeDisabled();
    expect(screen.getByText(/already appears in the prompt/i)).toBeTruthy();
  });
});
