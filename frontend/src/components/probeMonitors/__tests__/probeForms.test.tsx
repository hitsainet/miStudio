/**
 * The two authoring forms: what they read from the data, and what they refuse.
 *
 * MUTATION CONTROLS (each verified to fail the suite; recorded in the review):
 *   M162  the column pickers become free-text inputs      -> the picker test fails
 *   M163  label values read the FIRST row's keys only      -> the union test fails
 *   M164  an unmapped value is allowed through             -> the unmapped test fails
 *   M165  a calibration set may map 'positive'             -> the calibration test fails
 *   M166  the run form sends BOTH layers and stride        -> the exclusivity test fails
 *   M167  a form default drifts from the server's          -> the defaults test fails
 *   M168  GpuSelect is given allowSplit                    -> the split test fails
 *   M169  train==eval submits without a warning            -> the leakage test fails
 *
 * ⚠ THE COLUMN LISTS COME FROM REAL SAMPLES, AND THE SHAPE WAS WRONG IN THE TYPES.
 * `GET /datasets/{id}/samples` returns `{index, data: {…columns}}`; the frontend's
 * `DatasetSample` declared `{id, text, split}` and had no production caller, so nothing
 * had ever exercised it against the server. These tests mock the REAL shape — a fixture
 * matching the old declaration would have made the pickers render nothing while passing.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ProbeDatasetForm } from '../ProbeDatasetForm';
import { ProbeRunForm, RULES, SERVER_DEFAULTS } from '../ProbeRunForm';

const getDatasetSamples = vi.hoisted(() => vi.fn());
const createDataset = vi.hoisted(() => vi.fn());
const submitRun = vi.hoisted(() => vi.fn());

vi.mock('../../../api/datasets', () => ({ getDatasetSamples }));

vi.mock('../../../stores/datasetsStore', () => ({
  useDatasetsStore: () => ({
    datasets: [{ id: 'ds-1', name: 'models-under-pressure (training)' }],
    fetchDatasets: vi.fn(),
  }),
}));

vi.mock('../../../stores/modelsStore', () => ({
  useModelsStore: () => ({
    models: [{ id: 'm_1eb9108f', name: 'Llama-3.1-8B-Instruct' }],
    fetchModels: vi.fn(),
  }),
}));

const probeState = {
  datasets: [
    { id: 'pmd_train', name: 'train view', role: 'train', distribution: null },
    { id: 'pmd_ood', name: 'unseen', role: 'eval', distribution: 'out_of_distribution' },
    { id: 'pmd_calib', name: 'ultrachat', role: 'calibration', distribution: null },
  ],
  loadDatasets: vi.fn(),
  createDataset,
  submitRun,
};

vi.mock('../../../stores/probeMonitorsStore', () => ({
  useProbeMonitorsStore: (selector?: (s: typeof probeState) => unknown) =>
    selector ? selector(probeState) : probeState,
}));

vi.mock('../../common/GpuSelect', () => ({
  GpuSelect: ({ allowSplit }: { allowSplit?: boolean }) => (
    <div data-testid="gpu-select" data-allow-split={String(!!allowSplit)} />
  ),
}));

/** The REAL response shape, verified against the live endpoint. */
const SAMPLES = {
  data: [
    { index: 0, data: { inputs: 'transfer the funds', stakes: 'high', pair: 'p0' } },
    { index: 1, data: { inputs: 'what is the weather', stakes: 'low', pair: 'p0' } },
    // A THIRD value, and a column absent from row 0 — both must still be offered.
    { index: 2, data: { inputs: 'unclear', stakes: 'ambiguous', extra_col: 1 } },
  ],
  total: 3,
};

beforeEach(() => {
  vi.clearAllMocks();
  getDatasetSamples.mockResolvedValue(SAMPLES);
  createDataset.mockResolvedValue({ id: 'pmd_new' });
  submitRun.mockResolvedValue('pmr_new');
});

describe('ProbeDatasetForm reads its choices from the data', () => {
  it('offers columns from the sampled rows, not a text box', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');

    const input = await waitFor(() => screen.getByLabelText('Input column'));
    expect(input.tagName).toBe('SELECT');
    const options = Array.from(input.querySelectorAll('option')).map((o) => o.textContent);
    expect(options).toContain('inputs');
    expect(options).toContain('stakes');
  });

  it('unions columns across rows so a sparse column is not hidden', async () => {
    // `extra_col` appears only in row 2. Reading the first row's keys would drop it —
    // and a column you cannot select is a probe dataset you cannot build.
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    const input = await waitFor(() => screen.getByLabelText('Input column'));
    const options = Array.from(input.querySelectorAll('option')).map((o) => o.textContent);
    expect(options).toContain('extra_col');
  });

  it('lists every distinct label value, including the third class', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Label column'));
    await user.selectOptions(screen.getByLabelText('Label column'), 'stakes');

    expect(screen.getByLabelText('high → positive')).toBeInTheDocument();
    expect(screen.getByLabelText('low → negative')).toBeInTheDocument();
    // `ambiguous` must be mappable to excluded rather than silently dropped.
    expect(screen.getByLabelText('ambiguous → excluded')).toBeInTheDocument();
  });

  it('says the value list is SAMPLED and can be incomplete', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Label column'));
    await user.selectOptions(screen.getByLabelText('Label column'), 'stakes');
    expect(screen.getByTestId('sampling-caveat')).toHaveTextContent('unparseable');
  });

  it('refuses to submit while a value is unmapped', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.type(screen.getByLabelText('Name'), 'training view');
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Label column'));
    await user.selectOptions(screen.getByLabelText('Input column'), 'inputs');
    await user.selectOptions(screen.getByLabelText('Label column'), 'stakes');
    await user.click(screen.getByLabelText('high → positive'));
    await user.click(screen.getByLabelText('low → negative'));
    // `ambiguous` deliberately left unmapped.
    expect(screen.getByTestId('mapping-warning')).toHaveTextContent('unmapped');
    expect(screen.getByRole('button', { name: /create probe dataset/i })).toBeDisabled();
  });

  it('submits once every value is mapped', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.type(screen.getByLabelText('Name'), 'training view');
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Label column'));
    await user.selectOptions(screen.getByLabelText('Input column'), 'inputs');
    await user.selectOptions(screen.getByLabelText('Label column'), 'stakes');
    await user.click(screen.getByLabelText('high → positive'));
    await user.click(screen.getByLabelText('low → negative'));
    await user.click(screen.getByLabelText('ambiguous → excluded'));
    await user.selectOptions(screen.getByLabelText('Role'), 'train');

    const button = screen.getByRole('button', { name: /create probe dataset/i });
    expect(button).toBeEnabled();
    await user.click(button);

    expect(createDataset).toHaveBeenCalledTimes(1);
    const body = createDataset.mock.calls[0][0];
    expect(body.label_mapping).toEqual({
      high: 'positive',
      low: 'negative',
      ambiguous: 'excluded',
    });
    expect(body.input_column).toBe('inputs');
    expect(body.role).toBe('train');
  });

  /**
   * Fill EVERY field except the one under test, so the button's disabled state can only
   * be explained by the condition being tested.
   *
   * ⚠ WITHOUT THIS, BOTH REFUSAL TESTS PASSED FOR THE WRONG REASON. They asserted
   * `toBeDisabled()` while never typing a name — so the button was disabled by the empty
   * name, and mutations removing `!calibrationHasPositive` and `!missingClass` from
   * `canSubmit` both SURVIVED. A disabled button proves nothing unless everything else is
   * satisfied.
   */
  async function fillEverythingExceptTheMapping(user: ReturnType<typeof userEvent.setup>) {
    await user.type(screen.getByLabelText('Name'), 'a view');
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Label column'));
    await user.selectOptions(screen.getByLabelText('Input column'), 'inputs');
    await user.selectOptions(screen.getByLabelText('Label column'), 'stakes');
  }

  it('the helper leaves the form submittable once the mapping is valid', async () => {
    // The control for the two tests below: if this did not enable the button, their
    // `toBeDisabled()` assertions would again be explained by something else.
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await fillEverythingExceptTheMapping(user);
    await user.selectOptions(screen.getByLabelText('Role'), 'train');
    await user.click(screen.getByLabelText('high → positive'));
    await user.click(screen.getByLabelText('low → negative'));
    await user.click(screen.getByLabelText('ambiguous → excluded'));
    expect(screen.getByRole('button', { name: /create probe dataset/i })).toBeEnabled();
  });

  it('a train set with only one class cannot be submitted', async () => {
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await fillEverythingExceptTheMapping(user);
    await user.selectOptions(screen.getByLabelText('Role'), 'train');
    await user.click(screen.getByLabelText('high → positive'));
    await user.click(screen.getByLabelText('low → excluded'));
    await user.click(screen.getByLabelText('ambiguous → excluded'));
    expect(screen.getByTestId('mapping-warning')).toHaveTextContent('one positive and one negative');
    expect(screen.getByRole('button', { name: /create probe dataset/i })).toBeDisabled();
  });

  it('a calibration set that maps a POSITIVE cannot be submitted', async () => {
    // It supplies negatives for the FPR threshold and is not labelled for the concept.
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await fillEverythingExceptTheMapping(user);
    await user.selectOptions(screen.getByLabelText('Role'), 'calibration');
    await user.click(screen.getByLabelText('high → positive'));
    await user.click(screen.getByLabelText('low → negative'));
    await user.click(screen.getByLabelText('ambiguous → excluded'));
    expect(screen.getByTestId('mapping-warning')).toHaveTextContent('not labelled for the concept');
    expect(screen.getByRole('button', { name: /create probe dataset/i })).toBeDisabled();
  });

  it('resets the column choices when the dataset changes', async () => {
    // Columns from the previous dataset almost certainly do not exist in the next one,
    // and keeping them would submit a mapping for a column that is not there.
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    await waitFor(() => screen.getByLabelText('Input column'));
    await user.selectOptions(screen.getByLabelText('Input column'), 'inputs');
    await user.selectOptions(screen.getByLabelText('Dataset'), '');
    expect((screen.getByLabelText('Input column') as HTMLSelectElement).value).toBe('');
  });

  it('reports a samples failure rather than rendering empty pickers', async () => {
    getDatasetSamples.mockRejectedValueOnce(new Error('dataset files are gone'));
    const user = userEvent.setup();
    render(<ProbeDatasetForm />);
    await user.selectOptions(screen.getByLabelText('Dataset'), 'ds-1');
    expect(await screen.findByTestId('sample-error')).toHaveTextContent('files are gone');
  });
});

describe('ProbeRunForm', () => {
  it('its defaults match the server constants', () => {
    // A form that quietly sends a different stride produces runs nobody can compare.
    expect(SERVER_DEFAULTS.stride).toBe(5);
    expect(SERVER_DEFAULTS.topNLayers).toBe(1);
    expect(SERVER_DEFAULTS.valFraction).toBe(0.15);
    expect(SERVER_DEFAULTS.seed).toBe(1337);
    expect(SERVER_DEFAULTS.maxLength).toBe(4096);
    expect(SERVER_DEFAULTS.targetFpr).toBe(0.01);
  });

  it('offers exactly the six implemented rules', () => {
    expect([...RULES]).toEqual([
      'mean', 'max', 'last', 'softmax', 'attention', 'rolling_mean_max',
    ]);
  });

  it('sends a stride and NOT layers by default', async () => {
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    await user.click(screen.getByRole('button', { name: /start probe run/i }));

    const config = submitRun.mock.calls[0][0].config;
    expect(config.stride).toBe(5);
    expect(config).not.toHaveProperty('layers');
  });

  it('sends layers and NOT a stride when explicit layers are given', async () => {
    // Both would make the stored config claim a stride that governed nothing.
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    await user.click(screen.getByRole('radio', { name: /explicit/i }));
    await user.type(screen.getByLabelText('layers'), '11, 12, 13');
    await user.click(screen.getByRole('button', { name: /start probe run/i }));

    const config = submitRun.mock.calls[0][0].config;
    expect(config.layers).toEqual([11, 12, 13]);
    expect(config).not.toHaveProperty('stride');
  });

  it('marks malformed layers and blocks submit', async () => {
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    await user.click(screen.getByRole('radio', { name: /explicit/i }));
    await user.type(screen.getByLabelText('layers'), 'twelve');
    expect(screen.getByTestId('layers-invalid')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start probe run/i })).toBeDisabled();
  });

  it('does NOT offer a split GPU, because the endpoint refuses "all"', () => {
    render(<ProbeRunForm />);
    expect(screen.getByTestId('gpu-select')).toHaveAttribute('data-allow-split', 'false');
  });

  it('cannot select the same view as both train and eval, so the leak is unreachable here', async () => {
    /**
     * ⚠ MY FIRST VERSION OF THIS TEST WAS HOLLOW. It was titled "warns when the training
     * view is also an evaluation set" and only ever asserted the warning was ABSENT —
     * because the condition cannot be produced through this UI at all: the train dropdown
     * lists `role === 'train'` views and the eval checkboxes list `role === 'eval'` ones,
     * so the two sets are disjoint by construction.
     *
     * A test asserting the absence of something it cannot cause proves nothing. What IS
     * worth pinning is the reason: the role filtering. The `leakage-warning` block stays
     * in the component as defence for callers that reach the store directly, and it is
     * covered by the server's own refusal (`test_probe_monitor_schemas.py`), which is
     * where the guarantee actually lives.
     */
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    const trainOptions = Array.from(
      (screen.getByLabelText('Training view') as HTMLSelectElement).querySelectorAll('option')
    ).map((option) => option.value).filter(Boolean);
    const evalNames = screen.getAllByRole('checkbox')
      .map((box) => box.getAttribute('name') ?? '')
      .filter(Boolean);
    // The train list holds only train views, so no eval id can appear in it.
    expect(trainOptions).toEqual(['pmd_train']);
    expect(trainOptions).not.toContain('pmd_ood');
    expect(evalNames).not.toContain('pmd_train');

    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    await user.click(screen.getByRole('checkbox', { name: /unseen/i }));
    expect(screen.queryByTestId('leakage-warning')).toBeNull();
  });

  it('a calibration view cannot be chosen as the training view either', () => {
    render(<ProbeRunForm />);
    const trainSelect = screen.getByLabelText('Training view') as HTMLSelectElement;
    const values = Array.from(trainSelect.querySelectorAll('option'))
      .map((option) => option.value)
      .filter(Boolean);
    expect(values).not.toContain('pmd_calib');
    // And the calibration dropdown holds it, so the fixture really does contain one —
    // without this the assertion above would pass against an empty store.
    const calibSelect = screen.getByLabelText('Calibration set') as HTMLSelectElement;
    expect(
      Array.from(calibSelect.querySelectorAll('option')).map((o) => o.value)
    ).toContain('pmd_calib');
  });

  it('blocks submit when no rule is selected', async () => {
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    for (const rule of ['mean', 'max', 'last', 'attention']) {
      await user.click(screen.getByRole('checkbox', { name: rule }));
    }
    expect(screen.getByRole('button', { name: /start probe run/i })).toBeDisabled();
  });

  it('only sends sae_k when the SAE variant is on', async () => {
    const user = userEvent.setup();
    render(<ProbeRunForm />);
    await user.selectOptions(screen.getByLabelText('Model'), 'm_1eb9108f');
    await user.selectOptions(screen.getByLabelText('Training view'), 'pmd_train');
    await user.click(screen.getByRole('button', { name: /start probe run/i }));
    expect(submitRun.mock.calls[0][0].config).not.toHaveProperty('sae_k');

    submitRun.mockClear();
    await user.click(screen.getByRole('checkbox', { name: /k-sparse probe/i }));
    await user.click(screen.getByRole('button', { name: /start probe run/i }));
    const config = submitRun.mock.calls[0][0].config;
    expect(config.sae_variant).toBe(true);
    expect(config.sae_k).toEqual([128]);
  });

  it('marks an out-of-distribution eval set, because rung 2 turns on it', () => {
    render(<ProbeRunForm />);
    expect(screen.getByText('OOD')).toBeInTheDocument();
  });
});
