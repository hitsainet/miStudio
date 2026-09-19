/**
 * Start Training and Save as Template send what the form holds, through the REAL
 * trainings and templates stores (review round 3, R3-C).
 *
 * R2D-6. The start payload re-listed its keys and never sent `evaluate_ce_delta`,
 * `evaluation_token_budget`, `holdout_eval_tokens`, `holdout_eval_chunk_tokens` or
 * `seed`. A template carrying them loaded them into the form and lost them at launch.
 *
 * R2F-6. Save as Template listed its keys by hand and omitted `hook_types`,
 * `log_interval`, `holdout_fraction`, `dataset_weights`, `seed` and
 * `evaluate_ce_delta`. The backend stored its default for each.
 *
 * THE FIX THIS FILE ALSO GUARDS. Now that those fields are sent, a template that
 * lacks one must not leave the previous template's value in the form, and a
 * template's weights must pair with datasets by id, never by position.
 *
 * Only the HTTP edges are stubbed: the model and dataset lists, the WebSocket hooks,
 * the extraction list (`fetch`), and the two store actions that would POST. Each
 * test asserts the WHOLE request as sent on the wire (a JSON round trip), written out
 * literally, and the number of requests.
 *
 * MUTATION CONTROLS (the table, with results, is in
 * .claude/context/sessions/review_sae_remediation_R3_C_2026-09-15.md):
 *   M1  Start Training builds its hyperparameters by hand again (without the evaluation block)
 *   M2  Save as Template restores its hand-written key list (the d85e1654 block)
 *   M3  Save as Template pairs weights with extraction_ids, as Start does
 *   M4  handleTemplateLoad skips the written-out extras update
 *   M5  a template's dataset-keyed weights are not resolved for the selected sources
 *   M6  the unpaired-weights notice is never rendered
 *   M7  Start does not check the evaluation fields before sending
 *   M8  Save does not check the evaluation fields before sending
 *   M9  the Seed input's onChange writes nothing
 * Three controls SURVIVED the first run, each a gap now closed by a test below and
 * re-run as a negative control:
 *   W1  the mixture inputs show the raw map, not the resolved weight the request sends
 *   S2  Save does not check the LR schedule
 *   S3  Save shows axios's status line instead of the server's detail
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { TrainingPanel } from './TrainingPanel';
import { useTrainingsStore } from '../../stores/trainingsStore';
import type { TrainingConfig } from '../../stores/trainingsStore';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import { useDeletionProgressWebSocket } from '../../hooks/useDeletionProgressWebSocket';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { SAEArchitectureType } from '../../types/training';

vi.mock('../../stores/modelsStore');
vi.mock('../../stores/datasetsStore');
vi.mock('../../hooks/useTrainingWebSocket');
vi.mock('../../hooks/useDeletionProgressWebSocket');
vi.mock('../../contexts/WebSocketContext');

const MODEL = 'm_b5911e07';
const DS_WEB = 'ds_owt';
const DS_CODE = 'ds_code';
const EXT_WEB = 'ext_m_b5911e07_20260914_web';
const EXT_CODE = 'ext_m_b5911e07_20260914_code';

const extraction = (id: string, datasetId: string) => ({
  extraction_id: id,
  dataset_id: datasetId,
  status: 'completed',
  layer_indices: [11, 12, 13],
  hook_types: ['residual', 'mlp'],
  num_samples_processed: 100,
  created_at: '2026-09-14T10:00:00Z',
  statistics: { layer_11_residual: { shape: [100, 2048, 2048] } },
});

/** The form after choosing the model and two datasets, web first. */
const FORM: TrainingConfig = {
  model_id: MODEL,
  dataset_ids: [DS_WEB, DS_CODE],
  hidden_dim: 2048,
  latent_dim: 16384,
  architecture_type: SAEArchitectureType.STANDARD_SAELENS,
  training_layers: [0],
  hook_types: ['residual'],
  l1_alpha: 5e-4,
  target_l0: 0.05,
  normalize_activations: 'constant_norm_rescale',
  holdout_fraction: 0,
  initial_threshold: 0.5,
  bandwidth: 0.01,
  ste_bandwidth: 0.5,
  sparsity_coeff: 1e-3,
  normalize_decoder: true,
  learning_rate: 4e-4,
  batch_size: 2048,
  total_steps: 50000,
  warmup_steps: 2000,
  lr_decay_steps: 0,
  sparsity_warmup_steps: 5000,
  weight_decay: 0,
  grad_clip_norm: 1,
  checkpoint_interval: 5000,
  log_interval: 100,
  dead_neuron_threshold: 1000,
  resample_dead_neurons: true,
  resample_interval: 5000,
};

/** Stored as the backend stores a create: the complete dump, nulls included. */
const HP_COMMON = {
  hidden_dim: 2048,
  latent_dim: 16384,
  architecture_type: 'jumprelu',
  training_layers: [11, 12, 13],
  l1_alpha: null,
  target_l0: 0.05,
  top_k_sparsity: null,
  normalize_activations: 'constant_norm_rescale',
  top_k: null,
  aux_k: null,
  aux_loss_alpha: null,
  adam_epsilon: null,
  initial_threshold: 0.5,
  ste_bandwidth: 0.5,
  bandwidth: 0.01,
  sparsity_coeff: 0.001,
  normalize_decoder: true,
  learning_rate: 0.00007,
  batch_size: 2048,
  warmup_steps: 2000,
  sparsity_warmup_steps: 10000,
  weight_decay: 0.0,
  grad_clip_norm: 1.0,
  dead_neuron_threshold: 10000,
  resample_dead_neurons: true,
  resample_interval: 5000,
};

const template = (id: string, name: string, datasetIds: string[], hyperparameters: Record<string, unknown>) => ({
  id,
  name,
  description: '',
  model_id: null,
  dataset_ids: datasetIds,
  dataset_id: null,
  encoder_type: 'jumprelu',
  is_favorite: false,
  created_at: '2026-09-14T16:06:00Z',
  updated_at: '2026-09-14T16:06:00Z',
  hyperparameters,
});

/** A 16K template that carries every field this round sends. Weights follow ITS dataset order: code, web. */
const TEMPLATE_16K = template('tmpl_16k', 'LFM2.5-1.2B L11-13 JumpReLU 16K', [DS_CODE, DS_WEB], {
  ...HP_COMMON,
  hook_types: ['residual', 'mlp'],
  total_steps: 150000,
  lr_decay_steps: 30000,
  checkpoint_interval: 10000,
  log_interval: 250,
  holdout_fraction: 0.1,
  evaluate_ce_delta: false,
  evaluation_token_budget: 65536,
  holdout_eval_tokens: 200000,
  holdout_eval_chunk_tokens: 4096,
  seed: 1004206375,
  dataset_weights: [0.25, 0.75],
});

/** An older template, saved before any of this round's fields were stored. */
const TEMPLATE_OLDER = template('tmpl_older', 'JumpReLU before R3', [DS_CODE, DS_WEB], {
  ...HP_COMMON,
  total_steps: 50000,
  lr_decay_steps: 0,
  checkpoint_interval: 2000,
});

/** Weights for five datasets on a template that names none (the stored 16K template names none). */
const TEMPLATE_UNPAIRED = template('tmpl_unpaired', 'weights without datasets', [], {
  ...HP_COMMON,
  total_steps: 50000,
  lr_decay_steps: 0,
  checkpoint_interval: 2000,
  dataset_weights: [0.9, 0.1],
});

/** What the 16K template's JumpReLU run sends, whatever the mixture reference. */
const HP_16K_WITHOUT_WEIGHTS = {
  hidden_dim: 2048,
  latent_dim: 16384,
  architecture_type: 'jumprelu',
  training_layers: [11, 12, 13],
  hook_types: ['residual', 'mlp'],
  normalize_activations: 'constant_norm_rescale',
  learning_rate: 0.00007,
  batch_size: 2048,
  total_steps: 150000,
  warmup_steps: 2000,
  sparsity_warmup_steps: 10000,
  weight_decay: 0,
  grad_clip_norm: 1,
  checkpoint_interval: 10000,
  log_interval: 250,
  lr_decay_steps: 30000,
  resample_dead_neurons: true,
  resample_interval: 5000,
  dead_neuron_threshold: 10000,
  sparsity_coeff: 0.001,
  target_l0: 0.05,
  initial_threshold: 0.5,
  bandwidth: 0.01,
  ste_bandwidth: 0.5,
  normalize_decoder: true,
  holdout_fraction: 0.1,
  evaluate_ce_delta: false,
  evaluation_token_budget: 65536,
  holdout_eval_tokens: 200000,
  holdout_eval_chunk_tokens: 4096,
  seed: 1004206375,
};

/** What the older template sends: no evaluation key, no seed, no weights, and the defaults it lacks. */
const HP_OLDER = {
  hidden_dim: 2048,
  latent_dim: 16384,
  architecture_type: 'jumprelu',
  training_layers: [11, 12, 13],
  hook_types: ['residual'],
  normalize_activations: 'constant_norm_rescale',
  learning_rate: 0.00007,
  batch_size: 2048,
  total_steps: 50000,
  warmup_steps: 2000,
  sparsity_warmup_steps: 10000,
  weight_decay: 0,
  grad_clip_norm: 1,
  checkpoint_interval: 2000,
  log_interval: 100,
  lr_decay_steps: 0,
  resample_dead_neurons: true,
  resample_interval: 5000,
  dead_neuron_threshold: 10000,
  sparsity_coeff: 0.001,
  target_l0: 0.05,
  initial_threshold: 0.5,
  bandwidth: 0.01,
  ste_bandwidth: 0.5,
  normalize_decoder: true,
};

const createTraining = vi.fn();
const createTemplate = vi.fn();
let alertSpy: ReturnType<typeof vi.spyOn>;

type Mocked = { mockReturnValue: (value: unknown) => void };

const wire = (value: unknown) => JSON.parse(JSON.stringify(value));

beforeEach(() => {
  createTraining.mockReset().mockResolvedValue({ id: 'train_new' });
  createTemplate.mockReset().mockResolvedValue({ id: 'tmpl_new' });
  alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});

  useTrainingsStore.setState({
    trainings: [],
    isLoading: false,
    error: null,
    config: { ...FORM, dataset_ids: [...FORM.dataset_ids] },
    fetchTrainings: vi.fn().mockResolvedValue(undefined),
    fetchTraining: vi.fn().mockResolvedValue(undefined),
    createTraining,
    deleteTraining: vi.fn(),
  });
  useTrainingTemplatesStore.setState({
    templates: [TEMPLATE_16K, TEMPLATE_OLDER, TEMPLATE_UNPAIRED],
    loading: false,
    fetchTemplates: vi.fn(),
    createTemplate,
  } as never);

  (useModelsStore as never as Mocked).mockReturnValue({
    models: [{
      id: MODEL, name: 'LFM2.5-1.2B-Instruct', status: 'ready',
      architecture_config: { num_hidden_layers: 16, hidden_size: 2048 },
    }],
    fetchModels: vi.fn(),
  });
  (useDatasetsStore as never as Mocked).mockReturnValue({
    datasets: [
      { id: DS_WEB, name: 'OpenWebText-2M', status: 'ready' },
      { id: DS_CODE, name: 'github-code-clean', status: 'ready' },
    ],
    fetchDatasets: vi.fn(),
  });
  (useTrainingWebSocket as never as Mocked).mockReturnValue({});
  (useDeletionProgressWebSocket as never as Mocked).mockReturnValue(undefined);
  (useWebSocketContext as never as Mocked).mockReturnValue({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: true,
  });
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ extractions: [extraction(EXT_WEB, DS_WEB), extraction(EXT_CODE, DS_CODE)] }),
  }) as never;
});

afterEach(() => {
  alertSpy.mockRestore();
});

async function loadTemplate(id: string) {
  fireEvent.change(await screen.findByLabelText(/Load Template/), { target: { value: id } });
}

async function startTraining() {
  const button = screen.getByRole('button', { name: /Start Training/i });
  await waitFor(() => expect(button).not.toBeDisabled());
  fireEvent.click(button);
  await waitFor(() => expect(createTraining).toHaveBeenCalledTimes(1));
  return wire(createTraining.mock.calls[0][0]);
}

async function saveTemplate() {
  fireEvent.click(screen.getByRole('button', { name: /Save as Template/i }));
  fireEvent.change(screen.getByLabelText(/Template Name/), { target: { value: 'sixteen k' } });
  fireEvent.change(screen.getByLabelText(/Description/), { target: { value: 'from the panel' } });
  fireEvent.click(screen.getByRole('button', { name: /^Save Template$/ }));
}

/** The per-dataset extraction picker, found through the row that names the dataset. */
function extractionPicker(datasetName: string): HTMLSelectElement {
  const cell = screen
    .getAllByTitle(datasetName)
    .find((element) => element.parentElement?.querySelector('select'));
  if (!cell) throw new Error(`no extraction picker for ${datasetName}`);
  return cell.parentElement!.querySelector('select')!;
}

describe('R2D-6: a loaded template reaches Start Training', () => {
  it("sends the template's seed, evaluation fields, hook types and weights paired by dataset id", async () => {
    render(<TrainingPanel />);
    await loadTemplate('tmpl_16k');

    const request = await startTraining();

    expect(request).toStrictEqual({
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      hyperparameters: {
        ...HP_16K_WITHOUT_WEIGHTS,
        // The panel's datasets are [web, code]; the template stored [code 0.25, web 0.75].
        dataset_weights: [0.75, 0.25],
      },
      gpu: 'auto',
    });
    expect(createTraining).toHaveBeenCalledTimes(1);
  });

  it("gives a template without those fields the backend's defaults, not the previous template's values", async () => {
    render(<TrainingPanel />);
    await loadTemplate('tmpl_16k');
    await loadTemplate('tmpl_older');

    const request = await startTraining();

    expect(request).toStrictEqual({
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      hyperparameters: HP_OLDER,
      gpu: 'auto',
    });
    expect(createTraining).toHaveBeenCalledTimes(1);
  });

  it('shows weights it cannot pair with datasets, and does not apply them by position', async () => {
    render(<TrainingPanel />);
    await loadTemplate('tmpl_unpaired');

    const notice = await screen.findByTestId('template-weights-notice');
    expect(notice).toHaveAttribute('role', 'alert');
    expect(notice).toHaveTextContent(
      'Template "weights without datasets" carries dataset_weights [0.9, 0.1] for 0 dataset(s), so they ' +
        'cannot be paired with datasets and were not applied.'
    );

    const request = await startTraining();
    expect(request).toStrictEqual({
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      hyperparameters: HP_OLDER,
      gpu: 'auto',
    });
    expect(createTraining).toHaveBeenCalledTimes(1);

    // A later template that pairs, or has none, clears the notice.
    await loadTemplate('tmpl_older');
    await waitFor(() => expect(screen.queryByTestId('template-weights-notice')).not.toBeInTheDocument());
  });
});

describe('R2F-6: Save as Template sends what Start Training sends', () => {
  it('saves the same hyperparameters, on the fly', async () => {
    render(<TrainingPanel />);
    await loadTemplate('tmpl_16k');

    await saveTemplate();
    await waitFor(() => expect(createTemplate).toHaveBeenCalledTimes(1));
    const saved = wire(createTemplate.mock.calls[0][0]);
    expect(saved).toStrictEqual({
      name: 'sixteen k',
      description: 'from the panel',
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      encoder_type: 'jumprelu',
      hyperparameters: { ...HP_16K_WITHOUT_WEIGHTS, dataset_weights: [0.75, 0.25] },
      is_favorite: false,
    });

    const started = await startTraining();
    // On the fly both are positional over dataset_ids: one builder, one answer.
    expect(saved.hyperparameters).toStrictEqual(started.hyperparameters);
    expect(createTemplate).toHaveBeenCalledTimes(1);
    expect(createTraining).toHaveBeenCalledTimes(1);
  });

  it('on cached activations, Start pairs weights with extraction_ids as picked and the template stores dataset order', async () => {
    useTrainingsStore.setState({ config: { ...FORM, dataset_ids: [DS_WEB, DS_CODE], extraction_ids: [] } });
    render(<TrainingPanel />);

    // Picked code first, so extraction_ids run opposite to dataset_ids.
    await waitFor(() => expect(extractionPicker('github-code-clean').options).toHaveLength(2));
    fireEvent.change(extractionPicker('github-code-clean'), { target: { value: EXT_CODE } });
    fireEvent.change(extractionPicker('OpenWebText-2M'), { target: { value: EXT_WEB } });
    await loadTemplate('tmpl_16k');

    await saveTemplate();
    await waitFor(() => expect(createTemplate).toHaveBeenCalledTimes(1));
    expect(wire(createTemplate.mock.calls[0][0])).toStrictEqual({
      name: 'sixteen k',
      description: 'from the panel',
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      encoder_type: 'jumprelu',
      // A template's weights follow its dataset_ids: web 0.75, code 0.25.
      hyperparameters: { ...HP_16K_WITHOUT_WEIGHTS, dataset_weights: [0.75, 0.25] },
      is_favorite: false,
    });

    expect(await startTraining()).toStrictEqual({
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      extraction_ids: [EXT_CODE, EXT_WEB],
      // A training's weights follow extraction_ids: code 0.25, web 0.75.
      hyperparameters: { ...HP_16K_WITHOUT_WEIGHTS, dataset_weights: [0.25, 0.75] },
      gpu: 'auto',
    });
    expect(createTemplate).toHaveBeenCalledTimes(1);
    expect(createTraining).toHaveBeenCalledTimes(1);
  });
});

describe('the seed and evaluation controls', () => {
  const openAdvanced = () => fireEvent.click(screen.getByText(/Advanced Configuration/i));

  it('reach the request', async () => {
    render(<TrainingPanel />);
    openAdvanced();

    fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '42' } });
    fireEvent.change(screen.getByLabelText('Evaluation Token Budget'), { target: { value: '0' } });
    fireEvent.click(screen.getByLabelText('Evaluate the spliced SAE after training'));
    fireEvent.change(screen.getByLabelText('Held-out Fraction'), { target: { value: '0.2' } });
    fireEvent.change(await screen.findByLabelText('Held-out Tokens per Log Step'), { target: { value: '50000' } });
    fireEvent.change(screen.getByLabelText('Held-out Tokens per Chunk'), { target: { value: '1024' } });

    expect(await startTraining()).toStrictEqual({
      model_id: MODEL,
      dataset_ids: [DS_WEB, DS_CODE],
      hyperparameters: {
        hidden_dim: 2048,
        latent_dim: 16384,
        architecture_type: 'standard_saelens',
        training_layers: [0],
        hook_types: ['residual'],
        normalize_activations: 'constant_norm_rescale',
        learning_rate: 4e-4,
        batch_size: 2048,
        total_steps: 50000,
        warmup_steps: 2000,
        sparsity_warmup_steps: 5000,
        weight_decay: 0,
        grad_clip_norm: 1,
        checkpoint_interval: 5000,
        log_interval: 100,
        lr_decay_steps: 0,
        resample_dead_neurons: true,
        resample_interval: 5000,
        dead_neuron_threshold: 1000,
        l1_alpha: 5e-4,
        target_l0: 0.05,
        normalize_decoder: true,
        holdout_fraction: 0.2,
        evaluate_ce_delta: false,
        evaluation_token_budget: 0,
        holdout_eval_tokens: 50000,
        holdout_eval_chunk_tokens: 1024,
        seed: 42,
      },
      gpu: 'auto',
    });
    expect(createTraining).toHaveBeenCalledTimes(1);
  });

  it("show a template's weights in the mixture inputs, on cached activations, as the request sends them", async () => {
    // Survivor W1. The inputs read the raw map, which holds a template's weights
    // keyed by dataset while the inputs are keyed by extraction, so each showed 1
    // while the request sent 0.25 and 0.75. On the fly the ids coincide, so only
    // the cached path can tell the two apart.
    useTrainingsStore.setState({ config: { ...FORM, dataset_ids: [DS_WEB, DS_CODE], extraction_ids: [] } });
    render(<TrainingPanel />);
    await waitFor(() => expect(extractionPicker('github-code-clean').options).toHaveLength(2));
    fireEvent.change(extractionPicker('github-code-clean'), { target: { value: EXT_CODE } });
    fireEvent.change(extractionPicker('OpenWebText-2M'), { target: { value: EXT_WEB } });
    await loadTemplate('tmpl_16k');
    openAdvanced();

    const code = await screen.findByLabelText('github-code-clean');
    const web = screen.getByLabelText('OpenWebText-2M');
    expect(code.id).toBe(`mixture-weight-${EXT_CODE}`);
    expect(code).toHaveValue(0.25);
    expect(web.id).toBe(`mixture-weight-${EXT_WEB}`);
    expect(web).toHaveValue(0.75);

    const request = await startTraining();
    expect(request.hyperparameters.dataset_weights).toStrictEqual([0.25, 0.75]);
    expect(createTraining).toHaveBeenCalledTimes(1);
  });

  it('Save refuses a schedule Start would refuse, before sending', async () => {
    // Survivor S2: Save checked the evaluation fields and not the schedule.
    useTrainingsStore.setState({ config: { ...FORM, dataset_ids: [DS_WEB, DS_CODE], lr_decay_steps: 49000 } });
    render(<TrainingPanel />);

    await saveTemplate();

    expect(
      await screen.findByText('Warmup steps (2000) plus LR decay steps (49000) exceed total steps (50000)')
    ).toBeInTheDocument();
    expect(createTemplate).toHaveBeenCalledTimes(0);
  });

  it("Save shows the server's reason when it refuses the template, not axios's status line", async () => {
    // Survivor S3.
    createTemplate.mockReset().mockRejectedValue(
      Object.assign(new Error('Request failed with status code 422'), {
        response: {
          status: 422,
          data: {
            detail: [
              { loc: ['body', 'hyperparameters', 'seed'], msg: 'Input should be greater than or equal to 0', type: 'greater_than_equal' },
            ],
          },
        },
      })
    );
    render(<TrainingPanel />);

    await saveTemplate();

    expect(
      await screen.findByText('hyperparameters.seed: Input should be greater than or equal to 0')
    ).toBeInTheDocument();
    expect(screen.queryByText('Request failed with status code 422')).not.toBeInTheDocument();
    expect(createTemplate).toHaveBeenCalledTimes(1);
  });

  it('refuse a value the backend would reject, from Start and from Save, before sending', async () => {
    render(<TrainingPanel />);
    openAdvanced();
    fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '1.5' } });

    const start = screen.getByRole('button', { name: /Start Training/i });
    await waitFor(() => expect(start).not.toBeDisabled());
    fireEvent.click(start);
    await waitFor(() =>
      expect(alertSpy).toHaveBeenCalledWith('Seed must be a whole number from 0 to 9,007,199,254,740,991')
    );
    expect(alertSpy).toHaveBeenCalledTimes(1);

    await saveTemplate();
    expect(await screen.findByText('Seed must be a whole number from 0 to 9,007,199,254,740,991')).toBeInTheDocument();

    expect(createTraining).toHaveBeenCalledTimes(0);
    expect(createTemplate).toHaveBeenCalledTimes(0);
  });
});
