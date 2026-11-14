/**
 * A layer count is a measurement. Never invent one.
 *
 * Reported 2026-08-25: gemma-4-12B-it, a 48-layer model, offered layers 0-11
 * on the extraction page. `ActivationExtractionConfig` fell back to 12 when
 * the stored config recorded no depth, so the picker looked authoritative
 * while capping the model at L11 -- layers 12-47 were unreachable and nothing
 * said so. `ModelArchitectureViewer` was worse: its fallbacks (12 layers, 768
 * wide, 12 heads, 50257 vocab) are GPT-2's, so any model with an incomplete
 * config was drawn as GPT-2 under its own name.
 *
 * A missing picker is a visible bug someone reports in a day. A fabricated one
 * is a wrong answer that looks right, which is the failure mode this product
 * can least afford. Same rule the J-Lens bands already follow: no default
 * constant anywhere, by construction.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { ModelArchitectureViewer } from './ModelArchitectureViewer';
import { ActivationExtractionConfig } from './ActivationExtractionConfig';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useExtractionTemplatesStore } from '../../stores/extractionTemplatesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';

vi.mock('../../api/models', async (orig) => ({
  ...(await orig<typeof import('../../api/models')>()),
  getModelExtractions: vi.fn().mockResolvedValue({ extractions: [] }),
  estimateExtractionResources: vi.fn().mockResolvedValue({}),
}));

vi.mock('../../hooks/useModelProgress', () => ({
  useModelExtractionProgress: () => ({ progress: null }),
}));

function seedStores() {
  useDatasetsStore.setState({
    datasets: [], tokenizations: {},
    fetchDatasets: vi.fn().mockResolvedValue(undefined),
    fetchTokenizations: vi.fn().mockResolvedValue(undefined),
  } as never);
  useExtractionTemplatesStore.setState({
    templates: [], favorites: [],
    fetchTemplates: vi.fn().mockResolvedValue(undefined),
    fetchFavorites: vi.fn().mockResolvedValue(undefined),
    createTemplate: vi.fn().mockResolvedValue(undefined),
  } as never);
  useModelsStore.setState({ models: [] } as never);
  useSystemMonitorStore.setState({
    gpuList: [], fetchGPUList: vi.fn().mockResolvedValue(undefined),
  } as never);
}

function model(architecture_config: Record<string, unknown> | null) {
  return {
    id: 'm_b55c6926',
    name: 'gemma-4-12B-it',
    status: 'ready',
    params_count: 12_000_000_000,
    quantization: 'q4',
    architecture: 'gemma4_unified',
    architecture_config,
  } as never;
}

const COMPLETE = {
  num_hidden_layers: 48,
  hidden_size: 3840,
  num_attention_heads: 16,
  intermediate_size: 15360,
  vocab_size: 262144,
};

describe('ModelArchitectureViewer', () => {
  it('draws the real depth when the config records it', async () => {
    render(<ModelArchitectureViewer model={model(COMPLETE)} onClose={vi.fn()} />);
    expect(await screen.findByText(/TransformerBlock_47/)).toBeInTheDocument();
  });

  it('refuses to draw anything when the depth is unknown', async () => {
    render(
      <ModelArchitectureViewer
        model={model({ model_type: 'gemma4_unified', initializer_range: 0.02 })}
        onClose={vi.fn()}
      />
    );

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /does not record the dimensions/i
    );
    expect(screen.queryByText(/TransformerBlock_11/)).not.toBeInTheDocument();
  });

  it('never falls back to GPT-2 geometry', async () => {
    render(
      <ModelArchitectureViewer
        model={model({ num_hidden_layers: 48, hidden_size: 3840 })}
        onClose={vi.fn()}
      />
    );

    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument());
    // 50257 is GPT-2's vocabulary; it must not appear for a model that
    // declares none of its own.
    expect(screen.queryByText(/50257/)).not.toBeInTheDocument();
    expect(screen.queryByText(/768/)).not.toBeInTheDocument();
  });
});


describe('ActivationExtractionConfig layer picker', () => {
  beforeEach(() => seedStores());

  it('offers every layer the model actually has', async () => {
    render(
      <ActivationExtractionConfig
        model={model(COMPLETE)}
        onClose={vi.fn()}
        onExtract={vi.fn()}
      />
    );

    expect(await screen.findByRole('button', { name: 'L47' })).toBeInTheDocument();
  });

  it('offers no layers at all when the depth is unknown', async () => {
    render(
      <ActivationExtractionConfig
        model={model({ model_type: 'gemma4_unified', initializer_range: 0.02 })}
        onClose={vi.fn()}
        onExtract={vi.fn()}
      />
    );

    // The bug: a confident 0-11 picker for a model of unknown depth.
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'L11' })).not.toBeInTheDocument()
    );
    expect(screen.queryByRole('button', { name: 'L0' })).not.toBeInTheDocument();
    expect(
      await screen.findByText(/records no layer count/i)
    ).toBeInTheDocument();
  });
});
