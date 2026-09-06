/**
 * J-Lens readout API client (Feature 023, doc chain 022 substrate).
 *
 * One call. The response is the upstream wire format verbatim — this module
 * deliberately performs NO reshaping, because an adaptation layer here is
 * exactly what PADR IDL-45 forbids: it would let the panel drift into a
 * miStudio-only shape while still appearing to conform.
 */

import { fetchAPI } from './client';
import type {
  JLensAnnotateRequest,
  JLensAnnotation,
  JLensAcquirePreview,
  JLensAcquirePreviewRequest,
  JLensAcquireRequest,
  JLensInterventionRequest,
  JLensPublishRequest,
  JLensTokenCheck,
  JLensWatchlistRequest,
  JLensWatchlistResponse,
  JLensArtifactSummary,
  JLensFitAccepted,
  JLensFitRequest,
  JLensValidationResponse,
  ReadoutAccepted,
  ReadoutRequest,
  ReadoutResult,
} from '../types/jlens';

export const jlensApi = {
  /**
   * Request a position x layer readout.
   *
   * `types` defaults server-side to LOGIT_LENS, which needs no artifact
   * (BR-005). Requesting JACOBIAN_LENS without `artifact_id` is refused by the
   * server rather than silently served as logit data under a Jacobian label
   * (BR-019).
   */
  readout: (request: ReadoutRequest) =>
    fetchAPI<ReadoutAccepted>('/jlens/readout', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Poll a queued readout. Null `readout` until `status` is SUCCESS. */
  readoutResult: (taskId: string) =>
    fetchAPI<ReadoutResult>(`/jlens/readout/${encodeURIComponent(taskId)}`),

  /** Artifacts present in the mounted registry. Presence, not validity. */
  listArtifacts: () => fetchAPI<JLensArtifactSummary[]>('/jlens/artifacts'),

  /**
   * Run the validation suite. The model's dimensions are required because the
   * envelope bound must come from the model the artifact was fitted for — a
   * bound derived from the wrong model passes while missing a real
   * materialisation.
   */
  validateArtifact: (
    slug: string,
    dims: { d_model: number; n_layers: number; n_vocab: number }
  ) =>
    fetchAPI<JLensValidationResponse>(
      `/jlens/artifacts/${encodeURIComponent(slug)}/validate` +
        `?d_model=${dims.d_model}&n_layers=${dims.n_layers}&n_vocab=${dims.n_vocab}`,
      { method: 'POST' }
    ),

  /**
   * Is a hand-typed string a single token in THIS model's vocabulary?
   *
   * A direction is `W_U[id]`, so any single token has one — including tokens
   * the readout never surfaced, which are the interesting swap targets. But
   * whether a string is ONE token belongs to the model's vocabulary, and the
   * worker's refusal of a multi-token direction arrives only after a 202 and a
   * slot on a single-GPU queue. Weights are not loaded.
   */
  checkTokens: (modelId: string, tokens: string[]) =>
    fetchAPI<JLensTokenCheck[]>('/jlens/token-check', {
      method: 'POST',
      body: JSON.stringify({ model_id: modelId, tokens }),
    }),

  /**
   * What in this repo could be a lens, with sizes. READ-ONLY.
   *
   * Spending a request instead of a download: a mistyped path otherwise costs a
   * multi-gigabyte fetch and a slot on the single-GPU queue before anything
   * notices.
   */
  previewRepo: (request: JLensAcquirePreviewRequest) =>
    fetchAPI<JLensAcquirePreview>('/jlens/acquire/preview', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Download a published lens and validate it for a local model. */
  acquire: (request: JLensAcquireRequest) =>
    fetchAPI<JLensFitAccepted>('/jlens/acquire', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Upload this model's validated lens to HuggingFace. */
  publish: (request: JLensPublishRequest) =>
    fetchAPI<JLensFitAccepted>('/jlens/publish', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Queue an intervention AND its matched control. Poll the task id. */
  intervene: (request: JLensInterventionRequest) =>
    fetchAPI<JLensFitAccepted>('/jlens/interventions', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Describe an SAE feature in J-space. Rung 0 — present is not used. */
  annotate: (request: JLensAnnotateRequest) =>
    fetchAPI<JLensAnnotation>('/jlens/annotate', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Validate a watchlist definition. REFUSES one missing its scoring rule. */
  createWatchlist: (request: JLensWatchlistRequest) =>
    fetchAPI<JLensWatchlistResponse>('/jlens/watchlists', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  /** Queue a fit. GPU-bound and long-running; poll the task id. */
  fit: (request: JLensFitRequest) =>
    fetchAPI<JLensFitAccepted>('/jlens/fit', {
      method: 'POST',
      body: JSON.stringify(request),
    }),
};
