/**
 * J-space lens wire format — the UI mirror of backend/src/schemas/jlens.py.
 *
 * THE WIRE FORMAT IS NOT OURS TO DESIGN (BR-029, PADR IDL-45). These shapes
 * mirror Neuronpedia's lens stream exactly, so the readout panel renders a
 * miStudio stream and a Neuronpedia stream with no adaptation layer. Adding a
 * miStudio-shaped field here would silently break that property — the panel
 * would keep working against our server and fail against theirs.
 *
 *   meta  = { model, types, layers_by_type, top_n, prompt_len }
 *   token = { position, token, id, is_generated, results: slice[] }
 *   slice = { type, top_tokens[layer][k], top_probs[layer][k] }
 *
 * `top_tokens` entries are DECODED STRINGS, not ids — the backend enforces
 * this because ids type-check against a looser schema and render as
 * unreadable cells.
 */

/**
 * Lens types the stream can CARRY.
 *
 * DIFF is deliberately absent: it is a client-side rendering mode over two
 * transported slices, never a transported type. Adding it here would invite a
 * request for a type the server cannot emit.
 */
export type LensType = 'JACOBIAN_LENS' | 'LOGIT_LENS';

/** What the mode tabs offer. DIFF exists only at this layer. */
export type LensMode = LensType | 'DIFF';

/**
 * Which computations are meaningful at one layer.
 *
 * Layer kind is PER-LAYER state, not a model property: a hybrid model
 * interleaves convolutional and attention layers, so "freeze Q/K" is undefined
 * on some of them. Inapplicable is `null` (absent), never `false` — a `false`
 * gets averaged by a consumer and silently understates (BR-032).
 */
export interface LayerApplicability {
  layer: number;
  has_attention: boolean;
  frozen_qk_applicable: boolean | null;
  broadcast_metrics_applicable: boolean | null;
}

/**
 * One lens type's readout for one token position, across layers.
 *
 * `top_tokens[layerIdx][k]` is indexed by POSITION IN `meta.layers_by_type[type]`,
 * not by the model's absolute layer number. Indexing it with an absolute layer
 * number reads the wrong row wherever the two differ, and produces a plausible
 * grid rather than an error.
 */
export interface LensTypeSlice {
  type: LensType;
  top_tokens: string[][];
  top_probs: number[][];
}

export interface LensMetaMessage {
  kind: 'meta';
  model: string;
  types: LensType[];
  /** Absolute layer indices per lens type. DRIVES THE LAYER AXIS — never assume
   *  a count or a spacing; models here range from 16 to 26+ layers. */
  layers_by_type: Record<string, number[]>;
  top_n: number;
  prompt_len: number;
  layer_applicability?: LayerApplicability[] | null;
}

export interface LensTokenMessage {
  kind: 'token';
  position: number;
  token: string;
  id: number;
  is_generated: boolean;
  results: LensTypeSlice[];
}

export interface ReadoutRequest {
  model_id: string;
  prompt: string;
  types?: LensType[];
  layers?: number[] | null;
  top_n?: number;
  /** Required by the server when `types` includes JACOBIAN_LENS. The logit lens
   *  needs no artifact (BR-005). */
  artifact_id?: string | null;
}

/** Non-streaming envelope. CONTAINS a meta message rather than being one. */
export interface ReadoutResponse {
  meta: LensMetaMessage;
  tokens: LensTokenMessage[];
}

/**
 * Sensory / workspace / motor boundaries for one model.
 *
 * THERE IS DELIBERATELY NO DEFAULT VALUE AND NO CONSTANT ANYWHERE IN THIS
 * FEATURE. The reference implementation's L40/L90 are the source paper's
 * Sonnet-4.5 figures; BR-002 forbids porting them to another model and requires
 * the product make porting impossible by construction. Bands render only from a
 * report computed for the selected model, and are absent otherwise — a
 * fallback object would be those figures under another name.
 */
export interface BandReport {
  model: string;
  workspace_start: number;
  motor_start: number;
  /** How the boundaries were derived, surfaced next to the shading. */
  derivation: string;
}

/** Provenance behind the current readout (BR-007). */
export interface ReadoutProvenance {
  /** Absent for the logit lens, which involves no artifact at all. */
  artifact_id: string | null;
  target_layer?: string;
  attention_gradients?: string;
  target_position_scope?: string;
  aggregation?: string;
  corpus?: string;
  n_prompts?: number;
  seq_len?: number;
  dtype?: string;
}

/** One artifact as it exists on disk. PRESENCE, not validity. */
export interface JLensArtifactSummary {
  slug: string;
  directory: string;
  lens_file: string;
  size_bytes: number;
  has_config: boolean;
  /**
   * Layers this artifact covers.
   *
   * EMPTY MEANS UNKNOWN, not "none". An artifact whose config could not be
   * read still holds whatever it holds, and drawing that as zero coverage
   * would assert something the listing never checked.
   */
  layers?: number[];
  /**
   * Layers where J is the identity — the lens there IS the logit lens.
   *
   * The last decoder layer has no blocks after it, so its sub-network is the
   * identity by construction. A Diff at such a layer is empty because the two
   * lenses are the same lens, not because they happen to agree, and an empty
   * top row read without that context looks like a finding.
   */
  degenerate_layers?: number[];
  /**
   * Which block the Jacobian was taken TO.
   *
   * With a `penultimate` target a COMPLETE fit covers 0..N-2, so comparing
   * coverage against the model's layer count renders a full artifact as
   * "25/26" and colours it amber — reporting a recipe choice as a defect.
   */
  target_layer?: string | null;
}

export interface JLensCheckOutcome {
  check: string;
  /** 'pass' | 'fail' | 'not_run'. NOT_RUN is not a pass — see ValidationResponse. */
  status: string;
  detail: string;
  evidence?: Record<string, unknown>;
}

/**
 * The validation verdict, every class reported individually.
 *
 * `passed` is fail-closed and covers all six classes — the gate for handing an
 * artifact to an EXTERNAL consumer. Two of those classes need a live consumer
 * to run at all, so a validation performed from the workbench reports them
 * NOT_RUN and `passed` is false. That is the honest answer, not a defect, and
 * the UI says so rather than showing a red cross.
 */
export interface JLensValidationResponse {
  slug: string;
  passed: boolean;
  summary: string;
  results: JLensCheckOutcome[];
}

/**
 * Fixture for the SEMANTIC validation class.
 *
 * `expected_intermediate` must NOT appear in `prompt`: a token already present
 * is recovered by an artifact that encodes nothing at all, so a fixture that
 * breaks this passes against a broken lens. The server rejects it.
 */
export interface JLensSemanticProbe {
  prompt: string;
  expected_intermediate: string;
  /**
   * Omit to SCAN every fitted layer — the default.
   *
   * Naming a layer asserts WHERE an unspoken intermediate lives, which is a
   * property of the model and not something this project may assume (BR-002).
   * Two such defaults were shipped and both were wrong; the second discarded a
   * converged artifact whose mid-stack readout was the correct concept field.
   */
  layer?: number | null;
  /**
   * An unrelated prompt for which `expected_intermediate` would be absurd.
   *
   * A scan has more chances to hit than a single layer, so it needs a
   * false-positive control: if the token surfaces here too, the check FAILS
   * however well the real prompt scored.
   */
  control_prompt?: string | null;
  top_k?: number;
}

export interface JLensFitRequest {
  model_id: string;
  prompts: string[];
  layers?: number[] | null;
  freeze_qk?: boolean;
  corpus_name?: string;
  /** Without this NOTHING IS PUBLISHED — the suite fails closed on an unrun check. */
  semantic_probe?: JLensSemanticProbe | null;
  /**
   * Publish even when this fit is WEAKER evidence than the artifact it
   * replaces — fewer prompts, or not converged where the incumbent converged.
   *
   * Publishing is otherwise last-writer-wins, and "last" means finished last,
   * not best: a 400-prompt fit that never converged once published over a
   * 1097-prompt fit that did, because the weaker job had been queued hours
   * earlier and only got a worker once the queue drained.
   */
  allow_quality_regression?: boolean;
}

export interface JLensFitAccepted {
  task_id: string;
  model_id: string;
  queue: string;
}

/** A queued readout. The result arrives via the task, not this response. */
export interface ReadoutAccepted {
  task_id: string;
  model_id: string;
  status: string;
}

/**
 * A readout task's state.
 *
 * `readout` is null until `status` is SUCCESS. Treating a pending task as an
 * empty readout reproduces exactly the confusion this feature exists to
 * prevent, so `status` is always authoritative.
 */
export interface ReadoutResult {
  task_id: string;
  status: string;
  stage?: string | null;
  readout?: ReadoutResponse | null;
  error?: string | null;
}


/**
 * An intervention and the control it is meaningless without (BR-018).
 *
 * There is no way to request one WITHOUT a control: `k` and `control_seed`
 * define it and both are always sent. An intervention that moves the output
 * says nothing until compared with what a random direction of the same size
 * does.
 */
export interface JLensInterventionRequest {
  model_id: string;
  prompt: string;
  primitive: string;
  layers: number[];
  /** A SINGLE token; the server resolves its unembedding row. */
  direction_token?: string | null;
  /**
   * The token whose RANK is scored in the model's own output.
   *
   * Defaults server-side to `direction_token`. REQUIRED and must DIFFER for
   * `coordinate_swap`: a swap exchanges two coordinates, and one token would be
   * an additive steer wearing a swap's name — which is what the backend used to
   * run, labelled as a swap, before the hook grew a branch for it.
   */
  target_token?: string | null;
  /**
   * MORE PROMPTS, one TRIAL each.
   *
   * The result is a fraction of trials with a Wilson 95% interval; a single
   * prompt yields an interval spanning nearly the whole range, which is the
   * honest rendering of one observation and rarely a finding.
   */
  prompts?: string[] | null;
  strength?: number;
  k?: number;
  control_seed?: number;
  artifact_id?: string | null;
}

export interface JLensAnnotateRequest {
  /** All this needs. The rest is resolved server-side from the feature row. */
  feature_id: string;
  model_id?: string;
  sae_id?: string;
  layer?: number;
  label_tokens?: string[];
  top_k?: number;
}

export interface JLensAnnotation {
  feature_id: string;
  layer: number;
  lens_kurtosis: number | null;
  /** UNKNOWN without a band report for this model — a real answer, not a gap. */
  workspace_class: string;
  top_tokens: string[];
  disagreement_score?: number | null;
  has_disagreement?: boolean;
  evidence_rung?: number;
}

export interface JLensWatchlistRequest {
  name: string;
  artifact_ref: string;
  /** REQUIRED. A threshold applied to a differently computed score is a
   *  different detector, and the consumer cannot notice. */
  scoring_definition: string;
  concepts: Array<{ token: string; threshold: number }>;
  control_set?: string[];
}

export interface JLensWatchlistResponse {
  name: string;
  artifact_ref: string;
  scoring_definition: string;
  concept_count: number;
}

/** One hand-typed token, resolved against the model's own vocabulary. */
export interface JLensTokenCheck {
  token: string;
  /** The ids it encodes to — shown, because "[4874, 883]" is what makes
   *  "this is two tokens" concrete. */
  ids: number[];
  n_tokens: number;
  /** Usable as a lens direction, which is defined for exactly one id. */
  usable: boolean;
  detail: string;
}

/** One downloadable candidate in a HuggingFace repo. */
export interface JLensAcquireCandidate {
  path: string;
  size_bytes: number | null;
  /**
   * A `config.yaml` sits beside this file.
   *
   * PRESENCE, NOT A VERDICT. `check_weight_identity` still returns UNVERIFIED
   * when that config names no model, and REFUSES outright when it names a
   * different one — so this says identity CAN be looked for, not that it will
   * check out. The badge said "identity checkable" until a review pointed out
   * the same overclaim.
   */
  has_config: boolean;
  has_convergence: boolean;
  /**
   * Null when NO VERDICT WAS COMPUTED — which has three causes, not one.
   *
   * The endpoint computes one only `if dims and c.size_bytes`: no model named,
   * OR the Hub reported no size, OR the model row lacks the dimensions to
   * derive a bound. Treating null as "fine" permits exactly the download the
   * preview exists to prevent.
   */
  fits_envelope: boolean | null;
  envelope_detail: string | null;
}

export interface JLensAcquirePreview {
  repo_id: string;
  /** The RESOLVED commit. `main` moves, so an acquisition pinned to it is not a
   *  reproducible statement. */
  revision: string;
  candidates: JLensAcquireCandidate[];
}

export interface JLensAcquirePreviewRequest {
  repo_id: string;
  revision?: string;
  model_id?: string;
  access_token?: string;
}

export interface JLensAcquireRequest {
  model_id: string;
  repo_id: string;
  path_in_repo: string;
  revision?: string;
  access_token?: string;
  allow_coverage_loss?: boolean;
  allow_quality_regression?: boolean;
  replace_staged?: boolean;
}

export interface JLensPublishRequest {
  model_id: string;
  target_repo: string;
  access_token?: string;
  dataset?: string;
  create_repo?: boolean;
  private?: boolean;
}

