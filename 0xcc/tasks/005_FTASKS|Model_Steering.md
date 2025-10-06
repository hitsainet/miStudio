# Task List: Model Steering

**Feature:** Feature-Based Model Behavior Modification
**PRD Reference:** 005_FPRD|Model_Steering.md
**TDD Reference:** 005_FTDD|Model_Steering.md
**TID Reference:** 005_FTID|Model_Steering.md
**Created:** 2025-10-06
**Status:** Ready for Implementation

---

## Relevant Files

### Backend Files
- `backend/src/services/steering_service.py` - Core steering service with forward hooks, coefficient application, metrics calculation
- `backend/src/api/routes/steering.py` - FastAPI routes for steering generation (`POST /api/steering/generate`)
- `backend/src/schemas/steering_schemas.py` - Pydantic schemas for steering requests/responses
- `backend/tests/unit/test_steering_service.py` - Unit tests for steering algorithm, hook cleanup, metrics calculations
- `backend/tests/integration/test_steering_api.py` - Integration tests for end-to-end steering workflow

### Frontend Files
- `frontend/src/components/panels/SteeringPanel.tsx` - Main steering panel component (lines 3512-3951 in Mock UI)
- `frontend/src/stores/steeringStore.ts` - Zustand store for steering state management
- `frontend/src/types/steering.ts` - TypeScript types for steering feature
- `frontend/src/components/panels/SteeringPanel.test.tsx` - Component tests for SteeringPanel

### Dependencies
- `sentence-transformers` - For semantic similarity calculation (all-MiniLM-L6-v2 model)
- PyTorch forward hooks - For activation intervention
- Existing ModelRegistry (from 002_FTDD) - For model loading
- Existing feature search endpoint (from 004_FTDD) - For feature selection

---

## Tasks

- [ ] 1.0 Backend Infrastructure Setup
  - [ ] 1.1 Create `backend/src/services/steering_service.py` file
  - [ ] 1.2 Create `backend/src/api/routes/steering.py` file
  - [ ] 1.3 Create `backend/src/schemas/steering_schemas.py` file
  - [ ] 1.4 Install `sentence-transformers` package (add to requirements.txt)
  - [ ] 1.5 Download sentence model at container build: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` (add to Dockerfile)
  - [ ] 1.6 Create test files: `backend/tests/unit/test_steering_service.py`, `backend/tests/integration/test_steering_api.py`
  - [ ] 1.7 Import steering router in `backend/src/api/main.py`: `app.include_router(steering_router, prefix="/api/steering", tags=["steering"])`

- [ ] 2.0 Pydantic Schemas
  - [ ] 2.1 Create `SteeringFeatureInput` schema with fields: `feature_id` (int), `coefficient` (float, range -5 to 5)
  - [ ] 2.2 Create `SteeringGenerateRequest` schema with fields: `model_id` (str), `prompt` (str, 1-1024 tokens), `features` (List[SteeringFeatureInput], max 10), `intervention_layer` (int, >= 0), `temperature` (float, 0-2), `max_tokens` (int, 1-2048), `seed` (Optional[int])
  - [ ] 2.3 Create `ComparisonMetrics` schema with fields: `kl_divergence` (float), `perplexity_delta` (float), `semantic_similarity` (float), `word_overlap` (float)
  - [ ] 2.4 Create `SteeringGenerateResponse` schema with fields: `unsteered_output` (str), `steered_output` (str), `metrics` (ComparisonMetrics), `generation_time_ms` (int)
  - [ ] 2.5 Add validators: coefficient range [-5.0, 5.0], temperature range [0.0, 2.0], max_tokens range [1, 2048], features array length [1, 10]
  - [ ] 2.6 Add example request/response in schema docstrings (from TDD lines 260-303)

- [ ] 3.0 SteeringService Class - Initialization
  - [ ] 3.1 Create `SteeringService` class in `steering_service.py`
  - [ ] 3.2 Initialize `self.hook_handles = []` to track registered hooks
  - [ ] 3.3 Initialize `self.model_registry = ModelRegistry()` (reuse from 002_FTDD)
  - [ ] 3.4 Initialize `self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')` at service init (cached)
  - [ ] 3.5 Add `_load_sae_for_features()` method: load SAE encoder/decoder from feature's training_id
  - [ ] 3.6 Add error handling for missing SAE checkpoints

- [ ] 4.0 Forward Hook Implementation
  - [ ] 4.1 Implement `_register_steering_hook()` method: accepts model, sae_encoder, sae_decoder, intervention_layer, steering_config
  - [ ] 4.2 Define `steering_hook(module, input, output)` nested function inside `_register_steering_hook()`
  - [ ] 4.3 Extract activations from output: `activations = output[0] if isinstance(output, tuple) else output` (handle tuple outputs)
  - [ ] 4.4 Pass activations through SAE encoder: `feature_activations = torch.relu(sae_encoder(activations))` (shape: [batch, seq, latent_dim])
  - [ ] 4.5 Apply multiplicative steering: `for feature_id, coeff in steering_config.items(): feature_activations[:, :, feature_id] *= (1 + coeff)`
  - [ ] 4.6 Clip steered features: `feature_activations = torch.clamp(feature_activations, min=-10, max=10)` (prevent numerical instability)
  - [ ] 4.7 Pass through SAE decoder: `steered_activations = sae_decoder(feature_activations)` (reconstruct to hidden_dim)
  - [ ] 4.8 Return steered activations: `return (steered_activations,) if isinstance(output, tuple) else steered_activations` (match output format)
  - [ ] 4.9 Register hook on target layer: `target_layer = model.transformer.h[intervention_layer]` (GPT-2 architecture), `handle = target_layer.register_forward_hook(steering_hook)`
  - [ ] 4.10 Store hook handle: `self.hook_handles.append(handle)`
  - [ ] 4.11 Implement `_cleanup_hooks()` method: `for handle in self.hook_handles: handle.remove()`, `self.hook_handles.clear()`

- [ ] 5.0 Unsteered Generation
  - [ ] 5.1 Implement `_generate_unsteered()` method: accepts model, tokenizer, prompt, temperature, max_tokens, seed
  - [ ] 5.2 Set random seed if provided: `if seed is not None: torch.manual_seed(seed)`
  - [ ] 5.3 Tokenize prompt: `input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()`
  - [ ] 5.4 Generate with log probs: `with torch.no_grad(): outputs = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, return_dict_in_generate=True, output_scores=True)`
  - [ ] 5.5 Extract output text: `output_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)`
  - [ ] 5.6 Extract log probabilities: `logprobs = torch.stack(outputs.scores, dim=1)` (shape: [batch, seq_len, vocab_size])
  - [ ] 5.7 Return tuple: `(output_text, logprobs)`

- [ ] 6.0 Steered Generation
  - [ ] 6.1 Implement `_generate_steered()` method: accepts model, tokenizer, sae_encoder, sae_decoder, prompt, features, intervention_layer, temperature, max_tokens, seed
  - [ ] 6.2 Build steering config dict: `steering_config = {f["feature_id"]: f["coefficient"] for f in features}`
  - [ ] 6.3 Register steering hook using `_register_steering_hook()`
  - [ ] 6.4 Set random seed if provided: `if seed is not None: torch.manual_seed(seed)` (same as unsteered for fair comparison)
  - [ ] 6.5 Tokenize prompt: `input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()`
  - [ ] 6.6 Generate with steering active: `with torch.no_grad(): outputs = model.generate(...)` (same parameters as unsteered)
  - [ ] 6.7 Extract output text and log probs (same as unsteered)
  - [ ] 6.8 Use try/finally block: ensure `_cleanup_hooks()` is called even if generation fails
  - [ ] 6.9 Return tuple: `(output_text, logprobs)`

- [ ] 7.0 Main Generation Method
  - [ ] 7.1 Implement `generate_with_steering()` method (public API): accepts model_id, prompt, features, intervention_layer, temperature, max_tokens, seed
  - [ ] 7.2 Load model and tokenizer: `model, tokenizer = await self.model_registry.load_model(model_id)`
  - [ ] 7.3 Load SAE encoder and decoder: `sae_encoder, sae_decoder = await self._load_sae_for_features(features)`
  - [ ] 7.4 Call `_generate_unsteered()` to get baseline output and log probs
  - [ ] 7.5 Call `_generate_steered()` to get steered output and log probs
  - [ ] 7.6 Call `_calculate_metrics()` to compute comparison metrics
  - [ ] 7.7 Return dict: `{"unsteered_output": str, "steered_output": str, "metrics": ComparisonMetrics}`
  - [ ] 7.8 Add timeout handling: wrap generation in asyncio timeout (30 seconds)
  - [ ] 7.9 Add OOM error handling: catch CUDA OOM, suggest reducing max_tokens in error message
  - [ ] 7.10 Add logging: log model_id, num_features, generation times

- [ ] 8.0 Metrics Calculation - KL Divergence
  - [ ] 8.1 Implement `_calculate_kl_divergence()` method: accepts unsteered_logprobs, steered_logprobs (both Tensor [batch, seq, vocab])
  - [ ] 8.2 Convert log probs to probabilities: `p_unsteered = torch.softmax(unsteered_logprobs, dim=-1)`, `p_steered = torch.softmax(steered_logprobs, dim=-1)`
  - [ ] 8.3 Calculate KL divergence: `kl_div = torch.sum(p_unsteered * torch.log(p_unsteered / (p_steered + 1e-10)), dim=-1)` (avoid log(0))
  - [ ] 8.4 Average across tokens: `kl_div_mean = kl_div.mean().item()`
  - [ ] 8.5 Return float rounded to 4 decimals: `round(kl_div_mean, 4)`

- [ ] 9.0 Metrics Calculation - Perplexity Delta
  - [ ] 9.1 Implement `_calculate_perplexity_delta()` method: accepts unsteered_logprobs, steered_logprobs
  - [ ] 9.2 Extract max log prob per token (NLL approximation): `unsteered_nll = -torch.max(unsteered_logprobs, dim=-1).values.mean()`, `steered_nll = -torch.max(steered_logprobs, dim=-1).values.mean()`
  - [ ] 9.3 Calculate perplexities: `perp_unsteered = torch.exp(unsteered_nll).item()`, `perp_steered = torch.exp(steered_nll).item()`
  - [ ] 9.4 Calculate delta: `perplexity_delta = perp_steered - perp_unsteered` (positive = less confident, negative = more confident)
  - [ ] 9.5 Return float rounded to 2 decimals: `round(perplexity_delta, 2)`

- [ ] 10.0 Metrics Calculation - Semantic Similarity
  - [ ] 10.1 Implement `_calculate_semantic_similarity()` method: accepts unsteered_text, steered_text (both str)
  - [ ] 10.2 Encode both texts: `embeddings = self.sentence_model.encode([unsteered_text, steered_text], convert_to_tensor=True, show_progress_bar=False)` (returns [2, 384])
  - [ ] 10.3 Calculate cosine similarity: `similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))`
  - [ ] 10.4 Return float rounded to 2 decimals: `round(similarity.item(), 2)`
  - [ ] 10.5 Handle empty text edge case: if both texts empty, return 1.0 (identical)

- [ ] 11.0 Metrics Calculation - Word Overlap
  - [ ] 11.1 Implement `_calculate_word_overlap()` method: accepts unsteered_text, steered_text, tokenizer
  - [ ] 11.2 Tokenize both texts: `tokens_unsteered = set(tokenizer.tokenize(unsteered_text))`, `tokens_steered = set(tokenizer.tokenize(steered_text))`
  - [ ] 11.3 Calculate intersection: `intersection = len(tokens_unsteered & tokens_steered)`
  - [ ] 11.4 Calculate union: `union = len(tokens_unsteered | tokens_steered)`
  - [ ] 11.5 Calculate Jaccard similarity: `overlap = intersection / union if union > 0 else 0.0`
  - [ ] 11.6 Return float rounded to 2 decimals: `round(overlap, 2)`

- [ ] 12.0 Metrics Calculation - Integration
  - [ ] 12.1 Implement `_calculate_metrics()` method: accepts unsteered_text, steered_text, unsteered_logprobs, steered_logprobs, tokenizer
  - [ ] 12.2 Call `_calculate_kl_divergence()` and store result
  - [ ] 12.3 Call `_calculate_perplexity_delta()` and store result
  - [ ] 12.4 Call `_calculate_semantic_similarity()` and store result
  - [ ] 12.5 Call `_calculate_word_overlap()` and store result
  - [ ] 12.6 Return dict with all 4 metrics: `{"kl_divergence": float, "perplexity_delta": float, "semantic_similarity": float, "word_overlap": float}`
  - [ ] 12.7 Add error handling: log metric calculation errors without failing entire request

- [ ] 13.0 FastAPI Route - POST /api/steering/generate
  - [ ] 13.1 Create `steering.py` router file, initialize `router = APIRouter()`
  - [ ] 13.2 Define `@router.post("/generate")` endpoint with response model `SteeringGenerateResponse`
  - [ ] 13.3 Accept `request: SteeringGenerateRequest` body
  - [ ] 13.4 Add authentication dependency: `current_user: User = Depends(get_current_user)`
  - [ ] 13.5 Validate model exists and status='ready': query database, raise 404 if not found
  - [ ] 13.6 Validate features exist in database: raise 404 if any feature_id not found
  - [ ] 13.7 Validate intervention_layer does not exceed model's layer count: raise 400 if invalid
  - [ ] 13.8 Validate features belong to same training: raise 422 if mismatched trainings
  - [ ] 13.9 Initialize SteeringService and call `generate_with_steering()`
  - [ ] 13.10 Measure generation time: record start/end time, calculate `generation_time_ms`
  - [ ] 13.11 Return response with all fields populated
  - [ ] 13.12 Add error handling: 400 (validation), 404 (not found), 408 (timeout), 422 (unprocessable), 507 (OOM)
  - [ ] 13.13 Add rate limiting: 10 requests per minute per user (use SlowAPI or custom middleware)

- [ ] 14.0 Frontend Types & Store Setup
  - [ ] 14.1 Create `frontend/src/types/steering.ts` file
  - [ ] 14.2 Define `SteeringFeature` interface: `{ feature_id: number, coefficient: number }`
  - [ ] 14.3 Define `SteeringGenerateRequest` interface (matches backend schema)
  - [ ] 14.4 Define `ComparisonMetrics` interface (matches backend schema)
  - [ ] 14.5 Define `SteeringGenerateResponse` interface (matches backend schema)
  - [ ] 14.6 Create `frontend/src/stores/steeringStore.ts` file
  - [ ] 14.7 Initialize Zustand store with `create<SteeringStore>()` and `devtools()` wrapper

- [ ] 15.0 Zustand Store - State Definition
  - [ ] 15.1 Define state: `selectedFeatures: Feature[]` (array of Feature objects from feature discovery)
  - [ ] 15.2 Define state: `steeringCoefficients: Record<number, number>` (map feature_id to coefficient)
  - [ ] 15.3 Define state: `featureSearch: string` (search query for feature selection)
  - [ ] 15.4 Define state: `selectedModel: string` (model_id)
  - [ ] 15.5 Define state: `prompt: string` (input prompt)
  - [ ] 15.6 Define state: `interventionLayer: number` (default 12)
  - [ ] 15.7 Define state: `temperature: number` (default 0.7)
  - [ ] 15.8 Define state: `maxTokens: number` (default 100)
  - [ ] 15.9 Define state: `isGenerating: boolean` (loading state)
  - [ ] 15.10 Define state: `unsteeredOutput: string`, `steeredOutput: string`
  - [ ] 15.11 Define state: `comparisonMetrics: ComparisonMetrics | null`
  - [ ] 15.12 Define state: `error: string | null`

- [ ] 16.0 Zustand Store - Feature Management Actions
  - [ ] 16.1 Implement `setFeatureSearch(query: string)`: updates featureSearch state
  - [ ] 16.2 Implement `addFeature(feature: Feature)`: checks if already selected, adds to selectedFeatures, initializes coefficient to 0.0, clears search
  - [ ] 16.3 Implement `removeFeature(featureId: number)`: filters out from selectedFeatures, deletes from steeringCoefficients
  - [ ] 16.4 Implement `updateCoefficient(featureId: number, value: number)`: updates coefficient in steeringCoefficients map
  - [ ] 16.5 Add client-side coefficient validation: clamp to [-5.0, 5.0] range

- [ ] 17.0 Zustand Store - Generation Configuration Actions
  - [ ] 17.1 Implement `setSelectedModel(modelId: string)`: updates selectedModel state
  - [ ] 17.2 Implement `setPrompt(prompt: string)`: updates prompt state
  - [ ] 17.3 Implement `setInterventionLayer(layer: number)`: updates interventionLayer state
  - [ ] 17.4 Implement `setTemperature(temp: number)`: validates range [0, 2], updates temperature state
  - [ ] 17.5 Implement `setMaxTokens(tokens: number)`: validates range [1, 2048], updates maxTokens state

- [ ] 18.0 Zustand Store - Generation Action
  - [ ] 18.1 Implement `generateComparison()` async action
  - [ ] 18.2 Build request payload: map selectedFeatures to `{feature_id, coefficient}` array
  - [ ] 18.3 Set `isGenerating: true, error: null`
  - [ ] 18.4 Fetch API: `POST /api/steering/generate` with JSON body
  - [ ] 18.5 Add Authorization header: `Bearer ${getAuthToken()}`
  - [ ] 18.6 Handle response: parse JSON, update `unsteeredOutput`, `steeredOutput`, `comparisonMetrics`
  - [ ] 18.7 Handle errors: set `error` state with error message
  - [ ] 18.8 Set `isGenerating: false` after completion (in finally block)
  - [ ] 18.9 Implement `reset()` action: clears all outputs, metrics, errors, and selected features

- [ ] 19.0 SteeringPanel Component - Setup
  - [ ] 19.1 Create `frontend/src/components/panels/SteeringPanel.tsx` file
  - [ ] 19.2 Import required components: Search, X, Zap, Loader, Copy icons from lucide-react
  - [ ] 19.3 Import Zustand store: `useSteeringStore`, models store: `useModelsStore`
  - [ ] 19.4 Define component: `export const SteeringPanel: React.FC = () => { ... }`
  - [ ] 19.5 Extract all state from stores using Zustand selectors
  - [ ] 19.6 Add local state: `const [featureSearch, setFeatureSearch] = useState('')` (for search input)
  - [ ] 19.7 Filter models to ready only: `const models = useModelsStore(state => state.models.filter(m => m.status === 'ready'))`

- [ ] 20.0 SteeringPanel - Feature Selection Panel (Lines 3582-3706)
  - [ ] 20.1 Create two-column grid layout: `<div className="grid grid-cols-2 gap-6">`
  - [ ] 20.2 Create left column container: `<div className="space-y-4">`
  - [ ] 20.3 Create panel: `<div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">`
  - [ ] 20.4 Add title: `<h3 className="text-lg font-semibold mb-4">Select Features to Steer</h3>`
  - [ ] 20.5 Create search input with Search icon: `<div className="relative mb-4">`, position Search icon absolutely (left-3, top-1/2, transform -translate-y-1/2)
  - [ ] 20.6 Add input field: placeholder "Search features to add...", value={featureSearch}, onChange updates state
  - [ ] 20.7 Style input: `w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500`
  - [ ] 20.8 Add search results dropdown: conditional render if featureSearch is non-empty
  - [ ] 20.9 Style dropdown: `bg-slate-900 border border-slate-700 rounded-lg shadow-lg max-h-48 overflow-y-auto`
  - [ ] 20.10 Map filtered features: show feature ID (mono font, slate-400) + label + "Add" button (emerald-400)
  - [ ] 20.11 Add onClick handler: call `addFeature(feature)` when "Add" clicked
  - [ ] 20.12 Filter out already-selected features from search results (client-side)

- [ ] 21.0 SteeringPanel - Selected Features List (Lines 3621-3705)
  - [ ] 21.1 Create selected features section: `<div className="space-y-3">`
  - [ ] 21.2 Add header: `<h4 className="text-sm font-medium text-slate-300">Selected Features ({selectedFeatures.length})</h4>`
  - [ ] 21.3 Add empty state: if no features selected, show centered message "No features selected. Search and add features above." (slate-400, py-8)
  - [ ] 21.4 Map selected features: `selectedFeatures.map(feature => ...)`
  - [ ] 21.5 Create feature card: `<div className="bg-slate-800/30 rounded-lg p-4 space-y-3">`
  - [ ] 21.6 Add header row: Feature ID (mono, slate-400) + Label (font-medium) + Remove button (X icon, red-400 hover red-300)
  - [ ] 21.7 Add remove button onClick: call `removeFeature(feature.id)`

- [ ] 22.0 SteeringPanel - Coefficient Slider (Lines 3651-3676)
  - [ ] 22.1 Create slider container: `<div className="space-y-2">`
  - [ ] 22.2 Add coefficient display row: "Coefficient" label (slate-400) + current value (emerald-400, mono font, 2 decimals)
  - [ ] 22.3 Add range input: `type="range" min="-5" max="5" step="0.1"`
  - [ ] 22.4 Set value: `value={steeringCoefficients[feature.id] || 0}`
  - [ ] 22.5 Add onChange: `onChange={(e) => updateCoefficient(feature.id, parseFloat(e.target.value))}`
  - [ ] 22.6 Style slider: `w-full accent-emerald-500`
  - [ ] 22.7 Add slider labels row: "-5.0 (suppress)" | "0.0" | "+5.0 (amplify)" (xs, slate-500, justify-between)

- [ ] 23.0 SteeringPanel - Quick Presets (Lines 3678-3701)
  - [ ] 23.1 Create presets container: `<div className="flex gap-2">`
  - [ ] 23.2 Add "Suppress" button: onClick sets coefficient to -2.0, style `px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600`
  - [ ] 23.3 Add "Reset" button: onClick sets coefficient to 0.0, same style
  - [ ] 23.4 Add "Amplify" button: onClick sets coefficient to 2.0, same style

- [ ] 24.0 SteeringPanel - Generation Controls Panel (Lines 3710-3817)
  - [ ] 24.1 Create right column container (matching left column structure)
  - [ ] 24.2 Create panel: `<div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">`
  - [ ] 24.3 Add title: `<h3 className="text-lg font-semibold">Generation Configuration</h3>`
  - [ ] 24.4 Add model selection dropdown: label "Model", select with placeholder "Select model...", map ready models to options
  - [ ] 24.5 Style dropdown: `w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500`
  - [ ] 24.6 Add prompt textarea: label "Prompt", rows={4}, placeholder "Enter your prompt here...", resize-none
  - [ ] 24.7 Style textarea: same as dropdown + `resize-none`
  - [ ] 24.8 Add intervention layer slider: label "Intervention Layer: {interventionLayer}", range 0-24, value={interventionLayer}
  - [ ] 24.9 Style layer slider: `w-full accent-emerald-500`

- [ ] 25.0 SteeringPanel - Generation Parameters (Lines 3763-3795)
  - [ ] 25.1 Create 2-column grid: `<div className="grid grid-cols-2 gap-4">`
  - [ ] 25.2 Add temperature input: label "Temperature", type="number" min="0" max="2" step="0.1", value={temperature}
  - [ ] 25.3 Add max tokens input: label "Max Tokens", type="number" min="1" max="2048", value={maxTokens}
  - [ ] 25.4 Style both inputs: `w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500`

- [ ] 26.0 SteeringPanel - Generate Button (Lines 3797-3815)
  - [ ] 26.1 Create button: full width, `w-full px-6 py-3`
  - [ ] 26.2 Add disabled condition: `!prompt || selectedFeatures.length === 0 || isGenerating`
  - [ ] 26.3 Style button: `bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium`
  - [ ] 26.4 Add loading state: if isGenerating, show Loader icon (animate-spin) + "Generating..." text
  - [ ] 26.5 Add idle state: if not generating, show Zap icon + "Generate Comparison" text
  - [ ] 26.6 Add onClick: call `generateComparison()` from store

- [ ] 27.0 SteeringPanel - Comparative Output Display (Lines 3820-3895)
  - [ ] 27.1 Create conditional render: only show if unsteeredOutput OR steeredOutput exists
  - [ ] 27.2 Create output container: `<div className="space-y-6">`
  - [ ] 27.3 Create 2-column grid: `<div className="grid grid-cols-2 gap-6">`
  - [ ] 27.4 Create unsteered panel: header "Unsteered (Baseline)" with slate-400 dot indicator (`<div className="w-2 h-2 rounded-full bg-slate-400" />`)
  - [ ] 27.5 Add copy button: Copy icon, onClick copies unsteeredOutput to clipboard
  - [ ] 27.6 Create output box: `bg-slate-900/50 border border-slate-800 rounded-lg p-4 min-h-[200px]`
  - [ ] 27.7 Add loading state: if isGenerating, show centered Loader icon (slate-400)
  - [ ] 27.8 Add output text: `whitespace-pre-wrap text-sm leading-relaxed` (slate-300)
  - [ ] 27.9 Add empty state: if no output, show "No generation yet" (centered, slate-500)

- [ ] 28.0 SteeringPanel - Steered Output Panel (Lines 3860-3894)
  - [ ] 28.1 Create steered panel (mirror unsteered structure)
  - [ ] 28.2 Change header to "Steered" with emerald-400 dot indicator
  - [ ] 28.3 Change border to `border border-emerald-800/30` (emerald border for steered)
  - [ ] 28.4 Add loading state: if isGenerating, show Loader icon (emerald-400 instead of slate-400)
  - [ ] 28.5 Add output text (same styling as unsteered)
  - [ ] 28.6 Add empty state (same as unsteered)

- [ ] 29.0 SteeringPanel - Comparison Metrics Panel (Lines 3897-3947)
  - [ ] 29.1 Create conditional render: only show if comparisonMetrics exists
  - [ ] 29.2 Create metrics container: `<div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">`
  - [ ] 29.3 Add title: `<h3 className="text-lg font-semibold mb-4">Comparison Metrics</h3>`
  - [ ] 29.4 Create 4-column grid: `<div className="grid grid-cols-4 gap-4">`

- [ ] 30.0 SteeringPanel - KL Divergence Metric (Lines 3903-3911)
  - [ ] 30.1 Create metric card: `<div className="bg-slate-800/50 rounded-lg p-4">`
  - [ ] 30.2 Add label: "KL Divergence" (xs, slate-400)
  - [ ] 30.3 Add value: `{comparisonMetrics.kl_divergence.toFixed(4)}` (text-2xl font-bold text-purple-400)
  - [ ] 30.4 Add description: "Distribution shift" (xs, slate-500)

- [ ] 31.0 SteeringPanel - Perplexity Delta Metric (Lines 3913-3924)
  - [ ] 31.1 Create metric card: `<div className="bg-slate-800/50 rounded-lg p-4">`
  - [ ] 31.2 Add label: "Perplexity Δ" (xs, slate-400)
  - [ ] 31.3 Add value with conditional color: `text-2xl font-bold ${comparisonMetrics.perplexity_delta > 0 ? 'text-red-400' : 'text-emerald-400'}`
  - [ ] 31.4 Add sign prefix: `{comparisonMetrics.perplexity_delta > 0 ? '+' : ''}`
  - [ ] 31.5 Format value: `{comparisonMetrics.perplexity_delta.toFixed(2)}`
  - [ ] 31.6 Add conditional description: "{perplexity_delta > 0 ? 'Higher' : 'Lower'} uncertainty" (xs, slate-500)

- [ ] 32.0 SteeringPanel - Semantic Similarity Metric (Lines 3926-3934)
  - [ ] 32.1 Create metric card: `<div className="bg-slate-800/50 rounded-lg p-4">`
  - [ ] 32.2 Add label: "Similarity" (xs, slate-400)
  - [ ] 32.3 Add value as percentage: `{(comparisonMetrics.semantic_similarity * 100).toFixed(1)}%` (text-2xl font-bold text-blue-400)
  - [ ] 32.4 Add description: "Cosine similarity" (xs, slate-500)

- [ ] 33.0 SteeringPanel - Word Overlap Metric (Lines 3936-3944)
  - [ ] 33.1 Create metric card: `<div className="bg-slate-800/50 rounded-lg p-4">`
  - [ ] 33.2 Add label: "Word Overlap" (xs, slate-400)
  - [ ] 33.3 Add value as percentage: `{(comparisonMetrics.word_overlap * 100).toFixed(1)}%` (text-2xl font-bold text-emerald-400)
  - [ ] 33.4 Add description: "Shared tokens" (xs, slate-500)

- [ ] 34.0 Unit Tests - Steering Algorithm
  - [ ] 34.1 Create test file: `backend/tests/unit/test_steering_service.py`
  - [ ] 34.2 Test multiplicative steering: `test_multiplicative_steering()` - verify `steered = original * (1 + coeff)` for positive, negative, zero coefficients
  - [ ] 34.3 Test steering clipping: `test_steering_clipping()` - verify extreme coefficients get clipped to [-10, 10]
  - [ ] 34.4 Test hook registration: `test_hook_registration()` - verify hooks are registered and handle stored
  - [ ] 34.5 Test hook cleanup: `test_hook_cleanup()` - verify hooks are removed and handles cleared
  - [ ] 34.6 Test hook cleanup on error: `test_hook_cleanup_on_generation_error()` - verify hooks cleaned even if generation fails

- [ ] 35.0 Unit Tests - Metrics Calculations
  - [ ] 35.1 Test KL divergence with identical distributions: `test_kl_divergence_identical()` - should return ~0
  - [ ] 35.2 Test KL divergence with different distributions: `test_kl_divergence_different()` - should return > 0
  - [ ] 35.3 Test perplexity delta with identical outputs: `test_perplexity_delta_identical()` - should return ~0
  - [ ] 35.4 Test perplexity delta with different confidences: verify positive/negative deltas
  - [ ] 35.5 Test semantic similarity with identical texts: `test_semantic_similarity_identical()` - should return ~1.0
  - [ ] 35.6 Test semantic similarity with different texts: should return < 1.0
  - [ ] 35.7 Test word overlap with identical texts: `test_word_overlap_identical()` - should return 1.0
  - [ ] 35.8 Test word overlap with no shared words: should return 0.0

- [ ] 36.0 Integration Tests - End-to-End Steering
  - [ ] 36.1 Create test file: `backend/tests/integration/test_steering_api.py`
  - [ ] 36.2 Test complete steering workflow: `test_steering_generation_end_to_end()` - load model, features, POST /api/steering/generate, validate response structure
  - [ ] 36.3 Test steering with zero coefficients: `test_steering_with_zero_coefficients()` - outputs should be very similar (high similarity, low KL)
  - [ ] 36.4 Test steering with multiple features: `test_steering_with_multiple_features()` - verify 5+ features work correctly
  - [ ] 36.5 Test steering at different layers: `test_steering_at_different_layers()` - test early (layer 2), middle (layer 6), late (layer 10)
  - [ ] 36.6 Test extreme coefficient values: `test_extreme_coefficient_values()` - verify -5.0 and +5.0 work, produce observable effects
  - [ ] 36.7 Test invalid intervention layer: `test_invalid_intervention_layer()` - should return 400 error
  - [ ] 36.8 Test empty prompt: `test_empty_prompt()` - should return 400 error
  - [ ] 36.9 Test generation timeout: `test_generation_timeout()` - mock long-running generation, verify 408 timeout
  - [ ] 36.10 Test OOM handling: `test_oom_error_handling()` - mock CUDA OOM, verify graceful error with suggestion

- [ ] 37.0 Performance Tests
  - [ ] 37.1 Test generation speed: `test_generation_speed()` - verify 100-token generation (both outputs) completes within 3 seconds
  - [ ] 37.2 Test metrics calculation speed: `test_metrics_calculation_speed()` - verify all 4 metrics calculated within 500ms
  - [ ] 37.3 Test multiple features overhead: `test_multiple_features_performance()` - measure overhead with 1, 5, 10 features
  - [ ] 37.4 Test sentence embedding caching: verify sentence model loaded once, not reloaded per request

- [ ] 38.0 Frontend Component Tests
  - [ ] 38.1 Create test file: `frontend/src/components/panels/SteeringPanel.test.tsx`
  - [ ] 38.2 Test feature search: verify search input filters features
  - [ ] 38.3 Test adding features: verify feature added to selected list, coefficient initialized to 0
  - [ ] 38.4 Test removing features: verify feature removed from list, coefficient deleted
  - [ ] 38.5 Test coefficient slider: verify slider updates coefficient in real-time
  - [ ] 38.6 Test quick presets: verify Suppress/Reset/Amplify buttons set correct values
  - [ ] 38.7 Test generate button disabled state: verify disabled when prompt empty or no features selected
  - [ ] 38.8 Test loading state: verify loading indicator shows during generation
  - [ ] 38.9 Test comparison metrics display: verify all 4 metrics render with correct colors and formatting
  - [ ] 38.10 Test copy to clipboard: verify copy buttons work for both outputs

- [ ] 39.0 Error Handling & Edge Cases
  - [ ] 39.1 Add validation: reject features from different trainings (raise 422 error)
  - [ ] 39.2 Add validation: reject intervention layer exceeding model depth (raise 400 error)
  - [ ] 39.3 Add validation: reject coefficients outside [-5, 5] range (raise 400 error)
  - [ ] 39.4 Add timeout: 30 second hard limit on generation (raise 408 error)
  - [ ] 39.5 Add OOM handling: catch CUDA OOM errors, return 507 with suggestion to reduce max_tokens
  - [ ] 39.6 Add hook cleanup guarantee: use try/finally to ensure hooks always removed
  - [ ] 39.7 Handle empty outputs: if generation produces empty string, return null metrics
  - [ ] 39.8 Handle sentence embedding failures: log error but continue with null semantic_similarity
  - [ ] 39.9 Add rate limiting: 10 requests per minute per user (use SlowAPI or custom middleware)

- [ ] 40.0 Documentation & Deployment
  - [ ] 40.1 Add API documentation: document `/api/steering/generate` endpoint in OpenAPI/Swagger
  - [ ] 40.2 Add example requests/responses to API docs
  - [ ] 40.3 Update Dockerfile: add sentence-transformers installation, pre-download model at build time
  - [ ] 40.4 Add logging: log steering requests (model_id, num_features, generation time)
  - [ ] 40.5 Add monitoring: track generation speed P50/P95/P99, steering effect rate (% with KL > 0.01), OOM error rate
  - [ ] 40.6 Update main router: include steering router in FastAPI app
  - [ ] 40.7 Add CI/CD pipeline: test steering feature on push (run unit + integration tests)
  - [ ] 40.8 Update README: document steering feature usage, example commands

---

**Total Tasks:** 40 parent tasks, 195 sub-tasks
**Estimated Complexity:** High (ML hooks + dual generation + complex metrics)
**Critical Path:** Steering hook implementation → Metrics calculation → Frontend integration
**Testing Priority:** Hook cleanup (memory leaks), generation consistency, metrics accuracy

**Implementation Notes:**
- **PRIMARY UI REFERENCE:** Mock-embedded-interp-ui.tsx lines 3512-3951
- Multiplicative steering is recommended (scales with activation magnitude)
- Use try/finally for hook cleanup to prevent memory leaks
- Sequential generation (unsteered → steered) avoids CUDA context conflicts
- Sentence model cached at service initialization (not per-request)
- Coefficient range [-5, 5]: -5 = 83% suppression, 0 = neutral, +5 = 6x amplification
- Target performance: 3 seconds for 100-token generation (both outputs), 500ms for metrics
- Jetson Orin Nano memory budget: ~2.3GB (model 1.5GB + SAE 200MB + sentence model 80MB + buffers 500MB)
