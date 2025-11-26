# Product Requirements Document: SAE Feature Steering with Side-by-Side Comparison

## 1. Executive Summary

### 1.1 Overview
This PRD defines requirements for adding SAE (Sparse Autoencoder) feature steering capabilities to the application, enabling users to:
1. Select interpretable features from their trained SAEs or downloaded SAEs from HuggingFace
2. Create steering vectors that modify model behavior
3. Compare steered vs. unsteered outputs side-by-side
4. Evaluate steering effectiveness quantitatively

### 1.2 Value Proposition
- **For Researchers**: Test hypotheses about feature causality and model behavior
- **For ML Engineers**: Fine-tune model behavior without retraining
- **For Safety Teams**: Test and evaluate safety interventions
- **For Product Teams**: Prototype behavior modifications quickly

### 1.3 Success Metrics
- Time to create first steering experiment: < 5 minutes
- User ability to interpret steering results: > 80% success rate
- System support for concurrent steering comparisons: 4+ configurations simultaneously

---

## 2. Background & Context

### 2.1 What is Feature Steering?

**Feature steering** modifies language model outputs by intervening on specific SAE features during generation. Unlike fine-tuning or prompting:
- **No retraining required**: Inference-time intervention only
- **Interpretable**: Target specific, human-understandable concepts
- **Reversible**: Easy to enable/disable
- **Precise**: Control specific behaviors while minimizing side effects

### 2.2 Core Concepts

**Steering Vector**: A direction in activation space (derived from SAE decoder weights) that represents a feature concept

**Steering Strength (Î±)**: Scalar multiplier controlling intervention intensity

**Steering Methods**:
1. **Direct SAE Feature Steering**: Use SAE decoder vector at feature index
2. **Multi-feature Steering**: Combine multiple features with weights
3. **SAE-Targeted Steering (SAE-TS)**: Optimize to target features while minimizing side effects

**Key Equation**:
```
steered_activation = original_activation + (Î± Ã— steering_vector)
```

### 2.3 Use Cases

**Safety & Alignment**:
- Reduce harmful outputs (toxicity, bias, etc.)
- Increase refusal of dangerous requests
- Modify political/ideological slant

**Behavior Modification**:
- Make model more/less creative
- Adjust formality/casualness
- Control verbosity
- Steer toward/away from topics

**Research & Interpretability**:
- Test feature causality
- Validate SAE quality
- Understand feature interactions
- Map feature effects

---

## 3. User Stories & Personas

### 3.1 Primary Personas

**Research Scientist (Sarah)**
- Trained an SAE on GPT-2 Small
- Wants to test if "religious text" feature actually causes religious output
- Needs to see clear before/after comparison
- Values scientific rigor and metrics

**ML Engineer (Alex)**
- Downloaded Gemma-Scope SAE from HuggingFace
- Wants to reduce model's tendency toward certain topics
- Needs to deploy quickly
- Values practical effectiveness over perfect interpretability

**Safety Researcher (Jordan)**
- Testing safety interventions on refusal behavior
- Needs to compare multiple steering strengths
- Requires quantitative evaluation
- Values both safety and capability preservation

### 3.2 User Stories

**Story 1: Basic Feature Steering**
```
As Sarah (researcher),
I want to select a feature from my trained SAE and see how it affects generation,
So that I can validate that the feature is causally relevant.

Acceptance Criteria:
- Can browse features from my SAE
- Can set steering strength with slider
- See side-by-side unsteered vs steered output
- Generate multiple samples with same prompt
```

**Story 2: Multi-Feature Comparison**
```
As Alex (engineer),
I want to compare steering with different features simultaneously,
So that I can find the most effective intervention.

Acceptance Criteria:
- Can select up to 4 features to compare
- See all outputs side-by-side
- Can adjust steering strength per feature independently
- Can export comparison results
```

**Story 3: Strength Sweep**
```
As Jordan (safety researcher),
I want to test a range of steering strengths for one feature,
So that I can find the optimal balance of steering vs coherence.

Acceptance Criteria:
- Can specify range (e.g., Î± = 0, 10, 25, 50, 100)
- See outputs for all strengths at once
- Get quantitative metrics (perplexity, behavioral score)
- Can plot steering curves
```

**Story 4: HuggingFace Integration**
```
As Alex (engineer),
I want to load a pretrained SAE from HuggingFace and immediately start steering,
So that I don't need to train my own SAE first.

Acceptance Criteria:
- Can browse/search HuggingFace SAEs
- Download with one click
- Feature browser shows Neuronpedia links if available
- Works same as locally trained SAEs
```

---

## 4. Functional Requirements

### 4.1 Feature Selection

**FR-1.1: SAE Loading**
- MUST support loading SAEs from:
  - Local training (recently trained in app)
  - Local disk (user-uploaded)
  - HuggingFace Hub (with authentication)
- MUST validate SAE compatibility with selected model
- MUST cache downloaded SAEs for reuse

**FR-1.2: Feature Browser**
- MUST display list of all SAE features with:
  - Feature index
  - Activation frequency (if available)
  - Top activating tokens (if available)
  - Manual description (if available)
- SHOULD integrate with Neuronpedia for feature interpretation
- SHOULD allow filtering features by:
  - Activation frequency
  - Layer
  - Keywords in description
- SHOULD show feature activation distribution

**FR-1.3: Feature Selection UI**
- MUST support selecting 1-4 features for comparison
- MUST show feature details on hover/click
- SHOULD support searching features by ID or description
- SHOULD highlight recently used features

### 4.2 Steering Configuration

**FR-2.1: Steering Method Selection**
- MUST support at minimum:
  - Direct decoder steering (baseline)
  - Simple multi-feature linear combination
- SHOULD support:
  - SAE-Targeted Steering (SAE-TS)
  - Feature Guided Activation Additions (FGAA)
- MUST document which method is being used

**FR-2.2: Steering Strength Control**
- MUST provide slider for steering strength (Î±)
- MUST support range: -100 to +300 (configurable)
- MUST support negative steering (suppress feature)
- SHOULD provide presets: "Subtle" (Î±=10), "Moderate" (Î±=50), "Strong" (Î±=100)
- SHOULD show recommended range based on feature max activation

**FR-2.3: Layer Selection**
- MUST allow selecting which layer(s) to apply steering
- MUST support:
  - Single layer steering
  - All layers steering
  - Custom layer ranges
- SHOULD provide guidance on layer selection
- SHOULD default to SAE's training layer

**FR-2.4: Multi-Feature Composition**
- MUST support combining multiple features with individual weights
- MUST support both positive and negative weights
- SHOULD visualize the combined steering vector
- SHOULD warn about potential feature interference

### 4.3 Generation & Comparison

**FR-3.1: Prompt Input**
- MUST support text prompt input
- MUST support batch prompts (multiple prompts at once)
- SHOULD support prompt templates
- SHOULD save recent prompts

**FR-3.2: Generation Parameters**
- MUST expose:
  - Max new tokens
  - Temperature
  - Top-p / Top-k
  - Number of samples per configuration
- SHOULD provide presets for common use cases
- MUST use same generation params for all comparisons

**FR-3.3: Side-by-Side Comparison Layout**
- MUST display outputs in column format:
  ```
  [Unsteered] [Feature 1] [Feature 2] [Feature 3]
  ```
- MUST clearly label each column with:
  - Feature ID and name
  - Steering strength
  - Method used
- MUST support horizontal scrolling for >4 columns
- SHOULD highlight differences between outputs
- SHOULD support copying individual outputs

**FR-3.4: Batch Generation**
- MUST support generating N samples per configuration
- MUST display samples in expandable sections
- SHOULD show sample statistics (e.g., "5 of 10 samples mention X")
- SHOULD allow regenerating individual samples

### 4.4 Evaluation & Metrics

**FR-4.1: Automatic Metrics**
- MUST compute and display:
  - **Perplexity**: Language model quality (lower = more coherent)
  - **KL Divergence**: Change from original distribution
  - **Length**: Output token count
- SHOULD compute:
  - **Behavioral Score**: Achievement of steering objective (task-specific)
  - **Coherence Score**: Semantic coherence (via external judge)
  - **Vocabulary Coverage**: Presence of target vocabulary

**FR-4.2: Human Evaluation Interface**
- MUST support manual rating of outputs:
  - Steering effectiveness (1-5 scale)
  - Output quality (1-5 scale)
  - Free-form notes
- SHOULD support blind evaluation (hide which is which)
- SHOULD track ratings over time

**FR-4.3: Comparative Analysis**
- MUST provide comparison table showing metrics for all configurations
- MUST support exporting comparison results to CSV/JSON
- SHOULD visualize metrics with charts:
  - Steering strength vs perplexity curve
  - Behavioral score vs coherence trade-off plot
- SHOULD highlight best performing configuration

**FR-4.4: Statistical Significance**
- SHOULD indicate when differences are statistically significant
- SHOULD support confidence intervals for metrics
- SHOULD warn when sample size is too small

### 4.5 Visualization & Analysis

**FR-5.1: Feature Activation Tracking**
- MUST show which SAE features activated during steered generation
- MUST display activation magnitudes
- SHOULD visualize per-token feature activations
- SHOULD highlight unexpected feature activations (side effects)

**FR-5.2: Steering Effect Visualization**
- MUST show activation delta (steered - unsteered)
- SHOULD project steering effect into SAE feature space
- SHOULD identify most affected features beyond target
- SHOULD warn about large unintended effects

**FR-5.3: Token-Level Analysis**
- SHOULD show token-by-token differences
- SHOULD highlight divergence points
- SHOULD display alternative token probabilities
- SHOULD support hovering for detailed per-token info

### 4.6 Experiment Management

**FR-6.1: Saving & Loading**
- MUST support saving steering configurations as experiments
- MUST save:
  - SAE reference
  - Feature selections
  - Steering strengths
  - Generation parameters
  - Prompts
  - Outputs and metrics
- MUST support loading and re-running experiments
- SHOULD support experiment versioning

**FR-6.2: Experiment Comparison**
- MUST support comparing results across experiments
- SHOULD provide experiment diff view
- SHOULD track performance over time
- SHOULD support tagging experiments

**FR-6.3: Sharing & Collaboration**
- SHOULD support exporting experiment as shareable link
- SHOULD support exporting as notebook/script
- SHOULD generate reproducible code snippets
- SHOULD support team collaboration features

---

## 5. Non-Functional Requirements

### 5.1 Performance

**NFR-1.1: Generation Speed**
- MUST complete steering generation within 2x baseline generation time
- SHOULD support GPU acceleration
- SHOULD support batch processing for efficiency
- Target: <30 seconds for 4 configs Ã— 5 samples Ã— 100 tokens

**NFR-1.2: UI Responsiveness**
- MUST maintain <100ms UI response time
- MUST provide progress indicators for long operations
- MUST support background processing
- MUST allow canceling in-progress generations

**NFR-1.3: Scalability**
- MUST support SAEs up to 131k features
- MUST support models up to 10B parameters
- SHOULD handle 100+ concurrent experiments per user
- SHOULD support team workspaces with 1000+ experiments

### 5.2 Reliability

**NFR-2.1: Error Handling**
- MUST gracefully handle:
  - SAE/model incompatibility
  - Out-of-memory errors
  - Network failures (HuggingFace)
  - Invalid steering configurations
- MUST provide clear error messages with remediation steps
- MUST not lose work on errors

**NFR-2.2: Validation**
- MUST validate steering vector norms
- MUST warn about extreme steering strengths
- MUST check for NaN/Inf in computations
- SHOULD detect and warn about feature incompatibility

### 5.3 Usability

**NFR-3.1: Learning Curve**
- MUST provide interactive tutorial
- MUST include example experiments
- SHOULD provide contextual help
- SHOULD include tooltips for all parameters

**NFR-3.2: Discoverability**
- MUST make steering feature easily discoverable from SAE training
- SHOULD provide workflow guidance
- SHOULD suggest relevant features based on query
- SHOULD recommend steering strengths

### 5.4 Compatibility

**NFR-4.1: SAE Formats**
- MUST support SAELens format (.safetensors)
- SHOULD support other common formats
- MUST handle format version differences gracefully

**NFR-4.2: Model Compatibility**
- MUST support all TransformerLens models
- SHOULD support HuggingFace models
- MUST validate SAE-model layer compatibility

### 5.5 Security & Privacy

**NFR-5.1: Data Privacy**
- MUST not log user prompts without consent
- MUST support local-only mode (no cloud)
- SHOULD support encryption for saved experiments

**NFR-5.2: HuggingFace Integration**
- MUST securely store HF tokens
- MUST support SSO where available
- MUST respect model access controls

---

## 6. User Interface Specifications

### 6.1 Main Steering Interface

**Layout Structure**:
```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Top Bar: [SAE Selector] [Model Selector] [Save] [Export]   â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                             â”‚
â”‚ Left Sidebar (Feature Selection):                          â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                    â”‚
â”‚ â”‚ ðŸ” Search features...                â”‚                    â”‚
â”‚ â”‚                                      â”‚                    â”‚
â”‚ â”‚ Selected Features (2/4):             â”‚                    â”‚
â”‚ â”‚ â˜‘ Feature 7650 - "Religious text"   â”‚                    â”‚
â”‚ â”‚   Î±: [====|====] 50                  â”‚                    â”‚
â”‚ â”‚   Layer: [12 â–¼]                      â”‚                    â”‚
â”‚ â”‚                                      â”‚                    â”‚
â”‚ â”‚ â˜‘ Feature 12331 - "Scientific"      â”‚                    â”‚
â”‚ â”‚   Î±: [====|====] 30                  â”‚                    â”‚
â”‚ â”‚   Layer: [12 â–¼]                      â”‚                    â”‚
â”‚ â”‚                                      â”‚                    â”‚
â”‚ â”‚ [+ Add Feature]                      â”‚                    â”‚
â”‚ â”‚                                      â”‚                    â”‚
â”‚ â”‚ Feature Browser:                     â”‚                    â”‚
â”‚ â”‚ â€¢ Feature 7650 - "Religious text"    â”‚                    â”‚
â”‚ â”‚ â€¢ Feature 12331 - "Scientific"       â”‚                    â”‚
â”‚ â”‚ â€¢ Feature 4521 - "Paris/France"      â”‚                    â”‚
â”‚ â”‚ ... (scrollable list)                â”‚                    â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                    â”‚
â”‚                                                             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                             â”‚
â”‚ Center Panel (Prompt & Generation):                        â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚ â”‚ Prompt:                                                  â”‚â”‚
â”‚ â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚â”‚
â”‚ â”‚ â”‚ In the beginning,                                    â”‚ â”‚â”‚
â”‚ â”‚ â”‚                                                       â”‚ â”‚â”‚
â”‚ â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚â”‚
â”‚ â”‚                                                          â”‚â”‚
â”‚ â”‚ [Temperature: 0.7] [Max tokens: 100] [Samples: 5]      â”‚â”‚
â”‚ â”‚                                                          â”‚â”‚
â”‚ â”‚ [ðŸŽ¯ Generate Comparison]          [âš™ï¸ Advanced Options] â”‚â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â”‚                                                             â”‚
â”‚ Results (Side-by-Side):                                    â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚ â”‚ Unsteered   â”‚ Religious   â”‚ Scientific  â”‚ Multi-Feat  â”‚â”‚
â”‚ â”‚             â”‚ (Î±=50)      â”‚ (Î±=30)      â”‚ Combined    â”‚â”‚
â”‚ â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤â”‚
â”‚ â”‚ God created â”‚ God created â”‚ the universeâ”‚ In the      â”‚â”‚
â”‚ â”‚ the world   â”‚ the heavens â”‚ formed from â”‚ beginning,  â”‚â”‚
â”‚ â”‚ in seven    â”‚ and the     â”‚ a singul... â”‚ scientific  â”‚â”‚
â”‚ â”‚ days...     â”‚ earth...    â”‚             â”‚ understand..â”‚â”‚
â”‚ â”‚             â”‚             â”‚             â”‚             â”‚â”‚
â”‚ â”‚ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”‚â”‚
â”‚ â”‚ Metrics:                                               â”‚â”‚
â”‚ â”‚ Perplexity:  15.2  â”‚  45.3  â”‚  18.7  â”‚  22.1         â”‚â”‚
â”‚ â”‚ Behavioral:  0.0   â”‚  0.95  â”‚  0.82  â”‚  0.88         â”‚â”‚
â”‚ â”‚ Coherence:   0.95  â”‚  0.78  â”‚  0.91  â”‚  0.85         â”‚â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â”‚                                                             â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                             â”‚
â”‚ Bottom Panel (Analysis):                                   â”‚
â”‚ [ðŸ“Š Metrics] [ðŸ”¬ Feature Activations] [ðŸ“ˆ Visualizations]  â”‚
â”‚                                                             â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 6.2 Feature Selection Dialog

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Select Feature to Steer                          â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                  â”‚
â”‚ ðŸ” [Search features...]              [Filters â–¼]â”‚
â”‚                                                  â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚ â”‚ â˜ 7650 - "Religious text"                    â”‚â”‚
â”‚ â”‚      Activates on: Bible, Quran, prayer...   â”‚â”‚
â”‚ â”‚      Frequency: 2.3%  Max: 12.5              â”‚â”‚
â”‚ â”‚      [View in Neuronpedia] [Activation Dist] â”‚â”‚
â”‚ â”‚                                              â”‚â”‚
â”‚ â”‚ â˜ 12331 - "Scientific concepts"              â”‚â”‚
â”‚ â”‚      Activates on: research, experiment...   â”‚â”‚
â”‚ â”‚      Frequency: 4.1%  Max: 8.9               â”‚â”‚
â”‚ â”‚      [View in Neuronpedia] [Activation Dist] â”‚â”‚
â”‚ â”‚                                              â”‚â”‚
â”‚ â”‚ â˜ 4521 - "Paris/France"                      â”‚â”‚
â”‚ â”‚      Activates on: Paris, French, Eiffel...  â”‚â”‚
â”‚ â”‚      Frequency: 0.8%  Max: 15.3              â”‚â”‚
â”‚ â”‚      [View in Neuronpedia] [Activation Dist] â”‚â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â”‚                                                  â”‚
â”‚               [Cancel]  [Add Selected (2)]       â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 6.3 Metrics Dashboard

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ Steering Performance Metrics                         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                                                      â”‚
â”‚ Perplexity vs Steering Strength:                    â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚ â”‚  PPL                                            â”‚  â”‚
â”‚ â”‚  100â”¤                                     â•±     â”‚  â”‚
â”‚ â”‚   80â”¤                               â•±           â”‚  â”‚
â”‚ â”‚   60â”¤                         â•±                 â”‚  â”‚
â”‚ â”‚   40â”¤                   â•±                       â”‚  â”‚
â”‚ â”‚   20â”¤             â•±                             â”‚  â”‚
â”‚ â”‚    0â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  â”‚  â”‚
â”‚ â”‚      0    50   100   150   200   250   300     â”‚  â”‚
â”‚ â”‚                 Steering Strength (Î±)           â”‚  â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                      â”‚
â”‚ Behavioral vs Coherence Trade-off:                  â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚ â”‚  Behavioral                                     â”‚  â”‚
â”‚ â”‚  1.0â”¤                    â—                      â”‚  â”‚
â”‚ â”‚  0.8â”¤               â—    Feature 7650          â”‚  â”‚
â”‚ â”‚  0.6â”¤          â—                                â”‚  â”‚
â”‚ â”‚  0.4â”¤     â—                                     â”‚  â”‚
â”‚ â”‚  0.2â”¤â—                                          â”‚  â”‚
â”‚ â”‚  0.0â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€  â”‚  â”‚
â”‚ â”‚     0.0  0.2  0.4  0.6  0.8  1.0               â”‚  â”‚
â”‚ â”‚              Coherence Score                    â”‚  â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                      â”‚
â”‚ Summary Table:                                       â”‚
â”‚ â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚ â”‚ Config â”‚  PPL  â”‚Behav.  â”‚Coherence â”‚Combined  â”‚  â”‚
â”‚ â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤  â”‚
â”‚ â”‚Unsteered 15.2 â”‚  0.00  â”‚  0.95    â”‚  0.00    â”‚  â”‚
â”‚ â”‚Î±=50    â”‚ 45.3  â”‚  0.95  â”‚  0.78    â”‚â˜… 0.74   â”‚  â”‚
â”‚ â”‚Î±=100   â”‚ 78.1  â”‚  0.98  â”‚  0.65    â”‚  0.64    â”‚  â”‚
â”‚ â””â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                      â”‚
â”‚ [ðŸ“¥ Export Data] [ðŸ“Š Full Report] [ðŸ”„ Rerun]        â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## 7. Technical Architecture

### 7.1 Component Overview

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                   Frontend (React)                  â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚   Feature   â”‚   Steering   â”‚   Comparison    â”‚  â”‚
â”‚  â”‚   Browser   â”‚Configuration â”‚     Viewer      â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                        â†“ â†‘ API
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              Backend (Python/FastAPI)               â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚         Steering Controller                   â”‚  â”‚
â”‚  â”‚  â€¢ Feature selection                          â”‚  â”‚
â”‚  â”‚  â€¢ Steering vector construction               â”‚  â”‚
â”‚  â”‚  â€¢ Batch generation orchestration             â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â”‚                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
â”‚  â”‚         Evaluation Engine                     â”‚  â”‚
â”‚  â”‚  â€¢ Metric computation                         â”‚  â”‚
â”‚  â”‚  â€¢ Statistical analysis                       â”‚  â”‚
â”‚  â”‚  â€¢ Comparison logic                           â”‚  â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                        â†“ â†‘
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚           Model Inference Layer                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚ SAELens      â”‚  TransformerLens               â”‚ â”‚
â”‚  â”‚ â€¢ SAE loadingâ”‚  â€¢ Model loading               â”‚ â”‚
â”‚  â”‚ â€¢ Feature    â”‚  â€¢ Hook management             â”‚ â”‚
â”‚  â”‚   encoding   â”‚  â€¢ Forward passes              â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                        â†“ â†‘
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                Storage Layer                        â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”â”‚
â”‚  â”‚    SAEs    â”‚Experimentsâ”‚   HuggingFace Cache   â”‚â”‚
â”‚  â”‚  (Local)   â”‚  (SQLite) â”‚                       â”‚â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 7.2 Key Algorithms

**Algorithm 1: Basic Feature Steering**
```python
def apply_feature_steering(
    model: HookedTransformer,
    sae: SAE,
    feature_idx: int,
    steering_strength: float,
    layer: int,
    prompt: str,
    max_tokens: int
) -> str:
    """
    Apply steering by adding scaled decoder vector to activations.
    """
    # Get steering vector from SAE decoder
    steering_vector = sae.W_dec[feature_idx]  # shape: (d_model,)
    
    # Scale by strength
    scaled_vector = steering_strength * steering_vector
    
    # Create hook function
    def steering_hook(activations, hook):
        # activations shape: (batch, seq, d_model)
        # Add steering vector to all positions
        return activations + scaled_vector
    
    # Run with hook
    with model.hooks([(f"blocks.{layer}.hook_resid_post", steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_tokens)
    
    return output
```

**Algorithm 2: Multi-Feature Steering**
```python
def apply_multi_feature_steering(
    model: HookedTransformer,
    sae: SAE,
    feature_configs: List[Tuple[int, float]],  # [(idx, strength), ...]
    layer: int,
    prompt: str,
    max_tokens: int
) -> str:
    """
    Combine multiple features with individual strengths.
    """
    # Construct combined steering vector
    combined_vector = torch.zeros(sae.cfg.d_model)
    
    for feature_idx, strength in feature_configs:
        combined_vector += strength * sae.W_dec[feature_idx]
    
    # Apply combined vector
    def steering_hook(activations, hook):
        return activations + combined_vector
    
    with model.hooks([(f"blocks.{layer}.hook_resid_post", steering_hook)]):
        output = model.generate(prompt, max_new_tokens=max_tokens)
    
    return output
```

**Algorithm 3: Steering Comparison Batch**
```python
def generate_steering_comparison(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    configurations: List[SteeringConfig],
    num_samples: int = 5,
    generation_params: dict = None
) -> ComparisonResult:
    """
    Generate outputs for multiple steering configurations.
    """
    results = {
        'unsteered': [],
        'steered': {config.name: [] for config in configurations}
    }
    
    # Generate unsteered baseline
    for _ in range(num_samples):
        output = model.generate(prompt, **generation_params)
        results['unsteered'].append(output)
    
    # Generate steered versions
    for config in configurations:
        for _ in range(num_samples):
            output = apply_feature_steering(
                model, sae, 
                config.feature_idx,
                config.strength,
                config.layer,
                prompt,
                generation_params['max_tokens']
            )
            results['steered'][config.name].append(output)
    
    # Compute metrics
    metrics = compute_comparison_metrics(results)
    
    return ComparisonResult(
        outputs=results,
        metrics=metrics,
        config=configurations
    )
```

**Algorithm 4: Metric Computation**
```python
def compute_comparison_metrics(
    unsteered_outputs: List[str],
    steered_outputs: List[str],
    model: HookedTransformer,
    target_vocabulary: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for steering comparison.
    """
    metrics = {}
    
    # Perplexity
    metrics['perplexity'] = {
        'unsteered': compute_perplexity(model, unsteered_outputs),
        'steered': compute_perplexity(model, steered_outputs)
    }
    
    # KL Divergence
    metrics['kl_divergence'] = compute_kl_divergence(
        model, unsteered_outputs, steered_outputs
    )
    
    # Behavioral score (if vocabulary provided)
    if target_vocabulary:
        metrics['behavioral_score'] = compute_vocabulary_coverage(
            steered_outputs, target_vocabulary
        )
    
    # Coherence (using external judge or perplexity proxy)
    metrics['coherence'] = estimate_coherence(steered_outputs)
    
    return metrics
```

### 7.3 Data Models

**Steering Configuration**:
```python
@dataclass
class SteeringConfig:
    name: str
    sae_id: str
    feature_idx: int
    steering_strength: float
    layer: int
    method: str = "direct"  # "direct", "multi", "sae-ts"
    additional_params: Dict = field(default_factory=dict)
```

**Experiment**:
```python
@dataclass
class SteeringExperiment:
    id: str
    created_at: datetime
    model_name: str
    sae_id: str
    prompt: str
    configurations: List[SteeringConfig]
    generation_params: Dict
    results: ComparisonResult
    user_ratings: Optional[Dict] = None
    notes: str = ""
```

**Comparison Result**:
```python
@dataclass
class ComparisonResult:
    outputs: Dict[str, List[str]]  # {config_name: [outputs]}
    metrics: Dict[str, Any]
    config: List[SteeringConfig]
    timestamp: datetime
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export for analysis."""
        pass
    
    def export_json(self) -> str:
        """Export for sharing."""
        pass
```

---

## 8. API Specifications

### 8.1 REST API Endpoints

**Load SAE**
```
POST /api/steering/sae/load
Request:
{
  "source": "huggingface",  // or "local", "trained"
  "sae_id": "google/gemma-scope-2b-pt-res",
  "path": "layer_12/width_16k/canonical"
}

Response:
{
  "sae_id": "gemma-scope-layer12-16k",
  "n_features": 16384,
  "model_name": "gemma-2-2b",
  "layer": 12,
  "cached": true
}
```

**Get Features**
```
GET /api/steering/features?sae_id={sae_id}&limit=100&offset=0

Response:
{
  "features": [
    {
      "idx": 7650,
      "description": "Religious text",
      "activation_freq": 0.023,
      "max_activation": 12.5,
      "top_tokens": ["Bible", "prayer", "God"],
      "neuronpedia_url": "https://..."
    },
    ...
  ],
  "total": 16384
}
```

**Create Steering Experiment**
```
POST /api/steering/experiment
Request:
{
  "model_name": "gpt2-small",
  "sae_id": "gpt2-small-layer8",
  "prompt": "In the beginning,",
  "configurations": [
    {
      "name": "Religious",
      "feature_idx": 7650,
      "strength": 50.0,
      "layer": 8
    },
    {
      "name": "Scientific",
      "feature_idx": 12331,
      "strength": 30.0,
      "layer": 8
    }
  ],
  "generation_params": {
    "max_tokens": 100,
    "temperature": 0.7,
    "num_samples": 5
  }
}

Response:
{
  "experiment_id": "exp_abc123",
  "status": "running",
  "estimated_time": 45  // seconds
}
```

**Get Experiment Results**
```
GET /api/steering/experiment/{experiment_id}

Response:
{
  "experiment_id": "exp_abc123",
  "status": "complete",
  "results": {
    "unsteered": [...],
    "steered": {
      "Religious": [...],
      "Scientific": [...]
    },
    "metrics": {
      "perplexity": {...},
      "kl_divergence": {...},
      "behavioral_score": {...}
    }
  },
  "created_at": "2025-01-15T10:30:00Z"
}
```

### 8.2 WebSocket Events (Real-time Updates)

```javascript
// Connect to experiment
ws.send({
  type: "subscribe",
  experiment_id: "exp_abc123"
});

// Progress updates
{
  type: "progress",
  experiment_id: "exp_abc123",
  stage: "generating",
  config: "Religious",
  sample: 3,
  total_samples: 5,
  progress: 0.6
}

// Completion
{
  type: "complete",
  experiment_id: "exp_abc123",
  results: {...}
}
```

---

## 9. Implementation Phases

### Phase 1: MVP (4-6 weeks)
**Goal**: Basic steering with side-by-side comparison

**Features**:
- Load SAE from local training or HuggingFace
- Simple feature browser (list view)
- Single feature steering
- 2-column comparison (unsteered vs steered)
- Basic metrics (perplexity, length)
- Save/load experiments

**Success Criteria**:
- Can steer with any SAE feature
- Side-by-side view works
- Results reproducible

### Phase 2: Enhanced Comparison (3-4 weeks)
**Goal**: Multi-feature comparison and better evaluation

**Features**:
- Multi-feature steering (up to 4 simultaneous)
- 4-column comparison view
- Strength sweep mode
- Additional metrics (KL divergence, behavioral score)
- Visualization dashboard
- Export to CSV/JSON

**Success Criteria**:
- Can compare 4 configs simultaneously
- Metrics computed accurately
- Visualizations helpful for analysis

### Phase 3: Advanced Features (4-6 weeks)
**Goal**: Production-ready steering capabilities

**Features**:
- SAE-TS method implementation
- Feature activation tracking
- Token-level analysis
- Batch prompts
- Statistical significance testing
- Experiment versioning
- Team collaboration

**Success Criteria**:
- SAE-TS outperforms baseline
- Advanced users can debug steering issues
- Multiple users can collaborate

### Phase 4: Polish & Scale (2-3 weeks)
**Goal**: Performance and UX improvements

**Features**:
- Performance optimization
- Tutorial and documentation
- Example experiments
- Mobile-responsive design
- A/B testing framework
- Usage analytics

**Success Criteria**:
- <30s for typical comparison
- <5 min to first successful experiment
- Positive user feedback

---

## 10. Success Metrics & KPIs

### 10.1 Adoption Metrics
- **Primary**: % of SAE training users who try steering (Target: >50%)
- **Primary**: Avg experiments per active user per week (Target: >5)
- Time to first steering experiment (Target: <5 minutes)
- Feature utilization rate (Target: >70% of steering uses >1 feature)

### 10.2 Quality Metrics
- User-reported steering success rate (Target: >70%)
- Avg coherence score maintained (Target: >0.7)
- Feature selection accuracy (relevance to goal) (Target: >80%)

### 10.3 Engagement Metrics
- Experiment completion rate (Target: >85%)
- Repeat usage rate (weekly) (Target: >40%)
- Sharing rate (Target: >20%)
- HuggingFace SAE downloads via app (Target: >100/week)

### 10.4 Technical Metrics
- Generation latency p95 (Target: <45s)
- Error rate (Target: <2%)
- System uptime (Target: >99.5%)

---

## 11. Dependencies & Integrations

### 11.1 Required Libraries
- **SAELens**: Core SAE loading and manipulation
- **TransformerLens**: Model hooks and intervention
- **HuggingFace Hub**: SAE downloading
- **PyTorch**: Tensor operations
- **NumPy/Pandas**: Data manipulation
- **Plotly**: Visualizations

### 11.2 External Services
- **Neuronpedia API**: Feature interpretations
- **HuggingFace Hub**: SAE repository
- **GPT-4 API** (optional): Coherence scoring

### 11.3 Internal Dependencies
- SAE training module (for recently trained SAEs)
- Model management system
- User authentication
- Experiment database

---

## 12. Open Questions & Risks

### 12.1 Open Questions
1. **Steering Method Default**: Which method should be default? Direct vs SAE-TS?
2. **Computational Budget**: What's acceptable generation time for comparison?
3. **Feature Discovery**: How to help users find relevant features?
4. **Evaluation**: What automatic metrics correlate best with user goals?
5. **Multi-layer**: Should we support steering at multiple layers simultaneously?

### 12.2 Technical Risks

**Risk 1: Performance**
- **Issue**: Steering generation may be too slow for interactive use
- **Mitigation**: Batch processing, GPU optimization, progress indicators
- **Impact**: High

**Risk 2: SAE Quality**
- **Issue**: Poor SAE quality leads to bad steering results
- **Mitigation**: Provide quality indicators, validation checks
- **Impact**: Medium

**Risk 3: Feature Selection**
- **Issue**: Users struggle to find relevant features
- **Mitigation**: Search, recommendations, examples
- **Impact**: High

**Risk 4: Metric Reliability**
- **Issue**: Automatic metrics don't match user intent
- **Mitigation**: Provide multiple metrics, support human evaluation
- **Impact**: Medium

### 12.3 Product Risks

**Risk 1: Complexity**
- **Issue**: Too many options overwhelm users
- **Mitigation**: Progressive disclosure, smart defaults, wizards
- **Impact**: High

**Risk 2: Reproducibility**
- **Issue**: Steering results vary too much
- **Mitigation**: Fix random seeds, document all parameters
- **Impact**: Medium

**Risk 3: Adoption**
- **Issue**: Users don't understand value of steering
- **Mitigation**: Clear examples, use case documentation
- **Impact**: High

---

## 13. Future Enhancements (Post-V1)

### 13.1 Advanced Steering Methods
- Learned steering (optimization-based)
- Context-dependent steering
- Multi-modal steering (for vision-language models)
- Steering with confidence bounds

### 13.2 Analysis Tools
- Feature interaction analysis
- Steering effect attribution
- Cross-model steering transfer
- Steering vector decomposition

### 13.3 Automation
- Auto-suggest features based on goal
- Auto-tune steering strengths
- A/B test multiple configurations
- Continuous evaluation

### 13.4 Collaboration
- Shared steering libraries
- Community feature annotations
- Steering vector marketplace
- Team dashboards

---

## 14. Appendix

### 14.1 Glossary

**Steering Vector**: A direction in activation space used to modify model behavior
**Steering Strength (Î±)**: Scalar multiplier controlling intervention intensity
**Feature**: Interpretable direction learned by SAE corresponding to a concept
**Decoder Vector**: SAE's W_dec weights representing feature direction
**Behavioral Score**: Metric measuring achievement of steering objective
**Coherence Score**: Metric measuring output quality/sensibility
**L0**: Number of active features (sparsity measure)
**SAE-TS**: SAE-Targeted Steering method
**CAA**: Contrastive Activation Addition
**Hook**: TransformerLens mechanism for intervening on activations

### 14.2 References

**Key Papers**:
1. Templeton et al. (2024) - "Scaling and evaluating sparse autoencoders"
2. Chalnev et al. (2024) - "Improving Steering Vectors by Targeting SAE Features"
3. Zou et al. (2023) - "Representation Engineering"
4. Rimsky et al. (2024) - "Contrastive Activation Addition"

**Code References**:
- SAELens: github.com/decoderesearch/SAELens
- TransformerLens: github.com/neelnanda-io/TransformerLens
- SAE-TS: github.com/slavachalnev/SAE-TS

### 14.3 Example Use Cases

**Use Case 1: Safety Research**
- Goal: Reduce harmful outputs
- Method: Identify "harmful content" features, apply negative steering
- Success: Reduced harm score while maintaining helpfulness

**Use Case 2: Behavior Modification**
- Goal: Make model more creative
- Method: Identify "creativity" features, apply positive steering
- Success: Increased novelty metrics without gibberish

**Use Case 3: Debugging SAEs**
- Goal: Validate SAE feature quality
- Method: Steer with top features, check if outputs match interpretation
- Success: Confirmed 80% of features are causally relevant

---

## 15. Acceptance Criteria

### 15.1 Feature Complete
- âœ… All FR-1.x (Feature Selection) implemented
- âœ… All FR-2.x (Steering Configuration) implemented
- âœ… All FR-3.x (Generation & Comparison) implemented
- âœ… All FR-4.x (Evaluation & Metrics) implemented
- âœ… All NFR-1.x (Performance) requirements met

### 15.2 Quality Gates
- âœ… Unit test coverage >80%
- âœ… Integration tests pass
- âœ… User acceptance testing with 5+ researchers
- âœ… Performance benchmarks met
- âœ… Documentation complete

### 15.3 Launch Readiness
- âœ… Tutorial created and tested
- âœ… Example experiments available
- âœ… Error handling robust
- âœ… Monitoring in place
- âœ… Rollback plan documented

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-15  
**Author**: Product Team  
**Reviewers**: Engineering, Research, Design  
**Status**: Ready for Implementation