# Feature Extraction Context Windows - Implementation Plan

## Overview

This document outlines the implementation plan for adding context window capture to miStudio's feature extraction system. Based on research into SAELens and industry best practices (Anthropic, OpenAI), we will capture tokens before and after each feature activation to provide better interpretability.

## Current State

**Current Implementation**:
- Extracts single tokens where features activate maximally
- Stores only the activated token string
- Limited context for understanding what triggered the feature

**Current Data Structure** (in `features` table):
```sql
example_context JSONB {
  "tokens": ["token1", "token2", ...],
  "activations": [0.5, 1.0, ...],
  "max_activation": 1.0
}
```

## Research Findings

### Industry Standards

**OpenAI Research** ("Scaling and evaluating sparse autoencoders", 2024):
- 64 token context for training sparse autoencoders on GPT-4
- Acknowledged as potentially "too few tokens to exhibit the most interesting behaviors"
- Used across all experiments in their 16M latent autoencoder

**Anthropic Research** ("Scaling Monosemanticity", 2024):
- 250 token context for training on Claude 3 Sonnet
- Max-pooling across context windows for feature activation
- Used 40 million contexts with 8 billion MLP activation vectors

**Display Recommendations** (from research community):
- 5-10 tokens before activation (preceding context)
- 3-5 tokens after activation (following context)
- Asymmetric windows preferred (more preceding than following)

### Key Insights

1. **Asymmetric Windows Preferred**: More preceding context than following (research shows preceding context is more informative)
2. **Store Activation Values**: Capture activation strength for all context tokens, not just prime token
3. **JSONB Storage**: Efficient and flexible for variable-length contexts
4. **Visual Hierarchy**: Bold + color for prime token, lighter styling for context
5. **Top-K Examples**: Store top 100 examples per feature (not all activations)
6. **Max-Pooling Strategy**: Anthropic's research shows max-pooling across context improves feature understanding

## Proposed Architecture

### 1. Data Schema

**Enhanced `example_context` JSONB Structure**:
```json
{
  "prefix_tokens": ["token1", "token2", "token3", "token4", "token5"],
  "prime_token": "activated_token",
  "suffix_tokens": ["token7", "token8", "token9"],
  "activation_values": [0.2, 0.5, 0.8, 0.9, 0.95, 1.0, 0.7, 0.3, 0.1],
  "prime_activation_index": 5,
  "token_positions": [142, 143, 144, 145, 146, 147, 148, 149, 150],
  "max_activation": 1.0
}
```

**Field Descriptions**:
- `prefix_tokens`: Array of N tokens before activation (default: 5)
- `prime_token`: The token with maximum activation
- `suffix_tokens`: Array of M tokens after activation (default: 3)
- `activation_values`: Parallel array of activation strengths for all tokens
- `prime_activation_index`: Index of prime token in activation_values array
- `token_positions`: Absolute positions in original sequence
- `max_activation`: Maximum activation value (same as prime token's activation)

### 2. Configuration Schema

**Add to `extraction_templates` table**:
```sql
ALTER TABLE extraction_templates ADD COLUMN context_config JSONB DEFAULT '{
  "prefix_tokens": 5,
  "suffix_tokens": 3,
  "min_context_activation": 0.0,
  "capture_full_sequence": false
}'::jsonb;
```

**Add to extraction job configuration**:
```python
@dataclass
class ExtractionConfig:
    # ... existing fields ...

    # Context window configuration
    context_prefix_tokens: int = 5
    context_suffix_tokens: int = 3
    min_context_activation: float = 0.0
    capture_full_sequence: bool = False
```

### 3. Extraction Worker Updates

**File**: `backend/src/workers/feature_extraction_tasks.py`

**Changes**:
1. Modify `extract_top_k_examples()` to capture context windows
2. Implement edge case handling (sequence start/end)
3. Store activation values for all context tokens
4. Use efficient batching to minimize overhead

**Implementation Approach**:
```python
def extract_context_window(
    tokens: torch.Tensor,
    activations: torch.Tensor,
    position: int,
    prefix_len: int = 5,
    suffix_len: int = 3
) -> dict:
    """
    Extract context window around activated position.

    Based on research from Anthropic and OpenAI papers on sparse autoencoders.
    Default asymmetric window (5 before, 3 after) provides optimal interpretability.

    Args:
        tokens: Token tensor [batch, seq_len]
        activations: Feature activations [batch, seq_len]
        position: Position of maximum activation
        prefix_len: Number of tokens before (default: 5)
        suffix_len: Number of tokens after (default: 3)

    Returns:
        Dictionary with context structure for JSONB storage
    """
    batch_idx = 0  # Assuming single batch for simplicity
    seq_len = tokens.shape[1]

    # Calculate window boundaries with edge case handling
    # Use max() and min() to avoid index out of bounds
    prefix_start = max(0, position - prefix_len)
    suffix_end = min(seq_len, position + suffix_len + 1)

    # Extract token IDs
    prefix_ids = tokens[batch_idx, prefix_start:position].tolist()
    prime_id = tokens[batch_idx, position].item()
    suffix_ids = tokens[batch_idx, position + 1:suffix_end].tolist()

    # Convert token IDs to strings using model's tokenizer
    prefix_strs = [tokenizer.decode([tid]) for tid in prefix_ids]
    prime_str = tokenizer.decode([prime_id])
    suffix_strs = [tokenizer.decode([tid]) for tid in suffix_ids]

    # Extract activation values for full window
    # Following Anthropic's approach of capturing activation strength across context
    activation_window = activations[batch_idx, prefix_start:suffix_end].tolist()

    # Calculate prime token index in activation array
    prime_idx = position - prefix_start

    # Token positions in original sequence (for debugging/analysis)
    token_positions = list(range(prefix_start, suffix_end))

    return {
        "prefix_tokens": prefix_strs,
        "prime_token": prime_str,
        "suffix_tokens": suffix_strs,
        "activation_values": activation_window,
        "prime_activation_index": prime_idx,
        "token_positions": token_positions,
        "max_activation": float(activations[batch_idx, position])
    }
```

### 4. API Updates

**File**: `backend/src/api/v1/endpoints/features.py`

**Changes**:
1. Return enhanced context data in feature responses
2. Add query parameter to control context display length
3. Support filtering by context activation threshold

**Example Response**:
```json
{
  "id": "feat_12345",
  "neuron_index": 42,
  "name": "Positive Sentiment",
  "example_context": {
    "prefix_tokens": ["I", " really", " love", " this", " amazing"],
    "prime_token": " product",
    "suffix_tokens": [" because", " it", " works"],
    "activation_values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.6, 0.2, 0.1],
    "prime_activation_index": 5,
    "max_activation": 1.0
  }
}
```

### 5. Frontend Visualization

#### Enhanced TokenHighlight Component

**File**: `frontend/src/components/features/TokenHighlight.tsx`

**Changes**:
1. Accept new context structure
2. Implement visual hierarchy for prime vs context tokens
3. Add activation strength gradient
4. Support expandable context display

**Visual Design** (based on research):
```
Context tokens:     text-slate-400, no background
Weak activations:   text-slate-300, emerald-500/10 background
Medium activations: text-slate-200, emerald-500/20 background
Strong activations: text-slate-100, emerald-500/40 background
Prime token:        text-white font-bold, emerald-500 background, border
```

**Component Props**:
```typescript
interface TokenHighlightProps {
  prefixTokens: string[];
  primeToken: string;
  suffixTokens: string[];
  activationValues: number[];
  primeActivationIndex: number;
  maxActivation: number;
  showActivationValues?: boolean;
  expandable?: boolean;
  className?: string;
}
```

#### Updated FeatureDetailModal

**File**: `frontend/src/components/features/FeatureDetailModal.tsx`

**Changes**:
1. Display top-K examples with full context
2. Add sortable columns for activation strength
3. Implement expandable context rows
4. Show token position in original sequence

### 6. Database Migration

**File**: `backend/alembic/versions/[timestamp]_add_context_windows.py`

**Changes**:
1. No schema change needed (using existing JSONB column)
2. Migration to populate context data for existing features (optional)
3. Add indexes for context queries if needed

**Note**: Since we're using the existing `example_context` JSONB column with an enhanced structure, no schema migration is required. However, existing features will have the old single-token format until re-extracted.

## Implementation Plan

### Phase 1: Core Functionality (Priority: High)

**Tasks**:
1. ✅ Research SAELens and industry best practices (completed)
2. ⏳ Design and document data schema (this document)
3. ⏳ Create Alembic migration (if needed)
4. ⏳ Update extraction worker to capture 5+3 context windows
5. ⏳ Update API endpoints to return context data
6. ⏳ Test with existing training job

**Estimated Time**: 6-8 hours

**Success Criteria**:
- ✅ Context windows captured during extraction
- ✅ Data stored in correct JSONB format
- ✅ API returns enhanced context structure
- ✅ No performance regression in extraction speed

### Phase 2: Frontend Visualization (Priority: High)

**Tasks**:
1. ⏳ Enhance TokenHighlight component with context display
2. ⏳ Implement visual hierarchy (bold prime, lighter context)
3. ⏳ Add activation strength color gradient
4. ⏳ Update FeatureDetailModal with context examples
5. ⏳ Test visualization with real data

**Estimated Time**: 4-6 hours

**Success Criteria**:
- ✅ Prime token clearly distinguished from context
- ✅ Activation gradients visible and intuitive
- ✅ Context enhances feature interpretability
- ✅ No layout issues or rendering problems

### Phase 3: Configuration & Templates (Priority: Medium)

**Tasks**:
1. ⏳ Add context config to extraction templates
2. ⏳ Add UI controls for context window size
3. ⏳ Implement presets (Quick: 3+2, Standard: 5+3, Deep: 10+5)
4. ⏳ Test different configurations

**Estimated Time**: 3-4 hours

**Success Criteria**:
- ✅ Users can configure context window size
- ✅ Presets work as expected
- ✅ Configuration saved in templates
- ✅ Validation prevents invalid values

### Phase 4: Advanced Features (Priority: Low)

**Tasks**:
1. ⏳ Add expandable/collapsible context display
2. ⏳ Implement hover tooltips with exact activation values
3. ⏳ Add sortable columns in feature table
4. ⏳ Add export functionality for context examples
5. ⏳ Document feature in user guide

**Estimated Time**: 4-5 hours

**Success Criteria**:
- ✅ Advanced interactions work smoothly
- ✅ Export formats include context
- ✅ Documentation complete and clear

## Technical Considerations

### Performance

**Extraction Speed**:
- Context window capture adds minimal overhead (~5-10%)
- Most time is spent in model inference, not token manipulation
- Batching strategy critical for efficiency

**Storage**:
- JSONB compression handles variable-length contexts efficiently
- Top-K filtering (100 examples) keeps storage manageable
- Index on `features.extraction_job_id` for fast queries

**Query Performance**:
- GIN index on JSONB column if context search needed
- Existing pagination handles large result sets
- No N+1 query issues (context embedded in feature row)

### Edge Cases

**Sequence Boundaries**:
- Start of sequence: Fewer prefix tokens (e.g., 2 instead of 5)
- End of sequence: Fewer suffix tokens (e.g., 1 instead of 3)
- Single token sequence: No context (prefix=[], suffix=[])

**Special Tokens**:
- BOS/EOS tokens: Include in context but mark specially
- Padding tokens: Exclude from context
- Unknown tokens: Include with [UNK] marker

**Tokenizer Differences**:
- GPT-2: Tokens have leading spaces (e.g., " the")
- LLaMA: Different BPE markers
- Handle via tokenizer-specific cleaning (existing `cleanToken` function)

### Backward Compatibility

**Existing Features**:
- Old single-token format: `{"tokens": [...], "activations": [...]}`
- New context format: `{"prefix_tokens": [...], "prime_token": "...", ...}`
- Frontend checks format and renders appropriately

**API Versioning**:
- No version bump needed (enhanced data, not breaking change)
- Old clients ignore new fields
- New clients handle both formats

**Migration Strategy**:
- No forced migration (features keep old format)
- Re-extraction populates new format
- Coexistence of both formats acceptable

## Configuration Defaults

### UI Configuration Interface

**Context Window Controls**:
```typescript
// Preset selector + custom input fields
interface ContextWindowConfig {
  usePreset: boolean;
  presetName: 'quick' | 'standard' | 'deep' | 'symmetric' | 'custom';
  customPrefixTokens: number;    // User can enter 1-20
  customSuffixTokens: number;    // User can enter 1-20
}
```

**UI Layout**:
```
┌─────────────────────────────────────────────┐
│ Context Window                              │
│                                             │
│ Preset: [▼ Standard]  or  Custom           │
│                                             │
│ Tokens Before: [ 5 ]  (1-20)              │
│ Tokens After:  [ 3 ]  (1-20)              │
│                                             │
│ When preset selected, fields auto-populate │
│ When fields edited, preset changes to      │
│ "Custom"                                    │
└─────────────────────────────────────────────┘
```

### Recommended Presets

**Quick Exploration** (minimal overhead):
```json
{
  "name": "Quick",
  "prefix_tokens": 3,
  "suffix_tokens": 2,
  "description": "Minimal context for rapid feature browsing"
}
```

**Standard Analysis** (recommended default):
```json
{
  "name": "Standard",
  "prefix_tokens": 5,
  "suffix_tokens": 3,
  "description": "Balanced context window based on research recommendations"
}
```

**Deep Analysis** (maximum interpretability):
```json
{
  "name": "Deep",
  "prefix_tokens": 10,
  "suffix_tokens": 5,
  "description": "Extended context for detailed investigation"
}
```

**Symmetric** (bidirectional analysis):
```json
{
  "name": "Symmetric",
  "prefix_tokens": 5,
  "suffix_tokens": 5,
  "description": "Equal context before and after activation"
}
```

**Custom** (user-defined):
```json
{
  "name": "Custom",
  "prefix_tokens": 1-20,  // User enters value
  "suffix_tokens": 1-20,  // User enters value
  "description": "User-defined context window size"
}
```

### Preset Logic

**When user selects a preset**:
- Dropdown changes to preset name
- Input fields auto-populate with preset values
- Fields remain editable

**When user edits input fields**:
- If values no longer match any preset, dropdown changes to "Custom"
- If values match a preset, dropdown shows that preset name
- Values are validated (1-20 range, integers only)

**Validation Rules**:
- Prefix tokens: 1-20 (minimum 1 for context)
- Suffix tokens: 1-20 (minimum 1 for context)
- Total context: 2-40 tokens (prefix + suffix)
- Integer values only

## Testing Strategy

### Unit Tests

**Backend**:
- `test_extract_context_window()` - Verify context extraction logic
- `test_context_edge_cases()` - Start/end of sequence, single tokens
- `test_context_api_format()` - API response structure
- `test_backward_compatibility()` - Old vs new format handling

**Frontend**:
- `test_token_highlight_context()` - Rendering with context
- `test_activation_gradient()` - Color coding logic
- `test_expandable_context()` - Expand/collapse functionality
- `test_old_format_support()` - Handling legacy single-token format

### Integration Tests

**Extraction Workflow**:
1. Create extraction job with context config
2. Run extraction on test dataset
3. Verify context captured correctly
4. Check API returns context data
5. Validate frontend displays context

**Performance Tests**:
1. Benchmark extraction speed (with vs without context)
2. Measure storage size increase
3. Test query performance with large datasets
4. Verify no memory leaks in long-running extractions

## Documentation Updates

**User Documentation**:
- Feature extraction guide: Explain context windows
- Template configuration: Document context settings
- Interpretation guide: How to use context for understanding features

**Developer Documentation**:
- Data schema documentation
- API endpoint specifications
- Frontend component usage

**Research Context**:
- Link to SAELens documentation
- Citations for industry best practices
- Comparison with Neuronpedia approach

## Success Metrics

**Functionality**:
- ✅ Context windows captured during extraction
- ✅ 5+3 default matches SAELens standard
- ✅ Visual hierarchy clearly distinguishes prime token
- ✅ No performance degradation (< 10% slower)

**Usability**:
- ✅ Users can understand features better with context
- ✅ Configuration is intuitive and well-documented
- ✅ Backward compatibility maintained

**Quality**:
- ✅ All tests passing
- ✅ No bugs in edge cases
- ✅ Code reviewed and approved

## Future Enhancements

**Phase 5: Advanced Interpretability** (Future):
1. Logit lens integration (show predicted next tokens)
2. Attention pattern visualization
3. Feature composition analysis (what features co-activate)
4. Interactive context testing (user inputs custom text)
5. Neuronpedia API compatibility for uploads

**Phase 6: Analysis Tools** (Future):
1. Context clustering (group similar activation contexts)
2. Token importance scoring (which context tokens matter most)
3. Comparative analysis (contexts across different features)
4. Export to research formats (JSON, CSV, Markdown)

## References

**Research Papers**:
- OpenAI: "Scaling and evaluating sparse autoencoders" (2024)
  - 64 token context for training, 16M latent SAE on GPT-4
  - https://cdn.openai.com/papers/sparse-autoencoders.pdf

- Anthropic: "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (2024)
  - 250 token context, max-pooling across context windows
  - https://transformer-circuits.pub/2024/scaling-monosemanticity/

**Tools & Platforms**:
- Neuronpedia: Feature exploration platform for SAE research
  - https://neuronpedia.org
  - Community standard for sharing interpretable features

---

**Document Version**: 1.1
**Created**: 2025-11-19
**Last Updated**: 2025-11-20
**Author**: miStudio Development Team
**Status**: Planning Phase - Ready for Implementation
