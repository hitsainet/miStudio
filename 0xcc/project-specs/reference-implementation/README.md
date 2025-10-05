# Reference Implementation - Mock UI

**Status:** This is the authoritative UI/UX specification
**Purpose:** The production MVP MUST look and behave exactly like this mock

---

## Critical Understanding

This is **NOT** a rough prototype or wireframe. This is a **complete, functional specification** for how the production application should look, feel, and behave.

### What This Means

1. **Visual Design**: Every color, spacing, border, shadow, and animation should match
2. **User Interactions**: Every button click, form submission, and state change should work the same way
3. **Data Flow**: The way data moves through components should match this structure
4. **Component Hierarchy**: The organization of components should be replicated
5. **API Contracts**: Backend APIs should match the documented patterns in comments

---

## File Contents

### Mock-embedded-interp-ui.tsx (4000+ lines)

This single file contains:

1. **Complete API Documentation** (Lines 1-305)
   - Technology stack recommendations
   - API design principles
   - Rate limiting patterns
   - Error handling patterns
   - Performance requirements
   - Database schema overview
   - WebSocket protocol
   - Security considerations

2. **Type Definitions** (Lines 306-672)
   - All TypeScript interfaces
   - Backend API contracts for each type
   - Validation constraints
   - Status transitions

3. **Main Application** (Lines 675-1104)
   - Tab navigation (Datasets, Models, Training, Features, Steering)
   - State management
   - Template/preset system

4. **Component Implementations**:
   - **DatasetsPanel** (Lines 1108-1202)
     - HuggingFace download form with token support
     - Dataset cards with status indicators
     - Dataset detail modal

   - **ModelsPanel** (Lines 1204-1343)
     - Model download form
     - Quantization selection
     - Model cards with download progress
     - "Extract Activations" button

   - **ModelArchitectureViewer** (Lines 1345-1437)
     - Layer-by-layer visualization
     - Model statistics
     - Configuration details

   - **ActivationExtractionConfig** (Lines 1439-1625)
     - Dataset selection
     - Layer selection (multi-select)
     - Activation type selection
     - Extraction progress tracking

   - **TrainingPanel** (Lines 1627-1842)
     - Dataset/Model/Encoder selection
     - **Advanced Hyperparameters** (collapsible):
       - Learning Rate
       - Batch Size
       - L1 Coefficient
       - Expansion Factor
       - Training Steps
       - Optimizer
       - LR Schedule
       - Ghost Gradient Penalty toggle
     - Training jobs list

   - **TrainingCard** (Lines 1844-2115)
     - Progress bar with real-time updates
     - Metrics display (Loss, L0 Sparsity, Dead Neurons, GPU Util)
     - Training controls (Pause, Resume, Stop)
     - Checkpoint management
     - Live metrics toggle

   - **FeaturesPanel** (Lines 2116+)
     - Training selection
     - Feature extraction trigger
     - Feature browser/table
     - Feature detail modal
     - Max activating examples

   - **SteeringPanel**
     - Model selection
     - Feature selection
     - Coefficient sliders
     - Generation controls
     - Comparative output display

### src/styles/mock-ui.css

Custom CSS properties for dynamic styling:

```css
.progress-bar-model { --width: 0%; }
.progress-bar-training { --width: 0%; }
.progress-bar-upload { --width: 0%; }
.chart-bar-emerald { --height: 0%; }
.chart-bar-blue { --height: 0%; }
.token-highlight-attention { --opacity: 0; }
.token-highlight-gradient { --strength: 0; }
```

These enable JavaScript to dynamically update progress bars and visualizations via inline styles.

---

## How to Use This as a Specification

### For Frontend Development

1. **Component Structure**
   ```
   EmbeddedInterpretabilityUI (main)
   ├── Header
   ├── Navigation Tabs
   └── Tab Content
       ├── DatasetsPanel
       │   ├── HF Download Form
       │   ├── Dataset Cards
       │   └── DatasetDetailModal
       ├── ModelsPanel
       │   ├── Model Download Form
       │   ├── Model Cards
       │   ├── ModelArchitectureViewer
       │   └── ActivationExtractionConfig
       ├── TrainingPanel
       │   ├── Configuration Form
       │   │   ├── Basic Config
       │   │   └── Advanced Hyperparameters (collapsible)
       │   └── TrainingCard (multiple)
       │       ├── Progress Display
       │       ├── Metrics Dashboard
       │       ├── Controls
       │       └── Checkpoint Manager
       ├── FeaturesPanel
       │   ├── Training Selector
       │   ├── Feature Browser
       │   └── Feature Detail Modal
       └── SteeringPanel
           ├── Feature Selector
           ├── Coefficient Controls
           └── Generation Interface
   ```

2. **Styling Approach**
   - **Framework**: Tailwind CSS (as used in mock)
   - **Dark Theme**: `bg-slate-950` (background), `bg-slate-900` (cards)
   - **Accent Color**: `emerald-400/500/600` for primary actions
   - **Text Colors**: `text-white` (primary), `text-slate-400` (secondary)
   - **Border Colors**: `border-slate-800/700`
   - **Hover States**: All interactive elements have hover states
   - **Transitions**: All state changes are animated (`transition-colors`, `transition-all`)

3. **State Management**
   - Use React hooks (useState, useEffect)
   - Global state can be added later (Redux, Zustand) but start with hooks
   - Expose key functions to window object for debugging (as in mock)

4. **Real-time Updates**
   - Training progress: Poll or WebSocket (mock uses polling every 2s)
   - Progress bars: Smooth animations with CSS transitions
   - Metrics: Update every 2-5 seconds during training

### For Backend Development

1. **API Endpoints**: Extract from comments in Mock-embedded-interp-ui.tsx
   - Lines 315-322: Datasets API
   - Lines 336-344: Models API
   - Lines 392-410: Training API
   - Lines 428-436: Checkpoints API
   - Lines 451-458: Features API
   - Lines 565-571: Steering API
   - Lines 605-611: Templates API

2. **Response Formats**: Match the TypeScript interfaces
   - Success: `{ "data": {...}, "meta": {...} }`
   - Error: `{ "error": { "code": "...", "message": "...", "details": {...} } }`

3. **Status Codes**: Follow the documented patterns (Lines 106-116)
   - 200: Success
   - 201: Created
   - 202: Accepted (long-running job)
   - 400: Bad Request
   - 404: Not Found
   - 429: Rate Limit
   - 500: Server Error
   - 503: Service Unavailable

4. **WebSocket Protocol**: Lines 227-263
   - Connection: `ws://api.example.com/ws/training/:trainingId`
   - Messages: `training.progress`, `training.completed`, `training.error`
   - Heartbeat: Ping/pong every 30s

### For Testing

1. **Visual Regression**: Screenshot comparison against mock UI
2. **Interaction Testing**: All user flows should match mock behavior
3. **State Transitions**: Follow the documented state machines (e.g., training status transitions)
4. **Error Handling**: Match the error display patterns in mock
5. **Loading States**: All async operations show loading indicators as in mock

---

## Key Features to Replicate

### 1. Advanced Hyperparameters Panel
- **Location**: Training tab, collapsible section
- **Behavior**: Click to expand/collapse, smooth animation
- **Fields**: 8 hyperparameter inputs with proper validation
- **Toggle**: Ghost Gradient Penalty has custom toggle component

### 2. Real-time Progress Tracking
- **Progress Bars**: Smooth animations, gradient colors
- **Metrics Cards**: 4 metrics displayed (Loss, L0 Sparsity, Dead Neurons, GPU Util)
- **Update Frequency**: Every 2 seconds for training jobs
- **Visual Feedback**: Pulsing animation on "training" status

### 3. Checkpoint Management
- **Save Button**: "Save Now" creates checkpoint immediately
- **Checkpoint List**: Scrollable list with Load/Delete actions
- **Auto-save**: Configurable interval with toggle
- **Display**: Shows step, loss, and timestamp for each checkpoint

### 4. Model Architecture Viewer
- **Modal Dialog**: Full-screen modal with close button
- **Layer List**: Scrollable list of all model layers
- **Statistics**: 4 stat cards (Total Layers, Hidden Dim, Attention Heads, Parameters)
- **Configuration**: Additional model config details at bottom

### 5. Activation Extraction
- **Dataset Selector**: Dropdown with sample counts
- **Layer Selector**: Multi-select grid with Select All/Deselect All
- **Activation Type**: Dropdown for residual/MLP/attention
- **Progress**: Live progress bar during extraction

### 6. HuggingFace Integration
- **Token Input**: Password field for access tokens
- **Usage Note**: Explanation text for when tokens are needed
- **Download Progress**: Progress bar with percentage
- **Status Indicators**: Icons for ready/downloading/error states

### 7. Template/Preset System
- **Training Templates**: Save/load training configs
- **Extraction Templates**: Save/load extraction settings
- **Steering Presets**: Save/load steering configurations
- **Favoriting**: Star icon to mark favorites

---

## Colors & Styling Reference

### Background Colors
```
bg-slate-950  - App background
bg-slate-900  - Card backgrounds
bg-slate-800  - Input backgrounds, secondary elements
bg-slate-700  - Hover states
```

### Text Colors
```
text-white        - Primary text
text-slate-300    - Secondary headings
text-slate-400    - Labels, descriptions
text-slate-500    - Disabled text
```

### Accent Colors
```
emerald-400/500/600  - Primary actions, success states
blue-400/500         - Info, datasets
purple-400/500/600   - Models, advanced features
red-400/500          - Errors, warnings
yellow-400           - Warnings, paused states
```

### Border Colors
```
border-slate-800  - Primary borders
border-slate-700  - Input borders, secondary borders
border-emerald-400/500  - Active/focused borders
```

---

## Animation Patterns

1. **Progress Bars**: `transition-all duration-500`
2. **Button Hovers**: `transition-colors`
3. **Card Hovers**: `transition-colors` or `transition-opacity`
4. **Collapse/Expand**: Smooth height transitions
5. **Loading Spinners**: `animate-spin`
6. **Pulse Animations**: `animate-pulse` for active states

---

## Responsive Behavior

The mock uses `max-w-7xl mx-auto px-6` for main content, which:
- Constrains width on large screens (max 80rem / 1280px)
- Centers content
- Provides padding on mobile

All grids use responsive classes:
- 2 columns: `grid-cols-2`
- 3 columns: `grid-cols-3`
- 4 columns: `grid-cols-4`
- 6 columns: `grid-cols-6` (for layer selection)

---

## Implementation Priority

When building the production version, maintain this exact UI/UX while:

1. **Phase 1**: Copy structure and styling exactly
2. **Phase 2**: Replace mock data with real API calls
3. **Phase 3**: Add WebSocket for real-time updates
4. **Phase 4**: Implement template/preset system
5. **Phase 5**: Add advanced features from Gap Closure Instructions

**Do not redesign the UI** - users expect this exact interface. Any improvements should be additive, not replacements.

---

## Questions?

- **"Can I change the layout?"** → No, keep it exactly as shown
- **"Can I use different colors?"** → No, use the exact color scheme
- **"Can I reorganize components?"** → No, keep the same structure
- **"Can I add features?"** → Yes, but only after replicating the mock exactly
- **"Can I use a different CSS framework?"** → Only if you can match the styling exactly

---

**Remember**: This mock UI is the product specification. The production MVP should be indistinguishable from this mock in terms of look, feel, and behavior.
