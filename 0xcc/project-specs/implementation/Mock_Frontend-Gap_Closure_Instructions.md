# MechInterp Studio (miStudio): Frontend Gap Closure Implementation Plan

## Document Status & Context

**Last Updated:** 2025-10-05
**Status:** Active implementation roadmap
**Current Phase:** Foundation complete, ready for feature implementation

### Recent Updates & Current State

This document has been synchronized with recent project refinements:

- **Code Quality Improvements (2025-10-05)**: All TypeScript warnings and critical errors have been resolved in the mock UI. Type safety has been improved with proper union type narrowing, CSS module type declarations, and component prop alignment.
- **HuggingFace Integration Enhancement (2025-10-05)**: Added token authentication support for gated datasets in the DatasetsPanel, enabling access to restricted HuggingFace datasets.
- **Development Environment**: Mock UI is running successfully at `http://localhost:3000/` with hot module replacement (HMR) working correctly.

### Document Purpose

This implementation plan provides a **comprehensive, step-by-step roadmap** to transform the current mock UI into a fully functional MechInterp Studio frontend. The plan is based on:

1. **Gap analysis** between the mock UI and the technical specification
2. **Prioritized phases** (P0-P3) for systematic feature implementation
3. **Detailed implementation guidance** with code examples and API contracts
4. **Backend API requirements** for each frontend feature

### Related Documentation

This document should be read in conjunction with:

#### Core Specifications

- **[miStudio_Specification.md](./miStudio_Specification.md)** - Comprehensive technical workflow specification describing the system architecture across four phases: Dataset Management Pipeline, Model Loading & Activation Extraction, Autoencoder Training, and Feature Discovery & Analysis. This document defines the core requirements and workflows that the UI must support.

- **[Gap_Analysis_MockUI-vs-Technical_Specfication.md](./Gap_Analysis_MockUI-vs-Technical_Specfication.md)** - Detailed analysis comparing the current mock UI implementation against the technical specification. Identifies what's implemented well and what's missing across all phases.

#### Infrastructure Specifications

- **[000_SPEC|REDIS_GUIDANCE_USECASE.md](./000_SPEC|REDIS_GUIDANCE_USECASE.md)** - Redis implementation guide for job queues (Celery/BullMQ), WebSocket pub/sub for real-time training updates, rate limiting, and distributed coordination. Critical for understanding how training progress updates are streamed to the frontend.

- **[001_SPEC|Folder_File_Details.md](./001_SPEC|Folder_File_Details.md)** - Complete folder and file structure specification for the production application with `/backend` and `/frontend` separation. Defines the target production structure for component organization and project layout.

- **[003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)** - PostgreSQL database schema specification including all tables (datasets, models, trainings, training_metrics, checkpoints, features, feature_activations, templates). Documents the backend data model that the frontend must interact with via API endpoints.

- **[004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md)** - Template/preset system specification for saving and reusing training configurations, extraction configurations, and steering presets. Defines the data structures and workflows for the template system that should be integrated into the UI.

#### Implementation Files

- **[src/types/training.types.ts](./src/types/training.types.ts)** - TypeScript type definitions and API contract documentation for the training system. Includes interfaces for `Hyperparameters`, `TrainingMetrics`, `Training`, `Checkpoint`, and `TrainingWebSocketMessage` types. This file documents the expected backend API endpoints and data structures.

- **[Mock-embedded-interp-ui.tsx](./Mock-embedded-interp-ui.tsx)** - Current mock UI implementation. This is the main file that will be enhanced following the steps in this plan.

### How to Use This Document

1. **Phases are ordered by priority**: P0 (MVP) features should be implemented first, followed by P1, P2, and P3.
2. **Each step is self-contained**: Steps include "What to Add", "Implementation Details", and "Backend API Required" sections.
3. **Checkmarks indicate planning structure**: The ✅ marks in the Implementation Summary section indicate the recommended implementation order, not completion status.
4. **Code examples are templates**: The provided code snippets should be adapted to match the existing code style and structure in the mock UI.
5. **Backend coordination required**: Most features require corresponding backend API endpoints as documented in each step.

### Implementation Philosophy

- **Incremental enhancement**: Build features one at a time, ensuring each works before moving to the next
- **Type safety first**: All new features should use TypeScript interfaces from `src/types/`
- **Consistent UX patterns**: Follow the existing design patterns in the mock UI (tab navigation, card layouts, progress indicators)
- **Real-time updates**: Use WebSockets for streaming updates where specified in the API contracts
- **Edge device awareness**: Keep performance considerations in mind for resource-constrained deployment targets
- **Component organization**: When transitioning from mock to production, follow the structure defined in [001_SPEC|Folder_File_Details.md](./001_SPEC|Folder_File_Details.md#frontend-structure) for organizing components into `common/`, `datasets/`, `models/`, `training/`, `features/`, and `steering/` directories

---

## Overview

Based on the comprehensive review of the mock UI, technical specification, and gap analysis, this plan provides a systematic approach to close all identified gaps. The plan prioritizes by criticality and builds features incrementally.

---

# **Comprehensive Plan to Close All UI Gaps**

## **Priority Matrix Overview**

**P0 (Must Have for MVP):** Critical workflow features - 8 major updates
**P1 (Essential for Usability):** Important enhancements - 6 major updates  
**P2 (Production Ready):** Polish and advanced features - 5 major updates
**P3 (Nice to Have):** Future enhancements - 4 major updates

---

# **PHASE 1: Training Infrastructure (P0 - Sprint 1)**
**Goal:** Complete the training pipeline with full hyperparameter control and real-time monitoring

**Specification Reference:** This phase implements features described in [miStudio_Specification.md - Phase 3: Autoencoder Training](./miStudio_Specification.md#phase-3-autoencoder-training)

**Type Definitions:** All training-related interfaces are defined in [src/types/training.types.ts](./src/types/training.types.ts), including:
- `Hyperparameters` interface with validation constraints
- `TrainingMetrics` interface for real-time monitoring
- `Training` interface for job state management
- `Checkpoint` interface for checkpoint management
- `TrainingWebSocketMessage` types for real-time updates
- Backend API contract documentation

**Infrastructure Requirements:**
- **Redis Pub/Sub** ([000_SPEC|REDIS_GUIDANCE_USECASE.md](./000_SPEC|REDIS_GUIDANCE_USECASE.md#use-case-2-websocket-pubsub-for-real-time-updates)) - Real-time training metrics streaming from GPU workers to WebSocket server to frontend
- **PostgreSQL Tables** ([003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)) - `trainings`, `training_metrics`, `checkpoints` tables store job state and time-series data
- **Template System** ([004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md)) - Training templates allow saving and reusing hyperparameter configurations

## **Step 1.1: Add Training Hyperparameters Panel**

### **What to Add:**
Create an expandable "Advanced Configuration" section in the Training tab

### **Implementation Details:**
```tsx
<TrainingConfigurationAdvanced>
  <CollapsibleSection title="Hyperparameters">
    // Learning Rate
    <NumberInput 
      label="Learning Rate" 
      default={1e-4} 
      min={1e-6} 
      max={1e-2}
      step="log"
    />
    
    // Batch Size
    <NumberInput 
      label="Batch Size" 
      default={256} 
      options={[64, 128, 256, 512, 1024, 2048]}
    />
    
    // L1 Coefficient (Sparsity)
    <NumberInput 
      label="L1 Coefficient (λ)" 
      default={1e-3} 
      min={1e-5} 
      max={1e-1}
    />
    
    // Expansion Factor
    <SelectInput 
      label="Expansion Factor" 
      default="8x"
      options={["4x", "8x", "16x", "32x"]}
    />
    
    // Training Steps
    <NumberInput 
      label="Training Steps" 
      default={10000} 
      min={1000} 
      max={100000}
    />
    
    // Optimizer
    <SelectInput 
      label="Optimizer" 
      default="AdamW"
      options={["Adam", "AdamW", "SGD"]}
    />
    
    // Learning Rate Schedule
    <SelectInput 
      label="LR Schedule" 
      default="cosine"
      options={["constant", "linear", "cosine", "exponential"]}
    />
    
    // Ghost Gradient Penalty
    <Toggle 
      label="Ghost Gradient Penalty" 
      default={true}
    />
  </CollapsibleSection>
</TrainingConfigurationAdvanced>
```

### **Backend API Required:**
```javascript
POST /api/training/start
{
  "model_id": "m1",
  "dataset_id": "ds1",
  "encoder_type": "sparse",
  "hyperparameters": {
    "learning_rate": 1e-4,
    "batch_size": 256,
    "l1_coefficient": 1e-3,
    "expansion_factor": 8,
    "training_steps": 10000,
    "optimizer": "AdamW",
    "lr_schedule": "cosine",
    "ghost_grad_penalty": true
  }
}
```

---

## **Step 1.2: Real-Time Training Metrics Dashboard**

### **What to Add:**
Live visualization of training metrics using Recharts

### **Implementation Details:**
```tsx
<TrainingMetricsDashboard>
  {/* Metrics Summary Cards */}
  <MetricsGrid>
    <MetricCard 
      title="Current Loss" 
      value={currentLoss} 
      trend="down"
      color="emerald"
    />
    <MetricCard 
      title="L0 Sparsity" 
      value={l0Sparsity} 
      target={50}
      color="blue"
    />
    <MetricCard 
      title="Dead Neurons" 
      value={deadNeurons} 
      total={totalNeurons}
      color="red"
    />
    <MetricCard 
      title="GPU Utilization" 
      value={gpuUtil} 
      unit="%"
      color="purple"
    />
  </MetricsGrid>
  
  {/* Live Charts */}
  <ChartsSection className="grid grid-cols-2 gap-4">
    {/* Loss Curve */}
    <LineChart 
      data={lossHistory}
      xAxis="step"
      yAxis="loss"
      title="Reconstruction Loss"
      color="#10b981"
    />
    
    {/* Sparsity Over Time */}
    <LineChart 
      data={sparsityHistory}
      xAxis="step"
      yAxis="l0"
      title="L0 Sparsity"
      color="#3b82f6"
    />
    
    {/* Activation Frequency Histogram */}
    <BarChart 
      data={activationFreqDist}
      xAxis="neuron_bins"
      yAxis="count"
      title="Activation Distribution"
      color="#8b5cf6"
    />
    
    {/* Dead Neuron Tracking */}
    <LineChart 
      data={deadNeuronHistory}
      xAxis="step"
      yAxis="count"
      title="Dead Neurons Over Time"
      color="#ef4444"
    />
  </ChartsSection>
</TrainingMetricsDashboard>
```

### **WebSocket Integration:**
```javascript
useEffect(() => {
  const ws = new WebSocket('ws://localhost:8000/api/training/metrics');
  
  ws.onmessage = (event) => {
    const metrics = JSON.parse(event.data);
    updateMetrics(metrics);
  };
  
  return () => ws.close();
}, [trainingId]);
```

### **Backend API Required:**
```javascript
// WebSocket endpoint
WS /api/training/{training_id}/metrics

// Returns every N steps:
{
  "step": 1500,
  "loss": 0.0234,
  "l0_sparsity": 47.3,
  "dead_neurons": 156,
  "total_neurons": 8192,
  "gpu_utilization": 87.5,
  "learning_rate": 9.8e-5
}
```

---

## **Step 1.3: Training Logs Viewer**

### **What to Add:**
Terminal-style log viewer for training output

### **Implementation Details:**
```tsx
<TrainingLogsTerminal>
  <div className="bg-slate-950 rounded-lg p-4 font-mono text-xs">
    <div className="flex items-center justify-between mb-2">
      <span className="text-slate-400">Training Logs</span>
      <button 
        onClick={clearLogs}
        className="text-slate-500 hover:text-slate-300"
      >
        Clear
      </button>
    </div>
    
    <div className="h-64 overflow-y-auto space-y-1">
      {logs.map((log, i) => (
        <div 
          key={i}
          className={`${
            log.level === 'error' ? 'text-red-400' :
            log.level === 'warning' ? 'text-yellow-400' :
            'text-slate-300'
          }`}
        >
          <span className="text-slate-500">[{log.timestamp}]</span> {log.message}
        </div>
      ))}
    </div>
    
    <button className="mt-2 text-emerald-400 text-xs">
      Auto-scroll {autoScroll ? '✓' : ''}
    </button>
  </div>
</TrainingLogsTerminal>
```

### **Backend API Required:**
```javascript
// WebSocket or SSE for streaming logs
WS /api/training/{training_id}/logs

// Or polling endpoint
GET /api/training/{training_id}/logs?since={timestamp}
```

---

## **Step 1.4: Training Controls (Pause/Resume/Stop)**

### **What to Add:**
Control buttons for managing active training jobs

### **Implementation Details:**
```tsx
<TrainingControls trainingId={training.id} status={training.status}>
  {status === 'training' && (
    <>
      <button onClick={() => pauseTraining(trainingId)}>
        <Pause className="w-4 h-4" />
        Pause
      </button>
      <button onClick={() => stopTraining(trainingId)}>
        <StopCircle className="w-4 h-4" />
        Stop
      </button>
    </>
  )}
  
  {status === 'paused' && (
    <button onClick={() => resumeTraining(trainingId)}>
      <Play className="w-4 h-4" />
      Resume
    </button>
  )}
  
  {status === 'failed' && (
    <button onClick={() => retryTraining(trainingId)}>
      <RotateCw className="w-4 h-4" />
      Retry
    </button>
  )}
</TrainingControls>
```

### **Backend API Required:**
```javascript
POST /api/training/{training_id}/pause
POST /api/training/{training_id}/resume
POST /api/training/{training_id}/stop
POST /api/training/{training_id}/retry
```

---

## **Step 1.5: Checkpoint Management**

### **What to Add:**
Interface to save, load, and manage training checkpoints

### **Implementation Details:**
```tsx
<CheckpointManager trainingId={training.id}>
  <div className="space-y-3">
    {/* Save Checkpoint Button */}
    <button 
      onClick={saveCheckpoint}
      className="w-full"
    >
      <Save className="w-4 h-4" />
      Save Checkpoint
    </button>
    
    {/* Checkpoint List */}
    <div className="space-y-2">
      <h4 className="text-sm font-medium">Saved Checkpoints</h4>
      {checkpoints.map(cp => (
        <div key={cp.id} className="flex items-center justify-between bg-slate-800/30 p-3 rounded">
          <div>
            <div className="font-medium">Step {cp.step}</div>
            <div className="text-xs text-slate-400">
              Loss: {cp.loss.toFixed(4)} • {formatDate(cp.timestamp)}
            </div>
          </div>
          <div className="flex gap-2">
            <button onClick={() => loadCheckpoint(cp.id)}>
              <Download className="w-4 h-4" />
            </button>
            <button onClick={() => deleteCheckpoint(cp.id)}>
              <Trash className="w-4 h-4" />
            </button>
          </div>
        </div>
      ))}
    </div>
    
    {/* Auto-save Settings */}
    <div className="border-t border-slate-700 pt-3">
      <Toggle 
        label="Auto-save every N steps"
        value={autoSave}
        onChange={setAutoSave}
      />
      {autoSave && (
        <NumberInput 
          value={autoSaveInterval}
          onChange={setAutoSaveInterval}
          min={100}
          max={10000}
        />
      )}
    </div>
  </div>
</CheckpointManager>
```

### **Backend API Required:**
```javascript
POST /api/training/{training_id}/checkpoint/save
GET /api/training/{training_id}/checkpoints
POST /api/training/{training_id}/checkpoint/{checkpoint_id}/load
DELETE /api/training/{training_id}/checkpoint/{checkpoint_id}
```

---

# **PHASE 2: Feature Discovery Core (P0 - Sprint 2)**
**Goal:** Build complete feature extraction, browsing, and analysis capabilities

**Specification Reference:** This phase implements features described in [miStudio_Specification.md - Phase 4: Feature Discovery & Analysis](./miStudio_Specification.md#phase-4-feature-discovery--analysis)

**Gap Analysis:** See [Gap_Analysis_MockUI-vs-Technical_Specfication.md - Phase 4: Feature Discovery](./Gap_Analysis_MockUI-vs-Technical_Specfication.md#phase-4-feature-discovery--analysis) for detailed analysis of missing features in this area

**Infrastructure Requirements:**
- **PostgreSQL Tables** ([003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)) - `features`, `feature_activations` tables store discovered features and their activation examples with JSONB support for flexible data structures
- **Extraction Templates** ([004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md#2-extraction-templates)) - Reusable configurations for feature extraction targeting specific layers and hook types

## **Step 2.1: Feature Extraction Interface**

### **What to Add:**
Button and progress UI to extract features after training completes

### **Implementation Details:**
```tsx
<FeatureExtractionPanel training={completedTraining}>
  <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
    <h3 className="font-semibold mb-4">Extract Features</h3>
    
    {!training.featuresExtracted ? (
      <>
        <p className="text-sm text-slate-400 mb-4">
          Training complete. Extract interpretable features from the trained encoder.
        </p>
        
        <div className="space-y-3 mb-4">
          <NumberInput 
            label="Evaluation Samples"
            value={evalSamples}
            onChange={setEvalSamples}
            default={10000}
          />
          
          <NumberInput 
            label="Top-K Examples per Feature"
            value={topK}
            onChange={setTopK}
            default={100}
          />
        </div>
        
        <button 
          onClick={startFeatureExtraction}
          className="w-full bg-emerald-600"
        >
          <Zap className="w-4 h-4" />
          Extract Features
        </button>
      </>
    ) : extractionStatus === 'extracting' ? (
      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span>Extracting features...</span>
          <span className="text-emerald-400">{extractionProgress}%</span>
        </div>
        <ProgressBar progress={extractionProgress} />
        <p className="text-xs text-slate-400">
          Processing activation patterns...
        </p>
      </div>
    ) : (
      <div className="flex items-center gap-2 text-emerald-400">
        <CheckCircle className="w-5 h-5" />
        <span>Features extracted successfully</span>
      </div>
    )}
  </div>
</FeatureExtractionPanel>
```

### **Backend API Required:**
```javascript
POST /api/training/{training_id}/extract-features
{
  "eval_samples": 10000,
  "top_k_examples": 100
}

// Returns job ID, poll for status
GET /api/training/{training_id}/extraction-status
{
  "status": "extracting",
  "progress": 45.3,
  "features_found": 2048
}
```

---

## **Step 2.2: Feature Browser/Table**

### **What to Add:**
Searchable, sortable table of all discovered features

### **Implementation Details:**
```tsx
<FeatureBrowser trainingId={training.id}>
  <div className="space-y-4">
    {/* Search and Filters */}
    <div className="flex gap-3">
      <input
        type="text"
        placeholder="Search features..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      />
      
      <select 
        value={sortBy}
        onChange={(e) => setSortBy(e.target.value)}
        className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        <option value="activation_freq">Activation Frequency</option>
        <option value="interpretability">Interpretability Score</option>
        <option value="feature_id">Feature ID</option>
      </select>
      
      <button onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}>
        {sortOrder === 'asc' ? <ArrowUp /> : <ArrowDown />}
      </button>
    </div>
    
    {/* Feature Table */}
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead className="bg-slate-800/50">
          <tr>
            <th className="px-4 py-3 text-left">ID</th>
            <th className="px-4 py-3 text-left">Label</th>
            <th className="px-4 py-3 text-right">Activation Freq</th>
            <th className="px-4 py-3 text-right">Interpretability</th>
            <th className="px-4 py-3 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {filteredFeatures.map(feature => (
            <tr 
              key={feature.id}
              onClick={() => openFeatureDetail(feature.id)}
              className="border-t border-slate-800 hover:bg-slate-800/30 cursor-pointer"
            >
              <td className="px-4 py-3 font-mono text-sm">
                {feature.id}
              </td>
              <td className="px-4 py-3">
                {feature.label || <span className="text-slate-500">Unlabeled</span>}
              </td>
              <td className="px-4 py-3 text-right">
                <span className="text-emerald-400">
                  {(feature.activation_freq * 100).toFixed(2)}%
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <span className="text-blue-400">
                  {(feature.interpretability * 100).toFixed(1)}%
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <button onClick={(e) => { e.stopPropagation(); toggleFavorite(feature.id); }}>
                  <Star className={feature.favorited ? "fill-yellow-400" : ""} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    
    {/* Pagination */}
    <Pagination 
      currentPage={page}
      totalPages={totalPages}
      onPageChange={setPage}
    />
  </div>
</FeatureBrowser>
```

### **Backend API Required:**
```javascript
GET /api/training/{training_id}/features?
  search={query}&
  sort_by={field}&
  sort_order={asc|desc}&
  page={n}&
  per_page={50}

Response:
{
  "features": [...],
  "total": 2048,
  "page": 1,
  "per_page": 50
}
```

---

## **Step 2.3: Feature Detail Modal (CRITICAL)**

### **What to Add:**
Comprehensive view of individual feature with all analysis tools

### **Implementation Details:**
```tsx
<FeatureDetailModal featureId={selectedFeature} onClose={closeModal}>
  <div className="max-w-6xl w-full bg-slate-900 rounded-lg">
    {/* Header */}
    <div className="border-b border-slate-800 p-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-2xl font-bold">Feature #{feature.id}</div>
          <input
            type="text"
            value={feature.label}
            onChange={(e) => updateFeatureLabel(feature.id, e.target.value)}
            placeholder="Add label..."
            className="mt-2 bg-slate-800 px-3 py-1 rounded"
          />
        </div>
        <button onClick={onClose}>
          <X className="w-6 h-6" />
        </button>
      </div>
      
      {/* Feature Stats */}
      <div className="grid grid-cols-4 gap-4 mt-4">
        <StatCard title="Activation Frequency" value={`${(feature.activation_freq * 100).toFixed(2)}%`} />
        <StatCard title="Interpretability" value={`${(feature.interpretability * 100).toFixed(1)}%`} />
        <StatCard title="Max Activation" value={feature.max_activation.toFixed(3)} />
        <StatCard title="Active Samples" value={feature.active_samples} />
      </div>
    </div>
    
    {/* Tabs */}
    <div className="border-b border-slate-800">
      <div className="flex px-6">
        {['examples', 'logit-lens', 'correlations', 'ablation'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveFeatureTab(tab)}
            className={`px-4 py-3 border-b-2 capitalize ${
              activeFeatureTab === tab
                ? 'border-emerald-400 text-emerald-400'
                : 'border-transparent text-slate-400'
            }`}
          >
            {tab.replace('-', ' ')}
          </button>
        ))}
      </div>
    </div>
    
    {/* Content */}
    <div className="p-6 max-h-[60vh] overflow-y-auto">
      {activeFeatureTab === 'examples' && (
        <MaxActivatingExamples 
          examples={feature.max_activating_examples}
          featureId={feature.id}
        />
      )}
      
      {activeFeatureTab === 'logit-lens' && (
        <LogitLensView 
          topTokens={feature.logit_lens_top_tokens}
          probabilities={feature.logit_lens_probs}
        />
      )}
      
      {activeFeatureTab === 'correlations' && (
        <FeatureCorrelations 
          correlatedFeatures={feature.correlated_features}
          onFeatureClick={openFeatureDetail}
        />
      )}
      
      {activeFeatureTab === 'ablation' && (
        <AblationAnalysis 
          perplexityDelta={feature.ablation_perplexity_delta}
          impactScore={feature.ablation_impact}
        />
      )}
    </div>
  </div>
</FeatureDetailModal>
```

---

## **Step 2.4: Max Activating Examples Viewer (CRITICAL)**

### **What to Add:**
Display tokens with gradient-based highlighting showing activation intensity

### **Implementation Details:**
```tsx
<MaxActivatingExamples examples={examples} featureId={featureId}>
  <div className="space-y-4">
    <div className="flex items-center justify-between mb-4">
      <h4 className="font-semibold">Top Activating Contexts</h4>
      <div className="text-sm text-slate-400">
        Showing {examples.length} examples
      </div>
    </div>
    
    {examples.map((example, idx) => (
      <div key={idx} className="bg-slate-800/30 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-500">
            Example #{idx + 1}
          </span>
          <span className="text-xs text-emerald-400">
            Max activation: {example.max_activation.toFixed(3)}
          </span>
        </div>
        
        {/* Token Sequence with Highlighting */}
        <div className="flex flex-wrap gap-1 font-mono text-sm">
          {example.tokens.map((token, tokenIdx) => {
            const activation = example.activations[tokenIdx];
            const intensity = Math.min(activation / example.max_activation, 1.0);
            
            return (
              <span
                key={tokenIdx}
                className="px-1 py-0.5 rounded relative group"
                style={{
                  backgroundColor: `rgba(16, 185, 129, ${intensity * 0.3})`,
                  color: intensity > 0.5 ? '#fff' : '#e2e8f0'
                }}
              >
                {token}
                
                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block">
                  <div className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs whitespace-nowrap">
                    Activation: {activation.toFixed(4)}
                  </div>
                </div>
              </span>
            );
          })}
        </div>
        
        {/* Context Source */}
        {example.source && (
          <div className="mt-2 text-xs text-slate-500">
            Source: {example.source}
          </div>
        )}
      </div>
    ))}
  </div>
</MaxActivatingExamples>
```

### **Backend API Required:**
```javascript
GET /api/features/{feature_id}/examples?top_k=100

Response:
{
  "feature_id": 1337,
  "examples": [
    {
      "tokens": ["The", "cat", "sat", "on", "the", "mat"],
      "activations": [0.01, 0.05, 0.12, 0.03, 0.02, 0.08],
      "max_activation": 0.12,
      "source": "dataset_sample_4521"
    },
    ...
  ]
}
```

---

## **Step 2.5: Logit Lens Analysis**

### **What to Add:**
Show what the model "thinks" the feature represents by decoding through unembedding

### **Implementation Details:**
```tsx
<LogitLensView topTokens={topTokens} probabilities={probs}>
  <div className="space-y-4">
    <div className="mb-4">
      <h4 className="font-semibold mb-2">Predicted Tokens</h4>
      <p className="text-sm text-slate-400">
        If this feature alone determined the next token, these would be most likely:
      </p>
    </div>
    
    <div className="space-y-2">
      {topTokens.map((token, idx) => (
        <div 
          key={idx}
          className="flex items-center gap-3 bg-slate-800/30 rounded-lg p-3"
        >
          <div className="w-8 text-center text-slate-500 text-sm">
            #{idx + 1}
          </div>
          
          <div className="flex-1">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono font-medium">"{token}"</span>
              <span className="text-emerald-400 text-sm">
                {(probs[idx] * 100).toFixed(2)}%
              </span>
            </div>
            
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400"
                style={{ width: `${probs[idx] * 100}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
    
    {/* Interpretation Note */}
    <div className="mt-4 p-4 bg-blue-900/20 border border-blue-800/30 rounded-lg">
      <div className="flex gap-2">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />
        <div className="text-sm text-slate-300">
          <strong>Interpretation:</strong> This feature appears to represent 
          <span className="text-blue-400"> {interpretationHint}</span>
        </div>
      </div>
    </div>
  </div>
</LogitLensView>
```

### **Backend API Required:**
```javascript
GET /api/features/{feature_id}/logit-lens

Response:
{
  "feature_id": 1337,
  "top_tokens": ["the", "a", "an", "this", "that"],
  "probabilities": [0.23, 0.18, 0.15, 0.12, 0.09],
  "interpretation_hint": "definite and indefinite articles"
}
```

---

## **Step 2.6: Feature Activation Heatmap**

### **What to Add:**
Visual heatmap showing activation patterns across sequences

### **Implementation Details:**
```tsx
<ActivationHeatmap 
  featureId={featureId}
  sequences={sequences}
  activations={activations}
>
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h4 className="font-semibold">Activation Heatmap</h4>
      <select 
        value={colorScale}
        onChange={(e) => setColorScale(e.target.value)}
        className="px-3 py-1 bg-slate-800 rounded text-sm"
      >
        <option value="viridis">Viridis</option>
        <option value="plasma">Plasma</option>
        <option value="green">Green (default)</option>
      </select>
    </div>
    
    {/* Heatmap Grid */}
    <div className="overflow-x-auto">
      <div className="inline-block min-w-full">
        {sequences.map((seq, seqIdx) => (
          <div key={seqIdx} className="flex mb-1">
            {/* Sequence Label */}
            <div className="w-20 flex items-center text-xs text-slate-500 pr-2">
              Seq {seqIdx + 1}
            </div>
            
            {/* Token Cells */}
            <div className="flex gap-0.5">
              {seq.tokens.map((token, tokenIdx) => {
                const activation = activations[seqIdx][tokenIdx];
                const intensity = activation / maxActivation;
                
                return (
                  <div
                    key={tokenIdx}
                    className="group relative"
                    style={{
                      width: '32px',
                      height: '32px',
                      backgroundColor: getHeatmapColor(intensity, colorScale),
                      cursor: 'pointer'
                    }}
                  >
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                      <div className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs whitespace-nowrap">
                        <div>Token: {token}</div>
                        <div>Activation: {activation.toFixed(4)}</div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
    
    {/* Color Scale Legend */}
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400">Low</span>
      <div className="flex-1 h-4 rounded" style={{
        background: `linear-gradient(to right, ${getColorScaleGradient(colorScale)})`
      }} />
      <span className="text-xs text-slate-400">High</span>
    </div>
  </div>Continuing from where I left off...

---

</ActivationHeatmap>
```

### **Backend API Required:**
```javascript
GET /api/features/{feature_id}/heatmap?sequences=20

Response:
{
  "feature_id": 1337,
  "sequences": [
    {
      "tokens": ["The", "cat", "sat", ...],
      "activations": [0.01, 0.05, 0.12, ...]
    },
    ...
  ],
  "max_activation": 0.87
}
```

---

# **PHASE 3: Steering Tool (P0 - Sprint 3)**
**Goal:** Build complete steering interface - the most critical missing component

**Specification Reference:** This phase implements the steering/intervention capabilities described in [miStudio_Specification.md - Phase 5: Steering & Intervention](./miStudio_Specification.md#phase-5-steering--intervention)

**Note:** The steering interface is the most significant gap in the current mock UI and is essential for demonstrating the practical value of mechanistic interpretability

**Infrastructure Requirements:**
- **Steering Presets** ([004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md#3-steering-presets)) - Save and load feature steering configurations including feature IDs, coefficients, and intervention layers
- **PostgreSQL Tables** ([003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)) - `steering_presets` table stores saved steering configurations for quick reuse

## **Step 3.1: Add "Steering" Tab**

### **What to Add:**
New tab in main navigation for steering interface

### **Implementation Details:**
```tsx
// In main navigation array, add:
{['datasets', 'models', 'training', 'features', 'steering'].map(tab => (
  <button
    key={tab}
    onClick={() => setActiveTab(tab)}
    className={`py-4 px-2 border-b-2 transition-colors capitalize ${
      activeTab === tab
        ? 'border-emerald-400 text-emerald-400'
        : 'border-transparent text-slate-400 hover:text-slate-300'
    }`}
  >
    {tab}
  </button>
))}

// In main content area:
{activeTab === 'steering' && (
  <SteeringPanel trainings={completedTrainings} />
)}
```

---

## **Step 3.2: Feature Selection for Steering**

### **What to Add:**
Multi-select interface with coefficient sliders for each selected feature

### **Implementation Details:**
```tsx
<FeatureSelector trainingId={selectedTraining}>
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h3 className="text-xl font-semibold">Select Features to Steer</h3>
      <button 
        onClick={clearSelectedFeatures}
        className="text-sm text-slate-400 hover:text-slate-200"
      >
        Clear All
      </button>
    </div>
    
    {/* Feature Search */}
    <div className="relative">
      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
      <input
        type="text"
        placeholder="Search features to add..."
        value={featureSearch}
        onChange={(e) => setFeatureSearch(e.target.value)}
        className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      />
    </div>
    
    {/* Search Results Dropdown */}
    {featureSearch && (
      <div className="absolute z-10 w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg shadow-lg max-h-60 overflow-y-auto">
        {filteredFeatures.map(feature => (
          <button
            key={feature.id}
            onClick={() => addFeatureToSteering(feature)}
            className="w-full px-4 py-2 text-left hover:bg-slate-800 flex items-center justify-between"
          >
            <span>
              <span className="font-mono text-sm text-slate-400">#{feature.id}</span>
              {' '}
              {feature.label || 'Unlabeled'}
            </span>
            <Plus className="w-4 h-4 text-emerald-400" />
          </button>
        ))}
      </div>
    )}
    
    {/* Selected Features with Sliders */}
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-slate-300">
        Selected Features ({selectedFeatures.length})
      </h4>
      
      {selectedFeatures.length === 0 ? (
        <div className="text-center py-8 text-slate-400">
          No features selected. Search and add features above.
        </div>
      ) : (
        selectedFeatures.map(feature => (
          <div 
            key={feature.id}
            className="bg-slate-800/30 rounded-lg p-4 space-y-3"
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="font-mono text-sm text-slate-400">#{feature.id}</span>
                <span className="ml-2 font-medium">{feature.label}</span>
              </div>
              <button 
                onClick={() => removeFeatureFromSteering(feature.id)}
                className="text-red-400 hover:text-red-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            {/* Coefficient Slider */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Coefficient</span>
                <span className="text-emerald-400 font-mono">
                  {steeringCoefficients[feature.id].toFixed(2)}
                </span>
              </div>
              
              <input
                type="range"
                min="-5"
                max="5"
                step="0.1"
                value={steeringCoefficients[feature.id]}
                onChange={(e) => updateCoefficient(feature.id, parseFloat(e.target.value))}
                className="w-full accent-emerald-500"
              />
              
              <div className="flex justify-between text-xs text-slate-500">
                <span>-5.0 (suppress)</span>
                <span>0.0</span>
                <span>+5.0 (amplify)</span>
              </div>
            </div>
            
            {/* Quick Presets */}
            <div className="flex gap-2">
              <button 
                onClick={() => updateCoefficient(feature.id, -2.0)}
                className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600"
              >
                Suppress
              </button>
              <button 
                onClick={() => updateCoefficient(feature.id, 0.0)}
                className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600"
              >
                Reset
              </button>
              <button 
                onClick={() => updateCoefficient(feature.id, 2.0)}
                className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600"
              >
                Amplify
              </button>
            </div>
          </div>
        ))
      )}
    </div>
    
    {/* Steering Vector Summary */}
    {selectedFeatures.length > 0 && (
      <div className="bg-blue-900/20 border border-blue-800/30 rounded-lg p-4">
        <div className="text-sm">
          <div className="font-medium text-blue-300 mb-1">Steering Vector</div>
          <div className="text-slate-300">
            Manipulating {selectedFeatures.length} feature{selectedFeatures.length !== 1 ? 's' : ''}
          </div>
          <div className="text-xs text-slate-400 mt-1">
            L2 norm: {calculateSteeringVectorNorm().toFixed(3)}
          </div>
        </div>
      </div>
    )}
  </div>
</FeatureSelector>
```

---

## **Step 3.3: Generation Controls & Configuration**

### **What to Add:**
Prompt input, generation parameters, and layer selection

### **Implementation Details:**
```tsx
<GenerationControls>
  <div className="space-y-4">
    {/* Model Selection */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Model
      </label>
      <select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        {availableModels.map(model => (
          <option key={model.id} value={model.id}>{model.name}</option>
        ))}
      </select>
    </div>
    
    {/* Prompt Input */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Prompt
      </label>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter your prompt here..."
        rows={4}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg resize-none focus:outline-none focus:border-emerald-500"
      />
      <div className="flex items-center justify-between mt-1">
        <span className="text-xs text-slate-500">
          {prompt.length} characters
        </span>
        <button 
          onClick={loadExamplePrompt}
          className="text-xs text-emerald-400 hover:text-emerald-300"
        >
          Load Example
        </button>
      </div>
    </div>
    
    {/* Intervention Layer */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Intervention Layer
      </label>
      <select
        value={interventionLayer}
        onChange={(e) => setInterventionLayer(parseInt(e.target.value))}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        {modelLayers.map(layer => (
          <option key={layer} value={layer}>
            Layer {layer} {layer === Math.floor(modelLayers.length / 2) && '(middle)'}
          </option>
        ))}
      </select>
      <p className="text-xs text-slate-500 mt-1">
        Layer where steering vector will be added to residual stream
      </p>
    </div>
    
    {/* Generation Parameters */}
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Temperature
        </label>
        <input
          type="number"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          min="0"
          max="2"
          step="0.1"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Top-p
        </label>
        <input
          type="number"
          value={topP}
          onChange={(e) => setTopP(parseFloat(e.target.value))}
          min="0"
          max="1"
          step="0.05"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Max Tokens
        </label>
        <input
          type="number"
          value={maxTokens}
          onChange={(e) => setMaxTokens(parseInt(e.target.value))}
          min="1"
          max="2048"
          step="1"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Random Seed
        </label>
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(parseInt(e.target.value))}
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
    </div>
    
    {/* Generate Button */}
    <button
      onClick={generateComparison}
      disabled={!prompt || selectedFeatures.length === 0 || isGenerating}
      className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
    >
      {isGenerating ? (
        <>
          <Loader className="w-5 h-5 animate-spin" />
          Generating...
        </>
      ) : (
        <>
          <Zap className="w-5 h-5" />
          Generate Comparison
        </>
      )}
    </button>
  </div>
</GenerationControls>
```

---

## **Step 3.4: Comparative Output Display (CRITICAL)**

### **What to Add:**
Side-by-side view of unsteered vs steered generation with diff highlighting

### **Implementation Details:**
```tsx
<ComparativeOutput 
  unsteered={unsteeredOutput} 
  steered={steeredOutput}
  isGenerating={isGenerating}
>
  <div className="grid grid-cols-2 gap-6">
    {/* Unsteered Output */}
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-slate-400" />
          Unsteered (Baseline)
        </h3>
        {unsteeredOutput && (
          <button 
            onClick={() => copyToClipboard(unsteeredOutput)}
            className="text-sm text-slate-400 hover:text-slate-200"
          >
            <Copy className="w-4 h-4" />
          </button>
        )}
      </div>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 min-h-[300px]">
        {isGenerating ? (
          <div className="flex items-center justify-center h-full">
            <Loader className="w-6 h-6 animate-spin text-slate-400" />
          </div>
        ) : unsteeredOutput ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <div className="text-slate-300 whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {unsteeredOutput}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            No generation yet
          </div>
        )}
      </div>
      
      {unsteeredOutput && (
        <div className="text-xs text-slate-400">
          {unsteeredOutput.split(' ').length} words • 
          {unsteeredOutput.length} characters
        </div>
      )}
    </div>
    
    {/* Steered Output */}
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          Steered
        </h3>
        {steeredOutput && (
          <button 
            onClick={() => copyToClipboard(steeredOutput)}
            className="text-sm text-slate-400 hover:text-slate-200"
          >
            <Copy className="w-4 h-4" />
          </button>
        )}
      </div>
      
      <div className="bg-slate-900/50 border border-emerald-800/30 rounded-lg p-4 min-h-[300px]">
        {isGenerating ? (
          <div className="flex items-center justify-center h-full">
            <Loader className="w-6 h-6 animate-spin text-emerald-400" />
          </div>
        ) : steeredOutput ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <div className="text-slate-300 whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {renderDiffHighlighting(unsteeredOutput, steeredOutput)}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            No generation yet
          </div>
        )}
      </div>
      
      {steeredOutput && (
        <div className="text-xs text-slate-400">
          {steeredOutput.split(' ').length} words • 
          {steeredOutput.length} characters
        </div>
      )}
    </div>
  </div>
  
  {/* View Mode Toggle */}
  {unsteeredOutput && steeredOutput && (
    <div className="mt-6 flex items-center justify-center gap-4">
      <button
        onClick={() => setViewMode('side-by-side')}
        className={`px-4 py-2 rounded-lg ${
          viewMode === 'side-by-side'
            ? 'bg-emerald-600 text-white'
            : 'bg-slate-800 text-slate-300'
        }`}
      >
        Side by Side
      </button>
      <button
        onClick={() => setViewMode('unified-diff')}
        className={`px-4 py-2 rounded-lg ${
          viewMode === 'unified-diff'
            ? 'bg-emerald-600 text-white'
            : 'bg-slate-800 text-slate-300'
        }`}
      >
        Unified Diff
      </button>
      <button
        onClick={() => setViewMode('word-diff')}
        className={`px-4 py-2 rounded-lg ${
          viewMode === 'word-diff'
            ? 'bg-emerald-600 text-white'
            : 'bg-slate-800 text-slate-300'
        }`}
      >
        Word-level Diff
      </button>
    </div>
  )}
</ComparativeOutput>
```

### **Diff Highlighting Function:**
```tsx
function renderDiffHighlighting(baseline, steered) {
  const baselineWords = baseline.split(' ');
  const steeredWords = steered.split(' ');
  
  return steeredWords.map((word, idx) => {
    const isDifferent = baselineWords[idx] !== word;
    
    return (
      <span
        key={idx}
        className={isDifferent ? 'bg-emerald-500/30 border-b-2 border-emerald-400' : ''}
      >
        {word}{' '}
      </span>
    );
  });
}
```

---

## **Step 3.5: Steering Metrics Display**

### **What to Add:**
Quantitative comparison metrics between steered and unsteered outputs

### **Implementation Details:**
```tsx
<SteeringMetrics 
  unsteered={unsteeredOutput}
  steered={steeredOutput}
  metrics={comparisonMetrics}
>
  <div className="mt-6 bg-slate-900/50 border border-slate-800 rounded-lg p-6">
    <h3 className="text-lg font-semibold mb-4">Comparison Metrics</h3>
    
    <div className="grid grid-cols-4 gap-4">
      {/* KL Divergence */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <div className="text-xs text-slate-400 mb-1">KL Divergence</div>
        <div className="text-2xl font-bold text-purple-400">
          {metrics.kl_divergence?.toFixed(4) || '—'}
        </div>
        <div className="text-xs text-slate-500 mt-1">
          Distribution shift
        </div>
      </div>
      
      {/* Perplexity Change */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <div className="text-xs text-slate-400 mb-1">Perplexity Δ</div>
        <div className={`text-2xl font-bold ${
          metrics.perplexity_delta > 0 ? 'text-red-400' : 'text-emerald-400'
        }`}>
          {metrics.perplexity_delta > 0 ? '+' : ''}
          {metrics.perplexity_delta?.toFixed(2) || '—'}
        </div>
        <div className="text-xs text-slate-500 mt-1">
          {metrics.perplexity_delta > 0 ? 'Higher' : 'Lower'} uncertainty
        </div>
      </div>
      
      {/* Semantic Similarity */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <div className="text-xs text-slate-400 mb-1">Similarity</div>
        <div className="text-2xl font-bold text-blue-400">
          {(metrics.semantic_similarity * 100)?.toFixed(1) || '—'}%
        </div>
        <div className="text-xs text-slate-500 mt-1">
          Cosine similarity
        </div>
      </div>
      
      {/* Word Overlap */}
      <div className="bg-slate-800/50 rounded-lg p-4">
        <div className="text-xs text-slate-400 mb-1">Word Overlap</div>
        <div className="text-2xl font-bold text-emerald-400">
          {(metrics.word_overlap * 100)?.toFixed(1) || '—'}%
        </div>
        <div className="text-xs text-slate-500 mt-1">
          Shared tokens
        </div>
      </div>
    </div>
    
    {/* Probability Distribution Comparison */}
    {metrics.token_probability_shifts && (
      <div className="mt-6">
        <h4 className="text-sm font-medium text-slate-300 mb-3">
          Token Probability Shifts
        </h4>
        <div className="space-y-2">
          {metrics.token_probability_shifts.slice(0, 10).map((shift, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <span className="font-mono text-sm w-24">"{shift.token}"</span>
              <div className="flex-1 flex items-center gap-2">
                <div className="flex-1 h-6 bg-slate-700 rounded overflow-hidden flex">
                  <div 
                    className="bg-slate-500"
                    style={{ width: `${shift.baseline_prob * 100}%` }}
                    title={`Baseline: ${(shift.baseline_prob * 100).toFixed(2)}%`}
                  />
                </div>
                <ArrowRight className="w-4 h-4 text-slate-500" />
                <div className="flex-1 h-6 bg-slate-700 rounded overflow-hidden flex">
                  <div 
                    className="bg-emerald-500"
                    style={{ width: `${shift.steered_prob * 100}%` }}
                    title={`Steered: ${(shift.steered_prob * 100).toFixed(2)}%`}
                  />
                </div>
              </div>
              <span className={`text-xs font-mono ${
                shift.delta > 0 ? 'text-emerald-400' : 'text-red-400'
              }`}>
                {shift.delta > 0 ? '+' : ''}{(shift.delta * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
</SteeringMetrics>
```

### **Backend API Required:**
```javascript
POST /api/steering/generate
{
  "model_id": "m1",
  "prompt": "The cat sat on the",
  "intervention_layer": 12,
  "features": [
    {"feature_id": 42, "coefficient": 2.5},
    {"feature_id": 137, "coefficient": -1.0}
  ],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 100,
  "seed": 42
}

Response:
{
  "unsteered_output": "...",
  "steered_output": "...",
  "metrics": {
    "kl_divergence": 0.0234,
    "perplexity_delta": -2.3,
    "semantic_similarity": 0.87,
    "word_overlap": 0.65,
    "token_probability_shifts": [...]
  }
}
```

---

## **Step 3.6: Steering Presets/Saved Configurations**

### **What to Add:**
Ability to save and load steering configurations for quick reuse

### **Implementation Details:**
```tsx
<SteeringPresets>
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h3 className="text-lg font-semibold">Saved Steering Presets</h3>
      <button
        onClick={saveCurrentConfig}
        className="px-4 py-2 bg-emerald-600 rounded-lg flex items-center gap-2"
      >
        <Save className="w-4 h-4" />
        Save Current
      </button>
    </div>
    
    <div className="grid gap-3">
      {presets.map(preset => (
        <div 
          key={preset.id}
          className="bg-slate-800/30 rounded-lg p-4 flex items-center justify-between"
        >
          <div>
            <div className="font-medium">{preset.name}</div>
            <div className="text-sm text-slate-400">
              {preset.features.length} features • 
              Layer {preset.intervention_layer}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => loadPreset(preset)}
              className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600"
            >
              Load
            </button>
            <button
              onClick={() => deletePreset(preset.id)}
              className="px-3 py-1 bg-red-900/30 text-red-400 rounded hover:bg-red-900/50"
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  </div>
</SteeringPresets>
```

---

# **PHASE 4: Dataset Management Enhancements (P1 - Sprint 4)**
**Goal:** Add advanced dataset curation and preprocessing capabilities

**Specification Reference:** This phase implements features described in [miStudio_Specification.md - Phase 1: Dataset Management Pipeline](./miStudio_Specification.md#phase-1-dataset-management-pipeline)

**Recent Enhancement:** HuggingFace token authentication support was added (2025-10-05) for gated datasets, partially addressing this phase's requirements

**Infrastructure Requirements:**
- **PostgreSQL Tables** ([003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md)) - `datasets` table stores metadata including file paths, tokenization status, statistics, and full-text search capabilities
- **Job Queue** ([000_SPEC|REDIS_GUIDANCE_USECASE.md](./000_SPEC|REDIS_GUIDANCE_USECASE.md#use-case-1-job-queue-backend)) - Dataset download and processing operations run as Celery tasks with Redis as the message broker

## **Step 4.1: Dataset Detail Modal**

### **What to Add:**
Detailed view with tokenization settings, sample browser, and statistics

### **Implementation Details:**
```tsx
<DatasetDetailModal dataset={selectedDataset} onClose={closeModal}>
  <div className="max-w-6xl w-full bg-slate-900 rounded-lg">
    {/* Header */}
    <div className="border-b border-slate-800 p-6">
      <h2 className="text-2xl font-bold">{dataset.name}</h2>
      <p className="text-slate-400 mt-1">
        {dataset.size} • {dataset.num_samples} samples
      </p>
    </div>
    
    {/* Tabs */}
    <div className="border-b border-slate-800">
      <div className="flex px-6">
        {['overview', 'samples', 'tokenization', 'statistics'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveDatasetTab(tab)}
            className={`px-4 py-3 border-b-2 capitalize ${
              activeDatasetTab === tab
                ? 'border-emerald-400 text-emerald-400'
                : 'border-transparent text-slate-400'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>
    </div>
    
    {/* Content */}
    <div className="p-6">
      {activeDatasetTab === 'overview' && (
        <DatasetOverview dataset={dataset} />
      )}
      
      {activeDatasetTab === 'samples' && (
        <DatasetSampleBrowser 
          datasetId={dataset.id}
          samples={dataset.samples}
        />
      )}
      
      {activeDatasetTab === 'tokenization' && (
        <TokenizationSettings 
          datasetId={dataset.id}
          currentSettings={dataset.tokenization_config}
        />
      )}
      
      {activeDatasetTab === 'statistics' && (
        <DatasetStatistics 
          datasetId={dataset.id}
          stats={dataset.statistics}
        />
      )}
    </div>
  </div>
</DatasetDetailModal>
```

---

## **Step 4.2: Tokenization Configuration Panel**

### **What to Add:**
Interface to configure tokenization before processing

### **Implementation Details:**
```tsx
<TokenizationSettings datasetId={datasetId} currentSettings={settings}>
  <div className="space-y-6">
    <div>
      <h3 className="font-semibold mb-4">Tokenization Configuration</h3>
      
      {/* Tokenizer Selection */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Tokenizer
          </label>
          <select
            value={tokenizer}
            onChange={(e) => setTokenizer(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          >
            <option value="auto">Auto (from model)</option>
            <option value="gpt2">GPT-2</option>
            <option value="llama">LLaMA</option>
            <option value="custom">Custom</option>
          </select>
        </div>
        
        {/* Max Sequence Length */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Max Sequence Length
          </label>
          <input
            type="number"
            value={maxLength}
            onChange={(e) => setMaxLength(parseInt(e.target.value))}
            min="1"
            max="4096"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
        
        {/* Padding Strategy */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Padding Strategy
          </label>
          <select
            value={paddingStrategy}
            onChange={(e) => setPaddingStrategy(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          >
            <option value="max_length">Max Length (pad to max_length)</option>
            <option value="longest">Longest (pad to longest in batch)</option>
            <option value="do_not_pad">Do Not Pad</option>
          </select>
        </div>
        
        {/* Truncation Strategy */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Truncation Strategy
          </label>
          <select
            value={truncationStrategy}
            onChange={(e) => setTruncationStrategy(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          >
            <option value="longest_first">Longest First</option>
            <option value="only_first">Only First Sequence</option>
            <option value="only_second">Only Second Sequence</option>
            <option value="do_not_truncate">Do Not Truncate</option>
          </select>
        </div>
        
        {/* Add Special Tokens */}
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-slate-300">
            Add Special Tokens
          </label>
          <Toggle 
            value={addSpecialTokens}
            onChange={setAddSpecialTokens}
          />
        </div>
        
        {/* Return Attention Mask */}
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-slate-300">
            Return Attention Mask
          </label>
          <Toggle 
            value={returnAttentionMask}
            onChange={setReturnAttentionMask}
          />
        </div>
      </div>
    </div>
    
    {/* Preview Tokenization */}
    <div className="border-t border-slate-700 pt-6">
      <h4 className="font-semibold mb-3">Preview</h4>
      <textarea
        value={previewText}
        onChange={(e) => setPreviewText(e.target.value)}
        placeholder="Enter text to preview tokenization..."
        rows={3}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg mb-3"
      />
      
      <button
        onClick={previewTokenization}
        className="px-4 py-2 bg-slate-700 rounded hover:bg-slate-600 mb-3"
      >
        Tokenize Preview
      </button>
      
      {previewTokens && (
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="flex flex-wrap gap-1 mb-3">
            {previewTokens.map((token, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-slate-700 rounded text-sm font-mono"
                title={`Token ID: ${token.id}`}
              >
                {token.text}
              </span>
            ))}
          </div>
          <div className="text-xs text-slate-400">
            {previewTokens.length} tokens
          </div>
        </div>
      )}
    </div>
    
    {/* Apply Settings */}
    <div className="border-t border-slate-700 pt-6">
      <button
        onClick={applyTokenizationSettings}
        disabled={isProcessing}
        className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 rounded-lg flex items-center justify-center gap-2"
      >
        {isProcessing ? (
          <>
            <Loader className="w-5 h-5 animate-spin" />
            Processing Dataset...
          </>
        ) : (
          <>
            <CheckCircle className="w-5 h-5" />
            Apply & Tokenize Dataset
          </>
        )}
      </button>
    </div>
  </div>
</TokenizationSettings>
```

### **Backend API Required:**
```javascript
POST /api/datasets/{dataset_id}/tokenize
{
  "tokenizer": "auto",
  "max_length": 512,
  "padding": "max_length",
  "truncation": "longest_first",
  "add_special_tokens": true,
  "return_attention_mask": true
}

// Preview endpoint
POST /api/tokenize/preview
{
  "text": "Hello world",
  "tokenizer": "gpt2"
}
```

---

## **Step 4.3: Dataset Sample Browser**

### **What to Add:**
Paginated browser to view raw samples from dataset

### **Implementation Details:**
```tsx
<DatasetSampleBrowser datasetId={datasetId} samples={samples}>
  <div className="space-y-4">
    {/* Filter and Search */}
    <div className="flex gap-3">
      <input
        type="text"
        placeholder="Search in samples..."
        value={sampleSearch}
        onChange={(e) => setSampleSearch(e.target.value)}
        className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      />
      
      <select
        value={filterSplit}
        onChange={(e) => setFilterSplit(e.target.value)}
        className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        <option value="all">All Splits</option>
        <option value="train">Train</option>
        <option value="validation">Validation</option>
        <option value="test">Test</option>
      </select>
    </div>
    
    {/* Sample List */}
    <div className="space-y-3">
      {filteredSamples.map((sample, idx) => (
        <div 
          key={sample.id}
          className="bg-slate-800/30 rounded-lg p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-500">
              Sample #{sample.id} • {sample.split}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => copySample(sample.text)}
                className="text-slate-400 hover:text-slate-200"
              >
                <Copy className="w-4 h-4" />
              </button>
              <button
                onClick={() => expandSample(sample.id)}
                className="text-slate-400 hover:text-slate-200"
              >
                {expandedSamples.includes(sample.id) ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
          
          <div className={`text-sm text-slate-300 ${
            expandedSamples.includes(sample.id) ? '' : 'line-clamp-3'
          }`}>
            {sample.text}
          </div>
          
          {sample.metadata && (
            <div className="mt-2 text-xs text-slate-500">
              Length: {sample.text.length} chars • 
              {sample.metadata.source && ` Source: ${sample.metadata.source}`}
            </div>
          )}
        </div>
      ))}
    </div>
    
    {/* Pagination */}
    <div className="flex items-center justify-between">
      <span className="text-sm text-slate-400">
        Showing {startIdx + 1}-{Math.min(endIdx, totalSamples)} of {totalSamples}
      </span>
      
      <div className="flex gap-2">
        <button
          onClick={() => setPage(page - 1)}
          disabled={page === 0}
          className="px-3 py-1 bg-slate-800 rounded disabled:opacity-50"
        >
          Previous
        </button>
        <button
          onClick={() => setPage(page + 1)}
          disabled={endIdx >= totalSamples}
          className="px-3 py-1 bg-slate-800 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</DatasetSampleBrowser>
```

---

## **Step 4.4: Dataset Statistics Dashboard**

### **What to Add:**
Visualizations and statistics about dataset composition

### **Implementation Details:**
```tsx
<DatasetStatistics datasetId={datasetId} stats={stats}>
  <div className="space-y-6">
    {/* Summary Cards */}
    <div className="grid grid-cols-4 gap-4">
      <StatCard 
        title="Total Samples"
        value={stats.total_samples.toLocaleString()}
        icon={<Database />}
      />
      <StatCard 
        title="Total Tokens"
        value={stats.total_tokens.toLocaleString()}
        icon={<Hash />}
      />
      <StatCard 
        title="Avg Tokens/Sample"
        value={stats.avg_tokens_per_sample.toFixed(1)}
        icon={<TrendingUp />}
      />
      <StatCard 
        title="Unique Tokens"
        value={stats.unique_tokens.toLocaleString()}
        icon={<Layers />}
      />
    </div>
    
    {/* Sequence Length Distribution */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h4 className="font-semibold mb-4">Sequence Length Distribution</h4>
      <BarChart
        data={stats.sequence_length_distribution}
        xAxis="length_bucket"
        yAxis="count"
        height={250}
        color="#10b981"
      />
      <div className="mt-2 text-sm text-slate-400">
        Min: {stats.min_length} • 
        Median: {stats.median_length} • 
        Max: {stats.max_length} tokens
      </div>
    </div>
    
    {/* Token Frequency */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h4 className="font-semibold mb-4">Top Tokens by Frequency</h4>
      <div className="space-y-2">
        {stats.top_tokens.slice(0, 20).map((token, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <span className="w-8 text-slate-500 text-sm">{idx + 1}</span>
            <span className="font-mono text-sm w-32">{token.text}</span>
            <div className="flex-1 h-6 bg-slate-700 rounded overflow-hidden">
              <div 
                className="h-full bg-emerald-500"
                style={{ width: `${(token.frequency / stats.max_token_frequency) * 100}%` }}
              />
            </div>
            <span className="text-sm text-emerald-400 w-24 text-right">
              {token.count.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
    
    {/* Split Distribution */}
    {stats.split_distribution && (
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h4 className="font-semibold mb-4">Split Distribution</h4>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(stats.split_distribution).map(([split, count]) => (
            <div key={split} className="bg-slate-800/30 rounded-lg p-4">
              <div className="text-sm text-slate-400 mb-1 capitalize">{split}</div>
              <div className="text-2xl font-bold text-emerald-400">
                {count.toLocaleString()}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {((count / stats.total_samples) * 100).toFixed(1)}% of total
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
</DatasetStatistics>
```

### **Backend API Required:**
```javascript
GET /api/datasets/{dataset_id}/statistics

Response:
{
  "total_samples": 10000,
  "total_tokens": 5234567,
  "avg_tokens_per_sample": 523.4,
  "unique_tokens": 50257,
  "min_length": 10,
  "median_length": 487,
  "max_length": 2048,
  "sequence_length_distribution": [...],
  "top_tokens": [...],
  "split_distribution": {...}
}
```

---

# **PHASE 5: Model Management Enhancements (P1 - Sprint 5)**
**Goal:** Add layer selection, activation extraction, and model inspection tools

## **Step 5.1: Model Architecture Viewer**

### **What to Add:**
Visual representation of model layers with metadata

### **Implementation Details:**
```tsx
<ModelArchitectureViewer model={selectedModel}>
  <div className="space-y-6">
    <div className="flex items-center justify-between">
      <h3 className="text-xl font-semibold">Model Architecture</h3>
      <button
        onClick={refreshModelInfo}
        className="text-sm text-emerald-400"
      >
        <RefreshCw className="w-4 h-4" />
      </button>
    </div>
    
    {/* Model Overview */}
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="text-sm text-slate-400 mb-1">Total Layers</div>
        <div className="text-2xl font-bold text-purple-400">
          {model.num_layers}
        </div>
      </div>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="text-sm text-slate-400 mb-1">Hidden Dimension</div>
        <div className="text-2xl font-bold text-blue-400">
          {model.hidden_dim}
        </div>
      </div>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="text-sm text-slate-400 mb-1">Attention Heads</div>
        <div className="text-2xl font-bold text-emerald-400">
          {model.num_attention_heads}
        </div>
      </div>
    </div>
    
    {/* Layer List */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h4 className="font-semibold mb-4">Transformer Layers</h4>
      
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {Array.from({ length: model.num_layers }, (_, idx) => (
          <div 
            key={idx}
            className="bg-slate-800/30 rounded-lg p-3 flex items-center justify-between hover:bg-slate-800/50"
          >
            <div className="flex items-center gap-3">
              <Layers className="w-5 h-5 text-purple-400" />
              <div>
                <div className="font-medium">Layer {idx}</div>
                <div className="text-xs text-slate-400">
                  Transformer Block
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>d_model: {model.hidden_dim}</span>
              <span>•</span>
              <span>heads: {model.num_attention_heads}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
    
    {/* Model Config */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h4 className="font-semibold mb-4">Configuration</h4>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Vocabulary Size:</span>
          <span className="ml-2 text-slate-200">{model.vocab_size}</span>
        </div>
        <div>
          <span className="text-slate-400">Max Position:</span>
          <span className="ml-2 text-slate-200">{model.max_position_embeddings}</span>
        </div>
        <div>
          <span className="text-slate-400">MLP Ratio:</span>
          <span className="ml-2 text-slate-200">{model.mlp_ratio}x</span>
        </div>
        <div>
          <span className="text-slate-400">Architecture:</span>
          <span className="ml-2 text-slate-200">{model.architecture}</span>
        </div>
      </div>
    </div>
  </div>
</ModelArchitectureViewer>
```

---

## **Step 5.2: Activation Extraction Configuration (CRITICAL)**

### **What to Add:**
Interface to select layers and activation types for extraction

### **Implementation Details:**
```tsx
<ActivationExtractionPanel model={selectedModel}>
  <div className="space-y-6">
    <div>
      <h3 className="text-xl font-semibold mb-2">Extract Activations</h3>
      <p className="text-sm text-slate-400">
        Select layers and activation types to extract for autoencoder training
      </p>
    </div>
    
    {/* Dataset Selection */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Dataset
      </label>
      <select
        value={selectedDataset}
        onChange={(e) => setSelectedDataset(e.target.value)}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        <option value="">Select dataset...</option>
        {readyDatasets.map(ds => (
          <option key={ds.id} value={ds.id}>{ds.name}</option>
        ))}
      </select>
    </div>
    
    {/* Layer Selection */}
    <div>
      <div className="flex items-center justify-between mb-3">
        <label className="text-sm font-medium text-slate-300">
          Select Layers to Extract From
        </label>
        <div className="flex gap-2">
          <button
            onClick={selectAllLayers}
            className="text-xs text-emerald-400 hover:text-emerald-300"
          >
            Select All
          </button>
          <button
            onClick={deselectAllLayers}
            className="text-xs text-slate-400 hover:text-slate-300"
          >
            Deselect All
          </button>
        </div>
      </div>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 max-h-64 overflow-y-auto">
        <div className="grid grid-cols-4 gap-2">
          {Array.from({ length: model.num_layers }, (_, idx) => (
            <label
              key={idx}
              className={`flex items-center justify-center p-3 rounded cursor-pointer border-2 transition-colors ${
                selectedLayers.includes(idx)
                  ? 'border-emerald-500 bg-emerald-500/10'
                  : 'border-slate-700 bg-slate-800/30 hover:border-slate-600'
              }`}
            >
              <input
                type="checkbox"
                checked={selectedLayers.includes(idx)}
                onChange={() => toggleLayer(idx)}
                className="hidden"
              />
              <span className="font-medium">Layer {idx}</span>
            </label>
          ))}
        </div>
      </div>
      
      <div className="mt-2 text-xs text-slate-400">
        {selectedLayers.length} layer(s) selected
      </div>
    </div>
    
    {/* Activation Type Selection */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-3">
        Activation Types
      </label>
      
      <div className="space-y-2">
        <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer hover:bg-slate-800/50">
          <input
            type="checkbox"
            checked={activationTypes.includes('residual')}
            onChange={() => toggleActivationType('residual')}
            className="w-4 h-4 accent-emerald-500"
          />
          <div>
            <div className="font-medium">Residual Stream</div>
            <div className="text-xs text-slate-400">
              Main information pathway through the model
            </div>
          </div>
        </label>
        
        <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer hover:bg-slate-800/50">
          <input
            type="checkbox"
            checked={activationTypes.includes('mlp')}
            onChange={() => toggleActivationType('mlp')}
            className="w-4 h-4 accent-emerald-500"
          />
          <div>
            <div className="font-medium">MLP Output</div>
            <div className="text-xs text-slate-400">
              Feed-forward network activations
            </div>
          </div>
        </label>
        
        <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer hover:bg-slate-800/50">
          <input
            type="checkbox"
            checked={activationTypes.includes('attention')}
            onChange={() => toggleActivationType('attention')}
            className="w-4 h-4 accent-emerald-500"
          />
          <div>
            <div className="font-medium">Attention Output</div>
            <div className="text-xs text-slate-400">
              Multi-head attention activations
            </div>
          </div>
        </label>
      </div>
    </div>
    
    {/* Extraction Settings */}
    <div className="grid grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Batch Size
        </label>
        <input
          type="number"
          value={batchSize}
          onChange={(e) => setBatchSize(parseInt(e.target.value))}
          min="1"
          max="64"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Max Samples
        </label>
        <input
          type="number"
          value={maxSamples}
          onChange={(e) => setMaxSamples(parseInt(e.target.value))}
          min="100"
          max="1000000"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
        />
      </div>
    </div>
    
    {/* Start Extraction Button */}
    <button
      onClick={startExtraction}
      disabled={!selectedDataset || selectedLayers.length === 0 || activationTypes.length === 0 || isExtracting}
      className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
    >
      {isExtracting ? (
        <>
          <Loader className="w-5 h-5 animate-spin" />
          Extracting Activations...
        </>
      ) : (
        <>
          <Zap className="w-5 h-5" />
          Extract Activations
        </>
      )}
    </button>
    
    {/* Extraction Progress */}
    {isExtracting && (
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Extraction Progress</span>
            <span className="text-emerald-400 font-medium">
              {extractionProgress.toFixed(1)}%
            </span>
          </div>
          
          <ProgressBar progress={extractionProgress} />
          
          <div className="text-xs text-slate-400">
            Processing layer {extractionProgress.current_layer} • 
            {extractionProgress.samples_processed.toLocaleString()} / 
            {extractionProgress.total_samples.toLocaleString()} samples
          </div>
        </div>
      </div>
    )}
    
    {/* Extracted Activation Datasets */}
    {extractedActivations.length > 0 && (
      <div className="border-t border-slate-700 pt-6">
        <h4 className="font-semibold mb-3">Extracted Activation Datasets</h4>
        <div className="space-y-2">
          {extractedActivations.map(activation => (
            <div 
              key={activation.id}
              className="bg-slate-800/30 rounded-lg p-4 flex items-center justify-between"
            >
              <div>
                <div className="font-medium">
                  {activation.layer_range} • {activation.activation_type}
                </div>
                <div className="text-sm text-slate-400">
                  {activation.num_samples.toLocaleString()} samples • 
                  {activation.size} • 
                  Created {formatDate(activation.created_at)}
                </div>
              </div>
              
              <div className="flex gap-2">
                <button className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 text-sm">
                  View
                </button>
                <button className="px-3 py-1 bg-red-900/30 text-red-400 rounded hover:bg-red-900/50 text-sm">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
</ActivationExtractionPanel>
```

### **Backend API Required:**
```javascript
POST /api/models/{model_id}/extract-activations
{
  "dataset_id": "ds1",
  "layers": [0, 5, 10, 15, 20],
  "activation_types": ["residual", "mlp"],
  "batch_size": 8,
  "max_samples": 10000
}

// Poll for progress
GET /api/models/{model_id}/extraction-status
{
  "status": "extracting",
  "progress": 45.3,
  "current_layer": 10,
  "samples_processed": 4530,
  "total_samples": 10000
}
```

---

# **PHASE 6: Advanced Visualizations (P2 - Sprint 6)**
**Goal:** Add dimensionality reduction, correlation matrices, and advanced charts

## **Step 6.1: UMAP/t-SNE Feature Projection**

### **What to Add:**
Interactive scatter plot of features in 2D space

### **Implementation Details:**
```tsx
<FeatureProjection trainingId={trainingId} features={features}>
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h3 className="text-lg font-semibold">Feature Space Visualization</h3>
      
      <div className="flex gap-2">
        <select
          value={projectionMethod}
          onChange={(e) => setProjectionMethod(e.target.value)}
          className="px-3 py-1 bg-slate-900 border border-slate-700 rounded text-sm"
        >
          <option value="umap">UMAP</option>
          <option value="tsne">t-SNE</option>
          <option value="pca">PCA</option>
        </select>
        
        <button
          onClick={recomputeProjection}
          disabled={isComputing}
          className="px-3 py-1 bg-slate-700 rounded hover:bg-slate-600 text-sm"
        >
          {isComputing ? <Loader className="w-4 h-4 animate-spin" /> : 'Recompute'}
        </button>
      </div>
    </div>
    
    {/* Scatter Plot */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <svg width="100%" height="500" ref={svgRef}>
        {/* D3 will render scatter plot here */}
      </svg>
      
      {/* Legend */}
      <div className="mt-4 flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-emerald-500" />
          <span>High interpretability</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span>Medium interpretability</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-slate-500" />
          <span>Low interpretability</span>
        </div>
      </div>
    </div>
    
    {/* Color By Options */}
    <div className="flex items-center gap-3">
      <span className="text-sm text-slate-400">Color by:</span>
      <div className="flex gap-2">
        {['interpretability', 'activation_freq', 'cluster'].map(option => (
          <button
            key={option}
            onClick={() => setColorBy(option)}
            className={`px-3 py-1 rounded text-sm ${
              colorBy === option
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-800 text-slate-300'
            }`}
          >
            {option.replace('_', ' ')}
          </button>
        ))}
      </div>
    </div>
    
    {/* Selected Feature Detail */}
    {selectedFeatureFromPlot && <div className="bg-blue-900/20 border border-blue-800/30 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-semibold">Selected Feature</h4>
          <button
            onClick={() => setSelectedFeatureFromPlot(null)}
            className="text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        <div className="grid grid-cols-3 gap-4">
          <div>
            <div className="text-xs text-slate-400">Feature ID</div>
            <div className="font-mono text-emerald-400">
              #{selectedFeatureFromPlot.id}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Label</div>
            <div className="text-slate-200">
              {selectedFeatureFromPlot.label || 'Unlabeled'}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">Interpretability</div>
            <div className="text-blue-400">
              {(selectedFeatureFromPlot.interpretability * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        
        <button
          onClick={() => openFeatureDetail(selectedFeatureFromPlot.id)}
          className="mt-3 w-full px-4 py-2 bg-slate-800 rounded hover:bg-slate-700"
        >
          View Full Details
        </button>
      </div>
    )}
  </div>
</FeatureProjection>
```

### **Backend API Required:**
```javascript
GET /api/training/{training_id}/feature-projection?method=umap

Response:
{
  "projections": [
    {"feature_id": 1, "x": 12.3, "y": -5.7},
    {"feature_id": 2, "x": -8.2, "y": 14.1},
    ...
  ],
  "metadata": {
    "method": "umap",
    "perplexity": 30, // for t-SNE
    "n_neighbors": 15 // for UMAP
  }
}
```

---

## **Step 6.2: Feature Correlation Heatmap**

### **What to Add:**
Matrix showing correlations between features

### **Implementation Details:**
```tsx
<FeatureCorrelationMatrix trainingId={trainingId}>
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h3 className="text-lg font-semibold">Feature Correlations</h3>
      
      <div className="flex gap-2">
        <select
          value={correlationMethod}
          onChange={(e) => setCorrelationMethod(e.target.value)}
          className="px-3 py-1 bg-slate-900 border border-slate-700 rounded text-sm"
        >
          <option value="pearson">Pearson</option>
          <option value="spearman">Spearman</option>
          <option value="cosine">Cosine Similarity</option>
        </select>
        
        <input
          type="number"
          value={topN}
          onChange={(e) => setTopN(parseInt(e.target.value))}
          min="10"
          max="200"
          step="10"
          placeholder="Top N"
          className="w-20 px-2 py-1 bg-slate-900 border border-slate-700 rounded text-sm"
        />
      </div>
    </div>
    
    {/* Heatmap */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 overflow-auto">
      <div className="min-w-[600px]">
        <svg width="100%" height="600" ref={heatmapRef}>
          {/* D3 will render heatmap here */}
        </svg>
      </div>
      
      {/* Color Scale */}
      <div className="mt-4 flex items-center gap-3">
        <span className="text-xs text-slate-400">Correlation:</span>
        <div className="flex items-center gap-2">
          <span className="text-xs">-1.0</span>
          <div className="w-48 h-4 rounded" style={{
            background: 'linear-gradient(to right, #ef4444, #ffffff, #10b981)'
          }} />
          <span className="text-xs">+1.0</span>
        </div>
      </div>
    </div>
    
    {/* Threshold Control */}
    <div className="flex items-center gap-3">
      <span className="text-sm text-slate-400">Show correlations above:</span>
      <input
        type="range"
        value={correlationThreshold}
        onChange={(e) => setCorrelationThreshold(parseFloat(e.target.value))}
        min="0"
        max="1"
        step="0.05"
        className="flex-1 accent-emerald-500"
      />
      <span className="text-sm text-emerald-400 font-mono w-12">
        {correlationThreshold.toFixed(2)}
      </span>
    </div>
    
    {/* Highly Correlated Pairs */}
    {highlyCorrelatedPairs.length > 0 && (
      <div className="border-t border-slate-700 pt-4">
        <h4 className="font-semibold mb-3">Highly Correlated Feature Pairs</h4>
        <div className="space-y-2">
          {highlyCorrelatedPairs.slice(0, 10).map((pair, idx) => (
            <div 
              key={idx}
              className="bg-slate-800/30 rounded-lg p-3 flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <span className="font-mono text-sm text-slate-400">
                  #{pair.feature1_id}
                </span>
                <ArrowRightLeft className="w-4 h-4 text-slate-500" />
                <span className="font-mono text-sm text-slate-400">
                  #{pair.feature2_id}
                </span>
              </div>
              
              <div className="flex items-center gap-3">
                <span className="text-emerald-400 font-mono text-sm">
                  {pair.correlation.toFixed(3)}
                </span>
                <button
                  onClick={() => compareFeatures(pair.feature1_id, pair.feature2_id)}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  Compare
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
</FeatureCorrelationMatrix>
```

### **Backend API Required:**
```javascript
GET /api/training/{training_id}/feature-correlations?
  method=pearson&
  top_n=100&
  threshold=0.5

Response:
{
  "correlation_matrix": [[...], [...]],
  "feature_ids": [1, 2, 3, ...],
  "highly_correlated_pairs": [
    {"feature1_id": 5, "feature2_id": 42, "correlation": 0.87},
    ...
  ]
}
```

---

## **Step 6.3: Training Loss Curves (Live)**

### **What to Add:**
Real-time line charts for loss and sparsity metrics

### **Implementation Details:**
```tsx
<TrainingLossCurves trainingId={trainingId}>
  <div className="space-y-4">
    <h3 className="text-lg font-semibold">Training Metrics Over Time</h3>
    
    {/* Loss Chart */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium">Reconstruction Loss</h4>
        <div className="flex items-center gap-2 text-xs">
          <Toggle 
            value={logScale}
            onChange={setLogScale}
            label="Log Scale"
          />
        </div>
      </div>
      
      <LineChart
        data={lossHistory}
        xAxis="step"
        yAxis="loss"
        height={200}
        color="#10b981"
        logScale={logScale}
        showGrid={true}
        animate={isTraining}
      />
    </div>
    
    {/* Sparsity Chart */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium">L0 Sparsity</h4>
        <div className="text-xs text-slate-400">
          Target: ~{targetSparsity} active features
        </div>
      </div>
      
      <LineChart
        data={sparsityHistory}
        xAxis="step"
        yAxis="l0_norm"
        height={200}
        color="#3b82f6"
        targetLine={targetSparsity}
        showGrid={true}
        animate={isTraining}
      />
    </div>
    
    {/* Combined Metrics */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <h4 className="font-medium mb-3">All Metrics</h4>
      
      <MultiLineChart
        data={metricsHistory}
        xAxis="step"
        series={[
          { key: 'loss', color: '#10b981', label: 'Loss' },
          { key: 'l0_sparsity', color: '#3b82f6', label: 'L0 Sparsity' },
          { key: 'dead_neurons', color: '#ef4444', label: 'Dead Neurons' },
          { key: 'learning_rate', color: '#8b5cf6', label: 'Learning Rate' }
        ]}
        height={250}
        showLegend={true}
        animate={isTraining}
      />
    </div>
    
    {/* Download Data */}
    <div className="flex gap-2">
      <button
        onClick={downloadMetricsCSV}
        className="px-4 py-2 bg-slate-800 rounded hover:bg-slate-700 flex items-center gap-2"
      >
        <Download className="w-4 h-4" />
        Download CSV
      </button>
      
      <button
        onClick={downloadMetricsJSON}
        className="px-4 py-2 bg-slate-800 rounded hover:bg-slate-700 flex items-center gap-2"
      >
        <Download className="w-4 h-4" />
        Download JSON
      </button>
    </div>
  </div>
</TrainingLossCurves>
```

---

# **PHASE 7: System Settings & Management (P2 - Sprint 7)**
**Goal:** Add global settings, GPU monitoring, and system configuration

## **Step 7.1: Settings Page**

### **What to Add:**
New tab or modal for global application settings

### **Implementation Details:**
```tsx
<SettingsPage>
  <div className="max-w-4xl mx-auto space-y-8">
    <h2 className="text-2xl font-bold">Settings</h2>
    
    {/* HuggingFace Integration */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">HuggingFace Integration</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Default Access Token
          </label>
          <input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg font-mono text-sm"
          />
          <p className="text-xs text-slate-500 mt-1">
            Stored securely and used for gated model access
          </p>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Cache Directory
          </label>
          <input
            type="text"
            value={hfCacheDir}
            onChange={(e) => setHfCacheDir(e.target.value)}
            placeholder="/path/to/huggingface/cache"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg font-mono text-sm"
          />
        </div>
        
        <button
          onClick={testHFConnection}
          className="px-4 py-2 bg-slate-700 rounded hover:bg-slate-600"
        >
          Test Connection
        </button>
      </div>
    </div>
    
    {/* Hardware Configuration */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Hardware Configuration</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            GPU Device
          </label>
          <select
            value={selectedGPU}
            onChange={(e) => setSelectedGPU(e.target.value)}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          >
            <option value="auto">Auto-detect</option>
            {availableGPUs.map(gpu => (
              <option key={gpu.id} value={gpu.id}>
                {gpu.name} ({gpu.memory})
              </option>
            ))}
            <option value="cpu">CPU Only</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Mixed Precision Training
          </label>
          <Toggle
            value={mixedPrecision}
            onChange={setMixedPrecision}
          />
          <p className="text-xs text-slate-500 mt-1">
            Use FP16 for faster training on supported GPUs
          </p>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Max Memory Usage
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              value={maxMemoryPercent}
              onChange={(e) => setMaxMemoryPercent(parseInt(e.target.value))}
              min="50"
              max="95"
              className="flex-1 accent-emerald-500"
            />
            <span className="text-emerald-400 font-mono w-12">
              {maxMemoryPercent}%
            </span>
          </div>
        </div>
      </div>
    </div>
    
    {/* Default Training Hyperparameters */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">Default Hyperparameters</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Learning Rate
          </label>
          <input
            type="number"
            value={defaultLR}
            onChange={(e) => setDefaultLR(parseFloat(e.target.value))}
            step="0.0001"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Batch Size
          </label>
          <input
            type="number"
            value={defaultBatchSize}
            onChange={(e) => setDefaultBatchSize(parseInt(e.target.value))}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            L1 Coefficient
          </label>
          <input
            type="number"
            value={defaultL1}
            onChange={(e) => setDefaultL1(parseFloat(e.target.value))}
            step="0.0001"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Expansion Factor
          </label>
          <select
            value={defaultExpansion}
            onChange={(e) => setDefaultExpansion(parseInt(e.target.value))}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          >
            <option value="4">4x</option>
            <option value="8">8x</option>
            <option value="16">16x</option>
            <option value="32">32x</option>
          </select>
        </div>
      </div>
    </div>
    
    {/* API Configuration */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4">API Configuration</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Backend URL
          </label>
          <input
            type="text"
            value={backendURL}
            onChange={(e) => setBackendURL(e.target.value)}
            placeholder="http://localhost:8000"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            WebSocket URL
          </label>
          <input
            type="text"
            value={wsURL}
            onChange={(e) => setWsURL(e.target.value)}
            placeholder="ws://localhost:8000/ws"
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
          />
        </div>
      </div>
    </div>
    
    {/* Save Settings */}
    <div className="flex gap-3">
      <button
        onClick={saveSettings}
        className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 rounded-lg flex items-center gap-2"
      >
        <Save className="w-5 h-5" />
        Save Settings
      </button>
      
      <button
        onClick={resetToDefaults}
        className="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg"
      >
        Reset to Defaults
      </button>
    </div>
  </div>
</SettingsPage>
```

### **Backend API Required:**
```javascript
GET /api/settings
PUT /api/settings
{
  "hf_token": "...",
  "hf_cache_dir": "...",
  "gpu_device": "cuda:0",
  "mixed_precision": true,
  "default_hyperparameters": {...}
}
```

---

## **Step 7.2: System Monitoring Dashboard**

### **What to Add:**
Real-time GPU and system metrics

### **Implementation Details:**
```tsx
<SystemMonitor>
  <div className="space-y-4">
    <h3 className="text-lg font-semibold">System Status</h3>
    
    {/* GPU Metrics */}
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium">GPU Utilization</h4>
          <Cpu className="w-5 h-5 text-purple-400" />
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Compute</span>
            <span className="text-purple-400 font-mono">
              {gpuMetrics.utilization}%
            </span>
          </div>
          <ProgressBar 
            progress={gpuMetrics.utilization} 
            color="purple"
          />
          
          <div className="flex items-center justify-between text-sm mt-3">
            <span className="text-slate-400">Memory</span>
            <span className="text-blue-400 font-mono">
              {gpuMetrics.memory_used} / {gpuMetrics.memory_total}
            </span>
          </div>
          <ProgressBar 
            progress={(gpuMetrics.memory_used / gpuMetrics.memory_total) * 100} 
            color="blue"
          />
        </div>
      </div>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium">GPU Temperature</h4>
          <Thermometer className="w-5 h-5 text-orange-400" />
        </div>
        
        <div className="text-center">
          <div className="text-4xl font-bold text-orange-400 mb-1">
            {gpuMetrics.temperature}°C
          </div>
          <div className="text-xs text-slate-400">
            Max: {gpuMetrics.max_temp}°C
          </div>
        </div>
        
        <ProgressBar 
          progress={(gpuMetrics.temperature / gpuMetrics.max_temp) * 100}
          color="orange"
          className="mt-3"
        />
      </div>
    </div>
    
    {/* Disk Space */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium">Disk Space</h4>
        <HardDrive className="w-5 h-5 text-emerald-400" />
      </div>
      
      <div className="space-y-3">
        {diskMetrics.map(disk => (
          <div key={disk.mount}>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-slate-400">{disk.mount}</span>
              <span className="text-emerald-400 font-mono">
                {disk.used} / {disk.total} ({disk.percent}%)
              </span>
            </div>
            <ProgressBar progress={disk.percent} />
          </div>
        ))}
      </div>
    </div>
    
    {/* Active Processes */}
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4">
      <h4 className="font-medium mb-3">Active Jobs</h4>
      
      <div className="space-y-2">
        {activeJobs.map(job => (
          <div 
            key={job.id}
            className="flex items-center justify-between bg-slate-800/30 rounded p-3"
          >
            <div className="flex items-center gap-3">
              <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
              <div>
                <div className="font-medium text-sm">{job.type}</div>
                <div className="text-xs text-slate-400">{job.description}</div>
              </div>
            </div>
            
            <div className="text-sm text-emerald-400">
              {job.progress.toFixed(1)}%
            </div>
          </div>
        ))}
        
        {activeJobs.length === 0 && (
          <div className="text-center py-4 text-slate-400 text-sm">
            No active jobs
          </div>
        )}
      </div>
    </div>
  </div>
</SystemMonitor>
```

### **Backend API Required:**
```javascript
GET /api/system/metrics

Response:
{
  "gpu": {
    "utilization": 87.5,
    "memory_used": "8.2GB",
    "memory_total": "12GB",
    "temperature": 72,
    "max_temp": 95
  },
  "disk": [...],
  "active_jobs": [...]
}
```

---

# **PHASE 8: Additional Enhancements (P3 - Sprint 8+)**
**Goal:** Nice-to-have features for production polish

## **Step 8.1: Project/Experiment Management**

### **What to Add:**
Save and load complete experiment configurations

### **Implementation Details:**
```tsx
<ProjectManager>
  <div className="space-y-6">
    <div className="flex items-center justify-between">
      <h2 className="text-2xl font-bold">Projects</h2>
      <button
        onClick={createNewProject}
        className="px-4 py-2 bg-emerald-600 rounded-lg flex items-center gap-2"
      >
        <Plus className="w-4 h-4" />
        New Project
      </button>
    </div>
    
    {/* Project List */}
    <div className="grid gap-4">
      {projects.map(project => (
        <div 
          key={project.id}
          className="bg-slate-900/50 border border-slate-800 rounded-lg p-6"
        >
          <div className="flex items-start justify-between mb-3">
            <div>
              <h3 className="text-lg font-semibold">{project.name}</h3>
              <p className="text-sm text-slate-400">{project.description}</p>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => loadProject(project.id)}
                className="px-3 py-1 bg-emerald-600 rounded text-sm"
              >
                Load
              </button>
              <button
                onClick={() => exportProject(project.id)}
                className="px-3 py-1 bg-slate-700 rounded text-sm"
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={() => deleteProject(project.id)}
                className="px-3 py-1 bg-red-900/30 text-red-400 rounded text-sm"
              >
                <Trash className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-slate-400">Datasets:</span>
              <span className="ml-2 text-slate-200">{project.datasets_count}</span>
            </div>
            <div>
              <span className="text-slate-400">Models:</span>
              <span className="ml-2 text-slate-200">{project.models_count}</span>
            </div>
            <div>
              <span className="text-slate-400">Trainings:</span>
              <span className="ml-2 text-slate-200">{project.trainings_count}</span>
            </div>
            <div>
              <span className="text-slate-400">Created:</span>
              <span className="ml-2 text-slate-200">{formatDate(project.created_at)}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
</ProjectManager>
```

---

## **Step 8.2: Export/Import Functionality**

### **What to Add:**
Export trained autoencoders, features, and configurations

### **Implementation Details:**
```tsx
<ExportPanel trainingId={trainingId}>
  <div className="space-y-4">
    <h3 className="text-lg font-semibold">Export Training Results</h3>
    
    <div className="space-y-3">
      <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer">
        <input
          type="checkbox"
          checked={exportOptions.model}
          onChange={() => toggleExportOption('model')}
          className="w-4 h-4 accent-emerald-500"
        />
        <div>
          <div className="font-medium">Trained Autoencoder</div>
          <div className="text-xs text-slate-400">Model weights (.pt, .safetensors)</div>
        </div>
      </label>
      
      <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer">
        <input
          type="checkbox"
          checked={exportOptions.features}
          onChange={() => toggleExportOption('features')}
          className="w-4 h-4 accent-emerald-500"
        />
        <div>
          <div className="font-medium">Discovered Features</div>
          <div className="text-xs text-slate-400">Feature metadata and examples (.json)</div>
        </div>
      </label>
      
      <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer">
        <input
          type="checkbox"
          checked={exportOptions.metrics}
          onChange={() => toggleExportOption('metrics')}
          className="w-4 h-4 accent-emerald-500"
        />
        <div>
          <div className="font-medium">Training Metrics</div>
          <div className="text-xs text-slate-400">Loss curves and statistics (.csv)</div>
        </div>
      </label>
      
      <label className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg cursor-pointer">
        <input
          type="checkbox"
          checked={exportOptions.config}
          onChange={() => toggleExportOption('config')}
          className="w-4 h-4 accent-emerald-500"
        />
        <div>
          <div classNameContinuing from Step 8.2...

---

```tsx
          <div className="font-medium">Configuration</div>
          <div className="text-xs text-slate-400">Hyperparameters and settings (.yaml)</div>
        </div>
      </label>
    </div>
    
    {/* Export Format */}
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-2">
        Export Format
      </label>
      <select
        value={exportFormat}
        onChange={(e) => setExportFormat(e.target.value)}
        className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg"
      >
        <option value="zip">ZIP Archive</option>
        <option value="tar.gz">TAR.GZ Archive</option>
        <option value="folder">Folder Structure</option>
      </select>
    </div>
    
    {/* Export Button */}
    <button
      onClick={startExport}
      disabled={!hasSelectedExportOptions || isExporting}
      className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 rounded-lg flex items-center justify-center gap-2"
    >
      {isExporting ? (
        <>
          <Loader className="w-5 h-5 animate-spin" />
          Exporting...
        </>
      ) : (
        <>
          <Download className="w-5 h-5" />
          Export Selected Items
        </>
      )}
    </button>
  </div>
</ExportPanel>

<ImportPanel>
  <div className="space-y-4 mt-8 pt-8 border-t border-slate-700">
    <h3 className="text-lg font-semibold">Import Training</h3>
    
    <div
      onDrop={handleFileDrop}
      onDragOver={(e) => e.preventDefault()}
      className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-emerald-500 transition-colors cursor-pointer"
    >
      <Upload className="w-12 h-12 text-slate-400 mx-auto mb-3" />
      <p className="text-slate-300 mb-1">
        Drop exported training archive here
      </p>
      <p className="text-sm text-slate-400">
        or click to browse
      </p>
      <input
        type="file"
        accept=".zip,.tar.gz"
        onChange={handleFileSelect}
        className="hidden"
        ref={fileInputRef}
      />
    </div>
    
    {selectedFile && (
      <div className="bg-slate-800/30 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-emerald-400" />
            <div>
              <div className="font-medium">{selectedFile.name}</div>
              <div className="text-xs text-slate-400">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>
          </div>
          
          <button
            onClick={importTraining}
            className="px-4 py-2 bg-emerald-600 rounded hover:bg-emerald-700"
          >
            Import
          </button>
        </div>
      </div>
    )}
  </div>
</ImportPanel>
```

### **Backend API Required:**
```javascript
POST /api/training/{training_id}/export
{
  "include": ["model", "features", "metrics", "config"],
  "format": "zip"
}

// Returns download URL or streams file

POST /api/training/import
// Multipart form-data with file upload
```

---

## **Step 8.3: Onboarding Tour**

### **What to Add:**
Interactive tutorial for first-time users

### **Implementation Details:**
```tsx
<OnboardingTour>
  {tourActive && (
    <div className="fixed inset-0 z-50">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/70" />
      
      {/* Tour Step */}
      <div
        className="absolute bg-slate-900 border border-emerald-500 rounded-lg shadow-2xl p-6 max-w-md"
        style={{
          top: tourSteps[currentStep].position.top,
          left: tourSteps[currentStep].position.left
        }}
      >
        {/* Step Indicator */}
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm text-slate-400">
            Step {currentStep + 1} of {tourSteps.length}
          </span>
          <button
            onClick={skipTour}
            className="text-slate-400 hover:text-slate-200"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        
        {/* Step Content */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2 text-emerald-400">
            {tourSteps[currentStep].title}
          </h3>
          <p className="text-slate-300 text-sm">
            {tourSteps[currentStep].description}
          </p>
          
          {tourSteps[currentStep].image && (
            <img
              src={tourSteps[currentStep].image}
              alt={tourSteps[currentStep].title}
              className="mt-4 rounded-lg"
            />
          )}
        </div>
        
        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={previousStep}
            disabled={currentStep === 0}
            className="px-4 py-2 bg-slate-800 rounded disabled:opacity-50"
          >
            Previous
          </button>
          
          {currentStep < tourSteps.length - 1 ? (
            <button
              onClick={nextStep}
              className="px-4 py-2 bg-emerald-600 rounded hover:bg-emerald-700"
            >
              Next
            </button>
          ) : (
            <button
              onClick={finishTour}
              className="px-4 py-2 bg-emerald-600 rounded hover:bg-emerald-700"
            >
              Get Started
            </button>
          )}
        </div>
        
        {/* Progress Dots */}
        <div className="flex items-center justify-center gap-2 mt-4">
          {tourSteps.map((_, idx) => (
            <div
              key={idx}
              className={`w-2 h-2 rounded-full ${
                idx === currentStep ? 'bg-emerald-400' : 'bg-slate-600'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  )}
</OnboardingTour>

// Tour Steps Definition
const tourSteps = [
  {
    title: "Welcome to MechInterp Studio",
    description: "This platform helps you discover and manipulate interpretable features in language models running on edge devices. Let's walk through the main features.",
    position: { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }
  },
  {
    title: "Download Datasets",
    description: "Start by downloading datasets from HuggingFace. These will be used to extract activations from your models.",
    position: { top: '200px', left: '100px' }
  },
  {
    title: "Load Models",
    description: "Download and quantize models for edge deployment. Choose from various quantization formats to balance accuracy and memory.",
    position: { top: '200px', left: '100px' }
  },
  {
    title: "Train Autoencoders",
    description: "Configure and train sparse autoencoders to discover interpretable features. Monitor training progress in real-time.",
    position: { top: '200px', left: '100px' }
  },
  {
    title: "Explore Features",
    description: "Browse discovered features, view activation examples, and analyze what each feature represents.",
    position: { top: '200px', left: '100px' }
  },
  {
    title: "Steering Tool",
    description: "Manipulate model behavior by amplifying or suppressing specific features. Compare steered vs. unsteered outputs.",
    position: { top: '200px', left: '100px' }
  }
];
```

---

## **Step 8.4: Help & Documentation System**

### **What to Add:**
Contextual tooltips and help panel

### **Implementation Details:**
```tsx
<HelpSystem>
  {/* Help Button (Fixed Position) */}
  <button
    onClick={toggleHelpPanel}
    className="fixed bottom-6 right-6 w-14 h-14 bg-emerald-600 rounded-full shadow-lg hover:bg-emerald-700 flex items-center justify-center z-40"
  >
    <HelpCircle className="w-6 h-6 text-white" />
  </button>
  
  {/* Help Panel (Slide-in) */}
  {helpPanelOpen && (
    <div className="fixed right-0 top-0 h-full w-96 bg-slate-900 border-l border-slate-800 shadow-2xl z-50 overflow-y-auto">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold">Help & Documentation</h2>
          <button
            onClick={toggleHelpPanel}
            className="text-slate-400 hover:text-slate-200"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Search */}
        <input
          type="text"
          placeholder="Search documentation..."
          value={helpSearch}
          onChange={(e) => setHelpSearch(e.target.value)}
          className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg mb-6"
        />
        
        {/* Categories */}
        <div className="space-y-4">
          {helpCategories.map(category => (
            <div key={category.id} className="border-b border-slate-800 pb-4">
              <button
                onClick={() => toggleCategory(category.id)}
                className="w-full flex items-center justify-between text-left font-semibold mb-2"
              >
                <span>{category.title}</span>
                {expandedCategories.includes(category.id) ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
              
              {expandedCategories.includes(category.id) && (
                <div className="space-y-2 ml-4">
                  {category.articles.map(article => (
                    <button
                      key={article.id}
                      onClick={() => openArticle(article)}
                      className="block text-sm text-slate-400 hover:text-emerald-400"
                    >
                      {article.title}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Quick Links */}
        <div className="mt-8 pt-6 border-t border-slate-800">
          <h3 className="font-semibold mb-3">Quick Links</h3>
          <div className="space-y-2">
            
              href="/docs"
              target="_blank"
              className="block text-sm text-emerald-400 hover:text-emerald-300"
            >
              📘 Full Documentation
            </a>
            
              href="/api-reference"
              target="_blank"
              className="block text-sm text-emerald-400 hover:text-emerald-300"
            >
              🔌 API Reference
            </a>
            
              href="https://github.com/your-repo"
              target="_blank"
              className="block text-sm text-emerald-400 hover:text-emerald-300"
            >
              💻 GitHub Repository
            </a>
            <button
              onClick={startOnboardingTour}
              className="block text-sm text-emerald-400 hover:text-emerald-300"
            >
              🎓 Take the Tutorial
            </button>
          </div>
        </div>
      </div>
    </div>
  )}
</HelpSystem>

{/* Contextual Tooltips */}
<Tooltip content="This is the learning rate for optimizer">
  <InfoIcon className="w-4 h-4 text-slate-400" />
</Tooltip>
```

---

## **Step 8.5: Error Boundary & Better Error Handling**

### **What to Add:**
Graceful error handling with user-friendly messages

### **Implementation Details:**
```tsx
<ErrorBoundary>
  {/* Error Toast Notifications */}
  <ToastContainer>
    {toasts.map(toast => (
      <Toast
        key={toast.id}
        type={toast.type}
        message={toast.message}
        onClose={() => removeToast(toast.id)}
      />
    ))}
  </ToastContainer>
  
  {/* Global Error Modal */}
  {globalError && (
    <ErrorModal
      error={globalError}
      onRetry={retryFailedOperation}
      onDismiss={dismissError}
    />
  )}
</ErrorBoundary>

// Toast Component
function Toast({ type, message, onClose }) {
  return (
    <div className={`fixed top-4 right-4 max-w-md bg-slate-900 border rounded-lg shadow-lg p-4 flex items-start gap-3 ${
      type === 'error' ? 'border-red-500' :
      type === 'warning' ? 'border-yellow-500' :
      type === 'success' ? 'border-emerald-500' :
      'border-blue-500'
    }`}>
      {type === 'error' && <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />}
      {type === 'warning' && <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0" />}
      {type === 'success' && <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />}
      {type === 'info' && <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />}
      
      <div className="flex-1">
        <p className="text-sm text-slate-200">{message}</p>
      </div>
      
      <button onClick={onClose} className="text-slate-400 hover:text-slate-200">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

// Error Modal
function ErrorModal({ error, onRetry, onDismiss }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70" onClick={onDismiss} />
      
      <div className="relative bg-slate-900 border border-red-500 rounded-lg shadow-2xl p-6 max-w-md">
        <div className="flex items-start gap-3 mb-4">
          <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0" />
          <div>
            <h3 className="text-lg font-semibold text-red-400 mb-1">
              {error.title || 'An Error Occurred'}
            </h3>
            <p className="text-sm text-slate-300">
              {error.message}
            </p>
          </div>
        </div>
        
        {error.details && (
          <details className="mb-4">
            <summary className="text-xs text-slate-400 cursor-pointer">
              Technical Details
            </summary>
            <pre className="mt-2 text-xs bg-slate-950 rounded p-2 overflow-auto max-h-32">
              {error.details}
            </pre>
          </details>
        )}
        
        <div className="flex gap-3">
          {onRetry && (
            <button
              onClick={onRetry}
              className="flex-1 px-4 py-2 bg-emerald-600 rounded hover:bg-emerald-700"
            >
              Retry
            </button>
          )}
          <button
            onClick={onDismiss}
            className="flex-1 px-4 py-2 bg-slate-700 rounded hover:bg-slate-600"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

# **IMPLEMENTATION SUMMARY & PRIORITIZATION**

This prioritization is based on:
- Critical gaps identified in [Gap_Analysis_MockUI-vs-Technical_Specfication.md](./Gap_Analysis_MockUI-vs-Technical_Specfication.md)
- Core requirements from [miStudio_Specification.md](./miStudio_Specification.md)
- Current implementation state in [Mock-embedded-interp-ui.tsx](./Mock-embedded-interp-ui.tsx)

**Note:** The ✅ marks below indicate the recommended implementation order, **not completion status**. These are planning checkmarks showing the priority sequence.

## **Critical Path (MVP) - Implement First:**

1. ✅ **Training Hyperparameters Panel** (P0) - Step 1.1
2. ✅ **Real-Time Training Metrics** (P0) - Step 1.2
3. ✅ **Feature Extraction Interface** (P0) - Step 2.1
4. ✅ **Feature Browser/Table** (P0) - Step 2.2
5. ✅ **Feature Detail Modal** (P0) - Step 2.3
6. ✅ **Max Activating Examples** (P0) - Step 2.4
7. ✅ **Complete Steering Tab** (P0) - Steps 3.1-3.5

## **High Priority - Implement Second:**

8. ✅ **Checkpoint Management** (P1) - Step 1.5
9. ✅ **Activation Extraction UI** (P1) - Step 5.2
10. ✅ **Dataset Detail Modal** (P1) - Step 4.1
11. ✅ **Tokenization Config** (P1) - Step 4.2

## **Medium Priority - Implement Third:**

12. ✅ **UMAP/t-SNE Visualization** (P2) - Step 6.1
13. ✅ **Correlation Heatmap** (P2) - Step 6.2
14. ✅ **Settings Page** (P2) - Step 7.1
15. ✅ **System Monitor** (P2) - Step 7.2

## **Polish - Implement Last:**

16. ✅ **Project Management** (P3) - Step 8.1
17. ✅ **Export/Import** (P3) - Step 8.2
18. ✅ **Onboarding Tour** (P3) - Step 8.3
19. ✅ **Help System** (P3) - Step 8.4

---

# **INFRASTRUCTURE STACK OVERVIEW**

This section provides a high-level view of how all infrastructure components work together to support the miStudio frontend. Understanding this architecture is critical when implementing features that depend on backend services.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TypeScript)            │
│  ┌───────────┬──────────┬───────────┬──────────┬─────────┐ │
│  │ Datasets  │ Models   │ Training  │ Features │Steering │ │
│  └─────┬─────┴────┬─────┴─────┬─────┴────┬─────┴────┬────┘ │
│        │          │           │          │          │       │
│        └──────────┴───────────┴──────────┴──────────┘       │
│                        │                                     │
│                   REST API + WebSocket                       │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────┐
│               Backend (Python FastAPI)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Routes                           │  │
│  │  /api/datasets  /api/models  /api/training  /api/...  │  │
│  └────┬─────────────────────┬─────────────────────┬─────┘  │
│       │                     │                     │         │
│  ┌────▼────┐         ┌──────▼──────┐       ┌─────▼─────┐  │
│  │         │         │             │       │           │  │
│  │ PostSQL │◄────────┤   Redis     │──────►│  Celery   │  │
│  │         │         │             │       │  Workers  │  │
│  └─────────┘         └─────────────┘       └───────────┘  │
│      │                     │                      │         │
│  Metadata           Job Queue &             GPU Tasks      │
│  JSONB Data         Pub/Sub for            (Training,      │
│  Relations          WebSocket              Feature         │
│                                            Extraction)      │
└──────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Frontend Layer ([001_SPEC|Folder_File_Details.md](./001_SPEC|Folder_File_Details.md))
- **React Components**: Organized by domain (`datasets/`, `models/`, `training/`, `features/`, `steering/`)
- **TypeScript Types**: Defined in `src/types/` (training.types.ts, etc.)
- **State Management**: React hooks (useState, useEffect) for UI state
- **Real-time Updates**: WebSocket connections for training progress, metrics streaming
- **API Client**: REST calls to backend endpoints

### Backend API Layer
- **FastAPI Framework**: High-performance Python web framework
- **REST Endpoints**: CRUD operations for datasets, models, trainings, features
- **WebSocket Server**: Real-time streaming of training metrics and logs
- **Authentication**: Single-user mode (no auth required initially)
- **Validation**: Pydantic models ensure type safety and validation

### PostgreSQL Database ([003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md))
- **Primary metadata store** for all application data
- **Key Tables**:
  - `datasets` - Dataset metadata, file paths, tokenization status
  - `models` - Model metadata, architecture configs, quantization info
  - `trainings` - Training job state, hyperparameters (JSONB), progress
  - `training_metrics` - Time-series metrics (step, loss, sparsity)
  - `checkpoints` - Checkpoint metadata and file references
  - `features` - Discovered features with activation statistics
  - `feature_activations` - Max-activating examples (large table, partitioned)
  - `training_templates` - Saved training configurations
  - `extraction_templates` - Saved extraction configurations
  - `steering_presets` - Saved steering configurations
- **JSONB Support**: Flexible storage for hyperparameters, activation data
- **Full-text Search**: GIN indexes for dataset/model search
- **Relations**: Foreign keys maintain data integrity

### Redis Infrastructure ([000_SPEC|REDIS_GUIDANCE_USECASE.md](./000_SPEC|REDIS_GUIDANCE_USECASE.md))

#### Use Case 1: Job Queue Backend
- **Message Broker**: Celery uses Redis for task queue management
- **Priority Queues**: GPU tasks (training, feature extraction) get high priority
- **Result Backend**: Training results stored temporarily in Redis
- **Operations**: Dataset download, model download, SAE training, feature extraction

#### Use Case 2: WebSocket Pub/Sub
- **Real-time Metrics**: GPU workers publish training metrics to Redis channels
- **WebSocket Coordination**: Multiple WebSocket servers subscribe to same channels
- **Message Format**: `training:{training_id}` channels for per-job updates
- **Scalability**: Enables horizontal scaling of WebSocket servers

#### Use Case 3: Rate Limiting
- **API Protection**: Prevent excessive requests to HuggingFace or GPU resources
- **Per-endpoint Limits**: Different limits for different operations

### Celery Workers
- **GPU-bound Tasks**: SAE training, feature extraction (limited to 1 concurrent)
- **IO-bound Tasks**: Dataset/model downloads (limited to 2 concurrent)
- **CPU-bound Tasks**: Dataset processing, tokenization
- **Progress Updates**: Workers publish progress via Redis pub/sub
- **Fault Tolerance**: Tasks can be retried on failure

### Template System ([004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md))
- **Training Templates**: Save/load hyperparameter configurations
- **Extraction Templates**: Save/load feature extraction settings
- **Steering Presets**: Save/load steering configurations
- **Favoriting**: Mark frequently-used templates for quick access
- **Flexibility**: Templates can be generic or model/dataset-specific

## Data Flow Examples

### Example 1: Starting a Training Job
1. User configures training in frontend (Training tab)
2. Frontend sends POST to `/api/training/start` with config
3. Backend validates config, creates record in `trainings` table
4. Backend enqueues Celery task via Redis (`celery:queue:gpu_high`)
5. GPU worker picks up task, starts SAE training
6. Worker publishes progress updates to Redis channel `training:{id}`
7. WebSocket server subscribes to channel, forwards to frontend
8. Frontend displays real-time metrics charts
9. On completion, worker updates `trainings` table status to 'completed'

### Example 2: Browsing Features
1. User navigates to Features tab, selects completed training
2. Frontend sends GET to `/api/features?training_id=tr_123`
3. Backend queries PostgreSQL `features` table
4. Returns list of features with activation stats
5. User clicks feature to view details
6. Frontend sends GET to `/api/features/{id}/activations?limit=100`
7. Backend queries `feature_activations` table (partitioned for performance)
8. Returns max-activating examples with JSONB data (tokens, values)
9. Frontend displays highlighted tokens in modal

### Example 3: Applying Steering
1. User selects features in Steering tab, sets coefficients
2. User clicks "Generate" with prompt
3. Frontend sends POST to `/api/steering/generate` with config
4. Backend modifies model forward pass to intervene on selected features
5. Backend runs generation with intervention
6. Backend returns both baseline and steered outputs
7. Frontend displays side-by-side comparison

## Implementation Priority

When implementing frontend features, coordinate with backend in this order:

1. **Phase 0 (Foundation)**:
   - PostgreSQL schema setup
   - Redis configuration
   - Basic API structure

2. **Phase 1 (P0 - MVP)**:
   - Training hyperparameters API + Celery tasks
   - WebSocket pub/sub for real-time metrics
   - Checkpoint management API

3. **Phase 2 (P0 - MVP)**:
   - Feature extraction API + Celery tasks
   - Feature browsing APIs (list, detail, activations)
   - Feature activation database optimization

4. **Phase 3 (P0 - MVP)**:
   - Steering API endpoints
   - Steering presets (save/load)
   - Real-time generation with intervention

---

# **NEXT STEPS FOR IMPLEMENTATION**

With all specifications and infrastructure details in place, you can now:

1. **Start implementing Phase 1** - Begin with Training Infrastructure enhancements following the step-by-step guide above

2. **Set up backend development environment** - Follow [001_SPEC|Folder_File_Details.md](./001_SPEC|Folder_File_Details.md) to create the `/backend` directory structure

3. **Initialize database** - Use schema definitions from [003_SPEC|Postgres_Usecase_Details_and_Guidance.md](./003_SPEC|Postgres_Usecase_Details_and_Guidance.md) with Alembic migrations

4. **Configure Redis** - Follow [000_SPEC|REDIS_GUIDANCE_USECASE.md](./000_SPEC|REDIS_GUIDANCE_USECASE.md) to set up job queues and pub/sub

5. **Implement template system** - Use [004_SPEC|Template_System_Guidance.md](./004_SPEC|Template_System_Guidance.md) for training/extraction/steering templates
