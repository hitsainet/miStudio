import { useState, useEffect } from 'react';
import { Play, Download, Cpu, Database, Activity, CheckCircle, Loader, X, Star, Info, Search, ChevronDown, ChevronUp, Copy, Zap } from 'lucide-react';
import './src/styles/mock-ui.css';

// Window interface extension for global debugging properties
declare global {
  interface Window {
    pauseTraining: (trainingId: string) => void;
    resumeTraining: (trainingId: string) => void;
    stopTraining: (trainingId: string) => void;
    retryTraining: (trainingId: string) => void;
    saveCheckpoint: (trainingId: string) => void;
    loadCheckpoint: (trainingId: string, checkpointId: string) => void;
    deleteCheckpoint: (trainingId: string, checkpointId: string) => void;
    checkpoints: any;
    mockModels: any;
    mockDatasets: any;
    trainingTemplates: any;
    extractionTemplates: any;
    steeringPresets: any;
  }
}

// ============================================================================
// BACKEND IMPLEMENTATION GUIDE - COMPLETE API SPECIFICATION
// ============================================================================
/**
 * This mock UI serves as a SPECIFICATION DOCUMENT for the full application.
 *
 * PURPOSE: This single file demonstrates ALL UI functionality and documents
 * EXACTLY what backend APIs, data structures, and infrastructure are needed.
 *
 * FOR BACKEND DEVELOPERS: Read the comments throughout this file to understand:
 * - API endpoints required (method, path, request/response)
 * - Data models and relationships
 * - Real-time communication patterns (WebSocket, polling)
 * - Job queue requirements
 * - Storage requirements
 *
 * TECHNOLOGY STACK RECOMMENDATIONS:
 *
 * Backend Framework:
 * - Python: FastAPI (async, automatic OpenAPI docs, type hints)
 * - Node.js: NestJS (TypeScript, modular, enterprise-ready)
 * - Go: Fiber or Echo (high performance for Edge AI)
 *
 * Database:
 * - PostgreSQL 14+ (JSONB support, full-text search, reliability)
 * - Tables: datasets, models, trainings, training_metrics,
 *   checkpoints, features, feature_activations, steering_presets
 *
 * Job Queue:
 * - Python: Celery with Redis backend
 * - Node.js: BullMQ with Redis
 * - Priority queues for training vs. feature extraction
 *
 * Real-time Communication:
 * - WebSocket server (Socket.IO or native WebSockets)
 * - Redis pub/sub for multi-server scaling
 * - Channels per training job for isolation
 *
 * Storage:
 * - Local File System: Local storage for models/datasets/checkpoints/activation tensors
 * - Database: Metadata and relationships
 *
 * Caching:
 * - Redis for rate limiting, WebSocket pub/sub
 * - Application-level caching for expensive computations
 * - ETag support for efficient polling
 *
 * ML/AI Stack:
 * - PyTorch 2.0+ (primary framework)
 * - HuggingFace Transformers (model loading)
 * - HuggingFace Datasets (dataset loading)
 * - SAE Training: Custom PyTorch modules (sparse autoencoder)
 * - Quantization: bitsandbytes or llama.cpp
 *
 * DEPLOYMENT ARCHITECTURE (for Jetson/Edge):
 *
 * Single Device (Jetson Orin Nano):
 * - Backend API: FastAPI with Uvicorn
 * - Database: PostgreSQL (Docker)
 * - Redis: Redis (Docker)
 * - Job Worker: 1-2 Celery workers (GPU access)
 * - Storage: Local SSD + external USB drive
 *
 * Multi-Device Cluster:
 * - Load Balancer: Nginx
 * - API Servers: 2+ FastAPI instances (CPU-only)
 * - GPU Workers: 1 per Jetson device (training/inference)
 * - Shared Storage: NFS mount for activations
 * - Redis: Shared instance for coordination
 *
 * API DESIGN PRINCIPLES:
 *
 * 1. RESTful Conventions:
 *    - GET: Retrieve resources (idempotent)
 *    - POST: Create resources or trigger actions
 *    - PUT/PATCH: Update resources (idempotent)
 *    - DELETE: Remove resources (idempotent)
 *
 * 2. Response Format:
 *    Success: { "data": {...}, "meta": {...} }
 *    Error:   { "error": { "code": "...", "message": "...", "details": {...} } }
 *
 * 3. Status Codes:
 *    - 200: Success (GET, PUT, DELETE)
 *    - 201: Created (POST)
 *    - 202: Accepted (long-running job queued)
 *    - 400: Bad Request (validation error)
 *    - 404: Not Found
 *    - 409: Conflict (duplicate resource)
 *    - 429: Rate Limit Exceeded
 *    - 500: Internal Server Error
 *    - 503: Service Unavailable (GPU busy)
 *
 * 4. Pagination:
 *    - Query params: ?page=1&limit=50
 *    - Response: { "data": [...], "pagination": { "page": 1, "total": 234, "has_next": true } }
 *
 * 5. Filtering/Sorting:
 *    - Query params: ?search=query&sortBy=field&order=desc
 *    - Full-text search on name/description fields
 *
 * 6. Long-Running Operations:
 *    - POST returns 202 with job_id immediately
 *    - Client polls GET /api/jobs/:id/status or uses WebSocket
 *    - Include estimated_duration_seconds in response
 *
 * RATE LIMITING:
 *
 * Global Limits (stored in Redis):
 * - Global: 100 requests/minute
 * - Training: 1 concurrent training job (GPU limitation)
 * - Downloads: 10/hour (network/disk protection)
 * - Generation: 20/hour
 *
 * Headers:
 * - X-RateLimit-Limit: 100
 * - X-RateLimit-Remaining: 47
 * - X-RateLimit-Reset: 1633024800 (Unix timestamp)
 * - Retry-After: 60 (seconds, on 429 responses)
 *
 * ERROR HANDLING PATTERNS:
 *
 * Retryable Errors (client should retry):
 * - 429: Rate Limit (wait Retry-After seconds)
 * - 503: Service Unavailable (temporary)
 * - Network errors (connection timeout, etc.)
 *
 * Non-Retryable Errors (client should not retry):
 * - 400: Bad Request (fix request)
 * - 404: Not Found (resource doesn't exist)
 * - 409: Conflict (duplicate resource)
 *
 * Error Response Format:
 * {
 *   "error": {
 *     "code": "VALIDATION_ERROR",
 *     "message": "Invalid learning rate: must be between 1e-6 and 1e-2",
 *     "details": {
 *       "field": "learningRate",
 *       "value": 0.5,
 *       "constraints": { "min": 1e-6, "max": 1e-2 }
 *     },
 *     "retryable": false,
 *     "timestamp": "2025-10-05T12:34:56Z"
 *   }
 * }
 *
 * PERFORMANCE REQUIREMENTS:
 *
 * Response Times (p95):
 * - GET requests: <100ms
 * - POST requests: <500ms
 * - Job queueing: <2s
 * - WebSocket message delivery: <50ms
 *
 * Throughput:
 * - API requests: 100/sec (single Jetson)
 * - Concurrent WebSocket connections: 100
 * - Concurrent training jobs: 1 (GPU limitation)
 * - Feature extraction: Background, doesn't block API
 *
 * Storage:
 * - Models: 500MB - 5GB each (quantized)
 * - Datasets: 1GB - 50GB each
 * - Activations: 10GB - 100GB per extraction
 * - Checkpoints: 100MB - 1GB each
 * - Database: <10GB for metadata
 *
 * DATABASE SCHEMA OVERVIEW:
 * (See detailed schema in comments near type definitions)
 *
 * Note: Single-user system - no user authentication or authorization.
 * All resources are globally accessible.
 *
 * datasets:
 *   - id, name, source, size_bytes, status, file_path, created_at
 *
 * models:
 *   - id, name, architecture, params_count, quantization, file_path
 *
 * trainings:
 *   - id, model_id, dataset_id, encoder_type, hyperparameters (JSONB),
 *     status, current_step, total_steps, created_at, started_at, completed_at
 *
 * training_metrics (time-series):
 *   - id, training_id, step, loss, sparsity, dead_neurons, timestamp
 *   - INDEX on (training_id, step) for efficient queries
 *
 * checkpoints:
 *   - id, training_id, step, file_path, file_size_bytes, created_at
 *
 * features:
 *   - id, training_id, neuron_index, name, activation_frequency,
 *     interpretability_score, layer, created_at
 *
 * feature_activations (large table):
 *   - id, feature_id, dataset_sample_id, tokens (JSONB), activations (JSONB),
 *     max_activation
 *   - Partitioned by feature_id for performance
 *
 * steering_presets:
 *   - id, name, description, features (JSONB), intervention_layer
 *
 * WEBSOCKET PROTOCOL:
 *
 * Connection:
 *   ws://api.example.com/ws/training/:trainingId
 *
 * Client → Server Messages:
 *   { "type": "subscribe", "trainingId": "tr_abc123" }
 *   { "type": "ping" }
 *
 * Server → Client Messages:
 *   {
 *     "type": "training.progress",
 *     "data": {
 *       "step": 1234,
 *       "total_steps": 10000,
 *       "progress": 12.34,
 *       "metrics": { "loss": 0.342, "sparsity": 12.4 },
 *       "estimated_remaining_seconds": 427
 *     },
 *     "timestamp": "2025-10-05T12:34:56Z"
 *   }
 *
 *   { "type": "pong" }
 *
 *   {
 *     "type": "training.completed",
 *     "data": { "final_loss": 0.234, "total_time_seconds": 3600 }
 *   }
 *
 *   {
 *     "type": "training.error",
 *     "error": { "code": "GPU_OOM", "message": "GPU out of memory" }
 *   }
 *
 * Heartbeat: Client sends ping every 30s, server responds with pong.
 * Reconnection: Exponential backoff: 1s, 2s, 4s, 8s, max 30s
 *
 * SECURITY CONSIDERATIONS:
 *
 * 1. Input Validation:
 *    - Sanitize all inputs
 *    - Validate file uploads (size, type, magic bytes)
 *    - Prevent path traversal in file operations
 *
 * 2. Resource Limits:
 *    - Max file upload: 10GB
 *    - Max training steps: 1,000,000
 *    - Max concurrent jobs: 3 (system-wide)
 *    - Request timeout: 30s (except long-running jobs)
 *
 * 3. GPU Protection:
 *    - Monitor GPU memory before accepting jobs
 *    - Reject new trainings if GPU >90% utilized
 *    - Implement job preemption for priority tasks
 *
 * 4. Data Management:
 *    - Organized storage structure for models/datasets/checkpoints
 *    - WebSocket rooms per training job for progress updates
 *
 * MONITORING & OBSERVABILITY:
 *
 * Metrics to Track:
 * - API response times (histogram)
 * - Error rates by endpoint
 * - GPU utilization, memory, temperature
 * - Job queue depth
 * - WebSocket connection count
 * - Database query performance
 *
 * Logging:
 * - Structured logs (JSON format)
 * - Include: timestamp, request_id, endpoint, status_code, duration
 * - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
 *
 * Health Checks:
 * - GET /api/health → 200 OK (includes GPU status)
 * - GET /api/health/ready → 200 if accepting requests
 * - GET /api/health/live → 200 if process is running
 */

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Dataset representation from HuggingFace or other sources
 *
 * Backend API Contract:
 * - GET /api/datasets - List all datasets
 * - GET /api/datasets/:id - Get dataset details
 * - POST /api/datasets/download - Download from HuggingFace (body: { repoId, accessToken? })
 * - DELETE /api/datasets/:id - Delete dataset
 *
 * Status transitions: downloading -> ingesting -> ready
 * Real-time updates: Poll GET /api/datasets/:id for status updates every 2-5 seconds
 */
interface Dataset {
  id: string;
  name: string;
  source: string; // 'HuggingFace', 'Local', 'Custom'
  size: string; // Human-readable size (e.g., '2.3GB')
  status: 'downloading' | 'ingesting' | 'ready' | 'error';
  progress?: number; // 0-100, present during downloading/ingesting
  error?: string; // Error message if status is 'error'
}

/**
 * Model representation with quantization support
 *
 * Backend API Contract:
 * - GET /api/models - List all models
 * - GET /api/models/:id - Get model details with architecture
 * - POST /api/models/download - Download model (body: { repoId, quantization, accessToken? })
 * - DELETE /api/models/:id - Delete model
 *
 * Quantization options: 'Q4', 'Q8', 'FP16', 'FP32'
 * Status transitions: downloading -> loading -> ready
 */
interface Model {
  id: string;
  name: string;
  params: string; // Human-readable param count (e.g., '1.1B', '135M')
  quantized: string; // Quantization type
  memReq: string; // Memory requirement (e.g., '1.2GB')
  status: 'downloading' | 'loading' | 'ready' | 'error';
  progress?: number; // 0-100, present during downloading
  error?: string; // Error message if status is 'error'
}

/**
 * Training hyperparameters for SAE (Sparse Autoencoder) training
 *
 * Backend Validation:
 * - learningRate: 1e-6 to 1e-2
 * - batchSize: Power of 2, typically 32-512
 * - l1Coefficient: 1e-5 to 1e-1 (sparsity penalty)
 * - expansionFactor: 1-32 (hidden layer expansion)
 * - trainingSteps: 1000-1000000
 */
interface Hyperparameters {
  learningRate: number;
  batchSize: number;
  l1Coefficient: number;
  expansionFactor: number;
  trainingSteps: number;
  optimizer: 'AdamW' | 'Adam' | 'SGD';
  lrSchedule: 'constant' | 'cosine' | 'linear' | 'exponential';
  ghostGradPenalty: boolean; // Enable ghost gradient penalty for dead neurons
}

/**
 * Training metrics tracked during SAE training
 */
interface TrainingMetrics {
  loss: number | null;
  sparsity: number | null; // L0 sparsity (avg active features)
  reconstruction_error?: number;
  dead_neurons?: number;
  explained_variance?: number;
}

/**
 * Training job configuration and state
 *
 * Backend API Contract:
 * - POST /api/training/start - Start training (body: TrainingConfig)
 * - POST /api/training/:id/pause - Pause training
 * - POST /api/training/:id/resume - Resume training
 * - POST /api/training/:id/stop - Stop training permanently
 * - GET /api/training/:id/status - Get current status
 * - WebSocket /ws/training/:id - Real-time progress updates
 *
 * Status transitions:
 * initializing -> training -> completed
 *                  |  ^
 *                  v  |
 *                paused
 *                  |
 *                  v
 *                stopped
 *
 * Progress updates should stream via WebSocket with:
 * { progress: number, metrics: TrainingMetrics, timestamp: string }
 */
interface Training {
  id: string;
  model: string; // Model ID
  dataset: string; // Dataset ID
  encoderType: 'sparse' | 'vanilla' | 'topk'; // SAE architecture type
  status: 'initializing' | 'training' | 'paused' | 'stopped' | 'completed' | 'error';
  progress: number; // 0-100
  startTime: string; // ISO 8601 timestamp
  endTime?: string; // ISO 8601 timestamp
  hyperparameters: Hyperparameters;
  metrics: TrainingMetrics;
  error?: string;
}

/**
 * Training checkpoint for resuming or rollback
 *
 * Backend API Contract:
 * - GET /api/training/:trainingId/checkpoints - List checkpoints
 * - POST /api/training/:trainingId/checkpoint - Save checkpoint
 * - POST /api/training/:trainingId/checkpoint/:id/load - Load checkpoint
 * - DELETE /api/training/:trainingId/checkpoint/:id - Delete checkpoint
 *
 * Storage: Checkpoints should be stored in local file storage
 * with metadata in database for quick retrieval
 */
interface Checkpoint {
  id: string;
  trainingId: string;
  step: number; // Training step number
  loss: number;
  timestamp: string; // ISO 8601
  storageUrl?: string; // URL to checkpoint file in object storage
  size?: string; // File size
}

/**
 * Discovered feature from trained SAE
 *
 * Backend API Contract:
 * - GET /api/features?trainingId=:id - List features for training
 * - GET /api/features/:id - Get detailed feature info
 * - GET /api/features/:id/activations - Get max activating examples
 * - GET /api/features/:id/logit-lens - Get logit lens analysis
 * - PATCH /api/features/:id - Update feature (name, description, favorite)
 *
 * Feature extraction is a post-training analysis step
 */
interface Feature {
  id: number;
  trainingId?: string;
  name: string;
  description?: string;
  activation: number; // Average activation frequency (0-1)
  interpretability: number; // Interpretability score (0-1)
  layer?: number; // Which transformer layer
  neuronIndex?: number; // Index in SAE
  isFavorite?: boolean;
}

/**
 * Status of feature extraction process
 *
 * Feature extraction workflow:
 * 1. Run dataset through trained SAE
 * 2. Collect activation statistics
 * 3. Find max activating examples
 * 4. Compute interpretability scores
 *
 * Backend: Long-running job, use job queue (Celery/BullMQ)
 * Status polling: GET /api/extraction/:trainingId/status
 */
interface ExtractionStatus {
  status: 'idle' | 'extracting' | 'completed' | 'error';
  progress: number; // 0-100
  error?: string;
}

/**
 * Example text sample with activation information
 *
 * Used for "max activating examples" - texts where feature activates strongest
 */
interface ActivationExample {
  text: string;
  activation: number; // Activation strength for this sample
  tokens?: string[]; // Tokenized text
  tokenActivations?: number[]; // Per-token activation values
  highlights?: Array<{ start: number; end: number; value: number }>; // Token ranges to highlight
}

/**
 * Logit lens analysis for feature
 *
 * Shows which tokens the feature predicts/correlates with
 */
interface LogitLens {
  topTokens: string[];
  probabilities: number[];
  interpretation: string;
}

/**
 * Feature correlation with other features
 */
interface FeatureCorrelation {
  id: number;
  name: string;
  correlation: number; // -1 to 1
}

/**
 * Dataset sample/text for browsing and analysis
 *
 * Backend API Contract:
 * - GET /api/datasets/:id/samples?page=1&limit=50&split=train&search=query
 * - Pagination required for large datasets
 * - Full-text search should use database FTS or Elasticsearch
 */
interface DatasetSample {
  id: number;
  text: string;
  split: 'train' | 'validation' | 'test';
  metadata?: Record<string, any>; // Source, domain, etc.
  tokens?: string[]; // Tokenized representation
  tokenCount?: number;
}

/**
 * Dataset statistics for overview
 *
 * Computed during ingestion and cached
 */
interface DatasetStats {
  total_samples: number;
  total_tokens: number;
  avg_tokens_per_sample: number;
  unique_tokens: number;
  min_length: number;
  median_length: number;
  max_length: number;
}

/**
 * Distribution bucket for visualizations
 */
interface DistributionBucket {
  range: string; // e.g., '100-200'
  count: number;
}

/**
 * Model steering configuration
 *
 * Backend API Contract:
 * - POST /api/steering/generate - Generate steered output
 *   Body: { modelId, prompt, features: Array<{id, coefficient}>, interventionLayer, temperature }
 * - Response: { unsteeredOutput, steeredOutput, metrics }
 *
 * Steering applies feature vectors to model activations at specified layer
 * Coefficients typically range from -5.0 to +5.0
 */
interface SteeringConfig {
  selectedFeatures: Feature[];
  coefficients: Record<number, number>; // featureId -> coefficient
  interventionLayer: number; // Which transformer layer to intervene at
  temperature: number; // Sampling temperature (0.1-2.0)
}

/**
 * Comparison metrics between unsteered and steered outputs
 */
interface ComparisonMetrics {
  perplexityChange: number; // Perplexity delta
  activationMagnitude: number; // Total activation change
  semanticSimilarity: number; // Cosine similarity between outputs
}

/**
 * Tokenization settings for dataset processing
 *
 * Backend: Should match model's tokenizer exactly
 * Tokenizer loading: Use HuggingFace transformers library
 */
interface TokenizationSettings {
  maxLength: number; // Max sequence length
  truncation: boolean;
  padding: 'max_length' | 'longest' | 'do_not_pad';
  addSpecialTokens: boolean;
}

/**
 * Training configuration template (preset)
 *
 * Backend API Contract:
 * - GET /api/templates/training - List all training templates
 * - POST /api/templates/training - Create new template
 * - PUT /api/templates/training/:id - Update template
 * - DELETE /api/templates/training/:id - Delete template
 * - POST /api/templates/training/:id/apply - Load template into current config
 *
 * Templates allow users to save and reuse common training configurations.
 * Model and dataset can be null for generic templates that work with any model/dataset.
 */
interface TrainingTemplate {
  id: string;
  name: string;
  description?: string;
  model_id?: string | null;  // null = works with any model
  dataset_id?: string | null;  // null = works with any dataset
  encoder_type: 'sparse' | 'skip' | 'transcoder';
  hyperparameters: Hyperparameters;
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Feature extraction configuration template
 *
 * Backend API Contract:
 * - GET /api/templates/extraction - List extraction templates
 * - POST /api/templates/extraction - Create template
 * - PUT /api/templates/extraction/:id - Update template
 * - DELETE /api/templates/extraction/:id - Delete template
 */
interface ExtractionTemplate {
  id: string;
  name: string;
  description?: string;
  layers: number[];  // Which transformer layers to extract from
  hook_types: string[];  // ['residual', 'mlp', 'attention']
  max_samples?: number;  // Limit samples for testing
  top_k_examples: number;  // Max-activating examples per feature (typically 100)
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
}

/**
 * Steering preset (already in DB schema, adding interface)
 *
 * Backend API Contract:
 * - GET /api/steering/presets?trainingId=:id - List presets for training
 * - POST /api/steering/presets - Create preset
 * - PUT /api/steering/presets/:id - Update preset
 * - DELETE /api/steering/presets/:id - Delete preset
 * - POST /api/steering/presets/:id/apply - Apply preset to steering panel
 */
interface SteeringPreset {
  id: string;
  training_id: string;
  name: string;
  description?: string;
  features: Array<{ feature_id: number; coefficient: number }>;
  intervention_layer: number;
  temperature: number;
  is_favorite: boolean;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// Main App Component
export default function EmbeddedInterpretabilityUI() {
  const [activeTab, setActiveTab] = useState('datasets');
  const [datasets, setDatasets] = useState([]);
  const [models, setModels] = useState([]);
  const [trainings, setTrainings] = useState([]);
  const [checkpoints, setCheckpoints] = useState({});
  const [selectedConfig, setSelectedConfig] = useState({
    model: '',
    dataset: '',
    encoderType: 'sparse',
    hyperparameters: {
      learningRate: 1e-4,
      batchSize: 256,
      l1Coefficient: 1e-3,
      expansionFactor: 8,
      trainingSteps: 10000,
      optimizer: 'AdamW',
      lrSchedule: 'cosine',
      ghostGradPenalty: true
    }
  });

  // Template/Preset state management
  const [trainingTemplates, setTrainingTemplates] = useState<TrainingTemplate[]>([]);
  const [extractionTemplates, setExtractionTemplates] = useState<ExtractionTemplate[]>([]);
  const [steeringPresets, setSteeringPresets] = useState<SteeringPreset[]>([]);

  // Simulate API polling for training progress (OpenTelemetry pattern)
  useEffect(() => {
    const pollInterval = setInterval(async () => {
      setTrainings(prev => prev.map(t => {
        if (t.status === 'training' && t.progress < 100) {
          return { ...t, progress: Math.min(t.progress + Math.random() * 5, 100) };
        }
        if (t.status === 'training' && t.progress >= 100) {
          return { ...t, status: 'completed' };
        }
        return t;
      }));
    }, 2000);

    return () => clearInterval(pollInterval);
  }, []);

  // Load initial data and expose functions
  useEffect(() => {
    loadDatasets();
    loadModels();
    loadTemplates();

    // Expose functions globally for child components
    window.pauseTraining = pauseTraining;
    window.resumeTraining = resumeTraining;
    window.stopTraining = stopTraining;
    window.retryTraining = retryTraining;
    window.saveCheckpoint = saveCheckpoint;
    window.loadCheckpoint = loadCheckpoint;
    window.deleteCheckpoint = deleteCheckpoint;
    window.checkpoints = checkpoints;
    window.mockModels = models;
    window.mockDatasets = datasets;
    window.trainingTemplates = trainingTemplates;
    window.extractionTemplates = extractionTemplates;
    window.steeringPresets = steeringPresets;

    return () => {
      delete window.pauseTraining;
      delete window.resumeTraining;
      delete window.stopTraining;
      delete window.retryTraining;
      delete window.saveCheckpoint;
      delete window.loadCheckpoint;
      delete window.deleteCheckpoint;
      delete window.checkpoints;
      delete window.mockModels;
      delete window.mockDatasets;
      delete window.trainingTemplates;
      delete window.extractionTemplates;
      delete window.steeringPresets;
    };
  }, [checkpoints, models, datasets, trainingTemplates, extractionTemplates, steeringPresets]);

  const loadDatasets = async () => {
    setDatasets([
      { id: 'ds1', name: 'OpenWebText-10K', source: 'HuggingFace', size: '2.3GB', status: 'ready' },
      { id: 'ds2', name: 'TinyStories', source: 'HuggingFace', size: '450MB', status: 'ready' },
      { id: 'ds3', name: 'CodeParrot-Small', source: 'HuggingFace', size: '1.8GB', status: 'ingesting' }
    ]);
  };

  const loadModels = async () => {
    setModels([
      { id: 'm1', name: 'TinyLlama-1.1B', params: '1.1B', quantized: 'Q4', memReq: '1.2GB', status: 'ready' },
      { id: 'm2', name: 'Phi-2', params: '2.7B', quantized: 'Q4', memReq: '2.1GB', status: 'ready' },
      { id: 'm3', name: 'SmolLM-135M', params: '135M', quantized: 'FP16', memReq: '270MB', status: 'ready' }
    ]);
  };

  const loadTemplates = async () => {
    // Initialize training templates
    setTrainingTemplates([
      {
        id: 'tmpl_1',
        name: 'Fast Prototyping',
        description: 'Quick training with small expansion factor for rapid iteration',
        model_id: null,
        dataset_id: null,
        encoder_type: 'sparse',
        hyperparameters: {
          learningRate: 5e-4,
          batchSize: 128,
          l1Coefficient: 5e-4,
          expansionFactor: 4,
          trainingSteps: 5000,
          optimizer: 'AdamW',
          lrSchedule: 'constant',
          ghostGradPenalty: false
        },
        is_favorite: true,
        created_at: '2025-01-01T10:00:00Z',
        updated_at: '2025-01-01T10:00:00Z'
      },
      {
        id: 'tmpl_2',
        name: 'High Quality SAE',
        description: 'Production-quality sparse autoencoder with cosine schedule',
        model_id: 'm1',
        dataset_id: 'ds1',
        encoder_type: 'sparse',
        hyperparameters: {
          learningRate: 1e-4,
          batchSize: 256,
          l1Coefficient: 1e-3,
          expansionFactor: 8,
          trainingSteps: 10000,
          optimizer: 'AdamW',
          lrSchedule: 'cosine',
          ghostGradPenalty: true
        },
        is_favorite: true,
        created_at: '2025-01-02T14:30:00Z',
        updated_at: '2025-01-05T09:15:00Z'
      },
      {
        id: 'tmpl_3',
        name: 'Large Expansion',
        description: 'Very sparse with 16x expansion for detailed feature discovery',
        model_id: null,
        dataset_id: null,
        encoder_type: 'sparse',
        hyperparameters: {
          learningRate: 1e-4,
          batchSize: 512,
          l1Coefficient: 2e-3,
          expansionFactor: 16,
          trainingSteps: 20000,
          optimizer: 'AdamW',
          lrSchedule: 'linear',
          ghostGradPenalty: true
        },
        is_favorite: false,
        created_at: '2025-01-03T16:00:00Z',
        updated_at: '2025-01-03T16:00:00Z'
      }
    ]);

    // Initialize extraction templates
    setExtractionTemplates([
      {
        id: 'ext_tmpl_1',
        name: 'Quick Scan',
        description: 'Extract from key layers only',
        layers: [0, 6, 11],
        hook_types: ['residual'],
        max_samples: 5000,
        top_k_examples: 50,
        is_favorite: true,
        created_at: '2025-01-01T10:00:00Z',
        updated_at: '2025-01-01T10:00:00Z'
      },
      {
        id: 'ext_tmpl_2',
        name: 'Full Analysis',
        description: 'Comprehensive extraction from all layers',
        layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        hook_types: ['residual', 'mlp', 'attention'],
        max_samples: undefined,
        top_k_examples: 100,
        is_favorite: false,
        created_at: '2025-01-02T11:00:00Z',
        updated_at: '2025-01-02:11:00:00Z'
      }
    ]);

    // Initialize steering presets (would be loaded per training)
    setSteeringPresets([
      {
        id: 'steer_1',
        training_id: 'tr123',  // Example training ID
        name: 'Positive Sentiment',
        description: 'Boost positive sentiment features',
        features: [
          { feature_id: 42, coefficient: 3.5 },
          { feature_id: 128, coefficient: 2.0 },
          { feature_id: 456, coefficient: 1.5 }
        ],
        intervention_layer: 6,
        temperature: 1.0,
        is_favorite: true,
        created_at: '2025-01-01T10:00:00Z',
        updated_at: '2025-01-01T10:00:00Z'
      },
      {
        id: 'steer_2',
        training_id: 'tr123',
        name: 'Formal Tone',
        description: 'Increase formality and professional language',
        features: [
          { feature_id: 89, coefficient: 4.0 },
          { feature_id: 234, coefficient: 2.5 }
        ],
        intervention_layer: 8,
        temperature: 0.8,
        is_favorite: false,
        created_at: '2025-01-02T14:00:00Z',
        updated_at: '2025-01-02T14:00:00Z'
      }
    ]);
  };

  const startTraining = async () => {
    const newTraining = {
      id: `tr${Date.now()}`,
      ...selectedConfig,
      status: 'initializing',
      progress: 0,
      startTime: new Date().toISOString(),
      metrics: { loss: null, sparsity: null }
    };
    
    setTrainings(prev => [newTraining, ...prev]);
    setActiveTab('training');

    setTimeout(() => {
      setTrainings(prev => prev.map(t => 
        t.id === newTraining.id ? { ...t, status: 'training' } : t
      ));
    }, 1000);
  };

  const pauseTraining = (trainingId: string) => {
    setTrainings(prev => prev.map((t: any) =>
      t.id === trainingId ? { ...t, status: 'paused' } : t
    ));
  };

  const resumeTraining = (trainingId: string) => {
    setTrainings(prev => prev.map((t: any) =>
      t.id === trainingId ? { ...t, status: 'training' } : t
    ));
  };

  const stopTraining = (trainingId: string) => {
    setTrainings(prev => prev.map((t: any) =>
      t.id === trainingId ? { ...t, status: 'stopped', progress: t.progress } : t
    ));
  };

  const retryTraining = (trainingId: string) => {
    setTrainings(prev => prev.map((t: any) =>
      t.id === trainingId ? { ...t, status: 'training', progress: 0 } : t
    ));
  };

  const saveCheckpoint = (trainingId: string) => {
    const training = trainings.find((t: any) => t.id === trainingId);
    if (!training) return;

    const checkpoint = {
      id: `cp${Date.now()}`,
      trainingId,
      step: Math.floor(training.progress * 100),
      loss: training.progress > 10 ? (0.5 - training.progress * 0.003) : 0,
      timestamp: new Date().toISOString()
    };

    setCheckpoints(prev => ({
      ...prev,
      [trainingId]: [...(prev[trainingId] || []), checkpoint]
    }));
  };

  const loadCheckpoint = (trainingId: string, checkpointId: string) => {
    const trainingCheckpoints = checkpoints[trainingId] || [];
    const checkpoint = trainingCheckpoints.find((cp: any) => cp.id === checkpointId);
    if (!checkpoint) return;

    setTrainings(prev => prev.map((t: any) =>
      t.id === trainingId ? { ...t, progress: checkpoint.step, status: 'paused' } : t
    ));
  };

  const deleteCheckpoint = (trainingId: string, checkpointId: string) => {
    setCheckpoints(prev => ({
      ...prev,
      [trainingId]: (prev[trainingId] || []).filter((cp: any) => cp.id !== checkpointId)
    }));
  };

  const downloadFromHF = async (repoId: string, accessToken: string = '') => {
    const newDataset = {
      id: `ds${Date.now()}`,
      name: repoId,
      source: 'HuggingFace',
      size: 'Downloading...',
      status: 'downloading'
    };
    setDatasets(prev => [...prev, newDataset]);
    // In real implementation, pass accessToken to backend API
    // POST /api/datasets/download with body: { repoId, accessToken }
  };

  const downloadModel = async (repoId: string, quantization: string, accessToken: string = '') => {
    const newModel = {
      id: `m${Date.now()}`,
      name: repoId.split('/').pop(),
      params: 'Unknown',
      quantized: quantization,
      memReq: 'Calculating...',
      status: 'downloading',
      progress: 0
    };
    setModels(prev => [...prev, newModel]);

    const progressInterval = setInterval(() => {
      setModels(prev => prev.map(m => {
        if (m.id === newModel.id && m.status === 'downloading') {
          const newProgress = Math.min(m.progress + Math.random() * 15, 100);
          if (newProgress >= 100) {
            clearInterval(progressInterval);
            return { ...m, status: 'quantizing', progress: 100 };
          }
          return { ...m, progress: newProgress };
        }
        if (m.id === newModel.id && m.status === 'quantizing') {
          clearInterval(progressInterval);
          setTimeout(() => {
            setModels(prev => prev.map(model => 
              model.id === newModel.id 
                ? { ...model, status: 'ready', memReq: '1.5GB', params: '1.3B' }
                : model
            ));
          }, 2000);
        }
        return m;
      }));
    }, 500);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Cpu className="w-8 h-8 text-emerald-400" />
            <div>
              <h1 className="text-xl font-semibold">MechInterp Studio</h1>
              <p className="text-sm text-slate-400">Edge AI Feature Discovery Platform</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-sm text-slate-400">
              <span className="text-emerald-400">●</span> API Connected
            </div>
            <div className="text-sm text-slate-400">Jetson Orin Nano</div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="border-b border-slate-800 bg-slate-900/30">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-8">
            {['datasets', 'models', 'training', 'features', 'steering'].map(tab => (
              <button
                type="button"
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
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'datasets' && (
          <DatasetsPanel datasets={datasets} onDownload={downloadFromHF} />
        )}
        {activeTab === 'models' && (
          <ModelsPanel models={models} onDownloadModel={downloadModel} />
        )}
        {activeTab === 'training' && (
          <TrainingPanel
            trainings={trainings}
            models={models}
            datasets={datasets}
            selectedConfig={selectedConfig}
            setSelectedConfig={setSelectedConfig}
            onStartTraining={startTraining}
            checkpoints={checkpoints}
          />
        )}
        {activeTab === 'features' && (
          <FeaturesPanel trainings={trainings} />
        )}
        {activeTab === 'steering' && (
          <SteeringPanel models={models} />
        )}
      </main>
    </div>
  );
}

// Datasets Panel Component
function DatasetsPanel({ datasets, onDownload }) {
  const [hfRepo, setHfRepo] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [selectedDataset, setSelectedDataset] = useState(null);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Dataset Management</h2>

      {/* Download from HuggingFace Form */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Download from HuggingFace</h3>
        <div className="space-y-4">
          <div>
            <label htmlFor="dataset-repo-input" className="block text-sm font-medium text-slate-300 mb-2">
              Dataset Repository
            </label>
            <input
              id="dataset-repo-input"
              type="text"
              placeholder="e.g., roneneldan/TinyStories"
              value={hfRepo}
              onChange={(e) => setHfRepo(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div>
            <label htmlFor="dataset-token-input" className="block text-sm font-medium text-slate-300 mb-2">
              Access Token (optional, for gated datasets)
            </label>
            <input
              id="dataset-token-input"
              type="password"
              placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm"
            />
          </div>

          <button
            type="button"
            onClick={() => {
              onDownload(hfRepo, accessToken);
              setHfRepo('');
              setAccessToken('');
            }}
            disabled={!hfRepo}
            className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
          >
            <Download className="w-5 h-5" />
            Download Dataset from HuggingFace
          </button>
        </div>
      </div>

      <div className="grid gap-4">
        {datasets.map((ds: any) => (
          <div
            key={ds.id}
            onClick={() => ds.status === 'ready' && setSelectedDataset(ds)}
            className={`bg-slate-900/50 border border-slate-800 rounded-lg p-6 ${ds.status === 'ready' ? 'cursor-pointer hover:bg-slate-900/70 transition-colors' : ''}`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Database className="w-8 h-8 text-blue-400" />
                <div>
                  <h3 className="font-semibold text-lg">{ds.name}</h3>
                  <p className="text-sm text-slate-400">Source: {ds.source} • Size: {ds.size}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {ds.status === 'ready' && <CheckCircle className="w-5 h-5 text-emerald-400" />}
                {ds.status === 'downloading' && <Loader className="w-5 h-5 text-blue-400 animate-spin" />}
                {ds.status === 'ingesting' && <Activity className="w-5 h-5 text-yellow-400" />}
                <span className="text-sm capitalize px-3 py-1 bg-slate-800 rounded-full">
                  {ds.status}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Dataset Detail Modal */}
      {selectedDataset && (
        <DatasetDetailModal
          dataset={selectedDataset}
          onClose={() => setSelectedDataset(null)}
        />
      )}
    </div>
  );
}

// Models Panel Component
function ModelsPanel({ models, onDownloadModel }) {
  const [hfModelRepo, setHfModelRepo] = useState('');
  const [quantization, setQuantization] = useState('Q4');
  const [accessToken, setAccessToken] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);
  const [showExtractionConfig, setShowExtractionConfig] = useState(false);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold mb-4">Model Management</h2>
        <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                HuggingFace Model Repository
              </label>
              <input
                type="text"
                placeholder="e.g., TinyLlama/TinyLlama-1.1B"
                value={hfModelRepo}
                onChange={(e) => setHfModelRepo(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label htmlFor="model-quantization" className="block text-sm font-medium text-slate-300 mb-2">
                Quantization Format
              </label>
              <select
                id="model-quantization"
                value={quantization}
                onChange={(e) => setQuantization(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              >
                <option value="FP16">FP16 (Full Precision)</option>
                <option value="Q8">Q8 (8-bit)</option>
                <option value="Q4">Q4 (4-bit)</option>
                <option value="Q2">Q2 (2-bit)</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Access Token <span className="text-slate-500">(optional, for gated models)</span>
            </label>
            <input
              type="password"
              placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm"
            />
            <p className="mt-1 text-xs text-slate-500">
              Required for gated models like Llama, Gemma, or other restricted access models
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              onDownloadModel(hfModelRepo, quantization, accessToken);
              setHfModelRepo('');
              setAccessToken('');
            }}
            disabled={!hfModelRepo}
            className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
          >
            <Download className="w-5 h-5" />
            Download Model from HuggingFace
          </button>
        </div>
      </div>

      <div className="grid gap-4">
        {models.map((model: any) => (
          <div key={model.id} className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div
                className="flex items-center gap-4 cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => setSelectedModel(model)}
              >
                <Cpu className="w-8 h-8 text-purple-400" />
                <div>
                  <h3 className="font-semibold text-lg">{model.name}</h3>
                  <p className="text-sm text-slate-400">
                    {model.params} params • {model.quantized} quantization • {model.memReq} memory
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {model.status === 'ready' && (
                  <button
                    type="button"
                    onClick={() => setShowExtractionConfig(true)}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm font-medium transition-colors"
                  >
                    Extract Activations
                  </button>
                )}
                {model.status === 'ready' && <CheckCircle className="w-5 h-5 text-emerald-400" />}
                {model.status === 'downloading' && <Loader className="w-5 h-5 text-blue-400 animate-spin" />}
                {model.status === 'quantizing' && <Activity className="w-5 h-5 text-yellow-400" />}
                <span className={`px-3 py-1 rounded-full text-sm ${
                  model.status === 'ready'
                    ? 'bg-emerald-900/30 text-emerald-400'
                    : 'bg-slate-800 text-slate-300 capitalize'
                }`}>
                  {model.status || 'Edge-Ready'}
                </span>
              </div>
            </div>
            {model.status === 'downloading' && model.progress !== undefined && (
              <div className="mt-4 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Download Progress</span>
                  <span className="text-emerald-400 font-medium">{model.progress.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-500 progress-bar-model"
                    style={{ '--width': `${model.progress}%` } as React.CSSProperties}
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {selectedModel && (
        <ModelArchitectureViewer model={selectedModel} onClose={() => setSelectedModel(null)} />
      )}

      {showExtractionConfig && (
        <ActivationExtractionConfig onClose={() => setShowExtractionConfig(false)} />
      )}
    </div>
  );
}

// Model Architecture Viewer Modal
function ModelArchitectureViewer({ model, onClose }) {
  // Mock model architecture data
  const modelLayers = [
    { type: 'Embedding', size: '50257 × 768' },
    ...Array.from({ length: 12 }, (_, i) => ({
      type: `TransformerBlock_${i}`,
      attention: '12 heads × 64 dims',
      mlp: '768 → 3072 → 768'
    })),
    { type: 'LayerNorm', size: '768' },
    { type: 'Output', size: '768 × 50257' }
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold text-emerald-400">{model.name} Architecture</h2>
            <p className="text-sm text-slate-400 mt-1">{model.params} parameters • {model.quantized} quantization</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {/* Model Overview Stats */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Total Layers</div>
              <div className="text-2xl font-semibold text-emerald-400">{modelLayers.length}</div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Hidden Dimension</div>
              <div className="text-2xl font-semibold text-purple-400">768</div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Attention Heads</div>
              <div className="text-2xl font-semibold text-blue-400">12</div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Parameters</div>
              <div className="text-2xl font-semibold text-yellow-400">{model.params}</div>
            </div>
          </div>

          {/* Layer List */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold mb-3">Model Layers</h3>
            {modelLayers.map((layer, idx) => (
              <div key={idx} className="bg-slate-800/30 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-colors">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <span className="text-slate-500 font-mono text-sm w-8">{idx}</span>
                    <div>
                      <div className="font-medium text-slate-200">{layer.type}</div>
                      {'attention' in layer && (
                        <div className="text-sm text-slate-400 mt-1">
                          Attention: {layer.attention} | MLP: {layer.mlp}
                        </div>
                      )}
                      {'size' in layer && (
                        <div className="text-sm text-slate-400 mt-1">Shape: {layer.size}</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Model Configuration */}
          <div className="mt-6 bg-slate-800/30 border border-slate-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3">Model Configuration</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div><span className="text-slate-400">Vocabulary Size:</span> <span className="font-mono ml-2">50257</span></div>
              <div><span className="text-slate-400">Max Position:</span> <span className="font-mono ml-2">1024</span></div>
              <div><span className="text-slate-400">MLP Ratio:</span> <span className="font-mono ml-2">4x</span></div>
              <div><span className="text-slate-400">Architecture:</span> <span className="font-mono ml-2">Decoder-only</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Activation Extraction Configuration Modal
function ActivationExtractionConfig({ onClose }) {
  const [selectedDataset, setSelectedDataset] = useState('ds1');
  const [selectedLayers, setSelectedLayers] = useState([0, 5, 11]);
  const [activationType, setActivationType] = useState('residual');
  const [batchSize, setBatchSize] = useState(32);
  const [maxSamples, setMaxSamples] = useState(1000);
  const [extracting, setExtracting] = useState(false);
  const [progress, setProgress] = useState(0);

  const datasets = [
    { id: 'ds1', name: 'OpenWebText-10K', samples: 10000 },
    { id: 'ds2', name: 'TinyStories', samples: 8500 },
    { id: 'ds3', name: 'CodeParrot-Small', samples: 5200 }
  ];

  const layers = Array.from({ length: 12 }, (_, i) => i);

  const toggleLayer = (layer: number) => {
    if (selectedLayers.includes(layer)) {
      setSelectedLayers(selectedLayers.filter((l: number) => l !== layer));
    } else {
      setSelectedLayers([...selectedLayers, layer].sort((a: number, b: number) => a - b));
    }
  };

  const startExtraction = async () => {
    setExtracting(true);
    setProgress(0);

    // Simulate extraction progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setExtracting(false);
            setProgress(0);
            onClose();
          }, 1000);
          return 100;
        }
        return prev + Math.random() * 10;
      });
    }, 500);
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-3xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <h2 className="text-2xl font-semibold text-emerald-400">Extract Activations</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Dataset Selection */}
          <div>
            <label htmlFor="extraction-dataset" className="block text-sm font-medium text-slate-300 mb-2">Select Dataset</label>
            <select
              id="extraction-dataset"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              {datasets.map((ds: any) => (
                <option key={ds.id} value={ds.id}>{ds.name} ({ds.samples} samples)</option>
              ))}
            </select>
          </div>

          {/* Layer Selection */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-300">Select Layers</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedLayers(layers)}
                  className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedLayers([])}
                  className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <div className="grid grid-cols-6 gap-2">
              {layers.map(layer => (
                <button
                  type="button"
                  key={layer}
                  onClick={() => toggleLayer(layer)}
                  className={`px-3 py-2 rounded font-mono text-sm transition-colors ${
                    selectedLayers.includes(layer)
                      ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                  }`}
                >
                  L{layer}
                </button>
              ))}
            </div>
          </div>

          {/* Activation Type */}
          <div>
            <label htmlFor="extraction-activation-type" className="block text-sm font-medium text-slate-300 mb-2">Activation Type</label>
            <select
              id="extraction-activation-type"
              value={activationType}
              onChange={(e) => setActivationType(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="residual">Residual Stream</option>
              <option value="mlp">MLP Output</option>
              <option value="attention">Attention Output</option>
            </select>
          </div>

          {/* Extraction Settings */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="extraction-batch-size" className="block text-sm font-medium text-slate-300 mb-2">Batch Size</label>
              <input
                id="extraction-batch-size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label htmlFor="extraction-max-samples" className="block text-sm font-medium text-slate-300 mb-2">Max Samples</label>
              <input
                id="extraction-max-samples"
                type="number"
                value={maxSamples}
                onChange={(e) => setMaxSamples(parseInt(e.target.value))}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>

          {/* Extraction Progress */}
          {extracting && (
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-slate-300">Extracting Activations...</span>
                <span className="text-sm font-mono text-emerald-400">{progress.toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300 progress-bar-upload"
                  style={{ '--width': `${progress}%` } as React.CSSProperties}
                />
              </div>
            </div>
          )}
        </div>

        <div className="border-t border-slate-800 p-6">
          <button
            type="button"
            onClick={startExtraction}
            disabled={selectedLayers.length === 0 || extracting}
            className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {extracting ? 'Extracting...' : 'Start Extraction'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Training Panel Component - WITH ADVANCED HYPERPARAMETERS
function TrainingPanel({ trainings, models, datasets, selectedConfig, setSelectedConfig, onStartTraining, checkpoints }) {
  const readyDatasets = datasets.filter((d: any) => d.status === 'ready');
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);

  const updateHyperparameter = (key: string, value: any) => {
    setSelectedConfig({
      ...selectedConfig,
      hyperparameters: {
        ...selectedConfig.hyperparameters,
        [key]: value
      }
    });
  };

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Training Configuration</h2>
      
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label htmlFor="training-dataset" className="block text-sm font-medium text-slate-300 mb-2">Dataset</label>
            <select
              id="training-dataset"
              value={selectedConfig.dataset}
              onChange={(e) => setSelectedConfig({ ...selectedConfig, dataset: e.target.value })}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="">Select dataset...</option>
              {readyDatasets.map((d: any) => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="training-model" className="block text-sm font-medium text-slate-300 mb-2">Model</label>
            <select
              id="training-model"
              value={selectedConfig.model}
              onChange={(e) => setSelectedConfig({ ...selectedConfig, model: e.target.value })}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="">Select model...</option>
              {models.filter((m: any) => m.status === 'ready').map((m: any) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="training-encoder-type" className="block text-sm font-medium text-slate-300 mb-2">Encoder Type</label>
            <select
              id="training-encoder-type"
              value={selectedConfig.encoderType}
              onChange={(e) => setSelectedConfig({ ...selectedConfig, encoderType: e.target.value })}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="sparse">Sparse Autoencoder</option>
              <option value="skip">Skip Autoencoder</option>
              <option value="transcoder">Transcoder</option>
            </select>
          </div>
        </div>

        {/* Advanced Configuration Toggle */}
        <button
          type="button"
          onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
          className="w-full px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg flex items-center justify-between transition-colors"
        >
          <span className="font-medium">Advanced Hyperparameters</span>
          <svg className={`w-5 h-5 transition-transform ${showAdvancedConfig ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Advanced Hyperparameters */}
        {showAdvancedConfig && (
          <div className="border-t border-slate-700 pt-4 space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="hyperparam-learning-rate" className="block text-sm font-medium text-slate-300 mb-2">Learning Rate</label>
                <input
                  id="hyperparam-learning-rate"
                  type="number"
                  value={selectedConfig.hyperparameters.learningRate}
                  onChange={(e) => updateHyperparameter('learningRate', parseFloat(e.target.value))}
                  step="0.00001"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label htmlFor="hyperparam-batch-size" className="block text-sm font-medium text-slate-300 mb-2">Batch Size</label>
                <select
                  id="hyperparam-batch-size"
                  value={selectedConfig.hyperparameters.batchSize}
                  onChange={(e) => updateHyperparameter('batchSize', parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                >
                  <option value="64">64</option>
                  <option value="128">128</option>
                  <option value="256">256</option>
                  <option value="512">512</option>
                  <option value="1024">1024</option>
                  <option value="2048">2048</option>
                </select>
              </div>
              <div>
                <label htmlFor="hyperparam-l1-coefficient" className="block text-sm font-medium text-slate-300 mb-2">L1 Coefficient (λ)</label>
                <input
                  id="hyperparam-l1-coefficient"
                  type="number"
                  value={selectedConfig.hyperparameters.l1Coefficient}
                  onChange={(e) => updateHyperparameter('l1Coefficient', parseFloat(e.target.value))}
                  step="0.00001"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label htmlFor="hyperparam-expansion-factor" className="block text-sm font-medium text-slate-300 mb-2">Expansion Factor</label>
                <select
                  id="hyperparam-expansion-factor"
                  value={selectedConfig.hyperparameters.expansionFactor}
                  onChange={(e) => updateHyperparameter('expansionFactor', parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                >
                  <option value="4">4x</option>
                  <option value="8">8x</option>
                  <option value="16">16x</option>
                  <option value="32">32x</option>
                </select>
              </div>
              <div>
                <label htmlFor="hyperparam-training-steps" className="block text-sm font-medium text-slate-300 mb-2">Training Steps</label>
                <input
                  id="hyperparam-training-steps"
                  type="number"
                  value={selectedConfig.hyperparameters.trainingSteps}
                  onChange={(e) => updateHyperparameter('trainingSteps', parseInt(e.target.value))}
                  min="1000"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
              <div>
                <label htmlFor="hyperparam-optimizer" className="block text-sm font-medium text-slate-300 mb-2">Optimizer</label>
                <select
                  id="hyperparam-optimizer"
                  value={selectedConfig.hyperparameters.optimizer}
                  onChange={(e) => updateHyperparameter('optimizer', e.target.value)}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                >
                  <option value="Adam">Adam</option>
                  <option value="AdamW">AdamW</option>
                  <option value="SGD">SGD</option>
                </select>
              </div>
              <div>
                <label htmlFor="hyperparam-lr-schedule" className="block text-sm font-medium text-slate-300 mb-2">LR Schedule</label>
                <select
                  id="hyperparam-lr-schedule"
                  value={selectedConfig.hyperparameters.lrSchedule}
                  onChange={(e) => updateHyperparameter('lrSchedule', e.target.value)}
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                >
                  <option value="constant">Constant</option>
                  <option value="linear">Linear</option>
                  <option value="cosine">Cosine</option>
                  <option value="exponential">Exponential</option>
                </select>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg">
                <label className="text-sm font-medium text-slate-300">Ghost Gradient Penalty</label>
                <button
                  type="button"
                  onClick={() => updateHyperparameter('ghostGradPenalty', !selectedConfig.hyperparameters.ghostGradPenalty)}
                  className={`relative w-12 h-6 rounded-full transition-colors ${
                    selectedConfig.hyperparameters.ghostGradPenalty ? 'bg-emerald-600' : 'bg-slate-600'
                  }`}
                  aria-label={`Toggle ghost gradient penalty ${selectedConfig.hyperparameters.ghostGradPenalty ? 'off' : 'on'}`}
                >
                  <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                    selectedConfig.hyperparameters.ghostGradPenalty ? 'translate-x-7' : 'translate-x-1'
                  }`} />
                </button>
              </div>
            </div>
          </div>
        )}
        
        <button
          type="button"
          onClick={onStartTraining}
          disabled={!selectedConfig.model || !selectedConfig.dataset}
          className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium"
        >
          <Play className="w-5 h-5" />
          Start Training
        </button>
      </div>

      {/* Active Trainings */}
      <div className="space-y-4">
        <h3 className="text-xl font-semibold">Training Jobs</h3>
        {trainings.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            No training jobs yet. Configure and start training above.
          </div>
        ) : (
          trainings.map((training: any) => (
            <TrainingCard key={training.id} training={training} models={models} datasets={datasets} />
          ))
        )}
      </div>
    </div>
  );
}

// Enhanced Training Card with Metrics, Logs, Controls, and Checkpoints
function TrainingCard({ training, models, datasets }) {
  const model = models.find((m: any) => m.id === training.model);
  const dataset = datasets.find((d: any) => d.id === training.dataset);
  const [showMetrics, setShowMetrics] = useState(false);
  const [showCheckpoints, setShowCheckpoints] = useState(false);
  const [autoSave, setAutoSave] = useState(false);
  const [autoSaveInterval, setAutoSaveInterval] = useState(1000);

  const currentLoss = training.progress > 10 ? (0.5 - training.progress * 0.003) : 0;
  const l0Sparsity = training.progress > 10 ? (45 + training.progress * 0.1) : 0;
  const deadNeurons = training.progress > 10 ? Math.max(200 - training.progress * 1.5, 10) : 0;
  const gpuUtil = training.status === 'training' ? 75 + Math.random() * 15 : 0;

  const trainingCheckpoints = window.checkpoints?.[training.id] || [];

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-semibold text-lg">{model?.name} + {dataset?.name}</h4>
          <p className="text-sm text-slate-400">Encoder: {training.encoderType} • Started: {new Date(training.startTime).toLocaleTimeString()}</p>
        </div>
        <div className="flex items-center gap-3">
          {training.status === 'training' && <Activity className="w-5 h-5 text-emerald-400 animate-pulse" />}
          {training.status === 'completed' && <CheckCircle className="w-5 h-5 text-emerald-400" />}
          {training.status === 'paused' && <svg className="w-5 h-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20"><path d="M5 4a2 2 0 012-2h2a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V4zm8 0a2 2 0 012-2h2a2 2 0 012 2v12a2 2 0 01-2 2h-2a2 2 0 01-2-2V4z" /></svg>}
          {training.status === 'initializing' && <Loader className="w-5 h-5 text-blue-400 animate-spin" />}
          <span className="capitalize px-3 py-1 bg-slate-800 rounded-full text-sm">
            {training.status}
          </span>
        </div>
      </div>

      {(training.status === 'training' || training.status === 'completed' || training.status === 'paused') && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Training Progress</span>
            <span className="text-emerald-400 font-medium">{training.progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500 progress-bar-training"
              style={{ '--width': `${training.progress}%` } as React.CSSProperties}
            />
          </div>
          
          <div className="grid grid-cols-4 gap-3 pt-2">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Loss</div>
              <div className="text-lg font-semibold text-emerald-400">
                {training.progress > 10 ? currentLoss.toFixed(4) : '—'}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">L0 Sparsity</div>
              <div className="text-lg font-semibold text-blue-400">
                {training.progress > 10 ? l0Sparsity.toFixed(1) : '—'}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Dead Neurons</div>
              <div className="text-lg font-semibold text-red-400">
                {training.progress > 10 ? Math.floor(deadNeurons) : '—'}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">GPU Util</div>
              <div className="text-lg font-semibold text-purple-400">
                {training.status === 'training' ? `${gpuUtil.toFixed(0)}%` : '—'}
              </div>
            </div>
          </div>

          {training.status === 'training' && (
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setShowMetrics(!showMetrics)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Activity className="w-4 h-4" />
                <span>{showMetrics ? 'Hide' : 'Show'} Live Metrics</span>
              </button>
              <button
                type="button"
                onClick={() => setShowCheckpoints(!showCheckpoints)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                </svg>
                <span>Checkpoints ({trainingCheckpoints.length})</span>
              </button>
            </div>
          )}

          {showCheckpoints && (
            <div className="border-t border-slate-700 pt-4 mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h5 className="text-sm font-medium text-slate-300">Checkpoint Management</h5>
                <button
                  type="button"
                  onClick={() => window.saveCheckpoint?.(training.id)}
                  className="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 rounded text-sm flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4" />
                  </svg>
                  Save Now
                </button>
              </div>

              {trainingCheckpoints.length > 0 ? (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {trainingCheckpoints.map((cp: any) => (
                    <div key={cp.id} className="flex items-center justify-between bg-slate-800/30 p-3 rounded">
                      <div>
                        <div className="font-medium text-sm">Step {cp.step}</div>
                        <div className="text-xs text-slate-400">
                          Loss: {cp.loss.toFixed(4)} • {new Date(cp.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => window.loadCheckpoint?.(training.id, cp.id)}
                          className="p-1 hover:bg-slate-700 rounded"
                          title="Load"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => window.deleteCheckpoint?.(training.id, cp.id)}
                          className="p-1 hover:bg-red-900/30 text-red-400 rounded"
                          title="Delete"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-slate-500 text-sm">
                  No checkpoints saved yet
                </div>
              )}

              <div className="border-t border-slate-700 pt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">Auto-save every N steps</label>
                  <button
                    type="button"
                    onClick={() => setAutoSave(!autoSave)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      autoSave ? 'bg-emerald-600' : 'bg-slate-600'
                    }`}
                    aria-label={`Toggle auto-save ${autoSave ? 'off' : 'on'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      autoSave ? 'translate-x-7' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
                {autoSave && (
                  <input
                    type="number"
                    value={autoSaveInterval}
                    onChange={(e) => setAutoSaveInterval(parseInt(e.target.value))}
                    min="100"
                    max="10000"
                    step="100"
                    aria-label="Auto-save interval in steps"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-sm"
                    placeholder="Interval (steps)"
                  />
                )}
              </div>
            </div>
          )}

          {showMetrics && training.status === 'training' && (
            <div className="border-t border-slate-700 pt-4 mt-4 space-y-4">
              <div className="bg-slate-800/30 rounded-lg p-4">
                <h5 className="text-sm font-medium text-slate-300 mb-3">Loss Curve</h5>
                <div className="h-24 flex items-end gap-1">
                  {Array.from({ length: 20 }, (_, i) => {
                    const height = Math.max(10, 100 - i * 3 - Math.random() * 10);
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-emerald-500 rounded-t chart-bar-emerald"
                        style={{ '--height': `${height}%` } as React.CSSProperties}
                      />
                    );
                  })}
                </div>
              </div>

              <div className="bg-slate-800/30 rounded-lg p-4">
                <h5 className="text-sm font-medium text-slate-300 mb-3">L0 Sparsity</h5>
                <div className="h-24 flex items-end gap-1">
                  {Array.from({ length: 20 }, (_, i) => {
                    const height = Math.min(90, 30 + i * 2 + Math.random() * 10);
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-blue-500 rounded-t chart-bar-blue"
                        style={{ '--height': `${height}%` } as React.CSSProperties}
                      />
                    );
                  })}
                </div>
              </div>

              <div className="bg-slate-950 rounded-lg p-4 font-mono text-xs">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-400">Training Logs</span>
                  <span className="text-emerald-400 text-xs">Live</span>
                </div>
                <div className="h-32 overflow-y-auto space-y-1">
                  <div className="text-slate-300">
                    <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> Step {Math.floor(training.progress * 100)}: loss={currentLoss.toFixed(4)}
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> L0 sparsity: {l0Sparsity.toFixed(1)}
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> Dead neurons: {Math.floor(deadNeurons)}/8192
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">[{new Date().toLocaleTimeString()}]</span> GPU utilization: {gpuUtil.toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {training.status !== 'completed' && (
        <div className="border-t border-slate-700 pt-4 flex gap-2">
          {training.status === 'training' && (
            <>
              <button
                type="button"
                onClick={() => window.pauseTraining?.(training.id)}
                className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5 4a2 2 0 012-2h2a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V4zm8 0a2 2 0 012-2h2a2 2 0 012 2v12a2 2 0 01-2 2h-2a2 2 0 01-2-2V4z" />
                </svg>
                Pause
              </button>
              <button
                type="button"
                onClick={() => window.stopTraining?.(training.id)}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                </svg>
                Stop
              </button>
            </>
          )}
          
          {training.status === 'paused' && (
            <>
              <button
                type="button"
                onClick={() => window.resumeTraining?.(training.id)}
                className="flex-1 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Resume
              </button>
              <button
                type="button"
                onClick={() => window.stopTraining?.(training.id)}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                </svg>
                Stop
              </button>
            </>
          )}

          {training.status === 'stopped' && (
            <button
              type="button"
              onClick={() => window.retryTraining?.(training.id)}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Features Panel Component - ENHANCED with Feature Extraction and Browser
function FeaturesPanel({ trainings }) {
  const [selectedTraining, setSelectedTraining] = useState(null);
  const [extractionStatus, setExtractionStatus] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('activation_freq');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [favoritedFeatures, setFavoritedFeatures] = useState(new Set());

  const completedTrainings = trainings.filter((t: any) => t.status === 'completed');

  const mockFeatures = [
    {
      id: 1,
      name: 'Sentiment Positive',
      activation: 0.23,
      interpretability: 0.94,
      exampleTokens: ['I', 'really', 'love', 'this', 'amazing', 'product'],
      exampleActivations: [0.02, 0.05, 0.89, 0.03, 0.78, 0.04]
    },
    {
      id: 2,
      name: 'Code Syntax',
      activation: 0.18,
      interpretability: 0.89,
      exampleTokens: ['def', 'function', '(', 'arg', '):', 'return'],
      exampleActivations: [0.92, 0.15, 0.87, 0.12, 0.85, 0.76]
    },
    {
      id: 3,
      name: 'Question Pattern',
      activation: 0.31,
      interpretability: 0.92,
      exampleTokens: ['What', 'is', 'the', 'answer', 'to', 'this', '?'],
      exampleActivations: [0.88, 0.15, 0.08, 0.12, 0.09, 0.11, 0.91]
    },
    {
      id: 4,
      name: 'Temporal Reference',
      activation: 0.15,
      interpretability: 0.87,
      exampleTokens: ['We', 'met', 'yesterday', 'at', 'noon'],
      exampleActivations: [0.05, 0.08, 0.92, 0.04, 0.79]
    },
    {
      id: 5,
      name: 'Negation Logic',
      activation: 0.28,
      interpretability: 0.91,
      exampleTokens: ['This', 'is', 'not', 'correct', 'at', 'all'],
      exampleActivations: [0.06, 0.09, 0.94, 0.23, 0.07, 0.15]
    },
    {
      id: 6,
      name: 'Pronouns First Person',
      activation: 0.19,
      interpretability: 0.88,
      exampleTokens: ['I', 'think', 'my', 'opinion', 'matters'],
      exampleActivations: [0.91, 0.12, 0.87, 0.15, 0.08]
    },
    {
      id: 7,
      name: 'Math Operators',
      activation: 0.14,
      interpretability: 0.93,
      exampleTokens: ['5', '+', '3', '=', '8'],
      exampleActivations: [0.12, 0.95, 0.11, 0.89, 0.10]
    },
    {
      id: 8,
      name: 'Location References',
      activation: 0.21,
      interpretability: 0.86,
      exampleTokens: ['We', 'went', 'to', 'Paris', 'in', 'France'],
      exampleActivations: [0.05, 0.08, 0.09, 0.93, 0.12, 0.85]
    },
  ];

  const startFeatureExtraction = (trainingId: string) => {
    setExtractionStatus({
      ...extractionStatus,
      [trainingId]: { status: 'extracting', progress: 0 }
    });

    const interval = setInterval(() => {
      setExtractionStatus(prev => {
        const current = prev[trainingId];
        if (!current || current.progress >= 100) {
          clearInterval(interval);
          return {
            ...prev,
            [trainingId]: { status: 'completed', progress: 100 }
          };
        }
        return {
          ...prev,
          [trainingId]: { 
            status: 'extracting', 
            progress: Math.min(current.progress + Math.random() * 15, 100) 
          }
        };
      });
    }, 500);
  };

  const toggleFavorite = (featureId: number) => {
    setFavoritedFeatures(prev => {
      const newSet = new Set(prev);
      if (newSet.has(featureId)) {
        newSet.delete(featureId);
      } else {
        newSet.add(featureId);
      }
      return newSet;
    });
  };

  const filteredFeatures = mockFeatures
    .filter(f => f.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      const aVal = sortBy === 'activation_freq' ? a.activation :
                   sortBy === 'interpretability' ? a.interpretability : a.id;
      const bVal = sortBy === 'activation_freq' ? b.activation :
                   sortBy === 'interpretability' ? b.interpretability : b.id;
      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });

  // Auto-select first training if none selected
  useEffect(() => {
    if (completedTrainings.length > 0 && !selectedTraining) {
      setSelectedTraining(completedTrainings[0].id);
    }
  }, [completedTrainings, selectedTraining]);

  const currentTraining = completedTrainings.find((t: any) => t.id === selectedTraining);
  const status = currentTraining ? extractionStatus[currentTraining.id] : null;
  const isExtracted = status?.status === 'completed';
  const isExtracting = status?.status === 'extracting';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Feature Discovery</h2>
      </div>

      {completedTrainings.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          No completed trainings yet. Complete a training job to discover features.
        </div>
      ) : (
        <div className="space-y-6">
          {/* Training Selector */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <label htmlFor="analysis-training-selector" className="block text-sm font-medium text-slate-300 mb-3">
              Select Training Job to Analyze
            </label>
            <select
              id="analysis-training-selector"
              value={selectedTraining || ''}
              onChange={(e) => setSelectedTraining(e.target.value)}
              className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-lg"
            >
              {completedTrainings.map((training: any) => {
                const models = window.mockModels || [];
                const datasets = window.mockDatasets || [];
                const model = models.find((m: any) => m.id === training.model);
                const dataset = datasets.find((d: any) => d.id === training.dataset);

                return (
                  <option key={training.id} value={training.id}>
                    {training.encoderType} SAE • {model?.name || 'Model'} • {dataset?.name || 'Dataset'} • Started {new Date(training.startTime).toLocaleDateString()}
                  </option>
                );
              })}
            </select>

            {currentTraining && (
              <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
                <div className="bg-slate-800/30 rounded px-3 py-2">
                  <span className="text-slate-400">Model:</span>
                  <span className="ml-2 text-slate-200">{window.mockModels?.find((m: any) => m.id === currentTraining.model)?.name || 'Unknown'}</span>
                </div>
                <div className="bg-slate-800/30 rounded px-3 py-2">
                  <span className="text-slate-400">Dataset:</span>
                  <span className="ml-2 text-slate-200">{window.mockDatasets?.find((d: any) => d.id === currentTraining.dataset)?.name || 'Unknown'}</span>
                </div>
                <div className="bg-slate-800/30 rounded px-3 py-2">
                  <span className="text-slate-400">Encoder:</span>
                  <span className="ml-2 text-slate-200 capitalize">{currentTraining.encoderType}</span>
                </div>
              </div>
            )}
          </div>

          {/* Feature Extraction/Browser for Selected Training */}
          {currentTraining && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
              <h3 className="font-semibold text-lg">Feature Extraction & Analysis</h3>
                
                {/* Feature Extraction Interface */}
                {!isExtracted ? (
                  <div className="space-y-3">
                    <p className="text-sm text-slate-400">
                      Training complete. Extract interpretable features from the trained encoder.
                    </p>

                    {!isExtracting && (
                      <>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs text-slate-400 mb-1">Evaluation Samples</label>
                            <input
                              type="number"
                              defaultValue="10000"
                              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-xs text-slate-400 mb-1">Top-K Examples per Feature</label>
                            <input
                              type="number"
                              defaultValue="100"
                              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-sm"
                            />
                          </div>
                        </div>

                        <button
                          type="button"
                          onClick={() => startFeatureExtraction(currentTraining.id)}
                          className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded flex items-center justify-center gap-2"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                          Extract Features
                        </button>
                      </>
                    )}

                    {isExtracting && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-slate-400">Extracting features...</span>
                          <span className="text-emerald-400">{status?.progress?.toFixed(1) || 0}%</span>
                        </div>
                        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500 progress-bar-processing"
                            style={{ '--width': `${status?.progress || 0}%` } as React.CSSProperties}
                          />
                        </div>
                        <p className="text-xs text-slate-500">Processing activation patterns...</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Feature Statistics */}
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-slate-800/50 rounded-lg p-4">
                        <div className="text-sm text-slate-400 mb-1">Features Found</div>
                        <div className="text-2xl font-bold text-emerald-400">2,048</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-4">
                        <div className="text-sm text-slate-400 mb-1">Interpretable</div>
                        <div className="text-2xl font-bold text-blue-400">87.3%</div>
                      </div>
                      <div className="bg-slate-800/50 rounded-lg p-4">
                        <div className="text-sm text-slate-400 mb-1">Activation Rate</div>
                        <div className="text-2xl font-bold text-purple-400">12.4%</div>
                      </div>
                    </div>

                    {/* Feature Browser */}
                    <div className="border-t border-slate-700 pt-6 space-y-4">
                        {/* Search and Sort Controls */}
                        <div className="flex gap-3">
                          <input
                            type="text"
                            placeholder="Search features..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            aria-label="Search features"
                            className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                          />
                          
                          <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value)}
                            aria-label="Sort features by"
                            className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                          >
                            <option value="activation_freq">Activation Freq</option>
                            <option value="interpretability">Interpretability</option>
                            <option value="feature_id">Feature ID</option>
                          </select>
                          
                          <button
                            type="button"
                            onClick={() => setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc')}
                            className="px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg hover:bg-slate-800"
                            aria-label={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
                          >
                            <svg className={`w-5 h-5 transition-transform ${sortOrder === 'desc' ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                            </svg>
                          </button>
                        </div>

                        {/* Feature Table */}
                        <div className="overflow-x-auto">
                          <table className="w-full">
                            <thead className="bg-slate-800/50">
                              <tr>
                                <th className="px-4 py-3 text-left text-sm">ID</th>
                                <th className="px-4 py-3 text-left text-sm">Label</th>
                                <th className="px-4 py-3 text-left text-sm">Example Context</th>
                                <th className="px-4 py-3 text-right text-sm">Activation Freq</th>
                                <th className="px-4 py-3 text-right text-sm">Interpretability</th>
                                <th className="px-4 py-3 text-right text-sm">Actions</th>
                              </tr>
                            </thead>
                            <tbody>
                              {filteredFeatures.map(feature => {
                                const maxActivation = Math.max(...feature.exampleActivations);
                                return (
                                  <tr
                                    key={feature.id}
                                    onClick={() => setSelectedFeature(feature)}
                                    className="border-t border-slate-800 hover:bg-slate-800/30 cursor-pointer transition-colors"
                                  >
                                    <td className="px-4 py-3 font-mono text-sm text-slate-400">
                                      #{feature.id}
                                    </td>
                                    <td className="px-4 py-3 text-sm">
                                      {feature.name}
                                    </td>
                                    <td className="px-4 py-3">
                                      <div className="flex flex-wrap gap-1 font-mono text-xs max-w-md">
                                        {feature.exampleTokens.map((token, tokenIdx) => {
                                          const activation = feature.exampleActivations[tokenIdx];
                                          const intensity = Math.min(activation / maxActivation, 1.0);

                                          return (
                                            <span
                                              key={tokenIdx}
                                              className="px-1 py-0.5 rounded relative group token-highlight-attention"
                                              style={{
                                                '--bg-color': `rgba(16, 185, 129, ${intensity * 0.4})`,
                                                '--text-color': intensity > 0.6 ? '#fff' : '#cbd5e1',
                                                '--border': intensity > 0.7 ? '1px solid rgba(16, 185, 129, 0.5)' : 'none'
                                              } as React.CSSProperties}
                                              title={`Activation: ${activation.toFixed(3)}`}
                                            >
                                              {token}
                                            </span>
                                          );
                                        })}
                                      </div>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                      <span className="text-emerald-400 text-sm">
                                        {(feature.activation * 100).toFixed(2)}%
                                      </span>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                      <span className="text-blue-400 text-sm">
                                        {(feature.interpretability * 100).toFixed(1)}%
                                      </span>
                                    </td>
                                    <td className="px-4 py-3 text-right">
                                      <button
                                        type="button"
                                        onClick={(e) => { e.stopPropagation(); toggleFavorite(feature.id); }}
                                        className={favoritedFeatures.has(feature.id) ? "text-yellow-400" : "text-slate-500 hover:text-yellow-300"}
                                        aria-label={favoritedFeatures.has(feature.id) ? "Remove from favorites" : "Add to favorites"}
                                      >
                                        <Star className={`w-5 h-5 ${favoritedFeatures.has(feature.id) ? 'fill-yellow-400' : ''}`} />
                                      </button>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>

                        {filteredFeatures.length === 0 && (
                          <div className="text-center py-8 text-slate-500">
                            No features match your search
                          </div>
                        )}

                        {/* Pagination Placeholder */}
                        <div className="flex items-center justify-between pt-4 border-t border-slate-700">
                          <span className="text-sm text-slate-400">
                            Showing {filteredFeatures.length} of 2,048 features
                          </span>
                          <div className="flex gap-2">
                            <button type="button" className="px-3 py-1 bg-slate-800 rounded hover:bg-slate-700 text-sm">
                              Previous
                            </button>
                            <button type="button" className="px-3 py-1 bg-slate-800 rounded hover:bg-slate-700 text-sm">
                              Next
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
      )}

      {/* Feature Detail Modal */}
      {selectedFeature && (
        <FeatureDetailModal
          feature={selectedFeature}
          onClose={() => setSelectedFeature(null)}
        />
      )}
    </div>
  );
}

// Feature Detail Modal Component
function FeatureDetailModal({ feature, onClose }) {
  const [activeTab, setActiveTab] = useState('examples');
  const [featureLabel, setFeatureLabel] = useState(feature.name);

  // Mock data for feature details
  const mockExamples = [
    {
      tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat', 'yesterday'],
      activations: [0.01, 0.12, 0.08, 0.03, 0.02, 0.05, 0.11],
      max_activation: 0.12,
      source: 'dataset_sample_4521'
    },
    {
      tokens: ['A', 'dog', 'ran', 'through', 'the', 'park'],
      activations: [0.02, 0.15, 0.07, 0.04, 0.02, 0.06],
      max_activation: 0.15,
      source: 'dataset_sample_8832'
    },
    {
      tokens: ['She', 'walked', 'to', 'the', 'store'],
      activations: [0.03, 0.13, 0.04, 0.01, 0.05],
      max_activation: 0.13,
      source: 'dataset_sample_1203'
    }
  ];

  const mockLogitLens = {
    topTokens: ['the', 'a', 'an', 'this', 'that', 'my', 'your', 'his', 'her', 'its'],
    probabilities: [0.23, 0.18, 0.15, 0.12, 0.09, 0.07, 0.06, 0.04, 0.03, 0.03],
    interpretation: 'determiners and articles'
  };

  const mockCorrelations = [
    { id: 42, name: 'Noun Phrases', correlation: 0.87 },
    { id: 137, name: 'Subject Position', correlation: 0.76 },
    { id: 89, name: 'Sentence Start', correlation: 0.65 }
  ];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="max-w-6xl w-full bg-slate-900 rounded-lg shadow-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-800 p-6">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <div className="text-2xl font-bold mb-2">Feature #{feature.id}</div>
              <input
                type="text"
                value={featureLabel}
                onChange={(e) => setFeatureLabel(e.target.value)}
                placeholder="Add label..."
                aria-label="Feature label"
                className="w-full max-w-md px-3 py-1 bg-slate-800 border border-slate-700 rounded focus:outline-none focus:border-emerald-500"
              />
            </div>
            <button
              type="button"
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 p-2"
              aria-label="Close"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Feature Stats */}
          <div className="grid grid-cols-4 gap-4 mt-4">
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Activation Frequency</div>
              <div className="text-xl font-bold text-emerald-400">
                {(feature.activation * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Interpretability</div>
              <div className="text-xl font-bold text-blue-400">
                {(feature.interpretability * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Max Activation</div>
              <div className="text-xl font-bold text-purple-400">
                {(feature.activation * 2).toFixed(3)}
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Active Samples</div>
              <div className="text-xl font-bold text-yellow-400">
                {Math.floor(feature.activation * 10000)}
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-slate-800">
          <div className="flex px-6">
            {['examples', 'logit-lens', 'correlations', 'ablation'].map(tab => (
              <button
                type="button"
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 capitalize ${
                  activeTab === tab
                    ? 'border-emerald-400 text-emerald-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300'
                }`}
              >
                {tab.replace('-', ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {activeTab === 'examples' && (
            <MaxActivatingExamples examples={mockExamples} />
          )}

          {activeTab === 'logit-lens' && (
            <LogitLensView data={mockLogitLens} />
          )}

          {activeTab === 'correlations' && (
            <FeatureCorrelations correlations={mockCorrelations} />
          )}

          {activeTab === 'ablation' && (
            <AblationAnalysis
              perplexityDelta={2.3}
              impactScore={0.76}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Max Activating Examples Component
function MaxActivatingExamples({ examples }) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h4 className="font-semibold">Top Activating Contexts</h4>
        <div className="text-sm text-slate-400">
          Showing {examples.length} examples
        </div>
      </div>

      {examples.map((example: any, idx: number) => (
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
            {example.tokens.map((token: any, tokenIdx: number) => {
              const activation = example.activations[tokenIdx];
              const intensity = Math.min(activation / example.max_activation, 1.0);

              return (
                <span
                  key={tokenIdx}
                  className="px-1 py-0.5 rounded relative group token-highlight-gradient"
                  style={{
                    '--bg-color': `rgba(16, 185, 129, ${intensity * 0.3})`,
                    '--text-color': intensity > 0.5 ? '#fff' : '#e2e8f0'
                  } as React.CSSProperties}
                >
                  {token}

                  {/* Tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-10">
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
  );
}

// Logit Lens View Component
function LogitLensView({ data }) {
  return (
    <div className="space-y-4">
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Predicted Tokens</h4>
        <p className="text-sm text-slate-400">
          If this feature alone determined the next token, these would be most likely:
        </p>
      </div>

      <div className="space-y-2">
        {data.topTokens.map((token: string, idx: number) => (
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
                  {(data.probabilities[idx] * 100).toFixed(2)}%
                </span>
              </div>

              <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 progress-bar-probability"
                  style={{ '--width': `${data.probabilities[idx] * 100}%` } as React.CSSProperties}
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
            <strong>Interpretation:</strong> This feature appears to represent{' '}
            <span className="text-blue-400">{data.interpretation}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Feature Correlations Component
function FeatureCorrelations({ correlations }) {
  return (
    <div className="space-y-4">
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Correlated Features</h4>
        <p className="text-sm text-slate-400">
          Features that frequently activate together with this feature
        </p>
      </div>

      <div className="space-y-2">
        {correlations.map((corr: any) => (
          <div
            key={corr.id}
            className="flex items-center justify-between bg-slate-800/30 rounded-lg p-4 hover:bg-slate-800/50 cursor-pointer transition-colors"
          >
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-slate-400">#{corr.id}</span>
                <span className="font-medium">{corr.name}</span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-sm text-slate-400">Correlation</div>
                <div className="text-lg font-bold text-purple-400">
                  {corr.correlation.toFixed(2)}
                </div>
              </div>
              <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-purple-400 progress-bar-correlation"
                  style={{ '--width': `${corr.correlation * 100}%` } as React.CSSProperties}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Ablation Analysis Component
function AblationAnalysis({ perplexityDelta, impactScore }) {
  return (
    <div className="space-y-6">
      <div className="mb-4">
        <h4 className="font-semibold mb-2">Ablation Analysis</h4>
        <p className="text-sm text-slate-400">
          Impact of removing this feature on model performance
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-slate-800/30 rounded-lg p-6">
          <div className="text-sm text-slate-400 mb-2">Perplexity Change</div>
          <div className="text-4xl font-bold text-red-400 mb-2">
            +{perplexityDelta.toFixed(2)}
          </div>
          <p className="text-xs text-slate-500">
            Removing this feature increases perplexity, indicating it contributes to model performance
          </p>
        </div>

        <div className="bg-slate-800/30 rounded-lg p-6">
          <div className="text-sm text-slate-400 mb-2">Impact Score</div>
          <div className="text-4xl font-bold text-emerald-400 mb-2">
            {(impactScore * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-500">
            Relative importance of this feature compared to others
          </p>
        </div>
      </div>

      <div className="bg-blue-900/20 border border-blue-800/30 rounded-lg p-4">
        <div className="flex gap-2">
          <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />
          <div className="text-sm text-slate-300">
            <strong>Analysis:</strong> This feature has a significant impact on model performance.
            Ablating it results in measurable degradation, suggesting it encodes important information.
          </div>
        </div>
      </div>
    </div>
  );
}

// Dataset Detail Modal Component
function DatasetDetailModal({ dataset, onClose }) {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="max-w-6xl w-full bg-slate-900 rounded-lg shadow-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-800 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold">{dataset.name}</h2>
              <p className="text-slate-400 mt-1">
                {dataset.size} • 10,000 samples
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 p-2"
              aria-label="Close"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-slate-800">
          <div className="flex px-6">
            {['overview', 'samples', 'tokenization', 'statistics'].map(tab => (
              <button
                type="button"
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-3 border-b-2 capitalize ${
                  activeTab === tab
                    ? 'border-emerald-400 text-emerald-400'
                    : 'border-transparent text-slate-400 hover:text-slate-300'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {activeTab === 'overview' && (
            <DatasetOverview dataset={dataset} />
          )}

          {activeTab === 'samples' && (
            <DatasetSampleBrowser dataset={dataset} />
          )}

          {activeTab === 'tokenization' && (
            <TokenizationSettings dataset={dataset} />
          )}

          {activeTab === 'statistics' && (
            <DatasetStatistics dataset={dataset} />
          )}
        </div>
      </div>
    </div>
  );
}

// Dataset Overview Component
function DatasetOverview({ dataset }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">Dataset Information</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Source</div>
            <div className="text-lg font-medium">{dataset.source}</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Size</div>
            <div className="text-lg font-medium">{dataset.size}</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Format</div>
            <div className="text-lg font-medium">JSON Lines</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Status</div>
            <div className="text-lg font-medium capitalize">{dataset.status}</div>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-4">Description</h3>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <p className="text-slate-300">
            This dataset contains high-quality text samples suitable for training language models
            and interpretability research. The data has been preprocessed and filtered for quality.
          </p>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Total Samples</div>
            <div className="text-2xl font-bold text-emerald-400">10,000</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Avg Length</div>
            <div className="text-2xl font-bold text-blue-400">487</div>
            <div className="text-xs text-slate-500">tokens</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Vocabulary</div>
            <div className="text-2xl font-bold text-purple-400">50,257</div>
            <div className="text-xs text-slate-500">unique tokens</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Dataset Sample Browser Component
function DatasetSampleBrowser({ dataset: _dataset }) {
  const [sampleSearch, setSampleSearch] = useState('');
  const [filterSplit, setFilterSplit] = useState('all');
  const [expandedSamples, setExpandedSamples] = useState([]);
  const [page, setPage] = useState(0);
  const samplesPerPage = 10;

  const mockSamples = [
    { id: 1, text: 'The quick brown fox jumps over the lazy dog. This is a sample sentence from the dataset.', split: 'train', metadata: { source: 'web_crawl' } },
    { id: 2, text: 'Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.', split: 'train', metadata: { source: 'wikipedia' } },
    { id: 3, text: 'In the year 2024, technological advancements continue to reshape our understanding of what is possible.', split: 'validation', metadata: { source: 'news_articles' } },
    { id: 4, text: 'Python is a high-level programming language known for its readability and versatility in data science applications.', split: 'train', metadata: { source: 'technical_docs' } },
    { id: 5, text: 'The process of training neural networks involves iteratively adjusting weights to minimize loss functions.', split: 'test', metadata: { source: 'research_papers' } }
  ];

  const filteredSamples = mockSamples.filter(sample => {
    const matchesSearch = sample.text.toLowerCase().includes(sampleSearch.toLowerCase());
    const matchesSplit = filterSplit === 'all' || sample.split === filterSplit;
    return matchesSearch && matchesSplit;
  });

  const paginatedSamples = filteredSamples.slice(page * samplesPerPage, (page + 1) * samplesPerPage);

  const toggleExpand = (sampleId: number) => {
    setExpandedSamples(prev =>
      prev.includes(sampleId)
        ? prev.filter((id: number) => id !== sampleId)
        : [...prev, sampleId]
    );
  };

  return (
    <div className="space-y-4">
      {/* Filter and Search */}
      <div className="flex gap-3">
        <input
          type="text"
          placeholder="Search in samples..."
          value={sampleSearch}
          onChange={(e) => setSampleSearch(e.target.value)}
          aria-label="Search in dataset samples"
          className="flex-1 px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
        />

        <select
          value={filterSplit}
          onChange={(e) => setFilterSplit(e.target.value)}
          aria-label="Filter by dataset split"
          className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
        >
          <option value="all">All Splits</option>
          <option value="train">Train</option>
          <option value="validation">Validation</option>
          <option value="test">Test</option>
        </select>
      </div>

      {/* Sample List */}
      <div className="space-y-3">
        {paginatedSamples.map((sample) => (
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
                  type="button"
                  onClick={() => navigator.clipboard.writeText(sample.text)}
                  className="text-slate-400 hover:text-slate-200"
                  aria-label="Copy to clipboard"
                >
                  <Copy className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => toggleExpand(sample.id)}
                  className="text-slate-400 hover:text-slate-200"
                  aria-label={expandedSamples.includes(sample.id) ? "Collapse" : "Expand"}
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
          Showing {page * samplesPerPage + 1}-{Math.min((page + 1) * samplesPerPage, filteredSamples.length)} of {filteredSamples.length}
        </span>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setPage(page - 1)}
            disabled={page === 0}
            className="px-3 py-1 bg-slate-800 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <button
            type="button"
            onClick={() => setPage(page + 1)}
            disabled={(page + 1) * samplesPerPage >= filteredSamples.length}
            className="px-3 py-1 bg-slate-800 rounded hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

// Tokenization Settings Component
function TokenizationSettings({ dataset: _dataset }) {
  const [tokenizer, setTokenizer] = useState('auto');
  const [maxLength, setMaxLength] = useState(512);
  const [paddingStrategy, setPaddingStrategy] = useState('max_length');
  const [truncationStrategy, setTruncationStrategy] = useState('longest_first');
  const [addSpecialTokens, setAddSpecialTokens] = useState(true);
  const [returnAttentionMask, setReturnAttentionMask] = useState(true);
  const [previewText, setPreviewText] = useState('');
  const [previewTokens, setPreviewTokens] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const previewTokenization = () => {
    // Mock tokenization
    const words = previewText.split(/\s+/);
    const tokens = words.map((word, idx) => ({
      id: idx,
      text: word
    }));
    setPreviewTokens(tokens);
  };

  const applySettings = () => {
    setIsProcessing(true);
    setTimeout(() => {
      setIsProcessing(false);
      alert('Tokenization settings applied successfully!');
    }, 2000);
  };

  return (
    <div className="space-y-6">
      <div>
        <h3 className="font-semibold mb-4">Tokenization Configuration</h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="tokenization-tokenizer" className="block text-sm font-medium text-slate-300 mb-2">
              Tokenizer
            </label>
            <select
              id="tokenization-tokenizer"
              value={tokenizer}
              onChange={(e) => setTokenizer(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="auto">Auto (from model)</option>
              <option value="gpt2">GPT-2</option>
              <option value="llama">LLaMA</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          <div>
            <label htmlFor="tokenization-max-length" className="block text-sm font-medium text-slate-300 mb-2">
              Max Sequence Length
            </label>
            <input
              id="tokenization-max-length"
              type="number"
              value={maxLength}
              onChange={(e) => setMaxLength(parseInt(e.target.value))}
              min="1"
              max="4096"
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div>
            <label htmlFor="tokenization-padding-strategy" className="block text-sm font-medium text-slate-300 mb-2">
              Padding Strategy
            </label>
            <select
              id="tokenization-padding-strategy"
              value={paddingStrategy}
              onChange={(e) => setPaddingStrategy(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="max_length">Max Length (pad to max_length)</option>
              <option value="longest">Longest (pad to longest in batch)</option>
              <option value="do_not_pad">Do Not Pad</option>
            </select>
          </div>

          <div>
            <label htmlFor="tokenization-truncation-strategy" className="block text-sm font-medium text-slate-300 mb-2">
              Truncation Strategy
            </label>
            <select
              id="tokenization-truncation-strategy"
              value={truncationStrategy}
              onChange={(e) => setTruncationStrategy(e.target.value)}
              className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
            >
              <option value="longest_first">Longest First</option>
              <option value="only_first">Only First Sequence</option>
              <option value="only_second">Only Second Sequence</option>
              <option value="do_not_truncate">Do Not Truncate</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg">
            <label className="text-sm font-medium text-slate-300">
              Add Special Tokens
            </label>
            <button
              type="button"
              onClick={() => setAddSpecialTokens(!addSpecialTokens)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                addSpecialTokens ? 'bg-emerald-600' : 'bg-slate-600'
              }`}
              aria-label={`Toggle add special tokens ${addSpecialTokens ? 'off' : 'on'}`}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                addSpecialTokens ? 'translate-x-7' : 'translate-x-1'
              }`} />
            </button>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-lg">
            <label className="text-sm font-medium text-slate-300">
              Return Attention Mask
            </label>
            <button
              type="button"
              onClick={() => setReturnAttentionMask(!returnAttentionMask)}
              className={`relative w-12 h-6 rounded-full transition-colors ${
                returnAttentionMask ? 'bg-emerald-600' : 'bg-slate-600'
              }`}
              aria-label={`Toggle return attention mask ${returnAttentionMask ? 'off' : 'on'}`}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                returnAttentionMask ? 'translate-x-7' : 'translate-x-1'
              }`} />
            </button>
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
          aria-label="Text to preview tokenization"
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg mb-3 focus:outline-none focus:border-emerald-500"
        />

        <button
          type="button"
          onClick={previewTokenization}
          className="px-4 py-2 bg-slate-700 rounded hover:bg-slate-600 mb-3"
        >
          Tokenize Preview
        </button>

        {previewTokens && (
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="flex flex-wrap gap-1 mb-3">
              {previewTokens.map((token: any) => (
                <span
                  key={token.id}
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
          type="button"
          onClick={applySettings}
          disabled={isProcessing}
          className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2"
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
  );
}

// Dataset Statistics Component
function DatasetStatistics({ dataset: _dataset }) {
  const mockStats = {
    total_samples: 10000,
    total_tokens: 5234567,
    avg_tokens_per_sample: 523.4,
    unique_tokens: 50257,
    min_length: 10,
    median_length: 487,
    max_length: 2048
  };

  const mockLengthDistribution = [
    { range: '0-100', count: 150 },
    { range: '100-200', count: 450 },
    { range: '200-400', count: 2100 },
    { range: '400-600', count: 3800 },
    { range: '600-800', count: 2200 },
    { range: '800-1000', count: 900 },
    { range: '1000+', count: 400 }
  ];

  const maxCount = Math.max(...mockLengthDistribution.map(d => d.count));

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Total Samples</div>
          <div className="text-2xl font-bold text-emerald-400">
            {mockStats.total_samples.toLocaleString()}
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Total Tokens</div>
          <div className="text-2xl font-bold text-blue-400">
            {mockStats.total_tokens.toLocaleString()}
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Avg Tokens/Sample</div>
          <div className="text-2xl font-bold text-purple-400">
            {mockStats.avg_tokens_per_sample.toFixed(1)}
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Unique Tokens</div>
          <div className="text-2xl font-bold text-yellow-400">
            {mockStats.unique_tokens.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Sequence Length Distribution */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h4 className="font-semibold mb-4">Sequence Length Distribution</h4>
        <div className="space-y-2">
          {mockLengthDistribution.map((item, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="w-24 text-sm text-slate-400">{item.range}</div>
              <div className="flex-1 h-8 bg-slate-800 rounded overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 flex items-center justify-end pr-2 progress-bar-distribution"
                  style={{ '--width': `${(item.count / maxCount) * 100}%` } as React.CSSProperties}
                >
                  <span className="text-xs text-white font-medium">
                    {item.count.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 text-sm text-slate-400">
          Min: {mockStats.min_length} •
          Median: {mockStats.median_length} •
          Max: {mockStats.max_length} tokens
        </div>
      </div>

      {/* Split Distribution */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
        <h4 className="font-semibold mb-4">Split Distribution</h4>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Train</div>
            <div className="text-2xl font-bold text-emerald-400">8,000</div>
            <div className="text-xs text-slate-500 mt-1">80% of total</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Validation</div>
            <div className="text-2xl font-bold text-blue-400">1,500</div>
            <div className="text-xs text-slate-500 mt-1">15% of total</div>
          </div>
          <div className="bg-slate-800/30 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-1">Test</div>
            <div className="text-2xl font-bold text-purple-400">500</div>
            <div className="text-xs text-slate-500 mt-1">5% of total</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Steering Panel Component
function SteeringPanel({ models }) {
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [steeringCoefficients, setSteeringCoefficients] = useState({});
  const [prompt, setPrompt] = useState('');
  const [interventionLayer, setInterventionLayer] = useState(12);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(100);
  const [isGenerating, setIsGenerating] = useState(false);
  const [unsteeredOutput, setUnsteeredOutput] = useState('');
  const [steeredOutput, setSteeredOutput] = useState('');
  const [featureSearch, setFeatureSearch] = useState('');
  const [comparisonMetrics, setComparisonMetrics] = useState(null);

  const mockFeatures = [
    { id: 1, label: 'Sentiment Positive' },
    { id: 2, label: 'Code Syntax' },
    { id: 3, label: 'Question Pattern' },
    { id: 4, label: 'Temporal Reference' },
    { id: 5, label: 'Negation Logic' }
  ];

  const readyModels = models.filter((m: any) => m.status === 'ready');

  const addFeatureToSteering = (feature: any) => {
    if (!selectedFeatures.find(f => f.id === feature.id)) {
      setSelectedFeatures([...selectedFeatures, feature]);
      setSteeringCoefficients({ ...steeringCoefficients, [feature.id]: 0.0 });
    }
    setFeatureSearch('');
  };

  const removeFeatureFromSteering = (featureId: number) => {
    setSelectedFeatures(selectedFeatures.filter((f: any) => f.id !== featureId));
    const newCoeffs = { ...steeringCoefficients };
    delete newCoeffs[featureId];
    setSteeringCoefficients(newCoeffs);
  };

  const updateCoefficient = (featureId: number, value: number) => {
    setSteeringCoefficients({ ...steeringCoefficients, [featureId]: value });
  };

  const generateComparison = async () => {
    setIsGenerating(true);

    // Simulate generation
    setTimeout(() => {
      setUnsteeredOutput("The cat sat on the mat, looking peaceful and content in the afternoon sun.");
      setSteeredOutput("The cat sat on the mat joyfully, radiating happiness and pure delight in the warm afternoon sun.");
      setComparisonMetrics({
        kl_divergence: 0.0234,
        perplexity_delta: -2.3,
        semantic_similarity: 0.87,
        word_overlap: 0.65
      });
      setIsGenerating(false);
    }, 2000);
  };

  const filteredFeatures = mockFeatures.filter(f =>
    f.label.toLowerCase().includes(featureSearch.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Model Steering</h2>

      <div className="grid grid-cols-2 gap-6">
        {/* Feature Selection Panel */}
        <div className="space-y-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Select Features to Steer</h3>

            {/* Feature Search */}
            <div className="relative mb-4">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search features to add..."
                value={featureSearch}
                onChange={(e) => setFeatureSearch(e.target.value)}
                aria-label="Search features to add for steering"
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Search Results Dropdown */}
            {featureSearch && filteredFeatures.length > 0 && (
              <div className="mb-4 bg-slate-900 border border-slate-700 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                {filteredFeatures.map(feature => (
                  <button
                    type="button"
                    key={feature.id}
                    onClick={() => addFeatureToSteering(feature)}
                    className="w-full px-4 py-2 text-left hover:bg-slate-800 flex items-center justify-between transition-colors"
                  >
                    <span>
                      <span className="font-mono text-sm text-slate-400">#{feature.id}</span>
                      {' '}
                      {feature.label}
                    </span>
                    <span className="text-emerald-400 text-sm">Add</span>
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
                <div className="text-center py-8 text-slate-400 text-sm">
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
                        type="button"
                        onClick={() => removeFeatureFromSteering(feature.id)}
                        className="text-red-400 hover:text-red-300"
                        aria-label="Remove feature"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Coefficient Slider */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">Coefficient</span>
                        <span className="text-emerald-400 font-mono">
                          {(steeringCoefficients[feature.id] || 0).toFixed(2)}
                        </span>
                      </div>

                      <input
                        type="range"
                        min="-5"
                        max="5"
                        step="0.1"
                        value={steeringCoefficients[feature.id] || 0}
                        onChange={(e) => updateCoefficient(feature.id, parseFloat(e.target.value))}
                        aria-label={`Steering coefficient for feature ${feature.id}`}
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
                        type="button"
                        onClick={() => updateCoefficient(feature.id, -2.0)}
                        className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600"
                      >
                        Suppress
                      </button>
                      <button
                        type="button"
                        onClick={() => updateCoefficient(feature.id, 0.0)}
                        className="px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600"
                      >
                        Reset
                      </button>
                      <button
                        type="button"
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
          </div>
        </div>

        {/* Generation Controls Panel */}
        <div className="space-y-4">
          <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
            <h3 className="text-lg font-semibold">Generation Configuration</h3>

            {/* Model Selection */}
            <div>
              <label htmlFor="steering-model" className="block text-sm font-medium text-slate-300 mb-2">
                Model
              </label>
              <select
                id="steering-model"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
              >
                <option value="">Select model...</option>
                {readyModels.map((model: any) => (
                  <option key={model.id} value={model.id}>{model.name}</option>
                ))}
              </select>
            </div>

            {/* Prompt Input */}
            <div>
              <label htmlFor="steering-prompt" className="block text-sm font-medium text-slate-300 mb-2">
                Prompt
              </label>
              <textarea
                id="steering-prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt here..."
                rows={4}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg resize-none focus:outline-none focus:border-emerald-500"
              />
            </div>

            {/* Intervention Layer */}
            <div>
              <label htmlFor="steering-intervention-layer" className="block text-sm font-medium text-slate-300 mb-2">
                Intervention Layer: {interventionLayer}
              </label>
              <input
                id="steering-intervention-layer"
                type="range"
                min="0"
                max="24"
                value={interventionLayer}
                onChange={(e) => setInterventionLayer(parseInt(e.target.value))}
                className="w-full accent-emerald-500"
              />
            </div>

            {/* Generation Parameters */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="steering-temperature" className="block text-sm font-medium text-slate-300 mb-2">
                  Temperature
                </label>
                <input
                  id="steering-temperature"
                  type="number"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  min="0"
                  max="2"
                  step="0.1"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>

              <div>
                <label htmlFor="steering-max-tokens" className="block text-sm font-medium text-slate-300 mb-2">
                  Max Tokens
                </label>
                <input
                  id="steering-max-tokens"
                  type="number"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  min="1"
                  max="2048"
                  className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500"
                />
              </div>
            </div>

            {/* Generate Button */}
            <button
              type="button"
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
        </div>
      </div>

      {/* Comparative Output Display */}
      {(unsteeredOutput || steeredOutput) && (
        <div className="space-y-6">
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
                    type="button"
                    onClick={() => navigator.clipboard.writeText(unsteeredOutput)}
                    className="text-sm text-slate-400 hover:text-slate-200"
                    aria-label="Copy to clipboard"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                )}
              </div>

              <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-4 min-h-[200px]">
                {isGenerating ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader className="w-6 h-6 animate-spin text-slate-400" />
                  </div>
                ) : unsteeredOutput ? (
                  <div className="text-slate-300 whitespace-pre-wrap text-sm leading-relaxed">
                    {unsteeredOutput}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500">
                    No generation yet
                  </div>
                )}
              </div>
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
                    type="button"
                    onClick={() => navigator.clipboard.writeText(steeredOutput)}
                    className="text-sm text-slate-400 hover:text-slate-200"
                    aria-label="Copy to clipboard"
                  >
                    <Copy className="w-4 h-4" />
                  </button>
                )}
              </div>

              <div className="bg-slate-900/50 border border-emerald-800/30 rounded-lg p-4 min-h-[200px]">
                {isGenerating ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader className="w-6 h-6 animate-spin text-emerald-400" />
                  </div>
                ) : steeredOutput ? (
                  <div className="text-slate-300 whitespace-pre-wrap text-sm leading-relaxed">
                    {steeredOutput}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-slate-500">
                    No generation yet
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Comparison Metrics */}
          {comparisonMetrics && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4">Comparison Metrics</h3>

              <div className="grid grid-cols-4 gap-4">
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">KL Divergence</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {comparisonMetrics.kl_divergence.toFixed(4)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Distribution shift
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Perplexity Δ</div>
                  <div className={`text-2xl font-bold ${
                    comparisonMetrics.perplexity_delta > 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}>
                    {comparisonMetrics.perplexity_delta > 0 ? '+' : ''}
                    {comparisonMetrics.perplexity_delta.toFixed(2)}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {comparisonMetrics.perplexity_delta > 0 ? 'Higher' : 'Lower'} uncertainty
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Similarity</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {(comparisonMetrics.semantic_similarity * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Cosine similarity
                  </div>
                </div>

                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-400 mb-1">Word Overlap</div>
                  <div className="text-2xl font-bold text-emerald-400">
                    {(comparisonMetrics.word_overlap * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    Shared tokens
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
