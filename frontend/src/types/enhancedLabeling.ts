/** Types for the enhanced per-feature two-pass labeling system. */

export type EnhancedLabelingStatus = 'queued' | 'running' | 'completed' | 'failed';
export type EnhancedLabelingPhase = 'pass1' | 'pass2' | null;

export interface EnhancedLabelingJob {
  id: string;
  feature_id: string;
  status: EnhancedLabelingStatus;
  phase: EnhancedLabelingPhase;
  examples_total: number;
  examples_completed: number;
  workers: number;
  // MIS-E2E-111: the server no longer returns `endpoint`. It was the
  // internal LLM server URL, published to the browser by a schema that
  // listed every column while its sibling deliberately withheld it.
  model: string;
  celery_task_id: string | null;
  pass1_summaries: Array<{
    n: number;
    prime: string;
    activation: number;
    summary: string;
  }> | null;
  raw_synthesis: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface EnhancedLabelingProgressEvent {
  job_id: string;
  phase: 'pass1' | 'pass2';
  examples_completed: number;
  examples_total: number;
}

export interface EnhancedLabelingCompletedEvent {
  job_id: string;
  name: string;
  category: string;
  description: string;
  notes: string;
}

export interface EnhancedLabelingFailedEvent {
  job_id: string;
  error_message: string;
}
