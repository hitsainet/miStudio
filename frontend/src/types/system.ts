/**
 * Type definitions for System Monitor
 *
 * These types match the backend API response structures.
 */

export interface GPUUtilization {
  gpu: number;
  memory: number;
}

export interface GPUMemory {
  used: number;
  total: number;
  free: number;
  used_gb: number;
  total_gb: number;
  used_percent: number;
}

export interface GPUPower {
  usage: number;
  limit: number;
  usage_percent: number;
}

export interface GPUClocks {
  gpu: number;
  memory: number;
}

export interface GPUMetrics {
  gpu_id: number;
  utilization: GPUUtilization;
  memory: GPUMemory;
  temperature: number;
  power: GPUPower;
  fan_speed: number;
  clocks: GPUClocks;
}

export interface GPUInfo {
  gpu_id: number;
  name: string;
  uuid: string;
  pci_bus_id: string;
  driver_version: string;
  cuda_version: string;
  compute_capability: string;
  total_memory_gb: number;
}

export interface GPUProcess {
  pid: number;
  process_name: string;
  gpu_memory_used: number;
  gpu_memory_used_mb: number;
  gpu_id: number;
}

export interface CPUMetrics {
  percent: number;
  count: number;
}

export interface RAMMetrics {
  used: number;
  total: number;
  available: number;
  used_gb: number;
  total_gb: number;
  used_percent: number;
}

export interface SwapMetrics {
  used: number;
  total: number;
  used_gb: number;
  total_gb: number;
  used_percent: number;
}

export interface DiskIO {
  read_bytes: number;
  write_bytes: number;
  read_mb: number;
  write_mb: number;
}

export interface NetworkIO {
  sent_bytes: number;
  recv_bytes: number;
  sent_mb: number;
  recv_mb: number;
}

export interface SystemMetrics {
  cpu: CPUMetrics;
  ram: RAMMetrics;
  swap: SwapMetrics;
  disk_io: DiskIO;
  network_io: NetworkIO;
}

export interface DiskUsage {
  mount_point: string;
  total: number;
  used: number;
  free: number;
  total_gb: number;
  used_gb: number;
  free_gb: number;
  percent: number;
}

export interface NetworkRates {
  sent_rate: number;
  recv_rate: number;
  sent_mbps: number;
  recv_mbps: number;
}

export interface DiskRates {
  read_rate: number;
  write_rate: number;
  read_mbps: number;
  write_mbps: number;
}

// API Response types
export interface GPUListResponse {
  gpu_count: number;
  gpus: GPUInfo[];
}

export interface GPUMetricsResponse {
  gpu_id: number;
  metrics: GPUMetrics;
}

export interface GPUInfoResponse {
  gpu_id: number;
  info: GPUInfo;
}

export interface GPUProcessesResponse {
  gpu_id: number;
  process_count: number;
  processes: GPUProcess[];
}

export interface SystemMetricsResponse {
  metrics: SystemMetrics;
}

export interface DiskUsageResponse {
  mount_points: DiskUsage[];
}

export interface NetworkRatesResponse {
  network_rates: NetworkRates;
}

export interface DiskRatesResponse {
  disk_rates: DiskRates;
}

export interface AllMonitoringDataResponse {
  gpu_available: boolean;
  system: SystemMetrics;
  disk_usage: DiskUsage[];
  network_rates: NetworkRates;
  disk_rates: DiskRates;
  gpu?: {
    gpu_count: number;
    selected_gpu_id: number;
    metrics: GPUMetrics;
    info: GPUInfo;
    processes: GPUProcess[];
  };
}

// View mode for System Monitor
export type ViewMode = 'single' | 'compare';

// Resource Estimation Types
export interface ResourceEstimateRequest {
  training_id: string;
  evaluation_samples: number;
  top_k_examples: number;
  batch_size?: number;
  num_workers?: number;
  db_commit_batch?: number;
}

export interface SystemResources {
  cpu_cores: number;
  total_ram_gb: number;
  available_ram_gb: number;
  ram_percent_used: number;
  gpu_available: boolean;
  gpu_name?: string;
  gpu_total_memory_gb?: number;
  gpu_memory_allocated_gb?: number;
  gpu_memory_reserved_gb?: number;
  gpu_memory_available_gb?: number;
}

export interface ResourceSettings {
  batch_size: number;
  num_workers: number;
  db_commit_batch: number;
}

export interface ResourceEstimates {
  estimated_ram_gb: number;
  estimated_gpu_gb: number | null;
  estimated_duration_minutes: number;
  warnings: string[];
  errors: string[];
}

export interface ResourceEstimateResponse {
  system_resources: SystemResources;
  recommended_settings: ResourceSettings;
  current_settings: ResourceSettings;
  resource_estimates: ResourceEstimates;
}
