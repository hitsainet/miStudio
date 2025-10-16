/**
 * System Monitor API Client
 *
 * This module provides functions for fetching GPU and system metrics from the backend.
 */

import { fetchAPI, buildQueryString } from './client';
import type {
  GPUListResponse,
  GPUMetricsResponse,
  GPUInfoResponse,
  GPUProcessesResponse,
  SystemMetricsResponse,
  DiskUsageResponse,
  NetworkRatesResponse,
  DiskRatesResponse,
  AllMonitoringDataResponse,
} from '../types/system';

/**
 * Get list of available GPUs.
 *
 * @returns Promise resolving to GPU list with count and info for each GPU
 * @throws {APIError} If request fails or GPU monitoring unavailable (503)
 *
 * @example
 * ```typescript
 * const { gpu_count, gpus } = await getGPUList();
 * console.log(`Found ${gpu_count} GPU(s)`);
 * ```
 */
export async function getGPUList(): Promise<GPUListResponse> {
  return fetchAPI<GPUListResponse>('/system/gpu-list');
}

/**
 * Get current metrics for specified GPU.
 *
 * @param gpuId - GPU index (default: 0)
 * @returns Promise resolving to GPU metrics
 * @throws {APIError} If request fails or invalid GPU ID (400)
 *
 * @example
 * ```typescript
 * const { metrics } = await getGPUMetrics(0);
 * console.log(`GPU Temp: ${metrics.temperature}°C`);
 * console.log(`GPU Memory: ${metrics.memory.used_percent}%`);
 * ```
 */
export async function getGPUMetrics(gpuId: number = 0): Promise<GPUMetricsResponse> {
  const query = buildQueryString({ gpu_id: gpuId });
  return fetchAPI<GPUMetricsResponse>(`/system/gpu-metrics?${query}`);
}

/**
 * Get current metrics for all available GPUs.
 *
 * @returns Promise resolving to metrics for all GPUs
 * @throws {APIError} If request fails or GPU monitoring unavailable (503)
 *
 * @example
 * ```typescript
 * const { gpu_count, gpus } = await getAllGPUMetrics();
 * gpus.forEach(gpu => {
 *   console.log(`GPU ${gpu.gpu_id}: ${gpu.utilization.gpu}%`);
 * });
 * ```
 */
export async function getAllGPUMetrics(): Promise<{
  gpu_count: number;
  gpus: Array<ReturnType<GPUMetricsResponse['metrics']['toJSON']>>;
}> {
  return fetchAPI<{
    gpu_count: number;
    gpus: Array<ReturnType<GPUMetricsResponse['metrics']['toJSON']>>;
  }>('/system/gpu-metrics/all');
}

/**
 * Get static information about specified GPU.
 *
 * @param gpuId - GPU index (default: 0)
 * @returns Promise resolving to GPU information
 * @throws {APIError} If request fails or invalid GPU ID (400)
 *
 * @example
 * ```typescript
 * const { info } = await getGPUInfo(0);
 * console.log(`GPU: ${info.name}`);
 * console.log(`Driver: ${info.driver_version}`);
 * console.log(`CUDA: ${info.cuda_version}`);
 * ```
 */
export async function getGPUInfo(gpuId: number = 0): Promise<GPUInfoResponse> {
  const query = buildQueryString({ gpu_id: gpuId });
  return fetchAPI<GPUInfoResponse>(`/system/gpu-info?${query}`);
}

/**
 * Get list of processes using specified GPU.
 *
 * @param gpuId - GPU index (default: 0)
 * @returns Promise resolving to GPU processes
 * @throws {APIError} If request fails or invalid GPU ID (400)
 *
 * @example
 * ```typescript
 * const { processes } = await getGPUProcesses(0);
 * console.log(`Active processes: ${processes.length}`);
 * processes.forEach(proc => {
 *   console.log(`PID ${proc.pid}: ${proc.process_name} (${proc.gpu_memory_used_mb} MB)`);
 * });
 * ```
 */
export async function getGPUProcesses(gpuId: number = 0): Promise<GPUProcessesResponse> {
  const query = buildQueryString({ gpu_id: gpuId });
  return fetchAPI<GPUProcessesResponse>(`/system/gpu-processes?${query}`);
}

/**
 * Get current system resource metrics (CPU, RAM, swap, disk I/O, network I/O).
 *
 * @returns Promise resolving to system metrics
 * @throws {APIError} If request fails (500)
 *
 * @example
 * ```typescript
 * const { metrics } = await getSystemMetrics();
 * console.log(`CPU: ${metrics.cpu.percent}%`);
 * console.log(`RAM: ${metrics.ram.used_gb}/${metrics.ram.total_gb} GB`);
 * ```
 */
export async function getSystemMetrics(): Promise<SystemMetricsResponse> {
  return fetchAPI<SystemMetricsResponse>('/system/metrics');
}

/**
 * Get disk usage for specified mount points.
 *
 * @param mountPoints - Comma-separated list of mount points (default: "/,/data/")
 * @returns Promise resolving to disk usage information
 * @throws {APIError} If request fails (500)
 *
 * @example
 * ```typescript
 * const { mount_points } = await getDiskUsage('/,/data/');
 * mount_points.forEach(disk => {
 *   console.log(`${disk.mount_point}: ${disk.used_gb}/${disk.total_gb} GB (${disk.percent}%)`);
 * });
 * ```
 */
export async function getDiskUsage(mountPoints?: string): Promise<DiskUsageResponse> {
  const query = mountPoints ? buildQueryString({ mount_points: mountPoints }) : '';
  return fetchAPI<DiskUsageResponse>(`/system/disk-usage${query ? `?${query}` : ''}`);
}

/**
 * Get current network I/O rates (bytes per second).
 *
 * Note: Rates are calculated based on difference from last call.
 * First call may return 0 or inaccurate values.
 *
 * @returns Promise resolving to network rates
 * @throws {APIError} If request fails (500)
 *
 * @example
 * ```typescript
 * const { network_rates } = await getNetworkRates();
 * console.log(`Upload: ${network_rates.sent_mbps} Mbps`);
 * console.log(`Download: ${network_rates.recv_mbps} Mbps`);
 * ```
 */
export async function getNetworkRates(): Promise<NetworkRatesResponse> {
  return fetchAPI<NetworkRatesResponse>('/system/network-rates');
}

/**
 * Get current disk I/O rates (bytes per second).
 *
 * Note: Rates are calculated based on difference from last call.
 * First call may return 0 or inaccurate values.
 *
 * @returns Promise resolving to disk rates
 * @throws {APIError} If request fails (500)
 *
 * @example
 * ```typescript
 * const { disk_rates } = await getDiskRates();
 * console.log(`Read: ${disk_rates.read_mbps} MB/s`);
 * console.log(`Write: ${disk_rates.write_mbps} MB/s`);
 * ```
 */
export async function getDiskRates(): Promise<DiskRatesResponse> {
  return fetchAPI<DiskRatesResponse>('/system/disk-rates');
}

/**
 * Get all monitoring data in a single call (GPU metrics, system metrics, disk usage, I/O rates).
 *
 * This endpoint is optimized for dashboard views that need all metrics at once.
 * It reduces the number of HTTP requests from 7+ to 1.
 *
 * @param gpuId - GPU index for GPU-specific metrics (default: 0)
 * @returns Promise resolving to all monitoring data
 * @throws {APIError} If request fails (503 if GPU monitoring unavailable, 500 for other errors)
 *
 * @example
 * ```typescript
 * const data = await getAllMonitoringData(0);
 * console.log(`GPU Available: ${data.gpu_available}`);
 * console.log(`CPU: ${data.system.cpu.percent}%`);
 * if (data.gpu) {
 *   console.log(`GPU: ${data.gpu.metrics.utilization.gpu}%`);
 *   console.log(`GPU Temp: ${data.gpu.metrics.temperature}°C`);
 * }
 * ```
 */
export async function getAllMonitoringData(
  gpuId: number = 0
): Promise<AllMonitoringDataResponse> {
  const query = buildQueryString({ gpu_id: gpuId });
  return fetchAPI<AllMonitoringDataResponse>(`/system/all?${query}`);
}
