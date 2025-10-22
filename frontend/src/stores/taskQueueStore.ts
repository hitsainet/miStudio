/**
 * Task Queue Store
 *
 * Zustand store for managing task queue state.
 * Provides visibility and control over background operations.
 */

import { create } from 'zustand';
import { TaskQueueEntry, TaskQueueStatus, RetryRequest } from '../types/taskQueue';
import * as taskQueueApi from '../api/taskQueue';

interface TaskQueueState {
  // State
  tasks: TaskQueueEntry[];
  failedTasks: TaskQueueEntry[];
  activeTasks: TaskQueueEntry[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchTasks: (status?: string, entityType?: string) => Promise<void>;
  fetchFailedTasks: () => Promise<void>;
  fetchActiveTasks: () => Promise<void>;
  retryTask: (taskQueueId: string, request?: RetryRequest) => Promise<void>;
  deleteTask: (taskQueueId: string) => Promise<void>;
  clearError: () => void;
}

export const useTaskQueueStore = create<TaskQueueState>((set, get) => ({
  // Initial state
  tasks: [],
  failedTasks: [],
  activeTasks: [],
  loading: false,
  error: null,

  // Fetch all tasks with optional filtering
  fetchTasks: async (status?: string, entityType?: string) => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.getTaskQueue(status, entityType);
      set({ tasks: response.data, loading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch tasks:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch tasks',
        loading: false,
      });
    }
  },

  // Fetch failed tasks
  fetchFailedTasks: async () => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.getFailedTasks();
      set({ failedTasks: response.data, loading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch failed tasks:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch failed tasks',
        loading: false,
      });
    }
  },

  // Fetch active tasks
  fetchActiveTasks: async () => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.getActiveTasks();
      set({ activeTasks: response.data, loading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch active tasks:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch active tasks',
        loading: false,
      });
    }
  },

  // Retry a failed task
  retryTask: async (taskQueueId: string, request?: RetryRequest) => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.retryTask(taskQueueId, request);

      console.log('[TaskQueueStore] Task retry initiated:', response);

      // Remove from failed tasks list
      set((state) => ({
        failedTasks: state.failedTasks.filter((task) => task.id !== taskQueueId),
        loading: false,
      }));

      // Refresh active tasks to show the retry
      await get().fetchActiveTasks();
    } catch (error) {
      console.error('[TaskQueueStore] Failed to retry task:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to retry task',
        loading: false,
      });
      throw error;
    }
  },

  // Delete a task
  deleteTask: async (taskQueueId: string) => {
    try {
      set({ loading: true, error: null });
      await taskQueueApi.deleteTask(taskQueueId);

      // Remove from all task lists
      set((state) => ({
        tasks: state.tasks.filter((task) => task.id !== taskQueueId),
        failedTasks: state.failedTasks.filter((task) => task.id !== taskQueueId),
        activeTasks: state.activeTasks.filter((task) => task.id !== taskQueueId),
        loading: false,
      }));

      console.log('[TaskQueueStore] Task deleted:', taskQueueId);
    } catch (error) {
      console.error('[TaskQueueStore] Failed to delete task:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to delete task',
        loading: false,
      });
      throw error;
    }
  },

  // Clear error
  clearError: () => {
    set({ error: null });
  },
}));
