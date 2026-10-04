/** Zustand store for DB-backed application settings. */

import { create } from 'zustand';
import type { AppSetting, AppSettingUpsert, SettingCategory } from '../types/appSetting';
import * as settingsApi from '../api/settings';

interface SettingsState {
  settings: AppSetting[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchAll: () => Promise<void>;
  fetchByCategory: (category: SettingCategory) => Promise<void>;
  upsert: (data: AppSettingUpsert) => Promise<void>;
  remove: (key: string) => Promise<void>;

  // Selectors
  getByKey: (key: string) => AppSetting | undefined;
  getByCategory: (category: SettingCategory) => AppSetting[];
  getEndpoints: () => AppSetting[];
}

export const useSettingsStore = create<SettingsState>()((set, get) => ({
  settings: [],
  isLoading: false,
  error: null,

  fetchAll: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await settingsApi.listSettings();
      set({ settings: data, isLoading: false });
    } catch (err) {
      set({ error: (err as Error).message, isLoading: false });
    }
  },

  fetchByCategory: async (category: SettingCategory) => {
    set({ isLoading: true, error: null });
    try {
      const data = await settingsApi.listSettings(category);
      // Merge with existing settings, replacing those in this category
      set((state) => {
        const otherSettings = state.settings.filter((s) => s.category !== category);
        return { settings: [...otherSettings, ...data], isLoading: false };
      });
    } catch (err) {
      set({ error: (err as Error).message, isLoading: false });
    }
  },

  upsert: async (data: AppSettingUpsert) => {
    set({ error: null });
    try {
      const saved = await settingsApi.upsertSetting(data);
      set((state) => {
        const idx = state.settings.findIndex((s) => s.key === saved.key);
        const next = [...state.settings];
        if (idx >= 0) {
          next[idx] = saved;
        } else {
          next.push(saved);
        }
        return { settings: next };
      });
    } catch (err) {
      set({ error: (err as Error).message });
      throw err;
    }
  },

  remove: async (key: string) => {
    set({ error: null });
    try {
      await settingsApi.deleteSetting(key);
      set((state) => ({
        settings: state.settings.filter((s) => s.key !== key),
      }));
    } catch (err) {
      set({ error: (err as Error).message });
      throw err;
    }
  },

  getByKey: (key: string) => get().settings.find((s) => s.key === key),
  getByCategory: (category: SettingCategory) => get().settings.filter((s) => s.category === category),
  getEndpoints: () => get().settings.filter((s) => s.category === 'endpoints'),
}));
