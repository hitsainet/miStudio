/**
 * Cluster profiles store (Feature 014, IDL-30).
 *
 * Owns the durable profile list, save/import/export flows, and the
 * SaveProfileDialog open-state (so the Clusters panel can trigger the dialog
 * across panels after a hand-off). Loading a profile INTO steering lives in
 * steeringStore.loadProfileIntoSteering — this store never touches selection.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { clusterProfilesApi } from '../api/clusterProfiles';
import {
  ClusterProfile,
  ClusterProfileCreate,
  ImportResponse,
} from '../types/clusterProfile';

function downloadJson(data: Record<string, unknown>, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function slug(name: string): string {
  return (
    name
      .toLowerCase()
      .replace(/[^a-z0-9-_]+/g, '-')
      .slice(0, 60) || 'cluster'
  );
}

interface ClusterProfilesState {
  profiles: ClusterProfile[];
  total: number;
  isLoading: boolean;
  error: string | null;
  lastImport: ImportResponse | null;

  // SaveProfileDialog orchestration (openable from Steering AND Clusters panels)
  saveDialogOpen: boolean;

  fetchProfiles: (saeId?: string | null, search?: string) => Promise<void>;
  saveProfile: (data: ClusterProfileCreate) => Promise<ClusterProfile | null>;
  deleteProfile: (id: string) => Promise<boolean>;
  importPayload: (
    payload: Record<string, unknown>,
    bindSaeId?: string | null,
  ) => Promise<ImportResponse | null>;
  exportProfile: (profile: ClusterProfile) => Promise<void>;
  exportBundle: (ids: string[]) => Promise<void>;
  setSaveDialogOpen: (open: boolean) => void;
  clearError: () => void;
}

export const useClusterProfilesStore = create<ClusterProfilesState>()(
  devtools(
    (set, get) => ({
      profiles: [],
      total: 0,
      isLoading: false,
      error: null,
      lastImport: null,
      saveDialogOpen: false,

      fetchProfiles: async (saeId?: string | null, search?: string) => {
        set({ isLoading: true, error: null });
        try {
          const resp = await clusterProfilesApi.list({
            ...(saeId ? { sae_id: saeId } : {}),
            ...(search ? { search } : {}),
          });
          set({ profiles: resp.data, total: resp.total, isLoading: false });
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Failed to load profiles', isLoading: false });
        }
      },

      saveProfile: async (data: ClusterProfileCreate) => {
        set({ error: null });
        try {
          const profile = await clusterProfilesApi.create(data);
          set({ profiles: [profile, ...get().profiles], total: get().total + 1 });
          return profile;
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Failed to save profile' });
          return null;
        }
      },

      deleteProfile: async (id: string) => {
        try {
          await clusterProfilesApi.delete(id);
          set({
            profiles: get().profiles.filter((p) => p.id !== id),
            total: Math.max(0, get().total - 1),
          });
          return true;
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Failed to delete profile' });
          return false;
        }
      },

      importPayload: async (payload: Record<string, unknown>, bindSaeId?: string | null) => {
        set({ error: null });
        try {
          const resp = await clusterProfilesApi.import(payload, bindSaeId);
          set({ lastImport: resp });
          // Refresh the list — imports may have landed bound or unbound.
          await get().fetchProfiles();
          return resp;
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Import failed' });
          return null;
        }
      },

      exportProfile: async (profile: ClusterProfile) => {
        try {
          const definition = await clusterProfilesApi.exportDefinition(profile.id);
          downloadJson(definition, `${slug(profile.name)}.cluster.json`);
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Export failed' });
        }
      },

      exportBundle: async (ids: string[]) => {
        try {
          const bundle = await clusterProfilesApi.exportBundle(ids);
          downloadJson(bundle, 'clusters.bundle.json');
        } catch (e) {
          set({ error: e instanceof Error ? e.message : 'Bundle export failed' });
        }
      },

      setSaveDialogOpen: (open: boolean) => set({ saveDialogOpen: open }),
      clearError: () => set({ error: null }),
    }),
    { name: 'ClusterProfilesStore' },
  ),
);
