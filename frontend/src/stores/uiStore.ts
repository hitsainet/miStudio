import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ActivePanel } from '../config/panels';

interface SidebarState {
  collapsed: boolean;
}

interface UIState {
  sidebar: SidebarState;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  /**
   * The screen on show. miStudio has no router: this is the one place the active panel lives, so
   * anything (the sidebar, a Help link, a "Steer this" button) can switch screens without a
   * callback threaded down to it (ENH-004, decision D3). App restores it on mount (from a
   * `#/help` link or localStorage) and persists it; it is deliberately NOT in `partialize`, so the
   * restore rules stay in one place.
   */
  activePanel: ActivePanel;
  setActivePanel: (panel: ActivePanel) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebar: {
        collapsed: false,
      },
      toggleSidebar: () =>
        set((state) => ({
          sidebar: { ...state.sidebar, collapsed: !state.sidebar.collapsed },
        })),
      setSidebarCollapsed: (collapsed: boolean) =>
        set((state) => ({
          sidebar: { ...state.sidebar, collapsed },
        })),
      activePanel: 'datasets',
      setActivePanel: (panel: ActivePanel) => set({ activePanel: panel }),
    }),
    {
      name: 'mistudio-ui-preferences',
      partialize: (state) => ({ sidebar: state.sidebar }),
    }
  )
);
