import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SidebarState {
  collapsed: boolean;
}

interface UIState {
  sidebar: SidebarState;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
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
    }),
    {
      name: 'mistudio-ui-preferences',
      partialize: (state) => ({ sidebar: state.sidebar }),
    }
  )
);
