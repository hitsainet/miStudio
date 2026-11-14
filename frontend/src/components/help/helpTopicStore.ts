// Which Help topic a link asked for; the Help screen scrolls to it and marks it current.
// Origin: miDataworks frontend/src/components/help/helpTopicStore.ts @ 32cf406 (ENH-004).
import { create } from 'zustand';

export const useHelpTopic = create<{ topic: string | null; open: (topic: string | null) => void }>(
  set => ({
    topic: null,
    open: topic => set({ topic }),
  })
);
