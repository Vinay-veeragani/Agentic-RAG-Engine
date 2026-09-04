import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AppState {
  /** Developer Mode: reveals query/trace IDs, provider/model
   * names, iteration counts, token usage, latency, and raw JSON — never
   * hidden chain-of-thought, since none is ever produced or stored. */
  developerMode: boolean;
  toggleDeveloperMode: () => void;

  selectedCollectionId: string | null;
  setSelectedCollectionId: (id: string | null) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      developerMode: false,
      toggleDeveloperMode: () => set((s) => ({ developerMode: !s.developerMode })),

      selectedCollectionId: null,
      setSelectedCollectionId: (id) => set({ selectedCollectionId: id }),
    }),
    { name: "agentic-rag-ui" },
  ),
);
