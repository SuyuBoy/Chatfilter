import type { LaplaceEvent } from '@laplace.live/event-types';
import { create } from 'zustand';

interface ChatState {
  messages: LaplaceEvent[];
  addMessage: (m: LaplaceEvent) => void;
  clear: () => void;
}

export const useChatStore = create<ChatState>(set => ({
  messages: [],
  addMessage: m => set(s => ({ messages: [...s.messages, m].slice(-100) })),
  clear: () => set({ messages: [] }),
}));
