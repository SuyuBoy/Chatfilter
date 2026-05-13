import { create } from 'zustand';
import { PipelineEngine } from '../../app';

const engine = new PipelineEngine();

interface ClusterState {
  ingested: number;
  unique: number;
  clusters: { id: string; canonical: string; count: number; examples: string[] }[];
  permanent: { id: string; canonical: string; count: number }[];
  cacheHitRate: number;
  connected: boolean;
  modelReady: boolean;
  ingest: (text: string, user: string) => Promise<void>;
  setConnected: (c: boolean) => void;
  setModelReady: (r: boolean) => void;
}

export const useClusterStore = create<ClusterState>((set, get) => ({
  ingested: 0, unique: 0, clusters: [], permanent: [], cacheHitRate: 0,
  connected: false, modelReady: false,

  ingest: async (text: string, _user: string) => {
    const r = await engine.ingest(text);
    if (r.filtered) return;
    const s = engine.getState();
    set({
      ingested: s.ingested,
      unique: s.unique,
      clusters: s.clusters.map(c => ({
        id: c.clusterId, canonical: c.canonicalText,
        count: c.totalCount, examples: c.topExamples?.slice(0, 3) || [],
      })).sort((a, b) => b.count - a.count),
      permanent: s.permanent.map(p => ({
        id: p.clusterId, canonical: p.canonicalText, count: p.totalCount,
      })).sort((a, b) => b.count - a.count),
      cacheHitRate: s.cacheHitRate,
    });
  },

  setConnected: (c: boolean) => set({ connected: c }),
  setModelReady: (r: boolean) => set({ modelReady: r }),
}));
