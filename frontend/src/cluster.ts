/**
 * 在线聚类引擎 — Leader-Follower + 周期维护
 * 翻译自 Python cluster_engine.py + micro_cluster.py
 */
import { cosineSim } from './embedder';

// ── MicroCluster ──

function lengthPenalty(newLen: number, anchorAvgLen: number): number {
  const ratio = Math.max(newLen, anchorAvgLen) / Math.max(Math.min(newLen, anchorAvgLen), 1);
  if (ratio <= 1.5) return 1.0;
  if (ratio <= 3.0) return 0.90;
  return 0.80;
}

export interface SlotState {
  slotId: number;
  clusterId: string;
  canonicalText: string;
  totalCount: number;
  topExamples: string[];
  latestRaw: string;
  lastUpdate: number;
  centroid: number[] | null;
  memberEmbeddings: number[][];
}

function createSlot(
  slotId: number, clusterId: string, canonical: string,
  emb: number[], raw: string,
): SlotState {
  return {
    slotId, clusterId, canonicalText: canonical,
    totalCount: 1, topExamples: [raw], latestRaw: raw,
    lastUpdate: Date.now(),
    centroid: [...emb],
    memberEmbeddings: [[...emb]],
  };
}

function internalSimilarity(slot: SlotState): number {
  if (slot.memberEmbeddings.length < 2 || !slot.centroid) return 1.0;
  const sims = slot.memberEmbeddings.map(e => cosineSim(e, slot.centroid!));
  return sims.reduce((a, b) => a + b, 0) / sims.length;
}

function canJoin(
  slot: SlotState, embedding: number[], textLen: number,
  centroidTh: number, anchorTh: number,
): boolean {
  if (!slot.centroid) return true;

  const simCentroid = cosineSim(embedding, slot.centroid);
  if (simCentroid < centroidTh) return false;

  // Anchor check: use centroid as anchor
  if (simCentroid < anchorTh) return false;

  // Length penalty
  const avgLen = slot.canonicalText.length || 1;
  if (textLen > 0) {
    const penalty = lengthPenalty(textLen, avgLen);
    if (simCentroid * penalty < centroidTh) return false;
  }

  return true;
}

// ── ClusterEngine ──

export interface ClusterConfig {
  centroidThreshold: number;
  anchorThreshold: number;
  maxSlots: number;
  mergeThreshold: number;
  splitVarianceThreshold: number;
  maintenanceInterval: number;
  permanentThreshold: number;
  permanentCentroidThreshold: number;
  permanentAnchorThreshold: number;
  permanentMergeThreshold: number;
  permanentSplitVarianceThreshold: number;
  permanentTtlSeconds: number;
  maxMemberEmbeddings: number;
}

export const DEFAULT_CONFIG: ClusterConfig = {
  centroidThreshold: 0.40,
  anchorThreshold: 0.60,
  maxSlots: 40,
  mergeThreshold: 0.92,
  splitVarianceThreshold: 0.50,
  maintenanceInterval: 300,
  permanentThreshold: 100,
  permanentCentroidThreshold: 0.75,
  permanentAnchorThreshold: 0.82,
  permanentMergeThreshold: 0.94,
  permanentSplitVarianceThreshold: 0.45,
  permanentTtlSeconds: 600,
  maxMemberEmbeddings: 50,
};

export class ClusterEngine {
  config: ClusterConfig;
  slots: Map<string, SlotState> = new Map();
  permanent: Map<string, SlotState> = new Map();
  private nextPermId = 1;
  totalIngested = 0;

  constructor(config: Partial<ClusterConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  findCluster(emb: number[], textLen: number): { slot: SlotState; isPerm: boolean } | null {
    let r = this.findInDict(this.permanent, emb, textLen,
      this.config.permanentCentroidThreshold,
      this.config.permanentAnchorThreshold,
      this.config.permanentSplitVarianceThreshold);
    if (r) return { slot: r, isPerm: true };

    r = this.findInDict(this.slots, emb, textLen,
      this.config.centroidThreshold, this.config.anchorThreshold,
      this.config.splitVarianceThreshold);
    if (r) return { slot: r, isPerm: false };

    return null;
  }

  private findInDict(
    dict: Map<string, SlotState>, emb: number[], textLen: number,
    centroidTh: number, anchorTh: number, splitVarTh: number,
  ): SlotState | null {
    let bestSim = -1;
    let bestSlot: SlotState | null = null;
    for (const slot of dict.values()) {
      if (!slot.centroid) continue;
      if (!canJoin(slot, emb, textLen, centroidTh, anchorTh)) continue;
      const sim = cosineSim(emb, slot.centroid);
      if (slot.memberEmbeddings.length >= 3 && internalSimilarity(slot) < splitVarTh) continue;
      if (sim > bestSim) {
        bestSim = sim;
        bestSlot = slot;
      }
    }
    return bestSlot;
  }

  join(slot: SlotState, emb: number[], raw: string, canonical: string): void {
    slot.totalCount++;
    slot.latestRaw = raw;
    slot.lastUpdate = Date.now();
    const old = Math.max(slot.totalCount - 1, 1);
    if (slot.centroid) {
      slot.centroid = slot.centroid.map(
        (c, i) => (c * old + emb[i]) / slot.totalCount,
      );
    }
    if (!slot.topExamples.includes(raw)) {
      slot.topExamples.push(raw);
    }
    slot.topExamples.sort((a, b) => b.length - a.length);
    slot.topExamples = [...new Set(slot.topExamples)].slice(0, 3);
    this.addMemberEmb(slot, emb);
  }

  newSlot(canonical: string, emb: number[], raw: string): { slotId: number; clusterId: string } {
    const sid = this.allocSlot();
    const cid = `c${sid}`;
    this.slots.set(canonical, createSlot(sid, cid, canonical, emb, raw));
    return { slotId: sid, clusterId: cid };
  }

  private allocSlot(): number {
    const maxN = this.config.maxSlots;
    if (this.slots.size < maxN) {
      const used = new Set([...this.slots.values()].map(s => s.slotId));
      for (let i = 1; i <= maxN; i++) {
        if (!used.has(i)) return i;
      }
      return this.slots.size + 1;
    }
    // Evict oldest
    let oldest: SlotState | null = null;
    for (const s of this.slots.values()) {
      if (!oldest || s.lastUpdate < oldest.lastUpdate) oldest = s;
    }
    if (oldest) {
      this.slots.delete(oldest.canonicalText);
      return oldest.slotId;
    }
    return 1;
  }

  promote(slot: SlotState): void {
    const canonical = slot.canonicalText;
    if (!this.slots.has(canonical)) return;
    this.slots.delete(canonical);
    slot.slotId = this.nextPermId;
    slot.clusterId = `p${this.nextPermId}`;
    this.nextPermId++;
    this.permanent.set(canonical, slot);
  }

  maintenance(): void {
    const conf = this.config;
    if (this.totalIngested % conf.maintenanceInterval !== 0) return;
    this.mergeSimilar(this.slots, conf.mergeThreshold);
    this.splitDegraded(this.slots, conf.splitVarianceThreshold, conf.maxSlots);
    this.mergeSimilar(this.permanent, conf.permanentMergeThreshold);
    this.splitDegraded(this.permanent, conf.permanentSplitVarianceThreshold, 9999);
    // TTL eviction
    const now = Date.now();
    for (const [c, s] of this.permanent) {
      if ((now - s.lastUpdate) / 1000 > conf.permanentTtlSeconds) {
        this.permanent.delete(c);
      }
    }
  }

  private mergeSimilar(dict: Map<string, SlotState>, threshold: number): void {
    const entries = [...dict.entries()];
    const merged = new Set<string>();
    for (let i = 0; i < entries.length; i++) {
      const [ci, si] = entries[i];
      if (merged.has(ci)) continue;
      for (let j = i + 1; j < entries.length; j++) {
        const [cj, sj] = entries[j];
        if (merged.has(cj)) continue;
        if (!si.centroid || !sj.centroid) continue;
        if (cosineSim(si.centroid, sj.centroid) > threshold) {
          if (si.totalCount >= sj.totalCount) {
            this.absorb(si, sj); merged.add(cj);
          } else {
            this.absorb(sj, si); merged.add(ci);
          }
        }
      }
    }
    for (const c of merged) dict.delete(c);
  }

  private absorb(target: SlotState, source: SlotState): void {
    const n = target.totalCount + source.totalCount;
    if (target.centroid && source.centroid) {
      target.centroid = target.centroid.map(
        (c, i) => (c * target.totalCount + source.centroid![i] * source.totalCount) / n,
      );
    }
    target.totalCount = n;
    target.lastUpdate = Math.max(target.lastUpdate, source.lastUpdate);
    for (const ex of source.topExamples) {
      if (!target.topExamples.includes(ex)) target.topExamples.push(ex);
    }
    target.topExamples.sort((a, b) => b.length - a.length);
    target.topExamples = [...new Set(target.topExamples)].slice(0, 3);
    for (const emb of source.memberEmbeddings) {
      this.addMemberEmb(target, emb);
    }
  }

  private splitDegraded(
    dict: Map<string, SlotState>, threshold: number, maxSlots: number,
  ): void {
    const toSplit: [string, SlotState][] = [];
    for (const [canonical, slot] of dict) {
      if (slot.memberEmbeddings.length < 4) continue;
      if (internalSimilarity(slot) < threshold) {
        toSplit.push([canonical, slot]);
      }
    }
    for (const [canonical, slot] of toSplit) {
      if (!dict.has(canonical)) continue;
      const embs = slot.memberEmbeddings;
      if (embs.length < 4) continue;
      const labels = kmeans2(embs);
      const g0 = embs.filter((_, k) => labels[k] === 0);
      const g1 = embs.filter((_, k) => labels[k] === 1);
      if (g0.length < 2 || g1.length < 2) continue;

      let newId: number;
      if (maxSlots < 9999) {
        const used = new Set([...dict.values()].map(s => s.slotId));
        newId = 0;
        for (let i = 1; i <= maxSlots; i++) {
          if (!used.has(i)) { newId = i; break; }
        }
        if (!newId) continue;
      } else {
        newId = Math.max(...[...dict.values()].map(s => s.slotId)) + 1;
      }

      // Recompute centroid for g0
      const c0 = meanVector(g0);
      slot.centroid = normalize(c0);
      slot.totalCount = g0.length;
      slot.memberEmbeddings = g0;

      // Create new slot for g1
      const c1 = normalize(meanVector(g1));
      const prefix = maxSlots === 9999 ? 'p' : 'c';
      const newCanonical = `${slot.canonicalText}*`;
      dict.set(newCanonical, {
        slotId: newId,
        clusterId: `${prefix}${newId}`,
        canonicalText: slot.canonicalText,
        totalCount: g1.length,
        topExamples: slot.topExamples.slice(0, 1),
        latestRaw: slot.latestRaw,
        lastUpdate: slot.lastUpdate,
        centroid: c1,
        memberEmbeddings: g1,
      });
    }
  }

  private addMemberEmb(slot: SlotState, emb: number[]): void {
    slot.memberEmbeddings.push([...emb]);
    const maxKeep = this.config.maxMemberEmbeddings;
    if (slot.memberEmbeddings.length > maxKeep) {
      const half = Math.floor(maxKeep / 2);
      slot.memberEmbeddings = [
        ...slot.memberEmbeddings.slice(0, half),
        ...slot.memberEmbeddings.slice(-half),
      ];
    }
  }

  getClusters(): SlotState[] {
    return [...this.slots.values()].sort((a, b) => b.totalCount - a.totalCount);
  }

  getPermanent(): SlotState[] {
    return [...this.permanent.values()].sort((a, b) => b.totalCount - a.totalCount);
  }
}

// ── Helpers ──

function meanVector(vectors: number[][]): number[] {
  const dim = vectors[0].length;
  const mean = new Array(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) mean[i] += v[i];
  }
  for (let i = 0; i < dim; i++) mean[i] /= vectors.length;
  return mean;
}

function normalize(vec: number[]): number[] {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) + 1e-8;
  return vec.map(v => v / norm);
}

function kmeans2(vectors: number[][]): number[] {
  const n = vectors.length;
  if (n < 2) return new Array(n).fill(0);
  let c0 = [...vectors[0]];
  let c1 = [...vectors[Math.min(1, n - 1)]];
  const labels = new Array(n).fill(0);
  for (let iter = 0; iter < 10; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      const nl = cosineSim(vectors[i], c0) >= cosineSim(vectors[i], c1) ? 0 : 1;
      if (nl !== labels[i]) { changed = true; labels[i] = nl; }
    }
    if (!changed) break;
    const g0 = vectors.filter((_, i) => labels[i] === 0);
    const g1 = vectors.filter((_, i) => labels[i] === 1);
    if (g0.length) c0 = meanVector(g0);
    if (g1.length) c1 = meanVector(g1);
  }
  return labels;
}
