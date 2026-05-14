/**
 * PipelineEngine — 编排层
 * 预处理 → embedding → 聚类 → 查询
 */
import { preprocess, DedupStore } from './preprocess';
import { encode, isModelReady } from './embedder';
import { ClusterEngine, type ClusterConfig, type SlotState } from './cluster';

export type { ClusterConfig, SlotState };

export interface IngestResult {
  clusterId: string;
  canonical: string;
  rawText: string;
  slotId: number;
  isNew: boolean;
  filtered: boolean;
  permanent: boolean;
}

export interface PipelineState {
  ingested: number;
  unique: number;
  centroid: number;
  anchor: number;
  maxSlots: number;
  clusters: SlotState[];
  permanent: SlotState[];
  cacheHitRate: number;
  modelReady: boolean;
}

export class PipelineEngine {
  cluster: ClusterEngine;
  dedup = new DedupStore();
  cacheHits = 0;
  cacheTotal = 0;
  totalIngested = 0;

  constructor(config: Partial<ClusterConfig> = {}) {
    this.cluster = new ClusterEngine(config);
  }

  async ingest(rawText: string): Promise<IngestResult> {
    this.totalIngested++;
    this.cluster.totalIngested = this.totalIngested;

    const pp = preprocess(rawText);
    if (!pp) {
      return {
        clusterId: '', canonical: '', rawText,
        slotId: 0, isNew: false, filtered: true, permanent: false,
      };
    }

    const canonical = pp.normalized;
    this.cacheTotal++;
    const isNew = this.dedup.add(canonical, rawText);
    if (!isNew) this.cacheHits++;

    if (!isModelReady()) {
      return {
        clusterId: '', canonical, rawText,
        slotId: 0, isNew: false, filtered: false, permanent: false,
      };
    }

    const [emb] = await encode([canonical]);
    const found = this.cluster.findCluster(emb, rawText.length);

    if (found) {
      this.cluster.join(found.slot, emb, rawText, canonical);
      if (!found.isPerm && found.slot.totalCount >= this.cluster.config.permanentThreshold) {
        this.cluster.promote(found.slot);
      }
      this.cluster.maintenance();
      return {
        clusterId: found.slot.clusterId, canonical, rawText,
        slotId: found.slot.slotId, isNew: false, filtered: false,
        permanent: found.isPerm,
      };
    }

    const { slotId, clusterId } = this.cluster.newSlot(canonical, emb, rawText);
    this.cluster.maintenance();
    return {
      clusterId, canonical, rawText,
      slotId, isNew: true, filtered: false, permanent: false,
    };
  }

  async ingestBatch(texts: string[]): Promise<IngestResult[]> {
    const results: IngestResult[] = [];
    const needEmb: { idx: number; canonical: string }[] = [];
    const prelims: { idx: number; canonical: string; rawText: string }[] = [];

    for (let i = 0; i < texts.length; i++) {
      this.totalIngested++;
      const pp = preprocess(texts[i]);
      if (!pp) {
        results[i] = {
          clusterId: '', canonical: '', rawText: texts[i],
          slotId: 0, isNew: false, filtered: true, permanent: false,
        };
        continue;
      }
      const canonical = pp.normalized;
      this.cacheTotal++;
      const isNew = this.dedup.add(canonical, texts[i]);
      if (!isNew) this.cacheHits++;
      prelims.push({ idx: i, canonical, rawText: texts[i] });
      needEmb.push({ idx: i, canonical });
    }

    this.cluster.totalIngested = this.totalIngested;

    if (needEmb.length > 0 && isModelReady()) {
      const canonicals = needEmb.map(n => n.canonical);
      const embs = await encode(canonicals);

      const embMap = new Map<string, number[]>();
      for (let i = 0; i < needEmb.length; i++) {
        embMap.set(needEmb[i].canonical, embs[i]);
      }

      for (const { idx, canonical, rawText } of prelims) {
        const emb = embMap.get(canonical);
        if (!emb) {
          results[idx] = {
            clusterId: '', canonical, rawText, slotId: 0,
            isNew: false, filtered: false, permanent: false,
          };
          continue;
        }

        const found = this.cluster.findCluster(emb, rawText.length);
        if (found) {
          this.cluster.join(found.slot, emb, rawText, canonical);
          if (!found.isPerm && found.slot.totalCount >= this.cluster.config.permanentThreshold) {
            this.cluster.promote(found.slot);
          }
          results[idx] = {
            clusterId: found.slot.clusterId, canonical, rawText,
            slotId: found.slot.slotId, isNew: false, filtered: false,
            permanent: found.isPerm,
          };
        } else {
          const { slotId, clusterId } = this.cluster.newSlot(canonical, emb, rawText);
          results[idx] = {
            clusterId, canonical, rawText,
            slotId, isNew: true, filtered: false, permanent: false,
          };
        }
      }
      this.cluster.maintenance();
    }

    return results;
  }

  getState(): PipelineState {
    return {
      ingested: this.totalIngested,
      unique: this.dedup.size,
      centroid: this.cluster.config.centroidThreshold,
      anchor: this.cluster.config.anchorThreshold,
      maxSlots: this.cluster.config.maxSlots,
      clusters: this.cluster.getClusters(),
      permanent: this.cluster.getPermanent(),
      cacheHitRate: this.cacheTotal > 0 ? this.cacheHits / this.cacheTotal : 0,
      modelReady: isModelReady(),
    };
  }
}
