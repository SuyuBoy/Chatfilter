"""
PipelineEngine — 业务编排入口。

流程: 注册(预处理) → embedding → 在线聚类 → 周期维护
      缓存由 UnifiedCache 统一管理。聚类逻辑见 cluster_engine.py。
"""

import time

import numpy as np

from config.settings import get_settings, Settings
from src.engine.canonical_registry import CanonicalRegistry
from src.engine.embedder import Embedder
from src.engine.cluster_engine import ClusterEngine
from src.engine.pipeline_cache import CacheEntry, PipelineTimer


class PipelineEngine:
    """编排层: 注册 → embedding → 聚类 → 查询。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = CanonicalRegistry(self.settings)
        self.embedder = Embedder(self.settings)
        self.cluster = ClusterEngine(self.settings)
        self.total_ingested = 0
        self._initialized = False
        self.timer = PipelineTimer()

    async def initialize(self):
        if not self._initialized:
            await self.embedder.initialize()
            self._initialized = True

    def ingest_batch(self, texts: list[str]) -> list[dict]:
        """批量摄入: 全部注册 → 攒批编码 → 再聚类。"""
        results = []
        need_emb: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            r = self.registry.register(t, msg_id=str(self.total_ingested + i + 1))
            results.append(r)
            if not r.filtered and r.cached_embedding is None:
                entry = self.registry.cache.get(r.canonical_text)
                if entry is None or entry.embedding is None:
                    need_emb.append((i, r.canonical_text))

        self.total_ingested += len(texts)

        # 批量编码
        canonicals = [c for _, c in need_emb]
        if canonicals:
            t0 = time.perf_counter()
            embs = self.embedder.encode_batched(canonicals)
            emb_ms = (time.perf_counter() - t0) * 1000
            per_item_ms = emb_ms / len(canonicals)
            for _ in canonicals:
                self.timer.record("embedding", per_item_ms)
            for (idx, canonical), emb in zip(need_emb, embs):
                results[idx].cached_embedding = emb
                self.registry.cache.backfill(canonical, emb)

        return [self._cluster_result(r) for r in results]

    def ingest(self, raw_text: str, msg_id: str = "") -> dict:
        self.total_ingested += 1
        result = self.registry.register(raw_text, msg_id=msg_id or str(self.total_ingested))
        if result.filtered:
            return {"cluster_id": "", "canonical": "", "raw_text": raw_text,
                    "slot_id": 0, "is_new": False, "filtered": True,
                    "cache_hits": result.cache_hits or []}

        canonical = result.canonical_text
        emb = result.cached_embedding
        entry = self.registry.cache.get(canonical)
        if emb is None and entry is not None:
            emb = entry.embedding
        if emb is None:
            t0 = time.perf_counter()
            emb = self.embedder._encode_sync([canonical])[0]
            self.timer.record("embedding", (time.perf_counter() - t0) * 1000)
            self.registry.cache.backfill(canonical, emb)

        return self._cluster_result(result)

    def _cluster_result(self, result) -> dict:
        if result.filtered:
            return {"cluster_id": "", "canonical": "", "raw_text": result.raw_text,
                    "slot_id": 0, "is_new": False, "filtered": True,
                    "cache_hits": result.cache_hits or []}

        canonical = result.canonical_text
        emb = result.cached_embedding
        if emb is None:
            entry = self.registry.cache.get(canonical)
            emb = entry.embedding if entry else None
        if emb is None:
            t0 = time.perf_counter()
            emb = self.embedder._encode_sync([canonical])[0]
            self.timer.record("embedding", (time.perf_counter() - t0) * 1000)
            self.registry.cache.backfill(canonical, emb)

        conf = self.settings.cluster
        t0 = time.perf_counter()
        slot, is_perm = self.cluster.find_cluster(emb, len(result.raw_text), result.raw_text)

        if slot is not None:
            self.cluster.join(slot, emb, result.raw_text, canonical)
            self.registry.cache.put(canonical,
                                    CacheEntry(canonical=canonical, embedding=emb,
                                               cluster_id=slot.cluster_id),
                                    canonical=canonical, cluster_id=slot.cluster_id)
            if not is_perm and slot.total_count >= conf.permanent_threshold:
                self.cluster.promote(slot, self.embedder)
            self.cluster.maintenance(self.total_ingested, self.registry.cache)
            self.timer.record("cluster", (time.perf_counter() - t0) * 1000)
            return {"cluster_id": slot.cluster_id, "canonical": canonical,
                    "raw_text": result.raw_text, "slot_id": slot.slot_id,
                    "is_new": False, "filtered": False, "permanent": is_perm,
                    "cache_hits": result.cache_hits or []}

        sid, cid = self.cluster.new_slot(canonical, emb, result.raw_text)
        self.registry.cache.put(canonical,
                                CacheEntry(canonical=canonical, embedding=emb, cluster_id=cid),
                                canonical=canonical, cluster_id=cid)
        self.cluster.maintenance(self.total_ingested, self.registry.cache)
        self.timer.record("cluster", (time.perf_counter() - t0) * 1000)
        return {"cluster_id": cid, "canonical": canonical,
                "raw_text": result.raw_text, "slot_id": sid,
                "is_new": True, "filtered": False,
                "cache_hits": result.cache_hits or []}

    def get_clusters(self) -> list[dict]:
        return self.cluster.get_clusters(self.registry)

    def get_permanent(self) -> list[dict]:
        return self.cluster.get_permanent(self.registry)

    def get_state(self) -> dict:
        cache_stats = self.registry.cache.stats()
        emb_stage = self.timer.stages.get("embedding")
        clu_stage = self.timer.stages.get("cluster")
        return {"ingested": self.total_ingested, "unique": self.registry.unique_count,
                "centroid": self.settings.cluster.centroid_threshold,
                "anchor": self.settings.cluster.anchor_threshold,
                "max_slots": self.settings.cluster.max_slots,
                "clusters": self.get_clusters(), "permanent": self.get_permanent(),
                "perm_count": len(self.cluster.permanent),
                "cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "embedding_avg_ms": round(emb_stage.recent_avg_ms, 2) if emb_stage else None,
                "cluster_avg_ms": round(clu_stage.recent_avg_ms, 2) if clu_stage else None,
                "stage_timing": self.registry.timer.stats()}

    def render(self) -> str:
        saved = self.total_ingested - self.registry.unique_count
        lines = ["═" * 55, f" 弹幕语义聚类 | 摄入:{self.total_ingested} 唯一:{self.registry.unique_count} 省:{saved}"]
        timing = self.registry.timer.stats()
        if timing:
            parts = [f"{n.split('_',1)[-1]}:{s['recent_avg_ms']:.1f}ms" for n, s in timing.items()]
            lines.append(f" 耗时: {' | '.join(parts)}")
        emb_stage = self.timer.stages.get("embedding")
        clu_stage = self.timer.stages.get("cluster")
        if emb_stage or clu_stage:
            ep = f"emb:{emb_stage.recent_avg_ms:.1f}ms" if emb_stage else "emb:—"
            cp = f"cluster:{clu_stage.recent_avg_ms:.1f}ms" if clu_stage else "cluster:—"
            lines.append(f" 后端: {ep}  {cp}")
        lines.append(self.cluster.render())
        lines.append("✏️  输入: ")
        return "\n".join(lines)
