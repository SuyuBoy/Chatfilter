"""
弹幕语义聚类引擎入口 — PipelineEngine。

流程: 注册(6步预处理) → embedding → 在线聚类 → 周期维护
      缓存由 UnifiedCache 统一管理 (单表 + 反向索引)。
"""

import time
import random
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from config.settings import get_settings, Settings
from src.engine.canonical_registry import CanonicalRegistry
from src.engine.embedder import Embedder
from src.engine.micro_cluster import MicroCluster
from src.engine.pipeline_cache import CacheEntry


@dataclass
class SlotState:
    """一个语义槽位。"""
    slot_id: int
    cluster_id: str = ""
    canonical_text: str = ""
    total_count: int = 0
    top_examples: list[str] = field(default_factory=list)
    latest_raw: str = ""
    last_update: float = 0.0
    centroid: np.ndarray | None = None
    _member_embeddings: list[np.ndarray] = field(default_factory=list)
    _canonicals: set = field(default_factory=set)

    def _add_member_emb(self, emb: np.ndarray):
        self._member_embeddings.append(emb.copy())
        if len(self._member_embeddings) > 10:
            self._member_embeddings = self._member_embeddings[:5] + self._member_embeddings[-5:]

    def internal_similarity(self) -> float:
        if len(self._member_embeddings) < 2 or self.centroid is None:
            return 1.0
        sims = [float(np.dot(e, self.centroid)) for e in self._member_embeddings]
        return sum(sims) / len(sims)


class PipelineEngine:
    """弹幕语义聚类引擎。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = CanonicalRegistry(self.settings)
        self.embedder = Embedder(self.settings)
        self.slots: OrderedDict[str, SlotState] = OrderedDict()
        self.permanent: OrderedDict[str, SlotState] = OrderedDict()
        self._next_perm_id = 1
        self.total_ingested = 0
        self._initialized = False
        self._emb_times: list[float] = []
        self._cluster_times: list[float] = []

    async def initialize(self):
        if not self._initialized:
            await self.embedder.initialize()
            self._initialized = True

    def ingest(self, raw_text: str, msg_id: str = "") -> dict:
        self.total_ingested += 1
        result = self.registry.register(raw_text, msg_id=msg_id or str(self.total_ingested))
        if result.filtered:
            return {"cluster_id": "", "canonical": "", "raw_text": raw_text, "slot_id": 0,
                    "is_new": False, "filtered": True,
                    "stage_times": result.stage_times, "cache_hits": result.cache_hits or []}

        canonical = result.canonical_text

        # Embedding: 统一缓存优先
        t0 = time.perf_counter()
        emb = result.cached_embedding
        cache_entry = self.registry.cache.get(canonical)
        if emb is None and cache_entry is not None:
            emb = cache_entry.embedding
        if emb is None:
            emb = self.embedder._encode_sync([canonical])[0]
            self.registry.cache.backfill(canonical, emb)
        emb_ms = (time.perf_counter() - t0) * 1000
        self._emb_times.append(emb_ms)
        if len(self._emb_times) > 1000:
            self._emb_times = self._emb_times[-500:]

        t0 = time.perf_counter()

        # 聚类: 先热点 (严格) → 后常规 (普通)
        conf = self.settings.cluster
        perm_slot = self._find_in_dict(self.permanent, emb, len(result.raw_text),
                                       conf.permanent_centroid_threshold,
                                       conf.permanent_anchor_threshold,
                                       conf.permanent_split_variance_threshold)
        if perm_slot is not None:
            self._join_slot(perm_slot, emb, result.raw_text, canonical)
            # 更新缓存: 该 canonical 的 cluster_id
            self.registry.cache.put(canonical,
                                    CacheEntry(canonical=canonical, embedding=emb,
                                               cluster_id=perm_slot.cluster_id),
                                    canonical=canonical, cluster_id=perm_slot.cluster_id)
            self._maintenance()
            cluster_ms = (time.perf_counter() - t0) * 1000
            self._cluster_times.append(cluster_ms)
            return {"cluster_id": perm_slot.cluster_id, "canonical": canonical,
                    "raw_text": result.raw_text, "slot_id": perm_slot.slot_id,
                    "is_new": False, "filtered": False, "permanent": True,
                    "stage_times": result.stage_times, "cache_hits": result.cache_hits or [],
                    "embedding_ms": round(emb_ms, 3), "cluster_ms": round(cluster_ms, 3)}

        best_slot = self._find_in_dict(self.slots, emb, len(result.raw_text),
                                       conf.centroid_threshold, conf.anchor_threshold,
                                       conf.split_variance_threshold)
        if best_slot is not None and best_slot.centroid is not None:
            self._join_slot(best_slot, emb, result.raw_text, canonical)
            self.registry.cache.put(canonical,
                                    CacheEntry(canonical=canonical, embedding=emb,
                                               cluster_id=best_slot.cluster_id),
                                    canonical=canonical, cluster_id=best_slot.cluster_id)
            if best_slot.total_count >= conf.permanent_threshold:
                self._promote(best_slot)
            self._maintenance()
            cluster_ms = (time.perf_counter() - t0) * 1000
            self._cluster_times.append(cluster_ms)
            return {"cluster_id": best_slot.cluster_id, "canonical": canonical,
                    "raw_text": result.raw_text, "slot_id": best_slot.slot_id,
                    "is_new": False, "filtered": False,
                    "stage_times": result.stage_times, "cache_hits": result.cache_hits or [],
                    "embedding_ms": round(emb_ms, 3), "cluster_ms": round(cluster_ms, 3)}

        sid, cid = self._new_slot(canonical, emb, result.raw_text)
        self.registry.cache.put(canonical,
                                CacheEntry(canonical=canonical, embedding=emb, cluster_id=cid),
                                canonical=canonical, cluster_id=cid)
        self._maintenance()
        cluster_ms = (time.perf_counter() - t0) * 1000
        self._cluster_times.append(cluster_ms)
        return {"cluster_id": cid, "canonical": canonical,
                "raw_text": result.raw_text, "slot_id": sid,
                "is_new": True, "filtered": False,
                "stage_times": result.stage_times, "cache_hits": result.cache_hits or [],
                "embedding_ms": round(emb_ms, 3), "cluster_ms": round(cluster_ms, 3)}

    @staticmethod
    def _find_in_dict(d: OrderedDict, emb: np.ndarray, text_len: int,
                      centroid_th: float, anchor_th: float,
                      split_var_th: float) -> SlotState | None:
        best_sim, best_slot = -1.0, None
        for slot in d.values():
            if slot.centroid is None:
                continue
            mc = MicroCluster()
            mc.centroid = slot.centroid
            mc.anchor_examples = [slot.canonical_text]
            mc.anchor_embeddings = [slot.centroid]  # centroid 近似锚点
            if mc.can_join(emb, text_len, centroid_th, anchor_th):
                sim = float(np.dot(emb, slot.centroid))
                if len(slot._member_embeddings) >= 3 and slot.internal_similarity() < split_var_th:
                    continue
                if sim > best_sim:
                    best_sim, best_slot = sim, slot
        return best_slot

    def _join_slot(self, slot: SlotState, emb: np.ndarray, raw: str, canonical: str = ""):
        slot.total_count += 1
        slot.latest_raw = raw
        slot.last_update = time.time()
        old = max(slot.total_count - 1, 1)
        if slot.centroid is not None:
            slot.centroid = (slot.centroid * old + emb) / slot.total_count
        if raw not in slot.top_examples:
            slot.top_examples.append(raw)
        slot.top_examples = sorted(set(slot.top_examples), key=lambda t: -len(t))[:3]
        slot._add_member_emb(emb)
        if canonical:
            slot._canonicals.add(canonical)

    def _new_slot(self, canonical: str, emb: np.ndarray, raw: str) -> tuple[int, str]:
        sid = self._alloc_slot()
        cid = f"c{sid}"
        slot = SlotState(slot_id=sid, cluster_id=cid, canonical_text=canonical,
                         total_count=1, top_examples=[raw], latest_raw=raw,
                         last_update=time.time(), centroid=emb.copy(),
                         _canonicals={canonical})
        slot._add_member_emb(emb)
        self.slots[canonical] = slot
        return sid, cid

    def _alloc_slot(self) -> int:
        max_n = self.settings.cluster.max_slots
        if len(self.slots) < max_n:
            used = {s.slot_id for s in self.slots.values()}
            for i in range(1, max_n + 1):
                if i not in used:
                    return i
            return len(self.slots) + 1
        oldest = min(self.slots.values(), key=lambda s: s.last_update)
        sid = oldest.slot_id
        del self.slots[oldest.canonical_text]
        return sid

    # ── 热点晋升 ──

    def _recompute_centroid(self, slot: SlotState):
        if len(slot._member_embeddings) < 2:
            return
        new_c = np.mean(slot._member_embeddings, axis=0)
        slot.centroid = new_c / (np.linalg.norm(new_c) + 1e-8)
        best_text, best_sim = slot.canonical_text, -1.0
        for t in slot.top_examples:
            te = self.embedder._encode_sync([t])[0]
            sim = float(np.dot(te, slot.centroid))  # type: ignore[arg-type]
            if sim > best_sim:
                best_sim, best_text = sim, t
        if best_sim > 0:
            slot.canonical_text = best_text

    def _promote(self, slot: SlotState):
        canonical = slot.canonical_text
        if canonical not in self.slots:
            return
        self._recompute_centroid(slot)
        del self.slots[canonical]
        slot.slot_id = self._next_perm_id
        slot.cluster_id = f"p{self._next_perm_id}"
        self._next_perm_id += 1
        self.permanent[canonical] = slot

    # ── 周期维护 ──

    def _maintenance(self):
        conf = self.settings.cluster
        if self.total_ingested % conf.maintenance_interval != 0:
            return
        self._merge_similar(self.slots, conf.merge_threshold)
        self._split_degraded(self.slots, conf.split_variance_threshold, conf.max_slots)
        self._merge_similar(self.permanent, conf.permanent_merge_threshold, check_sentiment=True)
        self._split_degraded(self.permanent, conf.permanent_split_variance_threshold, 9999)
        now = time.time()
        expired = [c for c, s in self.permanent.items() if now - s.last_update > 600]
        for c in expired:
            del self.permanent[c]

    @staticmethod
    def _same_topic_diff_sentiment(a: SlotState, b: SlotState) -> bool:
        pos_words = {'好', '棒', '牛', '强', '厉害', '无敌', '神', '帅', '绝', '赞'}
        neg_words = {'烂', '差', '菜', '弱', '垃圾', '恶心', '难听', '丑', '蠢', '废'}
        texts_a = set(a.top_examples + [a.canonical_text])
        texts_b = set(b.top_examples + [b.canonical_text])
        words_a = set().union(*[set(t) for t in texts_a])
        words_b = set().union(*[set(t) for t in texts_b])
        shared = words_a & words_b
        return (len(shared) >= 2 and
                ((bool(words_a & pos_words) and bool(words_b & neg_words)) or
                 (bool(words_a & neg_words) and bool(words_b & pos_words))))

    def _merge_similar(self, d: OrderedDict, threshold: float, check_sentiment: bool = False):
        items = list(d.items())
        merged: set[str] = set()
        for i in range(len(items)):
            ci, si = items[i]
            if ci in merged: continue
            for j in range(i + 1, len(items)):
                cj, sj = items[j]
                if cj in merged: continue
                if si.centroid is None or sj.centroid is None: continue
                if float(np.dot(si.centroid, sj.centroid)) > threshold:  # type: ignore[arg-type]
                    if check_sentiment and self._same_topic_diff_sentiment(si, sj):
                        continue
                    if si.total_count >= sj.total_count:
                        self._absorb(si, sj); merged.add(cj)
                    else:
                        self._absorb(sj, si); merged.add(ci)
        for c in merged:
            del d[c]

    def _absorb(self, target: SlotState, source: SlotState):
        n = target.total_count + source.total_count
        if target.centroid is not None and source.centroid is not None:
            target.centroid = (target.centroid * target.total_count +
                               source.centroid * source.total_count) / n
        target.total_count = n
        target.last_update = max(target.last_update, source.last_update)
        for raw in source.top_examples:
            if raw not in target.top_examples:
                target.top_examples.append(raw)
        target.top_examples = sorted(set(target.top_examples), key=lambda t: -len(t))[:3]
        if source.latest_raw and not target.latest_raw:
            target.latest_raw = source.latest_raw
        for emb in source._member_embeddings:
            target._add_member_emb(emb)
        target._canonicals.update(source._canonicals)

    def _split_degraded(self, d: OrderedDict, threshold: float, max_slots: int):
        to_split: list[tuple[str, SlotState]] = []
        for canonical, slot in d.items():
            if len(slot._member_embeddings) < 4: continue
            if slot.internal_similarity() < threshold:
                to_split.append((canonical, slot))
        for canonical, slot in to_split:
            if canonical not in d: continue
            embs = slot._member_embeddings
            if len(embs) < 4: continue
            labels = self._kmeans_2(embs)
            g0 = [embs[k] for k in range(len(embs)) if labels[k] == 0]
            g1 = [embs[k] for k in range(len(embs)) if labels[k] == 1]
            if len(g0) < 2 or len(g1) < 2: continue
            if max_slots < 9999:
                used = {s.slot_id for s in d.values() if s.slot_id != slot.slot_id}
                new_id = None
                for i in range(1, max_slots + 1):
                    if i not in used: new_id = i; break
                if new_id is None: continue
            else:
                new_id = self._next_perm_id; self._next_perm_id += 1
            # 标记原簇缓存无效
            self.registry.cache.mark_invalid(slot.cluster_id)
            c0 = np.mean(g0, axis=0)
            slot.centroid = c0 / (np.linalg.norm(c0) + 1e-8)
            slot.total_count = len(g0)
            slot._member_embeddings = g0
            c1 = np.mean(g1, axis=0)
            c1 = c1 / (np.linalg.norm(c1) + 1e-8)
            new_canonical = f"{slot.canonical_text}*"
            prefix = "p" if max_slots == 9999 else "c"
            d[new_canonical] = SlotState(
                slot_id=new_id, cluster_id=f"{prefix}{new_id}",
                canonical_text=slot.canonical_text, total_count=len(g1),
                top_examples=slot.top_examples[:1], latest_raw=slot.latest_raw,
                last_update=slot.last_update, centroid=c1, _member_embeddings=g1)

    @staticmethod
    def _kmeans_2(embs: list[np.ndarray]) -> list[int]:
        n = len(embs)
        if n < 2: return [0] * n
        rng = random.Random(42)
        idx = rng.sample(range(n), min(2, n))
        c0, c1 = embs[idx[0]].copy(), embs[idx[1]].copy()
        labels = [0] * n
        for _ in range(10):
            changed = False
            for i, e in enumerate(embs):
                nl = 0 if float(np.dot(e, c0)) >= float(np.dot(e, c1)) else 1
                if nl != labels[i]: changed = True; labels[i] = nl
            if not changed: break
            g0 = [embs[i] for i in range(n) if labels[i] == 0]
            g1 = [embs[i] for i in range(n) if labels[i] == 1]
            if g0: c0 = np.mean(g0, axis=0)
            if g1: c1 = np.mean(g1, axis=0)
        return labels

    # ── 查询 ──

    def _serialize_slot(self, s: SlotState) -> dict:
        all_members: list[dict] = []
        for c in s._canonicals:
            all_members.extend(self.registry.get_canonical_members(c, limit=5))
        return {"cluster_id": s.cluster_id, "slot_id": s.slot_id,
                "canonical_text": s.canonical_text, "total_count": s.total_count,
                "type": "semantic", "top_examples": s.top_examples[:5],
                "members": [{"text": m["raw"], "count": 1} for m in all_members[:8]],
                "latest_raw": s.latest_raw}

    def get_clusters(self) -> list[dict]:
        return [self._serialize_slot(s) for s in
                sorted(self.slots.values(), key=lambda s: -s.total_count)]

    def get_permanent(self) -> list[dict]:
        return [self._serialize_slot(s) for s in
                sorted(self.permanent.values(), key=lambda s: -s.total_count)]

    def _avg(self, vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def get_state(self) -> dict:
        cache_stats = self.registry.cache.stats()
        return {"ingested": self.total_ingested, "unique": self.registry.unique_count,
                "centroid": self.settings.cluster.centroid_threshold,
                "anchor": self.settings.cluster.anchor_threshold,
                "max_slots": self.settings.cluster.max_slots,
                "clusters": self.get_clusters(), "permanent": self.get_permanent(),
                "perm_count": len(self.permanent),
                "cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "stage_timing": self.registry.timer.stats(),
                "embedding_avg_ms": round(self._avg(self._emb_times), 3),
                "cluster_avg_ms": round(self._avg(self._cluster_times), 3)}

    def render(self) -> str:
        saved = self.total_ingested - self.registry.unique_count
        ct, at = self.settings.cluster.centroid_threshold, self.settings.cluster.anchor_threshold
        lines = ["═" * 55,
                 f" 弹幕语义聚类 | 摄入:{self.total_ingested} 唯一:{self.registry.unique_count} 省:{saved}",
                 f" 粒度: centroid={ct:.2f} anchor={at:.2f} | 热点:{len(self.permanent)}"]
        lines.append("═" * 55)
        if self.permanent:
            lines.append("  🏛️ 热点语义:")
            for s in sorted(self.permanent.values(), key=lambda s: -s.total_count):
                ex = " / ".join(s.top_examples[:2])
                lines.append(f"  [P{s.slot_id:02d}] {s.canonical_text[:16]:16s} {s.total_count:>4}次  {ex[:30]}")
        lines.append("─" * 55)
        slot_list = sorted(self.slots.values(), key=lambda s: s.slot_id)
        max_n = self.settings.cluster.max_slots
        for i in range(max_n):
            if i < len(slot_list):
                s = slot_list[i]
                ex = " / ".join(s.top_examples[:3])
                lines.append(f"  [{s.slot_id:02d}] {s.canonical_text[:16]:16s} {s.total_count:>4}次  {ex[:30]}")
            else:
                lines.append(f"  [{i+1:02d}] —")
        lines.append("─" * 55)
        lines.append("✏️  输入: ")
        return "\n".join(lines)
