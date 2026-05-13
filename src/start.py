"""
弹幕语义聚类引擎入口 — PipelineEngine。

流程: 注册 (6步预处理) → embedding → 在线聚类 → 周期维护(合并/拆分)
       → 热点语义晋升 + 10min TTL 淘汰

组件:
  CanonicalRegistry: 三层归一引擎 + 统一缓存 + 计时
  Embedder:          BGE 模型推理 + 本地缓存
  SlotState:         单个语义槽位 (常规 + 永久)
  MicroCluster:      微簇判断 (can_join)

聚类流程:
  1. 新消息 embedding 到达
  2. 先匹配热点语义 (更严格的 0.75/0.82 阈值)
  3. 再匹配常规槽位 (普通 0.65/0.75 阈值)
  4. 都不匹配 → 新建常规槽位
  5. 常规槽位 count > 100 → 晋升热点语义
  6. 每 300 条: 合并相似簇 + K-Means 拆分退化簇
  7. 热点 10min 无更新 → 淘汰

内容穿透:
  SlotState._canonicals 追踪所有加入过的 canonical。
  get_clusters() 合并所有 canonical 的成员, 跨 canonical 也能穿透原始文本。
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


@dataclass
class SlotState:
    """一个语义槽位 — 聚类的基本单元。

    Attributes:
        slot_id:            槽位编号 (常规 1-40, 热点自增)
        cluster_id:         簇标识 ("c{sid}" 或 "p{sid}")
        canonical_text:     规范文本 (聚类摘要标签)
        total_count:        消息总数
        top_examples:       高频原始弹幕样本 (最多 3 个)
        latest_raw:         最新原始弹幕
        last_update:        最后更新时间戳 (用于 LRU 淘汰 / TTL)
        centroid:           归一化 centroid 向量
        _member_embeddings: 已存成员 embedding (最多 10 个, 用于一致性检查)
        _canonicals:        该簇包含的所有 canonical_text (跨 canonical 穿透)
    """
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
        """添加成员 embedding。超过 10 个时保留首尾各 5 个 (均匀采样)。"""
        self._member_embeddings.append(emb.copy())
        if len(self._member_embeddings) > 10:
            self._member_embeddings = self._member_embeddings[:5] + self._member_embeddings[-5:]

    def internal_similarity(self) -> float:
        """簇内成员与 centroid 的平均余弦相似度。成员少时返回 1.0。"""
        if len(self._member_embeddings) < 2 or self.centroid is None:
            return 1.0
        sims = [float(np.dot(e, self.centroid)) for e in self._member_embeddings]
        return sum(sims) / len(sims)


class PipelineEngine:
    """弹幕语义聚类引擎。

    槽位: self.slots (常规, max_slots 上限) + self.permanent (热点, 无上限)
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.registry = CanonicalRegistry(self.settings)
        self.embedder = Embedder(self.settings)
        self.slots: OrderedDict[str, SlotState] = OrderedDict()        # 常规槽位
        self.permanent: OrderedDict[str, SlotState] = OrderedDict()   # 热点语义
        self._next_perm_id = 1                                         # 热点自增 ID
        self.total_ingested = 0
        self._initialized = False
        self._emb_times: list[float] = []       # embedding 耗时记录 (最近 500)
        self._cluster_times: list[float] = []   # 聚类耗时记录 (最近 500)

    async def initialize(self):
        """加载 BGE 模型 (仅首次调用生效)。"""
        if not self._initialized:
            await self.embedder.initialize()
            self._initialized = True

    # ── 摄入 ──

    def ingest(self, raw_text: str, msg_id: str = "") -> dict:
        """摄入一条弹幕。

        Returns:
            dict with cluster_id, canonical, raw_text, slot_id, is_new, filtered,
                 stage_times, cache_hits, embedding_ms, cluster_ms
        """
        self.total_ingested += 1

        # 1) 注册: 预处理管线 → RegisterResult
        result = self.registry.register(raw_text, msg_id=msg_id or str(self.total_ingested))
        if result.filtered:
            return {"cluster_id": "", "canonical": "", "raw_text": raw_text, "slot_id": 0,
                    "is_new": False, "filtered": True,
                    "stage_times": result.stage_times, "cache_hits": result.cache_hits or []}

        canonical = result.canonical_text

        # 2) Embedding: 优先用统一缓存带出的, 否则计算并回填
        t0 = time.perf_counter()
        emb = result.cached_embedding                        # 统一缓存命中 → 已有 embedding
        if emb is None:
            emb = self.embedder._cache.peek(canonical)       # 本地缓存兜底
        if emb is None:
            emb = self.embedder._encode_sync([canonical])[0]  # 实际计算
            self.embedder._cache.put(canonical, emb)
            self.registry.cache.backfill(canonical, emb)      # 回填统一缓存
        emb_ms = (time.perf_counter() - t0) * 1000
        self._emb_times.append(emb_ms)
        if len(self._emb_times) > 1000:
            self._emb_times = self._emb_times[-500:]

        t0 = time.perf_counter()

        # 3) 聚类匹配: 先热点 (严格阈值) → 后常规 (普通阈值)
        conf = self.settings.cluster
        perm_slot = self._find_in_dict(
            self.permanent, emb, len(result.raw_text),
            conf.permanent_centroid_threshold, conf.permanent_anchor_threshold,
            conf.permanent_split_variance_threshold)
        if perm_slot is not None:
            self._join_slot(perm_slot, emb, result.raw_text, canonical)
            self._maintenance()
            cluster_ms = (time.perf_counter() - t0) * 1000
            self._cluster_times.append(cluster_ms)
            return {"cluster_id": perm_slot.cluster_id, "canonical": canonical,
                    "raw_text": result.raw_text, "slot_id": perm_slot.slot_id,
                    "is_new": False, "filtered": False, "permanent": True,
                    "stage_times": result.stage_times, "cache_hits": result.cache_hits or [],
                    "embedding_ms": round(emb_ms, 3), "cluster_ms": round(cluster_ms, 3)}

        best_slot = self._find_in_dict(
            self.slots, emb, len(result.raw_text),
            conf.centroid_threshold, conf.anchor_threshold,
            conf.split_variance_threshold)
        if best_slot is not None and best_slot.centroid is not None:
            self._join_slot(best_slot, emb, result.raw_text, canonical)
            # 晋升检查: count > 阈值 → 升级为热点
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

        # 4) 无匹配 → 新建常规槽位
        sid, cid = self._new_slot(canonical, emb, result.raw_text)
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
        """在给定槽位字典中查找最佳匹配簇。

        检查条件:
          1. MicroCluster.can_join() 通过 (centroid + anchor + length_penalty)
          2. 内部一致性: 退化簇 (内部相似度 < split_var_th) 拒绝新成员
        """
        best_sim, best_slot = -1.0, None
        for slot in d.values():
            if slot.centroid is None:
                continue
            # 构造临时 MicroCluster 用于判断
            mc = MicroCluster()
            mc.centroid = slot.centroid
            mc.anchor_examples = [slot.canonical_text]
            mc.anchor_embeddings = []
            for t in [slot.canonical_text]:
                mc.anchor_embeddings.append(slot.centroid)  # 用 centroid 近似锚点 embedding
            if mc.can_join(emb, text_len, centroid_th, anchor_th):
                sim = float(np.dot(emb, slot.centroid))
                # 退化簇拒绝新成员
                if len(slot._member_embeddings) >= 3 and slot.internal_similarity() < split_var_th:
                    continue
                if sim > best_sim:
                    best_sim, best_slot = sim, slot
        return best_slot

    def _join_slot(self, slot: SlotState, emb: np.ndarray, raw: str, canonical: str = ""):
        """将新消息加入已有槽位: 更新 centroid, top_examples, _canonicals。"""
        slot.total_count += 1
        slot.latest_raw = raw
        slot.last_update = time.time()
        # 增量更新 centroid: 加权平均
        old = max(slot.total_count - 1, 1)
        if slot.centroid is not None:
            slot.centroid = (slot.centroid * old + emb) / slot.total_count
        # 维护 top_examples (去重 + 按长度排序取 top 3)
        if raw not in slot.top_examples:
            slot.top_examples.append(raw)
        slot.top_examples = sorted(set(slot.top_examples), key=lambda t: -len(t))[:3]
        slot._add_member_emb(emb)
        if canonical:
            slot._canonicals.add(canonical)  # 跨 canonical 追踪

    def _new_slot(self, canonical: str, emb: np.ndarray, raw: str) -> tuple[int, str]:
        """新建常规槽位。如果满槽则 LRU 淘汰最旧的。"""
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
        """分配常规槽位 ID。未满则找最小空闲 ID, 满则 LRU 淘汰。"""
        max_n = self.settings.cluster.max_slots
        if len(self.slots) < max_n:
            used = {s.slot_id for s in self.slots.values()}
            for i in range(1, max_n + 1):
                if i not in used:
                    return i
            return len(self.slots) + 1
        # 满槽: 淘汰最近活跃时间最早的
        oldest = min(self.slots.values(), key=lambda s: s.last_update)
        sid = oldest.slot_id
        del self.slots[oldest.canonical_text]
        return sid

    # ── 热点晋升 ──

    def _recompute_centroid(self, slot: SlotState):
        """用所有已存成员 embedding 重算 centroid, 消除增量漂移。
        同时选最靠近新 centroid 的 top_example 作为 canonical_text (摘要)。
        """
        if len(slot._member_embeddings) < 2:
            return
        new_c = np.mean(slot._member_embeddings, axis=0)
        slot.centroid = new_c / (np.linalg.norm(new_c) + 1e-8)
        # 找最靠近新 centroid 的 top_example
        best_text, best_sim = slot.canonical_text, -1.0
        for t in slot.top_examples:
            te = self.embedder._cache.peek(t)
            if te is None:
                te = self.embedder._encode_sync([t])[0]
            sim = float(np.dot(te, slot.centroid))
            if sim > best_sim:
                best_sim, best_text = sim, t
        if best_sim > 0:
            slot.canonical_text = best_text

    def _promote(self, slot: SlotState):
        """晋升到热点语义区: 先重算 centroid + 选摘要, 再迁移。"""
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
        """每 maintenance_interval 条消息触发: 合并相似簇 + 拆分退化簇 + 热点 TTL 淘汰。"""
        conf = self.settings.cluster
        if self.total_ingested % conf.maintenance_interval != 0:
            return
        # 常规槽位
        self._merge_similar(self.slots, conf.merge_threshold)
        self._split_degraded(self.slots, conf.split_variance_threshold, conf.max_slots)
        # 热点语义 — 更严格合并 + 情感守卫
        self._merge_similar(self.permanent, conf.permanent_merge_threshold, check_sentiment=True)
        self._split_degraded(self.permanent, conf.permanent_split_variance_threshold, 9999)
        # TTL 淘汰: 10min 无更新 → 移除
        now = time.time()
        expired = [c for c, s in self.permanent.items() if now - s.last_update > 600]
        for c in expired:
            del self.permanent[c]

    @staticmethod
    def _same_topic_diff_sentiment(a: SlotState, b: SlotState) -> bool:
        """检查两个簇是否同一主题但相反情感。

        例如 "周杰伦唱得好" vs "周杰伦唱的烂" — 共享词 ≥2 且一方含正面词一方含负面词。
        用于热点合并时的情感守卫, 防止同主题正反面被错误合并。
        """
        pos_words = {'好', '棒', '牛', '强', '厉害', '无敌', '神', '帅', '绝', '赞'}
        neg_words = {'烂', '差', '菜', '弱', '垃圾', '恶心', '难听', '丑', '蠢', '废'}
        texts_a = set(a.top_examples + [a.canonical_text])
        texts_b = set(b.top_examples + [b.canonical_text])
        words_a = set().union(*[set(t) for t in texts_a])
        words_b = set().union(*[set(t) for t in texts_b])
        shared = words_a & words_b  # 共同字符
        has_pos_a = bool(words_a & pos_words)
        has_neg_a = bool(words_a & neg_words)
        has_pos_b = bool(words_b & pos_words)
        has_neg_b = bool(words_b & neg_words)
        return len(shared) >= 2 and ((has_pos_a and has_neg_b) or (has_neg_a and has_pos_b))

    def _merge_similar(self, d: OrderedDict, threshold: float, check_sentiment: bool = False):
        """合并 centroid 相似度 > threshold 的槽位。"""
        items = list(d.items())
        merged: set[str] = set()
        for i in range(len(items)):
            ci, si = items[i]
            if ci in merged:
                continue
            for j in range(i + 1, len(items)):
                cj, sj = items[j]
                if cj in merged:
                    continue
                if si.centroid is None or sj.centroid is None:
                    continue
                if float(np.dot(si.centroid, sj.centroid)) > threshold:
                    if check_sentiment and self._same_topic_diff_sentiment(si, sj):
                        continue  # 情感守卫: 同主题反情感不合并
                    if si.total_count >= sj.total_count:
                        self._absorb(si, sj); merged.add(cj)
                    else:
                        self._absorb(sj, si); merged.add(ci)
        for c in merged:
            del d[c]

    def _absorb(self, target: SlotState, source: SlotState):
        """将 source 的所有数据吸收到 target。"""
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
        """K-Means (k=2) 拆分内部相似度 < threshold 的退化槽位。"""
        to_split: list[tuple[str, SlotState]] = []
        for canonical, slot in d.items():
            if len(slot._member_embeddings) < 4:
                continue
            if slot.internal_similarity() < threshold:
                to_split.append((canonical, slot))

        for canonical, slot in to_split:
            if canonical not in d:
                continue
            embs = slot._member_embeddings
            if len(embs) < 4:
                continue
            # 极简 K-Means: 随机种子 → 分配 → 更新中心 → 迭代 (最多 10 轮)
            labels = self._kmeans_2(embs)
            g0 = [embs[k] for k in range(len(embs)) if labels[k] == 0]
            g1 = [embs[k] for k in range(len(embs)) if labels[k] == 1]
            if len(g0) < 2 or len(g1) < 2:
                continue

            # 分配新 slot_id
            if max_slots < 9999:
                used = {s.slot_id for s in d.values() if s.slot_id != slot.slot_id}
                new_id = None
                for i in range(1, max_slots + 1):
                    if i not in used:
                        new_id = i; break
                if new_id is None:
                    continue  # 常规槽位已满, 跳过拆分
            else:
                new_id = self._next_perm_id
                self._next_perm_id += 1

            # 子簇 0: 复用旧 slot
            c0 = np.mean(g0, axis=0)
            slot.centroid = c0 / (np.linalg.norm(c0) + 1e-8)
            slot.total_count = len(g0)
            slot._member_embeddings = g0

            # 子簇 1: 新建 slot
            c1 = np.mean(g1, axis=0)
            c1 = c1 / (np.linalg.norm(c1) + 1e-8)
            new_canonical = f"{slot.canonical_text}*"
            prefix = "p" if max_slots == 9999 else "c"
            d[new_canonical] = SlotState(
                slot_id=new_id, cluster_id=f"{prefix}{new_id}",
                canonical_text=slot.canonical_text,
                total_count=len(g1), top_examples=slot.top_examples[:1],
                latest_raw=slot.latest_raw, last_update=slot.last_update,
                centroid=c1, _member_embeddings=g1,
            )

    @staticmethod
    def _kmeans_2(embs: list[np.ndarray]) -> list[int]:
        """极简 K-Means (k=2): 随机种子 → 分配 → 更新中心 → 迭代收敛。"""
        n = len(embs)
        if n < 2:
            return [0] * n
        rng = random.Random(42)
        idx = rng.sample(range(n), min(2, n))
        c0, c1 = embs[idx[0]].copy(), embs[idx[1]].copy()
        labels = [0] * n
        for _ in range(10):
            changed = False
            for i, e in enumerate(embs):
                nl = 0 if float(np.dot(e, c0)) >= float(np.dot(e, c1)) else 1
                if nl != labels[i]:
                    changed = True; labels[i] = nl
            if not changed:
                break
            g0 = [embs[i] for i in range(n) if labels[i] == 0]
            g1 = [embs[i] for i in range(n) if labels[i] == 1]
            if g0: c0 = np.mean(g0, axis=0)
            if g1: c1 = np.mean(g1, axis=0)
        return labels

    # ── 查询 ──

    def _serialize_slot(self, s: SlotState) -> dict:
        """将 SlotState 序列化为前端可用的字典。"""
        all_members: list[dict] = []
        for c in s._canonicals:
            all_members.extend(self.registry.get_canonical_members(c, limit=5))
        return {
            "cluster_id": s.cluster_id, "slot_id": s.slot_id,
            "canonical_text": s.canonical_text, "total_count": s.total_count,
            "type": "semantic", "top_examples": s.top_examples[:5],
            "members": [{"text": m["raw"], "count": 1} for m in all_members[:8]],
            "latest_raw": s.latest_raw,
        }

    def get_clusters(self) -> list[dict]:
        """返回当前常规聚类摘要, 按 total_count 降序。"""
        return [self._serialize_slot(s) for s in
                sorted(self.slots.values(), key=lambda s: -s.total_count)]

    def get_permanent(self) -> list[dict]:
        """返回当前热点语义摘要, 按 total_count 降序。"""
        return [self._serialize_slot(s) for s in
                sorted(self.permanent.values(), key=lambda s: -s.total_count)]

    def _avg(self, vals: list[float]) -> float:
        """计算列表平均值。"""
        return sum(vals) / len(vals) if vals else 0.0

    def get_state(self) -> dict:
        """返回引擎完整快照 (供 SSE / API 查询)。"""
        cache_stats = self.registry.cache.stats()
        return {
            "ingested": self.total_ingested, "unique": self.registry.unique_count,
            "centroid": self.settings.cluster.centroid_threshold,
            "anchor": self.settings.cluster.anchor_threshold,
            "max_slots": self.settings.cluster.max_slots,
            "clusters": self.get_clusters(),
            "permanent": self.get_permanent(),
            "perm_count": len(self.permanent),
            "cache_hits": cache_stats["hits"], "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "stage_timing": self.registry.timer.stats(),
            "embedding_avg_ms": round(self._avg(self._emb_times), 3),
            "cluster_avg_ms": round(self._avg(self._cluster_times), 3),
        }

    # ── 终端渲染 ──

    def render(self) -> str:
        """终端 UI 字符串 (40 行固定, 兼容旧接口)。"""
        saved = self.total_ingested - self.registry.unique_count
        ct, at = self.settings.cluster.centroid_threshold, self.settings.cluster.anchor_threshold
        lines = ["═" * 55,
                 f" 弹幕语义聚类 | 摄入:{self.total_ingested} 唯一:{self.registry.unique_count} 省:{saved}",
                 f" 粒度: centroid={ct:.2f} anchor={at:.2f} | 热点:{len(self.permanent)}"]
        lines.append("═" * 55)
        # 热点区
        if self.permanent:
            lines.append("  🏛️ 热点语义:")
            for s in sorted(self.permanent.values(), key=lambda s: -s.total_count):
                ex = " / ".join(s.top_examples[:2])
                lines.append(f"  [P{s.slot_id:02d}] {s.canonical_text[:16]:16s} {s.total_count:>4}次  {ex[:30]}")
        lines.append("─" * 55)
        # 常规区
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
