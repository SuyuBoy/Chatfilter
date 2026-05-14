"""
在线聚类引擎 — Leader-Follower + 周期维护(合并/拆分) + 热点语义。

职责:
  - Leader-Follower 双阈值匹配 (先热点后常规)
  - 簇内一致性检查 + 周期合并/拆分
  - 热点语义晋升 (count>100) + 10min TTL 淘汰
  - 情感守卫 (同主题反情感不合并)
  - K-Means(k=2) 拆分退化簇
"""

import time
import random
from collections import OrderedDict, Counter
from dataclasses import dataclass, field

import numpy as np

from config.settings import Settings, get_settings
from src.engine.micro_cluster import MicroCluster
from src.engine.keyword_util import extract_keywords


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
    keyword_freq: dict = field(default_factory=dict)  # 词→出现次数
    top_keywords: list[str] = field(default_factory=list)  # 前 top_k 个关键词

    def _add_member_emb(self, emb: np.ndarray, max_keep: int = 10):
        self._member_embeddings.append(emb.copy())
        if len(self._member_embeddings) > max_keep:
            half = max_keep // 2
            self._member_embeddings = self._member_embeddings[:half] + self._member_embeddings[-half:]

    def internal_similarity(self) -> float:
        if len(self._member_embeddings) < 2 or self.centroid is None:
            return 1.0
        sims = [float(np.dot(e, self.centroid)) for e in self._member_embeddings]
        return sum(sims) / len(sims)


class ClusterEngine:
    """在线聚类引擎: 槽位管理 + 匹配 + 周期维护 + 热点晋升。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.slots: OrderedDict[str, SlotState] = OrderedDict()
        self.permanent: OrderedDict[str, SlotState] = OrderedDict()
        self._next_perm_id = 1

    def find_cluster(self, emb: np.ndarray, text_len: int,
                     raw_text: str = "") -> tuple[SlotState | None, bool]:
        """找最佳匹配簇。返回 (slot, is_permanent)。"""
        self._check_dim(emb)
        conf = self.settings.cluster
        keywords = extract_keywords(raw_text, conf.keyword_topk) if raw_text else None
        kw = conf.keyword_weight

        slot = self._find_in_dict(self.permanent, emb, text_len,
                                  conf.permanent_centroid_threshold,
                                  conf.permanent_anchor_threshold,
                                  conf.permanent_split_variance_threshold,
                                  keywords, kw)
        if slot is not None:
            return slot, True

        slot = self._find_in_dict(self.slots, emb, text_len,
                                  conf.centroid_threshold, conf.anchor_threshold,
                                  conf.split_variance_threshold,
                                  keywords, kw)
        return slot, False

    def _check_dim(self, emb: np.ndarray):
        """检测 embedding 维度是否与已有 centroid 匹配。不匹配则清空。"""
        for d in [self.slots, self.permanent]:
            for s in list(d.values()):
                if s.centroid is not None and len(s.centroid) != len(emb):
                    d.clear()
                    return

    @staticmethod
    def _find_in_dict(d: OrderedDict, emb: np.ndarray, text_len: int,
                      centroid_th: float, anchor_th: float,
                      split_var_th: float, keywords: list[str] | None = None,
                      keyword_weight: float = 0.7) -> SlotState | None:
        best_sim, best_slot = -1.0, None
        for slot in d.values():
            if slot.centroid is None:
                continue
            mc = MicroCluster()
            mc.centroid = slot.centroid
            mc.anchor_examples = [slot.canonical_text]
            mc.anchor_embeddings = [slot.centroid]
            if slot.top_keywords:
                mc.top_keywords = slot.top_keywords
            mc.keyword_weight = keyword_weight
            if mc.can_join(emb, text_len, keywords, centroid_th, anchor_th):
                sim = float(np.dot(emb, slot.centroid))
                if len(slot._member_embeddings) >= 3 and slot.internal_similarity() < split_var_th:
                    continue
                if sim > best_sim:
                    best_sim, best_slot = sim, slot
        return best_slot

    def join(self, slot: SlotState, emb: np.ndarray, raw: str, canonical: str = ""):
        """加入已有槽位。"""
        slot.total_count += 1
        slot.latest_raw = raw
        slot.last_update = time.time()
        old = max(slot.total_count - 1, 1)
        if slot.centroid is not None:
            slot.centroid = (slot.centroid * old + emb) / slot.total_count
        if raw not in slot.top_examples:
            slot.top_examples.append(raw)
        slot.top_examples = sorted(set(slot.top_examples), key=lambda t: -len(t))[:3]
        max_ke = self.settings.cluster.max_member_embeddings
        slot._add_member_emb(emb, max_ke)
        if canonical:
            slot._canonicals.add(canonical)
        # 更新关键词签名
        for kw in extract_keywords(raw, self.settings.cluster.keyword_topk):
            slot.keyword_freq[kw] = slot.keyword_freq.get(kw, 0) + 1
        topk = self.settings.cluster.keyword_topk
        slot.top_keywords = [w for w, _ in Counter(slot.keyword_freq).most_common(topk)]

    def new_slot(self, canonical: str, emb: np.ndarray, raw: str) -> tuple[int, str]:
        """新建常规槽位。"""
        sid = self._alloc_slot()
        cid = f"c{sid}"
        kws = extract_keywords(raw, self.settings.cluster.keyword_topk)
        kw_freq = {kw: 1 for kw in kws}
        slot = SlotState(slot_id=sid, cluster_id=cid, canonical_text=canonical,
                         total_count=1, top_examples=[raw], latest_raw=raw,
                         last_update=time.time(), centroid=emb.copy(),
                         _canonicals={canonical},
                         keyword_freq=kw_freq, top_keywords=kws)
        max_ke = self.settings.cluster.max_member_embeddings
        slot._add_member_emb(emb, max_ke)
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

    def promote(self, slot: SlotState, embedder):
        """晋升到热点语义区: 重算 centroid + 选摘要 → 迁移。"""
        canonical = slot.canonical_text
        if canonical not in self.slots:
            return
        self._recompute_centroid(slot, embedder)
        del self.slots[canonical]
        slot.slot_id = self._next_perm_id
        slot.cluster_id = f"p{self._next_perm_id}"
        self._next_perm_id += 1
        self.permanent[canonical] = slot

    def _recompute_centroid(self, slot: SlotState, embedder):
        """重算 centroid + 选最靠近中心的 top_example 为摘要。"""
        if len(slot._member_embeddings) < 2:
            return
        new_c = np.mean(slot._member_embeddings, axis=0)
        slot.centroid = new_c / (np.linalg.norm(new_c) + 1e-8)
        best_text, best_sim = slot.canonical_text, -1.0
        for t in slot.top_examples:
            te = embedder._encode_sync([t])[0]
            sim = float(np.dot(te, slot.centroid))  # type: ignore[arg-type]
            if sim > best_sim:
                best_sim, best_text = sim, t
        if best_sim > 0:
            slot.canonical_text = best_text

    # ── 周期维护 ──

    def maintenance(self, total_ingested: int, cache=None):
        """每 maintenance_interval 条触发: 合并+拆分+TTL淘汰。"""
        conf = self.settings.cluster
        if total_ingested % conf.maintenance_interval != 0:
            return
        self._merge_similar(self.slots, conf.merge_threshold)
        self._split_degraded(self.slots, conf.split_variance_threshold, conf.max_slots, cache)
        self._merge_similar(self.permanent, conf.permanent_merge_threshold, check_sentiment=True)
        self._split_degraded(self.permanent, conf.permanent_split_variance_threshold, 9999, cache)
        now = time.time()
        ttl = self.settings.cluster.permanent_ttl_seconds
        expired = [c for c, s in self.permanent.items() if now - s.last_update > ttl]
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
        max_ke = self.settings.cluster.max_member_embeddings
        for emb in source._member_embeddings:
            target._add_member_emb(emb, max_ke)
        target._canonicals.update(source._canonicals)
        # 合并关键词签名
        for kw, cnt in source.keyword_freq.items():
            target.keyword_freq[kw] = target.keyword_freq.get(kw, 0) + cnt
        topk = self.settings.cluster.keyword_topk
        target.top_keywords = [w for w, _ in Counter(target.keyword_freq).most_common(topk)]

    def _split_degraded(self, d: OrderedDict, threshold: float, max_slots: int, cache=None):
        to_split: list[tuple[str, SlotState]] = []
        for canonical, slot in d.items():
            if len(slot._member_embeddings) < 4: continue
            if slot.internal_similarity() < threshold:
                to_split.append((canonical, slot))
        for canonical, slot in to_split:
            if canonical not in d: continue
            embs = slot._member_embeddings
            if len(embs) < 4: continue
            labels = _kmeans_2(embs)
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
                new_id = max(s.slot_id for s in d.values()) + 1
            if cache:
                cache.mark_invalid(slot.cluster_id)
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

    # ── 序列化 ──

    def serialize_slot(self, s: SlotState, registry) -> dict:
        all_members: list[dict] = []
        for c in s._canonicals:
            all_members.extend(registry.get_canonical_members(c, limit=5))
        return {"cluster_id": s.cluster_id, "slot_id": s.slot_id,
                "canonical_text": s.canonical_text, "total_count": s.total_count,
                "type": "semantic", "top_examples": s.top_examples[:5],
                "top_keywords": s.top_keywords,
                "members": [{"text": m["raw"], "count": 1} for m in all_members[:8]],
                "latest_raw": s.latest_raw}

    def get_clusters(self, registry) -> list[dict]:
        return [self.serialize_slot(s, registry) for s in
                sorted(self.slots.values(), key=lambda s: -s.total_count)]

    def get_permanent(self, registry) -> list[dict]:
        return [self.serialize_slot(s, registry) for s in
                sorted(self.permanent.values(), key=lambda s: -s.total_count)]

    def render(self) -> str:
        ct = self.settings.cluster.centroid_threshold
        at = self.settings.cluster.anchor_threshold
        lines = ["═" * 55]
        lines.append(f" 聚类 | centroid={ct:.2f} anchor={at:.2f} | 热点:{len(self.permanent)}")
        if self.permanent:
            lines.append("  🏛️ 热点:")
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
        return "\n".join(lines)


def _kmeans_2(embs: list[np.ndarray]) -> list[int]:
    """极简 K-Means (k=2)。"""
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
