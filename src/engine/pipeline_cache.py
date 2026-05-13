"""
单表缓存 + 逐环节计时。

UnifiedCache: text_hash → (embedding, cluster_id, cluster_valid)
  反向索引1: canonical → {text_hash}   (embedding 回填)
  反向索引2: cluster_id → {text_hash}  (退化标记)

PipelineTimer: 逐环节耗时统计。
"""

import time
import hashlib
from dataclasses import dataclass, field

import numpy as np


def _hash_key(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) & ((1 << 64) - 1)


@dataclass
class CacheEntry:
    """单条缓存记录。"""
    canonical: str = ""                     # 规范文本 (去重的 key)
    embedding: np.ndarray | None = None     # None=尚未计算
    cluster_id: str | None = None           # 聚类归属
    cluster_valid: bool = True              # False=簇已退化, 需重新聚类


class UnifiedCache:
    """单表缓存 — text_hash → CacheEntry。

    索引:
      _canonical_index: canonical → set[text_hash]  (embedding 回填)
      _cluster_index:   cluster_id → set[text_hash] (退化标记)
    """

    def __init__(self, max_size: int = 50000, ttl: float = 600.0):
        self._table: dict[int, tuple[CacheEntry, float]] = {}  # hash → (entry, last_access)
        self._canonical_index: dict[str, set[int]] = {}         # canonical → {hash, ...}
        self._cluster_index: dict[str, set[int]] = {}           # cluster_id → {hash, ...}
        self._max_size = max_size
        self._ttl = ttl
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> CacheEntry | None:
        """查表。命中返回 CacheEntry, 未命中返回 None。"""
        h = _hash_key(text)
        pair = self._table.get(h)
        if pair is not None and (time.time() - pair[1] < self._ttl):
            self.hits += 1
            self._table[h] = (pair[0], time.time())  # 更新访问时间
            return pair[0]
        self.misses += 1
        return None

    def put(self, text: str, entry: CacheEntry, canonical: str = "",
            cluster_id: str = "") -> None:
        """写入。维护两个反向索引。"""
        h = _hash_key(text)
        self._evict_if_needed()
        self._table[h] = (entry, time.time())
        if canonical:
            self._canonical_index.setdefault(canonical, set()).add(h)
        if cluster_id:
            self._cluster_index.setdefault(cluster_id, set()).add(h)

    def backfill(self, canonical: str, embedding: np.ndarray) -> None:
        """embedding 计算后回填所有指向该 canonical 的条目。O(1) via 反向索引。"""
        for h in self._canonical_index.get(canonical, set()):
            pair = self._table.get(h)
            if pair is not None:
                pair[0].embedding = embedding.copy()

    def mark_invalid(self, cluster_id: str) -> None:
        """退化标记: 该 cluster_id 下所有条目 cluster_valid=False。O(1) via 反向索引。"""
        for h in self._cluster_index.get(cluster_id, set()):
            pair = self._table.get(h)
            if pair is not None:
                pair[0].cluster_valid = False

    def _evict_if_needed(self) -> None:
        """LRU: 超出 max_size 淘汰最旧 10%, 同时清理两个反向索引。"""
        if len(self._table) >= self._max_size:
            items = sorted(self._table.items(), key=lambda x: x[1][1])
            for (h, _) in items[: max(len(items) // 10, 1)]:
                del self._table[h]
            # 清理死引用 (惰性, 仅在淘汰时做)
            for idx in [self._canonical_index, self._cluster_index]:
                dead = [k for k, v in idx.items() if not v]
                for k in dead:
                    del idx[k]

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t > 0 else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 3)}


@dataclass
class StageTiming:
    name: str
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    _recent: list[float] = field(default_factory=lambda: [0.0] * 100)
    _recent_idx: int = 0

    def record(self, elapsed_ms: float) -> None:
        self.total_ms += elapsed_ms
        self.count += 1
        if elapsed_ms < self.min_ms: self.min_ms = elapsed_ms
        if elapsed_ms > self.max_ms: self.max_ms = elapsed_ms
        self._recent[self._recent_idx % 100] = elapsed_ms
        self._recent_idx += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_ms(self) -> float:
        n = min(self._recent_idx, 100)
        return sum(self._recent[:n]) / n if n > 0 else 0.0


class PipelineTimer:
    def __init__(self):
        self.stages: dict[str, StageTiming] = {}
        self._order: list[str] = []

    def _ensure(self, name: str) -> StageTiming:
        if name not in self.stages:
            self.stages[name] = StageTiming(name=name)
            self._order.append(name)
        return self.stages[name]

    def record(self, name: str, elapsed_ms: float) -> None:
        self._ensure(name).record(elapsed_ms)

    def stats(self) -> dict:
        return {name: {"avg_ms": round(self.stages[name].avg_ms, 3),
                       "recent_avg_ms": round(self.stages[name].recent_avg_ms, 3),
                       "min_ms": round(self.stages[name].min_ms, 3),
                       "max_ms": round(self.stages[name].max_ms, 3),
                       "count": self.stages[name].count}
                for name in self._order}
