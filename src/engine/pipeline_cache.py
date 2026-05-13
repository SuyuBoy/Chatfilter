"""
统一缓存池 + 逐环节时间统计。

UnifiedCache: 替换原来的 PipelineCachePool + EmbeddingCache。
每个确定性环节 (cleanse/alias/variant/cycle) 的输入 → (canonical, embedding)
共用一个 LRU 缓存。任一阶段命中即拿到 canonical + embedding，下游全部短路。

PipelineTimer: 每环节耗时统计 (avg/min/max/recent/count)。
"""

import time
import hashlib
from dataclasses import dataclass, field

import numpy as np


def _hash_key(text: str) -> int:
    """文本 → 64 位整数哈希 (MD5 截断)。"""
    return int(hashlib.md5(text.encode()).hexdigest(), 16) & ((1 << 64) - 1)


@dataclass
class CacheEntry:
    """缓存条目: 规范文本 + 可选的 embedding 向量。"""
    canonical: str
    embedding: np.ndarray | None = None  # None 表示尚未计算 (等 backfill)


class UnifiedCache:
    """统一缓存池 — Pipeline + Embedding 合二为一。

    结构:
      _entries: (stage, text_hash) → (canonical, embedding, last_access)
      _canonical_keys: canonical → set of keys (用于 backfill 回填)

    工作流:
      1. register() 每阶段查 get() — 命中则拿到 CacheEntry
      2. 全流程跑完 put() 写入 embedding=None 的占位条目
      3. ingest() 计算 embedding 后 backfill() 回填所有占位条目
    """

    def __init__(self, max_size: int = 50000, ttl: float = 600.0):
        self._entries: dict[tuple, tuple[str, np.ndarray | None, float]] = {}
        self._max_size = max_size
        self._ttl = ttl                     # 缓存过期时间 (秒)
        self.hits = 0
        self.misses = 0
        self._canonical_keys: dict[str, set[tuple]] = {}  # canonical → 所有指向它的 cache key

    def get(self, stage: str, text: str) -> CacheEntry | None:
        """查询缓存。命中返回 CacheEntry，未命中返回 None。自动更新访问时间。"""
        key = (stage, _hash_key(text))
        entry = self._entries.get(key)
        if entry is not None and (time.time() - entry[2] < self._ttl):
            self.hits += 1
            # 更新访问时间 (LRU 排序依据)
            self._entries[key] = (entry[0], entry[1], time.time())
            return CacheEntry(canonical=entry[0], embedding=entry[1])
        self.misses += 1
        return None

    def put(self, stage: str, text: str, canonical: str, embedding: np.ndarray | None = None) -> None:
        """写入缓存。embedding 为 None 时仅占位，等待 backfill 回填。"""
        key = (stage, _hash_key(text))
        self._evict_if_needed()
        emb_copy = embedding.copy() if embedding is not None else None
        self._entries[key] = (canonical, emb_copy, time.time())
        # 维护反向索引，供 backfill 查找
        if canonical not in self._canonical_keys:
            self._canonical_keys[canonical] = set()
        self._canonical_keys[canonical].add(key)

    def backfill(self, canonical: str, embedding: np.ndarray) -> None:
        """回填 embedding 到所有指向该 canonical 的缓存条目。
        在 ingest() 计算完 embedding 后调用，使得后续命中直接带出 embedding。
        """
        keys = self._canonical_keys.get(canonical, set())
        emb_copy = embedding.copy()
        for key in keys:
            if key in self._entries:
                entry = self._entries[key]
                self._entries[key] = (entry[0], emb_copy, entry[2])

    def _evict_if_needed(self) -> None:
        """LRU 淘汰: 超出 max_size 时清除最旧 10% 条目。"""
        if len(self._entries) >= self._max_size:
            items = sorted(self._entries.items(), key=lambda x: x[1][2])
            for (k, _) in items[: max(len(items) // 10, 1)]:
                canonical = self._entries[k][0]
                if canonical in self._canonical_keys:
                    self._canonical_keys[canonical].discard(k)
                    if not self._canonical_keys[canonical]:
                        del self._canonical_keys[canonical]
                del self._entries[k]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hit_rate, 3)}


@dataclass
class StageTiming:
    """单环节耗时统计。"""
    name: str
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    _recent: list[float] = field(default_factory=lambda: [0.0] * 100)  # 最近 100 次耗时
    _recent_idx: int = 0

    def record(self, elapsed_ms: float) -> None:
        """记录一次耗时。"""
        self.total_ms += elapsed_ms
        self.count += 1
        if elapsed_ms < self.min_ms:
            self.min_ms = elapsed_ms
        if elapsed_ms > self.max_ms:
            self.max_ms = elapsed_ms
        self._recent[self._recent_idx % 100] = elapsed_ms
        self._recent_idx += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    @property
    def recent_avg_ms(self) -> float:
        """最近 100 次的平均耗时 (反映当前性能)。"""
        n = min(self._recent_idx, 100)
        return sum(self._recent[:n]) / n if n > 0 else 0.0


class PipelineTimer:
    """全管线逐环节计时器。"""

    def __init__(self):
        self.stages: dict[str, StageTiming] = {}
        self._order: list[str] = []  # 保持插入顺序

    def _ensure(self, name: str) -> StageTiming:
        if name not in self.stages:
            self.stages[name] = StageTiming(name=name)
            self._order.append(name)
        return self.stages[name]

    def record(self, name: str, elapsed_ms: float) -> None:
        self._ensure(name).record(elapsed_ms)

    def stats(self) -> dict:
        """返回所有环节的统计摘要。"""
        return {
            name: {
                "avg_ms": round(self.stages[name].avg_ms, 3),
                "recent_avg_ms": round(self.stages[name].recent_avg_ms, 3),
                "min_ms": round(self.stages[name].min_ms, 3),
                "max_ms": round(self.stages[name].max_ms, 3),
                "count": self.stages[name].count,
            }
            for name in self._order
        }
