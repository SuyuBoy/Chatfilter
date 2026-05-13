"""
注册表模式: CanonicalRegistry — 三层归一引擎 + 单表缓存 + 计时。

管线: ① 清洗 → ② 归一化 (jieba+变体+拼音) → ③ 循环压缩 → ④ SimHash → ⑤ 去重
"""

from dataclasses import dataclass

import numpy as np

from src.engine.preprocessor import basic_cleanse
from src.engine.normalizer import Normalizer
from src.engine.cycle_compressor import compress_cycle
from src.engine.simhash_dedup import SimHashHelper
from src.engine.dedup_store import DedupStore
from src.engine.pipeline_cache import UnifiedCache, CacheEntry, PipelineTimer
from config.settings import Settings


@dataclass
class RegisterResult:
    """注册表返回结果。"""
    canonical_id: str = ""
    canonical_text: str = ""
    msg_id: str = ""
    raw_text: str = ""
    is_new: bool = False
    layer: int = 0
    filtered: bool = False
    cache_hits: list[str] | None = None
    cached_embedding: np.ndarray | None = None


class CanonicalRegistry:
    """三层归一引擎 + 单表缓存。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.normalizer = Normalizer(settings.preprocess.variants_path)
        self.simhash = SimHashHelper(
            high_conf_distance=settings.preprocess.simhash_high_conf_distance,
            candidate_distance=settings.preprocess.simhash_candidate_distance,
            min_text_length=settings.preprocess.simhash_min_text_length,
        )
        self.dedup_store = DedupStore()
        self._trusted: set[str] = set()  # 增量维护, count≥3 的 canonical
        emb_cfg = settings.embedding
        self.cache = UnifiedCache(max_size=emb_cfg.cache_max_size, ttl=emb_cfg.cache_ttl)
        self.timer = PipelineTimer()
        self.total_ingested = 0
        self.layer1_hits = 0
        self.layer3_new = 0

    def _skip_to_dedup(self, canonical: str, raw_text: str, msg_id: str,
                       cache_hits: list,
                       cached_emb: np.ndarray | None) -> RegisterResult:
        with self.timer.stage("⑤_dedup"):
            is_new = self.dedup_store.add(canonical, count=1, raw_text=raw_text, msg_id=msg_id)
            if self.dedup_store.get_count(canonical) >= 3:
                self._trusted.add(canonical)
        layer = 1 if not is_new else 3
        if not is_new: self.layer1_hits += 1
        else: self.layer3_new += 1
        return RegisterResult(canonical_id=canonical, canonical_text=canonical,
                              msg_id=msg_id, raw_text=raw_text,
                              is_new=not is_new, layer=layer,
                              cache_hits=cache_hits, cached_embedding=cached_emb)

    def register(self, raw_text: str, msg_id: str = "") -> RegisterResult:
        self.total_ingested += 1
        T = self.timer  # 上下文管理器: enabled=False 时零开销

        # ── ① 清洗 ──
        with T.stage("①_cleanse"):
            cleaned = basic_cleanse(raw_text, min_len=1, max_len=128)
        if cleaned is None:
            return RegisterResult(msg_id=msg_id, raw_text=raw_text, filtered=True,
                                  cache_hits=[])

        entry = self.cache.get(cleaned)
        if entry is not None:
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       ["cleanse"], entry.embedding)

        # ── ② 归一化 ──
        with T.stage("②_normalize"):
            normalized = self.normalizer.normalize(cleaned, self._trusted)

        entry = self.cache.get(normalized)
        if entry is not None:
            self.cache.put(cleaned, entry, canonical=entry.canonical)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       ["normalize"], entry.embedding)

        # ── ③ 循环压缩 ──
        with T.stage("③_cycle"):
            text = compress_cycle(normalized)

        entry = self.cache.get(text)
        if entry is not None:
            self.cache.put(cleaned, entry, canonical=entry.canonical)
            self.cache.put(normalized, entry, canonical=entry.canonical)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       ["cycle"], entry.embedding)

        # ── ④ SimHash ──
        with T.stage("④_simhash"):
            self.simhash.add(text)
            canonical_text, is_auto = self.simhash.find_canonical(text)
            if is_auto and canonical_text:
                text = canonical_text

        # 写入缓存
        final = text
        for t in [cleaned, normalized, final]:
            self.cache.put(t, CacheEntry(canonical=final, embedding=None), canonical=final)

        # ── ⑤ 去重 ──
        with T.stage("⑤_dedup"):
            is_new = self.dedup_store.add(final, count=1, raw_text=raw_text, msg_id=msg_id)
            # 增量维护可信集
            if self.dedup_store.get_count(final) >= 3:
                self._trusted.add(final)

        if not is_new:
            self.layer1_hits += 1
            return RegisterResult(canonical_id=final, canonical_text=final,
                                  msg_id=msg_id, raw_text=raw_text,
                                  is_new=False, layer=1, cache_hits=[])
        self.layer3_new += 1
        return RegisterResult(canonical_id=final, canonical_text=final,
                              msg_id=msg_id, raw_text=raw_text,
                              is_new=True, layer=3, cache_hits=[])

    def get_canonical_count(self, canonical_text: str) -> int:
        return self.dedup_store.get_count(canonical_text)

    def get_canonical_members(self, canonical_text: str, limit: int = 0) -> list[dict]:
        return self.dedup_store.get_members(canonical_text, limit)

    @property
    def unique_count(self) -> int:
        return len(self.dedup_store)

    @property
    def dedup_rate(self) -> float:
        if self.total_ingested == 0:
            return 0.0
        return 1.0 - (self.unique_count / self.total_ingested)
