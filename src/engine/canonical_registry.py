"""
注册表模式: CanonicalRegistry — 三层归一引擎 + 单表缓存 + 计时。

管线: ① 清洗 → ② 归一化 (jieba+变体+拼音) → ③ 循环压缩 → ④ SimHash → ⑤ 去重
"""

import time
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
    stage_times: dict[str, float] | None = None
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
        emb_cfg = settings.embedding
        self.cache = UnifiedCache(max_size=emb_cfg.cache_max_size, ttl=emb_cfg.cache_ttl)
        self.timer = PipelineTimer()
        self.total_ingested = 0
        self.layer1_hits = 0
        self.layer3_new = 0

    def _trusted_canonicals(self) -> set[str]:
        """返回当前可信 canonical 集合 (dedup 中出现 ≥3 次)。"""
        return {c for c in self.dedup_store if self.dedup_store.get_count(c) >= 3}

    def _skip_to_dedup(self, canonical: str, raw_text: str, msg_id: str,
                       stage_times: dict, cache_hits: list,
                       cached_emb: np.ndarray | None) -> RegisterResult:
        t0 = time.perf_counter()
        is_new = self.dedup_store.add(canonical, count=1, raw_text=raw_text, msg_id=msg_id)
        stage_times["⑤_dedup"] = (time.perf_counter() - t0) * 1000
        for name, ms in stage_times.items():
            self.timer.record(name, ms)
        layer = 1 if not is_new else 3
        if not is_new:
            self.layer1_hits += 1
        else:
            self.layer3_new += 1
        return RegisterResult(canonical_id=canonical, canonical_text=canonical,
                              msg_id=msg_id, raw_text=raw_text,
                              is_new=not is_new, layer=layer,
                              stage_times=stage_times, cache_hits=cache_hits,
                              cached_embedding=cached_emb)

    def register(self, raw_text: str, msg_id: str = "") -> RegisterResult:
        self.total_ingested += 1
        stage_times: dict[str, float] = {}

        # ── ① 清洗 ──
        t0 = time.perf_counter()
        cleaned = basic_cleanse(raw_text, min_len=1, max_len=128)
        stage_times["①_cleanse"] = (time.perf_counter() - t0) * 1000
        if cleaned is None:
            for name, ms in stage_times.items():
                self.timer.record(name, ms)
            return RegisterResult(msg_id=msg_id, raw_text=raw_text, filtered=True,
                                  stage_times=stage_times, cache_hits=[])

        entry = self.cache.get(cleaned)
        if entry is not None:
            self.timer.record("①_cleanse", stage_times["①_cleanse"])
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["cleanse"], entry.embedding)

        # ── ② 归一化 (alias + variant + pinyin) ──
        t0 = time.perf_counter()
        trusted = self._trusted_canonicals()
        normalized = self.normalizer.normalize(cleaned, trusted)
        stage_times["②_normalize"] = (time.perf_counter() - t0) * 1000

        entry = self.cache.get(normalized)
        if entry is not None:
            self.cache.put(cleaned, entry, canonical=entry.canonical)
            for name, ms in stage_times.items():
                self.timer.record(name, ms)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["normalize"], entry.embedding)

        # ── ③ 循环压缩 ──
        t0 = time.perf_counter()
        text = compress_cycle(normalized)
        stage_times["③_cycle"] = (time.perf_counter() - t0) * 1000

        entry = self.cache.get(text)
        if entry is not None:
            self.cache.put(cleaned, entry, canonical=entry.canonical)
            self.cache.put(normalized, entry, canonical=entry.canonical)
            for name, ms in stage_times.items():
                self.timer.record(name, ms)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["cycle"], entry.embedding)

        # ── ④ SimHash ──
        t0 = time.perf_counter()
        self.simhash.add(text)
        canonical_text, is_auto = self.simhash.find_canonical(text)
        if is_auto and canonical_text:
            text = canonical_text
        stage_times["④_simhash"] = (time.perf_counter() - t0) * 1000

        # ── 写入缓存 ──
        final = text
        for t in [cleaned, normalized, final]:
            self.cache.put(t, CacheEntry(canonical=final, embedding=None), canonical=final)

        # ── ⑤ 去重 ──
        t0 = time.perf_counter()
        is_new = self.dedup_store.add(final, count=1, raw_text=raw_text, msg_id=msg_id)
        stage_times["⑤_dedup"] = (time.perf_counter() - t0) * 1000
        for name, ms in stage_times.items():
            self.timer.record(name, ms)

        if not is_new:
            self.layer1_hits += 1
            return RegisterResult(canonical_id=final, canonical_text=final,
                                  msg_id=msg_id, raw_text=raw_text,
                                  is_new=False, layer=1,
                                  stage_times=stage_times, cache_hits=[])
        self.layer3_new += 1
        return RegisterResult(canonical_id=final, canonical_text=final,
                              msg_id=msg_id, raw_text=raw_text,
                              is_new=True, layer=3,
                              stage_times=stage_times, cache_hits=[])

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
