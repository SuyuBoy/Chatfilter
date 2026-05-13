"""
注册表模式: CanonicalRegistry — 三层归一引擎 + 统一缓存池 + 计时。

职责:
  将 6 步预处理封装为统一的 register(raw_text, msg_id) → RegisterResult。
  下游只需检查 is_new 来决定是否触发 embedding。

三层归一:
  层1: 精确匹配 (DedupStore O(1) 哈希) — 已有 canonical 直接命中
  层2: 模糊归一 (别名/谐音/循环/SimHash) — 变体归一化到已有 canonical
  层3: 全新文本 — 自身体为 canonical, 触发 embedding

缓存策略:
  每个确定性阶段 (cleanse/alias/variant/cycle) 查统一缓存 (UnifiedCache)。
  命中 → 拿到 canonical + embedding → 短路全部下游。
  全流程跑完 → 写入占位条目 → ingest() 计算 embedding → backfill 回填。
"""

import time
from dataclasses import dataclass

import numpy as np

from src.engine.preprocessor import basic_cleanse
from src.engine.alias_normalizer import AliasNormalizer
from src.engine.variant_normalizer import VariantNormalizer
from src.engine.cycle_compressor import compress_cycle
from src.engine.simhash_dedup import SimHashHelper
from src.engine.dedup_store import DedupStore
from src.engine.pipeline_cache import UnifiedCache, PipelineTimer
from config.settings import Settings


@dataclass
class RegisterResult:
    """注册表返回结果 — 封装归一化全过程的状态。"""
    canonical_id: str = ""              # 代表文本的唯一标识
    canonical_text: str = ""            # 代表文本 (规范后的文本)
    msg_id: str = ""                    # 原始消息 ID
    raw_text: str = ""                  # 原始文本 (内容穿透保留)
    is_new: bool = False                # 是否全新的 canonical (需触发 embedding)
    layer: int = 0                      # 命中层: 1=精确匹配, 3=全新
    filtered: bool = False              # 是否被基础清洗过滤
    stage_times: dict[str, float] | None = None   # 逐环节耗时 (ms)
    cache_hits: list[str] | None = None           # 最早命中缓存的环节
    cached_embedding: np.ndarray | None = None    # 统一缓存命中时带出的 embedding


class CanonicalRegistry:
    """三层归一引擎。

    管线顺序 (固定):
      ① 基础清洗 → ② 别名归一化 → ③ 变体归一化 → ④ 循环压缩
      → ⑤ SimHash 辅助 → ⑥ 精确去重

    每步先查统一缓存, 命中即短路。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # 初始化各步骤处理器
        self.alias_normalizer = AliasNormalizer(settings.preprocess.aliases_path)
        self.variant_normalizer = VariantNormalizer(settings.preprocess.variants_path)
        self.simhash = SimHashHelper(
            high_conf_distance=settings.preprocess.simhash_high_conf_distance,
            candidate_distance=settings.preprocess.simhash_candidate_distance,
            min_text_length=settings.preprocess.simhash_min_text_length,
        )
        self.dedup_store = DedupStore()
        self.cache = UnifiedCache()      # 统一缓存池
        self.timer = PipelineTimer()     # 逐环节计时器

        # 统计计数
        self.total_ingested = 0
        self.layer1_hits = 0            # 层1: 精确匹配命中
        self.layer3_new = 0             # 层3: 全新 canonical

    def _skip_to_dedup(self, canonical: str, raw_text: str, msg_id: str,
                       stage_times: dict, cache_hits: list,
                       cached_emb: np.ndarray | None) -> RegisterResult:
        """缓存命中后的快速通道: 跳过所有中间步骤, 直接进入 dedup。"""
        t0 = time.perf_counter()
        is_new = self.dedup_store.add(canonical, count=1, raw_text=raw_text, msg_id=msg_id)
        stage_times["⑥_dedup"] = (time.perf_counter() - t0) * 1000

        for name, ms in stage_times.items():
            self.timer.record(name, ms)

        layer = 1 if not is_new else 3
        if not is_new:
            self.layer1_hits += 1
        else:
            self.layer3_new += 1

        return RegisterResult(
            canonical_id=canonical, canonical_text=canonical,
            msg_id=msg_id, raw_text=raw_text,
            is_new=not is_new, layer=layer,
            stage_times=stage_times, cache_hits=cache_hits,
            cached_embedding=cached_emb,
        )

    def register(self, raw_text: str, msg_id: str = "") -> RegisterResult:
        """输入原始弹幕 → 输出注册结果。下游用 is_new 判断是否触发 embedding。"""
        self.total_ingested += 1
        stage_times: dict[str, float] = {}

        # ── ① 基础清洗: 查统一缓存 ──
        t0 = time.perf_counter()
        entry = self.cache.get("cleanse", raw_text)
        stage_times["①_cleanse"] = (time.perf_counter() - t0) * 1000
        if entry is not None:
            # 命中: 拿到 canonical + embedding, 跳过后面的清洗/别名/变体/循环/SimHash
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["cleanse"], entry.embedding)

        # 未命中 → 实际清洗
        t0 = time.perf_counter()
        cleaned = basic_cleanse(raw_text, min_len=1, max_len=128)
        stage_times["①_cleanse"] += (time.perf_counter() - t0) * 1000

        # 清洗后为空 → 过滤
        if cleaned is None:
            for name, ms in stage_times.items():
                self.timer.record(name, ms)
            return RegisterResult(msg_id=msg_id, raw_text=raw_text, filtered=True,
                                  stage_times=stage_times, cache_hits=[])

        # ── ② 别名归一化 ──
        t0 = time.perf_counter()
        entry = self.cache.get("alias", cleaned)
        stage_times["②_alias"] = (time.perf_counter() - t0) * 1000
        if entry is not None:
            # 上游未命中但本层命中: 补写上游映射
            self.cache.put("cleanse", raw_text, entry.canonical)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["alias"], entry.embedding)

        t0 = time.perf_counter()
        text = self.alias_normalizer.normalize(cleaned)
        stage_times["②_alias"] += (time.perf_counter() - t0) * 1000

        # ── ③ 变体归一化 ──
        t0 = time.perf_counter()
        entry = self.cache.get("variant", text)
        stage_times["③_variant"] = (time.perf_counter() - t0) * 1000
        if entry is not None:
            self.cache.put("cleanse", raw_text, entry.canonical)
            self.cache.put("alias", cleaned, entry.canonical)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["variant"], entry.embedding)

        t0 = time.perf_counter()
        alias_out = text  # 保存别名输出, 用于缓存写入
        text = self.variant_normalizer.normalize(alias_out)
        stage_times["③_variant"] += (time.perf_counter() - t0) * 1000

        # ── ④ 循环节压缩 ──
        t0 = time.perf_counter()
        entry = self.cache.get("cycle", text)
        stage_times["④_cycle"] = (time.perf_counter() - t0) * 1000
        if entry is not None:
            self.cache.put("cleanse", raw_text, entry.canonical)
            self.cache.put("alias", cleaned, entry.canonical)
            self.cache.put("variant", alias_out, entry.canonical)
            return self._skip_to_dedup(entry.canonical, raw_text, msg_id,
                                       stage_times, ["cycle"], entry.embedding)

        t0 = time.perf_counter()
        variant_out = text  # 保存变体输出, 用于缓存写入
        text = compress_cycle(variant_out)
        stage_times["④_cycle"] += (time.perf_counter() - t0) * 1000

        # ── ⑤ SimHash 辅助 ──
        t0 = time.perf_counter()
        self.simhash.add(text)
        canonical_text, is_auto = self.simhash.find_canonical(text)
        if is_auto and canonical_text:
            text = canonical_text
        stage_times["⑤_simhash"] = (time.perf_counter() - t0) * 1000

        # ── 全流程走完: 写入统一缓存占位条目 (embedding=None, 等 backfill) ──
        final = text
        self.cache.put("cleanse", raw_text, final, None)
        self.cache.put("alias", cleaned, final, None)
        self.cache.put("variant", alias_out, final, None)
        self.cache.put("cycle", variant_out, final, None)

        # ── ⑥ 层1: 精确匹配 ──
        t0 = time.perf_counter()
        is_new = self.dedup_store.add(final, count=1, raw_text=raw_text, msg_id=msg_id)
        stage_times["⑥_dedup"] = (time.perf_counter() - t0) * 1000

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
        """返回该 canonical 的总出现次数。"""
        return self.dedup_store.get_count(canonical_text)

    def get_canonical_members(self, canonical_text: str, limit: int = 0) -> list[dict]:
        """返回该 canonical 下所有原始弹幕成员 (内容穿透数据源)。"""
        return self.dedup_store.get_members(canonical_text, limit)

    @property
    def unique_count(self) -> int:
        """唯一 canonical 文本数量。"""
        return len(self.dedup_store)

    @property
    def dedup_rate(self) -> float:
        """去重率: (总摄入 - 唯一) / 总摄入。"""
        if self.total_ingested == 0:
            return 0.0
        return 1.0 - (self.unique_count / self.total_ingested)
