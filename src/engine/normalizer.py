"""
Step ②+③ 统一归一化 — jieba 分词 + 词级匹配 + 拼音候选。

合并原来的 AliasNormalizer 和 VariantNormalizer:
  - 统一用 jieba 分词 + 相邻合并 (词边界保护)
  - 同时查 alias 映射 (灰泽满酱→主播) 和 variant 映射 (牛批→牛逼)
  - 自动声调拼音反向映射, 用 dedup_store 做可信验证

拼音可信: 候选 canonical 必须在 dedup_store 中出现过 (≥3 次),
否则视为生造词, 拒绝替换。
"""

from functools import lru_cache

import yaml
import jieba

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None
    Style = None


@lru_cache(maxsize=2048)
def _cached_pinyin(text: str) -> str:
    """带声调拼音, 缓存 2048 条。"""
    if pinyin is None:
        return ""
    return "".join([item[0] for item in pinyin(text, style=Style.TONE3)])  # type: ignore[arg-type]


class Normalizer:
    """统一归一化器 — Alias + Variant + Pinyin 三合一。

    _word_map: variant_text → canonical (所有 variant 词, 用于词级匹配)
    _pinyin_map: pinyin → canonical (自动生成, 声调区分)
    """

    def __init__(self, variants_path: str):
        with open(variants_path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        raw: dict[str, list[str]] = data.get("variants", {})

        # 构建统一的 variant → canonical 字典
        self._word_map: dict[str, str] = {}
        for canonical, variant_list in raw.items():
            for v in variant_list:
                self._word_map[v] = canonical

        # 从规范词自动生成声调拼音反向映射
        # 冲突处理: 两个 canonical 拼音相同 → 都不映射
        self._pinyin_map: dict[str, str] = {}
        if pinyin is not None:
            for canonical in raw:
                if 2 <= len(canonical) <= 4 and all('一' <= ch <= '鿿' for ch in canonical):
                    pk = _cached_pinyin(canonical)
                    if pk not in self._pinyin_map:
                        self._pinyin_map[pk] = canonical
                    else:
                        self._pinyin_map[pk] = ""  # 冲突标记

        # 清理冲突项
        self._pinyin_map = {k: v for k, v in self._pinyin_map.items() if v}

    def normalize(self, text: str, trusted_canonicals: set[str] | None = None) -> str:
        """归一化入口。

        Args:
            text: 待归一化文本
            trusted_canonicals: 可信 canonical 集合 (拼音候选验证用)

        流程:
          1. jieba 分词 + 相邻合并 → 查 word_map
          2. 未匹配词 → 拼音匹配 → 可信验证 → 替换或保留
        """
        words = list(jieba.cut(text))
        result = []
        i = 0
        while i < len(words):
            # 相邻合并: 最长 4 词, 精确匹配
            matched = False
            for j in range(min(i + 4, len(words)), i, -1):
                chunk = "".join(words[i:j])
                if chunk in self._word_map:
                    result.append(self._word_map[chunk])
                    i = j
                    matched = True
                    break
            if matched:
                continue
            # 单词匹配
            w = words[i]
            if w in self._word_map:
                result.append(self._word_map[w])
            elif pinyin is not None and 2 <= len(w) <= 4 and all('一' <= ch <= '鿿' for ch in w):
                # 拼音候选
                pk = _cached_pinyin(w)
                candidate = self._pinyin_map.get(pk)
                if candidate and candidate != w:
                    # 可信验证: 候选 canonical 是否在可信集中
                    if trusted_canonicals is None or candidate in trusted_canonicals:
                        result.append(candidate)
                    else:
                        result.append(w)
                else:
                    result.append(w)
            else:
                result.append(w)
            i += 1
        return "".join(result)
