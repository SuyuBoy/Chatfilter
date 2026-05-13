"""
Step ③ 变体归一化 — 谐音字典 + 拼音自动反向映射。

职责:
  Layer 1: variants.yaml 谐音字典精确匹配 (含拉丁写法如 niubi/yyds)
  Layer 2: 从规范词自身拼音自动生成带声调反向映射 (nan2ting1 → 难听)
           声调天然防误匹配 (lai2le ≠ lai4le)
           冲突处理: 多个 canonical 拼音相同 → 都不映射

为什么不用 SimHash: 音近字替换 (如 "煞笔"→"傻逼") 汉明距离 ~40，
                  SimHash 完全无效，字典 + 拼音才是正解。
"""

import yaml

try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None   # type: ignore[assignment]
    Style = None    # type: ignore[assignment]


class VariantNormalizer:
    """变体归一化器。

    Attributes:
        _variant_map: 变体→规范词 字典 (从 variants.yaml 加载)
        _pinyin_map:  拼音→规范词 反向映射 (自动生成, 声调区分)
        _max_pinyin_len: 拼音匹配长度上限 (短文本更安全)
    """

    def __init__(self, variants_path: str):
        with open(variants_path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        raw: dict[str, list[str]] = data.get("variants", {})

        # 构建 变体→规范词 字典
        self._variant_map: dict[str, str] = {}
        for canonical, variant_list in raw.items():
            for v in variant_list:
                self._variant_map[v] = canonical

        # 从规范词自身拼音自动生成反向映射
        # nan2ting1 → 难听, niu2bi1 → 牛逼, ...
        self._pinyin_map: dict[str, str | None] = {}
        if pinyin is not None:
            for canonical in raw.keys():
                # 仅处理 2-4 字的纯中文规范词
                if 2 <= len(canonical) <= 4 and all('一' <= ch <= '鿿' for ch in canonical):
                    pk = "".join([item[0] for item in pinyin(canonical, style=Style.TONE3)])  # type: ignore[union-attr]
                    if pk not in self._pinyin_map:
                        self._pinyin_map[pk] = canonical
                    else:
                        # 拼音冲突: 两个不同的 canonical 拼音完全相同 → 都标记为 None
                        # 后续过滤掉，避免误匹配
                        self._pinyin_map[pk] = None

        # 清理冲突项 (值为 None 的条目)
        self._pinyin_map = {k: v for k, v in self._pinyin_map.items() if v is not None}  # type: ignore[dict-item]

        self._max_pinyin_len: int = 4  # 最多 4 字走拼音匹配

    def normalize(self, text: str) -> str:
        """归一化入口。"""
        # Layer 1: 谐音字典精确匹配 (含 niubi/yyds 等拉丁写法)
        if text in self._variant_map:
            return self._variant_map[text]

        # Layer 2: 自动拼音反向映射 — 纯中文短文本
        if pinyin is not None and self._pinyin_map and 2 <= len(text) <= self._max_pinyin_len:
            if all('一' <= ch <= '鿿' for ch in text):
                pk = "".join([item[0] for item in pinyin(text, style=Style.TONE3)])  # type: ignore[union-attr]
                candidate = self._pinyin_map.get(pk)
                if candidate is not None and candidate != text:
                    return candidate

        return text
