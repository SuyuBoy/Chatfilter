"""
Step ② jieba 分词 + 别名替换 — variants.yaml 联动。

职责:
  - 用 jieba 分词替代子串扫描，只替换完整词
  - variants.yaml 的 variant + canonical 词全部加入 jieba 词典
  - suggest_freq 调权确保变体词优先被识别为完整词
  - 防子串误匹配: "南亭" 在 "东南亭子" 中不会被替换

设计演进:
  v1: 子串扫描 text.replace(alias, canonical) — "东南亭子" 被误杀
  v2: jieba 分词 + 词级替换 — 只在完整词边界上做匹配
"""

import yaml
import jieba


class AliasNormalizer:
    """别名归一化器 — 词级替换。

    工作流:
      1. 加载 variants.yaml → 构建 variant→canonical 映射
      2. 将 variant + canonical 词全部加入 jieba 词典并调频
      3. normalize() 时: 分词 → 逐词查映射 → 替换 → 无空格拼接
    """

    def __init__(self, variants_path: str):
        with open(variants_path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        raw: dict[str, list[str]] = data.get("variants", {})

        # variant → canonical 映射
        self._mapping: dict[str, str] = {}
        for canonical, variant_list in raw.items():
            for v in variant_list:
                self._mapping[v] = canonical

        # 联动 jieba: variant + canonical 词都加入词典并调频
        # 确保如 "灰泽满酱" 被识别为一个完整词而不是 ["灰泽", "满酱"]
        for canonical in raw:
            jieba.add_word(canonical)
            jieba.suggest_freq(canonical, tune=True)
        for variant in self._mapping:
            jieba.add_word(variant)
            jieba.suggest_freq(variant, tune=True)

    def normalize(self, text: str) -> str:
        """分词 → 逐词替换 → 拼接。不匹配的词原样保留。"""
        words = list(jieba.cut(text))
        result = []
        for w in words:
            result.append(self._mapping.get(w, w))  # 命中则替换, 否则保留原词
        return "".join(result)  # 无空格拼接, 保持中文连贯
