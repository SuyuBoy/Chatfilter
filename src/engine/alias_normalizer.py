"""
Step ② jieba 分词 + 相邻词合并匹配 — variants.yaml 联动。

设计:
  - jieba 默认分词 (不调频, 不 add_word), 保持词典干净
  - 逐词查 variant_map + 相邻词合并检查 (最多 4 词)
  - 精确字符串比较, 不做子串匹配
  - "东南亭子" → ["东南","亭子"] → "南亭" 不会出现 → 不误匹配
  - "南亭" 被切成 ["南","亭"] → 相邻合并 = "南亭" → 命中
"""

import yaml
import jieba


class AliasNormalizer:
    """别名归一化器 — 默认 jieba 分词 + 相邻合并。

    相邻合并: 弥补 jieba 不认识变体词的短板。如果 2-4 个相邻词拼起来
    恰好命中 variant_map, 就合并替换。不做子串扫描。
    """

    def __init__(self, variants_path: str):
        with open(variants_path, "r", encoding="utf-8") as f:
            data: dict = yaml.safe_load(f) or {}
        raw: dict[str, list[str]] = data.get("variants", {})

        self._mapping: dict[str, str] = {}
        for canonical, variant_list in raw.items():
            for v in variant_list:
                self._mapping[v] = canonical

    def normalize(self, text: str) -> str:
        words = list(jieba.cut(text))
        result = []
        i = 0
        while i < len(words):
            # 尝试相邻合并: 从长到短, 最多 4 词
            matched = False
            for j in range(min(i + 4, len(words)), i, -1):
                chunk = "".join(words[i:j])
                if chunk in self._mapping:
                    result.append(self._mapping[chunk])
                    i = j
                    matched = True
                    break
            if not matched:
                result.append(self._mapping.get(words[i], words[i]))
                i += 1
        return "".join(result)
