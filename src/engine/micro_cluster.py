"""
在线聚类核心: 微簇 (MicroCluster) — 双阈值 + 锚点反漂移 + 长度惩罚 + 关键词双通道。

聚类流程:
  1. 新消息 embedding 到达
  2. 遍历所有 slot 的 centroid, 调用 can_join() 检查加入条件
  3. 条件:
     a) 与 centroid 余弦相似度 > centroid_threshold
     b) 与至少一个锚点余弦相似度 > anchor_threshold (防漂移)
     c) 长度差异惩罚 — 短情绪 ≠ 长评价
     d) 关键词 Jaccard 相似度 — 嵌入 + 文本双通道打分 (新增, alpha 控制权重)
  4. 选相似度最高的 slot 加入; 都不满足则新建 slot

设计原理:
  Leader-Follower 的经典问题: A→B→C→D 逐级相似, 最终 "鼓励" 漂成 "欢乐"。
  锚点机制: 新消息不仅要比 centroid 近, 还必须有至少一个锚点确认语义一致性。
  关键词双通道: BGE 嵌入对短文本信噪比低, 关键词 Jaccard 作为第二路信号纠正。
"""

from dataclasses import dataclass, field

import numpy as np

from src.engine.keyword_util import jaccard


def _length_penalty(new_len: int, anchor_avg_len: float) -> float:
    """长度差异惩罚因子。

    ratio≤1.5 → 1.00 (无惩罚)
    ratio≤3.0 → 0.90 (轻微)
    ratio>3.0 → 0.80 (显著)
    """
    ratio = max(new_len, anchor_avg_len) / max(min(new_len, anchor_avg_len), 1)
    if ratio <= 1.5:
        return 1.0
    elif ratio <= 3.0:
        return 0.90
    else:
        return 0.80


@dataclass
class MicroCluster:
    """微簇 — 最小的聚类单元。

    Attributes:
        centroid:          归一化均值向量 (增量更新)
        anchor_examples:   锚点文本列表 (top_examples 中最具代表性的)
        anchor_embeddings: 锚点 embedding (固定参照, 不随时间漂移)
        top_keywords:      簇签名关键词 (用于双通道相似度)
        keyword_weight:    关键词通道权重 (alpha=1.0 关闭关键词, 0.0 纯关键词)
    """

    centroid: np.ndarray | None = None
    anchor_examples: list[str] = field(default_factory=list)
    anchor_embeddings: list[np.ndarray] = field(default_factory=list)
    top_keywords: list[str] = field(default_factory=list)
    keyword_weight: float = 0.7   # alpha: 嵌入通道权重

    @property
    def _anchor_avg_len(self) -> float:
        if not self.anchor_examples:
            return 1.0
        return sum(len(t) for t in self.anchor_examples) / len(self.anchor_examples)

    def can_join(self, embedding: np.ndarray, new_text_len: int = 0,
                 new_keywords: list[str] | None = None,
                 centroid_threshold: float = 0.78,
                 anchor_threshold: float = 0.82) -> bool:
        """判断新消息是否可以加入此微簇。

        Args:
            embedding:          新消息的归一化 embedding
            new_text_len:       新消息文本长度 (用于长度惩罚)
            new_keywords:       新消息的关键词列表 (用于双通道打分)
            centroid_threshold: 与 centroid 的最低相似度
            anchor_threshold:   与锚点的最低相似度

        Returns:
            True 如果可以加入
        """
        if self.centroid is None:
            return True

        # 条件 1: embedding 余弦相似度
        sim_centroid = float(np.dot(embedding, self.centroid))

        # 条件 1b: 关键词双通道 — 嵌入 + 文本混合打分
        if self.top_keywords and new_keywords:
            kw_sim = jaccard(new_keywords, self.top_keywords)
            alpha = self.keyword_weight
            sim_combined = alpha * sim_centroid + (1 - alpha) * kw_sim
        else:
            sim_combined = sim_centroid

        if sim_combined < centroid_threshold:
            return False

        # 条件 2: 与至少一个锚点足够近 (防止语义漂移)
        if self.anchor_embeddings:
            max_anchor_sim = max(float(np.dot(embedding, a)) for a in self.anchor_embeddings)
            if max_anchor_sim < anchor_threshold:
                return False

        # 条件 3: 长度差异惩罚 (短前缀 ≠ 长评价)
        if new_text_len > 0 and self.anchor_examples:
            penalty = _length_penalty(new_text_len, self._anchor_avg_len)
            if sim_centroid * penalty < centroid_threshold:
                return False

        return True
