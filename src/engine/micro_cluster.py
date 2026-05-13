"""
在线聚类核心: 微簇 (MicroCluster) — 双阈值 + 锚点反漂移 + 长度惩罚。

聚类流程:
  1. 新消息 embedding 到达
  2. 遍历所有 slot 的 centroid, 调用 can_join() 检查加入条件
  3. 条件:
     a) 与 centroid 余弦相似度 > centroid_threshold
     b) 与至少一个锚点余弦相似度 > anchor_threshold (防漂移)
     c) 长度差异惩罚 — 短情绪 ≠ 长评价
  4. 选相似度最高的 slot 加入; 都不满足则新建 slot

设计原理:
  Leader-Follower 的经典问题: A→B→C→D 逐级相似, 最终 "鼓励" 漂成 "欢乐"。
  锚点机制: 新消息不仅要比 centroid 近, 还必须有至少一个锚点确认语义一致性。
  锚点是 slot 创建时的 top_examples — 它们代表 "这个簇最初的语义", 不会随时间漂移。
"""

from dataclasses import dataclass, field

import numpy as np


def _length_penalty(new_len: int, anchor_avg_len: float) -> float:
    """长度差异惩罚因子。

    问题: "帅啊"(3字) 和 "帅啊我看XXX也就那样了"(13字) 共享前缀,
          BGE embedding 会拉近 (sim≈0.76), 但短情绪和长评价不应同簇。

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
    """

    centroid: np.ndarray | None = None
    anchor_examples: list[str] = field(default_factory=list)
    anchor_embeddings: list[np.ndarray] = field(default_factory=list)

    @property
    def _anchor_avg_len(self) -> float:
        """锚点文本的平均长度 (用于长度惩罚计算)。"""
        if not self.anchor_examples:
            return 1.0
        return sum(len(t) for t in self.anchor_examples) / len(self.anchor_examples)

    def can_join(self, embedding: np.ndarray, new_text_len: int = 0,
                 centroid_threshold: float = 0.78,
                 anchor_threshold: float = 0.82) -> bool:
        """判断新消息是否可以加入此微簇。

        Args:
            embedding:          新消息的归一化 embedding
            new_text_len:       新消息文本长度 (用于长度惩罚)
            centroid_threshold: 与 centroid 的最低相似度
            anchor_threshold:   与锚点的最低相似度

        Returns:
            True 如果可以加入
        """
        # 空簇: 无条件接受
        if self.centroid is None:
            return True

        # 条件 1: 与 centroid 足够近
        sim_centroid = float(np.dot(embedding, self.centroid))
        if sim_centroid < centroid_threshold:
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
