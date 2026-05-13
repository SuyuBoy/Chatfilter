"""
Step ⑤ SimHash 辅助模糊 — 降级为辅助角色。

职责:
  - 变体字典 + 拼音已覆盖 80%+ 弹幕变体
  - SimHash 仅处理字典遗漏的长文本近重复
  - ≥8 字 + 汉明距离 ≤2 → 高置信度自动合并
  - 短文本 (<8 字) 仅写候选日志, 不自动合并

原理:
  SimHash 将文本映射为 64 位指纹, 相似文本的汉明距离小。
  但对音近字无效 — "煞笔" vs "傻逼" 汉明距离 ~40。
"""

import logging

logger = logging.getLogger(__name__)


def _ngrams(text: str, n: int = 3) -> list[str]:
    """生成字符 n-gram。短文本自动降级: 1 字 → 自身, 2 字 → bigram。"""
    if len(text) < n:
        if len(text) < 2:
            return [text]                    # 单字: 自身为唯一 token
        return [text[i : i + 2] for i in range(len(text) - 1)]  # bigram
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _hash_token(token: str) -> int:
    """字符串 → 64 位无符号整数哈希。"""
    return hash(token) & ((1 << 64) - 1)


def compute_simhash(text: str, bits: int = 64, ngram: int = 3) -> int:
    """计算文本的 SimHash 指纹。

    算法: 每个 n-gram 哈希的每一位投票 (+1/-1), 投票结果 >0 则该位为 1。
    """
    tokens = _ngrams(text, ngram)
    if not tokens:
        return 0
    vec = [0] * bits
    for token in tokens:
        h = _hash_token(token)
        for i in range(bits):
            if (h >> i) & 1:
                vec[i] += 1
            else:
                vec[i] -= 1
    fingerprint = 0
    for i in range(bits):
        if vec[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """两个 64 位整数的汉明距离 (不同 bit 数)。"""
    return (a ^ b).bit_count()


class SimHashHelper:
    """SimHash 辅助器 — 降级角色。

    Attributes:
        high_conf_distance: 高置信度汉明距离上限 (≤此值自动合并)
        candidate_distance:  候选汉明距离上限 (仅写日志)
        min_text_length:     短于该长度不触发自动合并
    """

    def __init__(
        self,
        high_conf_distance: int = 2,
        candidate_distance: int = 3,
        min_text_length: int = 8,
    ):
        self.high_conf_distance = high_conf_distance
        self.candidate_distance = candidate_distance
        self.min_text_length = min_text_length
        # canonical_text → (fingerprint, frequency)
        self._store: dict[str, tuple[int, int]] = {}

    def add(self, text: str) -> None:
        """将文本及其 SimHash 指纹加入存储。"""
        fp = compute_simhash(text)
        if text in self._store:
            _, freq = self._store[text]
            self._store[text] = (fp, freq + 1)
        else:
            self._store[text] = (fp, 1)

    def find_canonical(self, text: str) -> tuple[str | None, bool]:
        """查找与给定文本最相似的 canonical 文本。

        Returns:
            (canonical_text, is_auto_merged)
            - canonical_text: None 表示无匹配
            - is_auto_merged: True 表示高置信度自动合并
        """
        fp = compute_simhash(text)
        best = None
        best_dist = 999

        # 遍历所有已存储文本，找最小汉明距离
        for stored_text, (stored_fp, _) in self._store.items():
            dist = hamming_distance(fp, stored_fp)
            if dist < best_dist:
                best_dist = dist
                best = stored_text

        if best is None:
            return None, False

        # 高置信度: ≥8 字且汉明距离 ≤2 → 自动合并
        if best_dist <= self.high_conf_distance:
            if len(text) >= self.min_text_length and len(best) >= self.min_text_length:
                freq = self._store[best][1]
                if freq < self._store.get(text, (0, 1))[1]:
                    # 新文本频次更高 → 替换 canonical
                    self._store[text] = (fp, freq + 1)
                    del self._store[best]
                    return text, True
                return best, True

        # 候选: 汉明距离 ≤3 但 <8 字 → 仅日志, 不自动合并
        if best_dist <= self.candidate_distance:
            logger.info("simhash candidate: %r -> %r (distance=%d)", text, best, best_dist)
            return best, False

        return None, False
