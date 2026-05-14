"""
关键词提取 — jieba 分词 + 停用词过滤 + 高频签名
"""

import re
from collections import Counter

import jieba

# 停用词：单字、数字、符号、高频虚词
_STOP_WORDS = {
    '了', '的', '是', '我', '不', '在', '人', '有', '这', '个', '们', '也',
    '就', '都', '一', '个', '上', '他', '你', '她', '它', '说', '看', '来',
    '去', '和', '那', '要', '会', '吗', '吧', '呢', '啊', '哦', '嗯', '哈',
    '呀', '哇', '嘿', '呵', '嘻', '啦', '吧', '么', '没', '被', '让', '把',
    '对', '到', '很', '还', '大', '小', '多', '少', '太', '真', '好', '超',
    '可', '能', '什么', '怎么', '为什么', '啥', '哪', '怎么', '咋',
}


def extract_keywords(text: str, top_k: int = 5) -> list[str]:
    """从文本中提取 top_k 个关键词。

    Args:
        text: 输入文本
        top_k: 保留的关键词数量

    Returns:
        按频率降序的关键词列表
    """
    # 清洗：去符号、保留中文字和字母
    text = re.sub(r'[^一-鿿\w]', ' ', text)
    if not text.strip():
        return []

    words = jieba.lcut(text)
    counter = Counter()
    for w in words:
        w = w.strip().lower()
        if len(w) < 2:                # 过滤单字
            continue
        if w.isdigit():               # 过滤纯数字
            continue
        if w in _STOP_WORDS:          # 过滤虚词/停用词
            continue
        counter[w] += 1

    return [w for w, _ in counter.most_common(top_k)]


def jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard 相似度。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
