"""
Step ① 基础清洗 — 文本预处理的第一道防线。

职责:
  - 移除控制字符、统一全角半角、NFKC 归一化
  - 折叠空白符、长度裁剪 (1-128 字符)
  - 过滤纯数字 (>4 字) 但保留短数字梗 (666/520/233)
  - 放行纯符号/emoji — 由后续循环节压缩处理

设计原则: 只做确定性清洗，不确定的留给后续步骤。
"""

import re
import unicodedata


def basic_cleanse(text: str, min_len: int = 1, max_len: int = 128) -> str | None:
    """基础清洗入口。返回 None 表示该文本应被丢弃。

    Args:
        text: 原始弹幕文本
        min_len: 最短保留长度 (字符)
        max_len: 最长保留长度 (字符)

    Returns:
        清洗后的文本，或 None (应丢弃)
    """
    # 空值/非字符串直接丢弃
    if not text or not isinstance(text, str):
        return None

    # 移除控制字符 (保留换行/回车/制表符)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")
    # NFKC 归一化: 全角→半角、合字→分字、上标→普通
    text = unicodedata.normalize("NFKC", text)
    # 折叠连续空白符为单个空格，去除首尾空白
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    # 过滤纯数字 (>4 字): "12345" 是噪音，"666" 是梗
    if text.isdigit() and len(text) > 4:
        return None

    # 不在本步过滤纯符号/emoji — "🎶🎤🐛"×3 会由循环节压缩处理

    if len(text) < min_len or len(text) > max_len:
        return None

    return text
