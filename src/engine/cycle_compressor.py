"""
Step ④ 循环节压缩 — 正则反向引用消除弹幕刷屏重复。

职责:
  - 用 (.+?)\1{2,} 匹配 3 次以上的子串重复
  - 容忍尾部残余，不再要求整除
  - LRU 缓存加速重复查询

示例:
  "哈哈哈哈哈哈" → "哈"
  "加油加油加油加" → "加油加" (尾部残余保留)
  "🎶🎤🐛🎶🎤🐛🎶🎤🐛" → "🎶🎤🐛"
"""

import re
from functools import lru_cache


@lru_cache(maxsize=4096)
def compress_cycle(text: str) -> str:
    """压缩循环重复子串。使用 lru_cache 缓存已处理文本的结果。

    (.+?)  : 最小匹配 1+ 字符
    \1{2,} : 该捕获组重复 2 次以上 (即总共 ≥3 次)
    """
    return re.sub(r"(.+?)\1{2,}", r"\1", text)
