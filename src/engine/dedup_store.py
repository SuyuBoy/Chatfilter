"""
Step ⑥ 精确去重 + 频次计数 — 预处理管线的最后一步。

职责:
  - 哈希表 O(1) 查找 canonical_text 是否已存在
  - 频次累加: 相同 canonical 的 count 自动合并
  - 原始弹幕成员保留: 每个 canonical 下存储 {msg_id, raw, ts}
    用于内容穿透 — 聚类后可回查原始文本
"""

import time
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class DedupRecord:
    """一条去重记录。"""
    canonical_text: str
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    raw_examples: list[str] = field(default_factory=list)
    MAX_RAW_EXAMPLES = 5

    def update(self, count_delta: int = 1, raw_text: str = "", timestamp: float = 0) -> None:
        """更新记录: 累加计数 + 追加原始文本样本。"""
        self.count += count_delta
        self.last_seen = timestamp or time.time()
        if raw_text and len(self.raw_examples) < self.MAX_RAW_EXAMPLES:
            self.raw_examples.append(raw_text)


class DedupStore:
    """精确去重存储: OrderedDict + 频次计数 + 成员保留。"""

    def __init__(self):
        self._store: OrderedDict[str, DedupRecord] = OrderedDict()
        self._counts: dict[str, int] = {}            # canonical_text → 总出现次数
        self._members: dict[str, list[dict]] = {}    # canonical_text → [{msg_id, raw, ts}]
        self.last_canonical: str = ""

    def add(self, canonical_text: str, count: int = 1,
            raw_text: str = "", msg_id: str = "", timestamp: float = 0) -> bool:
        """添加一条 canonical 文本。返回 True 表示首次出现 (全新)。"""
        ts = timestamp or time.time()
        self.last_canonical = canonical_text

        # 记录原始弹幕成员 (内容穿透的数据源)
        if canonical_text not in self._members:
            self._members[canonical_text] = []
        self._members[canonical_text].append({"msg_id": msg_id, "raw": raw_text, "ts": ts})

        if canonical_text in self._store:
            self._store[canonical_text].update(count, raw_text, ts)
            self._counts[canonical_text] += count
            return False
        else:
            record = DedupRecord(canonical_text, count=count,
                                first_seen=ts, last_seen=ts)
            if raw_text:
                record.raw_examples.append(raw_text)
            self._store[canonical_text] = record
            self._counts[canonical_text] = count
            return True

    def get_members(self, canonical_text: str, limit: int = 0) -> list[dict]:
        """返回该 canonical 下所有原始弹幕成员。limit=0 返回全部。"""
        members = self._members.get(canonical_text, [])
        if limit > 0:
            return members[-limit:]  # 返回最近的 N 条
        return members

    def get_count(self, canonical_text: str) -> int:
        return self._counts.get(canonical_text, 0)

    def __contains__(self, canonical_text: str) -> bool:
        return canonical_text in self._store

    def __len__(self) -> int:
        return len(self._store)
