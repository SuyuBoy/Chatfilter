# 聚类增强方案（后端优先）

## Phase 1 — 关键词签名增强 (k-NLPmeans)

### 当前问题

BGE 嵌入对短文本（"帅啊" vs "真帅"）容易拉近，但对"帅"的褒义和"弱智吧"的反讽完全分不开。纯余弦相似度的语义空间在高维短文本上信噪比低。

### 方案：嵌入 + 关键词双通道相似度

每条弹幕同时计算两路相似度，加权合并：

```
sim_total = α × cosine(emb, centroid) + (1-α) × jaccard(keywords, centroid_keywords)
```

**关键词提取**：jieba 分词后，过滤停用词 + 单字，保留前 5 个高频词作为"签名"。

**簇签名维护**：每个 MicroCluster 维护一个关键词频率字典，增量更新。

### 实现细节

**`MicroCluster` 新增字段：**
```python
keyword_freq: dict[str, int]   # 词 → 出现次数
top_keywords: list[str]        # 前 5 个关键词
```

**`can_join()` 新增逻辑：**
```python
# 当前：只看 embedding cosine
sim_centroid = float(np.dot(embedding, self.centroid))

# 新增：同时看关键词重叠
kws_new = extract_keywords(raw_text)
kws_old = self.top_keywords
jaccard = len(kws_new & kws_old) / max(len(kws_new | kws_old), 1)
sim_combined = alpha * sim_centroid + (1 - alpha) * jaccard
```

**参数：**
- `alpha = 0.7` — 嵌入权重，可配置
- `keyword_topk = 5` — 每簇保留前 5 个关键词
- 关键词过滤：长度 >= 2 字符，非纯数字/符号

### 改动文件

- `src/engine/micro_cluster.py` — 新增 keyword_freq、top_keywords、can_join 双通道
- `src/engine/cluster_engine.py` — join 时更新关键词，新建 slot 时初始化关键词
- `config/config.yaml` — 新增 `keyword_alpha`、`keyword_topk` 参数
- `config/settings.py` — 新增配置类

### 复杂度

- 每条弹幕：额外 O(分词 + 5 词 × N 槽位) ≈ <0.5ms
- 内存：每个 slot 额外 ~200B

### 代码量：~80 行

---

## Phase 2 — HDBSCAN 周期重整

### 当前问题

`maintenance()` 里的 merge/split 是纯启发式：
- merge 只看 centroid 余弦 > 0.92 → 可能把"游戏"和"声卡"合并
- split 用 K-Means(k=2) 强制二分 → 从哪来有 3 个话题的簇拆不对

### 方案：周期收集 embedding → HDBSCAN 自动重构整个槽位布局

不是替换在线匹配层，而是把 `maintenance()` 的 merge/split 逻辑换成 HDBSCAN 全局重算。

### 流程

```
每 300 条触发 maintenance():
  1. 收集当前所有 slot 的 member embeddings (共 ~200-500 个向量)
  2. HDBSCAN(min_cluster_size=3, metric='cosine')
  3. 输出: N 个簇 + 噪声点
  4. 噪声点 → 丢弃（一次性弹幕不值得占槽位）
  5. N 个簇 → 重新分配 slot_id，更新 centroid/top_examples
  6. 清空被合并的旧 slot，保留 cluster_id 映射（前端平滑过渡）
```

### 关键设计

**簇 ID 映射**：HDBSCAN 每次的输出簇 ID 会变。维护一个新旧映射表：
```python
old_to_new: dict[str, str]  # 旧 cluster_id → 新 cluster_id
```
前端状态更新时查表转换，避免 UI 闪烁。

**噪声处理**：HDBSCAN 自动标记噪声点（label=-1）。这些是"一次性弹幕"，丢掉不占槽位。

**在线层的 centroid 更新**：重整后新 centroid 用 HDBSCAN 簇内成员的均值；后续在线匹配仍按老逻辑更新。

### 参数

- `min_cluster_size = 3` — 至少 3 条弹幕才成簇
- `metric = 'cosine'` — 和在线层一致
- `maintenance_interval = 300` — 和现在一致

### 改动文件

- `src/engine/cluster_engine.py` — maintenance() 重写 merge/split 部分
- `config/config.yaml` — 新增 `min_cluster_size` 参数
- `config/settings.py` — 新增配置项

### 复杂度

- 每 300 条跑一次：~200 个向量 × 512 维 → <100ms
- 新增依赖：`hdbscan`（纯 pip install，无需系统级依赖 Ubuntu 20.04+ 预编译 wheel 可用）

### 代码量：~120 行

---

## 执行顺序

```
Phase 1 ← 先做（零风险，立即改善短文本匹配）
Phase 2 ← 验证 Phase 1 效果后再做（解决根本架构问题）
```
