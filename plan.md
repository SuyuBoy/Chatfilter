# 聚类系统改进路线

## 第一梯队（必须改）

### 1. centroid → EMA
```python
# 现在: 算术平均 → 一次错误加入永久污染中心
slot.centroid = (slot.centroid * old + emb) / slot.total_count

# 改为: EMA → 对早期错误不敏感，可适应语义漂移
alpha = 0.05
slot.centroid = normalize((1-alpha) * slot.centroid + alpha * emb)
```

### 2. candidate buffer（防噪声）
新消息先不直接建 cluster。至少出现 N 次才新建槽位，单次弹幕不进聚类。

### 3. cluster radius + 动态阈值
维护每个槽的半径 `mean(1 - cos(e, centroid))`，用它调节 join 阈值——紧致簇放宽，松散簇收紧。

### 4. char-ngram signature（替换关键词）
```python
# 中文短文本字级比词级稳定
"典" → 2gram: "典"
"绷不住了" → 2gram: "绷不", "不住", "住了"
```
比 jieba 分词 + Jaccard 更适合弹幕。

## 第二梯队（质量暴涨）

### 5. multi-prototype
一个 cluster 不只有一个 centroid，而是多个语义原型。弹幕热点的嵌入空间不是单峰分布。

### 6. merge 安全化
不用 centroid dot 直接判 merge。做 cross-member similarity：簇间成员整体接近才允许合并，防止链式塌缩。

### 7. HNSW/Faiss 索引
O(N) 线性扫描 → O(log N) ANN 检索。40 槽位不明显，200+ 槽位时必需。

## 第三梯队（研究级）

### 8. semantic graph
cluster ↔ cluster 关系图。建模"笑死 ↔ 绷不住 ↔ 逆天"的语义关联。

### 9. cluster confidence
每个槽位维护 density / stability / growth / purity 综合评分。

### 10. hotness decay
热点不是永久——用指数衰减热度，让过气梗自然死亡。
