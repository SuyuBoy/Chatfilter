# 弹幕实时流式语义聚类方案设计

> 200 msg/s | 60s 滑动窗口 | Python + CPU 优先 | 在线增量聚类

---

## 修订记录

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v1.0 | 2026-05-10 | 初始方案 |
| v2.0 | 2026-05-10 | 综合 Claude/GPT/Gemini 三方评审修订 |
| v2.1 | 2026-05-10 | 新增长度惩罚防前缀误合并；质量过滤改为软惩罚三分类(语义/反应/噪声)；新增核心概念解释章节 |
| v2.2 | 2026-05-10 | 算法部分全部实现并测试通过；修复 SimHash 短文本指纹全零 bug；修复别名双重替换 bug |
| v3.0 | 2026-05-10 | 架构升级：三层归一引擎 + 注册表模式；新增终端交互 Demo |
| v4.0 | 2026-05-11 | Pipeline 公共缓冲池 + Embedding 统一缓存 (命中即短路)；逐环节时间统计；SSE 实时推送；内容穿透；Web 监控面板 |
| v4.1 | 2026-05-11 | 聚类质量改进：提高默认阈值、簇内一致性检查、周期合并+拆分(k=2 K-Means)；拼音自动反向生成(声调防误匹配)；jieba 分词联动 variants |
| v4.2 | 2026-05-12 | BGE 有监督微调(变体正样本对)；热点语义晋升+10min TTL 淘汰；聚类晋升前重算 centroid 选摘要；情感守卫防同主题反情感合并 |

---

## 实现状态 (v4.2)

> 更新时间: 2026-05-12

### 项目结构

```
chatfilter/
├── config/
│   ├── settings.py                # 预处理/embedding/聚类配置
│   └── variants.yaml              # 别名 + 谐音 + 拼音归一化字典
├── src/
│   ├── start.py                   # PipelineEngine 入口
│   ├── engine/
│   │   ├── canonical_registry.py  # 三层归一引擎 + 缓存池 + 计时
│   │   ├── preprocessor.py        # ① 基础清洗
│   │   ├── alias_normalizer.py    # ② 别名归一化 (子串扫描)
│   │   ├── variant_normalizer.py  # ③ 变体归一化 (谐音字典 + pinyin)
│   │   ├── cycle_compressor.py    # ④ 循环节压缩 (regex)
│   │   ├── simhash_dedup.py       # ⑤ SimHash 辅助模糊
│   │   ├── dedup_store.py         # ⑥ 精确去重 + 频次计数
│   │   ├── embedder.py            # Embedding 推理 + 长期缓存
│   │   ├── micro_cluster.py       # 微簇: 双阈值 + 锚点防漂移 + 长度惩罚
│   │   └── pipeline_cache.py      # 公共缓冲池 + 逐环节时间统计
│   └── network/
│       └── http_server.py         # 极简 HTTP 服务 + SSE 实时推送
├── static/
│   └── monitor.html               # Web 监控面板 (SSE 驱动, 3列)
├── scripts/
│   ├── demo_terminal_server.py    # 终端 + HTTP 双模式 Demo
│   └── demo_preprocess.py         # 预处理管道分步演示
├── demo/
│   ├── sender.py                  # CSV 弹幕发送端
│   ├── startdemo.sh               # 一键启动脚本
│   └── 弹幕列表.csv               # 测试数据 (~44k条)
└── docs/
    └── DESIGN.md                  # 本文档
```

### 已实现的核心功能

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置管理 | `config/settings.py` | 预处理 + embedding + 聚类全部配置 |
| 基础清洗 | `preprocessor.py` | 控制字符/全角半角/空白归一/长度裁剪 |
| 别名归一化 | `alias_normalizer.py` | 子串扫描 + 最长优先 + 防双重替换 |
| 变体归一化 | `variant_normalizer.py` | 谐音字典 + pypinyin 白名单 |
| 循环节压缩 | `cycle_compressor.py` | 正则 `(.+?)\1{2,}` |
| SimHash 辅助 | `simhash_dedup.py` | n-gram 降级 + 仅 auto-merge 替换 |
| 精确去重 | `dedup_store.py` | 频次计数 + 原始弹幕成员保留 |
| 注册表 | `canonical_registry.py` | 6步预处理统一入口 + is_new 判定 |
| Embedding | `embedder.py` | ONNX/ST 推理 + xxhash 长缓存 (TTL=10min) |
| 微簇聚类 | `micro_cluster.py` | 双阈值 + 锚点防漂移 + 长度惩罚 |
| Pipeline 缓存 | `pipeline_cache.py` | 每环节 (输入→最终canonical) 缓存, 命中即短路 |
| 时间统计 | `pipeline_cache.py` | 逐环节 avg/min/max/recent 耗时 |
| HTTP 服务 | `http_server.py` | 极简 asyncio + SSE 实时推送 |
| Web 监控 | `monitor.html` | SSE 响应式 + 3列实时 + 阈值调节 + 手动发送 + 热点栏 |
| 发送端 | `sender.py` | CSV 按时间戳顺序 POST, 支持倍速播放 |
| BGE 微调 | `finetune_bge.py` | 有监督对比学习 (变体→规范词正样本对), GPU 训练 39s |

### Pipeline 缓存机制

4 个确定性阶段各自的 LRU 缓存，key = 阶段输入文本，value = 最终 canonical_text。查缓存按管道正向顺序，首个命中即短路后续所有阶段：

```
raw_text → [cleanse 缓存?] ──命中──→ 直接 dedup (短路 alias/variant/cycle/simhash)
   ↓ 未命中, 执行清洗
cleaned → [alias 缓存?]   ──命中──→ 直接 dedup (短路 variant/cycle/simhash)
   ↓ 未命中, 执行别名归一化
text    → [variant 缓存?] ──命中──→ 直接 dedup (短路 cycle/simhash)
   ↓ 未命中, 执行变体归一化
text    → [cycle 缓存?]   ──命中──→ 直接 dedup (短路 simhash)
   ↓ 未命中, 执行循环压缩 → simhash → dedup → 写入全部4层缓存
```

网页上每条弹幕显示最早命中缓存的环节 (`清洗` `同义词替换` `变体归一` `循环节压缩`)，未命中则不显示。

### 内容穿透

`DedupStore._members` 为每个 canonical_text 保留原始弹幕列表 `{msg_id, raw, ts}`。聚类卡片同时展示归一化规范文本和紫色原始弹幕标签，可溯源每条弹幕的本来面目。

### SSE 实时推送

服务端内置 `/events` 端点。每次 ingest 后广播完整状态 JSON 到所有连接的浏览器。前端用 `EventSource` 接收，无需轮询，延迟 < 5ms。

### 修复的 Bug

| Bug | 症状 | 修复 |
|-----|------|------|
| SimHash 短文本全零 | "好帅"/"哈"/"牛批"指纹=0 | n-gram 降级: <3字→1/2-gram |
| 别名双重替换 | "灰泽满加油"→"灰泽满满加油" | canonical 已存在则跳过 |
| Embedding 缓存统计失真 | anchor 查询污染命中数，新文本不统计 miss | 新增 `peek()` 无统计查询 |
| 管道缓存重复命中 | 重复弹幕 4 个阶段全显示缓存命中 | 首个命中即短路，映射到最终 canonical |
| 网页重复弹幕不更新 | `raw+cluster_id` 做 key 导致去重 | 每条消息分配唯一递增 id |

---

## 架构概览

### 核心概念: ID 驱动的三层归一 + 回放

每条弹幕从进入系统起分配唯一的 `msg_id`。预处理不 "吃掉" 原始弹幕，而是建立映射关系：

```
原始弹幕 (id=A, "牛批")   ──→ canonical_id=X, canonical_text="牛逼"
原始弹幕 (id=B, "好听×10") ──→ canonical_id=Y, canonical_text="好听"
```

只有代表文本 (canonical_text) 进入 Embedding + 聚类。语义分类完成后通过 canonical_id 回查原始弹幕。

### 三层归一引擎

```
每条弹幕: msg_id + raw_text + count=1
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ [层1] 精确匹配 (哈希表 O(1))                          │
│   "好听" == "好听" → 同一 canonical_id                │
│   → count 累加, raw_text 追加到成员列表               │
│   → 不新增 embedding 计算                            │
├──────────────────────────────────────────────────────┤
│ [层2] 模糊归一 (别名/谐音/循环/1-2字差异)              │
│   "牛批"→"牛逼"  "小满好棒"→"灰泽满好棒"              │
│   → 保留原始 msg_id 和 raw_text                       │
│   → 已有 canonical 直接用 cached embedding            │
├──────────────────────────────────────────────────────┤
│ [层3] 完全新文本                                      │
│   → 自身为 canonical, msg_id = canonical_id          │
│   → 触发 embedding 推理                              │
└──────────────────────────────────────────────────────┘
    │
    ▼ 仅代表文本进入下游
Embedding → 在线聚类 → 语义分类
    │
    ▼ 完成后可回放
每个簇按 canonical_id 钻取原始弹幕
```

### 注册表模式 (CanonicalRegistry)

将 6 步预处理封装为统一的 `CanonicalRegistry.register(raw_text, msg_id) → RegisterResult`。下游只需检查 `is_new` 来决定是否触发 embedding。

### 数据流

```
原始弹幕 → CanonicalRegistry.register()
              │
              ├─ 层1/层2: is_new=False → 直接归入已有簇 (复用 cached embedding)
              │
              └─ 层3: is_new=True → Embedding(长缓存) → 微簇聚类(双阈值锚点)
```

- **缓冲层**: asyncio.Queue + 批量处理
- **Embedding**: ONNX Runtime + bge-small-zh-v1.5 (本地) + 长期缓存
- **聚类**: 在线微簇(双阈值+锚点防漂移)
- **回放**: 每个簇的 members 保留原始 raw_text + count

---

## 管道详解

### 管道顺序

```
清洗 → 别名(子串) → 变体(拼音+字典) → 压缩(regex) → SimHash(辅助) → 去重
  │        │              │               │             │            │
  │        │              │               │             │            └── 最后计数
  │        │              │               │             └── 残余模糊(降级)
  │        │              │               └── 内部重复归一
  │        │              └── 音近字/谐音/缩写 (SimHash盲区)
  │        └── 实体别名对齐 (子串扫描，覆盖嵌入文本)
  └── 基础噪声过滤
```

每一步解决一类特定的噪声，各司其职，互不重叠。

### 数据流示意

```
原始弹幕 (200条/s)
    │
    ▼ ① 基础清洗 — 过滤垃圾、统一格式
    │
    ▼ ② 别名归一化 (子串扫描, 长优先)
    │   "小满好棒"→"灰泽满好棒"
    │
    ▼ ③ 变体归一化 (谐音字典 + pypinyin)
    │   "煞笔"→"傻逼", "nb"→"牛逼"
    │
    ▼ ④ 循环节压缩 (正则反向引用)
    │   "哈哈哈哈哈哈"→"哈"
    │
    ▼ ⑤ SimHash 辅助模糊 (降级)
    │   长文本高置信度自动合并；短文本仅写候选日志
    │
    ▼ ⑥ 精确去重 + 频次计数
    │
    ▼ Embedding (ONNX + xxhash长缓存 TTL=10min)
    │
    ▼ 微簇聚类 (双阈值+锚点防漂移)
    │
    ▼ 输出
```

---

## 阶段 1: 项目骨架搭建

### 步骤 1-2: 项目结构 + 配置管理

- `pyproject.toml` / `requirements.txt`
- `config/settings.py`: 模型名、批大小(20)、批间隔(100ms)、窗口大小(60s)、聚类阈值
- 依赖: `sentence-transformers`, `onnxruntime`, `numpy`, `pypinyin`, `pyyaml`, `xxhash`

---

## 阶段 2: 预处理管道 (6 步)

### 步骤 3: 文本基础清洗 (`preprocessor.py`)

- 过滤: 空文本、超长纯数字(>4字)、超长/超短文本
- **保留**: 短纯数字 ("666"/"520")、短 emoji、混合文本
- **放行纯符号**: 不在此步过滤——后续循环节压缩会处理
- 清洗: 移除控制字符、统一全角/半角、统一空白符、长度裁剪 1~128 字符
- 设计原则: 预处理只做确定性清洗，不确定的留给后续步骤

### 步骤 4: 别名归一化 — 子串扫描 (`alias_normalizer.py`)

**修复**: 从全文精确匹配改为子串扫描替换。别名按长度降序排列，长别名优先匹配。防双重替换：canonical 已在文本中时跳过。

### 步骤 5: 变体归一化 — 字典 + 自动拼音 (`variant_normalizer.py`)

**核心修正**: SimHash 对音近字替换完全无效（"煞笔" vs "傻逼" 汉明距离约 40-50），必须用字典处理。

- **Layer 1**: 谐音字字典精确匹配（含拉丁写法如 `niubi`、`yyds`）
- **Layer 2**: 从规范词自身自动生成带声调拼音反向映射（`nan2ting1` → `难听`），无需手动配置。声调天然防误匹配（`lai2le` ≠ `lai4le`）

### 步骤 6: 循环节压缩 (`cycle_compressor.py`)

正则反向引用 `(.+?)\1{2,}`，容忍尾部残余。最少 3 次重复触发。

### 步骤 7: SimHash 辅助模糊 (`simhash_dedup.py`)

**角色降级**: 从主归一化器 → 辅助模糊候选生成器。
- 汉明距离 ≤ 2 且长度 ≥ 8 → 高置信度自动合并
- 短文本 → 仅写候选日志，不自动合并

### 步骤 8: 精确去重 + 频次计数 (`dedup_store.py`)

包含批内局部去重，避免同批次相同文本重复触发 embedding。

---

## 阶段 3: Embedding 服务

### 步骤 10: Embedding 推理引擎 (`embedder.py`)

- 默认模型 `BAAI/bge-small-zh-v1.5` (512维)，可通过配置切换
- ONNX Runtime / SentenceTransformer 双路径
- xxhash 长缓存 (TTL=10min)，`peek()` 方法用于 anchor 查询不污染统计
- L2 归一化

---

## 阶段 4: 在线聚类

### 核心架构

```
新消息 → Leader-Follower + 双阈值锚点 (实时, <1ms)
              ↓ 每 300 条
         周期维护: 合并相似簇 + K-Means 拆分退化簇
              ↓
           输出
```

- **簇内一致性检查**: 退化簇 (内部相似度 < 0.50) 拒绝新成员加入
- **周期合并**: centroid 相似度 > 0.92 的簇合并；热点合并含情感守卫(同主题反情感不合并)
- **K-Means 拆分**: 簇内相似度 < 0.50 时 k=2 拆分
- **热点语义**: count > 100 晋升，不限数量，晋升前重算 centroid 并选取最靠近中心的文本为摘要；10min 无更新自动淘汰
- LRU 淘汰: 满槽时淘汰最久未更新的簇

避免经典漂移问题：A→B→C→D 逐级相似导致「鼓励」漂成「欢乐」。

### 步骤 11: 微簇 (`micro_cluster.py`)

**双阈值 + 锚点防漂移 + 长度惩罚**:

```python
def can_join(self, embedding, new_text_len=0, centroid_threshold=0.78, anchor_threshold=0.82):
    # 条件1: 与 centroid 相似度 > T_centroid
    if sim_centroid < centroid_threshold: return False
    # 条件2: 与至少一个锚点相似度 > T_anchor (防漂移)
    if max_anchor_sim < anchor_threshold: return False
    # 条件3: 长度差异惩罚 — 前缀共享 ≠ 语义相同
    if new_text_len > 0:
        penalty = _length_penalty(new_text_len, anchor_avg_len)
        if sim_centroid * penalty < centroid_threshold: return False
    return True
```

长度惩罚: ratio≤1.5→1.0, ≤3.0→0.90, >3.0→0.80。防止 "帅啊"(3字) 和 "帅啊我看XXX也就那样了"(13字) 混入同一簇。

---

### Embedding 模型选型

| 模型 | 维度 | 大小 | 聚类 | CPU 推理 | 推荐 |
|------|------|------|------|----------|------|
| **bge-small-zh-v1.5** | 512 | ~100MB | 63.96 | ~2ms/条 | 快速验证 |
| **bge-base-zh-v1.5** | 768 | ~400MB | 68.07 | ~4ms/条 | 生产推荐 |
| bge-large-zh-v1.5 | 1024 | ~1.3GB | 69.13 | ~8ms/条 | GPU 可用时 |

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 别名归一化 | 子串扫描 + 长优先策略 | 修复全文匹配覆盖率极低的问题 |
| 变体归一化 | 谐音字字典 + pypinyin | SimHash 对音近字完全无效，字典才是正解 |
| SimHash 角色 | 降级为辅助候选生成器 | 变体字典+拼音已覆盖 80%+ |
| 循环节压缩 | 正则反向引用 `(.+?)\1{2,}` | 容忍尾部残余 |
| Embedding 长缓存 | xxhash + TTL=10min | 高频词跨窗口复用 |
| 双阈值锚点+长度惩罚 | centroid(0.78) + anchor(0.82) | 防漂移+防前缀误合并 |
| Pipeline 缓存 | 输入→最终 canonical，命中即短路 | 重复弹幕跳过全部预处理 |
| 实时推送 | SSE 替换轮询 | 延迟从 ~50ms 降到 < 5ms |
| 内容穿透 | `_canonicals` 集合追踪所有加入过的 canonical | 跨 canonical 聚类也能穿透原始文本 |
| 聚类维护 | 合并(>0.92) + K-Means 拆分(<0.50) | 相似簇合并减冗余，退化簇拆分恢复纯度 |
| 拼音匹配 | 从规范词自动反向生成声调拼音 | 无需手动配置，声调天然防误匹配 |
| 布隆过滤器 | 不使用 | 内存充足，需要精确计数 |

---

## 备选方案 (已排除)

- **K-Means**: 需预设 K，不支持流式增量更新
- **DBSCAN**: 批量算法，在线版实现复杂
- **GPT/API Embedding**: 延迟不可控，成本高
- **Spark Streaming**: 过重，延迟高
- **编辑距离全局去重**: O(n²) 复杂度

---

## 进一步考虑

1. **冷启动**: 前几秒使用更低双阈值缓解
2. **持久化**: 微簇 centroid + anchor 快照写入 Redis
3. **拼音字典自动扩充**: SimHash 候选日志中确认的映射自动加入 `variants.yaml`
4. **情绪轴**: 增加简单情感词典，区分夸/骂
5. **事件检测**: 基于突发检测自动标记直播事件

---

## 专项 Q&A

### Q: SimHash 能清理故意不同的错别字吗？

**不能。** 直播弹幕的"错别字"是高度模式化的规避/谐音词，不是自然 typo。采用谐音字字典 + 拼音归一化作为核心，SimHash 降级为辅助。

### Q: 聚类粒度能调吗？

**能，双阈值调节。** centroid 控制聚合范围，anchor 控制语义纯度。运行时通过网页或 API 动态调整。

### Q: 聚合后还能看到独立弹幕吗？

**能。** 每个簇的 members 保留每条独立弹幕的原始文本、频次、时间戳。聚类卡片展示紫色原始弹幕标签（内容穿透）。

### Q: 语义聚合有意义吗？

**有。** "加油"/"冲啊"/"你可以的" 在 BGE 嵌入空间中相近 → 自动归为「鼓励」簇，锚点机制防止漂移。

### Q: 聚类为什么有时不准确？

在线聚类的固有挑战：先入为主的消息定义锚点方向，后续消息顺序依赖。通过提高阈值 + 簇内一致性检查 + 周期性合并/拆分来缓解。
