# 弹幕语义聚类系统

实时弹幕语义聚类引擎 — 44k+ 条弹幕测试，71.5% 去重率，纯 CPU。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 下载 BGE 模型 (只需一次)
bash scripts/setup_models.sh
```

### 启动命令

| 命令 | 说明 |
|------|------|
| `.venv/bin/python scripts/demo_terminal_server.py --server --port 8766` | 后端服务 (Web UI + API) |
| `.venv/bin/python scripts/demo_terminal_server.py` | 交互终端模式 (手动输入) |
| `.venv/bin/python demo/sender.py --csv demo/弹幕列表.csv --server http://localhost:8766 --speed 60 --batch 3` | CSV 弹幕发送端 |
| `.venv/bin/python scripts/finetune_bge.py` | BGE 有监督微调 |
| `.venv/bin/python scripts/finetune_bge.py --smoke` | 微调烟测 (200条 1epoch) |
| `.venv/bin/python scripts/active_label.py --pairs 20` | 交互式主动标注 |
| `.venv/bin/python scripts/demo_preprocess.py` | 预处理管道分步演示 |
| `bash demo/startdemo.sh` | 一键启动 (服务端+发送端) |

启动后端后访问 **http://localhost:8766** 打开 Web 监控面板。

## 架构

```
CSV 发送端 ──→ MiniHttpServer ──→ 预处理管线 ──→ Embedding ──→ 在线聚类
                 (8766, SSE)        6-step         BGE-small     Leader-Follower
                                                     512d         20 槽位
```

### 预处理 6 步

| 步骤 | 模块 | 说明 |
|------|------|------|
| ① 基础清洗 | `preprocessor.py` | 控制字符移除、全角半角统一、长度裁剪 1-128 |
| ② 统一归一化 | `normalizer.py` | jieba分词+相邻合并+变体字典+拼音可信验证 |
| ③ 循环节压缩 | `cycle_compressor.py` | 正则 `(.+?)\1{2,}` (如 "哈哈哈哈哈哈"→"哈") |
| ④ SimHash 辅助 | `simhash_dedup.py` | 仅 ≥8 字符高置信度自动合并 |
| ⑤ 精确去重 | `dedup_store.py` | 哈希表 O(1) + 原始弹幕成员保留 |

### 在线聚类

- Leader-Follower 双阈值 (簇心 + 锚点)
- 锚点反漂移 + 长度惩罚
- 20 槽位 LRU 淘汰

### 统一缓存池

4 个确定性阶段共用一份 LRU 缓存，存储 `input → (canonical, embedding)`。任一阶段命中即全短路后续所有阶段（含 embedding），重复弹幕处理耗时 < 0.1ms。

### 实时推送

SSE (Server-Sent Events) 替代轮询。每次 ingest 后服务端即时推送完整状态到浏览器，延迟 < 5ms。

## 项目结构

```
chatfilter/
├── config/
│   ├── settings.py                # 预处理 / embedding / 聚类配置
│   └── variants.yaml              # 别名 + 谐音 + 拼音归一化字典
├── src/
│   ├── start.py                   # PipelineEngine 入口
│   ├── engine/
│   │   ├── canonical_registry.py  # 三层归一引擎 + 统一缓存 + 计时
│   │   ├── preprocessor.py        # ① 基础清洗
│   │   ├── alias_normalizer.py    # ② 别名归一化
│   │   ├── variant_normalizer.py  # ③ 变体归一化
│   │   ├── cycle_compressor.py    # ④ 循环节压缩
│   │   ├── simhash_dedup.py       # ⑤ SimHash 辅助
│   │   ├── dedup_store.py         # ⑥ 精确去重
│   │   ├── embedder.py            # Embedding 推理 + 本地缓存
│   │   ├── micro_cluster.py       # 微簇: 双阈值 + 锚点 + 长度惩罚
│   │   └── pipeline_cache.py      # 统一缓存池 + 逐环节计时
│   └── network/
│       └── http_server.py         # HTTP 服务 + SSE 推送
├── static/
│   └── monitor.html               # Web 监控面板 (SSE 响应式)
├── scripts/
│   ├── demo_terminal_server.py    # 服务端入口
│   └── demo_preprocess.py         # 预处理管道分步演示
├── demo/
│   ├── sender.py                  # CSV 弹幕发送端
│   ├── startdemo.sh               # 一键启动
│   └── 弹幕列表.csv               # 测试数据 (~44k 条)
└── docs/
    └── DESIGN.md                  # 设计文档
```

## Web UI

访问 `http://localhost:8766`，三列实时面板：

| 列 | 内容 |
|----|------|
| 📥 原始弹幕 | 实时原始弹幕流，标注缓存命中环节 |
| 🔄 预处理后 | 归一化后文本，标注归一化 / 缓存命中 |
| 🧠 语义聚类 | 20 槽位聚类卡片，蓝色标题 = 规范文本，紫色标签 = 原始弹幕（内容穿透） |

顶栏实时显示：摄入量、去重数、聚类数、统一缓存命中率、emb 耗时、聚类耗时、逐环节耗时。

## 粒度调节

网页上修改簇心/锚点阈值后点击「应用」即可实时生效。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 簇心阈值 | 0.20 | 控制簇的聚合范围，越低簇越大 |
| 锚点阈值 | 0.40 | 控制簇的语义纯度，越低越容易漂移 |

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 监控面板 |
| `/events` | GET | SSE 实时推送 |
| `/state` | GET | 完整状态 JSON |
| `/ingest?text=xxx` | POST | 摄入单条弹幕 |
| `/admin/threshold?centroid=0.2&anchor=0.4` | POST | 动态调整阈值 |
