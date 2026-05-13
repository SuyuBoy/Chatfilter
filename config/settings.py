from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PreprocessConfig:
    """弹幕预处理 — 6 步管线参数"""
    min_text_length: int = 1                    # 最短文本 (字符)
    max_text_length: int = 128                  # 最长文本 (字符)
    aliases_path: str = str(PROJECT_ROOT / "config" / "variants.yaml")   # 别名+谐音+拼音统一配置
    variants_path: str = str(PROJECT_ROOT / "config" / "variants.yaml")  # 同上
    cycle_min_repeats: int = 2                  # 循环压缩: (.+?)\1{N-1,}  如3→重复2次以上触发
    simhash_bits: int = 64                      # SimHash 指纹位数
    simhash_ngram: int = 3                      # SimHash n-gram 窗口
    simhash_high_conf_distance: int = 2         # 高置信度海明距离 (长文本自动合并)
    simhash_candidate_distance: int = 3         # 候选海明距离 (短文本仅日志)
    simhash_min_text_length: int = 8            # 短于该长度不触发自动合并


@dataclass
class EmbeddingConfig:
    """向量化 — 模型加载后自动检测维度"""
    model_name: str = "models/bge-base-zh-v1.5"                        # 空=自动检测本地模型, 否则 HF 名称
    onnx_path: str = ""                         # ONNX 模型导出路径 (空=用 SentenceTransformer)
    onnx_threads: int = 2                       # ONNX 推理线程数
    cache_ttl: float = 600.0                    # 缓存过期时间 (秒), 默认 10 分钟
    cache_max_size: int = 50000                 # 统一缓存池最大条目数
    embedding_dim: int = 0                      # 向量维度 (0=加载模型后自动检测)
    normalize: bool = True                      # 是否 L2 归一化
    ft_model_path: str = ""                     # 微调模型路径, 空=用原始模型
    embed_batch_size: int = 16                  # embedding 批量推理大小
    embed_batch_timeout_ms: int = 100            # 攒批超时 (ms), 不满也发


@dataclass
class ClusterConfig:
    """在线聚类 — Leader-Follower 双阈值 + 周期维护"""
    centroid_threshold: float = 0.4              # 新消息 vs 簇中心余弦相似度底线 (0~1)
    anchor_threshold: float = 0.6               # 新消息 vs 簇锚点余弦相似度底线 (0~1)
    max_slots: int = 40                          # 最大聚类槽位数
    merge_threshold: float = 0.92                # centroid 相似度超过此值则合并
    split_variance_threshold: float = 0.50       # 簇内平均相似度低于此值触发拆分
    maintenance_interval: int = 300              # 每 N 条消息执行一次合并+拆分
    # 永久语义 (count > threshold 晋升, 不限数量, 永不淘汰)
    max_member_embeddings: int = 50             # 簇内保留的成员 embedding 数上限
    permanent_ttl_seconds: int = 600             # 热点语义无更新淘汰时间 (秒)
    permanent_threshold: int = 100               # 消息数超过此值晋升为永久语义
    permanent_centroid_threshold: float = 0.75   # 永久语义聚类阈值 (更严格)
    permanent_anchor_threshold: float = 0.82     # 永久语义锚点阈值 (更严格)
    permanent_merge_threshold: float = 0.94      # 永久语义合并阈值 (更高)
    permanent_split_variance_threshold: float = 0.45  # 永久语义拆分阈值 (更低=更积极拆分)


@dataclass
class Settings:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
