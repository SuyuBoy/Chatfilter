import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class PreprocessConfig:
    """弹幕预处理 — 6 步管线参数"""
    min_text_length: int = 1
    max_text_length: int = 128
    aliases_path: str = str(PROJECT_ROOT / "config" / "variants.yaml")
    variants_path: str = str(PROJECT_ROOT / "config" / "variants.yaml")
    cycle_min_repeats: int = 2
    simhash_bits: int = 64
    simhash_ngram: int = 3
    simhash_high_conf_distance: int = 2
    simhash_candidate_distance: int = 3
    simhash_min_text_length: int = 8


@dataclass
class EmbeddingConfig:
    """向量化 — 模型加载后自动检测维度"""
    model_name: str = "models/bge-small-zh-v1.5"
    onnx_path: str = ""
    onnx_threads: int = 2
    cache_ttl: float = 600.0
    cache_max_size: int = 50000
    embedding_dim: int = 0
    normalize: bool = True
    ft_model_path: str = ""
    embed_batch_size: int = 32
    embed_batch_timeout_ms: int = 100


@dataclass
class ClusterConfig:
    """在线聚类 — Leader-Follower 双阈值 + 周期维护"""
    centroid_threshold: float = 0.7
    anchor_threshold: float = 0.8
    max_slots: int = 40
    merge_threshold: float = 0.92
    split_variance_threshold: float = 0.50
    maintenance_interval: int = 300
    max_member_embeddings: int = 50
    permanent_ttl_seconds: int = 600
    permanent_threshold: int = 100
    permanent_centroid_threshold: float = 0.75
    permanent_anchor_threshold: float = 0.82
    permanent_merge_threshold: float = 0.94
    permanent_split_variance_threshold: float = 0.45
    # 关键词双通道 (k-NLPmeans)
    keyword_weight: float = 0.7              # 嵌入通道权重 (1=纯嵌入, 0=纯关键词)
    keyword_topk: int = 100                  # 每簇保留的关键词数量
    # DBSCAN 周期重整
    dbscan_eps: float = 0.35                 # 常规槽位 DBSCAN eps (cosine 距离)
    permanent_dbscan_eps: float = 0.30       # 热点槽位 DBSCAN eps (更严格)


@dataclass
class Settings:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)


_settings: Settings | None = None


def _load_yaml(path: str) -> dict | None:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _apply_dict(settings: Settings, data: dict) -> None:
    for section_name in ("preprocess", "embedding", "cluster"):
        section_data = data.get(section_name, {})
        if not section_data:
            continue
        section_obj = getattr(settings, section_name)
        for key, value in section_data.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)


def _apply_env_overrides(settings: Settings) -> None:
    """环境变量: CHATFILTER__<section>__<field>=value (双下划线分隔)"""
    prefix = "CHATFILTER__"
    for env_key, env_val in sorted(os.environ.items()):
        if not env_key.startswith(prefix):
            continue
        key = env_key[len(prefix):]
        parts = key.split("__", 1)
        if len(parts) != 2:
            continue
        section_name, field_name = parts
        section_obj = getattr(settings, section_name, None)
        if section_obj is None or not hasattr(section_obj, field_name):
            continue
        current = getattr(section_obj, field_name)
        if isinstance(current, bool):
            setattr(section_obj, field_name, env_val.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(section_obj, field_name, int(env_val))
        elif isinstance(current, float):
            setattr(section_obj, field_name, float(env_val))
        else:
            setattr(section_obj, field_name, env_val)


def get_settings() -> Settings:
    global _settings
    if _settings is not None:
        return _settings

    _settings = Settings()

    # 1. YAML 配置文件 (env CHATFILTER_CONFIG 可指定路径)
    cfg_path = os.environ.get("CHATFILTER_CONFIG") or str(PROJECT_ROOT / "config" / "config.yaml")
    yaml_data = _load_yaml(cfg_path)
    if yaml_data:
        _apply_dict(_settings, yaml_data)

    # 2. 环境变量覆盖 (最高优先级)
    _apply_env_overrides(_settings)

    return _settings
