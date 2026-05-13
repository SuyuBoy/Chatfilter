"""
Embedding 服务: BGE 模型推理 + 本地缓存。

职责:
  - 加载 bge-small-zh-v1.5 (或微调版本) 进行文本向量化
  - 本地 EmbeddingCache: 快速 peek/put, 供 anchor 查询
  - 统一缓存 (UnifiedCache) 负责 pipeline 级别的 embedding 缓存

模型:
  默认: BAAI/bge-small-zh-v1.5 (512d, 100MB, ~2ms/条 CPU)
  微调: models/bge-small-zh-v1.5-ft (ft_model_path 配置切换)
  支持: SentenceTransformer (默认) / ONNX Runtime (onnx_path 配置)
"""

import time
import hashlib
from typing import Any

import numpy as np

from pathlib import Path

from config.settings import Settings

# 项目根目录 + 本地模型路径 (优先级: base > small > HF)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOCAL_BASE = _PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
_LOCAL_SMALL = _PROJECT_ROOT / "models" / "bge-small-zh-v1.5"

# 文本哈希: 优先 xxhash, 回退 md5
try:
    import xxhash as _xxhash

    def _hash_text(text: str) -> int:
        return _xxhash.xxh64(text).intdigest()
except ImportError:
    def _hash_text(text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest(), 16) & ((1 << 64) - 1)


class EmbeddingCache:
    """本地 embedding 缓存 — 轻量级, 仅用于 anchor 查询。

    UnifiedCache 负责 pipeline 级别的 embedding 缓存。
    此缓存作为快速本地存储, 供 _find_best_cluster 的锚点查询使用。
    """

    def __init__(self, ttl: float = 600.0, max_size: int = 50000):
        self._cache: dict[int, tuple[np.ndarray, float]] = {}  # hash → (embedding, last_access)
        self._ttl = ttl       # 缓存有效期 (秒)
        self._max_size = max_size

    def peek(self, text: str) -> np.ndarray | None:
        """查询缓存 (不影响命中率统计, 仅用于 anchor 查询)。"""
        key = _hash_text(text)
        entry = self._cache.get(key)
        if entry is not None and (time.time() - entry[1] < self._ttl):
            self._cache[key] = (entry[0], time.time())  # 更新访问时间
            return entry[0]
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """写入缓存。超容量时淘汰最旧 10% 条目。"""
        key = _hash_text(text)
        if len(self._cache) >= self._max_size:
            items = sorted(self._cache.items(), key=lambda x: x[1][1])
            for k, _ in items[: len(items) // 10]:
                del self._cache[k]
        self._cache[key] = (embedding.copy(), time.time())


class Embedder:
    """Embedding 推理引擎。

    模型加载优先级:
      1. ft_model_path (微调模型) > 2. model_name (配置指定) > 3. 本地模型 > 4. HF 下载
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        emb_cfg = settings.embedding
        self._dim = emb_cfg.embedding_dim
        self._cache = EmbeddingCache(ttl=emb_cfg.cache_ttl)
        self._model: Any = None
        self._model_name: str = ""

        # 微调模型优先
        if emb_cfg.ft_model_path:
            ft_path = _PROJECT_ROOT / emb_cfg.ft_model_path
            if ft_path.exists():
                self._model_name = str(ft_path.resolve())
        # 回退: 配置指定 → 本地 base → 本地 small → HF
        if not self._model_name:
            if emb_cfg.model_name:
                self._model_name = emb_cfg.model_name
            elif _LOCAL_BASE.exists():
                self._model_name = str(_LOCAL_BASE.resolve())
            elif _LOCAL_SMALL.exists():
                self._model_name = str(_LOCAL_SMALL.resolve())
            else:
                self._model_name = "BAAI/bge-small-zh-v1.5"

        self._onnx_path: str = emb_cfg.onnx_path
        self._initialized: bool = False

    async def initialize(self) -> None:
        """延迟加载模型 (SentenceTransformer 或 ONNX)。"""
        if self._initialized:
            return
        if self._onnx_path:
            await self._init_onnx()
        else:
            await self._init_sentence_transformers()
        self._initialized = True

    async def _init_sentence_transformers(self) -> None:
        """加载 SentenceTransformer 模型。"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

    async def _init_onnx(self) -> None:
        """加载 ONNX Runtime 模型 (占位, 需 tokenizer 配置)。"""
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.settings.embedding.onnx_threads
            sess_options.inter_op_num_threads = 1
            self._model = ort.InferenceSession(self._onnx_path, sess_options=sess_options)
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        """同步编码文本列表。返回 L2 归一化的 embedding 矩阵。"""
        if self._model is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")

        if hasattr(self._model, "encode"):
            # SentenceTransformer 路径
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.settings.embedding.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        else:
            # ONNX 路径 (需 tokenizer 配置)
            raise NotImplementedError("ONNX inference requires tokenizer setup")
