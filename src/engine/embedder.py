"""
Embedding 服务: BGE 模型推理。

UnifiedCache 现在是唯一的缓存层 — embedding 缓存由它管理。
此模块只负责模型加载和推理，不维护自己的缓存。
"""

import time
import hashlib
from typing import Any

import numpy as np

from pathlib import Path

from config.settings import Settings

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOCAL_BASE = _PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
_LOCAL_SMALL = _PROJECT_ROOT / "models" / "bge-small-zh-v1.5"


class Embedder:
    """BGE 模型推理引擎。缓存由 UnifiedCache 统一管理。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        emb_cfg = settings.embedding
        self._dim = emb_cfg.embedding_dim
        self._model: Any = None
        self._model_name: str = ""
        # 微调模型优先
        if emb_cfg.ft_model_path:
            ft_path = _PROJECT_ROOT / emb_cfg.ft_model_path
            if ft_path.exists():
                self._model_name = str(ft_path.resolve())
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
        if self._initialized:
            return
        if self._onnx_path:
            await self._init_onnx()
        else:
            await self._init_sentence_transformers()
        self._initialized = True

    async def _init_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name, device="cpu")
        self._dim = self._model.get_embedding_dimension()

    async def _init_onnx(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.settings.embedding.onnx_threads
        sess_options.inter_op_num_threads = 1
        self._model = ort.InferenceSession(self._onnx_path, sess_options=sess_options)

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        if hasattr(self._model, "encode"):
            embeddings = self._model.encode(
                texts, normalize_embeddings=self.settings.embedding.normalize,
                show_progress_bar=False)
            return np.asarray(embeddings, dtype=np.float32)
        raise NotImplementedError("ONNX inference requires tokenizer setup")
