"""
Embedding 服务: BGE 模型推理 (SentenceTransformer / ONNX Runtime)。

UnifiedCache 现在是唯一的缓存层 — embedding 缓存由它管理。
此模块只负责模型加载和推理，不维护自己的缓存。
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_LOCAL_BASE = _PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
_LOCAL_SMALL = _PROJECT_ROOT / "models" / "bge-small-zh-v1.5"


class Embedder:
    """BGE 模型推理引擎。支持 SentenceTransformer 和 ONNX Runtime 两种后端。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        emb_cfg = settings.embedding
        self._dim = emb_cfg.embedding_dim
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_onnx = False
        self._model_name: str = ""

        # 模型路径优先级: 微调 > 配置 model_name > 本地 base > 本地 small > HF
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

        self._onnx_dir: str = ""
        if emb_cfg.onnx_path:
            onnx_dir = Path(emb_cfg.onnx_path)
            if (onnx_dir / "model.onnx").exists():
                self._onnx_dir = str(onnx_dir.resolve())
        self._initialized: bool = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._onnx_dir:
            await self._init_onnx()
        else:
            await self._init_sentence_transformers()
        self._initialized = True

    async def _init_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name, device="cpu")
        self._dim = self._model.get_embedding_dimension()
        self.settings.embedding.embedding_dim = self._dim

    async def _init_onnx(self) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_dir = Path(self._onnx_dir)
        onnx_file = onnx_dir / "model.onnx"

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.settings.embedding.onnx_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._model = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
        )
        self._is_onnx = True

        # Load tokenizer from ONNX directory
        self._tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

        # Detect embedding dimension via test inference
        test_emb = self._encode_sync(["test"])
        self._dim = test_emb.shape[1]
        self.settings.embedding.embedding_dim = self._dim

    def encode_batched(self, texts: list[str]) -> np.ndarray:
        """批量编码, 自动分批。"""
        if isinstance(texts, str):
            texts = [texts]
        bs = self.settings.embedding.embed_batch_size
        if bs <= 1 or len(texts) <= 1:
            return self._encode_sync(texts)
        result = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            result.append(self._encode_sync(batch))
        return np.concatenate(result) if len(result) > 1 else result[0]

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")

        if self._is_onnx:
            # ── ONNX Runtime path ──
            encoded = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors="np",
            )
            ort_inputs = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            embs = self._model.run(None, ort_inputs)[0]
            return np.asarray(embs, dtype=np.float32)
        else:
            # ── SentenceTransformers path ──
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.settings.embedding.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)

    def flush_pending(self) -> np.ndarray | None:
        return None
