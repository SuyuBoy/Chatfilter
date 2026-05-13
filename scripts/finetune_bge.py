#!/usr/bin/env python3
"""
BGE 有监督微调: 用 variants.yaml 的 (变体, 规范词) 正样本对做对比学习.
用法: .venv/bin/python scripts/finetune_bge.py [--smoke]
"""

import sys
import yaml
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader

# ── 配置 ──
PROJECT = Path(__file__).parent.parent
VARIANTS_PATH = PROJECT / "config" / "variants.yaml"
MODEL_NAME = str(PROJECT / "models" / "bge-small-zh-v1.5")
OUTPUT_PATH = str(PROJECT / "models" / "bge-small-zh-v1.5-ft")
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-5
WARMUP_RATIO = 0.1

# ── 1. 加载 variants 正样本对 ──
print("1. Loading variant pairs...")
with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
variants: dict[str, list[str]] = data.get("variants", {})

pairs: list[tuple[str, str]] = []
for canonical, vlist in variants.items():
    # 规范词自身也加入 (增强表达)
    pairs.append((canonical, canonical))
    for v in vlist:
        pairs.append((v, canonical))

# 去重
pairs = list(set(pairs))
print(f"   positive pairs: {len(pairs)}")
for a, b in pairs[:8]:
    print(f"     {a:12s} → {b}")

# ── 2. smoke / full ──
smoke = "--smoke" in sys.argv
if smoke:
    pairs = pairs[:len(pairs)//4]
    _epochs = 1
    print(f"   SMOKE: {len(pairs)} pairs, {_epochs} epoch")
else:
    _epochs = EPOCHS

# ── 3. 训练 ──
print(f"3. Training on {torch.cuda.get_device_name(0)}...")
model = SentenceTransformer(MODEL_NAME)

train_examples = [InputExample(texts=[a, b]) for a, b in pairs]
train_dataloader = DataLoader(train_examples, batch_size=BATCH_SIZE, shuffle=True)  # type: ignore[arg-type]
train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = int(len(train_dataloader) * _epochs * WARMUP_RATIO)

if smoke:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=_epochs,
        warmup_steps=max(warmup_steps, 1),
        optimizer_params={"lr": LR},
        show_progress_bar=True,
    )
else:
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=_epochs,
        warmup_steps=max(warmup_steps, 1),
        optimizer_params={"lr": LR},
        show_progress_bar=True,
        output_path=OUTPUT_PATH,
    )

if smoke:
    print("Smoke OK. Run without --smoke for full training.")
else:
    print(f"4. Saved to {OUTPUT_PATH}")
print("Done.")
