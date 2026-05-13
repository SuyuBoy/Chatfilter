#!/usr/bin/env python3
"""
Export BGE model to ONNX with mean pooling + L2 normalization.

Usage:
  .venv/bin/python scripts/export_onnx.py
  .venv/bin/python scripts/export_onnx.py --model models/bge-base-zh-v1.5
  .venv/bin/python scripts/export_onnx.py --model BAAI/bge-small-zh-v1.5 --output models/bge-small-zh-v1.5-onnx
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class BGEModelWithHead(torch.nn.Module):
    """BERT encoder + mean pooling + L2 normalization."""

    def __init__(self, model: AutoModel):
        super().__init__()
        self.encoder = model

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs[0] shape: [batch, seq, hidden]
        hidden = outputs[0]

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.shape).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # L2 normalization
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def validate(model_name: str, onnx_path: str, tokenizer: AutoTokenizer):
    """Compare PyTorch vs ONNX outputs to verify correctness."""
    import onnxruntime as ort

    # Load original model
    pt_model = AutoModel.from_pretrained(model_name)
    pt_wrapped = BGEModelWithHead(pt_model)
    pt_wrapped.eval()

    # Load ONNX
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    test_texts = [
        "你好世界",
        "今天天气真好",
        "哈哈哈哈哈哈",
        "主播太强了",
    ]

    # PyTorch inference
    encoded = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        pt_out = pt_wrapped(encoded["input_ids"], encoded["attention_mask"])
    pt_vecs = pt_out.numpy()

    # ONNX inference
    ort_inputs = {
        "input_ids": encoded["input_ids"].numpy().astype(np.int64),
        "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
    }
    ort_out = sess.run(None, ort_inputs)[0]

    # Compare
    max_diff = 0.0
    for i, text in enumerate(test_texts):
        sim = float(np.dot(pt_vecs[i], ort_out[i]))
        diff = np.abs(pt_vecs[i] - ort_out[i]).max()
        max_diff = max(max_diff, diff)
        print(f"  [{text}] cos_sim={sim:.6f}  max_diff={diff:.6f}")

    print(f"\n  Max abs diff: {max_diff:.6f}")
    if max_diff < 1e-4:
        print("  ✅ Validation PASSED")
        return True
    else:
        print("  ⚠️  Validation: differences detected (may be acceptable for fp32)")
        return max_diff < 1e-2


def main():
    parser = argparse.ArgumentParser(description="Export BGE model to ONNX")
    parser.add_argument("--model", default="models/bge-small-zh-v1.5",
                        help="Model path or HuggingFace name")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <model_path>-onnx)")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Validate ONNX output vs PyTorch")
    args = parser.parse_args()

    model_name = args.model
    output_dir = Path(args.output) if args.output else Path(f"{model_name}-onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print(f"Exporting {model_name} → {onnx_path}")

    # Load model and tokenizer
    print("Loading model...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    wrapped = BGEModelWithHead(model)
    wrapped.eval()

    # Export with dynamic batch/shapes (new dynamo-based exporter)
    print("Exporting to ONNX...")
    dummy_input_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    dummy_mask = torch.ones((2, 16), dtype=torch.long)

    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["vec"],
        dynamic_shapes=[
            {0: "batch", 1: "seq"},
            {0: "batch", 1: "seq"},
        ],
        opset_version=17,
        export_params=True,
        dynamo=True,
    )

    # Copy tokenizer files
    src_dir = Path(model_name) if Path(model_name).exists() else None
    if src_dir and src_dir.is_dir():
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json", "config.json"]:
            src = src_dir / f
            if src.exists():
                shutil.copy2(src, output_dir / f)
    else:
        # Download tokenizer files from HF
        tokenizer.save_pretrained(str(output_dir))

    print(f"Exported to: {output_dir}")
    print(f"  model.onnx: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Validate
    if args.validate:
        print("\nValidating ONNX vs PyTorch...")
        validate(model_name, str(onnx_path), tokenizer)

    # Print config for settings
    config_path = output_dir / "ort_config.json"
    config_path.write_text(json.dumps({
        "model_type": "bge",
        "export_date": str(Path(__file__).stat().st_mtime),
        "onnx_path": str(output_dir.resolve()),
        "opset": 14,
    }, indent=2))
    print(f"\nTo enable ONNX, set in config.yaml:")
    print(f"  embedding.onnx_path: \"{output_dir}\"")


if __name__ == "__main__":
    main()
