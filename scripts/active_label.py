#!/usr/bin/env python3
"""
交互式主动标注: 找模型困惑的文本对，人工标注同义/不同义，写入 variants.yaml。
用法: .venv/bin/python scripts/active_label.py [--pairs 20] [--dry-run]
"""

import csv
import sys
import random
import yaml
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.preprocessor import basic_cleanse
from src.engine.alias_normalizer import AliasNormalizer
from src.engine.variant_normalizer import VariantNormalizer
from src.engine.cycle_compressor import compress_cycle
from sentence_transformers import SentenceTransformer
try:
    from pypinyin import pinyin, Style
except ImportError:
    pinyin = None

PROJECT = Path(__file__).parent.parent
CSV_PATH = PROJECT / "demo" / "弹幕列表.csv"
VARIANTS_PATH = PROJECT / "config" / "variants.yaml"
MODEL_PATH = str(PROJECT / "models" / "bge-small-zh-v1.5")
FT_MODEL_PATH = PROJECT / "models" / "bge-small-zh-v1.5-ft"
EMB_CONFUSION_RANGE = (0.50, 0.75)   # embedding 困惑区间
PINYIN_EDIT_MAX = 3                  # 拼音编辑距离上限
PINYIN_EMB_MAX = 0.50                # 拼音近但嵌入远的上限
MAX_CANDIDATES = 2000                # 候选对上限 (避免 O(n²) 爆炸)


def pinyin_edit_distance(a: str, b: str) -> int:
    """Levenshtein on pinyin sequences."""
    if pinyin is None:
        return 999
    pa = "".join([item[0] for item in pinyin(a, style=Style.TONE3)])
    pb = "".join([item[0] for item in pinyin(b, style=Style.TONE3)])
    m, n = len(pa), len(pb)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if pa[i-1] == pb[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def load_existing_pairs() -> set:
    """已存在于 variants.yaml 的 pairs (含方向)。"""
    existing = set()
    if VARIANTS_PATH.exists():
        with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("variants", {})
        for canonical, vlist in raw.items():
            existing.add(canonical)
            for v in vlist:
                existing.add(v)
    return existing


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20, help="标注题数")
    ap.add_argument("--dry-run", action="store_true", help="仅生成候选不交互")
    args = ap.parse_args()

    # 1. 加载 + 预处理
    print("Loading & preprocessing CSV...")
    alias_norm = AliasNormalizer(str(VARIANTS_PATH))
    variant_norm = VariantNormalizer(str(VARIANTS_PATH))
    canonical_freq: Counter[str] = Counter()

    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            msg = row.get("message", "").strip()
            if not msg:
                continue
            cleaned = basic_cleanse(msg, min_len=1, max_len=128)
            if cleaned is None:
                continue
            text = alias_norm.normalize(cleaned)
            text = variant_norm.normalize(text)
            text = compress_cycle(text)
            canonical_freq[text] += 1

    # 过滤: 3-16 字, 至少出现 2 次
    texts = sorted([t for t, c in canonical_freq.items() if 3 <= len(t) <= 16 and c >= 2])
    print(f"  unique canonicals (filtered): {len(texts)}")

    # 2. BGE 编码
    model_path = str(FT_MODEL_PATH) if FT_MODEL_PATH.exists() else MODEL_PATH
    print(f"Encoding with {model_path}...")
    model = SentenceTransformer(model_path)
    embs = model.encode(texts, show_progress_bar=True)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    # 3. 找两类困惑对
    existing = load_existing_pairs()
    emb_confused = []    # (a, b, sim)
    pinyin_confused = [] # (a, b, sim, py_dist)

    n = len(texts)
    indices = list(range(n))
    random.shuffle(indices)
    sample_n = min(MAX_CANDIDATES, n)
    sampled = sorted(indices[:sample_n])

    print(f"Finding confusion pairs (sampling {sample_n}/{n})...")
    for i_idx, i in enumerate(sampled):
        if texts[i] in existing:
            continue
        for j in sampled[i_idx+1:]:
            if texts[j] in existing:
                continue
            sim = float(np.dot(embs[i], embs[j]))
            if EMB_CONFUSION_RANGE[0] <= sim <= EMB_CONFUSION_RANGE[1]:
                emb_confused.append((texts[i], texts[j], sim))
            elif sim < PINYIN_EMB_MAX:
                py_dist = pinyin_edit_distance(texts[i], texts[j])
                if py_dist <= PINYIN_EDIT_MAX:
                    pinyin_confused.append((texts[i], texts[j], sim, py_dist))

    print(f"  embedding-confused: {len(emb_confused)}")
    print(f"  pinyin-confused:    {len(pinyin_confused)}")

    # 合并抽样
    random.shuffle(emb_confused)
    random.shuffle(pinyin_confused)
    half = args.pairs // 2
    candidates = emb_confused[:half] + pinyin_confused[:half]
    random.shuffle(candidates)
    candidates = candidates[:args.pairs]

    if args.dry_run:
        print(f"\n=== Dry run: {len(candidates)} candidates ===")
        for i, c in enumerate(candidates):
            if len(c) == 3:
                a, b, sim = c
                tag = "emb"
            else:
                a, b, sim, pd = c
                tag = f"pinyin(dist={pd})"
            print(f"  [{i+1:2d}] {a:20s} vs {b:20s}  sim={sim:.3f}  [{tag}]")
        return

    # 4. 交互标注
    confirmed: list[tuple[str, str]] = []
    skipped = 0
    for idx, c in enumerate(candidates):
        if len(c) == 3:
            a, b, sim = c
            tag = f"嵌入困惑 · sim={sim:.3f}"
        else:
            a, b, sim, pd = c
            tag = f"谐音困惑 · sim={sim:.3f} · py_dist={pd}"

        print(f"\n{'═'*55}")
        print(f" 主动标注 · 第 {idx+1}/{len(candidates)} 题  [{tag}]")
        print(f"{'═'*55}")
        print(f"  A: {a}")
        print(f"  B: {b}")
        print(f"{'─'*55}")
        print(f"  [Y] 同义并记录  [N] 不同义  [S] 跳过  [Q] 退出")
        while True:
            choice = input("  >>> ").strip().lower()
            if choice == 'y':
                confirmed.append((a, b))
                print(f"  ✅ 已记录: {a} ↔ {b}")
                break
            elif choice == 'n':
                break
            elif choice == 's':
                skipped += 1
                break
            elif choice == 'q':
                print(f"\n  提前退出。已确认: {len(confirmed)}")
                break
            else:
                print("  [Y/N/S/Q]")
        if choice == 'q':
            break

    # 5. 写入 variants.yaml
    if confirmed:
        print(f"\n Writing {len(confirmed)} pairs to {VARIANTS_PATH}...")
        with open(VARIANTS_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        new_entries = ""
        for a, b in confirmed:
            # 按频率高的做 canonical
            ca, cb = canonical_freq.get(a, 0), canonical_freq.get(b, 0)
            if ca >= cb:
                canonical, variant = a, b
            else:
                canonical, variant = b, a
            new_entries += f"  {canonical}: [{variant}]\n"

        if "variants:" not in content:
            content += "\nvariants:\n"

        # 追加到 variants 块末尾
        content = content.rstrip() + "\n" + new_entries + "\n"

        with open(VARIANTS_PATH, "w", encoding="utf-8") as f:
            f.write(content)

        print(f" Done. 已记录 {len(confirmed)} 对 (跳过 {skipped}).")
        print(f" 下一步: .venv/bin/python scripts/finetune_bge.py")
    else:
        print("\n 没有确认的标注。")

if __name__ == "__main__":
    main()
