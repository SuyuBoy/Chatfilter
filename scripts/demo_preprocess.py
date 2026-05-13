#!/usr/bin/env python3
"""
预处理管道分步演示脚本
—— 读取弹幕列表.csv，逐步展示效果并保存每步CSV
   输入 "next" 或回车进入下一步
"""

import csv
import os
from pathlib import Path

from src.engine.preprocessor import basic_cleanse
from src.engine.normalizer import Normalizer
from src.engine.cycle_compressor import compress_cycle
from src.engine.simhash_dedup import SimHashHelper
from src.engine.dedup_store import DedupStore
from config.settings import get_settings


# ── 加载配置 ──
settings = get_settings()
normalizer = Normalizer(settings.preprocess.variants_path)
simhash = SimHashHelper(
    high_conf_distance=settings.preprocess.simhash_high_conf_distance,
    candidate_distance=settings.preprocess.simhash_candidate_distance,
    min_text_length=settings.preprocess.simhash_min_text_length,
)
dedup = DedupStore()

# ── 输出目录 ──
OUT_DIR = Path(__file__).parent / "preprocess_output"
OUT_DIR.mkdir(exist_ok=True)


# ── 读取CSV ──
def load_csv(path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row.get("message", "").strip()
            if not msg:
                continue
            rows.append({"message": msg, "date": row.get("date", ""), "timestamp": row.get("timestamp", "")})
            if limit and len(rows) >= limit:
                break
    return rows


def save_csv(filename: str, rows: list[dict], fieldnames: list[str]):
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    size = os.path.getsize(path)
    print(f"  💾 已保存: {filename} ({len(rows)} 行, {size:,} bytes)")


CSV_PATH = Path(__file__).parent / "弹幕列表.csv"
raw_messages = load_csv(str(CSV_PATH))

STEP_NAMES = [
    "① 基础清洗 (basic_cleanse)",
    "② 统一归一化 (normalizer)",
    "③ 循环节压缩 (cycle_compressor)",
    "④ 循环节压缩 (cycle_compressor)",
    "⑤ SimHash 辅助模糊 (simhash_dedup)",
    "⑥ 精确去重+计数 (dedup_store)",
]


def wait_step(step_idx: int, step_name: str) -> bool:
    print(f"\n{'─' * 55}")
    print(f" 步骤 {step_idx + 1}/6: {step_name}")
    print(f"{'─' * 55}")
    print("  Enter/'next' → 下一步  |  'skip' → 跳过剩余  |  'quit' → 退出")
    cmd = input("  >>> ").strip().lower()
    if cmd == "quit":
        return False
    if cmd == "skip":
        return False
    return True


# ── 主流程 ──
def main():
    rows = raw_messages

    print("═" * 55)
    print(f" 📂 弹幕列表.csv — 共 {len(rows)} 条")
    print(f" 📁 输出目录: {OUT_DIR}/")
    print("═" * 55)
    print("  前 20 条预览:")
    for i, r in enumerate(rows[:20]):
        print(f"  [{i:3d}] {r['message'][:45]}")
    if len(rows) > 20:
        print(f"  ... 还有 {len(rows) - 20} 条")

    # ── 步骤 1: 基础清洗 ──
    if not wait_step(0, STEP_NAMES[0]):
        return

    kept: list[dict] = []
    filtered: list[dict] = []
    for r in rows:
        cleaned = basic_cleanse(
            r["message"],
            min_len=settings.preprocess.min_text_length,
            max_len=settings.preprocess.max_text_length,
        )
        if cleaned is None:
            filtered.append({"message": r["message"], "date": r["date"], "timestamp": r["timestamp"],
                             "reason": "filtered"})
        else:
            kept.append({"message": cleaned, "date": r["date"], "timestamp": r["timestamp"],
                         "original": r["message"]})

    save_csv("step1_kept.csv", kept, ["message", "date", "timestamp", "original"])
    save_csv("step1_filtered.csv", filtered, ["message", "date", "timestamp", "reason"])

    print(f"\n  ✅ 保留: {len(kept)} 条  |  ❌ 过滤: {len(filtered)} 条")
    if filtered:
        print(f"  过滤示例 (前5):")
        for r in filtered[:5]:
            print(f"    {r['message'][:40]!r}")
    if kept:
        print(f"  清洗后示例 (变化的前5):")
        shown = 0
        for r in kept:
            if r["message"] != r["original"] and shown < 5:
                print(f"    {r['original'][:30]!r} → {r['message'][:30]!r}")
                shown += 1

    # ── 步骤 2: 归一化 (alias + variant 合并) ──
    if not wait_step(1, STEP_NAMES[1]):
        return

    step2_rows = []
    hits = 0
    for r in kept:
        result = normalizer.normalize(r["message"])
        if result != r["message"]:
            hits += 1
        step2_rows.append({"message": result, "date": r["date"], "timestamp": r["timestamp"],
                           "before": r["message"]})

    save_csv("step2_normalized.csv", step2_rows, ["message", "date", "timestamp", "before"])
    print(f"\n  归一化命中: {hits} 条")
    if hits:
        print(f"  替换示例 (前10):")
        shown = 0
        for r in step2_rows:
            if r["message"] != r["before"] and shown < 10:
                print(f"    {r['before'][:35]!r} → {r['message'][:35]!r}")
                shown += 1

    # ── 步骤 3: 循环节压缩 ──
    if not wait_step(2, STEP_NAMES[2]):
        return

    step3_rows = []
    variant_hits = 0
    for r in step2_rows:
        result = compress_cycle(r["message"])
        if result != r["message"]:
            variant_hits += 1
        step3_rows.append({"message": result, "date": r["date"], "timestamp": r["timestamp"],
                           "before": r["message"]})

    save_csv("step3_variant.csv", step3_rows, ["message", "date", "timestamp", "before"])
    print(f"\n  变体命中: {variant_hits} 条")
    if variant_hits:
        print(f"  替换示例 (前10):")
        shown = 0
        for r in step3_rows:
            if r["message"] != r["before"] and shown < 10:
                print(f"    {r['before'][:35]!r} → {r['message'][:35]!r}")
                shown += 1

    # ── 步骤 4: 循环节压缩 ──
    if not wait_step(3, STEP_NAMES[3]):
        return

    step4_rows = []
    cycle_hits = 0
    for r in step3_rows:
        compressed = compress_cycle(r["message"])
        if compressed != r["message"]:
            cycle_hits += 1
        step4_rows.append({"message": compressed, "date": r["date"], "timestamp": r["timestamp"],
                           "before": r["message"]})

    save_csv("step4_cycle.csv", step4_rows, ["message", "date", "timestamp", "before"])
    print(f"\n  循环节压缩: {cycle_hits} 条")
    if cycle_hits:
        print(f"  压缩示例 (前10):")
        shown = 0
        for r in step4_rows:
            if r["message"] != r["before"] and shown < 10:
                print(f"    {r['before'][:40]!r} → {r['message'][:40]!r}")
                shown += 1

    # ── 步骤 5: SimHash ──
    if not wait_step(4, STEP_NAMES[4]):
        return

    step5_rows = []
    simhash_auto = 0
    simhash_candidate = 0
    for r in step4_rows:
        simhash.add(r["message"])
        canonical, is_auto = simhash.find_canonical(r["message"])
        final = canonical if (is_auto and canonical is not None) else r["message"]
        if is_auto and canonical is not None and canonical != r["message"]:
            simhash_auto += 1
        elif canonical is not None and not is_auto:
            simhash_candidate += 1
        step5_rows.append({"message": final, "date": r["date"], "timestamp": r["timestamp"],
                           "before": r["message"]})

    save_csv("step5_simhash.csv", step5_rows, ["message", "date", "timestamp", "before"])
    print(f"\n  SimHash 自动合并: {simhash_auto}  |  候选: {simhash_candidate}")
    if simhash_auto:
        print(f"  自动合并示例 (前5):")
        shown = 0
        for r in step5_rows:
            if r["message"] != r["before"] and shown < 5:
                print(f"    {r['before'][:35]!r} → {r['message'][:35]!r}")
                shown += 1

    # ── 步骤 6: 精确去重 + 频次计数 ──
    if not wait_step(5, STEP_NAMES[5]):
        return

    freq: dict[str, dict] = {}
    step6_rows = []
    for r in step5_rows:
        is_new = dedup.add(r["message"], count=1, raw_text=r.get("original", r["message"]))
        freq[r["message"]] = freq.get(r["message"], {"count": 0, "first_date": r["date"]})
        freq[r["message"]]["count"] += 1
        step6_rows.append({"message": r["message"], "date": r["date"], "timestamp": r["timestamp"],
                           "is_new_unique": str(is_new)})

    save_csv("step6_dedup.csv", step6_rows, ["message", "date", "timestamp", "is_new_unique"])

    total_valid = len(step5_rows)
    unique_count = len(freq)
    dup_count = total_valid - unique_count

    print(f"\n  总有效: {total_valid} 条  |  唯一文本: {unique_count} 条  |  去重率: {dup_count / max(total_valid, 1) * 100:.1f}%")

    sorted_freq = sorted(freq.items(), key=lambda x: -x[1]["count"])
    print(f"\n  📊 去重后 Top-20:")
    for rank, (text, info) in enumerate(sorted_freq[:20], 1):
        bar = "█" * min(info["count"], 30)
        print(f"  {rank:2d}. [{info['count']:4d}x] {text[:40]:40s} {bar}")

    # 保存频次统计
    freq_rows = [{"rank": i, "text": t, "count": c["count"], "first_date": c["first_date"]}
                 for i, (t, c) in enumerate(sorted_freq, 1)]
    save_csv("step6_frequencies.csv", freq_rows, ["rank", "text", "count", "first_date"])

    # ── 汇总 ──
    print(f"\n{'═' * 55}")
    print(" ✅ 预处理完毕")
    print(f"{'═' * 55}")
    print(f"  输入: {len(rows)}  →  清洗保留: {len(kept)}  →  唯一: {unique_count}")
    print(f"  别名: {alias_hits}  |  变体: {variant_hits}  |  循环: {cycle_hits}  |  SimHash: {simhash_auto}")
    print(f"  所有 CSV 已保存到: {OUT_DIR}/")
    print(f"  文件列表:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == ".csv":
            print(f"    {f.name}  ({os.path.getsize(f):,} bytes)")


if __name__ == "__main__":
    main()
