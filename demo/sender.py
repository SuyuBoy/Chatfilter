#!/usr/bin/env python3
"""
弹幕发送端 — 读取 CSV，按时间戳顺序向接收端 POST 弹幕
支持倍速播放 (speed=1.0 实时, speed=60 加速60倍)
"""

import asyncio
import csv
import sys
import time
from pathlib import Path
from urllib.parse import quote

import aiohttp


async def send_messages(csv_path: str, server_url: str, speed: float = 1.0,
                        limit: int = 0, batch_size: int = 20):
    """Read CSV and POST messages to server in timestamp order."""
    # Load and sort by timestamp
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row.get("message", "").strip()
            if not msg:
                continue
            ts = float(row.get("timestamp", 0)) / 1000.0
            rows.append({"text": msg, "timestamp": ts})
            if limit and len(rows) >= limit:
                break

    rows.sort(key=lambda r: r["timestamp"])
    if not rows:
        print("No messages found in CSV")
        return

    base_ts = rows[0]["timestamp"]
    real_start = time.time()

    print(f"📤 发送端启动: {len(rows)} 条消息 → {server_url}")
    print(f"   倍速: {speed}x  |  批次大小: {batch_size}")
    print(f"   第一条: {rows[0]['timestamp']:.0f}  |  最后一条: {rows[-1]['timestamp']:.0f}")

    sent = 0
    i = 0
    async with aiohttp.ClientSession() as session:
        while i < len(rows):
            now = time.time()
            elapsed_real = now - real_start
            target_elapsed = (rows[i]["timestamp"] - base_ts) / speed

            if target_elapsed > elapsed_real:
                await asyncio.sleep(target_elapsed - elapsed_real)

            # Send a batch
            batch_end = min(i + batch_size, len(rows))
            tasks = []
            for j in range(i, batch_end):
                r = rows[j]
                url = f"{server_url}/ingest?text={quote(r['text'])}&msg_id={j}"
                tasks.append(session.post(url))
            await asyncio.gather(*tasks)
            sent += batch_end - i
            i = batch_end

            if sent % 100 == 0:
                print(f"  已发送: {sent}/{len(rows)}  ({(sent/len(rows)*100):.0f}%)")

    print(f"✅ 发送完成: {sent} 条")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="弹幕发送端")
    parser.add_argument("--csv", default="弹幕列表.csv", help="CSV 路径")
    parser.add_argument("--server", default="http://localhost:8765", help="接收端 URL")
    parser.add_argument("--speed", type=float, default=1.0, help="倍速 (1=实时, 60=60倍)")
    parser.add_argument("--limit", type=int, default=0, help="限制条数 (0=全部)")
    parser.add_argument("--batch", type=int, default=2, help="批次大小")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV 不存在: {csv_path}")
        sys.exit(1)

    asyncio.run(send_messages(str(csv_path), args.server, args.speed, args.limit, args.batch))
