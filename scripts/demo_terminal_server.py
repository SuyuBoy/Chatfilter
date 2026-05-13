#!/usr/bin/env python3
"""
终端交互式语义聚类 Demo (网络版)
—— 网络层独立 + 键盘实时调粒度 + 2×10 表格

用法:
  .venv/bin/python demo_terminal_server.py              # 交互模式
  .venv/bin/python demo_terminal_server.py --server      # HTTP 服务模式
  .venv/bin/python sender.py --server http://localhost:8766
"""

import asyncio
import json
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.start import PipelineEngine
from src.network.http_server import MiniHttpServer
from config.settings import get_settings

settings = get_settings()

_RECENT_BUFFER = deque(maxlen=200)


async def run_server(port: int):
    """服务模式: HTTP 接收 + 终端 UI + 网页监控"""
    engine = PipelineEngine()
    await engine.initialize()

    html_path = Path(__file__).parent.parent / "static" / "monitor.html"
    html_content = html_path.read_text(encoding="utf-8") if html_path.exists() else "<h1>no html</h1>"

    httpd = MiniHttpServer(port=port)
    httpd.set_html(html_content)

    def get_state():
        state = engine.get_state()
        state["raw_log"] = list(_RECENT_BUFFER)
        state["queue_size"] = httpd.queue.qsize()
        return state
    httpd.set_state_getter(get_state)

    srv = await httpd.start()

    print(f"\033[2J\033[H🌐 HTTP: http://0.0.0.0:{port}  (Ctrl+C 退出)")
    print(f"🌐 网页: http://0.0.0.0:{port}")
    print(engine.render(), end="", flush=True)

    async def consume():
        while True:
            try:
                text = await asyncio.wait_for(httpd.queue.get(), timeout=0.02)
                batch = [text]
                while True:
                    try:
                        batch.append(httpd.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                results = engine.ingest_batch(batch)
                for t, r in zip(batch, results):
                    if not r.get("filtered"):
                        _RECENT_BUFFER.append({
                            "id": engine.total_ingested - len(batch) + len(_RECENT_BUFFER) + 1,
                            "raw": t,
                            "canonical": r.get("canonical", t),
                            "cluster_id": r.get("cluster_id", ""),
                            "slot_id": r.get("slot_id", 0),
                            "cache_hits": r.get("cache_hits", []),
                        })
                import json as _json
                state = get_state()
                state["pulse"] = state["ingested"]
                await httpd.broadcast_sse("state", _json.dumps(state, ensure_ascii=False))
                print(f"\033[2J\033[H" + engine.render(), end="", flush=True)
            except asyncio.TimeoutError:
                pass

    async def admin_loop():
        while True:
            try:
                cfg = await asyncio.wait_for(httpd.admin_queue.get(), timeout=0.5)
                settings.cluster.centroid_threshold = cfg["centroid"]
                settings.cluster.anchor_threshold = cfg["anchor"]
                print(f"\033[2J\033[H" + engine.render(), end="", flush=True)
            except asyncio.TimeoutError:
                pass

    consumer = asyncio.create_task(consume())
    admin = asyncio.create_task(admin_loop())
    try:
        await asyncio.gather(consumer, admin)
    except asyncio.CancelledError:
        pass
    finally:
        await httpd.stop()
        print("\n👋 再见!")


async def run_interactive():
    """交互模式: 手动输入弹幕 + 指令调粒度"""
    engine = PipelineEngine()
    await engine.initialize()

    print(f"\033[2J\033[H")
    print(engine.render(), end="", flush=True)

    print("💡 输入弹幕直接发送 | :+ :上调centroid | :- :下调 | :] :上调anchor | :[ :下调 | quit 退出")

    while True:
        try:
            text = input()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见!"); break
        if not text.strip():
            continue
        if text.strip().lower() == "quit":
            print("\n👋 再见!"); break
        if text.startswith(":+"):
            settings.cluster.centroid_threshold = min(0.99, settings.cluster.centroid_threshold + 0.02)
        elif text.startswith(":-"):
            settings.cluster.centroid_threshold = max(0.01, settings.cluster.centroid_threshold - 0.02)
        elif text.startswith(":]"):
            settings.cluster.anchor_threshold = min(0.99, settings.cluster.anchor_threshold + 0.02)
        elif text.startswith(":["):
            settings.cluster.anchor_threshold = max(0.01, settings.cluster.anchor_threshold - 0.02)
        elif text.startswith(":"):
            parts = text[1:].replace(",", " ").split()
            try:
                if len(parts) >= 1:
                    settings.cluster.centroid_threshold = float(parts[0])
                if len(parts) >= 2:
                    settings.cluster.anchor_threshold = float(parts[1])
            except ValueError:
                print("  格式: :centroid,anchor  如 :0.5,0.8")
        else:
            result = engine.ingest(text.strip())
            # Show timing on each ingest
            if result.get("stage_times"):
                times = " | ".join(f"{k}:{v:.1f}ms" for k, v in result["stage_times"].items())
                print(f"  ⏱ {times}  emb:{result.get('embedding_ms', 0):.1f}ms  cluster:{result.get('cluster_ms', 0):.1f}ms")
        print(f"\033[2J\033[H" + engine.render(), end="", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="终端语义聚类 Demo")
    p.add_argument("--server", action="store_true", help="HTTP 服务模式")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--centroid", type=float, default=None, help="centroid 阈值")
    p.add_argument("--anchor", type=float, default=None, help="anchor 阈值")
    args = p.parse_args()
    if args.centroid is not None:
        settings.cluster.centroid_threshold = args.centroid
    if args.anchor is not None:
        settings.cluster.anchor_threshold = args.anchor
    asyncio.run(run_server(args.port) if args.server else run_interactive())
