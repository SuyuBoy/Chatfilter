#!/bin/bash
# ═══════════════════════════════════════════════════════
# Demo 演示 — 一键启动 (服务端 + 发送端)
# ═══════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

cleanup() {
    echo ""
    echo "🛑 正在停止所有服务..."
    pkill -f "demo_terminal_server.py" 2>/dev/null || true
    pkill -f "demo/sender.py" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "══════════════════════════════════════════"
echo "  弹幕语义聚类 — Demo"
echo "══════════════════════════════════════════"

# 清理旧进程
pkill -f "demo_terminal_server.py" 2>/dev/null || true
sleep 1

# 启动服务端
echo "🚀 启动服务端..."
.venv/bin/python scripts/demo_terminal_server.py --server --port 8766 &
SERVER_PID=$!

# 等待服务端就绪
echo "⏳ 等待服务端就绪..."
for i in $(seq 1 60); do
    if curl -sf -o /dev/null http://localhost:8766/state 2>/dev/null; then
        echo "✅ 服务端就绪 (${i}s)"
        break
    fi
    sleep 1
done

if ! curl -sf -o /dev/null http://localhost:8766/state 2>/dev/null; then
    echo "❌ 服务端启动超时"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# 启动发送端
echo "📤 启动弹幕发送端..."
.venv/bin/python demo/sender.py --csv demo/弹幕列表.csv --server http://localhost:8766 --speed 120 --batch 2 & # 60倍速发送弹幕

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ Demo 已启动"
echo "  🖥️  Web UI:  http://localhost:8766"
echo "  ⌨️  终端:    输入 quit 退出"
echo "══════════════════════════════════════════"

wait
