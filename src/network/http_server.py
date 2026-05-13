"""
极简 HTTP 服务 + SSE 实时推送 — 纯网络层, 与业务解耦。

设计:
  - asyncio 原生事件循环, 零第三方 HTTP 框架依赖
  - POST /ingest?text=xxx  → 推入 asyncio.Queue, 上游消费者处理
  - GET  /events           → SSE 长连接, 持续推送状态 JSON
  - GET  /state            → 返回完整状态快照 (供轮询兜底)
  - POST /admin/threshold  → 动态调整聚类阈值

SSE 机制:
  - 每个 /events 连接注册到 _sse_clients 列表
  - ingest 处理后调用 broadcast_sse() 向所有连接推送
  - 30s 无消息发 keepalive 保活
  - 客户端断开自动清理
"""

import asyncio
import json
from urllib.parse import parse_qs


class MiniHttpServer:
    """极简 HTTP 服务器 — 接收弹幕 + SSE 推送 + 状态查询 + 阈值调节。

    Attributes:
        queue:       弹幕文本队列 (生产者: HTTP handler; 消费者: 上游业务)
        admin_queue: 管理员指令队列 (阈值调节等)
        _state_getter: 状态回调函数 → dict (由上游注入)
        _sse_clients: 活跃 SSE 客户端队列列表
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5000)
        self.admin_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self._state_getter = None          # callable → dict, 由上游注入
        self._html_content = b""           # 静态 HTML 内容
        self._server = None
        self._sse_clients: list[asyncio.Queue] = []

    def set_state_getter(self, fn):
        """注入状态查询函数: fn() → dict。"""
        self._state_getter = fn

    def set_html(self, html: str):
        """注入 HTML 页面内容 (monitor.html)。"""
        self._html_content = html.encode() if isinstance(html, str) else html

    async def broadcast_sse(self, event: str, data: str):
        """向所有 SSE 客户端推送事件。死连接自动清理。"""
        dead = []
        for q in self._sse_clients:
            try:
                q.put_nowait((event, data))
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._sse_clients.remove(q)

    async def _handle_sse(self, writer):
        """SSE 长连接处理: 持续推送状态直到客户端断开。

        连接建立时立即发送一次完整状态。
        每 30s 发 keepalive 注释行保活。
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._sse_clients.append(q)

        # SSE 响应头
        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(headers.encode())
        await writer.drain()

        # 连接时立刻发送完整状态
        if self._state_getter:
            state = self._state_getter()
            body = json.dumps(state, ensure_ascii=False)
            writer.write(f"event: state\ndata: {body}\n\n".encode())
            await writer.drain()

        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(q.get(), timeout=30)
                    msg = f"event: {event}\ndata: {data}\n\n"
                    writer.write(msg.encode())
                    await writer.drain()
                except asyncio.TimeoutError:
                    # keepalive: SSE 注释行 (以 : 开头, 客户端忽略)
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass  # 客户端断开, 正常退出
        finally:
            if q in self._sse_clients:
                self._sse_clients.remove(q)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle(self, reader, writer):
        """HTTP 请求分发 — 每个连接一个协程。"""
        try:
            # 读取请求行
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            if not line:
                return
            parts = line.decode(errors="replace").split()
            if len(parts) < 2:
                return
            method, path = parts[0].upper(), parts[1]

            # 读取 headers
            content_length = 0
            while True:
                l = await asyncio.wait_for(reader.readline(), timeout=2)
                if not l or l.strip() == b"":
                    break
                if l.lower().startswith(b"content-length:"):
                    try:
                        content_length = int(l.split(b":", 1)[1].strip())
                    except ValueError:
                        pass

            # 读取 body (如有)
            if content_length > 0:
                await asyncio.wait_for(reader.readexactly(content_length), timeout=5)

            # ── 路由分发 ──
            if method == "GET":
                if path.startswith("/events"):
                    # SSE 长连接 — 不走常规回复, 直接进入推送循环
                    await self._handle_sse(writer)
                    return

                if path in ("/", "/index.html"):
                    body = self._html_content or b"<h1>no html</h1>"
                    rsp = f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body

                elif path.startswith("/state"):
                    # 状态 JSON (兜底轮询)
                    state = self._state_getter() if self._state_getter else {}
                    body = json.dumps(state, ensure_ascii=False).encode()
                    rsp = f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body

                else:
                    rsp = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 21\r\n\r\nPOST /ingest?text=xxx"

            elif method == "POST":
                # 解析 query string
                qs = path.split("?", 1)[1] if "?" in path else ""
                params = parse_qs(qs)

                if path.startswith("/ingest"):
                    # 摄入弹幕 → 推入队列
                    text = params.get("text", [""])[0]
                    if text:
                        await self.queue.put(text)
                    rsp = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok"

                elif path.startswith("/admin/threshold"):
                    # 动态调整阈值
                    ct = params.get("centroid", [None])[0]
                    at = params.get("anchor", [None])[0]
                    if ct and at:
                        await self.admin_queue.put({
                            "centroid": float(ct), "anchor": float(at)
                        })
                    rsp = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok"

                else:
                    rsp = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 21\r\n\r\nPOST /ingest?text=xxx"
            else:
                rsp = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 21\r\n\r\nPOST /ingest?text=xxx"

            writer.write(rsp)
            await writer.drain()

        except Exception:
            pass  # 连接异常, 静默关闭
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        """启动 HTTP 服务器。"""
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        return self._server

    async def stop(self):
        """停止 HTTP 服务器。"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
