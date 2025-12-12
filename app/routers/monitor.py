from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import AsyncGenerator
import asyncio
import json
from app.core.request_logger import request_logs, request_subscribers

router = APIRouter(tags=["Monitor"])

MONITOR_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Request Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0d1117;
            color: #c9d1d9;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #161b22;
            padding: 16px 24px;
            border-bottom: 1px solid #30363d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 20px;
            color: #58a6ff;
        }
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #f85149;
        }
        .status-dot.connected { background: #3fb950; }
        .controls {
            display: flex;
            gap: 12px;
        }
        .btn {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn:hover { background: #30363d; }
        .btn.danger { border-color: #f85149; color: #f85149; }
        .main {
            flex: 1;
            display: flex;
            overflow: hidden;
        }
        .request-list {
            width: 400px;
            border-right: 1px solid #30363d;
            overflow-y: auto;
        }
        .request-item {
            padding: 12px 16px;
            border-bottom: 1px solid #21262d;
            cursor: pointer;
            transition: background 0.2s;
        }
        .request-item:hover { background: #161b22; }
        .request-item.selected { background: #1f6feb33; border-left: 3px solid #58a6ff; }
        .request-item .top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        .method {
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        .method.GET { background: #238636; color: white; }
        .method.POST { background: #1f6feb; color: white; }
        .method.PUT { background: #9e6a03; color: white; }
        .method.DELETE { background: #da3633; color: white; }
        .method.PATCH { background: #8957e5; color: white; }
        .status-code {
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .status-code.success { background: #238636; }
        .status-code.error { background: #da3633; }
        .path {
            font-family: monospace;
            font-size: 13px;
            color: #8b949e;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .time {
            font-size: 11px;
            color: #6e7681;
            margin-top: 4px;
        }
        .detail-panel {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .detail-section {
            margin-bottom: 20px;
        }
        .detail-section h3 {
            color: #58a6ff;
            font-size: 14px;
            margin-bottom: 8px;
            padding-bottom: 4px;
            border-bottom: 1px solid #30363d;
        }
        .detail-content {
            background: #161b22;
            border-radius: 6px;
            padding: 12px;
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 300px;
            overflow-y: auto;
        }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #6e7681;
        }
        .empty-state svg {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.5;
        }
        .badge {
            background: #30363d;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Request Monitor</h1>
        <div class="status">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">Connecting...</span>
            <span class="badge" id="requestCount">0</span>
        </div>
        <div class="controls">
            <button class="btn" onclick="clearLogs()">Clear</button>
            <button class="btn" id="pauseBtn" onclick="togglePause()">Pause</button>
        </div>
    </div>
    <div class="main">
        <div class="request-list" id="requestList">
            <div class="empty-state" id="emptyState">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p>Waiting for requests...</p>
            </div>
        </div>
        <div class="detail-panel" id="detailPanel">
            <div class="empty-state">
                <p>Select a request to view details</p>
            </div>
        </div>
    </div>

    <script>
        let requests = [];
        let selectedId = null;
        let paused = false;
        let eventSource = null;

        function connect() {
            eventSource = new EventSource('/monitor/stream');

            eventSource.onopen = () => {
                document.getElementById('statusDot').classList.add('connected');
                document.getElementById('statusText').textContent = 'Connected';
            };

            eventSource.onerror = () => {
                document.getElementById('statusDot').classList.remove('connected');
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };

            eventSource.addEventListener('request', (e) => {
                if (paused) return;
                const data = JSON.parse(e.data);
                addRequest(data);
            });

            eventSource.addEventListener('history', (e) => {
                const data = JSON.parse(e.data);
                requests = data;
                renderList();
            });
        }

        function addRequest(req) {
            requests.unshift(req);
            if (requests.length > 100) requests.pop();
            renderList();
        }

        function renderList() {
            const list = document.getElementById('requestList');
            const empty = document.getElementById('emptyState');
            document.getElementById('requestCount').textContent = requests.length;

            if (requests.length === 0) {
                empty.style.display = 'flex';
                return;
            }
            empty.style.display = 'none';

            list.innerHTML = requests.map(req => `
                <div class="request-item ${selectedId === req.id ? 'selected' : ''}" onclick="selectRequest(${req.id})">
                    <div class="top">
                        <span class="method ${req.method}">${req.method}</span>
                        <span class="status-code ${req.status_code < 400 ? 'success' : 'error'}">${req.status_code}</span>
                    </div>
                    <div class="path">${req.path}</div>
                    <div class="time">${new Date(req.timestamp).toLocaleTimeString()} · ${req.duration_ms}ms · ${req.client_ip}</div>
                </div>
            `).join('');
        }

        function selectRequest(id) {
            selectedId = id;
            renderList();
            const req = requests.find(r => r.id === id);
            if (!req) return;

            document.getElementById('detailPanel').innerHTML = `
                <div class="detail-section">
                    <h3>General</h3>
                    <div class="detail-content">Method: ${req.method}
Path: ${req.path}
Status: ${req.status_code}
Duration: ${req.duration_ms}ms
Client IP: ${req.client_ip}
Time: ${req.timestamp}</div>
                </div>
                <div class="detail-section">
                    <h3>Query Parameters</h3>
                    <div class="detail-content">${JSON.stringify(req.query, null, 2) || '{}'}</div>
                </div>
                <div class="detail-section">
                    <h3>Request Headers</h3>
                    <div class="detail-content">${JSON.stringify(req.headers, null, 2)}</div>
                </div>
                <div class="detail-section">
                    <h3>Request Body</h3>
                    <div class="detail-content">${req.body ? JSON.stringify(req.body, null, 2) : '(empty)'}</div>
                </div>
            `;
        }

        function clearLogs() {
            requests = [];
            selectedId = null;
            renderList();
            document.getElementById('detailPanel').innerHTML = '<div class="empty-state"><p>Select a request to view details</p></div>';
        }

        function togglePause() {
            paused = !paused;
            document.getElementById('pauseBtn').textContent = paused ? 'Resume' : 'Pause';
        }

        connect();
    </script>
</body>
</html>
"""


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page():
    """요청 모니터링 웹 페이지"""
    return HTMLResponse(content=MONITOR_HTML)


@router.get("/monitor/stream")
async def monitor_stream(request: Request):
    """요청 로그 SSE 스트림"""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        request_subscribers.append(queue)

        try:
            # 기존 로그 전송
            yield f"event: history\ndata: {json.dumps(list(reversed(request_logs)))}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                try:
                    log = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: request\ndata: {json.dumps(log)}\n\n"
                except asyncio.TimeoutError:
                    yield f"event: ping\ndata: {{}}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            if queue in request_subscribers:
                request_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/monitor/logs")
async def get_logs():
    """저장된 요청 로그 조회"""
    return {"logs": request_logs, "count": len(request_logs)}
