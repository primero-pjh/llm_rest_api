from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from typing import Optional
import json

router = APIRouter(tags=["Terminal"])


TERMINAL_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Terminal</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .terminal-header {
            background-color: #323233;
            padding: 8px 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .terminal-header .dots {
            display: flex;
            gap: 6px;
        }
        .terminal-header .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot.red { background-color: #ff5f56; }
        .dot.yellow { background-color: #ffbd2e; }
        .dot.green { background-color: #27ca40; }
        .terminal-header .title {
            flex: 1;
            text-align: center;
            color: #888;
            font-size: 13px;
        }
        .terminal-body {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
        }
        .output-line {
            margin-bottom: 4px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .output-line.request { color: #569cd6; }
        .output-line.response { color: #4ec9b0; }
        .output-line.error { color: #f44747; }
        .output-line.info { color: #dcdcaa; }
        .output-line.success { color: #6a9955; }
        .input-line {
            display: flex;
            align-items: center;
            margin-top: 8px;
        }
        .prompt {
            color: #6a9955;
            margin-right: 8px;
        }
        .input-field {
            flex: 1;
            background: transparent;
            border: none;
            color: #d4d4d4;
            font-family: inherit;
            font-size: 14px;
            outline: none;
        }
        .method-selector {
            background: #3c3c3c;
            color: #d4d4d4;
            border: 1px solid #555;
            padding: 4px 8px;
            margin-right: 8px;
            font-family: inherit;
            border-radius: 4px;
        }
        .help-text {
            color: #888;
            font-size: 12px;
            margin-top: 16px;
            border-top: 1px solid #333;
            padding-top: 8px;
        }
    </style>
</head>
<body>
    <div class="terminal-header">
        <div class="dots">
            <span class="dot red"></span>
            <span class="dot yellow"></span>
            <span class="dot green"></span>
        </div>
        <span class="title">Web Terminal - FastAPI</span>
    </div>
    <div class="terminal-body" id="terminal">
        <div class="output-line info">Welcome to Web Terminal!</div>
        <div class="output-line info">Type a URL path (e.g., /api/v1/items) and press Enter to send a request.</div>
        <div class="output-line info">Commands: help, clear, history</div>
        <div class="output-line info">─────────────────────────────────────</div>
        <div id="request-info"></div>
        <div id="output"></div>
        <div class="input-line">
            <span class="prompt">$</span>
            <select class="method-selector" id="method">
                <option value="GET">GET</option>
                <option value="POST">POST</option>
                <option value="PUT">PUT</option>
                <option value="DELETE">DELETE</option>
            </select>
            <input type="text" class="input-field" id="input" placeholder="Enter path or command..." autofocus>
        </div>
    </div>
    <div class="help-text" style="padding: 8px 16px;">
        Examples: /health, /api/v1/items, POST /api/v1/items {"name":"test","price":100}
    </div>

    <script>
        const terminal = document.getElementById('terminal');
        const output = document.getElementById('output');
        const input = document.getElementById('input');
        const methodSelect = document.getElementById('method');
        const requestInfo = document.getElementById('request-info');
        const history = [];
        let historyIndex = -1;

        // 현재 요청 정보 표시
        const currentRequest = REQUEST_INFO_PLACEHOLDER;
        if (currentRequest) {
            requestInfo.innerHTML = `
                <div class="output-line info">─── Current Request Info ───</div>
                <div class="output-line request">Method: ${currentRequest.method}</div>
                <div class="output-line request">Path: ${currentRequest.path}</div>
                <div class="output-line request">Query: ${JSON.stringify(currentRequest.query)}</div>
                <div class="output-line request">Headers: ${JSON.stringify(currentRequest.headers, null, 2)}</div>
                <div class="output-line info">────────────────────────────</div>
            `;
        }

        function appendOutput(text, className = '') {
            const line = document.createElement('div');
            line.className = 'output-line ' + className;
            line.textContent = text;
            output.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        }

        function appendJson(obj, className = '') {
            const line = document.createElement('div');
            line.className = 'output-line ' + className;
            line.innerHTML = '<pre>' + JSON.stringify(obj, null, 2) + '</pre>';
            output.appendChild(line);
            terminal.scrollTop = terminal.scrollHeight;
        }

        async function sendRequest(method, path, body = null) {
            appendOutput(`>>> ${method} ${path}`, 'request');

            try {
                const options = {
                    method: method,
                    headers: { 'Content-Type': 'application/json' }
                };

                if (body && (method === 'POST' || method === 'PUT')) {
                    options.body = JSON.stringify(body);
                }

                const response = await fetch(path, options);
                const data = await response.json();

                appendOutput(`<<< Status: ${response.status}`, response.ok ? 'success' : 'error');
                appendJson(data, 'response');
            } catch (error) {
                appendOutput(`Error: ${error.message}`, 'error');
            }
        }

        function processCommand(cmd) {
            const trimmed = cmd.trim();
            if (!trimmed) return;

            history.push(trimmed);
            historyIndex = history.length;

            if (trimmed === 'clear') {
                output.innerHTML = '';
                return;
            }
            if (trimmed === 'help') {
                appendOutput('Available commands:', 'info');
                appendOutput('  clear     - Clear terminal', 'info');
                appendOutput('  history   - Show command history', 'info');
                appendOutput('  help      - Show this help', 'info');
                appendOutput('', 'info');
                appendOutput('Request format:', 'info');
                appendOutput('  /path                    - GET request', 'info');
                appendOutput('  /path {"key":"value"}    - POST with JSON body', 'info');
                return;
            }
            if (trimmed === 'history') {
                history.forEach((h, i) => appendOutput(`${i + 1}: ${h}`, 'info'));
                return;
            }

            // Parse request
            const method = methodSelect.value;
            let path = trimmed;
            let body = null;

            // Check if there's a JSON body
            const jsonMatch = trimmed.match(/^(\\/[^\\s]*)?\\s*(\\{.*\\})$/);
            if (jsonMatch) {
                path = jsonMatch[1] || '/';
                try {
                    body = JSON.parse(jsonMatch[2]);
                } catch (e) {
                    appendOutput('Invalid JSON body', 'error');
                    return;
                }
            }

            if (!path.startsWith('/')) {
                path = '/' + path;
            }

            sendRequest(method, path, body);
        }

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                processCommand(input.value);
                input.value = '';
            } else if (e.key === 'ArrowUp') {
                if (historyIndex > 0) {
                    historyIndex--;
                    input.value = history[historyIndex];
                }
                e.preventDefault();
            } else if (e.key === 'ArrowDown') {
                if (historyIndex < history.length - 1) {
                    historyIndex++;
                    input.value = history[historyIndex];
                } else {
                    historyIndex = history.length;
                    input.value = '';
                }
                e.preventDefault();
            }
        });
    </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def terminal_page(request: Request):
    """터미널 웹 페이지 - 요청 정보를 표시하고 API 테스트 가능"""
    # 요청 정보 수집
    request_info = {
        "method": request.method,
        "path": str(request.url.path),
        "query": dict(request.query_params),
        "headers": dict(request.headers),
        "client": request.client.host if request.client else None,
    }

    # HTML에 요청 정보 주입
    html = TERMINAL_HTML.replace(
        "REQUEST_INFO_PLACEHOLDER",
        json.dumps(request_info)
    )

    return HTMLResponse(content=html)


@router.api_route("/echo", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def echo_request(request: Request):
    """모든 요청 정보를 그대로 반환"""
    body = None
    try:
        body = await request.json()
    except:
        body_bytes = await request.body()
        if body_bytes:
            body = body_bytes.decode('utf-8', errors='ignore')

    return {
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "headers": dict(request.headers),
        "body": body,
        "client": {
            "host": request.client.host if request.client else None,
            "port": request.client.port if request.client else None,
        },
        "cookies": dict(request.cookies),
    }
