from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from datetime import datetime
from typing import Callable
import asyncio
import json

# 요청 로그 저장소 (최대 100개)
request_logs: list[dict] = []
MAX_LOGS = 100

# SSE 구독자
request_subscribers: list[asyncio.Queue] = []


async def add_request_log(log: dict):
    """요청 로그 추가 및 구독자에게 브로드캐스트"""
    request_logs.append(log)

    # 최대 개수 유지
    if len(request_logs) > MAX_LOGS:
        request_logs.pop(0)

    # 모든 구독자에게 전송
    for queue in request_subscribers:
        try:
            await queue.put(log)
        except:
            pass


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 모니터 페이지와 SSE 엔드포인트는 로깅 제외
        if request.url.path in ["/monitor", "/monitor/stream", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        start_time = datetime.now()

        # 요청 바디 읽기
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        body = json.loads(body_bytes.decode())
                    except:
                        body = body_bytes.decode('utf-8', errors='ignore')
            except:
                pass

        # 응답 처리
        response = await call_next(request)

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # 로그 생성
        log = {
            "id": len(request_logs) + 1,
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "query": dict(request.query_params),
            "headers": dict(request.headers),
            "body": body,
            "client_ip": request.client.host if request.client else None,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        await add_request_log(log)

        return response
