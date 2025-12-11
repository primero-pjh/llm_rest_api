from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional
import asyncio
import json
from datetime import datetime

router = APIRouter(
    prefix="/sse",
    tags=["SSE"],
)

# 연결된 클라이언트 관리
connected_clients: list[asyncio.Queue] = []


async def event_generator(request: Request, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
    """SSE 이벤트 스트림 생성기"""
    try:
        # 연결 성공 이벤트
        yield format_sse({"type": "connected", "message": "SSE 연결 성공"}, event="connection")

        while True:
            # 클라이언트 연결 해제 확인
            if await request.is_disconnected():
                break

            try:
                # 큐에서 메시지 대기 (타임아웃 30초)
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield format_sse(data["data"], event=data.get("event"), id=data.get("id"))
            except asyncio.TimeoutError:
                # Keep-alive 핑
                yield format_sse({"type": "ping", "timestamp": datetime.now().isoformat()}, event="ping")

    except asyncio.CancelledError:
        pass
    finally:
        # 클라이언트 연결 해제 시 큐 제거
        if queue in connected_clients:
            connected_clients.remove(queue)


def format_sse(data: dict, event: Optional[str] = None, id: Optional[str] = None) -> str:
    """SSE 포맷으로 데이터 변환"""
    message = ""
    if id:
        message += f"id: {id}\n"
    if event:
        message += f"event: {event}\n"
    message += f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    return message


@router.get("/stream")
async def sse_stream(request: Request):
    """SSE 스트림 연결 엔드포인트"""
    queue: asyncio.Queue = asyncio.Queue()
    connected_clients.append(queue)

    return StreamingResponse(
        event_generator(request, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        }
    )


@router.post("/broadcast")
async def broadcast_message(message: dict):
    """모든 연결된 클라이언트에게 메시지 브로드캐스트"""
    if not connected_clients:
        return {"success": False, "message": "연결된 클라이언트가 없습니다", "clients": 0}

    event_data = {
        "event": message.get("event", "message"),
        "data": {
            "content": message.get("content", ""),
            "timestamp": datetime.now().isoformat(),
        },
        "id": message.get("id"),
    }

    for queue in connected_clients:
        await queue.put(event_data)

    return {
        "success": True,
        "message": "메시지 전송 완료",
        "clients": len(connected_clients)
    }


@router.post("/send/{client_index}")
async def send_to_client(client_index: int, message: dict):
    """특정 클라이언트에게 메시지 전송"""
    if client_index < 0 or client_index >= len(connected_clients):
        return {"success": False, "message": "유효하지 않은 클라이언트 인덱스"}

    event_data = {
        "event": message.get("event", "message"),
        "data": {
            "content": message.get("content", ""),
            "timestamp": datetime.now().isoformat(),
        },
        "id": message.get("id"),
    }

    await connected_clients[client_index].put(event_data)
    return {"success": True, "message": f"클라이언트 {client_index}에게 전송 완료"}


@router.get("/clients")
async def get_connected_clients():
    """연결된 클라이언트 수 조회"""
    return {"connected_clients": len(connected_clients)}


@router.get("/demo")
async def sse_demo_stream(request: Request):
    """데모용 SSE 스트림 - 1초마다 시간 전송"""
    async def demo_generator() -> AsyncGenerator[str, None]:
        count = 0
        try:
            yield format_sse({"message": "데모 스트림 시작"}, event="start")

            while True:
                if await request.is_disconnected():
                    break

                count += 1
                data = {
                    "count": count,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"이벤트 #{count}"
                }
                yield format_sse(data, event="tick", id=str(count))
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        demo_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
