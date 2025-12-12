"""
SSE (Server-Sent Events) Router
===============================
실시간 서버 푸시 이벤트를 처리하는 라우터입니다.
LLM 스트리밍 응답, 실시간 알림 등에 사용됩니다.

참고 문서:
- SSE 표준: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
- sse-starlette: https://github.com/sysid/sse-starlette
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from typing import AsyncGenerator, Optional
import asyncio
import json
from datetime import datetime
from app.templates import get_template
from app.services.llm_service import llm_service

router = APIRouter(
    prefix="/sse",
    tags=["SSE"],
)

# 연결된 클라이언트 관리
connected_clients: list[asyncio.Queue] = []


@router.get("/monitor", response_class=HTMLResponse)
async def sse_monitor_page():
    """SSE 모니터링 웹 페이지"""
    html = get_template("sse_monitor.html")
    return HTMLResponse(content=html)


async def event_generator(request: Request, queue: asyncio.Queue) -> AsyncGenerator[dict, None]:
    """SSE 이벤트 스트림 생성기"""
    try:
        # 연결 성공 이벤트
        yield {
            "event": "connection",
            "data": json.dumps({"type": "connected", "message": "SSE 연결 성공"}, ensure_ascii=False)
        }

        while True:
            if await request.is_disconnected():
                break

            try:
                data = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield {
                    "event": data.get("event", "message"),
                    "id": data.get("id"),
                    "data": json.dumps(data["data"], ensure_ascii=False)
                }
            except asyncio.TimeoutError:
                # Keep-alive 핑
                yield {
                    "event": "ping",
                    "data": json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()}, ensure_ascii=False)
                }

    except asyncio.CancelledError:
        pass
    finally:
        if queue in connected_clients:
            connected_clients.remove(queue)


@router.get("/stream")
async def sse_stream(request: Request):
    """SSE 스트림 연결 엔드포인트"""
    queue: asyncio.Queue = asyncio.Queue()
    connected_clients.append(queue)

    return EventSourceResponse(event_generator(request, queue))


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
    async def demo_generator() -> AsyncGenerator[dict, None]:
        count = 0
        try:
            yield {
                "event": "start",
                "data": json.dumps({"message": "데모 스트림 시작"}, ensure_ascii=False)
            }

            while True:
                if await request.is_disconnected():
                    break

                count += 1
                yield {
                    "event": "tick",
                    "id": str(count),
                    "data": json.dumps({
                        "count": count,
                        "timestamp": datetime.now().isoformat(),
                        "message": f"이벤트 #{count}"
                    }, ensure_ascii=False)
                }
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass

    return EventSourceResponse(demo_generator())


@router.get("/summary")
async def sse_summary_stream(request: Request):
    """Summary SSE 스트림 - ServerSentEvent 사용, 1초마다 메시지 전송, 5초 후 종료"""
    async def summary_generator() -> AsyncGenerator[ServerSentEvent, None]:
        try:
            # ServerSentEvent 객체로 명시적으로 생성
            yield ServerSentEvent(
                data=json.dumps({"message": "Summary 스트림 시작", "total_seconds": 5}, ensure_ascii=False),
                event="start"
            )

            for i in range(1, 6):
                if await request.is_disconnected():
                    break

                yield ServerSentEvent(
                    data=json.dumps({
                        "second": i,
                        "message": f"메시지 {i}/5",
                        "timestamp": datetime.now().isoformat(),
                        "remaining": 5 - i
                    }, ensure_ascii=False),
                    event="message",
                    id=str(i),
                    retry=3000  # 재연결 대기 시간 (ms)
                )
                await asyncio.sleep(1)

            yield ServerSentEvent(
                data=json.dumps({"message": "Summary 스트림 종료", "status": "completed"}, ensure_ascii=False),
                event="complete"
            )

        except asyncio.CancelledError:
            print('asyncio.CancelledError')

    return EventSourceResponse(summary_generator())


# =============================================================================
# LLM 스트리밍 관련 엔드포인트
# =============================================================================

class ChatRequest(BaseModel):
    """
    채팅 요청 스키마

    Attributes:
        prompt: 사용자 입력 메시지
        system_prompt: AI의 역할을 정의하는 시스템 프롬프트 (선택)
        max_tokens: 생성할 최대 토큰 수 (선택, 기본값: 512)
        temperature: 생성 다양성 0.0~1.0 (선택, 기본값: 0.7)
    """
    prompt: str                                          # 필수: 사용자 입력
    system_prompt: Optional[str] = "You are a helpful assistant."  # 시스템 프롬프트
    max_tokens: Optional[int] = 512                      # 최대 토큰 수
    temperature: Optional[float] = 0.7                   # 생성 다양성


@router.post("/chat")
async def llm_chat_stream(request: Request, chat_request: ChatRequest):
    """
    LLM 채팅 스트리밍 엔드포인트

    Llama 3.2-3B 모델을 사용하여 사용자 프롬프트에 대한 응답을
    실시간 스트리밍으로 반환합니다.

    Request Body:
        - prompt (str): 사용자 입력 메시지
        - system_prompt (str, optional): 시스템 프롬프트
        - max_tokens (int, optional): 최대 토큰 수
        - temperature (float, optional): 생성 다양성

    Returns:
        SSE 스트림으로 다음 이벤트들을 반환:
        - start: 생성 시작 알림
        - token: 생성된 토큰 (실시간)
        - complete: 생성 완료 알림
        - error: 에러 발생 시

    Example:
        ```javascript
        const eventSource = new EventSource('/sse/chat', {
            method: 'POST',
            body: JSON.stringify({ prompt: "Hello!" })
        });
        eventSource.addEventListener('token', (e) => {
            const data = JSON.parse(e.data);
            console.log(data.token);  // 생성된 토큰 출력
        });
        ```
    """
    async def llm_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """
        LLM 응답 스트리밍 제너레이터

        llm_service.generate_stream()에서 생성된 토큰을
        SSE 이벤트로 변환하여 yield합니다.
        """
        # 토큰 카운터 (이벤트 ID용)
        token_count = 0
        # 전체 응답 텍스트 (완료 시 반환용)
        full_response = ""

        try:
            # 모델 로드 확인
            if not llm_service.is_loaded:
                yield ServerSentEvent(
                    data=json.dumps({
                        "error": "모델이 로드되지 않았습니다. 서버 시작 시 모델을 로드해주세요.",
                        "hint": "POST /sse/llm/load 엔드포인트를 호출하세요."
                    }, ensure_ascii=False),
                    event="error"
                )
                return

            # 생성 시작 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "message": "LLM 응답 생성 시작",
                    "prompt": chat_request.prompt,
                    "model": llm_service.model_id,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # LLM에서 토큰 단위로 스트리밍 생성
            # - generate_stream()은 동기 제너레이터이므로
            # - asyncio.to_thread()로 비동기 처리
            for token in llm_service.generate_stream(
                prompt=chat_request.prompt,
                system_prompt=chat_request.system_prompt,
                max_new_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
            ):
                # 클라이언트 연결 확인
                if await request.is_disconnected():
                    break

                # 토큰 카운트 증가
                token_count += 1
                # 전체 응답에 추가
                full_response += token

                # 토큰 이벤트 전송
                yield ServerSentEvent(
                    data=json.dumps({
                        "token": token,           # 생성된 토큰
                        "full_text": full_response,  # 지금까지의 전체 텍스트
                        "token_count": token_count   # 토큰 순번
                    }, ensure_ascii=False),
                    event="token",
                    id=str(token_count)
                )

                # 비동기 이벤트 루프에 제어권 양보
                # - 다른 요청 처리 가능하도록
                await asyncio.sleep(0)

            # 생성 완료 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "message": "LLM 응답 생성 완료",
                    "full_response": full_response,
                    "total_tokens": token_count,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="complete"
            )

        except Exception as e:
            # 에러 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "error": str(e),
                    "type": type(e).__name__
                }, ensure_ascii=False),
                event="error"
            )

        except asyncio.CancelledError:
            # 클라이언트 연결 종료 시
            print(f"LLM 스트리밍 취소됨. 생성된 토큰 수: {token_count}")

    return EventSourceResponse(llm_generator())


@router.post("/llm/load")
async def load_llm_model(
    model_path: Optional[str] = None,
    use_quantization: bool = True
):
    """
    LLM 모델 로드 엔드포인트

    서버 시작 후 모델을 수동으로 로드할 때 사용합니다.
    모델 로드에는 시간이 소요됩니다 (처음 실행 시 다운로드 포함).

    Args:
        model_path: 커스텀 모델 경로 (None이면 기본 Llama 3.2-3B 사용)
        use_quantization: 4bit 양자화 사용 여부 (메모리 절약)

    Returns:
        로드 결과 메시지

    Note:
        - 모델 로드는 한 번만 수행됩니다
        - GPU 메모리가 부족하면 use_quantization=True 권장
        - 커스텀 학습 모델 사용 시 model_path에 경로 지정
    """
    try:
        # 이미 로드된 경우
        if llm_service.is_loaded:
            return {
                "success": True,
                "message": "모델이 이미 로드되어 있습니다.",
                "model_id": llm_service.model_id
            }

        # 모델 로드 (시간 소요)
        llm_service.load_model(
            model_path=model_path,
            use_quantization=use_quantization
        )

        return {
            "success": True,
            "message": "모델 로드 완료",
            "model_id": llm_service.model_id,
            "device": llm_service.device
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }


@router.get("/llm/status")
async def get_llm_status():
    """
    LLM 모델 상태 조회 엔드포인트

    현재 로드된 모델의 상태를 반환합니다.

    Returns:
        - is_loaded: 모델 로드 여부
        - model_id: 모델 ID/경로
        - device: 사용 중인 디바이스 (cuda/cpu)
    """
    return {
        "is_loaded": llm_service.is_loaded,
        "model_id": llm_service.model_id if llm_service.is_loaded else None,
        "device": llm_service.device if llm_service.is_loaded else None
    }
