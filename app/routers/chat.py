"""
Chat Router (SSE)
=================
LLM 챗봇 대화를 처리하는 라우터입니다.
SSE를 통한 실시간 스트리밍 응답을 지원합니다.
"""

from fastapi import APIRouter, Request, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, List
import asyncio
import json
from datetime import datetime

from app.core.database import get_db, AsyncSessionLocal
from app.models.chat import ChatSession, ChatMessage, MessageRole, ToolLog, ToolCallStatus
from app.services.llm_service import llm_service
from app.services.agent_service import AgentService, AgentState, AgentContext
from app.mcp.client import mcp_client

router = APIRouter(
    prefix="/sse/chat",
    tags=["Chat"],
)


# ============== Pydantic Models ==============

class SessionCreate(BaseModel):
    """세션 생성 요청 모델"""
    title: Optional[str] = None
    system_prompt: Optional[str] = "You are a helpful assistant."
    user_id: Optional[int] = None


class SessionResponse(BaseModel):
    """세션 응답 모델"""
    id: int
    title: Optional[str]
    total_messages: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """메시지 응답 모델"""
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ToolLogResponse(BaseModel):
    """도구 호출 로그 응답 모델"""
    id: int
    session_id: Optional[int]
    tool_name: str
    tool_arguments: Optional[dict]
    status: str
    result_content: Optional[str]
    error_message: Optional[str]
    execution_time_ms: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============== Session Endpoints ==============

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    새로운 채팅 세션을 생성합니다.

    채팅 세션은 사용자와 AI 간의 대화를 논리적으로 그룹화하는 단위입니다.
    세션을 생성하면 해당 세션 내에서 주고받은 모든 메시지가 저장되어
    대화 히스토리를 관리할 수 있습니다.

    사용 시나리오:
    - 새로운 주제로 대화를 시작할 때
    - 이전 대화와 분리된 새 대화가 필요할 때
    - 특정 사용자의 대화 기록을 관리하고 싶을 때

    Request Body:
        title: 세션 제목. 생략 시 첫 메시지 내용으로 자동 설정됩니다.
        system_prompt: AI의 역할을 정의하는 시스템 프롬프트. 기본값: "You are a helpful assistant."
        user_id: 세션을 소유할 사용자 ID. 생략 가능하며, 익명 세션도 지원합니다.

    Returns:
        SessionResponse: 생성된 세션 정보 (id, title, total_messages, created_at, updated_at)

    Raises:
        500: 데이터베이스 오류 발생 시
    """
    session = ChatSession(
        title=request.title,
        system_prompt=request.system_prompt,
        user_id=request.user_id,
        model_name=llm_service.model_id if llm_service.is_loaded else None,
        tokenizer_name=llm_service.model_id if llm_service.is_loaded else None  # 토크나이저는 보통 모델명과 동일
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return session


@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    user_id: Optional[int] = Query(
        None,
        description="특정 사용자의 세션만 조회할 때 사용합니다. 생략하면 모든 세션을 조회합니다."
    ),
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="한 번에 조회할 세션 수입니다. 범위: 1-100. 기본값: 20. 페이지네이션에 사용합니다."
    ),
    offset: int = Query(
        0,
        ge=0,
        description="건너뛸 세션 수입니다. 기본값: 0. 페이지네이션에 사용합니다. 예: offset=20이면 21번째 세션부터 조회합니다."
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    채팅 세션 목록을 조회합니다.

    저장된 모든 채팅 세션을 최신 업데이트 순으로 조회합니다.
    페이지네이션을 지원하여 대량의 세션도 효율적으로 조회할 수 있습니다.

    사용 시나리오:
    - 사용자의 이전 대화 목록을 표시할 때
    - 특정 사용자의 대화 기록을 조회할 때
    - 대화 히스토리 관리 화면에서 세션 목록을 불러올 때

    Returns:
        List[SessionResponse]: 세션 목록. 각 세션에는 id, title, total_messages, created_at, updated_at가 포함됩니다.
        결과는 updated_at 기준 내림차순(최신순)으로 정렬됩니다.
    """
    query = select(ChatSession).order_by(ChatSession.updated_at.desc())

    if user_id:
        query = query.where(ChatSession.user_id == user_id)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    sessions = result.scalars().all()

    return sessions


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    특정 채팅 세션의 상세 정보를 조회합니다.

    세션 ID를 통해 단일 세션의 상세 정보를 가져옵니다.
    세션의 메타데이터(제목, 총 메시지 수, 생성/수정 시간)를 확인할 수 있습니다.

    사용 시나리오:
    - 특정 대화의 상세 정보를 확인할 때
    - 세션이 존재하는지 확인할 때
    - 대화를 이어가기 전에 세션 정보를 확인할 때

    Path Parameters:
        session_id: 조회할 세션의 고유 ID (정수)

    Returns:
        SessionResponse: 세션 상세 정보 (id, title, total_messages, created_at, updated_at)

    Raises:
        404: 해당 ID의 세션이 존재하지 않을 때
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    채팅 세션을 삭제합니다.

    지정된 세션과 해당 세션에 속한 모든 메시지를 영구적으로 삭제합니다.
    이 작업은 되돌릴 수 없으므로 신중하게 사용해야 합니다.

    사용 시나리오:
    - 사용자가 대화 기록을 삭제하고 싶을 때
    - 불필요한 세션을 정리할 때
    - 개인정보 보호를 위해 대화 내역을 제거할 때

    Path Parameters:
        session_id: 삭제할 세션의 고유 ID (정수)

    Returns:
        {"message": "Session deleted successfully"}: 삭제 성공 메시지

    Raises:
        404: 해당 ID의 세션이 존재하지 않을 때
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()

    return {"message": "Session deleted successfully"}


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    session_id: int,
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="한 번에 조회할 메시지 수입니다. 범위: 1-200. 기본값: 50. 대화 기록이 길 경우 페이지네이션에 사용합니다."
    ),
    offset: int = Query(
        0,
        ge=0,
        description="건너뛸 메시지 수입니다. 기본값: 0. 페이지네이션에 사용합니다. 예: offset=50이면 51번째 메시지부터 조회합니다."
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    특정 세션의 메시지 목록을 조회합니다.

    지정된 세션에 속한 모든 대화 메시지(사용자 메시지 + AI 응답)를
    시간순으로 조회합니다. 대화 히스토리를 불러와 표시하거나,
    이전 대화 맥락을 확인할 때 사용합니다.

    사용 시나리오:
    - 이전 대화 내용을 화면에 표시할 때
    - 대화 맥락을 확인하여 이어서 대화할 때
    - 특정 대화 내용을 검색하거나 확인할 때

    Path Parameters:
        session_id: 메시지를 조회할 세션의 고유 ID (정수)

    Returns:
        List[MessageResponse]: 메시지 목록. 각 메시지에는 id, role(user/assistant/system), content, created_at이 포함됩니다.
        결과는 created_at 기준 오름차순(시간순)으로 정렬됩니다.

    Raises:
        404: 해당 ID의 세션이 존재하지 않을 때
    """
    # 세션 존재 확인
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")

    # 메시지 조회
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .limit(limit)
        .offset(offset)
    )
    messages = result.scalars().all()

    return [
        MessageResponse(
            id=msg.id,
            role=msg.role.value,
            content=msg.content,
            created_at=msg.created_at
        )
        for msg in messages
    ]


# ============== Chat Endpoints (SSE) ==============
@router.get("/send")
async def send_message(
    request: Request,
    message: str = Query(
        ...,
        description="LLM에게 전달할 사용자 메시지입니다. 질문, 요청, 대화 내용 등 자유 형식의 텍스트를 입력합니다. 한국어와 영어 모두 지원됩니다. 예: '오늘 날씨 어때?', 'Python으로 정렬 알고리즘 설명해줘'"
    ),
    model_name: str = Query(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        description="사용할 LLM 모델의 Hugging Face 모델 ID입니다. 지원 모델: (1) 'Bllossom/llama-3.2-Korean-Bllossom-3B' - 3B 파라미터의 경량 한국어 특화 모델로 빠른 응답이 필요할 때 적합합니다. (2) 'kakaocorp/kanana-2-30b-a3b-instruct' - Kakao에서 개발한 30B 파라미터의 고성능 모델로 복잡한 추론, 코드 생성, 고품질 응답이 필요할 때 적합합니다."
    ),
    system_prompt: str = Query(
        "You are a helpful assistant. Always respond in Korean.",
        description="LLM의 역할과 행동 방식을 정의하는 시스템 프롬프트입니다. AI의 페르소나, 응답 언어, 응답 스타일, 전문 분야 등을 지정할 수 있습니다. 예: '너는 Python 전문가야. 코드 예제와 함께 설명해줘.', '너는 친절한 상담사야. 공감하며 대화해줘.'"
    ),
    max_tokens: int = Query(
        1024,
        ge=1,
        le=4096,
        description="LLM이 생성할 최대 토큰 수입니다. 범위: 1-4096. 토큰은 대략 한글 0.5-1글자, 영어 0.75단어에 해당합니다. 권장값: 짧은 답변(256), 일반 대화(1024), 긴 설명/코드 생성(2048-4096)."
    ),
    temperature: float = Query(
        0.7,
        ge=0.0,
        le=2.0,
        description="응답의 창의성/무작위성을 조절하는 값입니다. 범위: 0.0-2.0. 낮은 값(0.1-0.3)은 일관되고 결정적인 응답(사실 기반 Q&A에 적합), 중간 값(0.5-0.8)은 균형 잡힌 응답(일반 대화에 적합), 높은 값(1.0-2.0)은 창의적이고 다양한 응답(창작, 브레인스토밍에 적합)을 생성합니다."
    ),
    use_tools: bool = Query(
        False,
        description="도구 호출 기능을 활성화합니다. True로 설정하면 LLM이 필요에 따라 웹 검색, DB 조회, API 호출, 계산기 등의 도구를 사용하여 답변합니다. 도구 호출 내역은 tool_logs 테이블에 기록됩니다."
    ),
    session_id: Optional[int] = Query(
        None,
        description="도구 호출 로그를 연결할 세션 ID입니다. use_tools=true일 때만 사용됩니다. 지정하면 해당 세션에 도구 호출 로그가 연결되어 나중에 조회할 수 있습니다."
    ),
    max_iterations: int = Query(
        5,
        ge=1,
        le=10,
        description="도구를 호출할 수 있는 최대 횟수입니다. use_tools=true일 때만 사용됩니다. 무한 루프를 방지합니다."
    ),
):
    """
    LLM에게 메시지를 전송하고 실시간 스트리밍 응답을 받습니다.

    이 API는 Server-Sent Events(SSE)를 통해 LLM의 응답을 토큰 단위로 실시간 스트리밍합니다.
    use_tools=true로 설정하면 LLM이 필요시 MCP 도구를 호출하여 답변합니다.

    사용 시나리오:
    - 사용자가 AI 어시스턴트와 실시간 대화하고 싶을 때
    - 텍스트 기반 질문에 대한 즉각적인 답변이 필요할 때
    - 코드 생성, 번역, 요약, 설명 등 텍스트 처리가 필요할 때
    - use_tools=true: 웹 검색, DB 조회 등 도구가 필요한 질문에 답변할 때

    SSE Events (각 이벤트는 JSON 형식):
    - start: 스트리밍 시작. {type: "start", model: "모델명", use_tools: bool, timestamp: "ISO8601"}
    - thinking: (use_tools=true) 에이전트가 생각 중. {type: "thinking", iteration: 숫자}
    - tool_call: (use_tools=true) 도구 호출 시작. {type: "tool_call", tool_name: "도구명", arguments: {...}, log_id: 숫자}
    - tool_result: (use_tools=true) 도구 결과. {type: "tool_result", tool_name: "도구명", success: bool, content/error: ...}
    - message: 토큰 생성 중. {type: "token", token: "생성된토큰", full_text: "전체텍스트", token_count: 숫자}
    - complete: 스트리밍 완료. {type: "complete", full_response: "전체응답", total_tokens: 숫자, generation_time_ms: 밀리초, tool_calls_count: 숫자}
    - error: 에러 발생. {type: "error", error: "에러메시지", error_type: "에러타입"}

    Returns:
        EventSourceResponse: SSE 스트림. 클라이언트는 EventSource API로 실시간 수신 가능합니다.

    Raises:
        503: 요청한 모델이 로드되지 않은 경우. 응답에 지원 모델 목록과 현재 로드된 모델 목록이 포함됩니다.
    """
    # 모델 로드 확인
    if not llm_service.is_model_loaded(model_name):
        return JSONResponse(
            status_code=503,
            content={
                "error": f"모델이 로드되지 않았습니다: {model_name}",
                "hint": "서버가 시작 중이거나 모델 로드에 실패했습니다.",
                "supported_models": llm_service.get_supported_models(),
                "loaded_models": llm_service.get_loaded_models()
            }
        )

    # use_tools=true인 경우 에이전트 모드로 처리
    if use_tools:
        return await _send_message_with_tools(
            request=request,
            message=message,
            model_name=model_name,
            session_id=session_id,
            max_iterations=max_iterations,
        )

    # 기존 로직: 도구 없이 단순 LLM 응답
    async def chat_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """채팅 응답 스트리밍 제너레이터"""
        token_count = 0
        full_response = ""
        start_time = datetime.now()

        try:
            # 시작 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "start",
                    "model": model_name,
                    "use_tools": False,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # LLM 스트리밍 생성
            for token in llm_service.generate_stream(
                prompt=message,
                model_name=model_name,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                # 클라이언트 연결 끊김 확인
                if await request.is_disconnected():
                    break

                token_count += 1
                full_response += token

                # 토큰 이벤트
                yield ServerSentEvent(
                    data=json.dumps({
                        "type": "token",
                        "token": token,
                        "full_text": full_response,
                        "token_count": token_count
                    }, ensure_ascii=False),
                    event="message",
                    id=str(token_count)
                )

                await asyncio.sleep(0)

            # 생성 시간 계산
            generation_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # 완료 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "complete",
                    "full_response": full_response,
                    "total_tokens": token_count,
                    "generation_time_ms": generation_time,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="complete"
            )

        except Exception as e:
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }, ensure_ascii=False),
                event="error"
            )

        except asyncio.CancelledError:
            print(f"채팅 스트리밍 취소됨. 토큰: {token_count}")

    return EventSourceResponse(chat_generator())


async def _send_message_with_tools(
    request: Request,
    message: str,
    model_name: str,
    session_id: Optional[int],
    max_iterations: int,
) -> EventSourceResponse:
    """도구 호출 기능이 포함된 메시지 전송 (내부 함수)"""

    async def agent_chat_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """도구 호출 가능한 에이전트 채팅 응답 스트리밍 제너레이터"""
        start_time = datetime.now()
        tool_calls_count = 0
        token_count = 0
        current_tool_log_id: Optional[int] = None
        tool_start_time: Optional[datetime] = None

        try:
            # 시작 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "start",
                    "model": model_name,
                    "use_tools": True,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # 에이전트 컨텍스트 생성
            context = AgentContext(max_iterations=max_iterations)

            # 에이전트 서비스 생성
            agent = AgentService(model_name=model_name)

            # 에이전트 실행
            async for event in agent.run_stream(message, context):
                # 클라이언트 연결 끊김 확인
                if await request.is_disconnected():
                    break

                state = event["state"]

                if state == AgentState.THINKING:
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "thinking",
                            "iteration": event.get("iteration", 1),
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="thinking"
                    )

                elif state == AgentState.TOOL_CALL:
                    tool_calls_count += 1
                    tool_name = event["tool_call"]["name"]
                    tool_args = event["tool_call"]["arguments"]
                    tool_start_time = datetime.now()

                    # 도구 호출 로그 생성 (PENDING 상태)
                    async with AsyncSessionLocal() as db:
                        tool_log = ToolLog(
                            session_id=session_id,
                            tool_name=tool_name,
                            tool_arguments=tool_args,
                            status=ToolCallStatus.PENDING
                        )
                        db.add(tool_log)
                        await db.commit()
                        await db.refresh(tool_log)
                        current_tool_log_id = tool_log.id

                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "tool_call",
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "log_id": current_tool_log_id,
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="tool_call"
                    )

                elif state == AgentState.TOOL_RESULT:
                    tool_result = event["tool_result"]
                    tool_end_time = datetime.now()
                    execution_time = int((tool_end_time - tool_start_time).total_seconds() * 1000) if tool_start_time else None

                    # 도구 호출 로그 업데이트
                    if current_tool_log_id:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(ToolLog).where(ToolLog.id == current_tool_log_id)
                            )
                            tool_log = result.scalar_one_or_none()
                            if tool_log:
                                if tool_result.get("success", False):
                                    tool_log.status = ToolCallStatus.SUCCESS
                                    tool_log.result_content = json.dumps(tool_result.get("content"), ensure_ascii=False) if tool_result.get("content") else None
                                else:
                                    tool_log.status = ToolCallStatus.FAILED
                                    tool_log.error_message = tool_result.get("error")
                                tool_log.execution_time_ms = execution_time
                                tool_log.completed_at = tool_end_time
                                await db.commit()

                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "tool_result",
                            "success": tool_result.get("success", False),
                            "content": tool_result.get("content") if tool_result.get("success") else None,
                            "error": tool_result.get("error") if not tool_result.get("success") else None,
                            "log_id": current_tool_log_id,
                            "execution_time_ms": execution_time,
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="tool_result"
                    )

                elif state == AgentState.RESPONDING:
                    if event.get("is_streaming"):
                        token_count += 1
                        yield ServerSentEvent(
                            data=json.dumps({
                                "type": "token",
                                "token": event["content"],
                                "full_text": event.get("full_text", ""),
                                "token_count": token_count
                            }, ensure_ascii=False),
                            event="message",
                            id=str(token_count)
                        )

                elif state == AgentState.COMPLETED:
                    generation_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "complete",
                            "full_response": event["content"],
                            "total_tokens": token_count,
                            "tool_calls_count": tool_calls_count,
                            "total_iterations": event.get("total_iterations", 1),
                            "generation_time_ms": generation_time,
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="complete"
                    )

                elif state == AgentState.ERROR:
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "error",
                            "error": event.get("error", "Unknown error"),
                            "message": event["content"]
                        }, ensure_ascii=False),
                        event="error"
                    )

                await asyncio.sleep(0)

        except Exception as e:
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                }, ensure_ascii=False),
                event="error"
            )

        except asyncio.CancelledError:
            print(f"에이전트 스트리밍 취소됨. 토큰: {token_count}, 도구 호출: {tool_calls_count}")

    return EventSourceResponse(agent_chat_generator())


# ============== Tool Logs Endpoints ==============

@router.get("/sessions/{session_id}/tool-logs", response_model=List[ToolLogResponse])
async def get_tool_logs(
    session_id: int,
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="한 번에 조회할 로그 수입니다. 범위: 1-200. 기본값: 50."
    ),
    offset: int = Query(
        0,
        ge=0,
        description="건너뛸 로그 수입니다. 기본값: 0. 페이지네이션에 사용합니다."
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    특정 세션의 도구 호출 로그를 조회합니다.

    use_tools=true로 /send를 호출했을 때 발생한 도구 호출 내역을 조회합니다.
    각 도구 호출의 이름, 인자, 결과, 실행 시간 등을 확인할 수 있습니다.

    Path Parameters:
        session_id: 로그를 조회할 세션의 고유 ID (정수)

    Returns:
        List[ToolLogResponse]: 도구 호출 로그 목록.
        각 로그에는 tool_name, tool_arguments, status, result_content, error_message,
        execution_time_ms, created_at, completed_at가 포함됩니다.

    Raises:
        404: 해당 ID의 세션이 존재하지 않을 때
    """
    # 세션 존재 확인
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")

    # 도구 로그 조회
    result = await db.execute(
        select(ToolLog)
        .where(ToolLog.session_id == session_id)
        .order_by(ToolLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    logs = result.scalars().all()

    return [
        ToolLogResponse(
            id=log.id,
            session_id=log.session_id,
            tool_name=log.tool_name,
            tool_arguments=log.tool_arguments,
            status=log.status.value,
            result_content=log.result_content,
            error_message=log.error_message,
            execution_time_ms=log.execution_time_ms,
            created_at=log.created_at,
            completed_at=log.completed_at
        )
        for log in logs
    ]


@router.get("/tool-logs", response_model=List[ToolLogResponse])
async def get_all_tool_logs(
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="한 번에 조회할 로그 수입니다. 범위: 1-200. 기본값: 50."
    ),
    offset: int = Query(
        0,
        ge=0,
        description="건너뛸 로그 수입니다. 기본값: 0. 페이지네이션에 사용합니다."
    ),
    status: Optional[str] = Query(
        None,
        description="필터링할 상태. 'pending', 'success', 'failed', 'timeout' 중 하나."
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    모든 도구 호출 로그를 조회합니다.

    세션과 관계없이 모든 도구 호출 내역을 조회합니다.
    상태별 필터링이 가능합니다.

    Returns:
        List[ToolLogResponse]: 도구 호출 로그 목록. 최신순으로 정렬됩니다.
    """
    query = select(ToolLog).order_by(ToolLog.created_at.desc())

    if status:
        try:
            status_enum = ToolCallStatus(status)
            query = query.where(ToolLog.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: pending, success, failed, timeout"
            )

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    logs = result.scalars().all()

    return [
        ToolLogResponse(
            id=log.id,
            session_id=log.session_id,
            tool_name=log.tool_name,
            tool_arguments=log.tool_arguments,
            status=log.status.value,
            result_content=log.result_content,
            error_message=log.error_message,
            execution_time_ms=log.execution_time_ms,
            created_at=log.created_at,
            completed_at=log.completed_at
        )
        for log in logs
    ]
