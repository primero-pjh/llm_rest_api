"""
Agent Router (SSE) - MCP 버전
==============================
MCP 기반 LLM 에이전트 엔드포인트를 제공합니다.
에이전트는 필요에 따라 MCP를 통해 도구를 호출하여 사용자 질문에 답변합니다.
"""

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from typing import AsyncGenerator
import asyncio
import json
from datetime import datetime

from app.services.agent_service import agent_service, AgentState, AgentContext
from app.mcp.client import mcp_client
from app.services.llm_service import llm_service

router = APIRouter(
    prefix="/sse/agent",
    tags=["Agent"],
)


@router.get("/tools")
async def list_tools():
    """
    사용 가능한 MCP 도구 목록을 조회합니다.

    MCP 서버에 등록된 모든 도구의 스키마 정보를 반환합니다.
    각 도구의 이름, 설명, 파라미터 정보를 확인할 수 있습니다.

    Returns:
        도구 목록과 각 도구의 상세 스키마
    """
    await mcp_client.initialize()
    tools = await mcp_client.list_tools()

    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            }
            for tool in tools
        ],
        "total": len(tools)
    }


@router.post("/tools/{tool_name}/execute")
async def execute_tool(
    tool_name: str,
    arguments: dict
):
    """
    특정 MCP 도구를 직접 실행합니다.

    에이전트를 거치지 않고 도구를 직접 테스트할 때 사용합니다.

    Path Parameters:
        tool_name: 실행할 도구 이름

    Request Body:
        도구에 전달할 파라미터 딕셔너리

    Returns:
        도구 실행 결과
    """
    await mcp_client.initialize()
    result = await mcp_client.call_tool(tool_name, arguments)

    return {
        "success": result.success,
        "content": result.content,
        "error": result.error
    }


@router.get("/chat")
async def agent_chat(
    request: Request,
    message: str = Query(
        ...,
        description="에이전트에게 전달할 사용자 메시지입니다. 에이전트는 필요에 따라 MCP 도구(웹 검색, DB 조회, API 호출 등)를 사용하여 답변합니다."
    ),
    model_name: str = Query(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        description="사용할 LLM 모델의 Hugging Face 모델 ID입니다."
    ),
    max_iterations: int = Query(
        5,
        ge=1,
        le=10,
        description="에이전트가 도구를 호출할 수 있는 최대 횟수입니다. 무한 루프 방지를 위해 사용됩니다."
    )
):
    """
    MCP 에이전트와 대화합니다 (SSE 스트리밍).

    에이전트는 사용자 질문을 분석하여:
    1. 바로 답변 가능하면 → 직접 답변
    2. 추가 정보가 필요하면 → MCP 도구 호출 → 결과로 답변

    SSE Events:
    - thinking: 에이전트가 생각 중 {"state": "thinking", "iteration": 1}
    - tool_call: 도구 호출 {"state": "tool_call", "tool_name": "...", "arguments": {...}}
    - tool_result: 도구 결과 {"state": "tool_result", "result": {...}}
    - message: 응답 토큰 {"state": "responding", "token": "...", "full_text": "..."}
    - complete: 완료 {"state": "completed", "full_response": "...", "iterations": N}
    - error: 에러 {"state": "error", "error": "..."}

    Returns:
        EventSourceResponse: SSE 스트림
    """
    # 모델 로드 확인
    if not llm_service.is_model_loaded(model_name):
        return JSONResponse(
            status_code=503,
            content={
                "error": f"모델이 로드되지 않았습니다: {model_name}",
                "supported_models": llm_service.get_supported_models(),
                "loaded_models": llm_service.get_loaded_models()
            }
        )

    async def agent_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """에이전트 응답 스트리밍 제너레이터"""
        start_time = datetime.now()

        try:
            # 시작 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "start",
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # 에이전트 컨텍스트 생성
            context = AgentContext(max_iterations=max_iterations)

            # 에이전트용 서비스 생성 (모델 지정)
            from app.services.agent_service import AgentService
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
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "tool_call",
                            "tool_name": event["tool_call"]["name"],
                            "arguments": event["tool_call"]["arguments"],
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="tool_call"
                    )

                elif state == AgentState.TOOL_RESULT:
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "tool_result",
                            "result": event["tool_result"],
                            "timestamp": datetime.now().isoformat()
                        }, ensure_ascii=False),
                        event="tool_result"
                    )

                elif state == AgentState.RESPONDING:
                    if event.get("is_streaming"):
                        yield ServerSentEvent(
                            data=json.dumps({
                                "type": "token",
                                "token": event["content"],
                                "full_text": event.get("full_text", "")
                            }, ensure_ascii=False),
                            event="message"
                        )

                elif state == AgentState.COMPLETED:
                    generation_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    yield ServerSentEvent(
                        data=json.dumps({
                            "type": "complete",
                            "full_response": event["content"],
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
            print("에이전트 스트리밍 취소됨")

    return EventSourceResponse(agent_generator())


@router.post("/chat")
async def agent_chat_sync(
    message: str = Query(..., description="에이전트에게 전달할 메시지"),
    model_name: str = Query(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        description="사용할 LLM 모델"
    ),
    max_iterations: int = Query(5, ge=1, le=10, description="최대 도구 호출 횟수")
):
    """
    MCP 에이전트와 대화합니다 (동기 방식).

    SSE 스트리밍이 아닌 일반 JSON 응답을 반환합니다.
    전체 처리가 완료된 후 최종 결과만 반환됩니다.

    Returns:
        최종 응답과 실행 로그
    """
    if not llm_service.is_model_loaded(model_name):
        return JSONResponse(
            status_code=503,
            content={
                "error": f"모델이 로드되지 않았습니다: {model_name}",
                "supported_models": llm_service.get_supported_models(),
                "loaded_models": llm_service.get_loaded_models()
            }
        )

    from app.services.agent_service import AgentService
    agent = AgentService(model_name=model_name)
    context = AgentContext(max_iterations=max_iterations)

    execution_log = []
    final_response = ""

    async for event in agent.run(message, context):
        state = event["state"]

        log_entry = {
            "state": state.value,
            "timestamp": datetime.now().isoformat()
        }

        if state == AgentState.TOOL_CALL:
            log_entry["tool_call"] = event.get("tool_call")
        elif state == AgentState.TOOL_RESULT:
            log_entry["tool_result"] = event.get("tool_result")
        elif state in [AgentState.COMPLETED, AgentState.RESPONDING]:
            final_response = event["content"]

        execution_log.append(log_entry)

    return {
        "response": final_response,
        "execution_log": execution_log,
        "total_iterations": context.current_iteration
    }
