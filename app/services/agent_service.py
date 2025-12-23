"""
Agent Service Module (MCP 버전)
===============================
MCP 클라이언트를 사용하여 LLM이 도구를 호출하는 에이전트 로직을 구현합니다.

에이전트 실행 흐름:
1. 사용자 질문 수신
2. MCP 클라이언트에서 도구 목록 조회
3. LLM에게 도구 목록과 함께 질문 전달
4. LLM이 도구 호출이 필요하다고 판단하면 → MCP를 통해 도구 실행 → 결과를 LLM에게 다시 전달
5. LLM이 최종 응답 생성
6. 응답 반환
"""

from typing import AsyncGenerator, Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from app.services.llm_service import llm_service
from app.mcp.client import mcp_client, InProcessMCPClient


class AgentState(Enum):
    """에이전트 상태"""
    THINKING = "thinking"       # LLM이 생각 중
    TOOL_CALL = "tool_call"     # 도구 호출 중
    TOOL_RESULT = "tool_result" # 도구 결과 처리 중
    RESPONDING = "responding"   # 최종 응답 생성 중
    COMPLETED = "completed"     # 완료
    ERROR = "error"             # 에러


@dataclass
class ToolCall:
    """도구 호출 정보"""
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentMessage:
    """에이전트 메시지"""
    role: str  # "user", "assistant", "tool", "system"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


@dataclass
class AgentContext:
    """에이전트 컨텍스트 - 대화 상태 관리"""
    messages: List[AgentMessage] = field(default_factory=list)
    max_iterations: int = 5  # 무한 루프 방지
    current_iteration: int = 0


class AgentService:
    """
    LLM 에이전트 서비스 (MCP 기반)

    MCP 클라이언트를 통해 도구를 호출하고 LLM 응답을 생성합니다.
    """

    def __init__(
        self,
        mcp: Optional[InProcessMCPClient] = None,
        model_name: Optional[str] = None
    ):
        self.mcp = mcp or mcp_client
        self.model_name = model_name
        self._tools_description: Optional[str] = None

    async def _get_tools_description(self) -> str:
        """MCP에서 도구 설명을 가져옵니다."""
        if self._tools_description is None:
            await self.mcp.initialize()
            self._tools_description = self.mcp.get_tools_description()
        return self._tools_description

    async def _build_system_prompt(self) -> str:
        """도구 정보가 포함된 시스템 프롬프트 생성"""
        tools_description = await self._get_tools_description()

        return f"""당신은 도구를 사용할 수 있는 AI 어시스턴트입니다.

## {tools_description}

## 도구 사용 방법
도구가 필요하면 다음 형식으로 호출하세요:
<tool_call>
{{"name": "도구이름", "arguments": {{"파라미터": "값"}}}}
</tool_call>

## 중요 규칙
1. 도구가 필요한 경우에만 도구를 사용하세요.
2. 도구 호출 결과를 받으면 그 정보를 바탕으로 사용자에게 답변하세요.
3. 한 번에 하나의 도구만 호출하세요.
4. 도구 없이 답변 가능하면 바로 답변하세요.
5. 항상 한국어로 답변하세요.
"""

    def _parse_tool_calls(self, response: str) -> Optional[ToolCall]:
        """LLM 응답에서 도구 호출 파싱"""
        # <tool_call>...</tool_call> 패턴 찾기
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            try:
                tool_data = json.loads(match.group(1))
                return ToolCall(
                    name=tool_data.get("name", ""),
                    arguments=tool_data.get("arguments", {})
                )
            except json.JSONDecodeError:
                return None
        return None

    def _remove_tool_call_from_response(self, response: str) -> str:
        """응답에서 도구 호출 부분 제거"""
        pattern = r'<tool_call>.*?</tool_call>'
        return re.sub(pattern, '', response, flags=re.DOTALL).strip()

    async def _execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """MCP를 통해 도구 실행"""
        result = await self.mcp.call_tool(
            tool_call.name,
            tool_call.arguments
        )
        return {
            "success": result.success,
            "content": result.content,
            "error": result.error
        }

    def _build_conversation_prompt(self, context: AgentContext) -> str:
        """대화 컨텍스트를 프롬프트로 변환"""
        prompt_parts = []

        for msg in context.messages:
            if msg.role == "user":
                prompt_parts.append(f"사용자: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"어시스턴트: {msg.content}")
            elif msg.role == "tool":
                prompt_parts.append(f"[도구 결과 - {msg.tool_call_id}]: {msg.content}")

        return "\n\n".join(prompt_parts)

    async def run(
        self,
        user_message: str,
        context: Optional[AgentContext] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        에이전트 실행 (비스트리밍)

        Yields:
            각 단계별 상태와 결과
            {
                "state": AgentState,
                "content": str,
                "tool_call": Optional[ToolCall],
                "tool_result": Optional[Dict]
            }
        """
        if context is None:
            context = AgentContext()

        # MCP 클라이언트 초기화
        await self.mcp.initialize()

        # 사용자 메시지 추가
        context.messages.append(AgentMessage(role="user", content=user_message))

        system_prompt = await self._build_system_prompt()

        while context.current_iteration < context.max_iterations:
            context.current_iteration += 1

            # 1. LLM에게 질문
            yield {
                "state": AgentState.THINKING,
                "content": "생각 중...",
                "iteration": context.current_iteration
            }

            conversation_prompt = self._build_conversation_prompt(context)

            # LLM 응답 생성 (비스트리밍)
            full_response = llm_service.generate(
                prompt=conversation_prompt,
                model_name=self.model_name,
                system_prompt=system_prompt,
                max_new_tokens=1024,
                temperature=0.7
            )

            # 2. 도구 호출 파싱
            tool_call = self._parse_tool_calls(full_response)

            if tool_call:
                # 도구 호출이 있는 경우
                yield {
                    "state": AgentState.TOOL_CALL,
                    "content": f"도구 호출: {tool_call.name}",
                    "tool_call": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }

                # 3. MCP를 통해 도구 실행
                tool_result = await self._execute_tool(tool_call)

                yield {
                    "state": AgentState.TOOL_RESULT,
                    "content": "도구 결과 수신",
                    "tool_result": tool_result
                }

                # 컨텍스트에 도구 결과 추가
                clean_response = self._remove_tool_call_from_response(full_response)
                if clean_response:
                    context.messages.append(AgentMessage(
                        role="assistant",
                        content=clean_response
                    ))

                context.messages.append(AgentMessage(
                    role="tool",
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tool_call.name
                ))

                # 다음 반복에서 LLM이 결과를 처리
                continue

            else:
                # 도구 호출 없음 → 최종 응답
                context.messages.append(AgentMessage(
                    role="assistant",
                    content=full_response
                ))

                yield {
                    "state": AgentState.RESPONDING,
                    "content": full_response
                }

                yield {
                    "state": AgentState.COMPLETED,
                    "content": full_response,
                    "total_iterations": context.current_iteration
                }
                return

        # 최대 반복 도달
        yield {
            "state": AgentState.ERROR,
            "content": "최대 반복 횟수에 도달했습니다.",
            "error": "MAX_ITERATIONS_REACHED"
        }

    async def run_stream(
        self,
        user_message: str,
        context: Optional[AgentContext] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        에이전트 실행 (스트리밍 버전)

        최종 응답을 토큰 단위로 스트리밍합니다.
        """
        if context is None:
            context = AgentContext()

        # MCP 클라이언트 초기화
        await self.mcp.initialize()

        context.messages.append(AgentMessage(role="user", content=user_message))
        system_prompt = await self._build_system_prompt()

        while context.current_iteration < context.max_iterations:
            context.current_iteration += 1

            yield {
                "state": AgentState.THINKING,
                "content": "생각 중...",
                "iteration": context.current_iteration
            }

            conversation_prompt = self._build_conversation_prompt(context)

            # 첫 번째 패스: 도구 호출 확인을 위해 전체 응답 수집
            full_response = ""
            for token in llm_service.generate_stream(
                prompt=conversation_prompt,
                model_name=self.model_name,
                system_prompt=system_prompt,
                max_new_tokens=1024,
                temperature=0.7
            ):
                full_response += token

            tool_call = self._parse_tool_calls(full_response)

            if tool_call:
                yield {
                    "state": AgentState.TOOL_CALL,
                    "content": f"도구 호출: {tool_call.name}",
                    "tool_call": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                }

                tool_result = await self._execute_tool(tool_call)

                yield {
                    "state": AgentState.TOOL_RESULT,
                    "content": "도구 결과 수신",
                    "tool_result": tool_result
                }

                clean_response = self._remove_tool_call_from_response(full_response)
                if clean_response:
                    context.messages.append(AgentMessage(
                        role="assistant",
                        content=clean_response
                    ))

                context.messages.append(AgentMessage(
                    role="tool",
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tool_call.name
                ))
                continue

            else:
                # 최종 응답 스트리밍
                context.messages.append(AgentMessage(
                    role="assistant",
                    content=full_response
                ))

                # 토큰 단위로 재스트리밍 (이미 수집한 응답을)
                for i, char in enumerate(full_response):
                    yield {
                        "state": AgentState.RESPONDING,
                        "content": char,
                        "full_text": full_response[:i+1],
                        "is_streaming": True
                    }

                yield {
                    "state": AgentState.COMPLETED,
                    "content": full_response,
                    "total_iterations": context.current_iteration
                }
                return

        yield {
            "state": AgentState.ERROR,
            "content": "최대 반복 횟수에 도달했습니다.",
            "error": "MAX_ITERATIONS_REACHED"
        }


# 전역 에이전트 서비스 인스턴스
agent_service = AgentService()
