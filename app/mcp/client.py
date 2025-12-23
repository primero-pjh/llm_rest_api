"""
MCP Client Module
=================
MCP 서버에 연결하여 도구를 호출하는 클라이언트입니다.

LLM 에이전트가 이 클라이언트를 통해 MCP 서버의 도구를 사용합니다.
"""

import asyncio
import json
from typing import Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ToolInfo:
    """도구 정보"""
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    content: Any
    error: Optional[str] = None


class MCPClient:
    """
    MCP 클라이언트

    MCP 서버에 연결하여 도구 목록을 조회하고 도구를 실행합니다.
    """

    def __init__(self):
        self._session: Optional[ClientSession] = None
        self._tools: dict[str, ToolInfo] = {}

    @asynccontextmanager
    async def connect(self, server_script_path: str):
        """
        MCP 서버에 연결합니다.

        Args:
            server_script_path: MCP 서버 스크립트 경로

        Usage:
            async with client.connect("app/mcp/server.py") as client:
                tools = await client.list_tools()
                result = await client.call_tool("calculator", {"expression": "2+2"})
        """
        server_params = StdioServerParameters(
            command="python",
            args=["-m", server_script_path.replace("/", ".").replace(".py", "")],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session

                # 초기화
                await session.initialize()

                # 도구 목록 캐싱
                await self._cache_tools()

                yield self

                self._session = None

    async def _cache_tools(self):
        """도구 목록을 캐싱합니다."""
        if not self._session:
            return

        result = await self._session.list_tools()

        self._tools = {
            tool.name: ToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema
            )
            for tool in result.tools
        }

    async def list_tools(self) -> list[ToolInfo]:
        """사용 가능한 도구 목록을 반환합니다."""
        return list(self._tools.values())

    def get_tools_for_llm(self) -> list[dict]:
        """
        LLM에게 전달할 형식으로 도구 정보를 반환합니다.
        OpenAI function calling 형식과 호환됩니다.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in self._tools.values()
        ]

    def get_tools_description(self) -> str:
        """
        LLM 시스템 프롬프트에 포함할 도구 설명을 생성합니다.
        """
        lines = ["사용 가능한 도구:"]
        for tool in self._tools.values():
            params = tool.input_schema.get("properties", {})
            param_str = ", ".join([
                f"{k}({v.get('type', 'any')})"
                for k, v in params.items()
            ])
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  파라미터: {param_str}")
        return "\n".join(lines)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        도구를 실행합니다.

        Args:
            name: 도구 이름
            arguments: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        if not self._session:
            return ToolResult(
                success=False,
                content=None,
                error="MCP 서버에 연결되지 않았습니다."
            )

        if name not in self._tools:
            return ToolResult(
                success=False,
                content=None,
                error=f"알 수 없는 도구: {name}"
            )

        try:
            result = await self._session.call_tool(name, arguments)

            # 결과 파싱
            content = []
            for item in result.content:
                if hasattr(item, 'text'):
                    try:
                        content.append(json.loads(item.text))
                    except json.JSONDecodeError:
                        content.append(item.text)

            return ToolResult(
                success=not result.isError if hasattr(result, 'isError') else True,
                content=content[0] if len(content) == 1 else content,
                error=None
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=None,
                error=str(e)
            )


class InProcessMCPClient:
    """
    In-Process MCP 클라이언트

    별도 프로세스 없이 직접 MCP 서버의 핸들러를 호출합니다.
    FastAPI 앱 내에서 사용할 때 유용합니다.
    """

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}
        self._initialized = False

    async def initialize(self):
        """클라이언트 초기화 - 도구 목록 로드"""
        if self._initialized:
            return

        # 서버 모듈에서 직접 도구 목록 가져오기
        from app.mcp.server import list_tools

        tools = await list_tools()

        self._tools = {
            tool.name: ToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema
            )
            for tool in tools
        }

        self._initialized = True

    async def list_tools(self) -> list[ToolInfo]:
        """사용 가능한 도구 목록 반환"""
        await self.initialize()
        return list(self._tools.values())

    def get_tools_for_llm(self) -> list[dict]:
        """LLM용 도구 정보 반환 (OpenAI 형식)"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in self._tools.values()
        ]

    def get_tools_description(self) -> str:
        """시스템 프롬프트용 도구 설명 생성"""
        lines = ["사용 가능한 도구:"]
        for tool in self._tools.values():
            params = tool.input_schema.get("properties", {})
            param_str = ", ".join([
                f"{k}({v.get('type', 'any')})"
                for k, v in params.items()
            ])
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  파라미터: {param_str}")
        return "\n".join(lines)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """도구 실행"""
        await self.initialize()

        if name not in self._tools:
            return ToolResult(
                success=False,
                content=None,
                error=f"알 수 없는 도구: {name}"
            )

        try:
            # 서버 핸들러 직접 호출
            from app.mcp.server import call_tool

            result = await call_tool(name, arguments)

            # 결과 파싱
            content = []
            for item in result.content:
                if hasattr(item, 'text'):
                    try:
                        content.append(json.loads(item.text))
                    except json.JSONDecodeError:
                        content.append(item.text)

            is_error = result.isError if hasattr(result, 'isError') else False

            return ToolResult(
                success=not is_error,
                content=content[0] if len(content) == 1 else content,
                error=content[0] if is_error and content else None
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=None,
                error=str(e)
            )


# 전역 In-Process 클라이언트 인스턴스
mcp_client = InProcessMCPClient()
