"""
MCP Server Module
=================
Python mcp 라이브러리를 사용한 표준 MCP 서버 구현입니다.

이 서버는 다음 도구들을 제공합니다:
- smart_db_query: 자연어 → SQL 변환 및 자동 실행
- analyze_db_data: DB 데이터 분석 및 통계
- web_search: 웹 검색
- database_query: DB 조회
- api_call: 외부 API 호출
- calculator: 수학 계산

실행 방법:
    python -m app.mcp.server

또는 stdio 모드로 Claude Desktop에서 사용:
    claude_desktop_config.json에 등록
"""

import asyncio
import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Optional
from collections import Counter
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)


# MCP 서버 인스턴스 생성
server = Server("llm-tools-server")


# ============== 도구 정의 ==============

# ============== DB 스키마 정보 ==============

DB_SCHEMA = {
    "chat_sessions": {
        "description": "대화 세션 테이블 - 채팅방/대화 스레드 정보",
        "columns": {
            "id": {"type": "INTEGER", "description": "세션 고유 ID (PK)"},
            "user_id": {"type": "INTEGER", "description": "사용자 ID (FK → users.id, nullable)"},
            "title": {"type": "VARCHAR(255)", "description": "대화 제목"},
            "summary": {"type": "TEXT", "description": "대화 요약"},
            "model_name": {"type": "VARCHAR(200)", "description": "사용된 LLM 모델명"},
            "system_prompt": {"type": "TEXT", "description": "시스템 프롬프트"},
            "temperature": {"type": "VARCHAR(10)", "description": "생성 온도 (0.0~2.0)"},
            "max_tokens": {"type": "INTEGER", "description": "최대 생성 토큰 수"},
            "total_messages": {"type": "INTEGER", "description": "총 메시지 수"},
            "total_tokens_used": {"type": "INTEGER", "description": "사용된 총 토큰 수"},
            "created_at": {"type": "DATETIME", "description": "생성 시각"},
            "updated_at": {"type": "DATETIME", "description": "업데이트 시각"},
        }
    },
    "chat_messages": {
        "description": "대화 메시지 테이블 - 개별 메시지 저장",
        "columns": {
            "id": {"type": "INTEGER", "description": "메시지 고유 ID (PK)"},
            "session_id": {"type": "INTEGER", "description": "소속 세션 ID (FK → chat_sessions.id)"},
            "role": {"type": "ENUM", "description": "메시지 역할 (user/assistant/system)"},
            "content": {"type": "TEXT", "description": "메시지 내용"},
            "input_tokens": {"type": "INTEGER", "description": "입력 토큰 수 (assistant만)"},
            "output_tokens": {"type": "INTEGER", "description": "출력 토큰 수 (assistant만)"},
            "model_name": {"type": "VARCHAR(200)", "description": "응답 생성 모델명"},
            "generation_time": {"type": "INTEGER", "description": "생성 시간 ms"},
            "created_at": {"type": "DATETIME", "description": "생성 시각"},
        }
    },
    "tool_logs": {
        "description": "도구 호출 로그 테이블 - MCP 도구 실행 기록",
        "columns": {
            "id": {"type": "INTEGER", "description": "로그 고유 ID (PK)"},
            "session_id": {"type": "INTEGER", "description": "소속 세션 ID (nullable)"},
            "tool_name": {"type": "VARCHAR(100)", "description": "호출된 도구 이름"},
            "tool_arguments": {"type": "JSON", "description": "도구에 전달된 인자"},
            "status": {"type": "ENUM", "description": "상태 (pending/success/failed/timeout)"},
            "result_content": {"type": "TEXT", "description": "도구 실행 결과"},
            "error_message": {"type": "TEXT", "description": "에러 메시지"},
            "execution_time_ms": {"type": "INTEGER", "description": "실행 소요 시간 ms"},
            "created_at": {"type": "DATETIME", "description": "호출 시작 시각"},
            "completed_at": {"type": "DATETIME", "description": "실행 완료 시각"},
        }
    },
    "users": {
        "description": "사용자 테이블",
        "columns": {
            "id": {"type": "INTEGER", "description": "사용자 고유 ID (PK)"},
            "username": {"type": "VARCHAR(100)", "description": "사용자명"},
            "email": {"type": "VARCHAR(255)", "description": "이메일"},
            "created_at": {"type": "DATETIME", "description": "생성 시각"},
        }
    },
    "calendars": {
        "description": "캘린더 테이블",
        "columns": {
            "id": {"type": "INTEGER", "description": "캘린더 고유 ID (PK)"},
            "user_id": {"type": "INTEGER", "description": "사용자 ID (FK → users.id)"},
            "name": {"type": "VARCHAR(100)", "description": "캘린더 이름"},
            "created_at": {"type": "DATETIME", "description": "생성 시각"},
        }
    },
    "calendar_events": {
        "description": "캘린더 이벤트 테이블",
        "columns": {
            "id": {"type": "INTEGER", "description": "이벤트 고유 ID (PK)"},
            "calendar_id": {"type": "INTEGER", "description": "캘린더 ID (FK → calendars.id)"},
            "title": {"type": "VARCHAR(255)", "description": "이벤트 제목"},
            "description": {"type": "TEXT", "description": "이벤트 설명"},
            "start_time": {"type": "DATETIME", "description": "시작 시간"},
            "end_time": {"type": "DATETIME", "description": "종료 시간"},
            "created_at": {"type": "DATETIME", "description": "생성 시각"},
        }
    }
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록 반환"""
    return [
        Tool(
            name="smart_db_query",
            description="""자연어 요청을 분석하여 SQL 쿼리를 자동 생성하고 실행합니다.

사용 예시:
- "오늘 생성된 채팅 세션 보여줘"
- "가장 많은 메시지를 가진 세션 TOP 5"
- "이번 주에 사용된 총 토큰 수"
- "user 역할의 메시지 중 가장 긴 것"
- "실패한 도구 호출 목록"

지원 테이블: chat_sessions, chat_messages, tool_logs, users, calendars, calendar_events""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "자연어로 작성된 데이터 조회 요청 (예: '오늘 생성된 세션 개수', '가장 활발한 사용자')"
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "True면 SQL만 생성하고 실행하지 않음 (기본값: False)",
                        "default": False
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="analyze_db_data",
            description="""DB 데이터를 분석하여 통계, 트렌드, 인사이트를 제공합니다.

사용 예시:
- "채팅 세션 사용 통계 분석해줘"
- "시간대별 메시지 생성 패턴"
- "모델별 토큰 사용량 비교"
- "도구 호출 성공률 분석"

분석 유형: statistics, trend, comparison, distribution""",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "분석할 테이블",
                        "enum": ["chat_sessions", "chat_messages", "tool_logs", "users", "calendars", "calendar_events"]
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "분석 유형",
                        "enum": ["statistics", "trend", "comparison", "distribution"],
                        "default": "statistics"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "그룹화 기준 컬럼 (선택사항)"
                    },
                    "time_range": {
                        "type": "string",
                        "description": "분석 기간 (today, this_week, this_month, all)",
                        "default": "all"
                    }
                },
                "required": ["table"]
            }
        ),
        Tool(
            name="get_db_schema",
            description="데이터베이스 테이블 스키마 정보를 조회합니다. 어떤 테이블과 컬럼이 있는지 확인할 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "조회할 테이블명 (생략시 전체 테이블 목록)",
                        "enum": ["chat_sessions", "chat_messages", "tool_logs", "users", "calendars", "calendar_events"]
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="web_search",
            description="인터넷에서 정보를 검색합니다. 최신 정보, 뉴스, 사실 확인이 필요할 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 키워드나 질문"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "반환할 검색 결과 수 (기본값: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="database_query",
            description="데이터베이스에서 정보를 조회합니다. 저장된 데이터를 검색할 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "조회할 테이블 이름",
                        "enum": ["chat_sessions", "chat_messages", "users"]
                    },
                    "conditions": {
                        "type": "object",
                        "description": "조회 조건 (예: {\"user_id\": 1})"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "최대 조회 건수 (기본값: 10)",
                        "default": 10
                    }
                },
                "required": ["table"]
            }
        ),
        Tool(
            name="api_call",
            description="외부 REST API를 호출합니다. 외부 서비스의 데이터를 가져올 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "호출할 API URL"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP 메서드",
                        "enum": ["GET", "POST"],
                        "default": "GET"
                    },
                    "headers": {
                        "type": "object",
                        "description": "요청 헤더"
                    },
                    "body": {
                        "type": "object",
                        "description": "요청 본문 (POST인 경우)"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="calculator",
            description="수학 계산을 수행합니다. 복잡한 계산이 필요할 때 사용합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "계산할 수학 표현식 (예: '2 + 3 * 4', 'sqrt(16)', 'sin(3.14)')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="check_calendar_conflicts",
            description="""주어진 시간대에 겹치는 기존 일정을 조회합니다.

사용 예시:
- 새로운 일정을 추가하기 전에 충돌 확인
- 특정 시간대의 일정 가용성 확인

충돌 조건: (기존.start < 새.end) AND (기존.end > 새.start)

반환 값:
- has_conflict: 충돌 여부 (boolean)
- conflicting_events: 충돌하는 기존 일정 목록
- conflict_count: 충돌하는 일정 수""",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "일정을 확인할 사용자 ID"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "시작 날짜 (YYYY-MM-DD 형식)"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "시작 시간 (HH:MM 형식, 종일 이벤트는 '00:00')"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "종료 날짜 (YYYY-MM-DD 형식)"
                    },
                    "end_time": {
                        "type": "string",
                        "description": "종료 시간 (HH:MM 형식, 종일 이벤트는 '23:59')"
                    }
                },
                "required": ["user_id", "start_date", "start_time", "end_date", "end_time"]
            }
        ),
    ]


# ============== 도구 실행 핸들러 ==============

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """도구 호출 처리"""

    if name == "smart_db_query":
        return await _handle_smart_db_query(arguments)

    elif name == "analyze_db_data":
        return await _handle_analyze_db_data(arguments)

    elif name == "get_db_schema":
        return await _handle_get_db_schema(arguments)

    elif name == "web_search":
        return await _handle_web_search(arguments)

    elif name == "database_query":
        return await _handle_database_query(arguments)

    elif name == "api_call":
        return await _handle_api_call(arguments)

    elif name == "calculator":
        return await _handle_calculator(arguments)

    elif name == "check_calendar_conflicts":
        return await _handle_check_calendar_conflicts(arguments)

    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"알 수 없는 도구: {name}")],
            isError=True
        )


async def _handle_smart_db_query(arguments: dict) -> CallToolResult:
    """자연어 → SQL 변환 및 실행"""
    query = arguments.get("query", "")
    dry_run = arguments.get("dry_run", False)

    try:
        # 자연어 분석하여 SQL 생성
        sql_info = _natural_language_to_sql(query)

        if sql_info.get("error"):
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({
                    "error": sql_info["error"],
                    "hint": sql_info.get("hint", "")
                }, ensure_ascii=False, indent=2))],
                isError=True
            )

        result = {
            "natural_query": query,
            "generated_sql": sql_info["sql"],
            "table": sql_info["table"],
            "operation": sql_info["operation"],
            "explanation": sql_info["explanation"],
        }

        if dry_run:
            result["dry_run"] = True
            result["message"] = "SQL이 생성되었습니다. dry_run=False로 설정하면 실행됩니다."
        else:
            # 실제 DB 실행 (현재는 mock)
            execution_result = await _execute_sql_query(sql_info["sql"], sql_info["table"])
            result["execution_result"] = execution_result

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2, default=str)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"쿼리 처리 오류: {str(e)}")],
            isError=True
        )


def _natural_language_to_sql(query: str) -> dict:
    """자연어 쿼리를 SQL로 변환"""
    query_lower = query.lower()

    # 테이블 감지
    table_keywords = {
        "chat_sessions": ["세션", "session", "대화", "채팅", "chat"],
        "chat_messages": ["메시지", "message", "응답", "질문"],
        "tool_logs": ["도구", "tool", "로그", "log", "호출"],
        "users": ["사용자", "user", "유저", "회원"],
        "calendars": ["캘린더", "calendar", "달력"],
        "calendar_events": ["이벤트", "event", "일정", "약속"],
    }

    detected_table = None
    for table, keywords in table_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected_table = table
            break

    if not detected_table:
        return {
            "error": "테이블을 감지할 수 없습니다.",
            "hint": "다음 키워드 중 하나를 포함해주세요: 세션, 메시지, 도구/로그, 사용자, 캘린더, 이벤트"
        }

    # 연산 유형 감지
    operation = "SELECT"
    if any(kw in query_lower for kw in ["개수", "몇 개", "count", "몇개"]):
        operation = "COUNT"
    elif any(kw in query_lower for kw in ["합계", "총", "sum", "total"]):
        operation = "SUM"
    elif any(kw in query_lower for kw in ["평균", "average", "avg"]):
        operation = "AVG"
    elif any(kw in query_lower for kw in ["최대", "max", "가장 큰", "가장 많은"]):
        operation = "MAX"
    elif any(kw in query_lower for kw in ["최소", "min", "가장 작은", "가장 적은"]):
        operation = "MIN"

    # 시간 조건 감지
    time_condition = None
    today = datetime.now().date()

    if "오늘" in query_lower or "today" in query_lower:
        time_condition = f"DATE(created_at) = '{today}'"
    elif "어제" in query_lower or "yesterday" in query_lower:
        yesterday = today - timedelta(days=1)
        time_condition = f"DATE(created_at) = '{yesterday}'"
    elif "이번 주" in query_lower or "this week" in query_lower:
        week_start = today - timedelta(days=today.weekday())
        time_condition = f"DATE(created_at) >= '{week_start}'"
    elif "이번 달" in query_lower or "this month" in query_lower:
        month_start = today.replace(day=1)
        time_condition = f"DATE(created_at) >= '{month_start}'"
    elif "최근" in query_lower:
        # 최근 N일 파싱
        days_match = re.search(r"최근\s*(\d+)\s*일", query)
        if days_match:
            days = int(days_match.group(1))
            past_date = today - timedelta(days=days)
            time_condition = f"DATE(created_at) >= '{past_date}'"
        else:
            time_condition = f"DATE(created_at) >= '{today - timedelta(days=7)}'"

    # 정렬 조건 감지
    order_clause = ""
    if any(kw in query_lower for kw in ["최신", "최근", "newest", "latest"]):
        order_clause = "ORDER BY created_at DESC"
    elif any(kw in query_lower for kw in ["오래된", "oldest"]):
        order_clause = "ORDER BY created_at ASC"

    # LIMIT 감지
    limit_clause = ""
    limit_match = re.search(r"(?:top|상위|최대)\s*(\d+)", query_lower)
    if limit_match:
        limit_clause = f"LIMIT {limit_match.group(1)}"
    elif any(kw in query_lower for kw in ["첫", "first", "하나"]):
        limit_clause = "LIMIT 1"
    else:
        limit_clause = "LIMIT 100"  # 기본 제한

    # 특정 조건 감지
    where_conditions = []
    if time_condition:
        where_conditions.append(time_condition)

    # role 조건 (chat_messages)
    if detected_table == "chat_messages":
        if "user" in query_lower or "사용자" in query_lower:
            where_conditions.append("role = 'user'")
        elif "assistant" in query_lower or "어시스턴트" in query_lower or "ai" in query_lower:
            where_conditions.append("role = 'assistant'")

    # status 조건 (tool_logs)
    if detected_table == "tool_logs":
        if any(kw in query_lower for kw in ["성공", "success"]):
            where_conditions.append("status = 'success'")
        elif any(kw in query_lower for kw in ["실패", "fail", "error"]):
            where_conditions.append("status = 'failed'")

    # SQL 생성
    where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""

    # 집계 컬럼 감지
    agg_column = "*"
    if operation in ["SUM", "AVG", "MAX", "MIN"]:
        if detected_table == "chat_sessions":
            if "토큰" in query_lower or "token" in query_lower:
                agg_column = "total_tokens_used"
            elif "메시지" in query_lower:
                agg_column = "total_messages"
        elif detected_table == "chat_messages":
            if "토큰" in query_lower or "token" in query_lower:
                agg_column = "output_tokens"
            elif "시간" in query_lower or "time" in query_lower:
                agg_column = "generation_time"
        elif detected_table == "tool_logs":
            if "시간" in query_lower or "time" in query_lower:
                agg_column = "execution_time_ms"

    if operation == "COUNT":
        sql = f"SELECT COUNT(*) as count FROM {detected_table} {where_clause}"
    elif operation in ["SUM", "AVG", "MAX", "MIN"]:
        sql = f"SELECT {operation}({agg_column}) as result FROM {detected_table} {where_clause}"
    else:
        sql = f"SELECT * FROM {detected_table} {where_clause} {order_clause} {limit_clause}".strip()

    # 설명 생성
    explanation = f"'{detected_table}' 테이블에서 "
    if where_conditions:
        explanation += f"조건({', '.join(where_conditions)})에 맞는 "
    if operation == "COUNT":
        explanation += "개수를 조회합니다."
    elif operation in ["SUM", "AVG", "MAX", "MIN"]:
        explanation += f"{agg_column}의 {operation} 값을 계산합니다."
    else:
        explanation += "데이터를 조회합니다."

    return {
        "sql": sql,
        "table": detected_table,
        "operation": operation,
        "explanation": explanation,
    }


async def _execute_sql_query(sql: str, table: str) -> dict:
    """SQL 쿼리 실행 (실제 DB 연동)"""
    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql))

            # SELECT 쿼리인 경우
            if sql.strip().upper().startswith("SELECT"):
                rows = result.fetchall()
                columns = result.keys()

                # 결과를 딕셔너리 리스트로 변환
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # datetime 객체 처리
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        row_dict[col] = value
                    data.append(row_dict)

                return {
                    "success": True,
                    "row_count": len(data),
                    "columns": list(columns),
                    "data": data[:100],  # 최대 100개로 제한
                    "truncated": len(data) > 100
                }
            else:
                await session.commit()
                return {
                    "success": True,
                    "affected_rows": result.rowcount,
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def _handle_analyze_db_data(arguments: dict) -> CallToolResult:
    """DB 데이터 분석 및 통계"""
    table = arguments.get("table", "")
    analysis_type = arguments.get("analysis_type", "statistics")
    group_by = arguments.get("group_by")
    time_range = arguments.get("time_range", "all")

    if table not in DB_SCHEMA:
        return CallToolResult(
            content=[TextContent(type="text", text=f"알 수 없는 테이블: {table}")],
            isError=True
        )

    try:
        # 시간 범위 조건 생성
        time_condition = _get_time_condition(time_range)

        # 분석 유형별 쿼리 및 분석 수행
        if analysis_type == "statistics":
            result = await _analyze_statistics(table, time_condition, group_by)
        elif analysis_type == "trend":
            result = await _analyze_trend(table, time_condition)
        elif analysis_type == "comparison":
            result = await _analyze_comparison(table, time_condition, group_by)
        elif analysis_type == "distribution":
            result = await _analyze_distribution(table, time_condition, group_by)
        else:
            result = {"error": f"지원하지 않는 분석 유형: {analysis_type}"}

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "table": table,
                    "analysis_type": analysis_type,
                    "time_range": time_range,
                    "result": result,
                    "analyzed_at": datetime.now().isoformat(),
                }, ensure_ascii=False, indent=2, default=str)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"분석 오류: {str(e)}")],
            isError=True
        )


def _get_time_condition(time_range: str) -> str:
    """시간 범위 조건 생성"""
    today = datetime.now().date()

    conditions = {
        "today": f"DATE(created_at) = '{today}'",
        "this_week": f"DATE(created_at) >= '{today - timedelta(days=today.weekday())}'",
        "this_month": f"DATE(created_at) >= '{today.replace(day=1)}'",
        "all": "",
    }
    return conditions.get(time_range, "")


async def _analyze_statistics(table: str, time_condition: str, group_by: Optional[str]) -> dict:
    """기본 통계 분석"""
    where_clause = f"WHERE {time_condition}" if time_condition else ""

    # 테이블별 통계 쿼리
    stats_queries = {
        "chat_sessions": [
            ("total_count", f"SELECT COUNT(*) FROM {table} {where_clause}"),
            ("avg_messages", f"SELECT AVG(total_messages) FROM {table} {where_clause}"),
            ("total_tokens", f"SELECT SUM(total_tokens_used) FROM {table} {where_clause}"),
            ("avg_tokens", f"SELECT AVG(total_tokens_used) FROM {table} {where_clause}"),
        ],
        "chat_messages": [
            ("total_count", f"SELECT COUNT(*) FROM {table} {where_clause}"),
            ("user_messages", f"SELECT COUNT(*) FROM {table} {where_clause} {'AND' if time_condition else 'WHERE'} role = 'user'"),
            ("assistant_messages", f"SELECT COUNT(*) FROM {table} {where_clause} {'AND' if time_condition else 'WHERE'} role = 'assistant'"),
            ("avg_generation_time", f"SELECT AVG(generation_time) FROM {table} {where_clause} {'AND' if time_condition else 'WHERE'} role = 'assistant'"),
        ],
        "tool_logs": [
            ("total_count", f"SELECT COUNT(*) FROM {table} {where_clause}"),
            ("success_count", f"SELECT COUNT(*) FROM {table} {where_clause} {'AND' if time_condition else 'WHERE'} status = 'success'"),
            ("failed_count", f"SELECT COUNT(*) FROM {table} {where_clause} {'AND' if time_condition else 'WHERE'} status = 'failed'"),
            ("avg_execution_time", f"SELECT AVG(execution_time_ms) FROM {table} {where_clause}"),
        ],
    }

    # 기본 통계 (모든 테이블)
    default_queries = [
        ("total_count", f"SELECT COUNT(*) FROM {table} {where_clause}"),
    ]

    queries = stats_queries.get(table, default_queries)
    results = {}

    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            for stat_name, sql in queries:
                result = await session.execute(text(sql))
                value = result.scalar()
                results[stat_name] = round(value, 2) if isinstance(value, float) else value
    except Exception as e:
        results["error"] = str(e)

    return results


async def _analyze_trend(table: str, time_condition: str) -> dict:
    """시간별 트렌드 분석"""
    where_clause = f"WHERE {time_condition}" if time_condition else ""

    sql = f"""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM {table} {where_clause}
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        LIMIT 30
    """

    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql))
            rows = result.fetchall()

            trend_data = [{"date": str(row[0]), "count": row[1]} for row in rows]

            # 트렌드 계산 (증가/감소)
            if len(trend_data) >= 2:
                recent = sum(r["count"] for r in trend_data[:7])
                previous = sum(r["count"] for r in trend_data[7:14]) if len(trend_data) >= 14 else recent
                trend = "increasing" if recent > previous else "decreasing" if recent < previous else "stable"
                change_rate = ((recent - previous) / previous * 100) if previous > 0 else 0
            else:
                trend = "insufficient_data"
                change_rate = 0

            return {
                "daily_counts": trend_data,
                "trend": trend,
                "change_rate_7d": round(change_rate, 2),
            }
    except Exception as e:
        return {"error": str(e)}


async def _analyze_comparison(table: str, time_condition: str, group_by: Optional[str]) -> dict:
    """그룹별 비교 분석"""
    where_clause = f"WHERE {time_condition}" if time_condition else ""

    # 테이블별 기본 그룹화 컬럼
    default_group_by = {
        "chat_sessions": "model_name",
        "chat_messages": "role",
        "tool_logs": "tool_name",
    }

    group_col = group_by or default_group_by.get(table, "id")

    sql = f"""
        SELECT {group_col}, COUNT(*) as count
        FROM {table} {where_clause}
        GROUP BY {group_col}
        ORDER BY count DESC
        LIMIT 20
    """

    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql))
            rows = result.fetchall()

            comparison_data = [{"group": str(row[0]), "count": row[1]} for row in rows]
            total = sum(r["count"] for r in comparison_data)

            # 비율 계산
            for item in comparison_data:
                item["percentage"] = round(item["count"] / total * 100, 2) if total > 0 else 0

            return {
                "group_by": group_col,
                "groups": comparison_data,
                "total": total,
            }
    except Exception as e:
        return {"error": str(e)}


async def _analyze_distribution(table: str, time_condition: str, group_by: Optional[str]) -> dict:
    """데이터 분포 분석"""
    where_clause = f"WHERE {time_condition}" if time_condition else ""

    # 테이블별 분포 분석 컬럼
    distribution_columns = {
        "chat_sessions": "total_messages",
        "chat_messages": "output_tokens",
        "tool_logs": "execution_time_ms",
    }

    dist_col = group_by or distribution_columns.get(table)

    if not dist_col:
        return {"error": "분포 분석을 위한 수치형 컬럼이 없습니다."}

    sql = f"""
        SELECT {dist_col}
        FROM {table} {where_clause}
        WHERE {dist_col} IS NOT NULL
    """

    try:
        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql))
            values = [row[0] for row in result.fetchall() if row[0] is not None]

            if not values:
                return {"error": "분석할 데이터가 없습니다."}

            # 기본 통계
            values.sort()
            n = len(values)
            mean_val = sum(values) / n
            min_val = values[0]
            max_val = values[-1]
            median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2

            # 분위수
            q1 = values[n // 4]
            q3 = values[3 * n // 4]

            # 히스토그램 (10개 구간)
            bucket_size = (max_val - min_val) / 10 if max_val > min_val else 1
            histogram = Counter()
            for v in values:
                bucket = int((v - min_val) / bucket_size) if bucket_size > 0 else 0
                bucket = min(bucket, 9)  # 최대 9번 버킷
                histogram[bucket] += 1

            histogram_data = [
                {
                    "range": f"{min_val + i * bucket_size:.0f}-{min_val + (i + 1) * bucket_size:.0f}",
                    "count": histogram.get(i, 0)
                }
                for i in range(10)
            ]

            return {
                "column": dist_col,
                "count": n,
                "min": min_val,
                "max": max_val,
                "mean": round(mean_val, 2),
                "median": median,
                "q1": q1,
                "q3": q3,
                "histogram": histogram_data,
            }
    except Exception as e:
        return {"error": str(e)}


async def _handle_get_db_schema(arguments: dict) -> CallToolResult:
    """DB 스키마 정보 조회"""
    table_name = arguments.get("table_name")

    try:
        if table_name:
            if table_name not in DB_SCHEMA:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"알 수 없는 테이블: {table_name}")],
                    isError=True
                )
            result = {
                "table": table_name,
                "schema": DB_SCHEMA[table_name]
            }
        else:
            result = {
                "tables": list(DB_SCHEMA.keys()),
                "schemas": {
                    table: {
                        "description": info["description"],
                        "column_count": len(info["columns"])
                    }
                    for table, info in DB_SCHEMA.items()
                }
            }

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"스키마 조회 오류: {str(e)}")],
            isError=True
        )


async def _handle_web_search(arguments: dict) -> CallToolResult:
    """웹 검색 처리"""
    query = arguments.get("query", "")
    num_results = arguments.get("num_results", 5)

    try:
        # TODO: 실제 검색 API 연동 (Google, Bing, SerpAPI 등)
        # 현재는 목업 응답
        results = [
            {"title": f"검색 결과 {i+1}: {query}", "snippet": f"{query}에 대한 정보입니다.", "url": f"https://example.com/result/{i+1}"}
            for i in range(num_results)
        ]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"검색 오류: {str(e)}")],
            isError=True
        )


async def _handle_database_query(arguments: dict) -> CallToolResult:
    """DB 조회 처리"""
    table = arguments.get("table", "")
    conditions = arguments.get("conditions", {})
    limit = arguments.get("limit", 10)

    # 보안: 허용된 테이블만 조회
    allowed_tables = ["chat_sessions", "chat_messages", "users"]
    if table not in allowed_tables:
        return CallToolResult(
            content=[TextContent(type="text", text=f"허용되지 않은 테이블: {table}")],
            isError=True
        )

    try:
        # TODO: 실제 DB 연동
        # 현재는 목업 응답
        mock_data = [
            {"id": i, "table": table, "sample": "data"}
            for i in range(min(limit, 3))
        ]

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "table": table,
                    "conditions": conditions,
                    "count": len(mock_data),
                    "data": mock_data
                }, ensure_ascii=False, indent=2)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"DB 조회 오류: {str(e)}")],
            isError=True
        )


async def _handle_api_call(arguments: dict) -> CallToolResult:
    """외부 API 호출 처리"""
    url = arguments.get("url", "")
    method = arguments.get("method", "GET").upper()
    headers = arguments.get("headers", {})
    body = arguments.get("body")

    # 보안: 허용된 도메인 검증 (필요시 활성화)
    # from urllib.parse import urlparse
    # allowed_domains = ["api.example.com"]
    # domain = urlparse(url).netloc
    # if domain not in allowed_domains:
    #     return CallToolResult(
    #         content=[TextContent(type="text", text=f"허용되지 않은 도메인: {domain}")],
    #         isError=True
    #     )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                response = await client.get(url, headers=headers)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=body)
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"지원하지 않는 메서드: {method}")],
                    isError=True
                )

            # 응답 처리
            try:
                result = response.json()
            except:
                result = response.text[:2000]  # 텍스트는 2000자 제한

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "status_code": response.status_code,
                        "data": result
                    }, ensure_ascii=False, indent=2)
                )]
            )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"API 호출 오류: {str(e)}")],
            isError=True
        )


async def _handle_calculator(arguments: dict) -> CallToolResult:
    """수학 계산 처리"""
    expression = arguments.get("expression", "")

    try:
        # 보안: 허용된 함수만 사용
        allowed_names = {
            "abs": abs, "round": round,
            "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)

        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "expression": expression,
                    "result": result
                }, ensure_ascii=False)
            )]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"계산 오류: {str(e)}")],
            isError=True
        )


async def _handle_check_calendar_conflicts(arguments: dict) -> CallToolResult:
    """일정 충돌 감지"""
    user_id = arguments.get("user_id")
    start_date = arguments.get("start_date")
    start_time = arguments.get("start_time", "00:00")
    end_date = arguments.get("end_date")
    end_time = arguments.get("end_time", "23:59")

    try:
        # 날짜/시간 문자열을 datetime으로 변환
        new_start = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        new_end = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")

        from app.core.database import AsyncSessionLocal
        from sqlalchemy import text

        async with AsyncSessionLocal() as session:
            # 충돌 조건: (기존.start < 새.end) AND (기존.end > 새.start)
            # 사용자의 캘린더에 있는 이벤트만 조회
            sql = text("""
                SELECT 
                    ce.id,
                    ce.title,
                    ce.description,
                    ce.start_date,
                    ce.start_time,
                    ce.end_date,
                    ce.end_time,
                    ce.is_all_day,
                    ce.location
                FROM calendar_events ce
                JOIN calendars c ON ce.calendar_id = c.id
                WHERE c.owner_id = :user_id
                  AND ce.status = 'scheduled'
                  AND (
                    CASE 
                      WHEN ce.is_all_day = 1 THEN
                        DATE(ce.start_date) <= DATE(:new_end_date) 
                        AND DATE(ce.end_date) >= DATE(:new_start_date)
                      ELSE
                        TIMESTAMP(ce.start_date, COALESCE(ce.start_time, '00:00:00')) < :new_end
                        AND TIMESTAMP(ce.end_date, COALESCE(ce.end_time, '23:59:59')) > :new_start
                    END
                  )
                ORDER BY ce.start_date, ce.start_time
                LIMIT 20
            """)

            result = await session.execute(sql, {
                "user_id": user_id,
                "new_start": new_start,
                "new_end": new_end,
                "new_start_date": start_date,
                "new_end_date": end_date,
            })
            rows = result.fetchall()

            # 충돌하는 이벤트 목록 생성
            conflicting_events = []
            for row in rows:
                event = {
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "start_date": str(row[3]) if row[3] else None,
                    "start_time": str(row[4]) if row[4] else None,
                    "end_date": str(row[5]) if row[5] else None,
                    "end_time": str(row[6]) if row[6] else None,
                    "is_all_day": bool(row[7]),
                    "location": row[8],
                }
                conflicting_events.append(event)

            response = {
                "has_conflict": len(conflicting_events) > 0,
                "conflict_count": len(conflicting_events),
                "conflicting_events": conflicting_events,
                "checked_period": {
                    "start": f"{start_date} {start_time}",
                    "end": f"{end_date} {end_time}",
                },
                "user_id": user_id,
            }

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps(response, ensure_ascii=False, indent=2, default=str)
                )]
            )
    except ValueError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"날짜/시간 형식 오류: {str(e)}. YYYY-MM-DD HH:MM 형식을 사용하세요.")],
            isError=True
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"충돌 감지 오류: {str(e)}")],
            isError=True
        )


# ============== 서버 실행 ==============

async def main():
    """MCP 서버 실행 (stdio 모드)"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
