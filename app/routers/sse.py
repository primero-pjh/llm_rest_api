"""
SSE (Server-Sent Events) Router
===============================
실시간 서버 푸시 이벤트를 처리하는 라우터입니다.
LLM 스트리밍 응답을 통한 텍스트 요약 기능을 제공합니다.
"""

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel
from typing import AsyncGenerator, Optional
import asyncio
import json
import re
from datetime import datetime
from app.services.llm_service import llm_service

router = APIRouter(
    prefix="/sse",
    tags=["SSE"],
)


@router.get("/extract-schedule-with-conflict")
async def extract_schedule_with_conflict(
    request: Request,
    text: str = Query(..., description="일정을 추출할 대화 내용/회의록"),
    user_id: int = Query(..., description="사용자 ID (충돌 감지용)"),
    max_tokens: Optional[int] = Query(1024, description="최대 토큰 수"),
    temperature: Optional[float] = Query(0.1, description="생성 온도")
):
    """
    텍스트에서 일정을 추출하고 기존 캘린더와의 충돌을 확인하는 통합 엔드포인트

    Query Parameters:
        - text (str): 일정이 포함된 대화 내용/회의록
        - user_id (int): 충돌을 확인할 사용자 ID
        - max_tokens (int, optional): 최대 토큰 수 (기본값: 1024)
        - temperature (float, optional): 생성 온도 (기본값: 0.1)

    Returns:
        SSE 스트림으로 다음 이벤트들을 반환:
        - start: 추출 시작 알림
        - message: 생성 중인 토큰 (실시간)
        - schedule: 추출된 개별 일정
        - conflict: 충돌 정보
        - complete: 처리 완료 알림
        - error: 에러 발생 시
    """
    if not llm_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": "모델이 로드되지 않았습니다.",
                "hint": "서버가 시작 중이거나 모델 로드에 실패했습니다."
            }
        )

    async def schedule_conflict_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """일정 추출 및 충돌 감지 스트리밍 제너레이터"""
        token_count = 0
        full_response = ""

        try:
            # 시스템 프롬프트
            system_prompt = """You are an AI assistant specialized in extracting schedule information from Korean text.
Extract all schedules from the given text and return them as a JSON array.

Return format:
[
  {
    "title": "일정 제목",
    "description": "상세 설명",
    "startDate": "YYYY-MM-DD",
    "startTime": "HH:MM",
    "endDate": "YYYY-MM-DD",
    "endTime": "HH:MM",
    "location": "장소 (있는 경우)",
    "attendees": ["참석자1", "참석자2"]
  }
]

Important:
- Extract ALL schedules from the text
- Use relative date expressions (내일, 다음주 등) converted to actual dates based on today
- If time is not specified, use "09:00" for start and "10:00" for end
- If end date/time is not specified, assume 1 hour duration
- Return ONLY valid JSON array, no additional text
- If no schedules found, return empty array: []"""

            user_prompt = f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}입니다.\n\n다음 텍스트에서 모든 일정을 추출하세요:\n\n{text}"

            # 시작 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "start",
                    "message": "일정 추출 시작",
                    "text_length": len(text),
                    "user_id": user_id,
                    "model": llm_service.model_id,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # LLM 스트리밍 생성
            for token in llm_service.generate_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                if await request.is_disconnected():
                    break

                token_count += 1
                full_response += token

                yield ServerSentEvent(
                    data=json.dumps({
                        "token": token,
                        "full_text": full_response,
                        "token_count": token_count
                    }, ensure_ascii=False),
                    event="message",
                    id=str(token_count)
                )

                await asyncio.sleep(0)

            # JSON 파싱 시도
            try:
                # JSON 배열 추출 (```json ... ``` 또는 [ ... ] 형태)
                json_match = re.search(r'\[[\s\S]*\]', full_response)
                if json_match:
                    schedules = json.loads(json_match.group())
                else:
                    schedules = []
            except json.JSONDecodeError:
                schedules = []

            # 각 일정에 대해 충돌 확인
            from app.core.database import AsyncSessionLocal
            from sqlalchemy import text

            all_conflicts = []

            async with AsyncSessionLocal() as session:
                for idx, schedule in enumerate(schedules):
                    start_date = schedule.get("startDate", "")
                    start_time = schedule.get("startTime", "09:00")
                    end_date = schedule.get("endDate", start_date)
                    end_time = schedule.get("endTime", "10:00")

                    if not start_date:
                        continue

                    # 일정 이벤트 전송
                    yield ServerSentEvent(
                        data=json.dumps({
                            "index": idx,
                            "schedule": schedule
                        }, ensure_ascii=False),
                        event="schedule"
                    )

                    # 충돌 확인 쿼리
                    try:
                        new_start = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                        new_end = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")

                        sql = text("""
                            SELECT 
                                ce.id, ce.title, ce.start_date, ce.start_time, 
                                ce.end_date, ce.end_time, ce.is_all_day
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
                            LIMIT 10
                        """)

                        result = await session.execute(sql, {
                            "user_id": user_id,
                            "new_start": new_start,
                            "new_end": new_end,
                            "new_start_date": start_date,
                            "new_end_date": end_date,
                        })
                        conflicts = result.fetchall()

                        if conflicts:
                            conflict_events = []
                            for row in conflicts:
                                conflict_events.append({
                                    "id": row[0],
                                    "title": row[1],
                                    "start_date": str(row[2]) if row[2] else None,
                                    "start_time": str(row[3]) if row[3] else None,
                                    "end_date": str(row[4]) if row[4] else None,
                                    "end_time": str(row[5]) if row[5] else None,
                                    "is_all_day": bool(row[6])
                                })

                            conflict_info = {
                                "schedule_index": idx,
                                "schedule_title": schedule.get("title", ""),
                                "has_conflict": True,
                                "conflicting_events": conflict_events
                            }
                            all_conflicts.append(conflict_info)

                            # 충돌 이벤트 전송
                            yield ServerSentEvent(
                                data=json.dumps(conflict_info, ensure_ascii=False, default=str),
                                event="conflict"
                            )
                    except Exception as e:
                        print(f"충돌 확인 오류: {e}")

            # 완료 이벤트
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "complete",
                    "message": "일정 추출 및 충돌 확인 완료",
                    "total_schedules": len(schedules),
                    "total_conflicts": len(all_conflicts),
                    "schedules": schedules,
                    "conflicts": all_conflicts,
                    "total_tokens": token_count,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False, default=str),
                event="complete"
            )

        except Exception as e:
            yield ServerSentEvent(
                data=json.dumps({
                    "error": str(e),
                    "type": type(e).__name__
                }, ensure_ascii=False),
                event="error"
            )

        except asyncio.CancelledError:
            print(f"일정 추출 스트리밍 취소됨. 생성된 토큰 수: {token_count}")

    return EventSourceResponse(schedule_conflict_generator())




@router.get("/extract-schedule")
async def extract_schedule_stream(
    request: Request,
    text: str,
    max_tokens: Optional[int] = 1024,
    temperature: Optional[float] = 0.1
):
    """
    텍스트에서 일정을 추출하여 JSON 형태로 반환하는 엔드포인트

    Query Parameters:
        - text (str): 일정이 포함된 텍스트
        - max_tokens (int, optional): 최대 토큰 수 (기본값: 1024)

    Returns:
        SSE 스트림으로 다음 이벤트들을 반환:
        - start: 추출 시작 알림
        - token: 생성된 토큰 (실시간)
        - complete: 추출 완료 알림
        - error: 에러 발생 시
    """
    if not llm_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": "모델이 로드되지 않았습니다.",
                "hint": "서버가 시작 중이거나 모델 로드에 실패했습니다."
            }
        )

    async def schedule_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """
        일정 추출 스트리밍 제너레이터

        텍스트에서 일정을 추출하여 JSON 형태로 반환합니다.
        """
        token_count = 0
        full_response = ""

        try:
            # 일정 추출을 위한 시스템 프롬프트
            system_prompt = """You are an AI assistant specialized in extracting schedule information from text.
Extract all schedules from the given text and return them as a JSON array.

Return format:
[
  {
    "title": "event title",
    "description": "detailed description",
    "startDate": "YYYY-MM-DD or YYYY-MM-DD HH:MM",
    "endDate": "YYYY-MM-DD or YYYY-MM-DD HH:MM"
  }
]

Important:
- Extract ALL schedules from the text
- If no date is specified, use null for startDate/endDate
- If only start date is mentioned, endDate can be the same as startDate
- Return ONLY valid JSON array, no additional text
- If no schedules found, return empty array: []"""

            # 사용자 프롬프트
            user_prompt = f"Extract all schedules from the following text:\n\n{text}"

            # 시작 이벤트 전송
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "start",
                    "message": "일정 추출 시작",
                    "text_length": len(text),
                    "model": llm_service.model_id,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"
            )

            # LLM 스트리밍 생성 (temperature=1.0으로 창의적 생성)
            for token in llm_service.generate_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                # 클라이언트 연결 끊김 확인
                if await request.is_disconnected():
                    break

                token_count += 1
                full_response += token

                # 각 토큰마다 SSE 이벤트 전송
                yield ServerSentEvent(
                    data=json.dumps({
                        "token": token,
                        "full_text": full_response,
                        "token_count": token_count
                    }, ensure_ascii=False),
                    event="message",
                    id=str(token_count)
                )

                await asyncio.sleep(0)

            # 완료 이벤트 전송
            yield ServerSentEvent(
                data=json.dumps({
                    "type": "complete",
                    "message": "일정 추출 완료",
                    "full_response": full_response,
                    "total_tokens": token_count,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="complete"
            )

        except Exception as e:
            # 에러 이벤트 전송
            yield ServerSentEvent(
                data=json.dumps({
                    "error": str(e),
                    "type": type(e).__name__
                }, ensure_ascii=False),
                event="error"
            )

        except asyncio.CancelledError:
            print(f"일정 추출 스트리밍 취소됨. 생성된 토큰 수: {token_count}")

    return EventSourceResponse(schedule_generator())


@router.get("/summary")
async def sse_summary_stream(
    request: Request,
    summary: str,
    max_tokens: Optional[int] = 512,
    temperature: Optional[float] = 0.7
):
    """
    텍스트 요약 스트리밍 엔드포인트

    Query Parameters:
        - summary (str): LLM에 전달할 요약 텍스트
        - max_tokens (int, optional): 최대 토큰 수 (기본값: 512)
        - temperature (float, optional): 생성 다양성 (기본값: 0.7)

    Returns:
        SSE 스트림으로 다음 이벤트들을 반환:
        - start: 요약 시작 알림
        - token: 생성된 토큰 (실시간)
        - complete: 요약 완료 알림
        - error: 에러 발생 시
    """
    if not llm_service.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": "모델이 로드되지 않았습니다.",
                "hint": "서버가 시작 중이거나 모델 로드에 실패했습니다."
            }
        )

    async def summary_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """
        SSE 스트리밍 제너레이터

        LLM이 생성하는 토큰을 실시간으로 클라이언트에 전송합니다.
        yield를 사용하여 이벤트를 하나씩 스트리밍합니다.
        """
        # 토큰 개수와 전체 응답 텍스트를 추적하는 변수
        token_count = 0
        full_response = ""

        try:
            # 1단계: 시작 이벤트 전송
            # 클라이언트에게 요약 처리가 시작되었음을 알림
            yield ServerSentEvent(
                data=json.dumps({
                    "message": "요약 처리 시작",
                    "text_length": len(summary),
                    "model": llm_service.model_id,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="start"  # 이벤트 타입: start
            )

            # 2단계: LLM 스트리밍 생성
            # llm_service.generate_stream()은 제너레이터로, 토큰을 하나씩 생성
            for token in llm_service.generate_stream(
                prompt=summary,  # 사용자가 보낸 summary 텍스트
                system_prompt="You are a helpful assistant.",  # LLM 역할 정의
                max_new_tokens=max_tokens,  # 최대 생성 토큰 수
                temperature=temperature,  # 생성 다양성 (0.0=결정적, 1.0=창의적)
            ):
                # 클라이언트 연결 끊김 확인
                if await request.is_disconnected():
                    break

                # 토큰 카운트 증가 및 전체 응답에 추가
                token_count += 1
                full_response += token

                # 각 토큰마다 SSE 이벤트 전송 (실시간 스트리밍)
                yield ServerSentEvent(
                    data=json.dumps({
                        "token": token,  # 현재 생성된 토큰
                        "full_text": full_response,  # 지금까지 생성된 전체 텍스트
                        "token_count": token_count  # 현재까지 생성된 토큰 수
                    }, ensure_ascii=False),
                    event="message",  # 이벤트 타입: token
                    id=str(token_count)  # 이벤트 ID (SSE 표준)
                )

                # 비동기 이벤트 루프에 제어권 양보 (다른 작업 처리 가능하도록)
                await asyncio.sleep(0)

            # 3단계: 완료 이벤트 전송
            # 모든 토큰 생성이 끝났음을 클라이언트에 알림
            yield ServerSentEvent(
                data=json.dumps({
                    "message": "요약 처리 완료",
                    "full_response": full_response,  # 최종 완성된 전체 응답
                    "total_tokens": token_count,  # 총 생성된 토큰 수
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False),
                event="complete"  # 이벤트 타입: complete
            )

        except Exception as e:
            # 에러 발생 시 에러 이벤트 전송
            yield ServerSentEvent(
                data=json.dumps({
                    "error": str(e),  # 에러 메시지
                    "type": type(e).__name__  # 에러 타입 (예: ValueError, RuntimeError)
                }, ensure_ascii=False),
                event="error"  # 이벤트 타입: error
            )

        except asyncio.CancelledError:
            # 클라이언트가 연결을 끊었을 때 로그 출력
            print(f"요약 스트리밍 취소됨. 생성된 토큰 수: {token_count}")

    return EventSourceResponse(summary_generator())
