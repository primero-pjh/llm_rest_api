# LLM REST API - Claude Code Guide

## 프로젝트 개요
FastAPI 기반 LLM 챗봇 REST API 서버

## 기술 스택
- **Framework**: FastAPI
- **Database**: MySQL (async SQLAlchemy)
- **LLM**: Hugging Face Transformers
  - Bllossom/llama-3.2-Korean-Bllossom-3B (기본값)
  - kakaocorp/kanana-2-30b-a3b-instruct (Kakao Kanana-2)
- **Streaming**: SSE (Server-Sent Events)

## 프로젝트 구조
```
llm_rest_api/
├── main.py                    # FastAPI 앱 엔트리포인트
├── app/
│   ├── core/
│   │   ├── config.py          # 설정
│   │   ├── database.py        # DB 연결
│   │   └── request_logger.py  # 요청 로깅 미들웨어
│   ├── models/
│   │   ├── user.py            # User 모델
│   │   ├── calendar.py        # Calendar 모델
│   │   └── chat.py            # ChatSession, ChatMessage 모델
│   ├── routers/
│   │   ├── terminal.py        # 터미널 라우터
│   │   ├── sse.py             # SSE 라우터 (일정추출, 요약)
│   │   ├── chat.py            # 채팅 라우터
│   │   ├── monitor.py         # 모니터링 라우터
│   │   └── mysql_viewer.py    # MySQL 뷰어 라우터
│   ├── services/
│   │   ├── llm_service.py     # LLM 서비스 매니저 (다중 모델 라우팅)
│   │   ├── bllossom_service.py # Bllossom 모델 서비스
│   │   └── kanana_service.py  # Kanana-2 모델 서비스
│   └── templates/
│       └── mysql_viewer.html  # MySQL 뷰어 템플릿
```

## API 엔드포인트

### SSE 라우터 (`/sse`)
> **중요**: 모든 SSE 엔드포인트는 **GET 방식**을 사용합니다.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sse/extract-schedule` | 텍스트에서 일정 추출 (SSE) |
| GET | `/sse/summarize` | 텍스트 요약 (SSE) |

### Chat 라우터 (`/sse/chat`)
> **중요**: SSE 스트리밍 엔드포인트는 **GET 방식**을 사용합니다.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/sse/chat/sessions` | 새 채팅 세션 생성 |
| GET | `/sse/chat/sessions` | 세션 목록 조회 |
| GET | `/sse/chat/sessions/{id}` | 특정 세션 조회 |
| DELETE | `/sse/chat/sessions/{id}` | 세션 삭제 |
| GET | `/sse/chat/sessions/{id}/messages` | 세션 메시지 목록 조회 |
| **GET** | `/sse/chat/send` | **메시지 전송 및 LLM 응답 스트리밍 (SSE)** |

### MySQL Viewer (`/mysql`)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mysql` | MySQL 테이블 구조 뷰어 (HTML) |

### 기타
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | 헬스체크 |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

## 코드 컨벤션

### SSE 엔드포인트 규칙
1. **모든 SSE 스트리밍 엔드포인트는 GET 방식 사용**
2. 파라미터는 Query Parameters로 전달
3. EventSourceResponse 반환

```python
@router.get("/endpoint")
async def sse_endpoint(
    request: Request,
    param: str = Query(..., description="설명"),
    db: AsyncSession = Depends(get_db)
):
    async def generator():
        yield ServerSentEvent(data=json.dumps({...}), event="start")
        # ... streaming logic
        yield ServerSentEvent(data=json.dumps({...}), event="complete")

    return EventSourceResponse(generator())
```

### SSE 이벤트 타입
- `start`: 스트리밍 시작
- `message`: 토큰 생성 (스트리밍 중)
- `complete`: 스트리밍 완료
- `error`: 에러 발생

### Temperature 설정
- 최소값: 0.1 (0.3 미만은 greedy decoding 사용)
- 기본값: 0.7
- 최대값: 2.0

---

## Router Description 작성 규칙 (MCP Tool Calling용)

> **목적**: MCP Server의 tool_calling을 통해 LLM이 API를 자동으로 호출할 수 있도록, Router의 description을 상세하고 명확하게 작성합니다.

### 1. 필수 작성 원칙

#### 1.1 함수 docstring 작성 규칙
```python
@router.get("/endpoint")
async def endpoint_name(
    param1: str = Query(..., description="파라미터 설명"),
):
    """
    [한 줄 요약]: 이 API가 무엇을 하는지 한 문장으로 설명

    [상세 설명]:
    - 이 API의 목적과 사용 시나리오
    - 어떤 상황에서 이 API를 호출해야 하는지
    - 반환되는 데이터의 의미

    [사용 예시]:
    - "사용자가 일정을 추가하고 싶을 때 이 API를 호출"
    - "텍스트에서 날짜/시간 정보를 추출할 때 사용"

    Args:
        param1: 파라미터의 용도와 허용되는 값의 범위/형식
        param2: 기본값이 있는 경우 기본값의 의미도 설명

    Returns:
        반환 데이터의 구조와 각 필드의 의미

    Raises:
        발생 가능한 에러와 그 원인
    """
```

#### 1.2 Query Parameter description 규칙
```python
# 좋은 예시
message: str = Query(
    ...,
    description="LLM에게 전달할 사용자 메시지. 질문, 요청, 대화 내용 등 자유 형식의 텍스트를 입력합니다."
)

model_name: str = Query(
    "Bllossom/llama-3.2-Korean-Bllossom-3B",
    description="사용할 LLM 모델의 Hugging Face 모델 ID. 'Bllossom/llama-3.2-Korean-Bllossom-3B'(경량, 빠른 응답)와 'kakaocorp/kanana-2-30b-a3b-instruct'(고품질, 복잡한 작업) 중 선택 가능합니다."
)

# 나쁜 예시 (너무 간략함)
message: str = Query(..., description="메시지")
model_name: str = Query("model", description="모델명")
```

### 2. Description 작성 체크리스트

- [ ] **What**: API가 무엇을 하는가?
- [ ] **When**: 언제 이 API를 사용해야 하는가?
- [ ] **How**: 어떻게 파라미터를 구성해야 하는가?
- [ ] **Result**: 어떤 결과를 기대할 수 있는가?
- [ ] **Error**: 어떤 에러가 발생할 수 있는가?

### 3. MCP Tool Calling을 위한 권장 표현

| 상황 | 권장 표현 |
|------|-----------|
| 필수 파라미터 | "필수. ~를 입력합니다" |
| 선택 파라미터 | "선택. 기본값: X. ~할 때 사용합니다" |
| 열거형 값 | "가능한 값: A, B, C. A는 ~, B는 ~" |
| 범위 제한 | "범위: 최소 X ~ 최대 Y. 권장값: Z" |
| 형식 지정 | "형식: YYYY-MM-DD 또는 ISO 8601" |

### 4. 예시: Chat Send API

```python
@router.get("/send")
async def send_message(
    request: Request,
    message: str = Query(
        ...,
        description="LLM에게 전달할 사용자 메시지입니다. 질문, 요청, 대화 내용 등 자유 형식의 텍스트를 입력합니다. 한국어와 영어 모두 지원됩니다."
    ),
    model_name: str = Query(
        "Bllossom/llama-3.2-Korean-Bllossom-3B",
        description="사용할 LLM 모델의 Hugging Face 모델 ID입니다. 지원 모델: (1) 'Bllossom/llama-3.2-Korean-Bllossom-3B' - 3B 파라미터의 경량 모델로 빠른 응답이 필요할 때 적합합니다. (2) 'kakaocorp/kanana-2-30b-a3b-instruct' - 30B 파라미터의 고성능 모델로 복잡한 추론이나 고품질 응답이 필요할 때 적합합니다."
    ),
    system_prompt: str = Query(
        "You are a helpful assistant. Always respond in Korean.",
        description="LLM의 역할과 행동 방식을 정의하는 시스템 프롬프트입니다. AI의 페르소나, 응답 언어, 응답 스타일 등을 지정할 수 있습니다. 기본값은 한국어로 응답하는 도움이 되는 어시스턴트입니다."
    ),
    max_tokens: int = Query(
        1024,
        ge=1,
        le=4096,
        description="LLM이 생성할 최대 토큰 수입니다. 범위: 1-4096. 짧은 답변은 256, 일반 대화는 1024, 긴 설명이나 코드 생성은 2048-4096을 권장합니다."
    ),
    temperature: float = Query(
        0.7,
        ge=0.0,
        le=2.0,
        description="응답의 창의성/무작위성을 조절하는 값입니다. 범위: 0.0-2.0. 낮은 값(0.1-0.3)은 일관되고 결정적인 응답, 중간 값(0.5-0.8)은 균형 잡힌 응답, 높은 값(1.0-2.0)은 창의적이고 다양한 응답을 생성합니다. 사실 기반 질문은 0.3, 일반 대화는 0.7, 창작은 1.0을 권장합니다."
    ),
):
    """
    LLM에게 메시지를 전송하고 실시간 스트리밍 응답을 받습니다.

    이 API는 Server-Sent Events(SSE)를 통해 LLM의 응답을 토큰 단위로 실시간 스트리밍합니다.
    사용자의 질문에 대한 AI 응답, 텍스트 생성, 대화 등 다양한 LLM 기반 작업에 사용됩니다.

    사용 시나리오:
    - 사용자가 AI 어시스턴트와 대화하고 싶을 때
    - 텍스트 기반 질문에 대한 답변이 필요할 때
    - 코드 생성, 번역, 요약 등 텍스트 처리가 필요할 때

    SSE Events:
    - start: 스트리밍 시작 (model, timestamp 포함)
    - message: 토큰 생성 중 (token, full_text, token_count 포함)
    - complete: 스트리밍 완료 (full_response, total_tokens, generation_time_ms 포함)
    - error: 에러 발생 (error, error_type 포함)

    Returns:
        EventSourceResponse: SSE 스트림
        - 각 이벤트는 JSON 형식의 data 필드를 포함합니다.

    Raises:
        503: 요청한 모델이 로드되지 않은 경우
    """
```

---

## 데이터베이스 테이블

### chat_sessions (채팅 세션)
- `id`: PK
- `user_id`: FK → users.id (nullable)
- `title`: 대화 제목
- `model_name`: 사용된 LLM 모델명
- `tokenizer_name`: 사용된 토크나이저명
- `system_prompt`: 시스템 프롬프트
- `temperature`: 생성 온도
- `max_tokens`: 최대 토큰 수
- `total_messages`: 총 메시지 수
- `total_tokens_used`: 사용된 총 토큰 수

### chat_messages (채팅 메시지)
- `id`: PK
- `session_id`: FK → chat_sessions.id
- `role`: user / assistant / system
- `content`: 메시지 내용
- `input_tokens`: 입력 토큰 수 (assistant만)
- `output_tokens`: 출력 토큰 수 (assistant만)
- `model_name`: 응답 생성 모델명 (assistant만)
- `generation_time`: 생성 시간 ms (assistant만)
