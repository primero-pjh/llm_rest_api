"""
Chat Models
============
LLM 챗봇 대화 내용을 저장하는 테이블 모델

테이블 구조:
- ChatSession: 대화 세션 (부모 테이블) - 하나의 채팅방/대화 스레드
- ChatMessage: 대화 메시지 (자식 테이블) - 개별 메시지 (사용자/LLM/시스템)

관계:
- User (1) : ChatSession (N) - 한 사용자가 여러 세션을 가질 수 있음
- ChatSession (1) : ChatMessage (N) - 한 세션에 여러 메시지가 포함됨

사용 예시:
    # 새 세션 생성
    session = ChatSession(title="새 대화", system_prompt="You are helpful.")

    # 메시지 추가
    user_msg = ChatMessage(session_id=1, role=MessageRole.USER, content="안녕?")
    assistant_msg = ChatMessage(session_id=1, role=MessageRole.ASSISTANT, content="안녕하세요!")
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum, JSON, func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class MessageRole(str, enum.Enum):
    """
    메시지 역할 열거형

    LLM 대화에서 각 메시지의 발신자를 구분합니다.

    Values:
        USER: 사용자가 입력한 메시지
        ASSISTANT: LLM이 생성한 응답 메시지
        SYSTEM: 시스템 프롬프트 또는 시스템 안내 메시지
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ToolCallStatus(str, enum.Enum):
    """
    도구 호출 상태 열거형

    MCP 도구 호출의 실행 상태를 나타냅니다.

    Values:
        PENDING: 대기 중 (호출 시작됨)
        SUCCESS: 성공적으로 완료됨
        FAILED: 실행 중 에러 발생
        TIMEOUT: 타임아웃으로 실패
    """
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ChatSession(Base):
    """
    대화 세션 테이블 (부모)

    하나의 채팅방/대화 스레드를 나타냅니다.
    사용자와 LLM 간의 대화 컨텍스트를 유지하고,
    대화에 사용된 LLM 설정을 저장합니다.

    Attributes:
        id (int): 세션 고유 식별자 (PK, Auto Increment)
        user_id (int, optional): 사용자 ID (FK → users.id)
            - NULL 허용: 비로그인 사용자도 대화 가능
            - 사용자 삭제 시 SET NULL
        title (str, optional): 대화 제목
            - 최대 255자
            - 첫 메시지 내용에서 자동 생성 가능
        summary (str, optional): 대화 요약
            - 긴 대화의 컨텍스트 압축용
        model_name (str, optional): 사용된 LLM 모델명
            - 예: "Bllossom/llama-3.2-Korean-Bllossom-3B"
        tokenizer_name (str, optional): 사용된 토크나이저명
            - 보통 모델명과 동일
        system_prompt (str, optional): 시스템 프롬프트
            - LLM의 역할/성격 정의
        temperature (str): 생성 온도 (기본값: "0.7")
            - 낮을수록 결정적, 높을수록 창의적
        max_tokens (int): 최대 생성 토큰 수 (기본값: 1024)
        total_messages (int): 세션 내 총 메시지 수 (기본값: 0)
        total_tokens_used (int): 사용된 총 토큰 수 (기본값: 0)
        created_at (datetime): 세션 생성 시각
        updated_at (datetime): 마지막 업데이트 시각

    Relationships:
        user: User 모델과의 관계 (N:1)
        messages: ChatMessage 목록 (1:N, cascade delete)
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="세션 고유 식별자")
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True, comment="사용자 ID (비로그인 시 NULL)")
    title = Column(String(255), nullable=True, comment="대화 제목")
    summary = Column(Text, nullable=True, comment="대화 요약")
    model_name = Column(String(200), nullable=True, comment="사용된 LLM 모델명")
    tokenizer_name = Column(String(200), nullable=True, comment="사용된 토크나이저명")
    system_prompt = Column(Text, nullable=True, comment="시스템 프롬프트")
    temperature = Column(String(10), default="0.7", comment="생성 온도 (0.0~2.0)")
    max_tokens = Column(Integer, default=1024, comment="최대 생성 토큰 수")
    total_messages = Column(Integer, default=0, comment="총 메시지 수")
    total_tokens_used = Column(Integer, default=0, comment="사용된 총 토큰 수")
    created_at = Column(DateTime, server_default=func.now(), comment="생성 시각")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), comment="업데이트 시각")

    # 관계 설정
    user = relationship("User", backref="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, title='{self.title}', user_id={self.user_id})>"


class ChatMessage(Base):
    """
    대화 메시지 테이블 (자식)

    각각의 메시지(사용자 입력 또는 LLM 응답)를 저장합니다.
    하나의 ChatSession에 여러 ChatMessage가 시간순으로 저장됩니다.

    Attributes:
        id (int): 메시지 고유 식별자 (PK, Auto Increment)
        session_id (int): 소속 세션 ID (FK → chat_sessions.id)
            - NOT NULL: 반드시 세션에 속해야 함
            - 세션 삭제 시 CASCADE 삭제
        role (MessageRole): 메시지 발신자 역할
            - USER: 사용자 입력
            - ASSISTANT: LLM 응답
            - SYSTEM: 시스템 메시지
        content (str): 메시지 내용 (TEXT, NOT NULL)
        input_tokens (int, optional): 입력 토큰 수
            - LLM 응답 메시지에만 해당
            - 프롬프트 토큰 수
        output_tokens (int, optional): 출력 토큰 수
            - LLM 응답 메시지에만 해당
            - 생성된 응답 토큰 수
        model_name (str, optional): 응답 생성에 사용된 모델명
            - LLM 응답 메시지에만 해당
        generation_time (int, optional): 생성 소요 시간 (밀리초)
            - LLM 응답 메시지에만 해당
            - 성능 모니터링용
        created_at (datetime): 메시지 생성 시각

    Relationships:
        session: ChatSession 모델과의 관계 (N:1)

    Notes:
        - 메시지는 created_at 기준으로 정렬됨
        - input_tokens, output_tokens, model_name, generation_time은
          role=ASSISTANT인 경우에만 의미 있음
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="메시지 고유 식별자")
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True, comment="소속 세션 ID")
    role = Column(Enum(MessageRole), nullable=False, comment="메시지 역할 (user/assistant/system)")
    content = Column(Text, nullable=False, comment="메시지 내용")
    input_tokens = Column(Integer, nullable=True, comment="입력 토큰 수 (LLM 응답만)")
    output_tokens = Column(Integer, nullable=True, comment="출력 토큰 수 (LLM 응답만)")
    model_name = Column(String(200), nullable=True, comment="응답 생성 모델명 (LLM 응답만)")
    generation_time = Column(Integer, nullable=True, comment="생성 시간 ms (LLM 응답만)")
    created_at = Column(DateTime, server_default=func.now(), comment="생성 시각")

    # 관계 설정
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<ChatMessage(id={self.id}, role='{self.role.value}', content='{content_preview}')>"


class ToolLog(Base):
    """
    도구 호출 로그 테이블

    LLM 에이전트가 MCP를 통해 호출한 도구의 실행 기록을 저장합니다.
    도구 호출의 입력, 출력, 상태, 실행 시간 등을 추적하여
    디버깅, 모니터링, 분석에 활용합니다.

    Attributes:
        id (int): 로그 고유 식별자 (PK, Auto Increment)
        session_id (int, optional): 소속 세션 ID (FK → chat_sessions.id)
            - NULL 허용: 세션 없이 도구만 호출하는 경우
            - 세션 삭제 시 CASCADE 삭제
        tool_name (str): 호출된 도구 이름
            - 예: "web_search", "database_query", "api_call", "calculator"
        tool_arguments (JSON, optional): 도구에 전달된 인자
            - 예: {"query": "서울 날씨", "num_results": 5}
        status (ToolCallStatus): 도구 호출 상태
            - PENDING: 호출 시작
            - SUCCESS: 성공
            - FAILED: 실패
            - TIMEOUT: 타임아웃
        result_content (str, optional): 도구 실행 결과
            - 성공 시 결과 내용
            - JSON 문자열로 저장
        error_message (str, optional): 에러 메시지
            - 실패 시 에러 상세 내용
        execution_time_ms (int, optional): 실행 소요 시간 (밀리초)
            - 성능 모니터링용
        created_at (datetime): 도구 호출 시작 시각
        completed_at (datetime, optional): 도구 실행 완료 시각

    Relationships:
        session: ChatSession 모델과의 관계 (N:1)

    Notes:
        - 도구 호출 시 PENDING 상태로 생성
        - 완료 시 status, result_content/error_message, completed_at 업데이트
    """
    __tablename__ = "tool_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="로그 고유 식별자")
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True, index=True, comment="소속 세션 ID (없을 수 있음)")
    tool_name = Column(String(100), nullable=False, comment="호출된 도구 이름")
    tool_arguments = Column(JSON, nullable=True, comment="도구에 전달된 인자")
    status = Column(Enum(ToolCallStatus), default=ToolCallStatus.PENDING, nullable=False, comment="도구 호출 상태")
    result_content = Column(Text, nullable=True, comment="도구 실행 결과")
    error_message = Column(Text, nullable=True, comment="에러 메시지 (실패 시)")
    execution_time_ms = Column(Integer, nullable=True, comment="실행 소요 시간 (ms)")
    created_at = Column(DateTime, server_default=func.now(), comment="호출 시작 시각")
    completed_at = Column(DateTime, nullable=True, comment="실행 완료 시각")

    # 관계 설정
    session = relationship("ChatSession", backref="tool_logs")

    def __repr__(self):
        return f"<ToolLog(id={self.id}, tool='{self.tool_name}', status='{self.status.value}')>"
