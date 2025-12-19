from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.request_logger import RequestLoggerMiddleware
from app.routers import terminal
from app.routers import sse
from app.routers import monitor
from app.routers import mysql_viewer
from app.routers import chat
from app.services.llm_service import llm_service
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()  # MySQL 접속 활성화
    print("✅ MySQL 데이터베이스 연결 완료")

    # LLM 모델 자동 로드 (모든 지원 모델)
    print("=" * 60)
    print("LLM 모델 자동 로드 시작...")
    print(f"지원 모델: {llm_service.get_supported_models()}")
    print("=" * 60)
    try:
        # 비동기 함수에서 동기 함수 호출 - 모든 모델 로드
        await asyncio.to_thread(
            llm_service.load_all_models,
            use_quantization=True  # 4bit 양자화 사용 (메모리 절약)
        )
        print("=" * 60)
        print(f"✅ LLM 모델 로드 완료: {llm_service.get_loaded_models()}")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"❌ LLM 모델 로드 실패: {e}")
        print("수동으로 POST /sse/llm/load 엔드포인트를 호출하세요.")
        print("=" * 60)

    yield

    # Shutdown
    await close_db()  # MySQL 접속 종료
    print("✅ MySQL 데이터베이스 연결 종료")
    print("서버 종료 중...")


app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI Backend Server",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 로깅 미들웨어
app.add_middleware(RequestLoggerMiddleware)

# 모니터 라우터
app.include_router(monitor.router)

# 터미널 라우터 (루트에 등록)
app.include_router(terminal.router)

# SSE 라우터
app.include_router(sse.router)

# MySQL Viewer 라우터
app.include_router(mysql_viewer.router)

# Chat 라우터
app.include_router(chat.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
