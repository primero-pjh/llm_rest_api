from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.request_logger import RequestLoggerMiddleware
from app.routers import terminal
from app.routers import sse
from app.routers import monitor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


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


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
