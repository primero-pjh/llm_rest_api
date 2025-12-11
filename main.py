from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.routers import items
from app.routers import terminal
from app.routers import sse

app = FastAPI(
    title=settings.APP_NAME,
    description="FastAPI Backend Server",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(items.router, prefix=settings.API_V1_PREFIX)

# 터미널 라우터 (루트에 등록)
app.include_router(terminal.router)

# SSE 라우터
app.include_router(sse.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
