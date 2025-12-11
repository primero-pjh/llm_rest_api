from pydantic import BaseModel
from typing import TypeVar, Generic, Optional, List
from datetime import datetime

T = TypeVar("T")


class ResponseBase(BaseModel, Generic[T]):
    """API 응답 기본 스키마"""
    success: bool = True
    message: str = "Success"
    data: Optional[T] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """페이지네이션 응답 스키마"""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class TimestampMixin(BaseModel):
    """타임스탬프 믹스인"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
