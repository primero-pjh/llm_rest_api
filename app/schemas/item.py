from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ItemBase(BaseModel):
    """Item 기본 스키마"""
    name: str
    description: Optional[str] = None
    price: float
    is_active: bool = True


class ItemCreate(ItemBase):
    """Item 생성 스키마"""
    pass


class ItemUpdate(BaseModel):
    """Item 수정 스키마"""
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    is_active: Optional[bool] = None


class ItemResponse(ItemBase):
    """Item 응답 스키마"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
