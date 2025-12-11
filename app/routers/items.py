from fastapi import APIRouter, HTTPException, status
from typing import List
from app.schemas.item import ItemCreate, ItemUpdate, ItemResponse
from app.schemas.base import ResponseBase
from app.services import item_service

router = APIRouter(
    prefix="/items",
    tags=["Items"],
    responses={404: {"description": "Not found"}},
)


@router.get("", response_model=ResponseBase[List[ItemResponse]])
async def get_items():
    """모든 아이템 조회"""
    items = item_service.get_all_items()
    return ResponseBase(data=items)


@router.get("/{item_id}", response_model=ResponseBase[ItemResponse])
async def get_item(item_id: int):
    """특정 아이템 조회"""
    item = item_service.get_item_by_id(item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id {item_id} not found"
        )
    return ResponseBase(data=item)


@router.post("", response_model=ResponseBase[ItemResponse], status_code=status.HTTP_201_CREATED)
async def create_item(item_data: ItemCreate):
    """아이템 생성"""
    item = item_service.create_item(item_data)
    return ResponseBase(message="Item created successfully", data=item)


@router.put("/{item_id}", response_model=ResponseBase[ItemResponse])
async def update_item(item_id: int, item_data: ItemUpdate):
    """아이템 수정"""
    item = item_service.update_item(item_id, item_data)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id {item_id} not found"
        )
    return ResponseBase(message="Item updated successfully", data=item)


@router.delete("/{item_id}", response_model=ResponseBase)
async def delete_item(item_id: int):
    """아이템 삭제"""
    success = item_service.delete_item(item_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id {item_id} not found"
        )
    return ResponseBase(message="Item deleted successfully")
