from typing import List, Optional
from datetime import datetime
from app.schemas.item import ItemCreate, ItemUpdate, ItemResponse


# 임시 인메모리 저장소 (실제 프로젝트에서는 DB 사용)
_items_db: dict[int, dict] = {}
_id_counter = 0


def get_next_id() -> int:
    global _id_counter
    _id_counter += 1
    return _id_counter


def get_all_items() -> List[ItemResponse]:
    """모든 아이템 조회"""
    return [ItemResponse(**item) for item in _items_db.values()]


def get_item_by_id(item_id: int) -> Optional[ItemResponse]:
    """ID로 아이템 조회"""
    item = _items_db.get(item_id)
    if item:
        return ItemResponse(**item)
    return None


def create_item(item_data: ItemCreate) -> ItemResponse:
    """아이템 생성"""
    item_id = get_next_id()
    now = datetime.now()
    item = {
        "id": item_id,
        "name": item_data.name,
        "description": item_data.description,
        "price": item_data.price,
        "is_active": item_data.is_active,
        "created_at": now,
        "updated_at": None,
    }
    _items_db[item_id] = item
    return ItemResponse(**item)


def update_item(item_id: int, item_data: ItemUpdate) -> Optional[ItemResponse]:
    """아이템 수정"""
    if item_id not in _items_db:
        return None

    item = _items_db[item_id]
    update_data = item_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        item[field] = value

    item["updated_at"] = datetime.now()
    _items_db[item_id] = item
    return ItemResponse(**item)


def delete_item(item_id: int) -> bool:
    """아이템 삭제"""
    if item_id in _items_db:
        del _items_db[item_id]
        return True
    return False
