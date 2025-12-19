"""
MySQL Database Viewer Router
/mysql 경로에서 DB 구조 및 레코드를 조회할 수 있는 웹 인터페이스 제공
"""

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from app.core.database import engine
from app.core.config import settings
from typing import Optional

router = APIRouter(prefix="/mysql", tags=["MySQL Viewer"])

# Jinja2 템플릿 환경 설정
templates_dir = Path(__file__).parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(templates_dir))


async def execute_query(query: str, params: dict = None):
    """비동기로 SQL 쿼리 실행"""
    async with engine.connect() as conn:
        result = await conn.execute(text(query), params or {})
        return result.fetchall(), result.keys()


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def mysql_viewer(
    table: Optional[str] = Query(None, description="조회할 테이블명"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    per_page: int = Query(50, ge=1, le=500, description="페이지당 레코드 수")
):
    """MySQL 데이터베이스 뷰어 메인 페이지"""

    # 모든 테이블 목록 조회
    tables_query = """
    SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, CREATE_TIME, TABLE_COMMENT
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = :db_name
    ORDER BY TABLE_NAME
    """
    tables_data, _ = await execute_query(tables_query, {"db_name": settings.DB_NAME})

    tables_info = [{
        "name": row[0],
        "rows": row[1] or 0,
        "size": round((row[2] or 0) / 1024, 2),
        "created": str(row[3])[:19] if row[3] else "N/A",
        "comment": row[4] or ""
    } for row in tables_data]

    # 템플릿 컨텍스트 초기화
    context = {
        "db_name": settings.DB_NAME,
        "tables_info": tables_info,
        "selected_table": table,
        "table_structure": [],
        "table_records": [],
        "column_names": [],
        "total_records": 0,
        "current_page": page,
        "total_pages": 0,
        "per_page": per_page,
        "create_statement": "",
        "indexes_info": [],
        "error_message": None
    }

    if table:
        try:
            # 테이블 구조 조회
            structure_data, _ = await execute_query(f"DESCRIBE `{table}`")
            context["table_structure"] = [{
                "field": row[0],
                "type": row[1],
                "null": row[2],
                "key": row[3],
                "default": str(row[4]) if row[4] is not None else "NULL",
                "extra": row[5] or ""
            } for row in structure_data]

            # CREATE TABLE 문 조회
            create_data, _ = await execute_query(f"SHOW CREATE TABLE `{table}`")
            if create_data:
                context["create_statement"] = create_data[0][1]

            # 인덱스 정보 조회
            index_data, _ = await execute_query(f"SHOW INDEX FROM `{table}`")
            context["indexes_info"] = [{
                "name": row[2],
                "column": row[4],
                "unique": "No" if row[1] else "Yes",
                "type": row[10]
            } for row in index_data]

            # 총 레코드 수 조회
            count_data, _ = await execute_query(f"SELECT COUNT(*) FROM `{table}`")
            total_records = count_data[0][0] if count_data else 0
            context["total_records"] = total_records
            context["total_pages"] = (total_records + per_page - 1) // per_page

            # 페이지네이션된 레코드 조회
            offset = (page - 1) * per_page
            records_data, keys = await execute_query(
                f"SELECT * FROM `{table}` LIMIT :limit OFFSET :offset",
                {"limit": per_page, "offset": offset}
            )
            context["column_names"] = list(keys)

            table_records = []
            for row in records_data:
                record = {}
                for i, col in enumerate(context["column_names"]):
                    value = row[i]
                    if value is None:
                        record[col] = "NULL"
                    elif isinstance(value, bytes):
                        record[col] = value.hex()[:50] + "..." if len(value) > 25 else value.hex()
                    else:
                        str_val = str(value)
                        record[col] = str_val[:100] + "..." if len(str_val) > 100 else str_val
                table_records.append(record)
            context["table_records"] = table_records

        except Exception as e:
            context["error_message"] = str(e)

    # Jinja2 템플릿 렌더링
    template = jinja_env.get_template("mysql_viewer.html")
    return HTMLResponse(content=template.render(**context))


@router.get("/api/tables")
async def get_tables():
    """테이블 목록 API"""
    tables_query = """
    SELECT TABLE_NAME, TABLE_ROWS, DATA_LENGTH, CREATE_TIME, TABLE_COMMENT
    FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = :db_name
    ORDER BY TABLE_NAME
    """
    tables_data, _ = await execute_query(tables_query, {"db_name": settings.DB_NAME})

    return [{
        "name": row[0],
        "rows": row[1] or 0,
        "size_kb": round((row[2] or 0) / 1024, 2),
        "created": str(row[3])[:19] if row[3] else None,
        "comment": row[4] or ""
    } for row in tables_data]


@router.get("/api/table/{table_name}/structure")
async def get_table_structure(table_name: str):
    """테이블 구조 API"""
    structure_data, _ = await execute_query(f"DESCRIBE `{table_name}`")

    return [{
        "field": row[0],
        "type": row[1],
        "null": row[2],
        "key": row[3],
        "default": str(row[4]) if row[4] is not None else None,
        "extra": row[5] or ""
    } for row in structure_data]


@router.get("/api/table/{table_name}/records")
async def get_table_records(
    table_name: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500)
):
    """테이블 레코드 API"""
    offset = (page - 1) * per_page

    # 총 레코드 수
    count_data, _ = await execute_query(f"SELECT COUNT(*) FROM `{table_name}`")
    total = count_data[0][0] if count_data else 0

    # 레코드 조회
    records_data, keys = await execute_query(
        f"SELECT * FROM `{table_name}` LIMIT :limit OFFSET :offset",
        {"limit": per_page, "offset": offset}
    )
    column_names = list(keys)

    records = []
    for row in records_data:
        record = {}
        for i, col in enumerate(column_names):
            value = row[i]
            if value is None:
                record[col] = None
            elif isinstance(value, bytes):
                record[col] = value.hex()
            else:
                record[col] = str(value)
        records.append(record)

    return {
        "columns": column_names,
        "records": records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page
    }
