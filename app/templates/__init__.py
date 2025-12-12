from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def get_template(name: str) -> str:
    """템플릿 파일 내용 반환"""
    template_path = TEMPLATES_DIR / name
    return template_path.read_text(encoding="utf-8")
