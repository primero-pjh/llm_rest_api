"""
LLM Service Module
==================
다중 LLM 모델을 관리하고 model_name에 따라 적절한 서비스로 라우팅하는 모듈입니다.

지원 모델:
- Bllossom/llama-3.2-Korean-Bllossom-3B (기본값)
- kakaocorp/kanana-2-30b-a3b-instruct

참고 문서:
- https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B
- https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct
"""

from typing import Generator, Optional, Dict, Any
from app.services.bllossom_service import bllossom_service, BllossomService
from app.services.kanana_service import kanana_service, KananaService


# 지원하는 모델 매핑 (Hugging Face model_name -> 서비스 인스턴스)
SUPPORTED_MODELS: Dict[str, Any] = {
    "Bllossom/llama-3.2-Korean-Bllossom-3B": bllossom_service,
    "kakaocorp/kanana-2-30b-a3b-instruct": kanana_service,
}

# 기본 모델
DEFAULT_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"


class LLMService:
    """
    LLM 서비스 매니저 클래스

    다중 모델을 관리하고, model_name에 따라 적절한 서비스로 요청을 라우팅합니다.
    """

    _instance: Optional["LLMService"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LLMService._initialized:
            return

        self.default_model = DEFAULT_MODEL
        self.model_id = DEFAULT_MODEL  # 하위 호환성 유지

        LLMService._initialized = True

    @property
    def is_loaded(self) -> bool:
        """최소 하나의 모델이 로드되었는지 확인"""
        return any(svc.is_loaded for svc in SUPPORTED_MODELS.values())

    def get_service(self, model_name: Optional[str] = None):
        """
        model_name에 해당하는 서비스 인스턴스를 반환합니다.

        Args:
            model_name: Hugging Face 모델 ID (None이면 기본 모델 사용)

        Returns:
            해당 모델의 서비스 인스턴스

        Raises:
            ValueError: 지원하지 않는 모델인 경우
        """
        if model_name is None:
            model_name = self.default_model

        if model_name not in SUPPORTED_MODELS:
            supported = list(SUPPORTED_MODELS.keys())
            raise ValueError(f"지원하지 않는 모델입니다: {model_name}. 지원 모델: {supported}")

        return SUPPORTED_MODELS[model_name]

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """특정 모델이 로드되었는지 확인"""
        try:
            service = self.get_service(model_name)
            return service.is_loaded
        except ValueError:
            return False

    def load_model(self, model_name: Optional[str] = None, use_quantization: bool = True):
        """
        특정 모델을 로드합니다.

        Args:
            model_name: Hugging Face 모델 ID (None이면 기본 모델)
            use_quantization: 4bit 양자화 사용 여부
        """
        service = self.get_service(model_name)
        service.load_model(use_quantization=use_quantization)

    def load_all_models(self, use_quantization: bool = True):
        """
        모든 지원 모델을 로드합니다.

        Args:
            use_quantization: 4bit 양자화 사용 여부
        """
        for model_name, service in SUPPORTED_MODELS.items():
            print(f"[LLMService] 모델 로드 시작: {model_name}")
            service.load_model(use_quantization=use_quantization)

    def generate_stream(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Generator[str, None, None]:
        """
        스트리밍 방식으로 텍스트를 생성합니다.

        Args:
            prompt: 사용자 입력 프롬프트
            model_name: 사용할 모델 (Hugging Face model ID)
            system_prompt: 시스템 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 생성 다양성
            top_p: nucleus sampling 확률 임계값
            top_k: top-k sampling에서 고려할 토큰 수

        Yields:
            str: 생성된 텍스트 조각
        """
        service = self.get_service(model_name)

        if not service.is_loaded:
            raise RuntimeError(f"모델이 로드되지 않았습니다: {model_name or self.default_model}")

        yield from service.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        일반(비스트리밍) 방식으로 텍스트를 생성합니다.
        """
        result = ""
        for text in self.generate_stream(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            result += text
        return result

    def get_supported_models(self) -> list:
        """지원하는 모델 목록을 반환합니다."""
        return list(SUPPORTED_MODELS.keys())

    def get_loaded_models(self) -> list:
        """현재 로드된 모델 목록을 반환합니다."""
        return [name for name, svc in SUPPORTED_MODELS.items() if svc.is_loaded]


# 전역 LLM 서비스 인스턴스 (싱글톤)
llm_service = LLMService()
