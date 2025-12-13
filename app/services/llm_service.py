"""
LLM Service Module
==================
Hugging Face Transformers를 사용하여 Llama 3.2-3B 모델을 로드하고
스트리밍 추론(inference)을 수행하는 서비스 모듈입니다.

참고 문서:
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
"""

import torch
from threading import Thread
from typing import Generator, Optional
from transformers import (
    AutoModelForCausalLM,  # Causal LM (텍스트 생성) 모델 자동 로드
    AutoTokenizer,         # 토크나이저 자동 로드
    TextIteratorStreamer,  # 스트리밍 출력을 위한 이터레이터
    BitsAndBytesConfig,    # 양자화 설정 (메모리 절약용)
)


class LLMService:
    """
    LLM 서비스 클래스

    Llama 3.2-3B-Instruct 모델을 로드하고 텍스트 생성을 수행합니다.
    싱글톤 패턴으로 구현되어 모델을 한 번만 로드합니다.
    """

    # 클래스 변수: 싱글톤 인스턴스 저장
    _instance: Optional["LLMService"] = None
    _initialized: bool = False

    def __new__(cls):
        """싱글톤 패턴 구현: 인스턴스가 없으면 생성, 있으면 기존 반환"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        초기화 메서드

        이미 초기화되었으면 스킵합니다 (싱글톤 패턴).
        실제 모델 로드는 load_model()에서 수행합니다.
        """
        if LLMService._initialized:
            return

        # 모델과 토크나이저를 저장할 변수
        self.model = None
        self.tokenizer = None

        # 모델 ID (Hugging Face 모델 허브 또는 로컬 경로)
        # 한국어 최적화된 Llama 3.2 모델 사용
        self.model_id = "Bllossom/llama-3.2-Korean-Bllossom-3B"

        # 모델 로드 상태
        self.is_loaded = False

        LLMService._initialized = True

    def load_model(self, model_path: Optional[str] = None, use_quantization: bool = True):
        """
        모델과 토크나이저를 로드합니다.

        Args:
            model_path: 커스텀 모델 경로 (None이면 기본 모델 사용)
            use_quantization: 4bit 양자화 사용 여부 (메모리 절약, 속도 향상)

        Note:
            - 처음 로드 시 모델 다운로드에 시간이 소요될 수 있습니다
            - GPU 메모리가 부족하면 use_quantization=True 권장
        """
        if self.is_loaded:
            print("모델이 이미 로드되어 있습니다.")
            return

        # 커스텀 모델 경로가 주어지면 사용
        if model_path:
            self.model_id = model_path

        print(f"모델 로드 중: {self.model_id}")

        # 디바이스 확인 (CUDA GPU가 있으면 사용, 없으면 CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 디바이스: {self.device}")

        # 토크나이저 로드
        # - 텍스트를 토큰(숫자)으로 변환하고, 토큰을 텍스트로 복원하는 역할
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,  # 커스텀 토크나이저 코드 신뢰
        )

        # 패딩 토큰 설정 (없으면 EOS 토큰 사용)
        # - 배치 처리 시 길이를 맞추기 위해 필요
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        model_kwargs = {
            "trust_remote_code": True,  # 커스텀 모델 코드 신뢰
            "device_map": "auto",       # 자동으로 GPU/CPU 배치
        }

        # 양자화 설정 (GPU에서만 사용 가능)
        if use_quantization and self.device == "cuda":
            # 4bit 양자화 설정
            # - 모델 크기와 메모리 사용량을 약 4배 줄임
            # - 약간의 성능 손실이 있지만 대부분의 경우 허용 가능
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                        # 4bit 양자화 활성화
                bnb_4bit_quant_type="nf4",               # NF4 양자화 타입 (권장)
                bnb_4bit_compute_dtype=torch.bfloat16,   # 연산은 bfloat16으로
                bnb_4bit_use_double_quant=True,          # 이중 양자화 (추가 압축)
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
            # 양자화 없이 로드 (bfloat16 또는 float16 사용)
            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float32

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )

        self.is_loaded = True
        print(f"모델 로드 완료: {self.model_id}")

    def generate_stream(
        self,
        prompt: str,
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
            system_prompt: 시스템 프롬프트 (AI의 역할 정의)
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 생성 다양성 (0.0=결정적, 1.0=무작위)
            top_p: nucleus sampling 확률 임계값
            top_k: top-k sampling에서 고려할 토큰 수

        Yields:
            str: 생성된 텍스트 조각 (토큰 단위)

        Raises:
            RuntimeError: 모델이 로드되지 않은 경우
        """
        # 모델 로드 확인
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        # 채팅 메시지 포맷 구성
        # - Llama 3.2 Instruct 모델은 특정 채팅 템플릿을 사용
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 채팅 템플릿 적용하여 입력 텍스트 생성
        # - 모델이 이해할 수 있는 형식으로 변환
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # 토큰화하지 않고 텍스트로 반환
            add_generation_prompt=True # 생성 시작 프롬프트 추가
        )

        # 입력 텍스트를 토큰으로 변환
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",      # PyTorch 텐서로 반환
            padding=True,
            truncation=True,
        ).to(self.model.device)       # 모델과 같은 디바이스로 이동

        # 스트리머 생성
        # - TextIteratorStreamer는 생성된 토큰을 순차적으로 반환
        # - skip_prompt=True: 입력 프롬프트는 출력에서 제외
        # - skip_special_tokens=True: 특수 토큰(EOS 등) 제외
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,  # 타임아웃 설정 (초)
        )

        # 생성 파라미터 설정
        generation_kwargs = {
            **inputs,                          # 입력 토큰
            "streamer": streamer,              # 스트리머
            "max_new_tokens": max_new_tokens,  # 최대 생성 토큰 수
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # temperature에 따라 샘플링 전략 결정
        if temperature > 0.0:
            # 샘플링 모드: temperature > 0일 때
            generation_kwargs.update({
                "do_sample": True,              # 샘플링 활성화
                "temperature": temperature,     # 생성 다양성
                "top_p": top_p,                # nucleus sampling
                "top_k": top_k,                # top-k sampling
            })
        else:
            # Greedy decoding: temperature = 0일 때
            # 가장 확률이 높은 토큰만 선택 (결정적 생성)
            generation_kwargs["do_sample"] = False

        # 별도 스레드에서 생성 실행
        # - model.generate()는 블로킹 함수이므로 별도 스레드에서 실행
        # - 메인 스레드에서는 streamer를 통해 결과를 순차적으로 받음
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 스트리머에서 생성된 텍스트를 순차적으로 yield
        # - 각 iteration마다 새로 생성된 텍스트 조각을 반환
        for text in streamer:
            if text:  # 빈 문자열 제외
                yield text

        # 생성 스레드 종료 대기
        thread.join()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        일반(비스트리밍) 방식으로 텍스트를 생성합니다.

        Args:
            prompt: 사용자 입력 프롬프트
            system_prompt: 시스템 프롬프트
            max_new_tokens: 생성할 최대 토큰 수
            temperature: 생성 다양성

        Returns:
            str: 생성된 전체 텍스트
        """
        # 스트리밍 생성 결과를 모아서 반환
        result = ""
        for text in self.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            result += text
        return result


# 전역 LLM 서비스 인스턴스 (싱글톤)
llm_service = LLMService()
