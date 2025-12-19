"""
Bllossom LLM Service Module
===========================
Bllossom/llama-3.2-Korean-Bllossom-3B 모델을 로드하고
스트리밍 추론(inference)을 수행하는 서비스 모듈입니다.

참고 문서:
- https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B
"""

import torch
from threading import Thread
from typing import Generator, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)


class BllossomService:
    """
    Bllossom LLM 서비스 클래스

    Bllossom/llama-3.2-Korean-Bllossom-3B 모델을 로드하고 텍스트 생성을 수행합니다.
    싱글톤 패턴으로 구현되어 모델을 한 번만 로드합니다.
    """

    _instance: Optional["BllossomService"] = None
    _initialized: bool = False

    # Hugging Face 모델 ID
    MODEL_ID = "Bllossom/llama-3.2-Korean-Bllossom-3B"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if BllossomService._initialized:
            return

        self.model = None
        self.tokenizer = None
        self.model_id = self.MODEL_ID
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        BllossomService._initialized = True

    def load_model(self, use_quantization: bool = True):
        """
        모델과 토크나이저를 로드합니다.

        Args:
            use_quantization: 4bit 양자화 사용 여부 (메모리 절약, 속도 향상)
        """
        if self.is_loaded:
            print(f"[Bllossom] 모델이 이미 로드되어 있습니다: {self.model_id}")
            return

        print(f"[Bllossom] 모델 로드 중: {self.model_id}")
        print(f"[Bllossom] 사용 디바이스: {self.device}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드 설정
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }

        # 4bit 양자화 설정
        if use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        else:
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
        print(f"[Bllossom] 모델 로드 완료: {self.model_id}")

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
        """
        if not self.is_loaded:
            raise RuntimeError("[Bllossom] 모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature >= 0.3:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        else:
            generation_kwargs["do_sample"] = False

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield text

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
        """
        result = ""
        for text in self.generate_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            result += text
        return result


# 전역 인스턴스
bllossom_service = BllossomService()
