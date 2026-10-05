from api.config import Settings
from api.llm.anthropic import AnthropicClient
from api.llm.base import LLMClient
from api.llm.fake import FakeLLMClient
from api.llm.openai_compat import OpenAICompatibleClient


def build_llm_client(settings: Settings) -> LLMClient:
    match settings.llm_provider:
        case "fake":
            return FakeLLMClient()
        case "vllm":
            return OpenAICompatibleClient(
                base_url=settings.vllm_base_url,
                model=settings.vllm_model,
                timeout_s=settings.llm_timeout_s,
                api_key=settings.vllm_api_key.get_secret_value() if settings.vllm_api_key else None,
                guided_json=settings.vllm_guided_json,
                enable_thinking=settings.vllm_enable_thinking,
                top_p=settings.vllm_top_p,
                top_k=settings.vllm_top_k,
                presence_penalty=settings.vllm_presence_penalty,
            )
        case "anthropic":
            # Settings' validator guarantees the key exists for this provider.
            assert settings.anthropic_api_key is not None
            return AnthropicClient(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.anthropic_model,
                base_url=settings.anthropic_base_url,
                timeout_s=settings.llm_timeout_s,
            )
