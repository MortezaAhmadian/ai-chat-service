from api.config import Settings
from api.llm.anthropic import AnthropicClient
from api.llm.base import LLMClient
from api.llm.fake import FakeLLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLMClient()
    # Settings' validator guarantees the key exists for the real provider.
    assert settings.anthropic_api_key is not None
    return AnthropicClient(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.anthropic_model,
        base_url=settings.anthropic_base_url,
        timeout_s=settings.llm_timeout_s,
    )
