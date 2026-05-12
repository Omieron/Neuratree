from pydantic_settings import BaseSettings, SettingsConfigDict


class DNTConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DNT_", env_file=".env")

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Bellek
    buffer_size: int = 50
    consolidate_every: int = 10
    max_depth: int = 5

    # GHSOM
    tau1: float = 0.5
    tau2: float = 0.01
    hebbian_lr: float = 0.1
    decay_factor: float = 0.99

    # Sorgu
    hop_limit: int = 3
    activation_threshold: float = 0.3
