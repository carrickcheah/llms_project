from typing import List, Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', 
        env_file_encoding='utf-8',
        env_file_overrides_env=True
    )
    url: str
    collection_name: str
    prefer_grpc: bool
    file_path: str
    hf_model: str
    content_payload_key: Optional[str]
    deepseek_api_key: str
    deepseek_url: str
    deepseek_model: str


config = Config()