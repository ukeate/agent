"""
应用配置管理
"""

import secrets
from functools import lru_cache

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""

    # 基础配置
    DEBUG: bool = Field(default=False, description="调试模式")
    TESTING: bool = Field(default=False, description="测试模式")
    HOST: str = Field(default="0.0.0.0", description="应用主机")
    PORT: int = Field(default=8000, description="应用端口")

    # 安全配置
    SECRET_KEY: str = Field(description="应用密钥")
    ALLOWED_HOSTS: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
        description="允许的跨域来源",
    )

    # 数据库配置
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5433/agent_db",
        description="数据库连接URL",
    )

    # Redis配置
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Redis连接URL"
    )

    # AI服务配置
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API密钥")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API密钥")
    USE_MOCK_EMBEDDINGS: bool = Field(default=False, description="使用Mock嵌入服务（用于测试和网络受限环境）")
    MAX_CONTEXT_LENGTH: int = Field(default=100000, description="最大上下文长度")
    SESSION_TIMEOUT_MINUTES: int = Field(default=60, description="会话超时时间（分钟）")

    # Qdrant配置
    QDRANT_HOST: str = Field(default="localhost", description="Qdrant主机")
    QDRANT_PORT: int = Field(default=6333, description="Qdrant端口")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def secret_key_validator(cls, v):
        """验证密钥 - 为开发环境生成加密安全的密钥"""
        if not v:
            # 生成加密安全的32字节密钥用于开发环境
            dev_key = secrets.token_urlsafe(32)
            return dev_key
        return v

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        """解析允许的主机列表"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",") if host.strip()]
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        env_nested_delimiter="__",
    )


@lru_cache
def get_settings() -> Settings:
    """获取应用设置（带缓存）"""
    return Settings()
