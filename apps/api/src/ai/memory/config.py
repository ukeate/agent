"""记忆系统配置"""
from typing import Optional
from pydantic import BaseModel
import os


class MemoryConfig(BaseModel):
    """记忆系统配置类"""
    # 存储配置
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = "memories"
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    db_url: str = os.getenv("DATABASE_URL", "postgresql://ai_agent_user:ai_agent_password@localhost:5433/ai_agent_db")
    
    # 记忆系统参数
    working_memory_capacity: int = 100  # 工作记忆容量
    episodic_memory_limit: int = 10000  # 情景记忆限制
    semantic_memory_limit: int = 5000   # 语义记忆限制
    
    # 性能参数
    batch_size: int = 100
    vector_dimension: int = 1536  # OpenAI嵌入维度
    similarity_threshold: float = 0.7  # 相似度阈值
    
    # 遗忘曲线参数
    decay_constant: float = 86400.0  # 24小时(秒)
    retention_threshold: float = 0.3  # 保留阈值
    
    # 巩固参数
    consolidation_threshold: int = 5  # 访问次数阈值
    compression_ratio: float = 0.3  # 压缩比率
    
    # 缓存配置
    cache_ttl: int = 3600  # 缓存过期时间(秒)
    cache_prefix: str = "mem:"
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


# 全局配置实例
memory_config = MemoryConfig()