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
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001", "http://localhost:3004", "http://127.0.0.1:3004"],
        description="允许的跨域来源",
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT算法")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="访问令牌过期时间（分钟）")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="刷新令牌过期时间（天）")
    
    # 安全增强配置
    FORCE_HTTPS: bool = Field(default=False, description="强制HTTPS")
    CSP_HEADER: str = Field(
        default="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';",
        description="Content Security Policy头"
    )
    SECURITY_THRESHOLD: float = Field(default=0.7, description="安全阈值")
    AUTO_BLOCK_THRESHOLD: float = Field(default=0.9, description="自动阻断阈值")
    MAX_REQUESTS_PER_MINUTE: int = Field(default=60, description="每分钟最大请求数")
    MAX_REQUEST_SIZE: int = Field(default=10485760, description="最大请求大小（字节）")
    DEFAULT_RATE_LIMIT: str = Field(default="100/minute", description="默认频率限制")

    # 数据库配置
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://ai_agent_user:ai_agent_password@localhost:5433/ai_agent_db",
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
    
    # Neo4j图数据库配置
    NEO4J_URI: str = Field(default="neo4j://localhost:7687", description="Neo4j连接URI")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j用户名")
    NEO4J_PASSWORD: str = Field(default="password", description="Neo4j密码")
    NEO4J_DATABASE: str = Field(default="neo4j", description="Neo4j数据库名称")
    NEO4J_MAX_POOL_SIZE: int = Field(default=50, description="Neo4j连接池最大大小")
    NEO4J_MAX_RETRY_TIME: float = Field(default=30.0, description="Neo4j事务最大重试时间")
    NEO4J_CONNECTION_TIMEOUT: float = Field(default=30.0, description="Neo4j连接超时时间")
    NEO4J_ENCRYPTED: bool = Field(default=False, description="Neo4j是否启用加密")
    NEO4J_TRUST_SYSTEM_CA: bool = Field(default=True, description="Neo4j是否信任系统CA")
    NEO4J_QUERY_TIMEOUT: int = Field(default=60, description="Neo4j查询超时时间(秒)")
    NEO4J_BATCH_SIZE: int = Field(default=1000, description="Neo4j批量操作大小")
    NEO4J_CACHE_TTL: int = Field(default=3600, description="Neo4j查询缓存TTL(秒)")
    NEO4J_MONITORING_ENABLED: bool = Field(default=True, description="启用Neo4j监控")
    NEO4J_POOL_MAX_LIFETIME: float = Field(default=3600.0, description="连接池中连接的最大生命周期(秒)")
    
    # pgvector配置
    PGVECTOR_ENABLED: bool = Field(default=True, description="启用pgvector支持")
    PGVECTOR_VERSION: str = Field(default="0.8.0", description="pgvector版本")
    PGVECTOR_DATABASE_URL: str = Field(
        default="postgresql://ai_agent_user:ai_agent_password@localhost:5433/ai_agent_db",
        description="pgvector专用数据库连接URL（纯PostgreSQL格式）"
    )
    
    # HNSW索引配置
    HNSW_EF_CONSTRUCTION: int = Field(default=64, description="HNSW构建时候选列表大小")
    HNSW_EF_SEARCH: int = Field(default=100, description="HNSW搜索时候选列表大小")
    HNSW_M: int = Field(default=16, description="HNSW每层连接数")
    HNSW_ITERATIVE_SCAN: str = Field(default="strict_order", description="HNSW迭代扫描模式")
    HNSW_MAX_SCAN_TUPLES: int = Field(default=20000, description="HNSW最大扫描元组数")
    HNSW_SCAN_MEM_MULTIPLIER: int = Field(default=2, description="HNSW扫描内存倍数")
    
    # IVFFlat索引配置
    IVFFLAT_LISTS: int = Field(default=1000, description="IVFFlat列表数量")
    IVFFLAT_PROBES: int = Field(default=10, description="IVFFlat探测数量")
    IVFFLAT_ITERATIVE_SCAN: str = Field(default="strict_order", description="IVFFlat迭代扫描模式")
    IVFFLAT_MAX_PROBES: int = Field(default=100, description="IVFFlat最大探测数")
    
    # 向量量化配置
    VECTOR_QUANTIZATION_ENABLED: bool = Field(default=True, description="启用向量量化")
    VECTOR_BINARY_QUANTIZATION: bool = Field(default=False, description="启用二进制量化")
    VECTOR_HALFVEC_ENABLED: bool = Field(default=True, description="启用半精度向量")
    VECTOR_SPARSEVEC_ENABLED: bool = Field(default=True, description="启用稀疏向量")
    
    # 向量索引性能配置
    VECTOR_INDEX_BUILD_MEMORY: str = Field(default="512MB", description="向量索引构建内存")
    VECTOR_INDEX_PARALLEL_WORKERS: int = Field(default=4, description="向量索引构建并行工作数")
    VECTOR_INDEX_MAINTENANCE_WORKERS: int = Field(default=4, description="向量索引维护工作数")
    VECTOR_INDEX_CONCURRENT_BUILD: bool = Field(default=True, description="启用并发索引构建")
    
    # 向量查询性能配置
    VECTOR_QUERY_CACHE_ENABLED: bool = Field(default=True, description="启用向量查询缓存")
    VECTOR_QUERY_CACHE_SIZE: int = Field(default=10000, description="向量查询缓存大小")
    VECTOR_QUERY_CACHE_TTL: int = Field(default=1800, description="向量查询缓存TTL（秒）")
    VECTOR_QUERY_TIMEOUT: int = Field(default=30, description="向量查询超时时间（秒）")
    VECTOR_BATCH_SIZE: int = Field(default=100, description="向量批处理大小")
    VECTOR_MAX_CONNECTIONS: int = Field(default=20, description="向量数据库最大连接数")
    
    # 向量监控配置
    VECTOR_MONITORING_ENABLED: bool = Field(default=True, description="启用向量监控")
    VECTOR_METRICS_COLLECTION_INTERVAL: int = Field(default=60, description="向量指标收集间隔（秒）")
    VECTOR_PERFORMANCE_LOGGING: bool = Field(default=True, description="启用向量性能日志")
    VECTOR_SLOW_QUERY_THRESHOLD: float = Field(default=1.0, description="慢查询阈值（秒）")
    
    # 向量数据完整性配置
    VECTOR_BACKUP_ENABLED: bool = Field(default=True, description="启用向量数据备份")
    VECTOR_VALIDATION_ENABLED: bool = Field(default=True, description="启用向量数据验证")
    VECTOR_MIGRATION_BATCH_SIZE: int = Field(default=1000, description="向量迁移批处理大小")
    
    # BM42混合搜索配置
    HYBRID_SEARCH_VECTOR_WEIGHT: float = Field(default=0.7, description="语义搜索权重")
    HYBRID_SEARCH_BM25_WEIGHT: float = Field(default=0.3, description="BM25搜索权重")
    HYBRID_SEARCH_TOP_K: int = Field(default=20, description="混合搜索返回结果数量")
    HYBRID_SEARCH_RERANK_SIZE: int = Field(default=100, description="重排序候选结果数量")
    HYBRID_SEARCH_STRATEGY: str = Field(default="hybrid_rrf", description="混合搜索策略")
    HYBRID_SEARCH_RRF_K: int = Field(default=60, description="RRF算法参数K")
    HYBRID_SEARCH_ENABLE_CACHE: bool = Field(default=True, description="启用混合搜索缓存")
    BM25_K1: float = Field(default=1.2, description="BM25算法参数K1")
    BM25_B: float = Field(default=0.75, description="BM25算法参数B")
    BM25_AVG_DOC_LENGTH: float = Field(default=1000.0, description="BM25平均文档长度")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    # 缓存配置
    CACHE_ENABLED: bool = Field(default=True, description="启用缓存")
    CACHE_BACKEND: str = Field(default="redis", description="缓存后端")
    CACHE_TTL_DEFAULT: int = Field(default=3600, description="默认缓存TTL（秒）")
    CACHE_MAX_ENTRIES: int = Field(default=10000, description="最大缓存条目数")
    CACHE_KEY_PREFIX: str = Field(default="langgraph:cache", description="缓存键前缀")
    CACHE_REDIS_URL: str = Field(default="redis://localhost:6379/1", description="缓存专用Redis URL")
    CACHE_COMPRESSION: bool = Field(default=True, description="启用缓存压缩")
    CACHE_MONITORING: bool = Field(default=True, description="启用缓存监控")
    CACHE_SERIALIZE_METHOD: str = Field(default="pickle", description="缓存序列化方法")
    CACHE_CLEANUP_INTERVAL: int = Field(default=300, description="缓存清理间隔（秒）")

    # 离线能力配置
    OFFLINE_STORAGE_PATH: str = Field(default="/tmp/ai_agent_offline", description="离线存储路径")
    OFFLINE_SYNC_INTERVAL: int = Field(default=30, description="离线同步间隔（秒）")
    OFFLINE_MAX_OPERATIONS: int = Field(default=10000, description="最大离线操作数")
    OFFLINE_BATCH_SIZE: int = Field(default=100, description="同步批量大小")

    # 用户反馈学习系统配置
    FEEDBACK_BUFFER_SIZE: int = Field(default=1000, description="反馈缓冲区大小")
    FEEDBACK_FLUSH_INTERVAL: float = Field(default=5.0, description="反馈刷新间隔（秒）")
    FEEDBACK_DEDUP_WINDOW: int = Field(default=300, description="反馈去重窗口（秒）")
    FEEDBACK_QUALITY_THRESHOLD: float = Field(default=0.6, description="反馈质量阈值")
    FEEDBACK_BATCH_SIZE: int = Field(default=100, description="反馈批处理大小")
    FEEDBACK_PROCESSING_ENABLED: bool = Field(default=True, description="启用反馈处理")
    FEEDBACK_COLLECTOR_THREADS: int = Field(default=4, description="反馈收集线程数")
    REWARD_SIGNAL_TTL: int = Field(default=86400, description="奖励信号TTL（秒）")
    REWARD_CALCULATION_STRATEGY: str = Field(default="weighted_average", description="奖励计算策略")
    OFFLINE_CONFLICT_RESOLUTION: str = Field(default="merge", description="冲突解决策略")
    OFFLINE_COMPRESSION: bool = Field(default=True, description="启用离线数据压缩")
    OFFLINE_ENCRYPTION: bool = Field(default=False, description="启用离线数据加密")
    OFFLINE_RETRY_MAX_COUNT: int = Field(default=3, description="最大重试次数")
    OFFLINE_RETRY_BACKOFF_FACTOR: float = Field(default=2.0, description="重试退避系数")
    OFFLINE_CONNECTION_TIMEOUT: int = Field(default=10, description="连接超时时间（秒）")
    OFFLINE_NETWORK_CHECK_INTERVAL: int = Field(default=5, description="网络检查间隔（秒）")
    OFFLINE_VECTOR_CLOCK_ENABLED: bool = Field(default=True, description="启用向量时钟")
    OFFLINE_MODEL_CACHE_SIZE: int = Field(default=1000, description="本地模型缓存大小（MB）")
    OFFLINE_MODEL_CACHE_PATH: str = Field(default="/tmp/ai_agent_models", description="本地模型缓存路径")

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
