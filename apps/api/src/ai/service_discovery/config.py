"""
Service Discovery Configuration

Configuration settings for the agent service discovery system.
"""

from typing import List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class ServiceDiscoveryConfig(BaseSettings):
    """Service discovery configuration"""
    
    # Redis存储配置
    redis_prefix: str = Field(
        default="service_discovery:agents:",
        description="Redis中存储agent元数据的key前缀",
    )
    agent_ttl_seconds: int = Field(
        default=90,
        description="Agent元数据TTL（秒），超时视为下线",
    )
    
    # Health check configuration
    health_check_interval: float = Field(
        default=30.0,
        description="Health check interval in seconds"
    )
    health_check_timeout: float = Field(
        default=5.0,
        description="Health check timeout in seconds"
    )
    health_check_retries: int = Field(
        default=3,
        description="Health check retry count"
    )
    health_check_failure_threshold: int = Field(
        default=3,
        description="Consecutive failures to mark agent unhealthy"
    )
    health_check_success_threshold: int = Field(
        default=1,
        description="Consecutive successes to mark agent healthy"
    )
    
    # Load balancer configuration
    default_lb_strategy: str = Field(
        default="capability_based",
        description="Default load balancing strategy"
    )
    connection_timeout: float = Field(
        default=30.0,
        description="Connection timeout for agent communication"
    )
    
    # Performance and limits
    max_agents_per_discovery: int = Field(
        default=100,
        description="Maximum number of agents returned in discovery"
    )
    discovery_cache_ttl: int = Field(
        default=60,
        description="Discovery result cache TTL in seconds"
    )
    metrics_collection_interval: float = Field(
        default=60.0,
        description="Metrics collection interval in seconds"
    )
    
    # Monitoring and alerting
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    enable_health_monitoring: bool = Field(
        default=True,
        description="Enable agent health monitoring"
    )
    enable_event_logging: bool = Field(
        default=True,
        description="Enable event logging"
    )
    
    model_config = SettingsConfigDict(env_prefix="SERVICE_DISCOVERY_", case_sensitive=False)

class LoadBalancerConfig:
    """Load balancer configuration"""
    
    STRATEGIES = {
        "round_robin": {
            "description": "Distribute requests evenly across agents",
            "suitable_for": "Uniform load distribution",
            "parameters": {}
        },
        "weighted_random": {
            "description": "Random selection with resource-based weighting",
            "suitable_for": "Dynamic load balancing based on resource usage",
            "parameters": {
                "cpu_weight": 0.4,
                "memory_weight": 0.4,
                "error_weight": 0.2
            }
        },
        "least_connections": {
            "description": "Select agent with fewest active connections",
            "suitable_for": "Connection-based load balancing",
            "parameters": {}
        },
        "least_response_time": {
            "description": "Select agent with fastest response time",
            "suitable_for": "Latency-sensitive applications",
            "parameters": {}
        },
        "capability_based": {
            "description": "Select based on capability requirements",
            "suitable_for": "Performance-sensitive task allocation",
            "parameters": {
                "performance_weight": 0.4,
                "resource_weight": 0.3,
                "reliability_weight": 0.3
            }
        },
        "resource_aware": {
            "description": "Select based on resource availability",
            "suitable_for": "Resource-intensive applications",
            "parameters": {
                "cpu_threshold": 0.8,
                "memory_threshold": 0.8,
                "connection_penalty": 0.1
            }
        }
    }
    
    @classmethod
    def get_strategy_info(cls, strategy: str) -> Dict[str, Any]:
        """Get information about a load balancing strategy"""
        return cls.STRATEGIES.get(strategy, {})
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available load balancing strategies"""
        return list(cls.STRATEGIES.keys())

class HealthCheckConfig:
    """Health check configuration presets"""
    
    PRESETS = {
        "fast": {
            "interval": 10.0,
            "timeout": 2.0,
            "retries": 2,
            "failure_threshold": 2,
            "success_threshold": 1
        },
        "normal": {
            "interval": 30.0,
            "timeout": 5.0,
            "retries": 3,
            "failure_threshold": 3,
            "success_threshold": 1
        },
        "conservative": {
            "interval": 60.0,
            "timeout": 10.0,
            "retries": 5,
            "failure_threshold": 5,
            "success_threshold": 2
        }
    }
    
    @classmethod
    def get_preset(cls, preset: str) -> Dict[str, Any]:
        """Get health check preset configuration"""
        return cls.PRESETS.get(preset, cls.PRESETS["normal"])
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """List all available health check presets"""
        return list(cls.PRESETS.keys())

# Default configuration instance
default_config = ServiceDiscoveryConfig()
