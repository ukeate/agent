"""
Service Discovery API Models

Pydantic models for request/response schemas in the service discovery API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class AgentStatusModel(str, Enum):
    """Agent status model"""
    ACTIVE = "active"
    INACTIVE = "inactive" 
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class AgentCapabilityModel(BaseModel):
    """Agent capability model"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    version: str = Field(..., description="Capability version")
    input_schema: Dict[str, Any] = Field(..., description="Input schema definition")
    output_schema: Dict[str, Any] = Field(..., description="Output schema definition")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Capability constraints")

    class Config:
        schema_extra = {
            "example": {
                "name": "text_generation",
                "description": "Generate text based on prompts",
                "version": "1.0.0",
                "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                "performance_metrics": {
                    "avg_latency": 1.5,
                    "throughput": 15.0,
                    "accuracy": 0.92,
                    "success_rate": 0.98
                }
            }
        }


class AgentRegistrationRequest(BaseModel):
    """Agent registration request"""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type")
    name: str = Field(..., description="Agent name")
    version: str = Field(..., description="Agent version")
    capabilities: List[AgentCapabilityModel] = Field(..., description="Agent capabilities")
    host: str = Field(..., description="Agent host address")
    port: int = Field(..., ge=1, le=65535, description="Agent port number")
    endpoint: str = Field(..., description="Agent API endpoint URL")
    health_endpoint: str = Field(..., description="Agent health check endpoint URL")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Resource information")
    tags: List[str] = Field(default_factory=list, description="Agent tags")
    group: str = Field(default="default", description="Agent group")
    region: str = Field(default="default", description="Agent region")

    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Agent ID cannot be empty')
        return v.strip()

    @validator('capabilities')
    def validate_capabilities(cls, v):
        if not v:
            raise ValueError('At least one capability is required')
        
        # Check for duplicate capability names
        capability_names = [cap.name for cap in v]
        if len(capability_names) != len(set(capability_names)):
            raise ValueError('Capability names must be unique')
        
        return v

    class Config:
        schema_extra = {
            "example": {
                "agent_id": "llm-agent-1",
                "agent_type": "language_model",
                "name": "GPT-4 Language Model Agent",
                "version": "1.0.0",
                "capabilities": [
                    {
                        "name": "text_generation",
                        "description": "Generate text based on prompts",
                        "version": "1.0.0",
                        "input_schema": {"type": "object", "properties": {"prompt": {"type": "string"}}},
                        "output_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                        "performance_metrics": {
                            "avg_latency": 1.5,
                            "throughput": 15.0,
                            "accuracy": 0.92,
                            "success_rate": 0.98
                        }
                    }
                ],
                "host": "localhost",
                "port": 8001,
                "endpoint": "http://localhost:8001",
                "health_endpoint": "http://localhost:8001/health",
                "tags": ["nlp", "text", "generation"],
                "group": "language_models",
                "region": "us-west-1"
            }
        }


class AgentMetadataResponse(BaseModel):
    """Agent metadata response"""
    agent_id: str
    agent_type: str
    name: str
    version: str
    capabilities: List[AgentCapabilityModel]
    host: str
    port: int
    endpoint: str
    health_endpoint: str
    resources: Dict[str, Any]
    tags: List[str]
    group: str
    region: str
    status: AgentStatusModel
    created_at: datetime
    last_heartbeat: datetime
    request_count: int
    error_count: int
    avg_response_time: float

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AgentDiscoveryRequest(BaseModel):
    """Agent discovery request"""
    capability: Optional[str] = Field(None, description="Required capability")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    status: Optional[AgentStatusModel] = Field(None, description="Agent status filter")
    group: Optional[str] = Field(None, description="Agent group filter")
    region: Optional[str] = Field(None, description="Agent region filter")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")

    class Config:
        schema_extra = {
            "example": {
                "capability": "text_generation",
                "tags": ["nlp"],
                "status": "active",
                "group": "language_models",
                "region": "us-west-1",
                "limit": 10
            }
        }


class AgentDiscoveryResponse(BaseModel):
    """Agent discovery response"""
    agents: List[AgentMetadataResponse] = Field(..., description="List of discovered agents")
    total_count: int = Field(..., description="Total number of matching agents")
    query_time: float = Field(..., description="Query execution time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "agents": [],
                "total_count": 0,
                "query_time": 0.05
            }
        }


class AgentStatusUpdate(BaseModel):
    """Agent status update request"""
    status: AgentStatusModel = Field(..., description="New agent status")

    class Config:
        schema_extra = {
            "example": {
                "status": "maintenance"
            }
        }


class AgentMetricsUpdate(BaseModel):
    """Agent metrics update request"""
    request_count: Optional[int] = Field(None, ge=0, description="Total request count")
    error_count: Optional[int] = Field(None, ge=0, description="Total error count")
    avg_response_time: Optional[float] = Field(None, ge=0.0, description="Average response time in seconds")

    @validator('error_count')
    def validate_error_count(cls, v, values):
        if v is not None and 'request_count' in values and values['request_count'] is not None:
            if v > values['request_count']:
                raise ValueError('Error count cannot exceed request count')
        return v

    class Config:
        schema_extra = {
            "example": {
                "request_count": 1000,
                "error_count": 15,
                "avg_response_time": 1.25
            }
        }


class LoadBalancerRequest(BaseModel):
    """Load balancer selection request"""
    capability: str = Field(..., description="Required capability")
    strategy: str = Field(default="capability_based", description="Load balancing strategy")
    tags: Optional[List[str]] = Field(None, description="Required tags")
    requirements: Optional[Dict[str, Any]] = Field(None, description="Performance requirements")

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = [
            "round_robin", "weighted_random", "least_connections",
            "least_response_time", "capability_based", "resource_aware"
        ]
        if v not in valid_strategies:
            raise ValueError(f'Invalid strategy. Must be one of: {", ".join(valid_strategies)}')
        return v

    class Config:
        schema_extra = {
            "example": {
                "capability": "text_generation",
                "strategy": "capability_based",
                "tags": ["nlp"],
                "requirements": {
                    "min_throughput": 10.0,
                    "min_accuracy": 0.9,
                    "max_latency": 2.0
                }
            }
        }


class LoadBalancerResponse(BaseModel):
    """Load balancer selection response"""
    selected_agent: Optional[AgentMetadataResponse] = Field(None, description="Selected agent")
    selection_time: float = Field(..., description="Selection time in seconds")
    strategy_used: str = Field(..., description="Strategy used for selection")

    class Config:
        schema_extra = {
            "example": {
                "selected_agent": None,
                "selection_time": 0.01,
                "strategy_used": "capability_based"
            }
        }


class ServiceStats(BaseModel):
    """Service discovery system statistics"""
    registry: Dict[str, Any] = Field(..., description="Registry statistics")
    load_balancer: Dict[str, Any] = Field(..., description="Load balancer statistics")
    system_status: str = Field(..., description="Overall system status")

    class Config:
        schema_extra = {
            "example": {
                "registry": {
                    "registered_agents": 5,
                    "active_agents": 4,
                    "health_checks": 1250,
                    "failed_health_checks": 12,
                    "discovery_requests": 450,
                    "agents_by_status": {
                        "active": 4,
                        "inactive": 0,
                        "unhealthy": 1,
                        "maintenance": 0
                    },
                    "agents_by_type": {
                        "language_model": 3,
                        "vision_model": 1,
                        "embedding_model": 1
                    },
                    "capabilities": ["text_generation", "image_analysis", "text_embedding"],
                    "tags": ["nlp", "vision", "embedding"],
                    "avg_response_time": 1.32
                },
                "load_balancer": {
                    "connection_stats": {},
                    "round_robin_counters": {},
                    "available_strategies": [
                        "round_robin", "weighted_random", "least_connections",
                        "least_response_time", "capability_based", "resource_aware"
                    ]
                },
                "system_status": "healthy"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-08-26T12:00:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.0
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid agent registration data",
                "details": {
                    "field_errors": [
                        {
                            "field": "agent_id",
                            "message": "Agent ID cannot be empty"
                        }
                    ]
                }
            }
        }