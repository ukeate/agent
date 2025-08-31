"""
AI Agent Service Discovery System

This module implements a distributed service discovery system for AI agents
based on etcd, providing agent registration, discovery, health monitoring,
and load balancing capabilities.
"""

from .core import (
    AgentMetadata,
    AgentCapability, 
    AgentStatus,
    HealthCheckStrategy,
    ServiceRegistry,
    LoadBalancer,
    AgentServiceDiscoverySystem
)

from .models import (
    AgentRegistrationRequest,
    AgentDiscoveryRequest,
    AgentStatusUpdate,
    LoadBalancerRequest,
    ServiceStats
)

__all__ = [
    "AgentMetadata",
    "AgentCapability",
    "AgentStatus", 
    "HealthCheckStrategy",
    "ServiceRegistry",
    "LoadBalancer",
    "AgentServiceDiscoverySystem",
    "AgentRegistrationRequest",
    "AgentDiscoveryRequest", 
    "AgentStatusUpdate",
    "LoadBalancerRequest",
    "ServiceStats"
]