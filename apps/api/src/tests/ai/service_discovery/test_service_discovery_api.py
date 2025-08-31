"""
Service Discovery API Tests

Tests for the service discovery API endpoints.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

# Test core service discovery functionality without full app
from ai.service_discovery.core import (
    AgentServiceDiscoverySystem, 
    AgentMetadata, 
    AgentCapability,
    AgentStatus,
    ServiceRegistry,
    LoadBalancer
)
from ai.service_discovery.core import AgentStatus
from ai.service_discovery.models import (
    AgentRegistrationRequest,
    AgentCapabilityModel,
    AgentStatusUpdate,
    LoadBalancerRequest
)


# Test data
SAMPLE_AGENT_REGISTRATION = {
    "agent_id": "test-agent-1",
    "agent_type": "language_model",
    "name": "Test Language Model Agent",
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


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client fixture"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_service_discovery_system():
    """Mock service discovery system"""
    with patch('api.v1.service_discovery.get_service_discovery_system') as mock:
        mock_system = Mock()
        mock_system.register_agent = AsyncMock(return_value=True)
        mock_system.registry.discover_agents = AsyncMock(return_value=[])
        mock_system.registry.get_agent = AsyncMock(return_value=None)
        mock_system.registry.update_agent_status = AsyncMock(return_value=True)
        mock_system.registry.update_agent_metrics = AsyncMock(return_value=True)
        mock_system.registry.deregister_agent = AsyncMock(return_value=True)
        mock_system.discover_and_select_agent = AsyncMock(return_value=None)
        mock_system.get_system_stats = Mock(return_value={
            "registry": {"registered_agents": 0},
            "load_balancer": {"connection_stats": {}},
            "system_status": "healthy"
        })
        
        mock.return_value = mock_system
        yield mock_system


class TestServiceDiscoveryAPI:
    """Service Discovery API tests"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test health check endpoint"""
        response = await async_client.get("/api/v1/service-discovery/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    @pytest.mark.asyncio
    async def test_get_configuration(self, async_client: AsyncClient):
        """Test configuration endpoint"""
        response = await async_client.get("/api/v1/service-discovery/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "load_balancer" in data
        assert "health_check" in data
        assert "system" in data
        assert "strategies" in data["load_balancer"]
        assert "presets" in data["health_check"]
    
    @pytest.mark.asyncio
    async def test_register_agent_success(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test successful agent registration"""
        response = await async_client.post(
            "/api/v1/service-discovery/agents",
            json=SAMPLE_AGENT_REGISTRATION
        )
        assert response.status_code == 201
        
        data = response.json()
        assert data["message"] == "Agent registered successfully"
        assert data["agent_id"] == "test-agent-1"
        assert data["status"] == "registered"
        
        # Verify mock was called
        mock_service_discovery_system.register_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_register_agent_validation_error(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent registration with validation error"""
        invalid_data = SAMPLE_AGENT_REGISTRATION.copy()
        invalid_data["agent_id"] = ""  # Invalid empty agent_id
        
        response = await async_client.post(
            "/api/v1/service-discovery/agents",
            json=invalid_data
        )
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_register_agent_system_failure(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent registration with system failure"""
        mock_service_discovery_system.register_agent.return_value = False
        
        response = await async_client.post(
            "/api/v1/service-discovery/agents",
            json=SAMPLE_AGENT_REGISTRATION
        )
        assert response.status_code == 400
        
        data = response.json()
        assert "Failed to register agent" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_discover_agents(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent discovery"""
        response = await async_client.get("/api/v1/service-discovery/agents")
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        assert "total_count" in data
        assert "query_time" in data
        assert isinstance(data["agents"], list)
    
    @pytest.mark.asyncio
    async def test_discover_agents_with_filters(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent discovery with filters"""
        response = await async_client.get(
            "/api/v1/service-discovery/agents",
            params={
                "capability": "text_generation",
                "tags": "nlp,text",
                "status_filter": "active",
                "group": "language_models",
                "region": "us-west-1",
                "limit": 10
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data
        
        # Verify mock was called with correct parameters
        mock_service_discovery_system.registry.discover_agents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_discover_agents_invalid_status(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent discovery with invalid status filter"""
        response = await async_client.get(
            "/api/v1/service-discovery/agents",
            params={"status_filter": "invalid_status"}
        )
        assert response.status_code == 400
        
        data = response.json()
        assert "Invalid status filter" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test getting non-existent agent"""
        response = await async_client.get("/api/v1/service-discovery/agents/nonexistent-agent")
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_update_agent_status_success(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test successful agent status update"""
        response = await async_client.put(
            "/api/v1/service-discovery/agents/test-agent-1/status",
            json={"status": "maintenance"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Agent status updated successfully"
        assert data["agent_id"] == "test-agent-1"
        assert data["new_status"] == "maintenance"
        
        # Verify mock was called
        mock_service_discovery_system.registry.update_agent_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_agent_status_not_found(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent status update for non-existent agent"""
        mock_service_discovery_system.registry.update_agent_status.return_value = False
        
        response = await async_client.put(
            "/api/v1/service-discovery/agents/nonexistent-agent/status",
            json={"status": "maintenance"}
        )
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_update_agent_metrics_success(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test successful agent metrics update"""
        response = await async_client.put(
            "/api/v1/service-discovery/agents/test-agent-1/metrics",
            json={
                "request_count": 1000,
                "error_count": 15,
                "avg_response_time": 1.25
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Agent metrics updated successfully"
        assert data["agent_id"] == "test-agent-1"
        
        # Verify mock was called
        mock_service_discovery_system.registry.update_agent_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_agent_metrics_validation_error(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent metrics update with validation error"""
        response = await async_client.put(
            "/api/v1/service-discovery/agents/test-agent-1/metrics",
            json={
                "request_count": 100,
                "error_count": 200,  # Error count cannot exceed request count
                "avg_response_time": 1.25
            }
        )
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_deregister_agent_success(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test successful agent deregistration"""
        response = await async_client.delete("/api/v1/service-discovery/agents/test-agent-1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Agent deregistered successfully"
        assert data["agent_id"] == "test-agent-1"
        assert data["status"] == "deregistered"
        
        # Verify mock was called
        mock_service_discovery_system.registry.deregister_agent.assert_called_once_with("test-agent-1")
    
    @pytest.mark.asyncio
    async def test_deregister_agent_not_found(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test deregistration of non-existent agent"""
        mock_service_discovery_system.registry.deregister_agent.return_value = False
        
        response = await async_client.delete("/api/v1/service-discovery/agents/nonexistent-agent")
        assert response.status_code == 404
        
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_select_agent(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent selection via load balancer"""
        response = await async_client.post(
            "/api/v1/service-discovery/load-balancer/select",
            json={
                "capability": "text_generation",
                "strategy": "capability_based",
                "tags": ["nlp"],
                "requirements": {
                    "min_throughput": 10.0,
                    "min_accuracy": 0.9,
                    "max_latency": 2.0
                }
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "selected_agent" in data
        assert "selection_time" in data
        assert "strategy_used" in data
        assert data["strategy_used"] == "capability_based"
        
        # Verify mock was called
        mock_service_discovery_system.discover_and_select_agent.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_select_agent_invalid_strategy(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test agent selection with invalid strategy"""
        response = await async_client.post(
            "/api/v1/service-discovery/load-balancer/select",
            json={
                "capability": "text_generation",
                "strategy": "invalid_strategy"
            }
        )
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_get_system_stats(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test getting system statistics"""
        response = await async_client.get("/api/v1/service-discovery/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "registry" in data
        assert "load_balancer" in data
        assert "system_status" in data
        
        # Verify mock was called
        mock_service_discovery_system.get_system_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test API error handling"""
        # Make register_agent raise an exception
        mock_service_discovery_system.register_agent.side_effect = Exception("Test error")
        
        response = await async_client.post(
            "/api/v1/service-discovery/agents",
            json=SAMPLE_AGENT_REGISTRATION
        )
        assert response.status_code == 500
        
        data = response.json()
        assert "Internal server error" in data["detail"]


class TestLoadBalancerStrategies:
    """Test load balancer strategy validation"""
    
    @pytest.mark.parametrize("strategy", [
        "round_robin",
        "weighted_random", 
        "least_connections",
        "least_response_time",
        "capability_based",
        "resource_aware"
    ])
    @pytest.mark.asyncio
    async def test_valid_strategies(self, async_client: AsyncClient, mock_service_discovery_system, strategy):
        """Test all valid load balancing strategies"""
        response = await async_client.post(
            "/api/v1/service-discovery/load-balancer/select",
            json={
                "capability": "text_generation",
                "strategy": strategy
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["strategy_used"] == strategy


class TestAgentStatusValidation:
    """Test agent status validation"""
    
    @pytest.mark.parametrize("status", ["active", "inactive", "unhealthy", "maintenance"])
    @pytest.mark.asyncio
    async def test_valid_status_values(self, async_client: AsyncClient, mock_service_discovery_system, status):
        """Test all valid status values"""
        response = await async_client.put(
            "/api/v1/service-discovery/agents/test-agent-1/status",
            json={"status": status}
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_invalid_status_value(self, async_client: AsyncClient, mock_service_discovery_system):
        """Test invalid status value"""
        response = await async_client.put(
            "/api/v1/service-discovery/agents/test-agent-1/status",
            json={"status": "invalid_status"}
        )
        assert response.status_code == 422  # Validation error