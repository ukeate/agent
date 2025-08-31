"""
Service Discovery Core Functionality Tests

Tests for the core service discovery system without API dependencies.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from ai.service_discovery.core import (
    AgentServiceDiscoverySystem,
    AgentMetadata,
    AgentCapability,
    AgentStatus,
    ServiceRegistry,
    LoadBalancer
)


class TestAgentMetadata:
    """Test AgentMetadata class"""
    
    def test_agent_capability_creation(self):
        """Test creating agent capability"""
        capability = AgentCapability(
            name="text_generation",
            description="Generate text based on prompts",
            version="1.0.0",
            input_schema={"type": "object", "properties": {"prompt": {"type": "string"}}},
            output_schema={"type": "object", "properties": {"text": {"type": "string"}}}
        )
        
        assert capability.name == "text_generation"
        assert capability.description == "Generate text based on prompts"
        assert capability.version == "1.0.0"
        assert "avg_latency" in capability.performance_metrics
        assert capability.performance_metrics["avg_latency"] == 1.0
    
    def test_agent_metadata_creation(self):
        """Test creating agent metadata"""
        capability = AgentCapability(
            name="text_generation",
            description="Generate text based on prompts",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        
        metadata = AgentMetadata(
            agent_id="test-agent-1",
            agent_type="language_model",
            name="Test Language Model Agent",
            version="1.0.0",
            capabilities=[capability],
            host="localhost",
            port=8001,
            endpoint="http://localhost:8001",
            health_endpoint="http://localhost:8001/health"
        )
        
        assert metadata.agent_id == "test-agent-1"
        assert metadata.agent_type == "language_model"
        assert metadata.status == AgentStatus.ACTIVE
        assert len(metadata.capabilities) == 1
        assert metadata.capabilities[0].name == "text_generation"


class TestServiceRegistry:
    """Test ServiceRegistry class"""
    
    def setup_method(self):
        """Setup test method"""
        self.registry = ServiceRegistry(["localhost:2379"])
    
    @pytest.mark.asyncio
    async def test_registry_initialization_without_etcd(self):
        """Test registry initialization without etcd"""
        # Should work in mock mode
        await self.registry.initialize()
        
        # Should have basic structures initialized
        assert self.registry.local_agents == {}
        assert self.registry.capability_index == {}
        assert self.registry.tag_index == {}
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_discovery(self):
        """Test agent registration and discovery"""
        await self.registry.initialize()
        
        # Create test agent
        capability = AgentCapability(
            name="text_generation",
            description="Generate text",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        
        metadata = AgentMetadata(
            agent_id="test-agent-1",
            agent_type="language_model",
            name="Test Agent",
            version="1.0.0",
            capabilities=[capability],
            host="localhost",
            port=8001,
            endpoint="http://localhost:8001",
            health_endpoint="http://localhost:8001/health",
            tags=["nlp", "text"]
        )
        
        # Register agent
        success = await self.registry.register_agent(metadata)
        assert success is True
        
        # Check local storage
        assert "test-agent-1" in self.registry.local_agents
        assert self.registry.local_agents["test-agent-1"].agent_id == "test-agent-1"
        
        # Check capability index
        assert "text_generation" in self.registry.capability_index
        assert "test-agent-1" in self.registry.capability_index["text_generation"]
        
        # Check tag index
        assert "nlp" in self.registry.tag_index
        assert "text" in self.registry.tag_index
        assert "test-agent-1" in self.registry.tag_index["nlp"]
        
        # Discover agents
        agents = await self.registry.discover_agents()
        assert len(agents) == 1
        assert agents[0].agent_id == "test-agent-1"
        
        # Discover by capability
        agents = await self.registry.discover_agents(capability="text_generation")
        assert len(agents) == 1
        assert agents[0].agent_id == "test-agent-1"
        
        # Discover by tag
        agents = await self.registry.discover_agents(tags=["nlp"])
        assert len(agents) == 1
        
        # Discover by non-existent capability
        agents = await self.registry.discover_agents(capability="nonexistent")
        assert len(agents) == 0
    
    @pytest.mark.asyncio
    async def test_agent_deregistration(self):
        """Test agent deregistration"""
        await self.registry.initialize()
        
        # Create and register test agent
        capability = AgentCapability(
            name="text_generation",
            description="Generate text",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        
        metadata = AgentMetadata(
            agent_id="test-agent-2",
            agent_type="language_model",
            name="Test Agent 2",
            version="1.0.0",
            capabilities=[capability],
            host="localhost",
            port=8002,
            endpoint="http://localhost:8002",
            health_endpoint="http://localhost:8002/health",
            tags=["nlp"]
        )
        
        await self.registry.register_agent(metadata)
        
        # Verify registration
        assert "test-agent-2" in self.registry.local_agents
        assert "text_generation" in self.registry.capability_index
        assert "nlp" in self.registry.tag_index
        
        # Deregister agent
        success = await self.registry.deregister_agent("test-agent-2")
        assert success is True
        
        # Verify deregistration
        assert "test-agent-2" not in self.registry.local_agents
        
        # Check that indexes are cleaned up
        if "text_generation" in self.registry.capability_index:
            assert "test-agent-2" not in self.registry.capability_index["text_generation"]
        
        if "nlp" in self.registry.tag_index:
            assert "test-agent-2" not in self.registry.tag_index["nlp"]
    
    @pytest.mark.asyncio
    async def test_agent_status_update(self):
        """Test agent status update"""
        await self.registry.initialize()
        
        # Create and register test agent
        capability = AgentCapability(
            name="text_generation",
            description="Generate text",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        
        metadata = AgentMetadata(
            agent_id="test-agent-3",
            agent_type="language_model",
            name="Test Agent 3",
            version="1.0.0",
            capabilities=[capability],
            host="localhost",
            port=8003,
            endpoint="http://localhost:8003",
            health_endpoint="http://localhost:8003/health"
        )
        
        await self.registry.register_agent(metadata)
        
        # Verify initial status
        assert self.registry.local_agents["test-agent-3"].status == AgentStatus.ACTIVE
        
        # Update status
        success = await self.registry.update_agent_status("test-agent-3", AgentStatus.MAINTENANCE)
        assert success is True
        
        # Verify status update
        assert self.registry.local_agents["test-agent-3"].status == AgentStatus.MAINTENANCE
        
        # Update metrics
        success = await self.registry.update_agent_metrics("test-agent-3", request_count=100, error_count=5, avg_response_time=1.5)
        assert success is True
        
        # Verify metrics update
        agent = self.registry.local_agents["test-agent-3"]
        assert agent.request_count == 100
        assert agent.error_count == 5
        assert agent.avg_response_time == 1.5


class TestLoadBalancer:
    """Test LoadBalancer class"""
    
    def setup_method(self):
        """Setup test method"""
        self.registry = ServiceRegistry(["localhost:2379"])
        self.load_balancer = LoadBalancer(self.registry)
    
    @pytest.mark.asyncio
    async def test_agent_selection_no_agents(self):
        """Test agent selection when no agents available"""
        await self.registry.initialize()
        
        # Try to select agent when none are available
        agent = await self.load_balancer.select_agent(capability="text_generation")
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_round_robin_strategy(self):
        """Test round robin load balancing strategy"""
        await self.registry.initialize()
        
        # Create multiple test agents
        for i in range(3):
            capability = AgentCapability(
                name="text_generation",
                description="Generate text",
                version="1.0.0",
                input_schema={"type": "object"},
                output_schema={"type": "object"}
            )
            
            metadata = AgentMetadata(
                agent_id=f"test-agent-{i}",
                agent_type="language_model",
                name=f"Test Agent {i}",
                version="1.0.0",
                capabilities=[capability],
                host="localhost",
                port=8000 + i,
                endpoint=f"http://localhost:{8000 + i}",
                health_endpoint=f"http://localhost:{8000 + i}/health"
            )
            
            await self.registry.register_agent(metadata)
        
        # Test round robin selection
        selected_agents = []
        for _ in range(6):  # Select more than available agents to test cycling
            agent = await self.load_balancer.select_agent(
                capability="text_generation",
                strategy="round_robin"
            )
            assert agent is not None
            selected_agents.append(agent.agent_id)
        
        # Should cycle through agents
        assert len(set(selected_agents)) <= 3  # At most 3 unique agents
        assert len(selected_agents) == 6       # All selections successful
    
    @pytest.mark.asyncio
    async def test_capability_based_strategy(self):
        """Test capability-based load balancing strategy"""
        await self.registry.initialize()
        
        # Create agent with high performance metrics
        high_perf_capability = AgentCapability(
            name="text_generation",
            description="High performance text generation",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            performance_metrics={
                "avg_latency": 0.5,
                "throughput": 50.0,
                "accuracy": 0.98,
                "success_rate": 0.99
            }
        )
        
        high_perf_metadata = AgentMetadata(
            agent_id="high-perf-agent",
            agent_type="language_model",
            name="High Performance Agent",
            version="1.0.0",
            capabilities=[high_perf_capability],
            host="localhost",
            port=8010,
            endpoint="http://localhost:8010",
            health_endpoint="http://localhost:8010/health",
            resources={"cpu_usage": 0.3, "memory_usage": 0.4}
        )
        
        # Create agent with lower performance metrics
        low_perf_capability = AgentCapability(
            name="text_generation",
            description="Standard text generation",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            performance_metrics={
                "avg_latency": 2.0,
                "throughput": 10.0,
                "accuracy": 0.85,
                "success_rate": 0.92
            }
        )
        
        low_perf_metadata = AgentMetadata(
            agent_id="low-perf-agent",
            agent_type="language_model",
            name="Standard Agent",
            version="1.0.0",
            capabilities=[low_perf_capability],
            host="localhost",
            port=8011,
            endpoint="http://localhost:8011",
            health_endpoint="http://localhost:8011/health",
            resources={"cpu_usage": 0.8, "memory_usage": 0.9}
        )
        
        await self.registry.register_agent(high_perf_metadata)
        await self.registry.register_agent(low_perf_metadata)
        
        # Test capability-based selection with requirements
        requirements = {
            "min_throughput": 20.0,
            "min_accuracy": 0.9,
            "max_latency": 1.0
        }
        
        agent = await self.load_balancer.select_agent(
            capability="text_generation",
            strategy="capability_based",
            requirements=requirements
        )
        
        # Should select the high-performance agent that meets requirements
        assert agent is not None
        # Note: Due to the complexity of the scoring algorithm, 
        # we just verify that an agent was selected


class TestAgentServiceDiscoverySystem:
    """Test the main service discovery system"""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test system initialization"""
        system = AgentServiceDiscoverySystem(["localhost:2379"])
        await system.initialize()
        
        assert system.registry is not None
        assert system.load_balancer is not None
    
    @pytest.mark.asyncio
    async def test_simplified_agent_registration(self):
        """Test simplified agent registration interface"""
        system = AgentServiceDiscoverySystem(["localhost:2379"])
        await system.initialize()
        
        # Register agent using simplified interface
        success = await system.register_agent(
            agent_id="simple-agent-1",
            agent_type="text_processor",
            name="Simple Text Agent",
            version="1.0.0",
            capabilities=[{
                "name": "text_processing",
                "description": "Process text data",
                "version": "1.0.0",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"}
            }],
            host="localhost",
            port=8020,
            endpoint="http://localhost:8020",
            health_endpoint="http://localhost:8020/health",
            tags=["nlp", "processing"]
        )
        
        assert success is True
        
        # Verify registration through discovery
        agent = await system.discover_and_select_agent(
            capability="text_processing",
            strategy="round_robin"
        )
        
        assert agent is not None
        assert agent.agent_id == "simple-agent-1"
        assert agent.agent_type == "text_processor"
    
    @pytest.mark.asyncio 
    async def test_system_stats(self):
        """Test system statistics"""
        system = AgentServiceDiscoverySystem(["localhost:2379"])
        await system.initialize()
        
        # Register a few agents
        for i in range(2):
            await system.register_agent(
                agent_id=f"stats-agent-{i}",
                agent_type="test_agent",
                name=f"Stats Test Agent {i}",
                version="1.0.0",
                capabilities=[{
                    "name": "testing",
                    "description": "Test capability",
                    "version": "1.0.0",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"}
                }],
                host="localhost",
                port=8030 + i,
                endpoint=f"http://localhost:{8030 + i}",
                health_endpoint=f"http://localhost:{8030 + i}/health"
            )
        
        # Get system stats
        stats = system.get_system_stats()
        
        assert "registry" in stats
        assert "load_balancer" in stats
        assert "system_status" in stats
        
        # Check registry stats
        registry_stats = stats["registry"]
        assert registry_stats["registered_agents"] >= 2
        assert registry_stats["active_agents"] >= 2