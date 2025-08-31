"""
Agent Service Discovery Core Implementation

Based on etcd for distributed agent service registry and discovery.
Supports agent registration, health monitoring, and load balancing.
"""

import asyncio
import json
import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
try:
    import etcd3
    from etcd3.exceptions import Etcd3Exception
    ETCD_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    etcd3 = None
    Etcd3Exception = Exception
    ETCD_AVAILABLE = False
    import warnings
    warnings.warn("etcd3 not available, using mock implementation for development")
import aiohttp
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class AgentCapability:
    """Agent capability description"""
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Default performance metrics
        if not self.performance_metrics:
            self.performance_metrics = {
                "avg_latency": 1.0,      # Average latency (seconds)
                "throughput": 10.0,       # Throughput (requests/sec)
                "accuracy": 0.95,         # Accuracy rate
                "success_rate": 0.98      # Success rate
            }


@dataclass 
class AgentMetadata:
    """Agent metadata"""
    agent_id: str
    agent_type: str
    name: str
    version: str
    capabilities: List[AgentCapability]
    
    # Network information
    host: str
    port: int
    endpoint: str
    health_endpoint: str
    
    # Resource information
    resources: Dict[str, Any] = field(default_factory=dict)
    
    # Classification and tags
    tags: List[str] = field(default_factory=list)
    group: str = "default"
    region: str = "default"
    
    # Status information
    status: AgentStatus = AgentStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    
    # Statistics
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0


class HealthCheckStrategy:
    """Health check strategy"""
    
    def __init__(
        self,
        interval: float = 30.0,        # Check interval (seconds)
        timeout: float = 5.0,          # Timeout (seconds)
        retries: int = 3,              # Retry count
        failure_threshold: int = 3,     # Failure threshold
        success_threshold: int = 1      # Recovery threshold
    ):
        self.interval = interval
        self.timeout = timeout
        self.retries = retries
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold


class ServiceRegistry:
    """Service registry"""
    
    def __init__(self, etcd_endpoints: List[str], prefix: str = "/agents/"):
        self.etcd_endpoints = etcd_endpoints
        self.prefix = prefix.rstrip('/') + '/'
        
        # etcd client
        self.etcd_client = None
        self.lease = None
        self.lease_ttl = 30  # Lease TTL (seconds)
        
        # Local cache
        self.local_agents: Dict[str, AgentMetadata] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability_name -> agent_ids
        self.tag_index: Dict[str, Set[str]] = {}         # tag -> agent_ids
        
        # Health check
        self.health_check_strategy = HealthCheckStrategy()
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        # Event callbacks
        self.event_handlers: Dict[str, List[Callable]] = {
            "agent_registered": [],
            "agent_deregistered": [],
            "agent_status_changed": [],
            "agent_updated": []
        }
        
        # Monitoring metrics
        self.metrics = {
            "registered_agents": 0,
            "active_agents": 0,
            "health_checks": 0,
            "failed_health_checks": 0,
            "discovery_requests": 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def initialize(self):
        """Initialize registry"""
        
        try:
            if ETCD_AVAILABLE:
                # Connect to etcd
                self.etcd_client = etcd3.client(
                    host=self.etcd_endpoints[0].split(':')[0] if ':' in self.etcd_endpoints[0] else self.etcd_endpoints[0],
                    port=int(self.etcd_endpoints[0].split(':')[1]) if ':' in self.etcd_endpoints[0] else 2379
                )
                
                # Create lease
                self.lease = self.etcd_client.lease(self.lease_ttl)
                
                # Start background tasks
                asyncio.create_task(self._start_watch_agents())
                asyncio.create_task(self._start_lease_renewal())
                asyncio.create_task(self._start_periodic_health_check())
                
                # Load existing agents
                await self._load_existing_agents()
                
                self.logger.info("Service registry initialized successfully with etcd")
            else:
                # Mock initialization for development/testing
                self.etcd_client = None
                self.lease = None
                
                # Start only health check for local agents
                asyncio.create_task(self._start_periodic_health_check())
                
                self.logger.warning("Service registry initialized in mock mode (etcd not available)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service registry: {e}")
            # In development mode, don't fail completely
            if not ETCD_AVAILABLE:
                self.logger.warning("Continuing with mock registry due to etcd unavailability")
            else:
                raise
    
    async def _load_existing_agents(self):
        """Load existing agents"""
        
        if not ETCD_AVAILABLE or not self.etcd_client:
            self.logger.info("No existing agents to load (etcd not available)")
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get all agents from etcd
            result = await loop.run_in_executor(
                self.executor,
                self.etcd_client.get_prefix,
                self.prefix
            )
            
            for value, metadata in result:
                try:
                    agent_data = json.loads(value.decode('utf-8'))
                    
                    # Reconstruct AgentMetadata object
                    capabilities = [
                        AgentCapability(**cap) for cap in agent_data["capabilities"]
                    ]
                    
                    agent_metadata = AgentMetadata(
                        agent_id=agent_data["agent_id"],
                        agent_type=agent_data["agent_type"],
                        name=agent_data["name"],
                        version=agent_data["version"],
                        capabilities=capabilities,
                        host=agent_data["host"],
                        port=agent_data["port"],
                        endpoint=agent_data["endpoint"],
                        health_endpoint=agent_data["health_endpoint"],
                        resources=agent_data.get("resources", {}),
                        tags=agent_data.get("tags", []),
                        group=agent_data.get("group", "default"),
                        region=agent_data.get("region", "default"),
                        status=AgentStatus(agent_data.get("status", "active")),
                        created_at=datetime.fromisoformat(agent_data["created_at"]),
                        last_heartbeat=datetime.fromisoformat(agent_data["last_heartbeat"]),
                        request_count=agent_data.get("request_count", 0),
                        error_count=agent_data.get("error_count", 0),
                        avg_response_time=agent_data.get("avg_response_time", 0.0)
                    )
                    
                    # Add to local cache
                    self.local_agents[agent_metadata.agent_id] = agent_metadata
                    self._update_indexes(agent_metadata)
                    
                    # Start health check
                    await self._start_health_check_for_agent(agent_metadata.agent_id)
                    
                except Exception as e:
                    self.logger.error(f"Error loading agent data: {e}")
            
            self._update_metrics()
            self.logger.info(f"Loaded {len(self.local_agents)} existing agents")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing agents: {e}")
    
    async def register_agent(self, metadata: AgentMetadata) -> bool:
        """Register agent"""
        
        try:
            # Validate agent information
            if not await self._validate_agent_metadata(metadata):
                return False
            
            # Serialize data
            agent_data = self._serialize_agent_metadata(metadata)
            
            # Write to etcd if available
            if ETCD_AVAILABLE and self.etcd_client:
                key = f"{self.prefix}{metadata.agent_id}"
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.etcd_client.put,
                    key,
                    agent_data,
                    self.lease
                )
            else:
                self.logger.info(f"Agent {metadata.agent_id} registered locally (etcd not available)")
            
            # Update local cache
            self.local_agents[metadata.agent_id] = metadata
            self._update_indexes(metadata)
            
            # Start health check
            await self._start_health_check_for_agent(metadata.agent_id)
            
            # Trigger event
            await self._trigger_event("agent_registered", metadata)
            
            # Update metrics
            self._update_metrics()
            
            self.logger.info(f"Agent {metadata.agent_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {metadata.agent_id}: {e}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent"""
        
        try:
            if agent_id not in self.local_agents:
                self.logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            metadata = self.local_agents[agent_id]
            
            # Delete from etcd if available
            if ETCD_AVAILABLE and self.etcd_client:
                key = f"{self.prefix}{agent_id}"
                
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.etcd_client.delete,
                    key
                )
            else:
                self.logger.info(f"Agent {agent_id} deregistered locally (etcd not available)")
            
            # Stop health check
            await self._stop_health_check_for_agent(agent_id)
            
            # Remove from local cache
            del self.local_agents[agent_id]
            self._remove_from_indexes(metadata)
            
            # Trigger event
            await self._trigger_event("agent_deregistered", metadata)
            
            # Update metrics
            self._update_metrics()
            
            self.logger.info(f"Agent {agent_id} deregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {e}")
            return False
    
    async def discover_agents(
        self,
        capability: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[AgentStatus] = None,
        group: Optional[str] = None,
        region: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AgentMetadata]:
        """Discover agents"""
        
        self.metrics["discovery_requests"] += 1
        
        try:
            # Start with capability index filtering
            candidate_ids = set()
            
            if capability:
                candidate_ids = self.capability_index.get(capability, set()).copy()
            else:
                candidate_ids = set(self.local_agents.keys())
            
            # Filter by tags
            if tags:
                for tag in tags:
                    tag_agents = self.tag_index.get(tag, set())
                    candidate_ids &= tag_agents
            
            # Apply other filter conditions
            matching_agents = []
            
            for agent_id in candidate_ids:
                if agent_id not in self.local_agents:
                    continue
                
                agent = self.local_agents[agent_id]
                
                # Status filter
                if status and agent.status != status:
                    continue
                
                # Group filter
                if group and agent.group != group:
                    continue
                
                # Region filter
                if region and agent.region != region:
                    continue
                
                matching_agents.append(agent)
            
            # Sort (by last heartbeat)
            matching_agents.sort(key=lambda a: a.last_heartbeat, reverse=True)
            
            # Limit results
            if limit:
                matching_agents = matching_agents[:limit]
            
            self.logger.debug(f"Discovered {len(matching_agents)} agents with criteria: capability={capability}, tags={tags}")
            return matching_agents
            
        except Exception as e:
            self.logger.error(f"Error discovering agents: {e}")
            return []
    
    async def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        """Get specific agent information"""
        return self.local_agents.get(agent_id)
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        
        try:
            if agent_id not in self.local_agents:
                return False
            
            old_status = self.local_agents[agent_id].status
            self.local_agents[agent_id].status = status
            self.local_agents[agent_id].last_heartbeat = datetime.now()
            
            # Update etcd
            metadata = self.local_agents[agent_id]
            agent_data = self._serialize_agent_metadata(metadata)
            key = f"{self.prefix}{agent_id}"
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.etcd_client.put,
                key,
                agent_data,
                self.lease
            )
            
            # Trigger status change event
            if old_status != status:
                await self._trigger_event("agent_status_changed", metadata)
            
            # Update metrics
            self._update_metrics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent status {agent_id}: {e}")
            return False
    
    async def update_agent_metrics(
        self, 
        agent_id: str, 
        request_count: Optional[int] = None,
        error_count: Optional[int] = None,
        avg_response_time: Optional[float] = None
    ) -> bool:
        """Update agent metrics"""
        
        try:
            if agent_id not in self.local_agents:
                return False
            
            agent = self.local_agents[agent_id]
            
            if request_count is not None:
                agent.request_count = request_count
            if error_count is not None:
                agent.error_count = error_count
            if avg_response_time is not None:
                agent.avg_response_time = avg_response_time
            
            agent.last_heartbeat = datetime.now()
            
            # Update etcd (async, don't wait for result)
            asyncio.create_task(self._update_agent_in_etcd(agent))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent metrics {agent_id}: {e}")
            return False
    
    async def _update_agent_in_etcd(self, metadata: AgentMetadata):
        """Update agent information to etcd"""
        
        try:
            agent_data = self._serialize_agent_metadata(metadata)
            key = f"{self.prefix}{metadata.agent_id}"
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.etcd_client.put,
                key,
                agent_data,
                self.lease
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update agent {metadata.agent_id} in etcd: {e}")
    
    async def _validate_agent_metadata(self, metadata: AgentMetadata) -> bool:
        """Validate agent metadata"""
        
        # Basic field validation
        if not metadata.agent_id or not metadata.agent_type or not metadata.name:
            self.logger.error("Agent metadata missing required fields")
            return False
        
        # Network information validation
        if not metadata.host or not metadata.port or not metadata.endpoint:
            self.logger.error("Agent metadata missing network information")
            return False
        
        # Capability validation
        if not metadata.capabilities:
            self.logger.error("Agent must have at least one capability")
            return False
        
        # Capability name uniqueness check
        capability_names = [cap.name for cap in metadata.capabilities]
        if len(capability_names) != len(set(capability_names)):
            self.logger.error("Agent capabilities must have unique names")
            return False
        
        return True
    
    def _serialize_agent_metadata(self, metadata: AgentMetadata) -> str:
        """Serialize agent metadata"""
        
        data = asdict(metadata)
        
        # Handle datetime serialization
        data["created_at"] = metadata.created_at.isoformat()
        data["last_heartbeat"] = metadata.last_heartbeat.isoformat()
        data["status"] = metadata.status.value
        
        return json.dumps(data, ensure_ascii=False)
    
    def _update_indexes(self, metadata: AgentMetadata):
        """Update indexes"""
        
        agent_id = metadata.agent_id
        
        # Capability index
        for capability in metadata.capabilities:
            if capability.name not in self.capability_index:
                self.capability_index[capability.name] = set()
            self.capability_index[capability.name].add(agent_id)
        
        # Tag index
        for tag in metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(agent_id)
    
    def _remove_from_indexes(self, metadata: AgentMetadata):
        """Remove from indexes"""
        
        agent_id = metadata.agent_id
        
        # Remove from capability index
        for capability in metadata.capabilities:
            if capability.name in self.capability_index:
                self.capability_index[capability.name].discard(agent_id)
                
                # Delete key if set is empty
                if not self.capability_index[capability.name]:
                    del self.capability_index[capability.name]
        
        # Remove from tag index
        for tag in metadata.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(agent_id)
                
                # Delete key if set is empty
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
    
    def _update_metrics(self):
        """Update monitoring metrics"""
        
        self.metrics["registered_agents"] = len(self.local_agents)
        self.metrics["active_agents"] = len([
            a for a in self.local_agents.values()
            if a.status == AgentStatus.ACTIVE
        ])
    
    async def _start_watch_agents(self):
        """Start watching agent changes"""
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run watch in thread pool
            await loop.run_in_executor(
                self.executor,
                self._watch_etcd_changes
            )
            
        except Exception as e:
            self.logger.error(f"Error watching agent changes: {e}")
    
    def _watch_etcd_changes(self):
        """Watch etcd changes (runs in thread pool)"""
        
        try:
            events_iterator, cancel = self.etcd_client.watch_prefix(self.prefix)
            
            for event in events_iterator:
                try:
                    # Process event in main event loop
                    asyncio.create_task(self._handle_etcd_event(event))
                    
                except Exception as e:
                    self.logger.error(f"Error processing etcd event: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in etcd watch: {e}")
    
    async def _handle_etcd_event(self, event):
        """Handle etcd event"""
        
        try:
            key = event.key.decode('utf-8')
            agent_id = key.replace(self.prefix, '')
            
            if event.type == etcd3.EventType.PUT:
                # Agent updated or added
                agent_data = json.loads(event.value.decode('utf-8'))
                
                # Reconstruct AgentMetadata object
                capabilities = [
                    AgentCapability(**cap) for cap in agent_data["capabilities"]
                ]
                
                metadata = AgentMetadata(
                    agent_id=agent_data["agent_id"],
                    agent_type=agent_data["agent_type"],
                    name=agent_data["name"],
                    version=agent_data["version"],
                    capabilities=capabilities,
                    host=agent_data["host"],
                    port=agent_data["port"],
                    endpoint=agent_data["endpoint"],
                    health_endpoint=agent_data["health_endpoint"],
                    resources=agent_data.get("resources", {}),
                    tags=agent_data.get("tags", []),
                    group=agent_data.get("group", "default"),
                    region=agent_data.get("region", "default"),
                    status=AgentStatus(agent_data.get("status", "active")),
                    created_at=datetime.fromisoformat(agent_data["created_at"]),
                    last_heartbeat=datetime.fromisoformat(agent_data["last_heartbeat"]),
                    request_count=agent_data.get("request_count", 0),
                    error_count=agent_data.get("error_count", 0),
                    avg_response_time=agent_data.get("avg_response_time", 0.0)
                )
                
                # Update local cache
                old_metadata = self.local_agents.get(agent_id)
                self.local_agents[agent_id] = metadata
                
                if old_metadata:
                    self._remove_from_indexes(old_metadata)
                
                self._update_indexes(metadata)
                
                # Trigger event
                if old_metadata:
                    await self._trigger_event("agent_updated", metadata)
                else:
                    await self._trigger_event("agent_registered", metadata)
                    await self._start_health_check_for_agent(agent_id)
            
            elif event.type == etcd3.EventType.DELETE:
                # Agent deleted
                if agent_id in self.local_agents:
                    metadata = self.local_agents[agent_id]
                    
                    del self.local_agents[agent_id]
                    self._remove_from_indexes(metadata)
                    
                    # Stop health check
                    await self._stop_health_check_for_agent(agent_id)
                    
                    # Trigger event
                    await self._trigger_event("agent_deregistered", metadata)
            
            # Update metrics
            self._update_metrics()
            
        except Exception as e:
            self.logger.error(f"Error handling etcd event: {e}")
    
    async def _start_lease_renewal(self):
        """Start lease renewal"""
        
        while True:
            try:
                await asyncio.sleep(self.lease_ttl // 3)  # Renew every 1/3 TTL
                
                if self.lease:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.executor,
                        self.lease.refresh
                    )
                    
            except Exception as e:
                self.logger.error(f"Error renewing lease: {e}")
                
                # Recreate lease
                try:
                    self.lease = self.etcd_client.lease(self.lease_ttl)
                except Exception as lease_error:
                    self.logger.error(f"Failed to recreate lease: {lease_error}")
    
    async def _start_periodic_health_check(self):
        """Start periodic health check"""
        
        while True:
            try:
                await asyncio.sleep(self.health_check_strategy.interval)
                
                # Check health status of all agents
                for agent_id in list(self.local_agents.keys()):
                    if agent_id not in self.health_check_tasks:
                        await self._start_health_check_for_agent(agent_id)
                
            except Exception as e:
                self.logger.error(f"Error in periodic health check: {e}")
    
    async def _start_health_check_for_agent(self, agent_id: str):
        """Start health check for agent"""
        
        if agent_id not in self.local_agents:
            return
        
        if agent_id in self.health_check_tasks:
            # Health check task already running
            return
        
        # Create health check task
        task = asyncio.create_task(self._health_check_loop(agent_id))
        self.health_check_tasks[agent_id] = task
        
        self.logger.debug(f"Started health check for agent {agent_id}")
    
    async def _stop_health_check_for_agent(self, agent_id: str):
        """Stop health check for agent"""
        
        if agent_id in self.health_check_tasks:
            task = self.health_check_tasks[agent_id]
            task.cancel()
            del self.health_check_tasks[agent_id]
            
            self.logger.debug(f"Stopped health check for agent {agent_id}")
    
    async def _health_check_loop(self, agent_id: str):
        """Health check loop"""
        
        consecutive_failures = 0
        consecutive_successes = 0
        
        while True:
            try:
                await asyncio.sleep(self.health_check_strategy.interval)
                
                if agent_id not in self.local_agents:
                    break
                
                agent = self.local_agents[agent_id]
                
                # Perform health check
                is_healthy = await self._perform_health_check(agent)
                
                self.metrics["health_checks"] += 1
                
                if is_healthy:
                    consecutive_failures = 0
                    consecutive_successes += 1
                    
                    # If previously unhealthy, now recovered
                    if (agent.status == AgentStatus.UNHEALTHY and 
                        consecutive_successes >= self.health_check_strategy.success_threshold):
                        await self.update_agent_status(agent_id, AgentStatus.ACTIVE)
                        self.logger.info(f"Agent {agent_id} recovered to healthy state")
                
                else:
                    consecutive_successes = 0
                    consecutive_failures += 1
                    
                    self.metrics["failed_health_checks"] += 1
                    
                    # If consecutive failures exceed threshold, mark as unhealthy
                    if (agent.status != AgentStatus.UNHEALTHY and
                        consecutive_failures >= self.health_check_strategy.failure_threshold):
                        await self.update_agent_status(agent_id, AgentStatus.UNHEALTHY)
                        self.logger.warning(f"Agent {agent_id} marked as unhealthy after {consecutive_failures} failures")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop for agent {agent_id}: {e}")
    
    async def _perform_health_check(self, agent: AgentMetadata) -> bool:
        """Perform health check"""
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.health_check_strategy.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(agent.health_endpoint) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.debug(f"Health check failed for agent {agent.agent_id}: {e}")
            return False
    
    async def _trigger_event(self, event_type: str, metadata: AgentMetadata):
        """Trigger event callbacks"""
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(metadata)
                    else:
                        handler(metadata)
                except Exception as e:
                    self.logger.error(f"Error in event handler {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        stats = self.metrics.copy()
        
        # Add detailed statistics
        stats.update({
            "agents_by_status": {
                status.value: len([a for a in self.local_agents.values() if a.status == status])
                for status in AgentStatus
            },
            "agents_by_type": {},
            "capabilities": list(self.capability_index.keys()),
            "tags": list(self.tag_index.keys()),
            "avg_response_time": sum(a.avg_response_time for a in self.local_agents.values()) / len(self.local_agents) if self.local_agents else 0
        })
        
        # Statistics by type
        for agent in self.local_agents.values():
            agent_type = agent.agent_type
            if agent_type not in stats["agents_by_type"]:
                stats["agents_by_type"][agent_type] = 0
            stats["agents_by_type"][agent_type] += 1
        
        return stats
    
    async def cleanup(self):
        """Clean up resources"""
        
        try:
            # Stop all health check tasks
            for task in self.health_check_tasks.values():
                task.cancel()
            
            # Wait for tasks to finish
            if self.health_check_tasks:
                await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
            
            # Close etcd connection
            if self.etcd_client:
                self.etcd_client.close()
            
            # Close thread pool
            self.executor.shutdown(wait=True)
            
            self.logger.info("Service registry cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class LoadBalancer:
    """Load balancer"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin_strategy,
            "weighted_random": self._weighted_random_strategy,
            "least_connections": self._least_connections_strategy,
            "least_response_time": self._least_response_time_strategy,
            "capability_based": self._capability_based_strategy,
            "resource_aware": self._resource_aware_strategy
        }
        
        # Connection statistics
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        
        # Round robin counters
        self._round_robin_counters: Dict[str, int] = {}
    
    async def select_agent(
        self,
        capability: Optional[str] = None,
        strategy: str = "capability_based",
        tags: Optional[List[str]] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Select most suitable agent"""
        
        try:
            # Discover candidate agents
            candidates = await self.registry.discover_agents(
                capability=capability,
                tags=tags,
                status=AgentStatus.ACTIVE
            )
            
            if not candidates:
                self.logger.warning(f"No available agents for capability: {capability}")
                return None
            
            # Apply load balancing strategy
            if strategy in self.strategies:
                selected_agent = await self.strategies[strategy](candidates, requirements)
                
                # Update connection statistics
                if selected_agent:
                    await self._update_connection_stats(selected_agent.agent_id, "selected")
                
                return selected_agent
            else:
                self.logger.error(f"Unknown load balancing strategy: {strategy}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error selecting agent: {e}")
            return None
    
    async def _round_robin_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Round robin strategy"""
        
        if not agents:
            return None
        
        # Use capability as round robin group identifier
        group_key = "default"
        if agents and agents[0].capabilities:
            group_key = agents[0].capabilities[0].name
        
        if group_key not in self._round_robin_counters:
            self._round_robin_counters[group_key] = 0
        
        # Select next agent
        selected_index = self._round_robin_counters[group_key] % len(agents)
        selected_agent = agents[selected_index]
        
        # Update counter
        self._round_robin_counters[group_key] += 1
        
        return selected_agent
    
    async def _weighted_random_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Weighted random strategy"""
        
        import random
        
        if not agents:
            return None
        
        # Calculate weights
        weights = []
        for agent in agents:
            # Weight based on resource usage
            cpu_usage = agent.resources.get("cpu_usage", 0.5)
            memory_usage = agent.resources.get("memory_usage", 0.5)
            
            # Error rate impact
            error_rate = agent.error_count / max(agent.request_count, 1)
            
            # Weight is inversely proportional to load and proportional to reliability
            weight = (2 - cpu_usage - memory_usage) * (1 - error_rate) / 2
            weights.append(max(0.1, weight))  # Minimum weight 0.1
        
        # Weighted random selection
        selected_agent = random.choices(agents, weights=weights)[0]
        return selected_agent
    
    async def _least_connections_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Least connections strategy"""
        
        if not agents:
            return None
        
        # Select agent with least connections
        min_connections = float('inf')
        selected_agent = None
        
        for agent in agents:
            connections = self._get_connection_count(agent.agent_id)
            if connections < min_connections:
                min_connections = connections
                selected_agent = agent
        
        return selected_agent
    
    async def _least_response_time_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Least response time strategy"""
        
        if not agents:
            return None
        
        # Select agent with shortest average response time
        selected_agent = min(agents, key=lambda a: a.avg_response_time)
        return selected_agent
    
    async def _capability_based_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Capability-based strategy"""
        
        if not agents:
            return None
        
        # If no special requirements, use weighted random
        if not requirements:
            return await self._weighted_random_strategy(agents, requirements)
        
        # Calculate adaptation score for each agent
        scored_agents = []
        
        for agent in agents:
            score = await self._calculate_capability_score(agent, requirements)
            scored_agents.append((agent, score))
        
        # Select agent with highest score
        if scored_agents:
            scored_agents.sort(key=lambda x: x[1], reverse=True)
            return scored_agents[0][0]
        
        return None
    
    async def _resource_aware_strategy(
        self,
        agents: List[AgentMetadata],
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentMetadata]:
        """Resource-aware strategy"""
        
        if not agents:
            return None
        
        # Calculate resource score for each agent
        scored_agents = []
        
        for agent in agents:
            # Calculate resource availability score
            cpu_available = 1.0 - agent.resources.get("cpu_usage", 0.5)
            memory_available = 1.0 - agent.resources.get("memory_usage", 0.5)
            disk_available = 1.0 - agent.resources.get("disk_usage", 0.3)
            
            resource_score = (cpu_available + memory_available + disk_available) / 3
            
            # Consider connection load
            connection_count = self._get_connection_count(agent.agent_id)
            connection_penalty = min(connection_count * 0.1, 0.5)  # Maximum penalty 0.5
            
            # Final score
            final_score = resource_score - connection_penalty
            
            scored_agents.append((agent, final_score))
        
        # Select agent with highest score
        if scored_agents:
            scored_agents.sort(key=lambda x: x[1], reverse=True)
            return scored_agents[0][0]
        
        return None
    
    async def _calculate_capability_score(
        self, 
        agent: AgentMetadata, 
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate agent capability matching score"""
        
        score = 0.0
        
        # Performance requirement matching
        required_throughput = requirements.get("min_throughput", 0)
        required_accuracy = requirements.get("min_accuracy", 0)
        max_latency = requirements.get("max_latency", float('inf'))
        
        for capability in agent.capabilities:
            metrics = capability.performance_metrics
            
            # Throughput score
            throughput = metrics.get("throughput", 0)
            if throughput >= required_throughput:
                score += min(throughput / max(required_throughput, 1), 2.0)  # Maximum 2 points
            
            # Accuracy score
            accuracy = metrics.get("accuracy", 0)
            if accuracy >= required_accuracy:
                score += accuracy * 2  # Maximum 2 points
            
            # Latency score
            latency = metrics.get("avg_latency", float('inf'))
            if latency <= max_latency:
                latency_score = max_latency / max(latency, 0.1)
                score += min(latency_score, 2.0)  # Maximum 2 points
        
        # Resource adequacy score
        cpu_available = 1.0 - agent.resources.get("cpu_usage", 0.5)
        memory_available = 1.0 - agent.resources.get("memory_usage", 0.5)
        resource_score = (cpu_available + memory_available) / 2
        score += resource_score * 2  # Maximum 2 points
        
        # Reliability score
        success_rate = 1.0 - (agent.error_count / max(agent.request_count, 1))
        score += success_rate * 2  # Maximum 2 points
        
        return score
    
    def _get_connection_count(self, agent_id: str) -> int:
        """Get current connection count for agent"""
        
        if agent_id not in self.connection_stats:
            self.connection_stats[agent_id] = {
                "active_connections": 0,
                "total_requests": 0,
                "last_selected": None
            }
        
        return self.connection_stats[agent_id]["active_connections"]
    
    async def _update_connection_stats(self, agent_id: str, action: str):
        """Update connection statistics"""
        
        if agent_id not in self.connection_stats:
            self.connection_stats[agent_id] = {
                "active_connections": 0,
                "total_requests": 0,
                "last_selected": None
            }
        
        stats = self.connection_stats[agent_id]
        
        if action == "selected":
            stats["active_connections"] += 1
            stats["total_requests"] += 1
            stats["last_selected"] = datetime.now()
        elif action == "released":
            stats["active_connections"] = max(0, stats["active_connections"] - 1)
    
    async def release_connection(self, agent_id: str):
        """Release connection"""
        await self._update_connection_stats(agent_id, "released")
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        
        return {
            "connection_stats": self.connection_stats,
            "round_robin_counters": self._round_robin_counters,
            "available_strategies": list(self.strategies.keys())
        }


# Usage examples and factory class
class AgentServiceDiscoverySystem:
    """Agent service discovery system main class"""
    
    def __init__(self, etcd_endpoints: List[str]):
        self.etcd_endpoints = etcd_endpoints
        self.registry = None
        self.load_balancer = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize system"""
        
        # Initialize service registry
        self.registry = ServiceRegistry(self.etcd_endpoints)
        await self.registry.initialize()
        
        # Initialize load balancer
        self.load_balancer = LoadBalancer(self.registry)
        
        self.logger.info("Agent service discovery system initialized")
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        version: str,
        capabilities: List[Dict[str, Any]],
        host: str,
        port: int,
        endpoint: str,
        health_endpoint: str,
        **kwargs
    ) -> bool:
        """Register agent (simplified interface)"""
        
        # Build capability objects
        capability_objects = [
            AgentCapability(**cap) for cap in capabilities
        ]
        
        # Build metadata object
        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type=agent_type,
            name=name,
            version=version,
            capabilities=capability_objects,
            host=host,
            port=port,
            endpoint=endpoint,
            health_endpoint=health_endpoint,
            **kwargs
        )
        
        return await self.registry.register_agent(metadata)
    
    async def discover_and_select_agent(
        self,
        capability: str,
        strategy: str = "capability_based",
        **kwargs
    ) -> Optional[AgentMetadata]:
        """Discover and select agent (integrated interface)"""
        
        return await self.load_balancer.select_agent(
            capability=capability,
            strategy=strategy,
            **kwargs
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        
        registry_stats = self.registry.get_registry_stats()
        lb_stats = self.load_balancer.get_load_balancer_stats()
        
        return {
            "registry": registry_stats,
            "load_balancer": lb_stats,
            "system_status": "healthy" if registry_stats["active_agents"] > 0 else "no_agents"
        }
    
    async def cleanup(self):
        """Clean up system resources"""
        
        if self.registry:
            await self.registry.cleanup()
        
        self.logger.info("Agent service discovery system cleanup completed")