"""
Service Discovery API Endpoints

FastAPI endpoints for agent service discovery and registration.
"""

import time
import asyncio
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from ai.service_discovery.core import (
    AgentServiceDiscoverySystem,
    AgentMetadata,
    AgentCapability,
    AgentStatus
)
from ai.service_discovery.models import (
    AgentRegistrationRequest,
    AgentMetadataResponse,
    AgentDiscoveryRequest,
    AgentDiscoveryResponse,
    AgentStatusUpdate,
    AgentMetricsUpdate,
    LoadBalancerRequest,
    LoadBalancerResponse,
    ServiceStats,
    HealthCheckResponse,
    ErrorResponse
)
from ai.service_discovery.config import default_config
from core.config import get_settings


router = APIRouter(prefix="/service-discovery", tags=["service-discovery"])
logger = logging.getLogger(__name__)

# Global service discovery system instance
_service_discovery_system: Optional[AgentServiceDiscoverySystem] = None
_system_start_time = time.time()


async def get_service_discovery_system() -> AgentServiceDiscoverySystem:
    """Get or initialize the service discovery system"""
    global _service_discovery_system
    
    if _service_discovery_system is None:
        settings = get_settings()
        
        # Use etcd endpoints from environment or default config
        etcd_endpoints = getattr(settings, 'etcd_endpoints', default_config.etcd_endpoints)
        
        _service_discovery_system = AgentServiceDiscoverySystem(etcd_endpoints)
        
        try:
            await _service_discovery_system.initialize()
            logger.info("Service discovery system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize service discovery system: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service discovery system initialization failed: {str(e)}"
            )
    
    return _service_discovery_system


def _convert_to_response_model(agent_metadata: AgentMetadata) -> AgentMetadataResponse:
    """Convert core AgentMetadata to response model"""
    return AgentMetadataResponse(
        agent_id=agent_metadata.agent_id,
        agent_type=agent_metadata.agent_type,
        name=agent_metadata.name,
        version=agent_metadata.version,
        capabilities=[
            {
                "name": cap.name,
                "description": cap.description,
                "version": cap.version,
                "input_schema": cap.input_schema,
                "output_schema": cap.output_schema,
                "performance_metrics": cap.performance_metrics,
                "constraints": cap.constraints
            } for cap in agent_metadata.capabilities
        ],
        host=agent_metadata.host,
        port=agent_metadata.port,
        endpoint=agent_metadata.endpoint,
        health_endpoint=agent_metadata.health_endpoint,
        resources=agent_metadata.resources,
        tags=agent_metadata.tags,
        group=agent_metadata.group,
        region=agent_metadata.region,
        status=agent_metadata.status.value,
        created_at=agent_metadata.created_at,
        last_heartbeat=agent_metadata.last_heartbeat,
        request_count=agent_metadata.request_count,
        error_count=agent_metadata.error_count,
        avg_response_time=agent_metadata.avg_response_time
    )


@router.post("/agents", 
             response_model=dict,
             status_code=status.HTTP_201_CREATED,
             summary="Register Agent",
             description="Register a new agent in the service discovery system")
async def register_agent(
    request: AgentRegistrationRequest,
    background_tasks: BackgroundTasks,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Register an agent"""
    try:
        # Convert request to core model
        capabilities = []
        for cap_data in request.capabilities:
            capability = AgentCapability(
                name=cap_data.name,
                description=cap_data.description,
                version=cap_data.version,
                input_schema=cap_data.input_schema,
                output_schema=cap_data.output_schema,
                performance_metrics=cap_data.performance_metrics,
                constraints=cap_data.constraints
            )
            capabilities.append(capability)
        
        # Register agent
        success = await system.register_agent(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            name=request.name,
            version=request.version,
            capabilities=[cap.__dict__ for cap in capabilities],
            host=request.host,
            port=request.port,
            endpoint=request.endpoint,
            health_endpoint=request.health_endpoint,
            resources=request.resources,
            tags=request.tags,
            group=request.group,
            region=request.region
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register agent. Check agent data validity."
            )
        
        logger.info(f"Agent {request.agent_id} registered successfully")
        
        return {
            "message": "Agent registered successfully",
            "agent_id": request.agent_id,
            "status": "registered"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering agent {request.agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during agent registration: {str(e)}"
        )


@router.get("/agents", 
            response_model=AgentDiscoveryResponse,
            summary="Discover Agents",
            description="Discover agents based on capabilities, tags, and other criteria")
async def discover_agents(
    capability: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated string
    status_filter: Optional[str] = None,
    group: Optional[str] = None,
    region: Optional[str] = None,
    limit: Optional[int] = None,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Discover agents"""
    try:
        start_time = time.time()
        
        # Parse tags from comma-separated string
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Parse status
        agent_status = None
        if status_filter:
            try:
                agent_status = AgentStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status filter: {status_filter}. Must be one of: {[s.value for s in AgentStatus]}"
                )
        
        # Discover agents
        agents = await system.registry.discover_agents(
            capability=capability,
            tags=tag_list,
            status=agent_status,
            group=group,
            region=region,
            limit=limit
        )
        
        query_time = time.time() - start_time
        
        # Convert to response models
        response_agents = [_convert_to_response_model(agent) for agent in agents]
        
        return AgentDiscoveryResponse(
            agents=response_agents,
            total_count=len(response_agents),
            query_time=query_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error discovering agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during agent discovery: {str(e)}"
        )


@router.get("/agents/{agent_id}",
            response_model=AgentMetadataResponse,
            summary="Get Agent Details",
            description="Get detailed information about a specific agent")
async def get_agent(
    agent_id: str,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Get agent details"""
    try:
        agent = await system.registry.get_agent(agent_id)
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        return _convert_to_response_model(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/agents/{agent_id}/status",
            response_model=dict,
            summary="Update Agent Status",
            description="Update the status of a specific agent")
async def update_agent_status(
    agent_id: str,
    request: AgentStatusUpdate,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Update agent status"""
    try:
        # Convert status to enum
        status_enum = AgentStatus(request.status.value)
        
        success = await system.registry.update_agent_status(agent_id, status_enum)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        logger.info(f"Agent {agent_id} status updated to {request.status}")
        
        return {
            "message": "Agent status updated successfully",
            "agent_id": agent_id,
            "new_status": request.status.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent status {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/agents/{agent_id}/metrics",
            response_model=dict,
            summary="Update Agent Metrics",
            description="Update performance metrics for a specific agent")
async def update_agent_metrics(
    agent_id: str,
    request: AgentMetricsUpdate,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Update agent metrics"""
    try:
        success = await system.registry.update_agent_metrics(
            agent_id=agent_id,
            request_count=request.request_count,
            error_count=request.error_count,
            avg_response_time=request.avg_response_time
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        logger.info(f"Agent {agent_id} metrics updated")
        
        return {
            "message": "Agent metrics updated successfully",
            "agent_id": agent_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent metrics {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/agents/{agent_id}",
               response_model=dict,
               summary="Deregister Agent",
               description="Remove an agent from the service discovery system")
async def deregister_agent(
    agent_id: str,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Deregister agent"""
    try:
        success = await system.registry.deregister_agent(agent_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        
        logger.info(f"Agent {agent_id} deregistered successfully")
        
        return {
            "message": "Agent deregistered successfully",
            "agent_id": agent_id,
            "status": "deregistered"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deregistering agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/load-balancer/select",
             response_model=LoadBalancerResponse,
             summary="Select Agent",
             description="Select the best agent using load balancing strategies")
async def select_agent(
    request: LoadBalancerRequest,
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Select agent using load balancer"""
    try:
        start_time = time.time()
        
        selected_agent = await system.discover_and_select_agent(
            capability=request.capability,
            strategy=request.strategy,
            tags=request.tags,
            requirements=request.requirements
        )
        
        selection_time = time.time() - start_time
        
        response_agent = None
        if selected_agent:
            response_agent = _convert_to_response_model(selected_agent)
        
        return LoadBalancerResponse(
            selected_agent=response_agent,
            selection_time=selection_time,
            strategy_used=request.strategy
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/stats",
            response_model=ServiceStats,
            summary="Get System Statistics",
            description="Get comprehensive statistics about the service discovery system")
async def get_system_stats(
    system: AgentServiceDiscoverySystem = Depends(get_service_discovery_system)
):
    """Get system statistics"""
    try:
        stats = system.get_system_stats()
        
        return ServiceStats(
            registry=stats["registry"],
            load_balancer=stats["load_balancer"],
            system_status=stats["system_status"]
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/health",
            response_model=HealthCheckResponse,
            summary="Health Check",
            description="Check the health status of the service discovery system")
async def health_check():
    """Health check endpoint"""
    try:
        current_time = time.time()
        uptime = current_time - _system_start_time
        
        # Check if system is initialized
        system_healthy = _service_discovery_system is not None
        
        return HealthCheckResponse(
            status="healthy" if system_healthy else "initializing",
            timestamp=current_time,
            version="1.0.0",
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/config",
            response_model=dict,
            summary="Get Configuration",
            description="Get current service discovery configuration")
async def get_configuration():
    """Get configuration"""
    try:
        from ai.service_discovery.config import LoadBalancerConfig, HealthCheckConfig
        
        return {
            "load_balancer": {
                "strategies": LoadBalancerConfig.list_strategies(),
                "strategy_details": {
                    strategy: LoadBalancerConfig.get_strategy_info(strategy)
                    for strategy in LoadBalancerConfig.list_strategies()
                }
            },
            "health_check": {
                "presets": HealthCheckConfig.list_presets(),
                "preset_details": {
                    preset: HealthCheckConfig.get_preset(preset)
                    for preset in HealthCheckConfig.list_presets()
                }
            },
            "system": {
                "etcd_endpoints": default_config.etcd_endpoints,
                "etcd_prefix": default_config.etcd_prefix,
                "max_agents_per_discovery": default_config.max_agents_per_discovery
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


# Startup event to ensure system is initialized
@router.on_event("startup")
async def startup_event():
    """Initialize service discovery system on startup"""
    try:
        # Force initialization of the service discovery system
        await get_service_discovery_system()
        logger.info("Service discovery API started successfully")
    except Exception as e:
        logger.error(f"Failed to start service discovery API: {e}")
        # Don't raise here as it would prevent the API from starting


# Shutdown event for cleanup
@router.on_event("shutdown") 
async def shutdown_event():
    """Clean up service discovery system on shutdown"""
    global _service_discovery_system
    
    if _service_discovery_system:
        try:
            await _service_discovery_system.cleanup()
            logger.info("Service discovery system cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during service discovery cleanup: {e}")
        finally:
            _service_discovery_system = None