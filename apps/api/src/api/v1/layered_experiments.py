"""
分层实验管理API端点
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, Depends, HTTPException, status, Query
from collections import defaultdict
from src.services.layered_experiment_manager import (
    LayeredExperimentManager, ExperimentLayer, ExperimentGroup, LayerType, ConflictResolution
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/experiments/layers", tags=["layered-experiments"])

_layered_experiment_manager = LayeredExperimentManager()

def get_layered_experiment_manager(
) -> LayeredExperimentManager:
    return _layered_experiment_manager

def _serialize_layer(layer: ExperimentLayer) -> Dict[str, Any]:
    return {
        "layer_id": layer.layer_id,
        "name": layer.name,
        "description": layer.description,
        "layer_type": layer.layer_type.value,
        "traffic_percentage": layer.traffic_percentage,
        "priority": layer.priority,
        "is_active": layer.is_active,
        "max_experiments": layer.max_experiments,
        "conflict_resolution": layer.conflict_resolution.value,
        "metadata": layer.metadata,
        "created_at": layer.created_at.isoformat(),
    }

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_experiment_layer(
    layer_data: Dict[str, Any],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """创建实验层"""
    try:
        if not layer_data.get("layer_id") or not layer_data.get("name") or layer_data.get("traffic_percentage") is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="layer_id、name、traffic_percentage 必填")
        layer = ExperimentLayer(
            layer_id=layer_data.get("layer_id"),
            name=layer_data.get("name"),
            description=layer_data.get("description", ""),
            layer_type=LayerType(layer_data.get("layer_type", LayerType.MUTUALLY_EXCLUSIVE.value)),
            traffic_percentage=layer_data.get("traffic_percentage"),
            priority=layer_data.get("priority", 0),
            is_active=layer_data.get("is_active", True),
            max_experiments=layer_data.get("max_experiments"),
            conflict_resolution=ConflictResolution(layer_data.get("conflict_resolution", ConflictResolution.PRIORITY_BASED.value)),
            metadata=layer_data.get("metadata", {})
        )
        
        created_layer = manager.create_layer(layer)
        
        return {
            "layer_id": created_layer.layer_id,
            "name": created_layer.name,
            "layer_type": created_layer.layer_type.value,
            "traffic_percentage": created_layer.traffic_percentage,
            "priority": created_layer.priority,
            "is_active": created_layer.is_active,
            "created_at": created_layer.created_at,
            "message": "Layer created successfully"
        }
        
    except ValueError as e:
        logger.error(f"Failed to create layer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating layer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create layer"
        )

@router.get("/")
async def list_experiment_layers(
    active_only: bool = Query(True, description="只返回活跃的层"),
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取实验层列表"""
    try:
        layers = [
            _serialize_layer(layer)
            for layer in manager.list_layers()
            if (layer.is_active or not active_only)
        ]
        return {"layers": layers, "total": len(layers)}
        
    except Exception as e:
        logger.error(f"Failed to list layers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve layers"
        )

@router.get("/metrics")
async def get_system_metrics(
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取系统指标"""
    try:
        layers = manager.list_layers()
        active_layers = [l for l in layers if l.is_active]
        total_traffic_allocated = sum(l.traffic_percentage for l in active_layers)
        avg_layer_traffic = total_traffic_allocated / len(active_layers) if active_layers else 0.0
        traffic_utilization = total_traffic_allocated
        return {
            "total_traffic_allocated": total_traffic_allocated,
            "avg_layer_traffic": avg_layer_traffic,
            "traffic_utilization": traffic_utilization,
            "active_layers": len(active_layers),
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

@router.get("/conflicts")
async def list_conflicts(
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager),
):
    """获取冲突列表"""
    try:
        validation = manager.validate_layer_configuration()
        conflicts: List[Dict[str, Any]] = []

        for idx, msg in enumerate(validation.get("errors") or []):
            conflicts.append(
                {
                    "id": f"error_{idx}",
                    "type": "error",
                    "description": msg,
                    "affected_layers": [],
                }
            )

        for idx, msg in enumerate(validation.get("warnings") or []):
            conflicts.append(
                {
                    "id": f"warning_{idx}",
                    "type": "warning",
                    "description": msg,
                    "affected_layers": [],
                }
            )

        return conflicts
    except Exception as e:
        logger.error(f"Failed to list conflicts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conflicts"
        )

@router.get("/{layer_id}")
async def get_layer_details(
    layer_id: str,
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取层详细信息"""
    try:
        layer = manager.get_layer(layer_id)
        return _serialize_layer(layer)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get layer {layer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve layer details"
        )

@router.put("/{layer_id}")
async def update_layer(
    layer_id: str,
    layer_data: Dict[str, Any],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager),
):
    """更新实验层"""
    try:
        updated = manager.update_layer(layer_id, layer_data)
        return _serialize_layer(updated)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update layer {layer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update layer"
        )

@router.delete("/{layer_id}")
async def delete_layer(
    layer_id: str,
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager),
):
    """删除实验层"""
    try:
        manager.delete_layer(layer_id)
        return {"layer_id": layer_id, "message": "Layer deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete layer {layer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete layer"
        )

@router.post("/{layer_id}/groups", status_code=status.HTTP_201_CREATED)
async def create_experiment_group(
    layer_id: str,
    group_data: Dict[str, Any],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """在指定层中创建实验组"""
    try:
        group = ExperimentGroup(
            group_id=group_data.get("group_id"),
            name=group_data.get("name"),
            layer_id=layer_id,
            experiments=group_data.get("experiments", []),
            traffic_split=group_data.get("traffic_split", {}),
            is_active=group_data.get("is_active", True),
            priority=group_data.get("priority", 0),
            max_group_size=group_data.get("max_group_size"),
            metadata=group_data.get("metadata", {})
        )
        
        created_group = manager.create_experiment_group(group)
        
        return {
            "group_id": created_group.group_id,
            "name": created_group.name,
            "layer_id": created_group.layer_id,
            "experiments": created_group.experiments,
            "traffic_split": created_group.traffic_split,
            "is_active": created_group.is_active,
            "message": "Experiment group created successfully"
        }
        
    except ValueError as e:
        logger.error(f"Failed to create group: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating group: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create experiment group"
        )

@router.post("/assign/{user_id}")
async def assign_user_to_layers(
    user_id: str,
    user_attributes: Optional[Dict[str, Any]] = None,
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """为用户分配实验层"""
    try:
        assignments = manager.assign_user_to_layers(user_id, user_attributes or {})
        
        result = {
            "user_id": user_id,
            "total_layers_assigned": len(assignments),
            "assignments": []
        }
        
        for layer_id, assignment in assignments.items():
            result["assignments"].append({
                "layer_id": layer_id,
                "assigned_experiment": assignment.assigned_experiment,
                "assignment_reason": assignment.assignment_reason,
                "assigned_at": assignment.assigned_at,
                "metadata": assignment.metadata
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to assign user {user_id} to layers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign user to layers"
        )

@router.get("/user/{user_id}/experiments")
async def get_user_experiments(
    user_id: str,
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取用户参与的所有实验"""
    try:
        user_experiments = manager.get_user_experiments(user_id)
        
        return {
            "user_id": user_id,
            "total_experiments": len(user_experiments),
            "experiments": user_experiments
        }
        
    except Exception as e:
        logger.error(f"Failed to get user experiments for {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user experiments"
        )

@router.post("/user/{user_id}/check-conflicts")
async def check_experiment_conflicts(
    user_id: str,
    experiment_id: str = Query(..., description="要检查的实验ID"),
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """检查实验冲突"""
    try:
        conflict_result = manager.check_experiment_conflicts(user_id, experiment_id)
        
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            **conflict_result
        }
        
    except Exception as e:
        logger.error(f"Failed to check conflicts for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check experiment conflicts"
        )

@router.post("/validate")
async def validate_layer_configuration(
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """验证层配置"""
    try:
        validation_result = manager.validate_layer_configuration()
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Failed to validate layer configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate configuration"
        )

@router.delete("/user/{user_id}/assignments")
async def clear_user_assignments(
    user_id: str,
    layer_id: Optional[str] = Query(None, description="特定层ID，为空则清理所有层"),
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """清理用户分配"""
    try:
        manager.clear_user_assignments(user_id, layer_id)
        
        return {
            "user_id": user_id,
            "layer_id": layer_id,
            "message": "User assignments cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear assignments for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear user assignments"
        )

@router.post("/batch-assign")
async def batch_assign_users(
    user_requests: List[Dict[str, Any]],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """批量分配用户到层"""
    try:
        results = []
        
        for user_req in user_requests:
            user_id = user_req.get("user_id")
            user_attributes = user_req.get("user_attributes", {})
            
            try:
                assignments = manager.assign_user_to_layers(user_id, user_attributes)
                
                result = {
                    "user_id": user_id,
                    "success": True,
                    "total_layers_assigned": len(assignments),
                    "assigned_experiments": [
                        assignment.assigned_experiment 
                        for assignment in assignments.values() 
                        if assignment.assigned_experiment
                    ]
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to assign user {user_id}: {str(e)}")
                results.append({
                    "user_id": user_id,
                    "success": False,
                    "error": str(e)
                })
        
        successful_assignments = len([r for r in results if r.get("success")])
        failed_assignments = len(results) - successful_assignments
        
        return {
            "total_requests": len(user_requests),
            "successful_assignments": successful_assignments,
            "failed_assignments": failed_assignments,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to batch assign users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to batch assign users"
        )

@router.get("/statistics/overview")
async def get_layers_overview(
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取所有层的统计概览"""
    try:
        overview = manager.get_all_layer_overview()
        return overview
        
    except Exception as e:
        logger.error(f"Failed to get layers overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve layers overview"
        )

@router.post("/simulate/assignment")
async def simulate_layer_assignment(
    simulation_config: Dict[str, Any],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """模拟层分配"""
    try:
        users = simulation_config.get("users")
        if not users:
            num_users = int(simulation_config.get("num_users", 100))
            users = [{"user_id": f"sim_user_{i}", "attributes": {}} for i in range(num_users)]

        assignment_results: Dict[str, Any] = {}
        layer_counts: Dict[str, int] = defaultdict(int)
        conflicts_detected = 0

        for item in users:
            user_id = item.get("user_id")
            if not user_id:
                continue
            attributes = item.get("attributes") or {}

            assignments = manager.assign_user_to_layers(user_id, attributes)
            assignment_results[user_id] = {
                "layers": {
                    layer_id: {
                        "assigned_experiment": a.assigned_experiment,
                        "assignment_reason": a.assignment_reason,
                        "assigned_at": a.assigned_at.isoformat(),
                        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
                        "metadata": a.metadata,
                    }
                    for layer_id, a in assignments.items()
                },
                "assigned_experiments": [
                    a.assigned_experiment for a in assignments.values() if a.assigned_experiment
                ],
            }

            for layer_id, a in assignments.items():
                if a.assigned_experiment:
                    layer_counts[layer_id] += 1
                else:
                    conflicts_detected += 1

        total_users = len(assignment_results) or 1
        layer_utilization = {
            layer_id: {
                "assigned_users": count,
                "utilization_rate": count / total_users,
            }
            for layer_id, count in layer_counts.items()
        }

        return {
            "simulation_config": simulation_config,
            "total_users": len(assignment_results),
            "assignment_results": assignment_results,
            "layer_utilization": layer_utilization,
            "conflicts_detected": conflicts_detected,
        }
    
    except Exception as e:
        logger.error(f"Failed to simulate layer assignment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simulate layer assignment"
        )
