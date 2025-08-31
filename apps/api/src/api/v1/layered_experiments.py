"""
分层实验管理API端点
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from core.database import get_db
from repositories.experiment_repository import ExperimentRepository
from services.layered_experiment_manager import (
    LayeredExperimentManager, ExperimentLayer, ExperimentGroup, LayerType, ConflictResolution
)
from core.logging import logger


router = APIRouter(prefix="/experiments/layers", tags=["layered-experiments"])


# 依赖注入
def get_experiment_repository(db: Session = Depends(get_db)) -> ExperimentRepository:
    return ExperimentRepository(db)

def get_layered_experiment_manager(
    experiment_repo: ExperimentRepository = Depends(get_experiment_repository)
) -> LayeredExperimentManager:
    return LayeredExperimentManager(experiment_repo)


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_experiment_layer(
    layer_data: Dict[str, Any],
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """创建实验层"""
    try:
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
        overview = manager.get_all_layer_overview()
        
        if active_only:
            overview["layers"] = [
                layer for layer in overview["layers"]
                if layer.get("is_active", False)
            ]
        
        return overview
        
    except Exception as e:
        logger.error(f"Failed to list layers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve layers"
        )


@router.get("/{layer_id}")
async def get_layer_details(
    layer_id: str,
    manager: LayeredExperimentManager = Depends(get_layered_experiment_manager)
):
    """获取层详细信息"""
    try:
        statistics = manager.get_layer_statistics(layer_id)
        
        if "error" in statistics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=statistics["error"]
            )
        
        return statistics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get layer {layer_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve layer details"
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
        num_users = simulation_config.get("num_users", 10000)
        user_attributes_samples = simulation_config.get("user_attributes_samples", [])
        
        simulation_results = {
            "simulation_config": {
                "num_users": num_users,
                "has_user_samples": len(user_attributes_samples) > 0
            },
            "assignment_results": {},
            "layer_utilization": {},
            "conflicts_detected": 0
        }
        
        # 模拟用户分配
        conflicts = 0
        for i in range(num_users):
            user_id = f"sim_user_{i:06d}"
            user_attrs = user_attributes_samples[i % len(user_attributes_samples)] if user_attributes_samples else {}
            
            try:
                assignments = manager.assign_user_to_layers(user_id, user_attrs)
                
                for layer_id, assignment in assignments.items():
                    if layer_id not in simulation_results["assignment_results"]:
                        simulation_results["assignment_results"][layer_id] = {}
                    
                    experiment_id = assignment.assigned_experiment or "no_experiment"
                    if experiment_id not in simulation_results["assignment_results"][layer_id]:
                        simulation_results["assignment_results"][layer_id][experiment_id] = 0
                    
                    simulation_results["assignment_results"][layer_id][experiment_id] += 1
                    
            except Exception as e:
                conflicts += 1
                logger.debug(f"Simulation conflict for user {user_id}: {str(e)}")
        
        # 计算层利用率
        for layer_id, assignments in simulation_results["assignment_results"].items():
            total_in_layer = sum(assignments.values())
            simulation_results["layer_utilization"][layer_id] = {
                "total_users": total_in_layer,
                "utilization_percentage": (total_in_layer / num_users) * 100,
                "experiment_distribution": {
                    exp_id: (count / total_in_layer) * 100 if total_in_layer > 0 else 0
                    for exp_id, count in assignments.items()
                }
            }
        
        simulation_results["conflicts_detected"] = conflicts
        
        return simulation_results
        
    except Exception as e:
        logger.error(f"Failed to simulate layer assignment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to simulate layer assignment"
        )