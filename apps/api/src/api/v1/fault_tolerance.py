from src.core.utils.timezone_utils import utc_now
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from src.ai.fault_tolerance import (
    FaultToleranceSystem, 
    FaultType, 
    FaultSeverity,
    BackupType,
    RecoveryStrategy
)
from src.api.base_model import ApiBaseModel
from src.core.dependencies import get_fault_tolerance_system
from fastapi import WebSocket, WebSocketDisconnect

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/fault-tolerance", tags=["Fault Tolerance"])

# Pydantic模型
class BackupRequest(ApiBaseModel):
    component_ids: List[str]
    backup_type: str = "full_backup"

class ConsistencyCheckRequest(ApiBaseModel):
    data_keys: List[str]

class FaultInjectionRequest(ApiBaseModel):
    component_id: str
    fault_type: str
    duration_seconds: int = 60

class RestoreBackupRequest(ApiBaseModel):
    backup_id: str
    target_component_id: Optional[str] = None

class ForceConsistencyRepairRequest(ApiBaseModel):
    data_key: str
    authoritative_component_id: str

# 状态和概览端点
@router.get("/status")
async def get_fault_tolerance_status(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取容错系统状态"""
    try:
        return await system.get_system_status()
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")

@router.get("/health")
async def get_health_summary(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取健康状态摘要"""
    try:
        summary = system.fault_detector.get_system_health_summary()
        services = []
        for health_status in system.fault_detector.health_status.values():
            services.append(
                {
                    "service_id": health_status.component_id,
                    "status": health_status.status,
                    "last_check": health_status.last_check.isoformat(),
                    "response_time_ms": round(health_status.response_time * 1000, 2),
                    "error_rate": health_status.error_rate,
                    "resource_usage": health_status.resource_usage,
                    "custom_metrics": health_status.custom_metrics,
                }
            )
        return {**summary, "services": services}
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health summary: {e}")

@router.get("/health/{component_id}")
async def get_component_health(
    component_id: str,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取组件健康状态"""
    try:
        health_status = await system.get_component_health(component_id)
        return health_status
    except Exception as e:
        logger.error(f"Error getting component health for {component_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get component health: {e}")

@router.get("/metrics")
async def get_fault_tolerance_metrics(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取容错系统指标"""
    try:
        return await system.get_system_metrics()
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {e}")

@router.get("/report")
async def get_detailed_system_report(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取详细的系统报告"""
    try:
        return await system.get_detailed_system_report()
    except Exception as e:
        logger.error(f"Error generating system report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate system report: {e}")

# 故障事件相关端点
@router.get("/faults")
async def get_fault_events(
    fault_type: Optional[str] = Query(None, description="Fault type filter"),
    severity: Optional[str] = Query(None, description="Severity filter"),
    resolved: Optional[bool] = Query(None, description="Resolved status filter"),
    limit: int = Query(100, description="Maximum number of events", ge=1, le=1000),
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> List[Dict[str, Any]]:
    """获取故障事件列表"""
    try:
        # 验证参数
        if fault_type:
            try:
                FaultType(fault_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid fault type: {fault_type}")
        
        if severity:
            try:
                FaultSeverity(severity)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        fault_events = await system.get_fault_events(
            fault_type=fault_type,
            severity=severity,
            resolved=resolved,
            limit=limit
        )
        
        return fault_events
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fault events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get fault events: {e}")

# 恢复相关端点
@router.get("/recovery/statistics")
async def get_recovery_statistics(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取恢复统计信息"""
    try:
        return system.recovery_manager.get_recovery_statistics()
    except Exception as e:
        logger.error(f"Error getting recovery statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery statistics: {e}")

# 备份相关端点
@router.post("/backup/manual")
async def trigger_manual_backup(
    request: BackupRequest,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """触发手动备份"""
    try:
        # 验证备份类型
        try:
            BackupType(request.backup_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid backup type: {request.backup_type}")
        
        results = await system.trigger_manual_backup(request.component_ids)
        
        return {
            "backup_results": results,
            "success_count": len([r for r in results.values() if r]),
            "total_count": len(results),
            "timestamp": utc_now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering manual backup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger manual backup: {e}")

@router.get("/backup/statistics")
async def get_backup_statistics(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取备份统计信息"""
    try:
        return system.backup_manager.get_backup_statistics()
    except Exception as e:
        logger.error(f"Error getting backup statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get backup statistics: {e}")

@router.post("/backup/restore")
async def restore_backup(
    request: RestoreBackupRequest,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """恢复备份"""
    try:
        record = next(
            (r for r in system.backup_manager.backup_records if r.backup_id == request.backup_id),
            None,
        )
        if not record:
            raise HTTPException(status_code=404, detail="备份不存在")
        if not record.valid:
            raise HTTPException(status_code=400, detail="备份无效")

        success = await system.restore_backup(request.backup_id, request.target_component_id)
        
        if success:
            return {
                "status": "restored",
                "backup_id": request.backup_id,
                "target_component": request.target_component_id or "original",
                "timestamp": utc_now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Backup restoration failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring backup: {e}")
        raise HTTPException(status_code=500, detail=f"Backup restoration failed: {e}")

@router.post("/backup/validate")
async def validate_all_backups(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """验证所有备份"""
    try:
        validation_results = await system.validate_all_backups()
        
        valid_count = sum(1 for is_valid in validation_results.values() if is_valid)
        total_count = len(validation_results)
        
        return {
            "validation_results": validation_results,
            "valid_count": valid_count,
            "total_count": total_count,
            "validation_rate": valid_count / max(total_count, 1),
            "timestamp": utc_now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error validating backups: {e}")
        raise HTTPException(status_code=500, detail=f"Backup validation failed: {e}")

# 一致性相关端点
@router.post("/consistency/check")
async def trigger_consistency_check(
    request: ConsistencyCheckRequest,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """触发一致性检查"""
    try:
        result = await system.trigger_manual_consistency_check(request.data_keys)
        
        return {
            "check_id": result.check_id,
            "checked_at": result.checked_at.isoformat(),
            "components": result.components,
            "consistent": result.consistent,
            "inconsistencies_count": len(result.inconsistencies),
            "inconsistencies": result.inconsistencies,
            "repair_actions": result.repair_actions
        }
    except Exception as e:
        logger.error(f"Error triggering consistency check: {e}")
        raise HTTPException(status_code=500, detail=f"Consistency check failed: {e}")

@router.get("/consistency/statistics")
async def get_consistency_statistics(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, Any]:
    """获取一致性统计信息"""
    try:
        return system.consistency_manager.get_consistency_statistics()
    except Exception as e:
        logger.error(f"Error getting consistency statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consistency statistics: {e}")

@router.post("/consistency/{check_id}/repair")
async def repair_consistency_issues(
    check_id: str,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """修复一致性问题"""
    try:
        if not system.consistency_manager.get_check_result_by_id(check_id):
            raise HTTPException(status_code=404, detail="Consistency check not found")
        success = await system.repair_consistency_issues(check_id)
        
        if success:
            return {
                "status": "repaired",
                "check_id": check_id,
                "timestamp": utc_now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Consistency repair failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error repairing consistency issues: {e}")
        raise HTTPException(status_code=500, detail=f"Consistency repair failed: {e}")

@router.post("/consistency/force-repair")
async def force_consistency_repair(
    request: ForceConsistencyRepairRequest,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """强制一致性修复"""
    try:
        success = await system.force_consistency_repair(
            request.data_key,
            request.authoritative_component_id
        )
        
        if success:
            return {
                "status": "force_repaired",
                "data_key": request.data_key,
                "authoritative_component": request.authoritative_component_id,
                "timestamp": utc_now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Force consistency repair failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in force consistency repair: {e}")
        raise HTTPException(status_code=500, detail=f"Force consistency repair failed: {e}")

# 测试相关端点
@router.post("/testing/inject-fault")
async def inject_fault_for_testing(
    request: FaultInjectionRequest,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """注入故障进行测试"""
    try:
        # 验证故障类型
        try:
            fault_type = FaultType(request.fault_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid fault type: {request.fault_type}")
        
        # 验证持续时间
        if request.duration_seconds <= 0 or request.duration_seconds > 3600:
            raise HTTPException(status_code=400, detail="Duration must be between 1 and 3600 seconds")
        
        fault_id = await system.simulate_fault_injection(
            request.component_id,
            fault_type,
            request.duration_seconds
        )
        
        return {
            "status": "fault_injected",
            "fault_id": fault_id,
            "component_id": request.component_id,
            "fault_type": request.fault_type,
            "duration_seconds": request.duration_seconds,
            "timestamp": utc_now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error injecting fault: {e}")
        raise HTTPException(status_code=500, detail=f"Fault injection failed: {e}")

# 枚举端点（用于前端显示）
@router.get("/enums/fault-types")
async def get_fault_types():
    """获取故障类型枚举"""
    return {
        "fault_types": [
            {"value": ft.value, "description": ft.name.replace("_", " ").title()}
            for ft in FaultType
        ]
    }

@router.get("/enums/severities")
async def get_fault_severities():
    """获取故障严重程度枚举"""
    return {
        "severities": [
            {"value": fs.value, "description": fs.name.replace("_", " ").title()}
            for fs in FaultSeverity
        ]
    }

@router.get("/enums/backup-types")
async def get_backup_types():
    """获取备份类型枚举"""
    return {
        "backup_types": [
            {"value": bt.value, "description": bt.name.replace("_", " ").title()}
            for bt in BackupType
        ]
    }

@router.get("/enums/recovery-strategies")
async def get_recovery_strategies():
    """获取恢复策略枚举"""
    return {
        "recovery_strategies": [
            {"value": rs.value, "description": rs.name.replace("_", " ").title()}
            for rs in RecoveryStrategy
        ]
    }

# 高级操作端点
@router.post("/system/start")
async def start_fault_tolerance_system(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """启动容错系统"""
    try:
        await system.start()
        return {
            "status": "started",
            "timestamp": utc_now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting fault tolerance system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start system: {e}")

@router.post("/system/stop")
async def stop_fault_tolerance_system(
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
) -> Dict[str, str]:
    """停止容错系统"""
    try:
        await system.stop()
        return {
            "status": "stopped",
            "timestamp": utc_now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping fault tolerance system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop system: {e}")

# WebSocket端点用于实时监控

@router.websocket("/ws/monitoring")
async def websocket_monitoring(
    websocket: WebSocket,
    system: FaultToleranceSystem = Depends(get_fault_tolerance_system)
):
    """实时监控WebSocket端点"""
    await websocket.accept()
    
    try:
        while True:
            try:
                # 获取实时状态
                status = await system.get_system_status()
                metrics = await system.get_system_metrics()
                
                # 发送实时数据
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "data": {
                        "status": status,
                        "metrics": metrics,
                        "timestamp": utc_now().isoformat()
                    }
                }))
                
                # 等待10秒再发送下次更新
                await asyncio.sleep(10)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket monitoring error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": utc_now().isoformat()
                }))
                break
    finally:
        logger.info("WebSocket monitoring connection closed")
