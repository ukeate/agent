from src.core.utils.timezone_utils import utc_now
import asyncio
import time
from typing import Dict, List, Optional, Any
from .fault_detector import FaultDetector, FaultEvent, FaultType, FaultSeverity
from .recovery_manager import RecoveryManager
from .backup_manager import BackupManager, BackupType
from .consistency_manager import ConsistencyManager, ConsistencyCheckResult

from src.core.logging import get_logger
class FaultToleranceSystem:
    """容错和恢复系统主类"""
    
    def __init__(
        self, 
        cluster_manager, 
        task_coordinator, 
        lifecycle_manager,
        metrics_collector,
        config: Dict[str, Any]
    ):
        self.cluster_manager = cluster_manager
        self.task_coordinator = task_coordinator
        self.lifecycle_manager = lifecycle_manager
        self.metrics_collector = metrics_collector
        self.config = config
        self.logger = get_logger(__name__)
        
        # 初始化核心组件
        self.fault_detector = FaultDetector(
            cluster_manager, 
            metrics_collector, 
            config.get("fault_detection", {})
        )
        
        self.recovery_manager = RecoveryManager(
            cluster_manager,
            task_coordinator,
            lifecycle_manager,
            config.get("recovery", {})
        )
        
        self.backup_manager = BackupManager(
            storage_backend=None,  # 需要实际存储后端
            config=config.get("backup", {})
        )
        
        self.consistency_manager = ConsistencyManager(
            cluster_manager,
            storage_backend=None,  # 需要实际存储后端
            config=config.get("consistency", {})
        )
        
        # 连接故障检测和恢复管理
        self.fault_detector.register_fault_callback(
            self.recovery_manager.handle_fault_event
        )
        
        # 启动标志
        self.started = False
    
    async def start(self):
        """启动容错系统"""
        
        if self.started:
            return
        
        try:
            self.logger.info("Starting fault tolerance system...")
            
            # 启动各个组件
            await self.fault_detector.start()
            await self.recovery_manager.start()
            await self.backup_manager.start()
            await self.consistency_manager.start()
            
            self.started = True
            self.logger.info("Fault tolerance system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start fault tolerance system: {e}")
            raise
    
    async def stop(self):
        """停止容错系统"""
        
        if not self.started:
            return
        
        try:
            self.logger.info("Stopping fault tolerance system...")
            
            # 停止各个组件
            await self.consistency_manager.stop()
            await self.backup_manager.stop()
            await self.recovery_manager.stop()
            await self.fault_detector.stop()
            
            self.started = False
            self.logger.info("Fault tolerance system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping fault tolerance system: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        
        try:
            # 获取各组件状态
            health_summary = self.fault_detector.get_system_health_summary()
            recovery_stats = self.recovery_manager.get_recovery_statistics()
            backup_stats = self.backup_manager.get_backup_statistics()
            consistency_stats = self.consistency_manager.get_consistency_statistics()
            
            # 获取活跃故障
            active_faults = self.fault_detector.get_fault_events(resolved=False, limit=50)
            
            return {
                "system_started": self.started,
                "health_summary": health_summary,
                "recovery_statistics": recovery_stats,
                "backup_statistics": backup_stats,
                "consistency_statistics": consistency_stats,
                "active_faults": [
                    {
                        "fault_id": fault.fault_id,
                        "fault_type": fault.fault_type.value,
                        "severity": fault.severity.value,
                        "affected_components": fault.affected_components,
                        "detected_at": fault.detected_at.isoformat(),
                        "description": fault.description
                    }
                    for fault in active_faults
                ],
                "last_updated": utc_now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "system_started": self.started,
                "error": str(e),
                "last_updated": utc_now().isoformat()
            }
    
    async def trigger_manual_backup(self, component_ids: List[str]) -> Dict[str, bool]:
        """触发手动备份"""
        
        results = {}
        
        for component_id in component_ids:
            try:
                backup_record = await self.backup_manager.create_backup(component_id)
                results[component_id] = backup_record is not None
                
                if backup_record:
                    self.logger.info(f"Manual backup created for {component_id}: {backup_record.backup_id}")
                else:
                    self.logger.warning(f"Manual backup failed for {component_id}")
                    
            except Exception as e:
                self.logger.error(f"Manual backup failed for {component_id}: {e}")
                results[component_id] = False
        
        return results
    
    async def trigger_manual_consistency_check(self, data_keys: List[str]) -> ConsistencyCheckResult:
        """触发手动一致性检查"""
        
        try:
            result = await self.consistency_manager.check_data_consistency(data_keys)
            self.logger.info(f"Manual consistency check completed: {result.check_id}, consistent: {result.consistent}")
            return result
        except Exception as e:
            self.logger.error(f"Manual consistency check failed: {e}")
            raise
    
    async def simulate_fault_injection(
        self, 
        component_id: str, 
        fault_type: FaultType,
        duration_seconds: int = 60
    ):
        """模拟故障注入（用于测试）"""
        
        self.logger.warning(f"Injecting fault {fault_type.value} for {component_id} (duration: {duration_seconds}s)")
        
        # 创建模拟故障事件
        fault_event = FaultEvent(
            fault_id=f"injected_{fault_type.value}_{int(time.time())}",
            fault_type=fault_type,
            severity=FaultSeverity.HIGH,
            affected_components=[component_id],
            detected_at=utc_now(),
            description=f"Injected fault for testing: {fault_type.value}",
            context={"injected": True, "duration": duration_seconds}
        )
        
        # 触发故障处理
        await self.recovery_manager.handle_fault_event(fault_event)
        
        # 设置自动恢复
        async def auto_resolve():
            await asyncio.sleep(duration_seconds)
            fault_event.resolved = True
            fault_event.resolved_at = utc_now()
            self.logger.info(f"Injected fault {fault_event.fault_id} auto-resolved")
        
        asyncio.create_task(auto_resolve())
        
        return fault_event.fault_id
    
    async def get_component_health(self, component_id: str) -> Dict[str, Any]:
        """获取特定组件的健康状态"""
        
        try:
            health_status = await self.fault_detector.check_component_health(component_id)
            return {
                "component_id": health_status.component_id,
                "status": health_status.status,
                "last_check": health_status.last_check.isoformat(),
                "response_time": health_status.response_time,
                "error_rate": health_status.error_rate,
                "resource_usage": health_status.resource_usage,
                "custom_metrics": health_status.custom_metrics
            }
        except Exception as e:
            self.logger.error(f"Error getting component health for {component_id}: {e}")
            return {
                "component_id": component_id,
                "status": "error",
                "error": str(e),
                "last_check": utc_now().isoformat()
            }
    
    async def get_fault_events(
        self,
        fault_type: Optional[str] = None,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取故障事件"""
        
        try:
            # 转换字符串参数为枚举
            fault_type_enum = None
            if fault_type:
                fault_type_enum = FaultType(fault_type)
            
            severity_enum = None
            if severity:
                severity_enum = FaultSeverity(severity)
            
            fault_events = self.fault_detector.get_fault_events(
                fault_type=fault_type_enum,
                severity=severity_enum,
                resolved=resolved,
                limit=limit
            )
            
            return [
                {
                    "fault_id": event.fault_id,
                    "fault_type": event.fault_type.value,
                    "severity": event.severity.value,
                    "affected_components": event.affected_components,
                    "detected_at": event.detected_at.isoformat(),
                    "description": event.description,
                    "resolved": event.resolved,
                    "resolved_at": event.resolved_at.isoformat() if event.resolved_at else None,
                    "recovery_actions": event.recovery_actions,
                    "context": event.context
                }
                for event in fault_events
            ]
        except Exception as e:
            self.logger.error(f"Error getting fault events: {e}")
            return []
    
    async def restore_backup(self, backup_id: str, target_component_id: Optional[str] = None) -> bool:
        """恢复备份"""
        
        try:
            success = await self.backup_manager.restore_backup(backup_id, target_component_id)
            
            if success:
                self.logger.info(f"Backup restored successfully: {backup_id}")
            else:
                self.logger.error(f"Backup restoration failed: {backup_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Backup restoration failed for {backup_id}: {e}")
            return False
    
    async def repair_consistency_issues(self, check_id: str) -> bool:
        """修复一致性问题"""
        
        try:
            # 查找检查结果
            check_result = self.consistency_manager.get_check_result_by_id(check_id)
            
            if not check_result:
                self.logger.error(f"Consistency check not found: {check_id}")
                return False
            
            success = await self.consistency_manager.repair_inconsistencies(check_result)
            
            if success:
                self.logger.info(f"Consistency issues repaired for check: {check_id}")
            else:
                self.logger.warning(f"Some consistency repairs failed for check: {check_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Consistency repair failed for {check_id}: {e}")
            return False
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        
        try:
            status = await self.get_system_status()
            
            return {
                "fault_detection_metrics": {
                    "total_components": status["health_summary"]["total_components"],
                    "healthy_components": status["health_summary"]["status_counts"]["healthy"],
                    "health_ratio": status["health_summary"]["health_ratio"],
                    "avg_response_time": status["health_summary"]["avg_response_time"],
                    "active_faults": status["health_summary"]["active_faults"]
                },
                "recovery_metrics": {
                    "total_recoveries": status["recovery_statistics"]["total_recoveries"],
                    "success_rate": status["recovery_statistics"]["success_rate"],
                    "avg_recovery_time": status["recovery_statistics"]["avg_recovery_time"]
                },
                "backup_metrics": {
                    "total_backups": status["backup_statistics"]["total_backups"],
                    "valid_backups": status["backup_statistics"]["valid_backups"],
                    "total_size": status["backup_statistics"]["total_size"]
                },
                "consistency_metrics": {
                    "total_checks": status["consistency_statistics"]["total_checks"],
                    "consistency_rate": status["consistency_statistics"]["consistency_rate"]
                },
                "system_availability": self._calculate_system_availability(),
                "last_updated": utc_now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e), "last_updated": utc_now().isoformat()}
    
    def _calculate_system_availability(self) -> float:
        """计算系统可用性"""
        
        try:
            # 基于健康状态计算系统可用性
            health_summary = self.fault_detector.get_system_health_summary()
            
            if health_summary["total_components"] == 0:
                return 1.0
            
            # 简化的可用性计算：基于健康组件的比例
            availability = health_summary["health_ratio"]
            
            # 如果有活跃故障，降低可用性评分
            active_faults = health_summary["active_faults"]
            if active_faults > 0:
                fault_penalty = min(0.1 * active_faults, 0.3)  # 最多降低30%
                availability = max(0.0, availability - fault_penalty)
            
            return availability
            
        except Exception as e:
            self.logger.error(f"Error calculating system availability: {e}")
            return 0.0
    
    async def validate_all_backups(self) -> Dict[str, bool]:
        """验证所有备份"""
        
        try:
            validation_results = await self.backup_manager.validate_all_backups()
            
            valid_count = sum(1 for is_valid in validation_results.values() if is_valid)
            total_count = len(validation_results)
            
            self.logger.info(f"Backup validation completed: {valid_count}/{total_count} backups valid")
            
            return validation_results
        except Exception as e:
            self.logger.error(f"Backup validation failed: {e}")
            return {}
    
    async def force_consistency_repair(
        self, 
        data_key: str, 
        authoritative_component_id: str
    ) -> bool:
        """强制一致性修复"""
        
        try:
            success = await self.consistency_manager.force_consistency_repair(
                data_key, 
                authoritative_component_id
            )
            
            if success:
                self.logger.info(f"Force consistency repair completed for {data_key}")
            else:
                self.logger.error(f"Force consistency repair failed for {data_key}")
            
            return success
        except Exception as e:
            self.logger.error(f"Force consistency repair failed: {e}")
            return False
    
    async def get_detailed_system_report(self) -> Dict[str, Any]:
        """获取详细的系统报告"""
        
        try:
            status = await self.get_system_status()
            metrics = await self.get_system_metrics()
            
            # 获取最近的故障事件
            recent_faults = await self.get_fault_events(limit=20)
            
            # 获取备份记录
            backup_records = self.backup_manager.get_backup_records()
            recent_backups = [
                {
                    "backup_id": record.backup_id,
                    "component_id": record.component_id,
                    "backup_type": record.backup_type.value,
                    "created_at": record.created_at.isoformat(),
                    "size": record.size,
                    "valid": record.valid
                }
                for record in sorted(backup_records, key=lambda x: x.created_at, reverse=True)[:10]
            ]
            
            return {
                "report_generated_at": utc_now().isoformat(),
                "system_status": status,
                "system_metrics": metrics,
                "recent_faults": recent_faults,
                "recent_backups": recent_backups,
                "recommendations": await self._generate_recommendations(status, metrics)
            }
        except Exception as e:
            self.logger.error(f"Error generating system report: {e}")
            return {
                "report_generated_at": utc_now().isoformat(),
                "error": str(e)
            }
    
    async def _generate_recommendations(
        self, 
        status: Dict[str, Any], 
        metrics: Dict[str, Any]
    ) -> List[str]:
        """生成系统建议"""
        
        recommendations = []
        
        try:
            # 基于健康状态生成建议
            health_ratio = status["health_summary"]["health_ratio"]
            if health_ratio < 0.8:
                recommendations.append(f"System health is degraded ({health_ratio:.1%}). Consider investigating unhealthy components.")
            
            # 基于活跃故障生成建议
            active_faults = status["health_summary"]["active_faults"]
            if active_faults > 5:
                recommendations.append(f"High number of active faults ({active_faults}). Review fault resolution strategies.")
            
            # 基于恢复成功率生成建议
            recovery_success_rate = status["recovery_statistics"]["success_rate"]
            if recovery_success_rate < 0.9:
                recommendations.append(f"Recovery success rate is low ({recovery_success_rate:.1%}). Review recovery strategies.")
            
            # 基于一致性检查生成建议
            consistency_rate = status["consistency_statistics"]["consistency_rate"]
            if consistency_rate < 0.95:
                recommendations.append(f"Data consistency rate is low ({consistency_rate:.1%}). Consider more frequent consistency checks.")
            
            # 基于备份状态生成建议
            total_backups = status["backup_statistics"]["total_backups"]
            valid_backups = status["backup_statistics"]["valid_backups"]
            if total_backups > 0 and (valid_backups / total_backups) < 0.95:
                recommendations.append("Some backups are invalid. Run backup validation and cleanup.")
            
            if len(recommendations) == 0:
                recommendations.append("System is operating normally. No immediate actions required.")
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to system error.")
        
        return recommendations
