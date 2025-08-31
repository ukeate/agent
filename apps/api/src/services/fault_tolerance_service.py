"""
容错和恢复系统服务层

提供容错系统的业务逻辑和服务接口
"""

from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from ai.fault_tolerance import FaultToleranceSystem, FaultType, BackupType

logger = logging.getLogger(__name__)

class FaultToleranceService:
    """容错系统服务"""
    
    def __init__(self, fault_tolerance_system: FaultToleranceSystem):
        self.fault_tolerance_system = fault_tolerance_system
        self.logger = logging.getLogger(__name__)
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """获取系统概览"""
        try:
            status = await self.fault_tolerance_system.get_system_status()
            metrics = await self.fault_tolerance_system.get_system_metrics()
            
            # 计算关键指标
            health_ratio = status["health_summary"]["health_ratio"]
            active_faults = status["health_summary"]["active_faults"]
            recovery_success_rate = status["recovery_statistics"]["success_rate"]
            consistency_rate = status["consistency_statistics"]["consistency_rate"]
            
            # 确定系统整体状态
            overall_status = self._determine_overall_status(
                health_ratio, active_faults, recovery_success_rate, consistency_rate
            )
            
            return {
                "overall_status": overall_status,
                "system_started": status["system_started"],
                "key_metrics": {
                    "health_ratio": health_ratio,
                    "active_faults": active_faults,
                    "recovery_success_rate": recovery_success_rate,
                    "consistency_rate": consistency_rate,
                    "system_availability": metrics.get("system_availability", 0.0)
                },
                "component_summary": {
                    "total_components": status["health_summary"]["total_components"],
                    "healthy_components": status["health_summary"]["status_counts"]["healthy"],
                    "degraded_components": status["health_summary"]["status_counts"]["degraded"],
                    "unhealthy_components": status["health_summary"]["status_counts"]["unhealthy"]
                },
                "recent_activity": {
                    "recent_faults": len([f for f in status["active_faults"] if self._is_recent(f["detected_at"])]),
                    "recent_recoveries": status["recovery_statistics"].get("recent_recoveries", []),
                    "recent_backups": len(status["backup_statistics"].get("components", {}))
                },
                "last_updated": status["last_updated"]
            }
        except Exception as e:
            self.logger.error(f"Error getting system overview: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def perform_health_assessment(self, component_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """执行健康评估"""
        try:
            assessment_results = {}
            
            if component_ids:
                # 评估指定组件
                for component_id in component_ids:
                    health = await self.fault_tolerance_system.get_component_health(component_id)
                    assessment_results[component_id] = self._analyze_component_health(health)
            else:
                # 评估系统整体健康
                status = await self.fault_tolerance_system.get_system_status()
                assessment_results["system"] = self._analyze_system_health(status["health_summary"])
            
            return {
                "assessment_results": assessment_results,
                "recommendations": await self._generate_health_recommendations(assessment_results),
                "assessed_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error performing health assessment: {e}")
            return {
                "error": str(e),
                "assessed_at": datetime.now().isoformat()
            }
    
    async def create_backup_plan(self, component_ids: List[str]) -> Dict[str, Any]:
        """创建备份计划"""
        try:
            backup_plan = {
                "plan_id": f"backup_plan_{int(datetime.now().timestamp())}",
                "components": component_ids,
                "backup_strategy": await self._determine_backup_strategy(component_ids),
                "estimated_duration": self._estimate_backup_duration(component_ids),
                "created_at": datetime.now().isoformat()
            }
            
            return backup_plan
        except Exception as e:
            self.logger.error(f"Error creating backup plan: {e}")
            return {"error": str(e)}
    
    async def execute_backup_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行备份计划"""
        try:
            component_ids = plan["components"]
            backup_results = await self.fault_tolerance_system.trigger_manual_backup(component_ids)
            
            execution_result = {
                "plan_id": plan["plan_id"],
                "executed_at": datetime.now().isoformat(),
                "results": backup_results,
                "success_rate": len([r for r in backup_results.values() if r]) / len(backup_results),
                "execution_summary": self._summarize_backup_execution(backup_results)
            }
            
            return execution_result
        except Exception as e:
            self.logger.error(f"Error executing backup plan: {e}")
            return {"error": str(e)}
    
    async def analyze_fault_patterns(self, days: int = 7) -> Dict[str, Any]:
        """分析故障模式"""
        try:
            # 获取最近的故障事件
            fault_events = await self.fault_tolerance_system.get_fault_events(limit=1000)
            
            # 过滤最近N天的故障
            recent_faults = [
                f for f in fault_events 
                if self._is_within_days(f["detected_at"], days)
            ]
            
            # 分析故障模式
            patterns = {
                "fault_frequency": self._analyze_fault_frequency(recent_faults),
                "fault_types_distribution": self._analyze_fault_types(recent_faults),
                "affected_components": self._analyze_affected_components(recent_faults),
                "severity_distribution": self._analyze_severity_distribution(recent_faults),
                "recovery_effectiveness": self._analyze_recovery_effectiveness(recent_faults),
                "temporal_patterns": self._analyze_temporal_patterns(recent_faults)
            }
            
            return {
                "analysis_period_days": days,
                "total_faults_analyzed": len(recent_faults),
                "patterns": patterns,
                "insights": self._generate_pattern_insights(patterns),
                "analyzed_at": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error analyzing fault patterns: {e}")
            return {"error": str(e)}
    
    async def optimize_recovery_strategies(self) -> Dict[str, Any]:
        """优化恢复策略"""
        try:
            # 获取恢复统计
            recovery_stats = self.fault_tolerance_system.recovery_manager.get_recovery_statistics()
            
            # 分析策略效果
            strategy_analysis = {}
            for strategy, success_rate in recovery_stats.get("strategy_success_rates", {}).items():
                strategy_analysis[strategy] = {
                    "current_success_rate": success_rate,
                    "optimization_potential": self._calculate_optimization_potential(success_rate),
                    "recommended_improvements": self._suggest_strategy_improvements(strategy, success_rate)
                }
            
            optimization_recommendations = {
                "strategy_analysis": strategy_analysis,
                "overall_recommendations": self._generate_optimization_recommendations(strategy_analysis),
                "implementation_priority": self._prioritize_optimizations(strategy_analysis),
                "analyzed_at": datetime.now().isoformat()
            }
            
            return optimization_recommendations
        except Exception as e:
            self.logger.error(f"Error optimizing recovery strategies: {e}")
            return {"error": str(e)}
    
    def _determine_overall_status(
        self, 
        health_ratio: float, 
        active_faults: int, 
        recovery_success_rate: float,
        consistency_rate: float
    ) -> str:
        """确定系统整体状态"""
        if health_ratio >= 0.95 and active_faults <= 2 and recovery_success_rate >= 0.95 and consistency_rate >= 0.98:
            return "excellent"
        elif health_ratio >= 0.85 and active_faults <= 5 and recovery_success_rate >= 0.90 and consistency_rate >= 0.95:
            return "good"
        elif health_ratio >= 0.70 and active_faults <= 10 and recovery_success_rate >= 0.80 and consistency_rate >= 0.90:
            return "fair"
        elif health_ratio >= 0.50 and active_faults <= 20:
            return "degraded"
        else:
            return "critical"
    
    def _is_recent(self, timestamp_str: str, hours: int = 24) -> bool:
        """检查时间戳是否在最近N小时内"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now()
            return (now - timestamp).total_seconds() < hours * 3600
        except:
            return False
    
    def _is_within_days(self, timestamp_str: str, days: int) -> bool:
        """检查时间戳是否在最近N天内"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now()
            return (now - timestamp).days < days
        except:
            return False
    
    def _analyze_component_health(self, health: Dict[str, Any]) -> Dict[str, Any]:
        """分析组件健康状态"""
        status = health.get("status", "unknown")
        response_time = health.get("response_time", 0)
        error_rate = health.get("error_rate", 0)
        resource_usage = health.get("resource_usage", {})
        
        analysis = {
            "status": status,
            "health_score": self._calculate_health_score(status, response_time, error_rate, resource_usage),
            "issues": [],
            "recommendations": []
        }
        
        # 识别问题和建议
        if response_time > 2.0:
            analysis["issues"].append("High response time")
            analysis["recommendations"].append("Investigate performance bottlenecks")
        
        if error_rate > 0.05:
            analysis["issues"].append("High error rate")
            analysis["recommendations"].append("Check error logs and fix underlying issues")
        
        cpu_usage = resource_usage.get("cpu", 0)
        memory_usage = resource_usage.get("memory", 0)
        
        if cpu_usage > 80:
            analysis["issues"].append("High CPU usage")
            analysis["recommendations"].append("Consider scaling or optimization")
        
        if memory_usage > 80:
            analysis["issues"].append("High memory usage")
            analysis["recommendations"].append("Check for memory leaks or increase resources")
        
        return analysis
    
    def _analyze_system_health(self, health_summary: Dict[str, Any]) -> Dict[str, Any]:
        """分析系统整体健康"""
        health_ratio = health_summary.get("health_ratio", 0)
        active_faults = health_summary.get("active_faults", 0)
        
        return {
            "health_ratio": health_ratio,
            "active_faults": active_faults,
            "system_health_score": min(health_ratio * 100 - active_faults * 5, 100),
            "status": "healthy" if health_ratio > 0.8 and active_faults < 5 else "degraded"
        }
    
    def _calculate_health_score(
        self, 
        status: str, 
        response_time: float, 
        error_rate: float, 
        resource_usage: Dict[str, float]
    ) -> float:
        """计算健康评分"""
        base_score = {"healthy": 100, "degraded": 70, "unhealthy": 30, "unknown": 0}.get(status, 0)
        
        # 响应时间惩罚
        if response_time > 1.0:
            base_score -= min(response_time * 10, 30)
        
        # 错误率惩罚
        base_score -= min(error_rate * 100, 40)
        
        # 资源使用惩罚
        cpu_usage = resource_usage.get("cpu", 0)
        memory_usage = resource_usage.get("memory", 0)
        
        if cpu_usage > 80:
            base_score -= (cpu_usage - 80) / 2
        if memory_usage > 80:
            base_score -= (memory_usage - 80) / 2
        
        return max(0, base_score)
    
    async def _generate_health_recommendations(self, assessment_results: Dict[str, Any]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        for component_id, analysis in assessment_results.items():
            if isinstance(analysis, dict) and "recommendations" in analysis:
                for rec in analysis["recommendations"]:
                    recommendations.append(f"{component_id}: {rec}")
        
        return recommendations
    
    async def _determine_backup_strategy(self, component_ids: List[str]) -> Dict[str, Any]:
        """确定备份策略"""
        # 简化实现，实际中应该根据组件类型和重要性确定策略
        return {
            "backup_type": BackupType.FULL_BACKUP.value,
            "parallel_backups": min(len(component_ids), 3),
            "priority_components": component_ids[:3]  # 前3个组件优先
        }
    
    def _estimate_backup_duration(self, component_ids: List[str]) -> int:
        """估算备份时长（秒）"""
        # 简化估算，每个组件大约需要30秒
        return len(component_ids) * 30
    
    def _summarize_backup_execution(self, backup_results: Dict[str, bool]) -> Dict[str, Any]:
        """总结备份执行结果"""
        success_count = sum(backup_results.values())
        total_count = len(backup_results)
        
        return {
            "total_components": total_count,
            "successful_backups": success_count,
            "failed_backups": total_count - success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "failed_components": [comp for comp, success in backup_results.items() if not success]
        }
    
    def _analyze_fault_frequency(self, faults: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析故障频率"""
        if not faults:
            return {"daily_average": 0, "trend": "stable"}
        
        # 按天分组统计
        daily_counts = {}
        for fault in faults:
            try:
                date = datetime.fromisoformat(fault["detected_at"].replace("Z", "+00:00")).date()
                daily_counts[str(date)] = daily_counts.get(str(date), 0) + 1
            except:
                continue
        
        if daily_counts:
            daily_average = sum(daily_counts.values()) / len(daily_counts)
            return {
                "daily_average": daily_average,
                "daily_counts": daily_counts,
                "trend": "increasing" if len(daily_counts) > 3 and list(daily_counts.values())[-1] > daily_average else "stable"
            }
        
        return {"daily_average": 0, "trend": "stable"}
    
    def _analyze_fault_types(self, faults: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析故障类型分布"""
        type_counts = {}
        for fault in faults:
            fault_type = fault.get("fault_type", "unknown")
            type_counts[fault_type] = type_counts.get(fault_type, 0) + 1
        return type_counts
    
    def _analyze_affected_components(self, faults: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析受影响组件"""
        component_counts = {}
        for fault in faults:
            for component in fault.get("affected_components", []):
                component_counts[component] = component_counts.get(component, 0) + 1
        return component_counts
    
    def _analyze_severity_distribution(self, faults: List[Dict[str, Any]]) -> Dict[str, int]:
        """分析严重程度分布"""
        severity_counts = {}
        for fault in faults:
            severity = fault.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts
    
    def _analyze_recovery_effectiveness(self, faults: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析恢复效果"""
        resolved_faults = [f for f in faults if f.get("resolved", False)]
        
        if not faults:
            return {"recovery_rate": 0, "avg_resolution_time": 0}
        
        recovery_rate = len(resolved_faults) / len(faults)
        
        resolution_times = []
        for fault in resolved_faults:
            try:
                detected_at = datetime.fromisoformat(fault["detected_at"].replace("Z", "+00:00"))
                resolved_at = datetime.fromisoformat(fault["resolved_at"].replace("Z", "+00:00"))
                resolution_times.append((resolved_at - detected_at).total_seconds())
            except:
                continue
        
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        return {
            "recovery_rate": recovery_rate,
            "avg_resolution_time": avg_resolution_time,
            "total_faults": len(faults),
            "resolved_faults": len(resolved_faults)
        }
    
    def _analyze_temporal_patterns(self, faults: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间模式"""
        hourly_counts = {}
        
        for fault in faults:
            try:
                dt = datetime.fromisoformat(fault["detected_at"].replace("Z", "+00:00"))
                hour = dt.hour
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            except:
                continue
        
        # 找到故障最多的时间段
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else 0
        
        return {
            "hourly_distribution": hourly_counts,
            "peak_hour": peak_hour,
            "peak_count": hourly_counts.get(peak_hour, 0)
        }
    
    def _generate_pattern_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """生成模式洞察"""
        insights = []
        
        # 故障频率洞察
        freq = patterns.get("fault_frequency", {})
        if freq.get("daily_average", 0) > 5:
            insights.append("High fault frequency detected - consider preventive measures")
        
        # 时间模式洞察
        temporal = patterns.get("temporal_patterns", {})
        peak_hour = temporal.get("peak_hour", 0)
        if peak_hour:
            insights.append(f"Most faults occur around {peak_hour}:00 - schedule maintenance accordingly")
        
        # 组件洞察
        components = patterns.get("affected_components", {})
        if components:
            most_affected = max(components.items(), key=lambda x: x[1])
            insights.append(f"Component {most_affected[0]} is most frequently affected - requires attention")
        
        return insights
    
    def _calculate_optimization_potential(self, success_rate: float) -> str:
        """计算优化潜力"""
        if success_rate >= 0.95:
            return "low"
        elif success_rate >= 0.85:
            return "medium"
        else:
            return "high"
    
    def _suggest_strategy_improvements(self, strategy: str, success_rate: float) -> List[str]:
        """建议策略改进"""
        improvements = []
        
        if success_rate < 0.8:
            improvements.append(f"Review {strategy} implementation")
            improvements.append("Add additional validation steps")
            improvements.append("Implement retry mechanisms")
        elif success_rate < 0.9:
            improvements.append(f"Fine-tune {strategy} parameters")
            improvements.append("Add monitoring and alerting")
        
        return improvements
    
    def _generate_optimization_recommendations(self, strategy_analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        low_performing_strategies = [
            strategy for strategy, analysis in strategy_analysis.items()
            if analysis.get("current_success_rate", 1.0) < 0.85
        ]
        
        if low_performing_strategies:
            recommendations.append(f"Focus on improving strategies: {', '.join(low_performing_strategies)}")
        
        recommendations.append("Implement continuous monitoring of recovery effectiveness")
        recommendations.append("Regular testing of recovery procedures")
        
        return recommendations
    
    def _prioritize_optimizations(self, strategy_analysis: Dict[str, Any]) -> List[str]:
        """优化优先级排序"""
        # 按成功率排序，最低的优先
        sorted_strategies = sorted(
            strategy_analysis.items(),
            key=lambda x: x[1].get("current_success_rate", 1.0)
        )
        
        return [strategy for strategy, _ in sorted_strategies]