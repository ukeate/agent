"""
风险评估和回滚服务

评估实验风险并提供自动回滚能力
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass, field
import statistics

from ..core.database import async_session_manager
from ..services.realtime_metrics_service import RealtimeMetricsService
from ..services.anomaly_detection_service import AnomalyDetectionService, AnomalyType
from ..services.alert_rules_service import AlertRulesEngine, AlertSeverity


class RiskLevel(str, Enum):
    """风险等级"""
    MINIMAL = "minimal"  # 极低风险
    LOW = "low"  # 低风险
    MEDIUM = "medium"  # 中等风险
    HIGH = "high"  # 高风险
    CRITICAL = "critical"  # 严重风险


class RiskCategory(str, Enum):
    """风险类别"""
    PERFORMANCE = "performance"  # 性能风险
    BUSINESS = "business"  # 业务风险
    TECHNICAL = "technical"  # 技术风险
    USER_EXPERIENCE = "user_experience"  # 用户体验风险
    DATA_QUALITY = "data_quality"  # 数据质量风险
    COMPLIANCE = "compliance"  # 合规风险


class RollbackStrategy(str, Enum):
    """回滚策略"""
    IMMEDIATE = "immediate"  # 立即回滚
    GRADUAL = "gradual"  # 渐进回滚
    PARTIAL = "partial"  # 部分回滚
    MANUAL = "manual"  # 手动确认


@dataclass
class RiskFactor:
    """风险因素"""
    category: RiskCategory
    name: str
    description: str
    severity: float  # 0-1
    likelihood: float  # 0-1
    impact: float  # 0-1
    current_value: Any
    threshold_value: Any
    mitigation: str
    
    @property
    def risk_score(self) -> float:
        """计算风险分数"""
        return self.severity * self.likelihood * self.impact


@dataclass
class RiskAssessment:
    """风险评估结果"""
    experiment_id: str
    assessment_time: datetime
    overall_risk_level: RiskLevel
    overall_risk_score: float
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    requires_rollback: bool
    rollback_strategy: Optional[RollbackStrategy]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackPlan:
    """回滚计划"""
    experiment_id: str
    trigger_reason: str
    strategy: RollbackStrategy
    target_state: Dict[str, Any]
    steps: List[Dict[str, Any]]
    estimated_duration_minutes: int
    auto_execute: bool
    approval_required: bool
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RollbackExecution:
    """回滚执行记录"""
    plan_id: str
    experiment_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, completed, failed
    steps_completed: int = 0
    total_steps: int = 0
    errors: List[str] = field(default_factory=list)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)


class RiskAssessmentService:
    """风险评估服务"""
    
    def __init__(self):
        self.assessments: Dict[str, List[RiskAssessment]] = {}
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.rollback_executions: Dict[str, RollbackExecution] = {}
        self.risk_thresholds = self._initialize_risk_thresholds()
        self.metrics_service = RealtimeMetricsService()
        self.anomaly_service = AnomalyDetectionService()
        self.alert_engine = AlertRulesEngine()
        
    def _initialize_risk_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """初始化风险阈值"""
        return {
            "performance": {
                "latency_increase": 1.5,  # 延迟增加50%
                "error_rate": 0.05,  # 错误率5%
                "timeout_rate": 0.01  # 超时率1%
            },
            "business": {
                "conversion_drop": 0.1,  # 转化率下降10%
                "revenue_impact": 0.05,  # 收入影响5%
                "user_retention_drop": 0.15  # 留存下降15%
            },
            "technical": {
                "cpu_usage": 0.8,  # CPU使用率80%
                "memory_usage": 0.9,  # 内存使用率90%
                "db_connections": 0.95  # 数据库连接池95%
            },
            "user_experience": {
                "bounce_rate_increase": 0.2,  # 跳出率增加20%
                "page_load_time": 3.0,  # 页面加载时间3秒
                "crash_rate": 0.001  # 崩溃率0.1%
            }
        }
        
    async def assess_risk(
        self,
        experiment_id: str,
        include_predictions: bool = True
    ) -> RiskAssessment:
        """
        评估实验风险
        
        Args:
            experiment_id: 实验ID
            include_predictions: 是否包含预测性分析
            
        Returns:
            风险评估结果
        """
        # 收集风险因素
        risk_factors = []
        
        # 评估性能风险
        perf_factors = await self._assess_performance_risk(experiment_id)
        risk_factors.extend(perf_factors)
        
        # 评估业务风险
        business_factors = await self._assess_business_risk(experiment_id)
        risk_factors.extend(business_factors)
        
        # 评估技术风险
        tech_factors = await self._assess_technical_risk(experiment_id)
        risk_factors.extend(tech_factors)
        
        # 评估用户体验风险
        ux_factors = await self._assess_user_experience_risk(experiment_id)
        risk_factors.extend(ux_factors)
        
        # 评估数据质量风险
        data_factors = await self._assess_data_quality_risk(experiment_id)
        risk_factors.extend(data_factors)
        
        # 如果需要，进行预测性分析
        if include_predictions:
            predicted_factors = await self._predict_future_risks(experiment_id)
            risk_factors.extend(predicted_factors)
            
        # 计算总体风险
        overall_score = self._calculate_overall_risk_score(risk_factors)
        overall_level = self._determine_risk_level(overall_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(risk_factors, overall_level)
        
        # 判断是否需要回滚
        requires_rollback = overall_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        rollback_strategy = self._determine_rollback_strategy(overall_level, risk_factors)
        
        assessment = RiskAssessment(
            experiment_id=experiment_id,
            assessment_time=datetime.utcnow(),
            overall_risk_level=overall_level,
            overall_risk_score=overall_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            requires_rollback=requires_rollback,
            rollback_strategy=rollback_strategy,
            confidence=self._calculate_confidence(risk_factors)
        )
        
        # 保存评估结果
        if experiment_id not in self.assessments:
            self.assessments[experiment_id] = []
        self.assessments[experiment_id].append(assessment)
        
        return assessment
        
    async def _assess_performance_risk(self, experiment_id: str) -> List[RiskFactor]:
        """评估性能风险"""
        factors = []
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查延迟
        latency = metrics.get("latency", {})
        baseline = latency.get("baseline", 100)
        current = latency.get("current", 100)
        
        if current > baseline * self.risk_thresholds["performance"]["latency_increase"]:
            factors.append(RiskFactor(
                category=RiskCategory.PERFORMANCE,
                name="延迟增加",
                description=f"延迟从{baseline}ms增加到{current}ms",
                severity=0.7,
                likelihood=0.9,
                impact=0.8,
                current_value=current,
                threshold_value=baseline * 1.5,
                mitigation="优化代码性能或增加资源"
            ))
            
        # 检查错误率
        error_rate = metrics.get("error_rate", {}).get("value", 0)
        if error_rate > self.risk_thresholds["performance"]["error_rate"]:
            factors.append(RiskFactor(
                category=RiskCategory.PERFORMANCE,
                name="错误率过高",
                description=f"错误率达到{error_rate:.2%}",
                severity=0.9,
                likelihood=1.0,
                impact=0.9,
                current_value=error_rate,
                threshold_value=0.05,
                mitigation="排查错误原因并修复"
            ))
            
        return factors
        
    async def _assess_business_risk(self, experiment_id: str) -> List[RiskFactor]:
        """评估业务风险"""
        factors = []
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查转化率
        conversion = metrics.get("conversion_rate", {})
        baseline = conversion.get("baseline", 0.1)
        current = conversion.get("current", 0.1)
        
        drop = (baseline - current) / baseline if baseline > 0 else 0
        if drop > self.risk_thresholds["business"]["conversion_drop"]:
            factors.append(RiskFactor(
                category=RiskCategory.BUSINESS,
                name="转化率下降",
                description=f"转化率下降{drop:.1%}",
                severity=0.8,
                likelihood=0.8,
                impact=1.0,
                current_value=current,
                threshold_value=baseline * 0.9,
                mitigation="分析用户行为，优化转化漏斗"
            ))
            
        return factors
        
    async def _assess_technical_risk(self, experiment_id: str) -> List[RiskFactor]:
        """评估技术风险"""
        factors = []
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查资源使用
        cpu_usage = metrics.get("cpu_usage", {}).get("value", 0)
        if cpu_usage > self.risk_thresholds["technical"]["cpu_usage"]:
            factors.append(RiskFactor(
                category=RiskCategory.TECHNICAL,
                name="CPU使用率过高",
                description=f"CPU使用率达到{cpu_usage:.0%}",
                severity=0.6,
                likelihood=0.7,
                impact=0.7,
                current_value=cpu_usage,
                threshold_value=0.8,
                mitigation="优化算法或扩容"
            ))
            
        return factors
        
    async def _assess_user_experience_risk(self, experiment_id: str) -> List[RiskFactor]:
        """评估用户体验风险"""
        factors = []
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 检查页面加载时间
        load_time = metrics.get("page_load_time", {}).get("value", 0)
        if load_time > self.risk_thresholds["user_experience"]["page_load_time"]:
            factors.append(RiskFactor(
                category=RiskCategory.USER_EXPERIENCE,
                name="页面加载缓慢",
                description=f"页面加载时间{load_time:.1f}秒",
                severity=0.7,
                likelihood=0.9,
                impact=0.8,
                current_value=load_time,
                threshold_value=3.0,
                mitigation="优化前端资源和加载策略"
            ))
            
        return factors
        
    async def _assess_data_quality_risk(self, experiment_id: str) -> List[RiskFactor]:
        """评估数据质量风险"""
        factors = []
        
        # 检查异常
        anomalies = await self.anomaly_service.detect_sample_ratio_mismatch(
            experiment_id=experiment_id,
            control_count=5000,
            treatment_count=4500,
            expected_ratio=0.5
        )
        
        if anomalies:
            factors.append(RiskFactor(
                category=RiskCategory.DATA_QUALITY,
                name="样本比例不匹配",
                description="检测到SRM问题",
                severity=1.0,
                likelihood=1.0,
                impact=0.9,
                current_value="SRM detected",
                threshold_value="No SRM",
                mitigation="检查分流逻辑"
            ))
            
        return factors
        
    async def _predict_future_risks(self, experiment_id: str) -> List[RiskFactor]:
        """预测未来风险"""
        factors = []
        
        # 基于趋势预测
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 简化示例：如果错误率在上升
        error_trend = metrics.get("error_rate", {}).get("trend", "stable")
        if error_trend == "increasing":
            factors.append(RiskFactor(
                category=RiskCategory.TECHNICAL,
                name="错误率上升趋势",
                description="预测未来24小时错误率将继续上升",
                severity=0.6,
                likelihood=0.7,
                impact=0.8,
                current_value="increasing",
                threshold_value="stable",
                mitigation="提前介入，防止恶化"
            ))
            
        return factors
        
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """计算总体风险分数"""
        if not risk_factors:
            return 0.0
            
        # 使用加权平均
        category_weights = {
            RiskCategory.PERFORMANCE: 1.2,
            RiskCategory.BUSINESS: 1.5,
            RiskCategory.TECHNICAL: 1.0,
            RiskCategory.USER_EXPERIENCE: 1.3,
            RiskCategory.DATA_QUALITY: 1.4,
            RiskCategory.COMPLIANCE: 2.0
        }
        
        weighted_scores = []
        for factor in risk_factors:
            weight = category_weights.get(factor.category, 1.0)
            weighted_scores.append(factor.risk_score * weight)
            
        # 返回最高风险分数（保守策略）
        return max(weighted_scores) if weighted_scores else 0.0
        
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """确定风险等级"""
        if risk_score < 0.2:
            return RiskLevel.MINIMAL
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
            
    def _generate_recommendations(
        self,
        risk_factors: List[RiskFactor],
        risk_level: RiskLevel
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("立即停止实验并回滚")
            recommendations.append("召集团队进行紧急评审")
            
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("考虑暂停实验或减少流量")
            recommendations.append("密切监控关键指标")
            
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("增加监控频率")
            recommendations.append("准备回滚计划")
            
        # 基于具体风险因素的建议
        for factor in risk_factors:
            if factor.risk_score > 0.7:
                recommendations.append(factor.mitigation)
                
        return list(set(recommendations))[:5]  # 返回前5个建议
        
    def _determine_rollback_strategy(
        self,
        risk_level: RiskLevel,
        risk_factors: List[RiskFactor]
    ) -> Optional[RollbackStrategy]:
        """确定回滚策略"""
        if risk_level == RiskLevel.CRITICAL:
            return RollbackStrategy.IMMEDIATE
        elif risk_level == RiskLevel.HIGH:
            # 检查是否有数据质量问题
            has_data_issues = any(
                f.category == RiskCategory.DATA_QUALITY 
                for f in risk_factors
            )
            if has_data_issues:
                return RollbackStrategy.IMMEDIATE
            else:
                return RollbackStrategy.GRADUAL
        elif risk_level == RiskLevel.MEDIUM:
            return RollbackStrategy.MANUAL
        else:
            return None
            
    def _calculate_confidence(self, risk_factors: List[RiskFactor]) -> float:
        """计算评估置信度"""
        if not risk_factors:
            return 0.5
            
        # 基于风险因素数量和严重性计算置信度
        factor_count = len(risk_factors)
        avg_severity = statistics.mean(f.severity for f in risk_factors)
        
        # 更多因素和更高严重性意味着更高置信度
        confidence = min(0.5 + (factor_count * 0.05) + (avg_severity * 0.3), 1.0)
        
        return confidence
        
    async def create_rollback_plan(
        self,
        experiment_id: str,
        assessment: RiskAssessment
    ) -> RollbackPlan:
        """
        创建回滚计划
        
        Args:
            experiment_id: 实验ID
            assessment: 风险评估结果
            
        Returns:
            回滚计划
        """
        # 确定目标状态
        target_state = {
            "traffic_percentage": 0,
            "status": "rolled_back",
            "variant_enabled": False
        }
        
        # 生成回滚步骤
        steps = []
        
        if assessment.rollback_strategy == RollbackStrategy.IMMEDIATE:
            steps = [
                {"step": 1, "action": "停止新流量", "duration_minutes": 1},
                {"step": 2, "action": "切换到对照组", "duration_minutes": 1},
                {"step": 3, "action": "清理缓存", "duration_minutes": 2},
                {"step": 4, "action": "验证回滚", "duration_minutes": 1}
            ]
            estimated_duration = 5
            auto_execute = True
            
        elif assessment.rollback_strategy == RollbackStrategy.GRADUAL:
            steps = [
                {"step": 1, "action": "减少流量到50%", "duration_minutes": 10},
                {"step": 2, "action": "监控指标", "duration_minutes": 30},
                {"step": 3, "action": "减少流量到20%", "duration_minutes": 10},
                {"step": 4, "action": "减少流量到5%", "duration_minutes": 10},
                {"step": 5, "action": "完全停止", "duration_minutes": 5}
            ]
            estimated_duration = 65
            auto_execute = False
            
        else:  # MANUAL
            steps = [
                {"step": 1, "action": "发送告警", "duration_minutes": 1},
                {"step": 2, "action": "等待确认", "duration_minutes": 0},
                {"step": 3, "action": "执行回滚", "duration_minutes": 5}
            ]
            estimated_duration = 6
            auto_execute = False
            
        plan = RollbackPlan(
            experiment_id=experiment_id,
            trigger_reason=f"风险等级: {assessment.overall_risk_level}",
            strategy=assessment.rollback_strategy or RollbackStrategy.MANUAL,
            target_state=target_state,
            steps=steps,
            estimated_duration_minutes=estimated_duration,
            auto_execute=auto_execute,
            approval_required=not auto_execute
        )
        
        plan_id = f"rollback_{experiment_id}_{datetime.utcnow().timestamp()}"
        self.rollback_plans[plan_id] = plan
        
        return plan
        
    async def execute_rollback(
        self,
        plan_id: str,
        force: bool = False
    ) -> RollbackExecution:
        """
        执行回滚
        
        Args:
            plan_id: 回滚计划ID
            force: 是否强制执行
            
        Returns:
            回滚执行记录
        """
        if plan_id not in self.rollback_plans:
            raise ValueError(f"回滚计划 {plan_id} 不存在")
            
        plan = self.rollback_plans[plan_id]
        
        # 检查是否需要批准
        if plan.approval_required and not force:
            raise PermissionError("需要批准才能执行回滚")
            
        # 获取回滚前的指标
        metrics_before = await self.metrics_service.get_experiment_metrics(
            plan.experiment_id
        )
        
        # 创建执行记录
        execution = RollbackExecution(
            plan_id=plan_id,
            experiment_id=plan.experiment_id,
            started_at=datetime.utcnow(),
            total_steps=len(plan.steps),
            metrics_before=metrics_before
        )
        
        exec_id = f"exec_{plan_id}_{datetime.utcnow().timestamp()}"
        self.rollback_executions[exec_id] = execution
        
        # 执行回滚步骤
        for step in plan.steps:
            try:
                await self._execute_rollback_step(step, plan.experiment_id)
                execution.steps_completed += 1
                
                # 等待步骤完成
                await asyncio.sleep(step["duration_minutes"] * 60)
                
            except Exception as e:
                execution.errors.append(f"步骤{step['step']}失败: {str(e)}")
                execution.status = "failed"
                break
                
        # 获取回滚后的指标
        if execution.status != "failed":
            execution.metrics_after = await self.metrics_service.get_experiment_metrics(
                plan.experiment_id
            )
            execution.status = "completed"
            execution.completed_at = datetime.utcnow()
            
        return execution
        
    async def _execute_rollback_step(self, step: Dict[str, Any], experiment_id: str):
        """执行回滚步骤"""
        action = step["action"]
        
        if "停止新流量" in action:
            # 停止新用户进入实验
            print(f"停止实验 {experiment_id} 的新流量")
            
        elif "切换到对照组" in action:
            # 将所有用户切换到对照组
            print(f"将实验 {experiment_id} 的用户切换到对照组")
            
        elif "减少流量" in action:
            # 提取百分比并调整流量
            import re
            match = re.search(r"(\d+)%", action)
            if match:
                percentage = int(match.group(1))
                print(f"将实验 {experiment_id} 的流量减少到 {percentage}%")
                
        elif "清理缓存" in action:
            # 清理相关缓存
            print(f"清理实验 {experiment_id} 的缓存")
            
        elif "验证回滚" in action:
            # 验证回滚是否成功
            print(f"验证实验 {experiment_id} 的回滚状态")
            
    async def monitor_risk_continuously(
        self,
        experiment_id: str,
        check_interval_minutes: int = 5
    ):
        """
        持续监控风险
        
        Args:
            experiment_id: 实验ID
            check_interval_minutes: 检查间隔
        """
        while True:
            try:
                # 评估风险
                assessment = await self.assess_risk(experiment_id)
                
                # 如果需要回滚
                if assessment.requires_rollback:
                    # 创建回滚计划
                    plan = await self.create_rollback_plan(experiment_id, assessment)
                    
                    # 如果是自动执行
                    if plan.auto_execute:
                        # 获取计划ID
                        plan_id = None
                        for pid, p in self.rollback_plans.items():
                            if p == plan:
                                plan_id = pid
                                break
                                
                        if plan_id:
                            await self.execute_rollback(plan_id, force=True)
                            break  # 回滚后停止监控
                            
                # 等待下一次检查
                await asyncio.sleep(check_interval_minutes * 60)
                
            except Exception as e:
                print(f"风险监控错误: {e}")
                await asyncio.sleep(60)  # 错误后等待1分钟