"""
自动扩量规则引擎服务

基于实验指标和规则自动调整流量
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass, field
import statistics

from ..core.database import async_session_manager
from ..services.realtime_metrics_service import RealtimeMetricsService
from ..services.anomaly_detection_service import AnomalyDetectionService
from ..services.statistical_analysis_service import StatisticalAnalysisService
from ..services.hypothesis_testing_service import HypothesisTestingService


class ScalingDirection(str, Enum):
    """扩量方向"""
    UP = "up"  # 增加流量
    DOWN = "down"  # 减少流量
    HOLD = "hold"  # 保持不变


class ScalingTrigger(str, Enum):
    """扩量触发器"""
    METRIC_THRESHOLD = "metric_threshold"  # 指标阈值
    STATISTICAL_SIGNIFICANCE = "statistical_significance"  # 统计显著性
    TIME_BASED = "time_based"  # 基于时间
    SAMPLE_SIZE = "sample_size"  # 样本量达标
    CONFIDENCE_INTERVAL = "confidence_interval"  # 置信区间
    CUSTOM_RULE = "custom_rule"  # 自定义规则


class ScalingMode(str, Enum):
    """扩量模式"""
    AGGRESSIVE = "aggressive"  # 激进模式
    CONSERVATIVE = "conservative"  # 保守模式
    BALANCED = "balanced"  # 平衡模式
    ADAPTIVE = "adaptive"  # 自适应模式


@dataclass
class ScalingCondition:
    """扩量条件"""
    trigger: ScalingTrigger
    metric_name: Optional[str]
    operator: str  # >, <, ==, !=, >=, <=
    threshold: float
    confidence_level: float = 0.95
    min_sample_size: int = 1000
    evaluation_window_minutes: int = 30
    
    def evaluate(self, metrics: Dict[str, Any], stats: Dict[str, Any]) -> bool:
        """评估条件是否满足"""
        if self.trigger == ScalingTrigger.METRIC_THRESHOLD:
            if self.metric_name and self.metric_name in metrics:
                value = metrics[self.metric_name].get("value", 0)
                return self._compare(value, self.threshold)
                
        elif self.trigger == ScalingTrigger.STATISTICAL_SIGNIFICANCE:
            p_value = stats.get("p_value", 1.0)
            return p_value < (1 - self.confidence_level)
            
        elif self.trigger == ScalingTrigger.SAMPLE_SIZE:
            sample_size = stats.get("sample_size", 0)
            return sample_size >= self.min_sample_size
            
        elif self.trigger == ScalingTrigger.CONFIDENCE_INTERVAL:
            ci_lower = stats.get("ci_lower", 0)
            ci_upper = stats.get("ci_upper", 0)
            # 检查置信区间是否不包含0（表示有显著效果）
            return ci_lower > 0 or ci_upper < 0
            
        return False
        
    def _compare(self, value: float, threshold: float) -> bool:
        """比较操作"""
        if self.operator == ">":
            return value > threshold
        elif self.operator == "<":
            return value < threshold
        elif self.operator == ">=":
            return value >= threshold
        elif self.operator == "<=":
            return value <= threshold
        elif self.operator == "==":
            return value == threshold
        elif self.operator == "!=":
            return value != threshold
        return False


@dataclass
class ScalingRule:
    """扩量规则"""
    id: str
    name: str
    description: str
    experiment_id: str
    variant: str
    mode: ScalingMode
    conditions: List[ScalingCondition]
    scale_up_conditions: List[ScalingCondition]
    scale_down_conditions: List[ScalingCondition]
    scale_increment: float = 10.0  # 每次扩量百分比
    scale_decrement: float = 5.0  # 每次缩量百分比
    min_percentage: float = 1.0  # 最小流量百分比
    max_percentage: float = 100.0  # 最大流量百分比
    cooldown_minutes: int = 30  # 冷却时间
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScalingDecision:
    """扩量决策"""
    rule_id: str
    experiment_id: str
    timestamp: datetime
    direction: ScalingDirection
    current_percentage: float
    target_percentage: float
    reason: str
    confidence: float
    metrics_snapshot: Dict[str, Any]
    applied: bool = False


@dataclass
class ScalingHistory:
    """扩量历史"""
    experiment_id: str
    decisions: List[ScalingDecision] = field(default_factory=list)
    last_scaled_at: Optional[datetime] = None
    total_scale_ups: int = 0
    total_scale_downs: int = 0
    current_percentage: float = 50.0


class AutoScalingService:
    """自动扩量服务"""
    
    def __init__(self):
        self.rules: Dict[str, ScalingRule] = {}
        self.histories: Dict[str, ScalingHistory] = {}
        self.active_monitors: Dict[str, asyncio.Task] = {}
        self.metrics_service = RealtimeMetricsService()
        self.anomaly_service = AnomalyDetectionService()
        self.stats_service = StatisticalAnalysisService()
        self.hypothesis_service = HypothesisTestingService()
        
    async def create_scaling_rule(
        self,
        experiment_id: str,
        name: str,
        mode: ScalingMode = ScalingMode.BALANCED,
        **kwargs
    ) -> ScalingRule:
        """
        创建扩量规则
        
        Args:
            experiment_id: 实验ID
            name: 规则名称
            mode: 扩量模式
            **kwargs: 其他参数
            
        Returns:
            扩量规则
        """
        # 根据模式设置默认参数
        if mode == ScalingMode.AGGRESSIVE:
            scale_increment = kwargs.get("scale_increment", 20.0)
            scale_decrement = kwargs.get("scale_decrement", 10.0)
            cooldown_minutes = kwargs.get("cooldown_minutes", 15)
        elif mode == ScalingMode.CONSERVATIVE:
            scale_increment = kwargs.get("scale_increment", 5.0)
            scale_decrement = kwargs.get("scale_decrement", 2.5)
            cooldown_minutes = kwargs.get("cooldown_minutes", 60)
        else:  # BALANCED
            scale_increment = kwargs.get("scale_increment", 10.0)
            scale_decrement = kwargs.get("scale_decrement", 5.0)
            cooldown_minutes = kwargs.get("cooldown_minutes", 30)
            
        # 创建默认条件
        scale_up_conditions = kwargs.get("scale_up_conditions", [
            ScalingCondition(
                trigger=ScalingTrigger.STATISTICAL_SIGNIFICANCE,
                metric_name="conversion_rate",
                operator=">",
                threshold=0.0,
                confidence_level=0.95
            ),
            ScalingCondition(
                trigger=ScalingTrigger.SAMPLE_SIZE,
                metric_name=None,
                operator=">=",
                threshold=1000,
                min_sample_size=1000
            )
        ])
        
        scale_down_conditions = kwargs.get("scale_down_conditions", [
            ScalingCondition(
                trigger=ScalingTrigger.METRIC_THRESHOLD,
                metric_name="error_rate",
                operator=">",
                threshold=0.05
            )
        ])
        
        rule = ScalingRule(
            id=f"rule_{experiment_id}_{datetime.utcnow().timestamp()}",
            name=name,
            description=kwargs.get("description", ""),
            experiment_id=experiment_id,
            variant=kwargs.get("variant", "treatment"),
            mode=mode,
            conditions=kwargs.get("conditions", []),
            scale_up_conditions=scale_up_conditions,
            scale_down_conditions=scale_down_conditions,
            scale_increment=scale_increment,
            scale_decrement=scale_decrement,
            min_percentage=kwargs.get("min_percentage", 1.0),
            max_percentage=kwargs.get("max_percentage", 100.0),
            cooldown_minutes=cooldown_minutes,
            enabled=kwargs.get("enabled", True)
        )
        
        self.rules[rule.id] = rule
        
        # 初始化历史记录
        if experiment_id not in self.histories:
            self.histories[experiment_id] = ScalingHistory(
                experiment_id=experiment_id,
                current_percentage=50.0
            )
            
        return rule
        
    async def start_auto_scaling(self, rule_id: str) -> bool:
        """
        启动自动扩量
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功启动
        """
        if rule_id not in self.rules:
            return False
            
        rule = self.rules[rule_id]
        
        if rule_id in self.active_monitors:
            return False  # 已经在运行
            
        # 启动监控任务
        task = asyncio.create_task(self._monitor_and_scale(rule))
        self.active_monitors[rule_id] = task
        
        return True
        
    async def stop_auto_scaling(self, rule_id: str) -> bool:
        """停止自动扩量"""
        if rule_id in self.active_monitors:
            self.active_monitors[rule_id].cancel()
            del self.active_monitors[rule_id]
            return True
        return False
        
    async def _monitor_and_scale(self, rule: ScalingRule):
        """监控并执行扩量"""
        while rule.enabled:
            try:
                # 获取实验指标
                metrics = await self.metrics_service.get_experiment_metrics(
                    rule.experiment_id
                )
                
                # 获取统计数据
                stats = await self._calculate_statistics(
                    rule.experiment_id,
                    rule.variant
                )
                
                # 检查是否在冷却期
                history = self.histories.get(rule.experiment_id)
                if history and history.last_scaled_at:
                    time_since_last = datetime.utcnow() - history.last_scaled_at
                    if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                        await asyncio.sleep(60)  # 等待1分钟后重试
                        continue
                        
                # 评估扩量决策
                decision = await self._evaluate_scaling(rule, metrics, stats)
                
                if decision and decision.direction != ScalingDirection.HOLD:
                    # 执行扩量
                    await self._apply_scaling(decision)
                    
                    # 记录历史
                    if history:
                        history.decisions.append(decision)
                        history.last_scaled_at = datetime.utcnow()
                        history.current_percentage = decision.target_percentage
                        
                        if decision.direction == ScalingDirection.UP:
                            history.total_scale_ups += 1
                        else:
                            history.total_scale_downs += 1
                            
                # 等待下一个评估周期
                await asyncio.sleep(300)  # 5分钟
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"自动扩量监控错误: {e}")
                await asyncio.sleep(60)
                
    async def _evaluate_scaling(
        self,
        rule: ScalingRule,
        metrics: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> Optional[ScalingDecision]:
        """评估扩量决策"""
        history = self.histories.get(rule.experiment_id)
        current_percentage = history.current_percentage if history else 50.0
        
        # 检查扩量条件
        should_scale_up = all(
            cond.evaluate(metrics, stats) 
            for cond in rule.scale_up_conditions
        )
        
        # 检查缩量条件
        should_scale_down = any(
            cond.evaluate(metrics, stats) 
            for cond in rule.scale_down_conditions
        )
        
        # 决定方向和目标
        if should_scale_up and not should_scale_down:
            direction = ScalingDirection.UP
            target_percentage = min(
                current_percentage + rule.scale_increment,
                rule.max_percentage
            )
            reason = "满足扩量条件"
            confidence = self._calculate_confidence(metrics, stats, "up")
            
        elif should_scale_down:
            direction = ScalingDirection.DOWN
            target_percentage = max(
                current_percentage - rule.scale_decrement,
                rule.min_percentage
            )
            reason = "触发缩量条件"
            confidence = self._calculate_confidence(metrics, stats, "down")
            
        else:
            direction = ScalingDirection.HOLD
            target_percentage = current_percentage
            reason = "保持当前流量"
            confidence = 1.0
            
        # 如果目标等于当前，不需要扩量
        if target_percentage == current_percentage:
            direction = ScalingDirection.HOLD
            
        return ScalingDecision(
            rule_id=rule.id,
            experiment_id=rule.experiment_id,
            timestamp=datetime.utcnow(),
            direction=direction,
            current_percentage=current_percentage,
            target_percentage=target_percentage,
            reason=reason,
            confidence=confidence,
            metrics_snapshot=metrics
        )
        
    async def _apply_scaling(self, decision: ScalingDecision):
        """应用扩量决策"""
        # 这里应该调用实际的流量调整服务
        print(f"扩量决策: {decision.experiment_id} "
              f"从 {decision.current_percentage}% "
              f"到 {decision.target_percentage}%")
        
        decision.applied = True
        
    async def _calculate_statistics(
        self,
        experiment_id: str,
        variant: str
    ) -> Dict[str, Any]:
        """计算统计数据"""
        # 获取实验数据
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        
        # 简化示例：计算转化率的统计显著性
        control_data = metrics.get("control", {})
        treatment_data = metrics.get(variant, {})
        
        if control_data and treatment_data:
            # 进行假设检验
            test_result = await self.hypothesis_service.proportion_test(
                control_data.get("conversions", 0),
                control_data.get("samples", 1),
                treatment_data.get("conversions", 0),
                treatment_data.get("samples", 1)
            )
            
            return {
                "p_value": test_result.get("p_value", 1.0),
                "sample_size": treatment_data.get("samples", 0),
                "ci_lower": test_result.get("ci_lower", 0),
                "ci_upper": test_result.get("ci_upper", 0)
            }
            
        return {}
        
    def _calculate_confidence(
        self,
        metrics: Dict[str, Any],
        stats: Dict[str, Any],
        direction: str
    ) -> float:
        """计算决策置信度"""
        # 基于多个因素计算置信度
        confidence_factors = []
        
        # 统计显著性
        p_value = stats.get("p_value", 1.0)
        if p_value < 0.01:
            confidence_factors.append(0.95)
        elif p_value < 0.05:
            confidence_factors.append(0.85)
        else:
            confidence_factors.append(0.5)
            
        # 样本量
        sample_size = stats.get("sample_size", 0)
        if sample_size > 10000:
            confidence_factors.append(0.9)
        elif sample_size > 1000:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
            
        # 效果大小
        ci_lower = stats.get("ci_lower", 0)
        ci_upper = stats.get("ci_upper", 0)
        effect_size = abs(ci_upper - ci_lower)
        
        if effect_size > 0.1:
            confidence_factors.append(0.8)
        elif effect_size > 0.05:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
            
        # 计算平均置信度
        if confidence_factors:
            return statistics.mean(confidence_factors)
        return 0.5
        
    async def get_scaling_recommendations(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        获取扩量建议
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            扩量建议
        """
        # 获取当前指标
        metrics = await self.metrics_service.get_experiment_metrics(experiment_id)
        stats = await self._calculate_statistics(experiment_id, "treatment")
        
        # 获取历史记录
        history = self.histories.get(experiment_id, None)
        
        # 分析趋势
        recommendations = []
        
        # 基于统计显著性
        p_value = stats.get("p_value", 1.0)
        if p_value < 0.05:
            recommendations.append({
                "type": "statistical",
                "action": "scale_up",
                "reason": f"统计显著 (p={p_value:.4f})",
                "confidence": 0.9
            })
            
        # 基于样本量
        sample_size = stats.get("sample_size", 0)
        if sample_size < 1000:
            recommendations.append({
                "type": "sample_size",
                "action": "hold",
                "reason": f"样本量不足 ({sample_size})",
                "confidence": 0.8
            })
            
        # 基于历史趋势
        if history and len(history.decisions) >= 3:
            recent_decisions = history.decisions[-3:]
            up_count = sum(1 for d in recent_decisions if d.direction == ScalingDirection.UP)
            
            if up_count >= 2:
                recommendations.append({
                    "type": "trend",
                    "action": "scale_up",
                    "reason": "持续正向趋势",
                    "confidence": 0.7
                })
                
        # 综合建议
        if recommendations:
            # 选择置信度最高的建议
            best_rec = max(recommendations, key=lambda x: x["confidence"])
            action = best_rec["action"]
        else:
            action = "hold"
            
        return {
            "experiment_id": experiment_id,
            "current_percentage": history.current_percentage if history else 50.0,
            "recommended_action": action,
            "recommendations": recommendations,
            "metrics_summary": {
                "p_value": stats.get("p_value"),
                "sample_size": stats.get("sample_size"),
                "confidence_interval": [
                    stats.get("ci_lower"),
                    stats.get("ci_upper")
                ]
            }
        }
        
    async def simulate_scaling(
        self,
        experiment_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        模拟扩量过程
        
        Args:
            experiment_id: 实验ID
            days: 模拟天数
            
        Returns:
            模拟结果
        """
        simulations = []
        current_percentage = 10.0
        
        for day in range(days):
            # 模拟每天的指标
            sample_size = 1000 * (day + 1)
            conversion_rate = 0.1 + (day * 0.01)  # 逐渐提升
            p_value = max(0.001, 0.1 - (day * 0.015))  # 逐渐显著
            
            # 决定是否扩量
            if p_value < 0.05 and sample_size >= 1000:
                # 扩量
                new_percentage = min(current_percentage + 15, 100)
                action = "scale_up"
            else:
                new_percentage = current_percentage
                action = "hold"
                
            simulations.append({
                "day": day + 1,
                "current_percentage": current_percentage,
                "new_percentage": new_percentage,
                "action": action,
                "metrics": {
                    "sample_size": sample_size,
                    "conversion_rate": conversion_rate,
                    "p_value": p_value
                }
            })
            
            current_percentage = new_percentage
            
        return simulations


class ScalingTemplates:
    """扩量模板"""
    
    @staticmethod
    async def create_safe_scaling_rule(
        service: AutoScalingService,
        experiment_id: str
    ) -> ScalingRule:
        """创建安全扩量规则"""
        return await service.create_scaling_rule(
            experiment_id=experiment_id,
            name="安全扩量",
            mode=ScalingMode.CONSERVATIVE,
            scale_increment=5.0,
            scale_decrement=10.0,
            min_percentage=1.0,
            max_percentage=50.0,
            cooldown_minutes=60,
            scale_up_conditions=[
                ScalingCondition(
                    trigger=ScalingTrigger.STATISTICAL_SIGNIFICANCE,
                    metric_name="conversion_rate",
                    operator=">",
                    threshold=0.0,
                    confidence_level=0.99  # 高置信度
                ),
                ScalingCondition(
                    trigger=ScalingTrigger.SAMPLE_SIZE,
                    metric_name=None,
                    operator=">=",
                    threshold=5000,  # 大样本量
                    min_sample_size=5000
                )
            ],
            scale_down_conditions=[
                ScalingCondition(
                    trigger=ScalingTrigger.METRIC_THRESHOLD,
                    metric_name="error_rate",
                    operator=">",
                    threshold=0.01  # 低错误率阈值
                )
            ]
        )
        
    @staticmethod
    async def create_aggressive_scaling_rule(
        service: AutoScalingService,
        experiment_id: str
    ) -> ScalingRule:
        """创建激进扩量规则"""
        return await service.create_scaling_rule(
            experiment_id=experiment_id,
            name="激进扩量",
            mode=ScalingMode.AGGRESSIVE,
            scale_increment=25.0,
            scale_decrement=5.0,
            min_percentage=5.0,
            max_percentage=100.0,
            cooldown_minutes=15,
            scale_up_conditions=[
                ScalingCondition(
                    trigger=ScalingTrigger.STATISTICAL_SIGNIFICANCE,
                    metric_name="conversion_rate",
                    operator=">",
                    threshold=0.0,
                    confidence_level=0.90  # 较低置信度
                ),
                ScalingCondition(
                    trigger=ScalingTrigger.SAMPLE_SIZE,
                    metric_name=None,
                    operator=">=",
                    threshold=500,  # 小样本量
                    min_sample_size=500
                )
            ]
        )