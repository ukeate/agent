"""
处理模式选择器

根据请求特征、系统负载和历史性能智能选择最优的处理模式。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import logging
import time
import statistics
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    """选择策略"""
    HEURISTIC = "heuristic"           # 启发式规则
    PERFORMANCE_BASED = "performance" # 基于性能历史
    LOAD_AWARE = "load_aware"        # 负载感知
    ML_PREDICTED = "ml_predicted"    # 机器学习预测（未实现）
    HYBRID = "hybrid"                # 混合策略


@dataclass
class SystemLoadMetrics:
    """系统负载指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_depth: int = 0
    active_connections: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    
    # 流式处理指标
    streaming_sessions: int = 0
    streaming_throughput: float = 0.0
    
    # 批处理指标
    batch_jobs_active: int = 0
    batch_queue_size: int = 0
    batch_processing_rate: float = 0.0
    
    # 计算负载评分 (0-1)
    @property
    def load_score(self) -> float:
        """综合负载评分"""
        components = [
            self.cpu_usage,
            self.memory_usage,
            min(1.0, self.queue_depth / 1000),  # 假设1000为满负载
            min(1.0, self.active_connections / 100),  # 假设100为满负载
            min(1.0, self.average_response_time / 10),  # 假设10秒为高延迟
            self.error_rate
        ]
        return sum(components) / len(components)


@dataclass
class ModePerformanceHistory:
    """模式性能历史"""
    mode: str
    request_count: int = 0
    total_processing_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    
    # 性能指标
    average_processing_time: float = 0.0
    success_rate: float = 1.0
    throughput: float = 0.0  # 请求/秒
    
    # 时间窗口
    window_start: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_metrics(self, processing_time: float, success: bool):
        """更新性能指标"""
        self.request_count += 1
        self.total_processing_time += processing_time
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # 计算平均值
        self.average_processing_time = self.total_processing_time / self.request_count
        self.success_rate = self.success_count / self.request_count
        
        # 计算吞吐量（基于时间窗口）
        elapsed_hours = (utc_now() - self.window_start).total_seconds() / 3600
        if elapsed_hours > 0:
            self.throughput = self.request_count / elapsed_hours
        
        self.last_updated = utc_now()
    
    @property
    def performance_score(self) -> float:
        """性能评分 (0-1，越高越好)"""
        if self.request_count == 0:
            return 0.5  # 中性评分
        
        # 基于成功率、处理时间和吞吐量的综合评分
        time_score = max(0, 1 - (self.average_processing_time / 60))  # 60秒为基准
        throughput_score = min(1, self.throughput / 100)  # 100请求/小时为基准
        
        return (self.success_rate * 0.4 + time_score * 0.3 + throughput_score * 0.3)


class ModeSelector:
    """处理模式选择器"""
    
    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.HYBRID):
        self.strategy = strategy
        
        # 性能历史记录
        from .processing_engine import ProcessingMode
        self.performance_history: Dict[ProcessingMode, ModePerformanceHistory] = {
            mode: ModePerformanceHistory(mode.value) for mode in ProcessingMode
        }
        
        # 系统负载监控
        self.current_load = SystemLoadMetrics()
        self.load_history: List[SystemLoadMetrics] = []
        
        # 启发式规则权重
        self.heuristic_weights = {
            "item_count": 0.3,
            "real_time_requirement": 0.25,
            "aggregation_requirement": 0.2,
            "system_load": 0.15,
            "complexity": 0.1
        }
        
        # 决策历史（用于学习和调优）
        self.decision_history: List[Dict[str, Any]] = []
        
    async def select_mode(self, request) -> 'ProcessingMode':
        """选择最优处理模式"""
        from .processing_engine import ProcessingMode, ProcessingRequest
        
        # 更新系统负载
        await self._update_system_load()
        
        # 根据策略选择模式
        if self.strategy == SelectionStrategy.HEURISTIC:
            mode = self._heuristic_selection(request)
        elif self.strategy == SelectionStrategy.PERFORMANCE_BASED:
            mode = self._performance_based_selection(request)
        elif self.strategy == SelectionStrategy.LOAD_AWARE:
            mode = self._load_aware_selection(request)
        elif self.strategy == SelectionStrategy.HYBRID:
            mode = self._hybrid_selection(request)
        else:
            # 默认启发式
            mode = self._heuristic_selection(request)
        
        # 记录决策
        decision_record = {
            "timestamp": utc_now().isoformat(),
            "selected_mode": mode.value,
            "strategy": self.strategy.value,
            "request_features": self._extract_request_features(request),
            "system_load": self._serialize_load_metrics(),
            "performance_scores": {
                m.value: hist.performance_score 
                for m, hist in self.performance_history.items()
            }
        }
        
        self.decision_history.append(decision_record)
        
        # 限制决策历史大小
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
        
        logger.info(f"选择处理模式: {mode.value} (策略: {self.strategy.value})")
        return mode
    
    def _heuristic_selection(self, request) -> 'ProcessingMode':
        """启发式模式选择"""
        from .processing_engine import ProcessingMode
        
        scores = {}
        
        # 流式处理评分
        stream_score = 0
        if request.requires_real_time:
            stream_score += 0.4
        if request.streaming_enabled and request.item_count <= 10:
            stream_score += 0.3
        if not request.requires_aggregation:
            stream_score += 0.2
        if self.current_load.load_score < 0.5:
            stream_score += 0.1
        
        scores[ProcessingMode.STREAM] = stream_score
        
        # 批处理评分
        batch_score = 0
        if request.item_count > 20:
            batch_score += 0.4
        if not request.requires_real_time:
            batch_score += 0.3
        if request.requires_aggregation:
            batch_score += 0.2
        if self.current_load.batch_queue_size < 100:
            batch_score += 0.1
        
        scores[ProcessingMode.BATCH] = batch_score
        
        # 混合处理评分
        hybrid_score = 0
        if 5 < request.item_count <= 20:
            hybrid_score += 0.3
        if request.requires_real_time and request.requires_aggregation:
            hybrid_score += 0.4
        if 0.3 < self.current_load.load_score < 0.7:
            hybrid_score += 0.2
        if request.streaming_enabled:
            hybrid_score += 0.1
        
        scores[ProcessingMode.HYBRID] = hybrid_score
        
        # 流水线处理评分
        pipeline_score = 0
        if request.item_count > 50:
            pipeline_score += 0.3
        if hasattr(request, 'pipeline_stages') and request.pipeline_stages:
            pipeline_score += 0.4
        if self.current_load.load_score < 0.6:
            pipeline_score += 0.3
        
        scores[ProcessingMode.PIPELINE] = pipeline_score
        
        # 选择评分最高的模式
        best_mode = max(scores.keys(), key=lambda k: scores[k])
        
        # 如果所有评分都很低，返回默认流式处理
        if scores[best_mode] < 0.2:
            return ProcessingMode.STREAM
        
        return best_mode
    
    def _performance_based_selection(self, request) -> 'ProcessingMode':
        """基于性能历史的选择"""
        from .processing_engine import ProcessingMode
        
        # 找到性能最好的模式
        best_mode = ProcessingMode.STREAM
        best_score = 0
        
        for mode, history in self.performance_history.items():
            if history.request_count > 5:  # 至少需要一定的样本数
                # 根据请求特征调整性能评分
                adjusted_score = history.performance_score
                
                # 根据请求类型调整
                if mode == ProcessingMode.STREAM and request.requires_real_time:
                    adjusted_score *= 1.2
                elif mode == ProcessingMode.BATCH and request.item_count > 20:
                    adjusted_score *= 1.2
                elif mode == ProcessingMode.HYBRID and request.requires_aggregation:
                    adjusted_score *= 1.1
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_mode = mode
        
        return best_mode
    
    def _load_aware_selection(self, request) -> 'ProcessingMode':
        """负载感知选择"""
        from .processing_engine import ProcessingMode
        
        load_score = self.current_load.load_score
        
        if load_score < 0.3:
            # 低负载：优先选择流式处理
            if request.requires_real_time or request.item_count <= 10:
                return ProcessingMode.STREAM
            else:
                return ProcessingMode.HYBRID
        
        elif load_score < 0.7:
            # 中等负载：选择混合或批处理
            if request.requires_real_time:
                return ProcessingMode.STREAM
            elif request.item_count > 20:
                return ProcessingMode.BATCH
            else:
                return ProcessingMode.HYBRID
        
        else:
            # 高负载：优先批处理
            if request.requires_real_time and request.item_count <= 5:
                return ProcessingMode.STREAM
            else:
                return ProcessingMode.BATCH
    
    def _hybrid_selection(self, request) -> 'ProcessingMode':
        """混合策略选择"""
        # 结合启发式、性能历史和负载感知
        heuristic_mode = self._heuristic_selection(request)
        performance_mode = self._performance_based_selection(request)
        load_aware_mode = self._load_aware_selection(request)
        
        # 投票机制
        votes = {}
        for mode in [heuristic_mode, performance_mode, load_aware_mode]:
            votes[mode] = votes.get(mode, 0) + 1
        
        # 选择票数最多的模式
        best_mode = max(votes.keys(), key=lambda k: votes[k])
        
        # 如果有平票，优先选择性能最好的
        max_votes = max(votes.values())
        tied_modes = [mode for mode, vote_count in votes.items() if vote_count == max_votes]
        
        if len(tied_modes) > 1:
            best_mode = max(
                tied_modes, 
                key=lambda mode: self.performance_history[mode].performance_score
            )
        
        return best_mode
    
    async def _update_system_load(self):
        """更新系统负载指标"""
        # 这里应该集成实际的系统监控
        # 目前使用模拟数据
        
        import psutil
        import random
        
        # 获取真实系统指标
        try:
            self.current_load.cpu_usage = psutil.cpu_percent() / 100.0
            self.current_load.memory_usage = psutil.virtual_memory().percent / 100.0
        except:
            # 如果无法获取真实数据，使用模拟数据
            self.current_load.cpu_usage = random.uniform(0.1, 0.8)
            self.current_load.memory_usage = random.uniform(0.2, 0.7)
        
        # 其他指标可以从相关服务获取
        # self.current_load.queue_depth = await self._get_queue_depth()
        # self.current_load.active_connections = await self._get_active_connections()
        
        # 添加到历史记录
        self.load_history.append(self.current_load)
        
        # 限制历史记录大小
        if len(self.load_history) > 100:
            self.load_history = self.load_history[-50:]
    
    def _extract_request_features(self, request) -> Dict[str, Any]:
        """提取请求特征"""
        return {
            "item_count": request.item_count,
            "requires_real_time": request.requires_real_time,
            "streaming_enabled": request.streaming_enabled,
            "requires_aggregation": request.requires_aggregation,
            "batch_size": request.batch_size,
            "max_parallel_tasks": request.max_parallel_tasks,
            "has_timeout": request.timeout is not None,
            "has_callback": request.callback is not None
        }
    
    def _serialize_load_metrics(self) -> Dict[str, float]:
        """序列化负载指标"""
        return {
            "cpu_usage": self.current_load.cpu_usage,
            "memory_usage": self.current_load.memory_usage,
            "queue_depth": self.current_load.queue_depth,
            "active_connections": self.current_load.active_connections,
            "load_score": self.current_load.load_score
        }
    
    def update_mode_performance(self, mode: 'ProcessingMode', processing_time: float, success: bool):
        """更新模式性能历史"""
        if mode in self.performance_history:
            self.performance_history[mode].update_metrics(processing_time, success)
            
            logger.debug(f"更新模式性能: {mode.value} "
                        f"(处理时间: {processing_time:.2f}s, 成功: {success})")
    
    def get_mode_recommendations(self, request) -> List[Dict[str, Any]]:
        """获取模式推荐（带评分）"""
        from .processing_engine import ProcessingMode
        
        recommendations = []
        
        # 计算每个模式的评分
        for mode in ProcessingMode:
            if mode == ProcessingMode.AUTO:
                continue
            
            # 基于启发式规则的评分
            if mode == ProcessingMode.STREAM:
                score = self._calculate_stream_score(request)
            elif mode == ProcessingMode.BATCH:
                score = self._calculate_batch_score(request)
            elif mode == ProcessingMode.HYBRID:
                score = self._calculate_hybrid_score(request)
            elif mode == ProcessingMode.PIPELINE:
                score = self._calculate_pipeline_score(request)
            else:
                score = 0.1
            
            # 结合性能历史
            performance_score = self.performance_history[mode].performance_score
            combined_score = (score * 0.7) + (performance_score * 0.3)
            
            recommendations.append({
                "mode": mode.value,
                "score": combined_score,
                "heuristic_score": score,
                "performance_score": performance_score,
                "request_count": self.performance_history[mode].request_count,
                "success_rate": self.performance_history[mode].success_rate,
                "avg_processing_time": self.performance_history[mode].average_processing_time
            })
        
        # 按评分排序
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def _calculate_stream_score(self, request) -> float:
        """计算流式处理评分"""
        score = 0.1  # 基础分
        
        if request.requires_real_time:
            score += 0.3
        if request.item_count <= 10:
            score += 0.2
        if request.streaming_enabled:
            score += 0.2
        if not request.requires_aggregation:
            score += 0.1
        if self.current_load.load_score < 0.5:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_batch_score(self, request) -> float:
        """计算批处理评分"""
        score = 0.1  # 基础分
        
        if request.item_count > 20:
            score += 0.3
        if not request.requires_real_time:
            score += 0.2
        if request.requires_aggregation:
            score += 0.2
        if request.batch_size and request.batch_size > 10:
            score += 0.1
        if self.current_load.batch_queue_size < 100:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_hybrid_score(self, request) -> float:
        """计算混合处理评分"""
        score = 0.1  # 基础分
        
        if 5 < request.item_count <= 20:
            score += 0.2
        if request.requires_real_time and request.requires_aggregation:
            score += 0.3
        if request.streaming_enabled:
            score += 0.1
        if 0.3 < self.current_load.load_score < 0.7:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_pipeline_score(self, request) -> float:
        """计算流水线处理评分"""
        score = 0.1  # 基础分
        
        if request.item_count > 50:
            score += 0.3
        if hasattr(request, 'pipeline_stages'):
            score += 0.2
        if self.current_load.load_score < 0.6:
            score += 0.2
        
        return min(1.0, score)
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        # 统计最近的决策
        recent_decisions = self.decision_history[-100:] if self.decision_history else []
        
        mode_counts = {}
        strategy_counts = {}
        
        for decision in recent_decisions:
            mode = decision["selected_mode"]
            strategy = decision["strategy"]
            
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 计算平均系统负载
        avg_load = 0
        if self.load_history:
            avg_load = sum(load.load_score for load in self.load_history) / len(self.load_history)
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions": len(recent_decisions),
            "mode_distribution": mode_counts,
            "strategy_distribution": strategy_counts,
            "current_strategy": self.strategy.value,
            "average_system_load": avg_load,
            "current_system_load": self.current_load.load_score,
            "performance_history": {
                mode.value: {
                    "request_count": hist.request_count,
                    "success_rate": hist.success_rate,
                    "average_processing_time": hist.average_processing_time,
                    "performance_score": hist.performance_score
                }
                for mode, hist in self.performance_history.items()
            }
        }
    
    def set_strategy(self, strategy: SelectionStrategy):
        """设置选择策略"""
        self.strategy = strategy
        logger.info(f"模式选择策略更改为: {strategy.value}")
    
    def clear_history(self):
        """清理历史数据"""
        self.decision_history.clear()
        self.load_history.clear()
        
        # 重置性能历史
        from .processing_engine import ProcessingMode
        self.performance_history = {
            mode: ModePerformanceHistory(mode.value) for mode in ProcessingMode
        }
        
        logger.info("选择器历史数据已清理")