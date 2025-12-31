"""
智能体行为监控系统
负责记录和监控Q-Learning智能体的行为、决策过程和性能指标
"""

import asyncio
import json
import time
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from ..reinforcement_learning.qlearning.base import QLearningAgent, AgentState, Experience

from src.core.logging import get_logger
logger = get_logger(__name__)

class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ActionType(str, Enum):
    """动作类型"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    RANDOM = "random"
    FORCED = "forced"

@dataclass
class AgentAction:
    """智能体动作记录"""
    timestamp: datetime
    agent_id: str
    session_id: str
    episode: int
    step: int
    state: Union[List[float], np.ndarray]
    action: int
    action_type: ActionType
    q_values: Optional[List[float]]
    confidence: float
    exploration_rate: float
    reward: Optional[float] = None
    next_state: Optional[Union[List[float], np.ndarray]] = None
    done: Optional[bool] = None

@dataclass
class AgentDecision:
    """智能体决策过程记录"""
    timestamp: datetime
    agent_id: str
    session_id: str
    state: Union[List[float], np.ndarray]
    available_actions: List[int]
    q_values: List[float]
    selected_action: int
    action_type: ActionType
    exploration_rate: float
    decision_time: float  # 决策耗时(毫秒)
    reasoning: Optional[str] = None

@dataclass
class PerformanceMetric:
    """性能指标记录"""
    timestamp: datetime
    agent_id: str
    session_id: str
    episode: int
    metric_name: str
    metric_value: float
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class AgentEvent:
    """智能体事件记录"""
    timestamp: datetime
    agent_id: str
    session_id: str
    event_type: str
    level: LogLevel
    message: str
    data: Optional[Dict[str, Any]] = None

class AgentMonitor:
    """智能体行为监控器"""
    
    def __init__(self, max_memory_size: int = 10000, buffer_flush_interval: int = 60):
        """
        初始化监控器
        
        Args:
            max_memory_size: 内存中最大记录数量
            buffer_flush_interval: 缓冲区刷新间隔(秒)
        """
        self.max_memory_size = max_memory_size
        self.buffer_flush_interval = buffer_flush_interval
        
        # 内存缓冲区
        self.action_buffer: deque = deque(maxlen=max_memory_size)
        self.decision_buffer: deque = deque(maxlen=max_memory_size)
        self.performance_buffer: deque = deque(maxlen=max_memory_size)
        self.event_buffer: deque = deque(maxlen=max_memory_size)
        
        # 实时统计
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.session_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 监控状态
        self.is_monitoring = False
        self.last_flush_time = time.time()
        
        logger.info("智能体监控器已初始化")
    
    async def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            logger.warning("监控器已在运行")
            return
            
        self.is_monitoring = True
        logger.info("开始智能体行为监控")
        
        # 启动定时任务
        create_task_with_logging(self._periodic_flush())
        create_task_with_logging(self._update_real_time_stats())
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self.is_monitoring:
            logger.warning("监控器未在运行")
            return
            
        self.is_monitoring = False
        
        # 刷新剩余数据
        await self._flush_buffers()
        
        logger.info("智能体行为监控已停止")
    
    def log_action(self, action_record: AgentAction):
        """记录智能体动作"""
        try:
            self.action_buffer.append(action_record)
            
            # 更新实时统计
            agent_id = action_record.agent_id
            session_id = action_record.session_id
            
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = {
                    'total_actions': 0,
                    'exploration_count': 0,
                    'exploitation_count': 0,
                    'average_reward': 0.0,
                    'last_activity': action_record.timestamp
                }
            
            stats = self.agent_stats[agent_id]
            stats['total_actions'] += 1
            stats['last_activity'] = action_record.timestamp
            
            if action_record.action_type == ActionType.EXPLORATION:
                stats['exploration_count'] += 1
            elif action_record.action_type == ActionType.EXPLOITATION:
                stats['exploitation_count'] += 1
            
            if action_record.reward is not None:
                # 更新平均奖励(指数移动平均)
                alpha = 0.1
                current_avg = stats['average_reward']
                stats['average_reward'] = alpha * action_record.reward + (1 - alpha) * current_avg
            
            logger.debug(f"记录智能体动作: {agent_id}, 动作: {action_record.action}")
            
        except Exception as e:
            logger.error(f"记录智能体动作失败: {e}")
    
    def log_decision(self, decision_record: AgentDecision):
        """记录智能体决策过程"""
        try:
            self.decision_buffer.append(decision_record)
            
            # 更新决策统计
            agent_id = decision_record.agent_id
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = {}
            
            if 'decision_times' not in self.agent_stats[agent_id]:
                self.agent_stats[agent_id]['decision_times'] = deque(maxlen=100)
            
            self.agent_stats[agent_id]['decision_times'].append(decision_record.decision_time)
            
            logger.debug(f"记录智能体决策: {agent_id}, 耗时: {decision_record.decision_time}ms")
            
        except Exception as e:
            logger.error(f"记录智能体决策失败: {e}")
    
    def log_performance(self, metric_record: PerformanceMetric):
        """记录性能指标"""
        try:
            self.performance_buffer.append(metric_record)
            
            # 更新性能统计
            agent_id = metric_record.agent_id
            metric_name = metric_record.metric_name
            
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = {}
            
            if 'performance_metrics' not in self.agent_stats[agent_id]:
                self.agent_stats[agent_id]['performance_metrics'] = {}
            
            if metric_name not in self.agent_stats[agent_id]['performance_metrics']:
                self.agent_stats[agent_id]['performance_metrics'][metric_name] = deque(maxlen=50)
            
            self.agent_stats[agent_id]['performance_metrics'][metric_name].append(
                metric_record.metric_value
            )
            
            logger.debug(f"记录性能指标: {agent_id}, {metric_name}: {metric_record.metric_value}")
            
        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
    
    def log_event(self, event_record: AgentEvent):
        """记录智能体事件"""
        try:
            self.event_buffer.append(event_record)
            
            # 统计事件数量
            agent_id = event_record.agent_id
            if agent_id not in self.agent_stats:
                self.agent_stats[agent_id] = {}
            
            if 'event_counts' not in self.agent_stats[agent_id]:
                self.agent_stats[agent_id]['event_counts'] = defaultdict(int)
            
            self.agent_stats[agent_id]['event_counts'][event_record.level] += 1
            
            logger.log(
                getattr(logger, event_record.level.value, logger.info),
                f"智能体事件 [{agent_id}]: {event_record.message}"
            )
            
        except Exception as e:
            logger.error(f"记录智能体事件失败: {e}")
    
    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """获取智能体统计摘要"""
        if agent_id not in self.agent_stats:
            return {}
        
        stats = self.agent_stats[agent_id].copy()
        
        # 计算派生指标
        if 'decision_times' in stats:
            decision_times = list(stats['decision_times'])
            stats['avg_decision_time'] = np.mean(decision_times) if decision_times else 0
            stats['max_decision_time'] = np.max(decision_times) if decision_times else 0
        
        # 计算探索率
        total_actions = stats.get('total_actions', 0)
        if total_actions > 0:
            exploration_rate = stats.get('exploration_count', 0) / total_actions
            stats['exploration_rate'] = exploration_rate
        
        # 性能指标汇总
        if 'performance_metrics' in stats:
            metrics_summary = {}
            for metric_name, values in stats['performance_metrics'].items():
                if values:
                    metrics_summary[metric_name] = {
                        'current': values[-1],
                        'average': np.mean(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
                    }
            stats['metrics_summary'] = metrics_summary
        
        return stats
    
    def get_recent_actions(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的智能体动作"""
        actions = [
            asdict(action) for action in self.action_buffer
            if action.agent_id == agent_id
        ]
        
        # 按时间倒序排列
        actions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return actions[:limit]
    
    def get_performance_trend(self, agent_id: str, metric_name: str, 
                            hours: int = 24) -> List[Dict[str, Any]]:
        """获取性能趋势数据"""
        cutoff_time = utc_now() - timedelta(hours=hours)
        
        trend_data = []
        for metric in self.performance_buffer:
            if (metric.agent_id == agent_id and 
                metric.metric_name == metric_name and
                metric.timestamp >= cutoff_time):
                
                trend_data.append({
                    'timestamp': metric.timestamp,
                    'value': metric.metric_value,
                    'episode': metric.episode
                })
        
        # 按时间排序
        trend_data.sort(key=lambda x: x['timestamp'])
        
        return trend_data
    
    async def _periodic_flush(self):
        """定期刷新缓冲区到存储"""
        while self.is_monitoring:
            try:
                current_time = time.time()
                if current_time - self.last_flush_time >= self.buffer_flush_interval:
                    await self._flush_buffers()
                    self.last_flush_time = current_time
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"定期刷新任务异常: {e}")
                await asyncio.sleep(60)  # 出错时等待更久
    
    async def _flush_buffers(self):
        """刷新缓冲区数据到持久存储"""
        try:
            # 这里可以实现将数据写入数据库或文件的逻辑
            # 现在只是简单地记录一下缓冲区大小
            
            logger.info(f"刷新监控数据 - 动作: {len(self.action_buffer)}, "
                       f"决策: {len(self.decision_buffer)}, "
                       f"性能: {len(self.performance_buffer)}, "
                       f"事件: {len(self.event_buffer)}")
            
            # 可选：将旧数据序列化到文件
            # await self._save_to_storage()
            
        except Exception as e:
            logger.error(f"刷新缓冲区失败: {e}")
    
    async def _update_real_time_stats(self):
        """更新实时统计信息"""
        while self.is_monitoring:
            try:
                # 清理过期的统计信息
                current_time = utc_now()
                cutoff_time = current_time - timedelta(hours=24)
                
                for agent_id in list(self.agent_stats.keys()):
                    stats = self.agent_stats[agent_id]
                    last_activity = stats.get('last_activity')
                    
                    if last_activity and last_activity < cutoff_time:
                        # 标记为非活跃
                        stats['is_active'] = False
                    else:
                        stats['is_active'] = True
                
                await asyncio.sleep(300)  # 每5分钟更新一次
                
            except Exception as e:
                logger.error(f"更新实时统计异常: {e}")
                await asyncio.sleep(300)

# 全局监控器实例
global_monitor = AgentMonitor()

def get_agent_monitor() -> AgentMonitor:
    """获取全局智能体监控器实例"""
    return global_monitor

class MonitoredQLearningAgent(QLearningAgent):
    """带监控功能的Q-Learning智能体包装器"""
    
    def __init__(self, base_agent: QLearningAgent, monitor: Optional[AgentMonitor] = None):
        """
        初始化监控包装器
        
        Args:
            base_agent: 基础Q-Learning智能体
            monitor: 监控器实例，如果为None则使用全局监控器
        """
        super().__init__(base_agent.agent_id, base_agent.state_size, base_agent.action_size, base_agent.config)
        self.base_agent = base_agent
        self.monitor = monitor or get_agent_monitor()
        self.session_id = f"session_{int(time.time())}"
        if hasattr(base_agent, "action_space"):
            self.action_space = base_agent.action_space
        self.epsilon = base_agent.epsilon
        
        # 复制Q表或模型
        if hasattr(base_agent, 'q_table'):
            self.q_table = base_agent.q_table
        if hasattr(base_agent, 'model'):
            self.model = base_agent.model
    
    def _state_to_list(self, state: AgentState) -> List[float]:
        if isinstance(state.features, dict):
            sorted_keys = sorted(state.features.keys())
            return [float(state.features[key]) for key in sorted_keys]
        if isinstance(state.features, (list, np.ndarray)):
            return [float(x) for x in state.features]
        return []

    def _q_values_to_list(self, q_values: Optional[Dict[str, float]]) -> List[float]:
        if not q_values:
            return []
        action_space = getattr(self, "action_space", None)
        if action_space:
            return [float(q_values.get(action, 0.0)) for action in action_space]
        return [float(v) for _, v in sorted(q_values.items(), key=lambda kv: str(kv[0]))]

    def get_action(self, state: AgentState, exploration: bool = True) -> str:
        """选择动作（带监控）"""
        start_time = time.time()
        normalized_state = self._normalize_state(state)
        q_values = self.base_agent.get_q_values(normalized_state)
        action_label = self.base_agent.get_action(normalized_state, exploration=exploration)
        action_index = self._action_label_to_index(action_label)

        q_values_list = self._q_values_to_list(q_values)
        confidence = 0.0
        if q_values_list:
            max_q = max(q_values_list)
            min_q = min(q_values_list)
            if 0 <= action_index < len(q_values_list):
                confidence = 1.0 if max_q == min_q else (q_values_list[action_index] - min_q) / (max_q - min_q)

        decision_time = (time.time() - start_time) * 1000
        decision_record = AgentDecision(
            timestamp=utc_now(),
            agent_id=self.agent_id,
            session_id=self.session_id,
            state=self._state_to_list(normalized_state),
            available_actions=list(range(self.action_size)),
            q_values=q_values_list,
            selected_action=action_index,
            action_type=ActionType.EXPLORATION if exploration else ActionType.EXPLOITATION,
            exploration_rate=self.base_agent.epsilon,
            decision_time=decision_time
        )
        self.monitor.log_decision(decision_record)
        return action_label

    def update_q_value(self, experience: Experience) -> Optional[float]:
        return self.base_agent.update_q_value(experience)

    def get_q_values(self, state: AgentState) -> Dict[str, float]:
        return self.base_agent.get_q_values(state)

    def save_model(self, filepath: str) -> None:
        self.base_agent.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self.base_agent.load_model(filepath)

    def learn(self, state: Union[List[float], np.ndarray], action: int,
              reward: float, next_state: Union[List[float], np.ndarray], done: bool):
        """学习（带监控）"""
        normalized_state = self._normalize_state(state)
        normalized_next_state = self._normalize_state(next_state)
        action_label = self._action_index_to_label(action)
        action_index = self._action_label_to_index(action_label)
        q_values_list = self._q_values_to_list(self.base_agent.get_q_values(normalized_state))
        confidence = 0.0
        if q_values_list and 0 <= action_index < len(q_values_list):
            max_q = max(q_values_list)
            min_q = min(q_values_list)
            confidence = 1.0 if max_q == min_q else (q_values_list[action_index] - min_q) / (max_q - min_q)

        action_record = AgentAction(
            timestamp=utc_now(),
            agent_id=self.agent_id,
            session_id=self.session_id,
            episode=getattr(self, 'current_episode', 0),
            step=getattr(self, 'current_step', 0),
            state=self._state_to_list(normalized_state),
            action=action_index,
            action_type=ActionType.EXPLOITATION,
            q_values=q_values_list,
            confidence=confidence,
            exploration_rate=self.base_agent.epsilon,
            reward=reward,
            next_state=self._state_to_list(normalized_next_state),
            done=done
        )
        self.monitor.log_action(action_record)
        loss = self.base_agent.learn(normalized_state, action, reward, normalized_next_state, done)
        self.epsilon = self.base_agent.epsilon
        return loss
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             additional_data: Optional[Dict[str, Any]] = None):
        """记录性能指标"""
        metric_record = PerformanceMetric(
            timestamp=utc_now(),
            agent_id=self.agent_id,
            session_id=self.session_id,
            episode=getattr(self, 'current_episode', 0),
            metric_name=metric_name,
            metric_value=value,
            additional_data=additional_data
        )
        
        self.monitor.log_performance(metric_record)
    
    def log_event(self, event_type: str, message: str, level: LogLevel = LogLevel.INFO,
                  data: Optional[Dict[str, Any]] = None):
        """记录事件"""
        event_record = AgentEvent(
            timestamp=utc_now(),
            agent_id=self.agent_id,
            session_id=self.session_id,
            event_type=event_type,
            level=level,
            message=message,
            data=data
        )
        
        self.monitor.log_event(event_record)

async def initialize_agent_monitoring():
    """初始化智能体监控系统"""
    monitor = get_agent_monitor()
    await monitor.start_monitoring()
    logger.info("智能体监控系统已启动")

async def shutdown_agent_monitoring():
    """关闭智能体监控系统"""
    monitor = get_agent_monitor()
    await monitor.stop_monitoring()
    logger.info("智能体监控系统已关闭")
