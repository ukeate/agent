"""
推荐效果评估系统

实现在线评估指标、离线回放测试和A/B测试支持，用于评估多臂老虎机推荐算法的性能。
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
from .bandits.base import MultiArmedBandit

@dataclass
class InteractionEvent:
    """交互事件数据结构"""
    event_id: str
    user_id: str
    item_id: str
    timestamp: datetime
    action_type: str  # 'view', 'click', 'like', 'purchase', etc.
    reward: float
    context: Optional[Dict[str, Any]] = None
    algorithm: Optional[str] = None
    experiment_id: Optional[str] = None

@dataclass
class EvaluationMetrics:
    """评估指标数据结构"""
    click_through_rate: float = 0.0
    conversion_rate: float = 0.0
    average_reward: float = 0.0
    cumulative_reward: float = 0.0
    regret: float = 0.0
    exploration_rate: float = 0.0
    coverage: float = 0.0  # 物品覆盖率
    diversity: float = 0.0  # 推荐多样性
    novelty: float = 0.0  # 新颖性
    precision_at_k: Optional[Dict[int, float]] = None
    recall_at_k: Optional[Dict[int, float]] = None
    ndcg_at_k: Optional[Dict[int, float]] = None

class MetricCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def calculate_ctr(events: List[InteractionEvent]) -> float:
        """计算点击率"""
        if not events:
            return 0.0
        
        clicks = sum(1 for e in events if e.action_type in ['click', 'like', 'purchase'])
        views = sum(1 for e in events if e.action_type == 'view')
        
        return clicks / max(views, 1)
    
    @staticmethod
    def calculate_conversion_rate(events: List[InteractionEvent]) -> float:
        """计算转化率"""
        if not events:
            return 0.0
        
        purchases = sum(1 for e in events if e.action_type == 'purchase')
        clicks = sum(1 for e in events if e.action_type in ['click', 'view'])
        
        return purchases / max(clicks, 1)
    
    @staticmethod
    def calculate_coverage(events: List[InteractionEvent], total_items: int) -> float:
        """计算物品覆盖率"""
        if not events or total_items == 0:
            return 0.0
        
        unique_items = len(set(e.item_id for e in events))
        return unique_items / total_items
    
    @staticmethod
    def calculate_diversity(events: List[InteractionEvent]) -> float:
        """计算推荐多样性（基于香农熵）"""
        if not events:
            return 0.0
        
        # 计算每个物品的推荐频率
        item_counts = {}
        for event in events:
            item_counts[event.item_id] = item_counts.get(event.item_id, 0) + 1
        
        if len(item_counts) <= 1:
            return 0.0
        
        # 计算香农熵
        total_events = len(events)
        entropy = 0.0
        
        for count in item_counts.values():
            prob = count / total_events
            entropy -= prob * np.log2(prob)
        
        # 标准化到[0,1]
        max_entropy = np.log2(len(item_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    @staticmethod
    def calculate_precision_at_k(
        recommended_items: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """计算Precision@K"""
        if not recommended_items or k <= 0:
            return 0.0
        
        top_k_items = recommended_items[:k]
        relevant_set = set(relevant_items)
        relevant_in_top_k = sum(1 for item in top_k_items if item in relevant_set)
        
        return relevant_in_top_k / min(k, len(top_k_items))
    
    @staticmethod
    def calculate_recall_at_k(
        recommended_items: List[str], 
        relevant_items: List[str], 
        k: int
    ) -> float:
        """计算Recall@K"""
        if not relevant_items or k <= 0:
            return 0.0
        
        top_k_items = recommended_items[:k]
        relevant_set = set(relevant_items)
        relevant_in_top_k = sum(1 for item in top_k_items if item in relevant_set)
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def calculate_ndcg_at_k(
        recommended_items: List[str], 
        item_relevances: Dict[str, float], 
        k: int
    ) -> float:
        """计算NDCG@K"""
        if not recommended_items or k <= 0:
            return 0.0
        
        def dcg_at_k(items: List[str], relevances: Dict[str, float], k: int) -> float:
            dcg = 0.0
            for i, item in enumerate(items[:k]):
                if item in relevances:
                    rel = relevances[item]
                    dcg += (2**rel - 1) / np.log2(i + 2)
            return dcg
        
        # 计算DCG@K
        dcg = dcg_at_k(recommended_items, item_relevances, k)
        
        # 计算IDCG@K（理想排序的DCG）
        sorted_items = sorted(item_relevances.items(), key=lambda x: x[1], reverse=True)
        ideal_items = [item for item, _ in sorted_items]
        idcg = dcg_at_k(ideal_items, item_relevances, k)
        
        return dcg / idcg if idcg > 0 else 0.0

class OnlineEvaluator:
    """在线评估器"""
    
    def __init__(self, window_size: int = 1000):
        """
        初始化在线评估器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.events_buffer = []
        self.metric_history = []
        self.calculator = MetricCalculator()
    
    def add_interaction(self, event: InteractionEvent):
        """添加交互事件"""
        self.events_buffer.append(event)
        
        # 维护滑动窗口
        if len(self.events_buffer) > self.window_size:
            self.events_buffer = self.events_buffer[-self.window_size:]
    
    def calculate_current_metrics(self, total_items: int = 1000) -> EvaluationMetrics:
        """计算当前评估指标"""
        if not self.events_buffer:
            return EvaluationMetrics()
        
        # 基础指标
        ctr = self.calculator.calculate_ctr(self.events_buffer)
        conversion_rate = self.calculator.calculate_conversion_rate(self.events_buffer)
        
        # 奖励相关指标
        rewards = [e.reward for e in self.events_buffer]
        avg_reward = np.mean(rewards) if rewards else 0.0
        cumulative_reward = sum(rewards)
        
        # 覆盖率和多样性
        coverage = self.calculator.calculate_coverage(self.events_buffer, total_items)
        diversity = self.calculator.calculate_diversity(self.events_buffer)
        
        # 探索率（基于算法信息）
        exploration_rate = self._calculate_exploration_rate()
        
        return EvaluationMetrics(
            click_through_rate=ctr,
            conversion_rate=conversion_rate,
            average_reward=avg_reward,
            cumulative_reward=cumulative_reward,
            coverage=coverage,
            diversity=diversity,
            exploration_rate=exploration_rate
        )
    
    def _calculate_exploration_rate(self) -> float:
        """计算探索率"""
        if not self.events_buffer:
            return 0.0
        
        # 简单启发式：基于物品多样性估算探索率
        unique_items = len(set(e.item_id for e in self.events_buffer))
        total_interactions = len(self.events_buffer)
        
        return unique_items / total_interactions if total_interactions > 0 else 0.0
    
    def get_real_time_metrics(self, last_n_events: int = 100) -> Dict[str, float]:
        """获取实时指标（最近N个事件）"""
        if not self.events_buffer:
            return {}
        
        recent_events = self.events_buffer[-last_n_events:]
        
        return {
            "recent_ctr": self.calculator.calculate_ctr(recent_events),
            "recent_avg_reward": np.mean([e.reward for e in recent_events]),
            "recent_diversity": self.calculator.calculate_diversity(recent_events),
            "events_count": len(recent_events)
        }

class OfflineEvaluator:
    """离线评估器（回放测试）"""
    
    def __init__(self):
        """初始化离线评估器"""
        self.calculator = MetricCalculator()
        self.replay_results = {}
    
    def replay_evaluation(
        self,
        bandit: MultiArmedBandit,
        historical_events: List[InteractionEvent],
        policy_name: str = "replay_policy"
    ) -> Dict[str, Any]:
        """
        回放评估
        
        Args:
            bandit: 多臂老虎机算法
            historical_events: 历史交互事件
            policy_name: 策略名称
            
        Returns:
            评估结果
        """
        # 重置算法状态
        bandit.reset()
        
        simulated_events = []
        total_reward = 0.0
        correct_predictions = 0
        
        for event in historical_events:
            # 使用历史上下文进行预测
            context = event.context
            predicted_arm = bandit.select_arm(context)
            
            # 检查预测是否正确（如果历史选择了相同的臂）
            if str(predicted_arm) == event.item_id or predicted_arm == int(event.item_id):
                correct_predictions += 1
                # 使用历史奖励更新算法
                bandit.update(predicted_arm, event.reward, context)
                total_reward += event.reward
                
                # 创建模拟事件
                simulated_event = InteractionEvent(
                    event_id=f"sim_{event.event_id}",
                    user_id=event.user_id,
                    item_id=str(predicted_arm),
                    timestamp=event.timestamp,
                    action_type=event.action_type,
                    reward=event.reward,
                    context=context,
                    algorithm=policy_name
                )
                simulated_events.append(simulated_event)
        
        # 计算评估指标
        accuracy = correct_predictions / len(historical_events) if historical_events else 0.0
        avg_reward = total_reward / len(historical_events) if historical_events else 0.0
        
        # 计算其他指标
        metrics = EvaluationMetrics(
            average_reward=avg_reward,
            cumulative_reward=total_reward,
            coverage=self.calculator.calculate_coverage(simulated_events, bandit.n_arms),
            diversity=self.calculator.calculate_diversity(simulated_events)
        )
        
        result = {
            "policy_name": policy_name,
            "accuracy": accuracy,
            "num_events": len(historical_events),
            "correct_predictions": correct_predictions,
            "total_reward": total_reward,
            "metrics": asdict(metrics),
            "bandit_stats": bandit.get_performance_metrics()
        }
        
        self.replay_results[policy_name] = result
        return result
    
    def compare_policies(
        self,
        bandits: Dict[str, MultiArmedBandit],
        historical_events: List[InteractionEvent]
    ) -> Dict[str, Any]:
        """
        比较多个策略的离线性能
        
        Args:
            bandits: 算法字典
            historical_events: 历史事件
            
        Returns:
            比较结果
        """
        comparison_results = {}
        
        for policy_name, bandit in bandits.items():
            result = self.replay_evaluation(bandit, historical_events, policy_name)
            comparison_results[policy_name] = result
        
        # 排序结果
        sorted_policies = sorted(
            comparison_results.items(),
            key=lambda x: x[1]["metrics"]["average_reward"],
            reverse=True
        )
        
        return {
            "comparison_results": comparison_results,
            "ranking": [policy for policy, _ in sorted_policies],
            "best_policy": sorted_policies[0][0] if sorted_policies else None,
            "evaluation_summary": {
                "num_policies": len(bandits),
                "num_events": len(historical_events),
                "best_avg_reward": sorted_policies[0][1]["metrics"]["average_reward"] if sorted_policies else 0.0
            }
        }

class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self):
        """初始化A/B测试管理器"""
        self.active_experiments = {}
        self.experiment_results = {}
        self.calculator = MetricCalculator()
    
    def create_experiment(
        self,
        experiment_name: str,
        bandits: Dict[str, MultiArmedBandit],
        traffic_split: Dict[str, float],
        start_time: datetime,
        end_time: datetime,
        min_sample_size: int = 100
    ) -> str:
        """
        创建A/B测试实验
        
        Args:
            experiment_name: 实验名称
            bandits: 参与测试的算法
            traffic_split: 流量分配比例
            start_time: 开始时间
            end_time: 结束时间
            min_sample_size: 最小样本量
            
        Returns:
            实验ID
        """
        experiment_id = str(uuid.uuid4())
        
        # 验证流量分配
        total_traffic = sum(traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"流量分配总和必须为1.0，当前为{total_traffic}")
        
        experiment = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "bandits": bandits,
            "traffic_split": traffic_split,
            "start_time": start_time,
            "end_time": end_time,
            "min_sample_size": min_sample_size,
            "status": "active",
            "events": {variant: [] for variant in bandits.keys()},
            "created_at": utc_now()
        }
        
        self.active_experiments[experiment_id] = experiment
        return experiment_id
    
    def assign_user_to_variant(self, experiment_id: str, user_id: str) -> Optional[str]:
        """
        将用户分配到实验变体
        
        Args:
            experiment_id: 实验ID
            user_id: 用户ID
            
        Returns:
            分配的变体名称
        """
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        # 检查实验是否在活动时间内
        now = utc_now()
        if now < experiment["start_time"] or now > experiment["end_time"]:
            return None
        
        # 使用用户ID的哈希值进行一致性分配
        user_hash = hash(user_id) % 10000 / 10000.0
        
        cumulative_prob = 0.0
        for variant, prob in experiment["traffic_split"].items():
            cumulative_prob += prob
            if user_hash <= cumulative_prob:
                return variant
        
        # 默认分配到第一个变体
        return list(experiment["traffic_split"].keys())[0]
    
    def log_interaction(
        self,
        experiment_id: str,
        variant: str,
        event: InteractionEvent
    ):
        """
        记录实验交互
        
        Args:
            experiment_id: 实验ID
            variant: 变体名称
            event: 交互事件
        """
        if experiment_id not in self.active_experiments:
            return
        
        event.experiment_id = experiment_id
        event.algorithm = variant
        self.active_experiments[experiment_id]["events"][variant].append(event)
    
    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验结果
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验结果
        """
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        results = {}
        
        for variant, events in experiment["events"].items():
            if not events:
                results[variant] = {"metrics": EvaluationMetrics(), "sample_size": 0}
                continue
            
            # 计算指标
            metrics = EvaluationMetrics(
                click_through_rate=self.calculator.calculate_ctr(events),
                conversion_rate=self.calculator.calculate_conversion_rate(events),
                average_reward=np.mean([e.reward for e in events]),
                cumulative_reward=sum(e.reward for e in events),
                coverage=self.calculator.calculate_coverage(events, 1000),  # 假设1000个物品
                diversity=self.calculator.calculate_diversity(events)
            )
            
            results[variant] = {
                "metrics": asdict(metrics),
                "sample_size": len(events),
                "bandit_stats": experiment["bandits"][variant].get_performance_metrics()
            }
        
        # 统计显著性测试（简单t检验）
        significance_results = self._calculate_significance(experiment["events"])
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment["experiment_name"],
            "status": experiment["status"],
            "start_time": experiment["start_time"].isoformat(),
            "end_time": experiment["end_time"].isoformat(),
            "traffic_split": experiment["traffic_split"],
            "results": results,
            "significance": significance_results,
            "recommendation": self._generate_recommendation(results, significance_results)
        }
    
    def _calculate_significance(self, variant_events: Dict[str, List[InteractionEvent]]) -> Dict[str, Any]:
        """计算统计显著性"""
        from scipy import stats
        
        if len(variant_events) < 2:
            return {"significant": False, "p_value": 1.0}
        
        # 获取所有变体的奖励
        variant_rewards = {}
        for variant, events in variant_events.items():
            variant_rewards[variant] = [e.reward for e in events]
        
        # 两两比较（简化版本，只比较前两个变体）
        variants = list(variant_rewards.keys())
        if len(variants) >= 2:
            rewards_a = variant_rewards[variants[0]]
            rewards_b = variant_rewards[variants[1]]
            
            if len(rewards_a) > 1 and len(rewards_b) > 1:
                t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)
                return {
                    "significant": p_value < 0.05,
                    "p_value": p_value,
                    "t_statistic": t_stat,
                    "compared_variants": [variants[0], variants[1]]
                }
        
        return {"significant": False, "p_value": 1.0}
    
    def _generate_recommendation(
        self,
        results: Dict[str, Any],
        significance: Dict[str, Any]
    ) -> str:
        """生成实验建议"""
        if not results:
            return "暂无足够数据"
        
        # 找到平均奖励最高的变体
        best_variant = max(
            results.keys(),
            key=lambda v: results[v]["metrics"]["average_reward"]
        )
        
        best_reward = results[best_variant]["metrics"]["average_reward"]
        sample_size = results[best_variant]["sample_size"]
        
        if significance.get("significant", False):
            return f"建议采用变体 {best_variant}（平均奖励: {best_reward:.3f}，样本量: {sample_size}，统计显著）"
        else:
            return f"变体 {best_variant} 表现最好（平均奖励: {best_reward:.3f}），但差异不显著，建议继续收集数据"
    
    def end_experiment(self, experiment_id: str) -> bool:
        """
        结束实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否成功结束
        """
        if experiment_id not in self.active_experiments:
            return False
        
        self.active_experiments[experiment_id]["status"] = "completed"
        self.active_experiments[experiment_id]["end_time"] = utc_now()
        
        # 保存最终结果
        final_results = self.get_experiment_results(experiment_id)
        self.experiment_results[experiment_id] = final_results
        
        return True
    
    def list_active_experiments(self) -> List[Dict[str, Any]]:
        """列出所有活动实验"""
        active_experiments = []
        
        for exp_id, experiment in self.active_experiments.items():
            if experiment["status"] == "active":
                active_experiments.append({
                    "experiment_id": exp_id,
                    "experiment_name": experiment["experiment_name"],
                    "start_time": experiment["start_time"].isoformat(),
                    "end_time": experiment["end_time"].isoformat(),
                    "variants": list(experiment["traffic_split"].keys()),
                    "total_events": sum(len(events) for events in experiment["events"].values())
                })
        
        return active_experiments
