"""
经典Q-Learning算法实现

基于表格的Q-Learning算法，适用于离散状态和动作空间。
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .base import (
    QLearningAgent, 
    AgentState, 
    Experience, 
    QLearningConfig,
    AlgorithmType,
    EpsilonGreedyStrategy
)


class ClassicQLearningAgent(QLearningAgent):
    """经典Q-Learning智能体实现"""
    
    def __init__(self, agent_id: str, state_size: int, action_size: int, config: QLearningConfig, action_space: List[str]):
        # 确保是经典Q-Learning算法
        config.algorithm_type = AlgorithmType.Q_LEARNING
        super().__init__(agent_id, state_size, action_size, config)
        
        self.action_space = action_space
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.state_visit_count: Dict[str, int] = defaultdict(int)
        self.state_action_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.exploration_strategy = EpsilonGreedyStrategy()
        
        # 初始化Q表（可选）
        self._initialize_q_table()
    
    def _initialize_q_table(self) -> None:
        """初始化Q表，所有Q值设为0"""
        pass  # Q表使用defaultdict，会自动初始化为0
    
    def _state_to_key(self, state: AgentState) -> str:
        """将状态转换为Q表的键"""
        # 将特征向量转换为可哈希的键
        if isinstance(state.features, dict):
            # 对特征进行排序以确保一致性
            sorted_features = sorted(state.features.items())
            return str(tuple((k, round(v, 6)) for k, v in sorted_features))
        else:
            # 如果是列表或数组，转换为元组
            return str(tuple(round(x, 6) for x in state.features.values()))
    
    def get_action(self, state: AgentState, exploration: bool = True) -> str:
        """使用epsilon-greedy策略选择动作"""
        state_key = self._state_to_key(state)
        self.state_visit_count[state_key] += 1
        
        if not exploration:
            # 贪婪策略，选择Q值最高的动作
            return self._get_greedy_action(state_key)
        
        # 使用exploration策略
        q_values = dict(self.q_table[state_key])
        
        # 确保所有动作都在Q值字典中
        for action in self.action_space:
            if action not in q_values:
                q_values[action] = 0.0
        
        return self.exploration_strategy.select_action(q_values, self.action_space, self.epsilon)
    
    def _get_greedy_action(self, state_key: str) -> str:
        """获取贪婪动作"""
        if state_key not in self.q_table or not self.q_table[state_key]:
            return np.random.choice(self.action_space)
        
        # 找到Q值最高的动作
        q_values = self.q_table[state_key]
        max_q_value = max(q_values.values())
        best_actions = [action for action, q_val in q_values.items() if q_val == max_q_value]
        
        return np.random.choice(best_actions)
    
    def update_q_value(self, experience: Experience) -> Optional[float]:
        """使用Q-Learning更新规则更新Q值"""
        state_key = self._state_to_key(experience.state)
        next_state_key = self._state_to_key(experience.next_state)
        
        # 记录状态-动作访问次数
        self.state_action_count[state_key][experience.action] += 1
        
        # 获取当前Q值
        current_q = self.q_table[state_key][experience.action]
        
        # 计算目标Q值
        if experience.done:
            target_q = experience.reward
        else:
            # 找到下一个状态的最大Q值
            next_q_values = self.q_table[next_state_key]
            if next_q_values:
                max_next_q = max(next_q_values.values())
            else:
                max_next_q = 0.0
            target_q = experience.reward + self.config.discount_factor * max_next_q
        
        # Q-Learning更新公式
        td_error = target_q - current_q
        self.q_table[state_key][experience.action] = current_q + self.config.learning_rate * td_error
        
        # 记录经验
        self.add_experience(experience)
        
        # 返回TD误差的绝对值作为损失
        return abs(td_error)
    
    def get_q_values(self, state: AgentState) -> Dict[str, float]:
        """获取状态的所有Q值"""
        state_key = self._state_to_key(state)
        q_values = dict(self.q_table[state_key])
        
        # 确保所有动作都有Q值
        for action in self.action_space:
            if action not in q_values:
                q_values[action] = 0.0
                
        return q_values
    
    def get_value_function(self, state: AgentState) -> float:
        """获取状态价值函数 V(s) = max_a Q(s,a)"""
        q_values = self.get_q_values(state)
        return max(q_values.values()) if q_values else 0.0
    
    def get_policy(self, state: AgentState) -> Dict[str, float]:
        """获取当前策略概率分布"""
        q_values = self.get_q_values(state)
        
        if not q_values:
            # 均匀随机策略
            prob = 1.0 / len(self.action_space)
            return {action: prob for action in self.action_space}
        
        # epsilon-greedy策略的概率分布
        max_q = max(q_values.values())
        best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
        
        policy = {}
        for action in self.action_space:
            if action in best_actions:
                # 最优动作获得 (1-epsilon)/|A*| + epsilon/|A| 的概率
                policy[action] = (1.0 - self.epsilon) / len(best_actions) + self.epsilon / len(self.action_space)
            else:
                # 非最优动作获得 epsilon/|A| 的概率
                policy[action] = self.epsilon / len(self.action_space)
        
        return policy
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """获取状态访问统计"""
        total_visits = sum(self.state_visit_count.values())
        unique_states = len(self.state_visit_count)
        
        # 计算状态访问熵（探索程度指标）
        if total_visits > 0:
            visit_probs = [count / total_visits for count in self.state_visit_count.values()]
            state_entropy = -sum(p * np.log(p) for p in visit_probs if p > 0)
        else:
            state_entropy = 0.0
        
        return {
            "unique_states_visited": unique_states,
            "total_state_visits": total_visits,
            "state_visit_entropy": state_entropy,
            "q_table_size": len(self.q_table),
            "average_visits_per_state": total_visits / max(unique_states, 1)
        }
    
    def prune_q_table(self, min_visits: int = 1) -> int:
        """修剪Q表，移除访问次数少的状态"""
        states_to_remove = [
            state for state, count in self.state_visit_count.items() 
            if count < min_visits
        ]
        
        for state in states_to_remove:
            if state in self.q_table:
                del self.q_table[state]
            if state in self.state_visit_count:
                del self.state_visit_count[state]
            if state in self.state_action_count:
                del self.state_action_count[state]
        
        return len(states_to_remove)
    
    def save_model(self, filepath: str) -> None:
        """保存Q表和统计信息"""
        model_data = {
            "agent_id": self.agent_id,
            "config": {
                "algorithm_type": self.config.algorithm_type.value,
                "learning_rate": self.config.learning_rate,
                "discount_factor": self.config.discount_factor,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay": self.config.epsilon_decay,
            },
            "q_table": dict(self.q_table),
            "state_visit_count": dict(self.state_visit_count),
            "state_action_count": dict(self.state_action_count),
            "action_space": self.action_space,
            "current_epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "training_history": self.training_history
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """加载Q表和统计信息"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        # 恢复Q表
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in model_data["q_table"].items():
            for action, q_value in actions.items():
                self.q_table[state][action] = q_value
        
        # 恢复统计信息
        self.state_visit_count = defaultdict(int, model_data.get("state_visit_count", {}))
        
        # 恢复状态-动作计数
        self.state_action_count = defaultdict(lambda: defaultdict(int))
        for state, actions in model_data.get("state_action_count", {}).items():
            for action, count in actions.items():
                self.state_action_count[state][action] = count
        
        # 恢复训练状态
        self.action_space = model_data.get("action_space", self.action_space)
        self.epsilon = model_data.get("current_epsilon", self.config.epsilon_start)
        self.step_count = model_data.get("step_count", 0)
        self.episode_count = model_data.get("episode_count", 0)
        self.training_history = model_data.get("training_history", {"rewards": [], "losses": [], "epsilon_values": []})
    
    def export_q_table_csv(self, filepath: str) -> None:
        """导出Q表为CSV格式便于分析"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            header = ["state"] + self.action_space + ["visit_count", "max_q", "best_action"]
            writer.writerow(header)
            
            # 写入数据
            for state_key, q_values in self.q_table.items():
                row = [state_key]
                
                # Q值
                for action in self.action_space:
                    row.append(q_values.get(action, 0.0))
                
                # 统计信息
                visit_count = self.state_visit_count.get(state_key, 0)
                max_q = max(q_values.values()) if q_values else 0.0
                best_action = max(q_values.keys(), key=lambda a: q_values[a]) if q_values else "N/A"
                
                row.extend([visit_count, max_q, best_action])
                writer.writerow(row)
    
    def reset(self) -> None:
        """重置智能体，但保留已学习的Q表"""
        super().reset()
        # 不重置Q表，允许继续学习