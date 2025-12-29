"""
智能体监控系统使用示例

演示如何使用监控包装器记录智能体行为和性能指标
"""

import asyncio
import numpy as np
from typing import List
from .tabular_qlearning_agent import TabularQLearningAgent
from .agent_monitor import (

    MonitoredQLearningAgent, 
    get_agent_monitor,
    initialize_agent_monitoring,
    shutdown_agent_monitoring,
    LogLevel
)
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class SimpleGridEnvironment:
    """简单的网格世界环境用于演示"""
    
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.state_space = grid_size * grid_size
        self.action_space = 4  # 上下左右
        self.goal_position = (grid_size - 1, grid_size - 1)
        self.current_position = (0, 0)
        
        # 动作映射：上(0), 右(1), 下(2), 左(3)
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1)   # 左
        }
    
    def reset(self):
        """重置环境"""
        self.current_position = (0, 0)
        return self._get_state()
    
    def step(self, action: int):
        """执行动作"""
        # 计算新位置
        dy, dx = self.action_effects[action]
        new_y = max(0, min(self.grid_size - 1, self.current_position[0] + dy))
        new_x = max(0, min(self.grid_size - 1, self.current_position[1] + dx))
        
        self.current_position = (new_y, new_x)
        
        # 计算奖励
        if self.current_position == self.goal_position:
            reward = 10.0
            done = True
        else:
            # 距离目标越近奖励越高
            distance = abs(self.current_position[0] - self.goal_position[0]) + \
                      abs(self.current_position[1] - self.goal_position[1])
            reward = -0.1 - distance * 0.01
            done = False
        
        return self._get_state(), reward, done
    
    def _get_state(self):
        """获取当前状态"""
        # 将2D位置转换为1D状态
        return self.current_position[0] * self.grid_size + self.current_position[1]

async def run_monitoring_demo():
    """运行智能体监控演示"""
    logger.info("智能体监控系统演示")
    
    # 1. 启动监控系统
    logger.info("启动监控系统")
    await initialize_agent_monitoring()
    monitor = get_agent_monitor()
    logger.info("监控状态", is_monitoring=monitor.is_monitoring)
    
    # 2. 创建环境和智能体
    logger.info("创建环境和智能体")
    env = SimpleGridEnvironment(grid_size=4)
    
    # 创建基础智能体
    base_agent = TabularQLearningAgent(
        state_size=env.state_space,
        action_size=env.action_space,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=0.3
    )
    
    # 包装为监控智能体
    monitored_agent = MonitoredQLearningAgent(base_agent, monitor)
    monitored_agent.current_episode = 0
    monitored_agent.current_step = 0
    
    logger.info("智能体信息", agent_id=monitored_agent.agent_id)
    logger.info("环境空间", state_space=env.state_space, action_space=env.action_space)
    
    # 3. 运行训练演示
    logger.info("开始训练演示", episodes=10)
    
    total_rewards = []
    
    for episode in range(10):
        monitored_agent.current_episode = episode
        
        state = env.reset()
        episode_reward = 0
        step_count = 0
        
        # 记录回合开始事件
        monitored_agent.log_event(
            "episode_start",
            f"开始第 {episode + 1} 回合",
            LogLevel.INFO,
            {"episode": episode, "initial_state": state}
        )
        
        while step_count < 50:  # 最多50步
            monitored_agent.current_step = step_count
            
            # 选择动作
            action = monitored_agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 学习
            monitored_agent.learn(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            step_count += 1
            
            if done:
                # 记录成功到达目标事件
                monitored_agent.log_event(
                    "goal_reached",
                    f"成功到达目标！用时 {step_count} 步",
                    LogLevel.INFO,
                    {"steps": step_count, "reward": episode_reward}
                )
                break
        
        total_rewards.append(episode_reward)
        
        # 记录性能指标
        monitored_agent.log_performance_metric("episode_reward", episode_reward)
        monitored_agent.log_performance_metric("episode_steps", step_count)
        monitored_agent.log_performance_metric("exploration_rate", monitored_agent.exploration_rate)
        
        # 记录回合结束事件
        monitored_agent.log_event(
            "episode_end",
            f"回合 {episode + 1} 结束，奖励: {episode_reward:.2f}",
            LogLevel.INFO,
            {
                "episode": episode,
                "reward": episode_reward,
                "steps": step_count,
                "success": done
            }
        )
        
        logger.info(
            "回合结果",
            episode=episode + 1,
            reward=round(episode_reward, 2),
            steps=step_count,
        )
        
        # 每5个回合显示一次统计
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(total_rewards[-5:])
            monitored_agent.log_performance_metric("avg_reward_5", avg_reward)
            logger.info("最近5回合平均奖励", avg_reward=round(avg_reward, 2))
    
    # 4. 显示监控统计
    logger.info("监控统计摘要")
    summary = monitor.get_agent_summary(monitored_agent.agent_id)
    
    if summary:
        logger.info("动作统计", total_actions=summary.get("total_actions", 0))
        logger.info("探索动作", exploration_count=summary.get("exploration_count", 0))
        logger.info("利用动作", exploitation_count=summary.get("exploitation_count", 0))
        logger.info("平均奖励", average_reward=round(summary.get("average_reward", 0), 3))
        logger.info(
            "平均决策时间",
            avg_decision_time_ms=round(summary.get("avg_decision_time", 0), 2),
        )
        
        if 'event_counts' in summary:
            logger.info("事件统计", event_counts=dict(summary["event_counts"]))
        
        if 'metrics_summary' in summary:
            logger.info("性能指标摘要")
            for metric, data in summary['metrics_summary'].items():
                logger.info(
                    "性能指标",
                    metric=metric,
                    current=round(data["current"], 2),
                    average=round(data["average"], 2),
                    trend=data["trend"],
                )
    
    # 5. 显示最近动作历史
    logger.info("最近动作历史", limit=5)
    recent_actions = monitor.get_recent_actions(monitored_agent.agent_id, 5)
    
    for i, action_data in enumerate(recent_actions[:5]):
        logger.info(
            "动作记录",
            index=i + 1,
            state=action_data["state"],
            action=action_data["action"],
            action_type=action_data["action_type"],
            reward=round(action_data["reward"], 2),
        )
    
    # 6. 显示性能趋势
    logger.info("回合奖励趋势")
    trend_data = monitor.get_performance_trend(monitored_agent.agent_id, "episode_reward", 1)
    
    if trend_data:
        logger.info("时间奖励趋势")
        for data in trend_data[-5:]:  # 显示最后5个数据点
            logger.info(
                "趋势数据点",
                timestamp=data["timestamp"].strftime("%H:%M:%S"),
                value=round(data["value"], 2),
            )
    
    # 7. 全局监控统计
    logger.info("全局监控统计")
    logger.info("监控状态", is_monitoring=monitor.is_monitoring)
    logger.info(
        "缓冲区大小",
        action=len(monitor.action_buffer),
        decision=len(monitor.decision_buffer),
        performance=len(monitor.performance_buffer),
        event=len(monitor.event_buffer),
    )
    logger.info("监控的智能体数量", total=len(monitor.agent_stats))
    
    # 8. 关闭监控系统
    logger.info("关闭监控系统")
    await shutdown_agent_monitoring()
    logger.info("监控系统已关闭")
    
    logger.info("演示完成")

async def run_multi_agent_monitoring_demo():
    """多智能体监控演示"""
    logger.info("多智能体监控演示")
    
    await initialize_agent_monitoring()
    monitor = get_agent_monitor()
    
    # 创建多个智能体
    agents = []
    for i in range(3):
        base_agent = TabularQLearningAgent(
            state_size=16,
            action_size=4,
            learning_rate=0.1 + i * 0.05,  # 不同的学习率
            discount_factor=0.95 + i * 0.01,
            exploration_rate=0.2 + i * 0.1
        )
        monitored_agent = MonitoredQLearningAgent(base_agent, monitor)
        agents.append(monitored_agent)
        
        logger.info("智能体创建", index=i + 1, agent_id=monitored_agent.agent_id)
    
    # 同时运行多个智能体
    env = SimpleGridEnvironment(grid_size=4)
    
    for episode in range(5):
        logger.info("回合开始", episode=episode + 1)
        
        for i, agent in enumerate(agents):
            agent.current_episode = episode
            
            state = env.reset()
            episode_reward = 0
            
            for step in range(20):
                agent.current_step = step
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            agent.log_performance_metric("episode_reward", episode_reward)
            logger.info(
                "智能体回合奖励",
                agent_index=i + 1,
                reward=round(episode_reward, 2),
            )
    
    # 显示所有智能体的统计
    logger.info("多智能体统计摘要")
    for i, agent in enumerate(agents):
        summary = monitor.get_agent_summary(agent.agent_id)
        logger.info("智能体统计", agent_index=i + 1)
        logger.info("总动作", total_actions=summary.get("total_actions", 0))
        logger.info("平均奖励", average_reward=round(summary.get("average_reward", 0), 3))
    
    await shutdown_agent_monitoring()
    logger.info("多智能体演示完成")

if __name__ == "__main__":
    # 运行单智能体演示
    setup_logging()
    asyncio.run(run_monitoring_demo())
    
    # 运行多智能体演示
    asyncio.run(run_multi_agent_monitoring_demo())
