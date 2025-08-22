"""
Q-Learning API端点实现

提供Q-Learning智能体的REST API接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging

from ...services.qlearning_service import QLearningService
from ...services.qlearning_strategy_service import (
    QLearningStrategyService, 
    StrategyInferenceRequest, 
    BatchInferenceRequest
)
from ...services.qlearning_recommendation_service import (
    QLearningRecommendationService,
    HybridRecommendationRequest,
    RecommendationContext,
    StrategyCombinationMode
)
from ...ai.qlearning.agent_monitor import (
    get_agent_monitor, MonitoredQLearningAgent, initialize_agent_monitoring
)

logger = logging.getLogger(__name__)

# 全局服务实例
qlearning_service = QLearningService()
strategy_service = QLearningStrategyService(qlearning_service)
recommendation_service = QLearningRecommendationService(
    qlearning_service, 
    strategy_service, 
    bandit_service=None  # 可以后续集成实际的bandit服务
)

router = APIRouter(prefix="/qlearning", tags=["Q-Learning"])


# Pydantic模型定义
class AgentConfig(BaseModel):
    """智能体配置"""
    agent_type: str = Field(default="tabular", description="智能体类型")
    state_size: int = Field(default=16, description="状态空间大小")
    action_size: int = Field(default=4, description="动作空间大小")
    learning_rate: float = Field(default=0.1, description="学习率")
    gamma: float = Field(default=0.99, description="折扣因子")
    epsilon: float = Field(default=0.1, description="探索率")
    epsilon_decay: float = Field(default=0.995, description="探索率衰减")
    epsilon_min: float = Field(default=0.01, description="最小探索率")
    
    environment: Dict[str, Any] = Field(default_factory=dict, description="环境配置")
    exploration: Dict[str, Any] = Field(default_factory=dict, description="探索策略配置")
    reward: Dict[str, Any] = Field(default_factory=dict, description="奖励函数配置")


class TrainingConfig(BaseModel):
    """训练配置"""
    max_episodes: int = Field(default=1000, description="最大训练回合数")
    max_steps_per_episode: int = Field(default=1000, description="每回合最大步数")
    evaluation_frequency: int = Field(default=100, description="评估频率")
    save_frequency: int = Field(default=500, description="保存频率")
    learning_rate: float = Field(default=0.001, description="学习率")
    early_stopping: bool = Field(default=True, description="是否启用早停")
    patience: int = Field(default=200, description="早停耐心值")


class PredictionRequest(BaseModel):
    """预测请求"""
    state: List[float] = Field(..., description="状态向量")


class EvaluationRequest(BaseModel):
    """评估请求"""
    num_episodes: int = Field(default=10, description="评估回合数")


class StrategyInferenceRequestModel(BaseModel):
    """策略推理请求模型"""
    state: List[float] = Field(..., description="状态向量")
    evaluation_mode: bool = Field(default=True, description="是否使用评估模式")
    return_q_values: bool = Field(default=True, description="是否返回Q值")
    return_confidence: bool = Field(default=True, description="是否返回置信度")
    context: Optional[Dict[str, Any]] = Field(default=None, description="额外上下文信息")


class BatchInferenceRequestModel(BaseModel):
    """批量推理请求模型"""
    states: List[List[float]] = Field(..., description="状态向量列表")
    evaluation_mode: bool = Field(default=True, description="是否使用评估模式")
    return_details: bool = Field(default=False, description="是否返回详细信息")


class StrategyComparisonRequest(BaseModel):
    """策略比较请求"""
    agent_ids: List[str] = Field(..., description="要比较的智能体ID列表")
    state: List[float] = Field(..., description="用于比较的状态")


class WarmupRequest(BaseModel):
    """预热请求"""
    num_warmup_states: int = Field(default=10, description="预热状态数量")


class HybridRecommendationRequestModel(BaseModel):
    """混合推荐请求模型"""
    state: List[float] = Field(..., description="当前状态向量")
    bandit_arm_id: Optional[str] = Field(default=None, description="Bandit臂ID")
    combination_mode: str = Field(default="weighted_average", description="组合模式")
    q_learning_weight: float = Field(default=0.5, description="Q-Learning权重")
    bandit_weight: float = Field(default=0.5, description="Bandit权重")
    fallback_to_random: bool = Field(default=True, description="是否允许随机fallback")
    context: Optional[Dict[str, Any]] = Field(default=None, description="推荐上下文")


class FeedbackRequest(BaseModel):
    """反馈请求"""
    decision_id: str = Field(..., description="决策ID")
    reward: float = Field(..., description="奖励值")
    success: Optional[bool] = Field(default=None, description="是否成功")


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    created_at: str
    agent_type: str
    status: str


# API端点
@router.get("/info")
async def get_algorithm_info():
    """获取Q-Learning算法信息"""
    try:
        return await qlearning_service.get_algorithm_info()
    except Exception as e:
        logger.error(f"获取算法信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents", response_model=SessionResponse)
async def create_agent(config: AgentConfig):
    """创建Q-Learning智能体"""
    try:
        session_id = await qlearning_service.create_agent_session(config.dict())
        
        session_info = await qlearning_service.get_session_info(session_id)
        
        return SessionResponse(
            session_id=session_id,
            created_at=session_info["created_at"],
            agent_type=session_info["agent_type"],
            status="created"
        )
        
    except Exception as e:
        logger.error(f"创建智能体失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents")
async def list_agents():
    """列出所有智能体会话"""
    try:
        return await qlearning_service.list_sessions()
    except Exception as e:
        logger.error(f"列出智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}")
async def get_agent_info(session_id: str):
    """获取智能体信息"""
    try:
        return await qlearning_service.get_session_info(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取智能体信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{session_id}")
async def delete_agent(session_id: str):
    """删除智能体会话"""
    try:
        success = await qlearning_service.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"message": "会话删除成功", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除智能体失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{session_id}/train")
async def start_training(session_id: str, config: TrainingConfig, background_tasks: BackgroundTasks):
    """开始训练智能体"""
    try:
        success = await qlearning_service.start_training(session_id, config.dict())
        if not success:
            raise HTTPException(status_code=400, detail="无法开始训练")
        
        return {
            "message": "训练已开始",
            "session_id": session_id,
            "config": config.dict()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"开始训练失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/agents/{session_id}/stop")
async def stop_training(session_id: str):
    """停止训练智能体"""
    try:
        success = await qlearning_service.stop_training(session_id)
        if not success:
            raise HTTPException(status_code=400, detail="无法停止训练")
        
        return {"message": "训练已停止", "session_id": session_id}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{session_id}/progress")
async def get_training_progress(session_id: str):
    """获取训练进度"""
    try:
        return await qlearning_service.get_training_progress(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取训练进度失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{session_id}/predict")
async def predict_action(session_id: str, request: PredictionRequest):
    """预测动作"""
    try:
        return await qlearning_service.predict_action(session_id, request.state)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"预测动作失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/agents/{session_id}/evaluate")
async def evaluate_policy(session_id: str, request: EvaluationRequest):
    """评估策略"""
    try:
        return await qlearning_service.evaluate_policy(session_id, request.num_episodes)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"策略评估失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{session_id}/statistics")
async def get_agent_statistics(session_id: str):
    """获取智能体统计信息"""
    try:
        session_info = await qlearning_service.get_session_info(session_id)
        
        # 提取统计信息
        statistics = {
            "session_info": {
                "session_id": session_info["session_id"],
                "created_at": session_info["created_at"],
                "last_updated": session_info["last_updated"],
                "is_training": session_info["is_training"]
            },
            "agent_info": {
                "agent_type": session_info["agent_type"],
                "state_size": session_info["state_size"],
                "action_size": session_info["action_size"]
            },
            "exploration": session_info.get("exploration", {}),
            "training_metrics": session_info.get("training_metrics", {}),
            "agent_statistics": session_info.get("agent_statistics", {})
        }
        
        return statistics
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 批量操作端点
@router.post("/batch/create")
async def create_multiple_agents(configs: List[AgentConfig]):
    """批量创建智能体"""
    try:
        results = []
        for config in configs:
            session_id = await qlearning_service.create_agent_session(config.dict())
            session_info = await qlearning_service.get_session_info(session_id)
            
            results.append({
                "session_id": session_id,
                "created_at": session_info["created_at"],
                "agent_type": session_info["agent_type"],
                "status": "created"
            })
        
        return {
            "message": f"成功创建 {len(results)} 个智能体",
            "agents": results
        }
        
    except Exception as e:
        logger.error(f"批量创建智能体失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/batch/cleanup")
async def cleanup_sessions():
    """清理所有会话"""
    try:
        sessions = await qlearning_service.list_sessions()
        deleted_count = 0
        
        for session in sessions:
            success = await qlearning_service.delete_session(session["session_id"])
            if success:
                deleted_count += 1
        
        return {
            "message": f"清理完成，删除了 {deleted_count} 个会话"
        }
        
    except Exception as e:
        logger.error(f"清理会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 配置模板端点
@router.get("/templates/configs")
async def get_config_templates():
    """获取配置模板"""
    return {
        "agent_templates": {
            "simple_tabular": {
                "agent_type": "tabular",
                "state_size": 16,
                "action_size": 4,
                "learning_rate": 0.1,
                "gamma": 0.99,
                "epsilon": 0.1,
                "environment": {
                    "type": "grid_world",
                    "grid_size": [4, 4],
                    "start_position": [0, 0],
                    "goal_position": [3, 3],
                    "obstacles": []
                },
                "exploration": {
                    "mode": "decaying_epsilon",
                    "initial_exploration": 1.0,
                    "final_exploration": 0.01,
                    "decay_steps": 5000
                },
                "reward": {
                    "type": "step",
                    "parameters": {
                        "positive_reward": 10.0,
                        "negative_reward": -1.0,
                        "neutral_reward": -0.01
                    }
                }
            },
            "dqn_continuous": {
                "agent_type": "dqn",
                "state_size": 4,
                "action_size": 2,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "environment": {
                    "type": "continuous",
                    "state_bounds": [[-10, 10], [-10, 10], [-1, 1], [-1, 1]]
                },
                "exploration": {
                    "mode": "decaying_epsilon",
                    "initial_exploration": 1.0,
                    "final_exploration": 0.01,
                    "decay_steps": 10000
                },
                "reward": {
                    "type": "gaussian",
                    "parameters": {
                        "target_value": 0.0,
                        "sigma": 1.0,
                        "amplitude": 1.0
                    }
                }
            }
        },
        "training_templates": {
            "quick_test": {
                "max_episodes": 100,
                "max_steps_per_episode": 200,
                "evaluation_frequency": 20,
                "early_stopping": True,
                "patience": 50
            },
            "full_training": {
                "max_episodes": 2000,
                "max_steps_per_episode": 1000,
                "evaluation_frequency": 100,
                "early_stopping": True,
                "patience": 200
            }
        }
    }


# WebSocket支持
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = []
            self.session_connections[session_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, session_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if session_id and session_id in self.session_connections:
            if websocket in self.session_connections[session_id]:
                self.session_connections[session_id].remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
    
    async def broadcast_to_session(self, message: str, session_id: str):
        if session_id in self.session_connections:
            disconnected = []
            for connection in self.session_connections[session_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"广播消息失败: {e}")
                    disconnected.append(connection)
            
            # 清理断开的连接
            for connection in disconnected:
                self.session_connections[session_id].remove(connection)

manager = ConnectionManager()


@router.websocket("/agents/{session_id}/monitor")
async def websocket_training_monitor(websocket: WebSocket, session_id: str):
    """WebSocket实时训练监控"""
    await manager.connect(websocket, session_id)
    
    try:
        # 发送连接确认
        await manager.send_personal_message(
            json.dumps({
                "type": "connected",
                "session_id": session_id,
                "timestamp": asyncio.get_event_loop().time()
            }), 
            websocket
        )
        
        while True:
            try:
                # 获取训练进度
                progress = await qlearning_service.get_training_progress(session_id)
                
                # 发送进度更新
                message = {
                    "type": "progress_update",
                    "session_id": session_id,
                    "data": progress,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                await manager.send_personal_message(
                    json.dumps(message), 
                    websocket
                )
                
            except ValueError:
                # 会话不存在
                error_message = {
                    "type": "error",
                    "message": "会话不存在",
                    "session_id": session_id
                }
                await manager.send_personal_message(
                    json.dumps(error_message), 
                    websocket
                )
                break
            except Exception as e:
                logger.error(f"WebSocket监控错误: {e}")
                error_message = {
                    "type": "error", 
                    "message": str(e),
                    "session_id": session_id
                }
                await manager.send_personal_message(
                    json.dumps(error_message), 
                    websocket
                )
            
            # 每2秒更新一次
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        manager.disconnect(websocket, session_id)


@router.get("/health")
async def health_check():
    """Q-Learning服务健康检查"""
    try:
        sessions = await qlearning_service.list_sessions()
        active_sessions = len(sessions)
        training_sessions = len([s for s in sessions if s.get("is_training", False)])
        
        return {
            "status": "healthy",
            "service": "qlearning",
            "active_sessions": active_sessions,
            "training_sessions": training_sessions,
            "websocket_connections": len(manager.active_connections),
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="Q-Learning服务不可用")


@router.get("/environments")
async def get_supported_environments():
    """获取支持的环境类型"""
    return {
        "environments": [
            {
                "type": "grid_world",
                "name": "网格世界环境",
                "description": "经典的网格世界环境，适合测试基础强化学习算法",
                "parameters": {
                    "grid_size": {
                        "type": "tuple",
                        "description": "网格大小，如(4,4)",
                        "default": [4, 4],
                        "range": [[2, 2], [10, 10]]
                    },
                    "start_position": {
                        "type": "tuple", 
                        "description": "智能体起始位置",
                        "default": [0, 0]
                    },
                    "goal_position": {
                        "type": "tuple",
                        "description": "目标位置",
                        "default": [3, 3]
                    },
                    "obstacles": {
                        "type": "list",
                        "description": "障碍物位置列表",
                        "default": [],
                        "example": [[1, 1], [2, 2]]
                    }
                }
            }
        ]
    }


@router.get("/exploration_strategies")
async def get_exploration_strategies():
    """获取支持的探索策略"""
    return {
        "strategies": [
            {
                "mode": "epsilon_greedy",
                "name": "Epsilon贪婪策略",
                "description": "以epsilon概率随机探索，否则贪婪选择最优动作",
                "parameters": {
                    "initial_exploration": {
                        "type": "float",
                        "description": "初始探索率",
                        "default": 1.0,
                        "range": [0.0, 1.0]
                    }
                }
            },
            {
                "mode": "decaying_epsilon",
                "name": "衰减Epsilon贪婪策略", 
                "description": "epsilon值随时间衰减的贪婪策略",
                "parameters": {
                    "initial_exploration": {
                        "type": "float",
                        "description": "初始探索率",
                        "default": 1.0,
                        "range": [0.0, 1.0]
                    },
                    "final_exploration": {
                        "type": "float",
                        "description": "最终探索率",
                        "default": 0.01,
                        "range": [0.0, 1.0]
                    },
                    "decay_steps": {
                        "type": "int",
                        "description": "衰减步数",
                        "default": 10000,
                        "range": [100, 100000]
                    }
                }
            },
            {
                "mode": "upper_confidence_bound",
                "name": "上置信界策略",
                "description": "基于不确定性的探索策略，选择置信上界最大的动作",
                "parameters": {
                    "ucb_c": {
                        "type": "float",
                        "description": "置信界系数",
                        "default": 2.0,
                        "range": [0.1, 10.0]
                    }
                }
            }
        ]
    }


@router.get("/reward_functions")
async def get_reward_functions():
    """获取支持的奖励函数"""
    return {
        "reward_functions": [
            {
                "type": "step",
                "name": "步长奖励",
                "description": "每步给予固定奖励值",
                "parameters": {
                    "positive_reward": {
                        "type": "float",
                        "description": "正向奖励（到达目标）",
                        "default": 10.0
                    },
                    "negative_reward": {
                        "type": "float", 
                        "description": "负向奖励（撞墙/障碍）",
                        "default": -1.0
                    },
                    "neutral_reward": {
                        "type": "float",
                        "description": "中性奖励（普通移动）",
                        "default": -0.01
                    }
                }
            },
            {
                "type": "linear",
                "name": "线性奖励",
                "description": "基于状态特征线性组合计算奖励",
                "parameters": {
                    "coefficients": {
                        "type": "list",
                        "description": "线性系数向量",
                        "default": [1.0, 0.0]
                    },
                    "bias": {
                        "type": "float",
                        "description": "偏置项",
                        "default": 0.0
                    }
                }
            },
            {
                "type": "threshold",
                "name": "阈值奖励",
                "description": "基于阈值判断给予不同奖励",
                "parameters": {
                    "threshold": {
                        "type": "float",
                        "description": "阈值",
                        "default": 0.5
                    },
                    "reward_above": {
                        "type": "float",
                        "description": "超过阈值的奖励",
                        "default": 1.0
                    },
                    "reward_below": {
                        "type": "float",
                        "description": "低于阈值的奖励", 
                        "default": -1.0
                    }
                }
            },
            {
                "type": "gaussian",
                "name": "高斯奖励",
                "description": "基于高斯分布的连续奖励函数",
                "parameters": {
                    "mean": {
                        "type": "float",
                        "description": "高斯分布均值",
                        "default": 0.0
                    },
                    "std": {
                        "type": "float", 
                        "description": "高斯分布标准差",
                        "default": 1.0
                    },
                    "max_reward": {
                        "type": "float",
                        "description": "最大奖励值",
                        "default": 1.0
                    }
                }
            }
        ]
    }


# =============================================================================
# 策略推理API端点
# =============================================================================

@router.post("/agents/{session_id}/inference", summary="策略推理")
async def strategy_inference(session_id: str, request: StrategyInferenceRequestModel):
    """对单个状态进行策略推理"""
    try:
        inference_request = StrategyInferenceRequest(
            agent_id=session_id,
            state=request.state,
            evaluation_mode=request.evaluation_mode,
            return_q_values=request.return_q_values,
            return_confidence=request.return_confidence,
            context=request.context
        )
        
        response = await strategy_service.single_inference(inference_request)
        
        return {
            "success": True,
            "agent_id": response.agent_id,
            "action": response.action,
            "action_name": response.action_name,
            "q_values": response.q_values,
            "confidence_score": response.confidence_score,
            "exploration_info": response.exploration_info,
            "inference_time_ms": response.inference_time_ms,
            "timestamp": response.timestamp.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"策略推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{session_id}/batch_inference", summary="批量策略推理")
async def batch_strategy_inference(session_id: str, request: BatchInferenceRequestModel):
    """对多个状态进行批量策略推理"""
    try:
        batch_request = BatchInferenceRequest(
            agent_id=session_id,
            states=request.states,
            evaluation_mode=request.evaluation_mode,
            return_details=request.return_details
        )
        
        response = await strategy_service.batch_inference(batch_request)
        
        result = {
            "success": True,
            "agent_id": response.agent_id,
            "actions": response.actions,
            "total_inference_time_ms": response.total_inference_time_ms,
            "average_inference_time_ms": response.average_inference_time_ms,
            "num_states": len(request.states)
        }
        
        if response.details:
            result["details"] = [
                {
                    "action": detail.action,
                    "action_name": detail.action_name,
                    "q_values": detail.q_values,
                    "confidence_score": detail.confidence_score,
                    "inference_time_ms": detail.inference_time_ms
                }
                for detail in response.details
            ]
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"批量策略推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/strategies/compare", summary="策略比较")
async def compare_strategies(request: StrategyComparisonRequest):
    """比较多个智能体在同一状态下的策略"""
    try:
        comparison = await strategy_service.compare_strategies(request.agent_ids, request.state)
        
        return {
            "success": True,
            "comparison": {
                "agent_ids": comparison.agent_ids,
                "state": comparison.state,
                "best_agent_id": comparison.best_agent_id,
                "recommendations": comparison.recommendations,
                "comparison_metrics": comparison.comparison_metrics
            }
        }
        
    except Exception as e:
        logger.error(f"策略比较失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}/insights", summary="策略洞察分析")
async def get_strategy_insights(session_id: str, num_states: int = 100):
    """获取智能体的策略洞察分析"""
    try:
        insights = await strategy_service.get_strategy_insights(session_id, num_states)
        
        return {
            "success": True,
            "insights": insights
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"策略洞察分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}/performance", summary="推理性能指标")
async def get_inference_performance(session_id: str):
    """获取智能体的推理性能指标"""
    try:
        metrics = await strategy_service.get_performance_metrics(session_id)
        
        return {
            "success": True,
            "performance_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"获取推理性能指标失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{session_id}/warmup", summary="智能体预热")
async def warmup_agent(session_id: str, request: WarmupRequest):
    """预热智能体以提高推理性能"""
    try:
        warmup_result = await strategy_service.warmup_agent(session_id, request.num_warmup_states)
        
        return {
            "success": True,
            "warmup": warmup_result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"智能体预热失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{session_id}/cache", summary="清理推理缓存")
async def clear_inference_cache(session_id: str):
    """清理指定智能体的推理缓存"""
    try:
        await strategy_service.clear_cache(session_id)
        
        return {
            "success": True,
            "message": f"已清理智能体 {session_id} 的推理缓存"
        }
        
    except Exception as e:
        logger.error(f"清理推理缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache", summary="清理所有推理缓存")
async def clear_all_inference_cache():
    """清理所有智能体的推理缓存"""
    try:
        await strategy_service.clear_cache()
        
        return {
            "success": True,
            "message": "已清理所有推理缓存"
        }
        
    except Exception as e:
        logger.error(f"清理所有推理缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 推荐协调API端点
# =============================================================================

@router.post("/agents/{session_id}/hybrid_recommendation", summary="混合推荐")
async def hybrid_recommendation(session_id: str, request: HybridRecommendationRequestModel):
    """Q-Learning与推荐引擎的混合推荐"""
    try:
        # 构建推荐上下文
        context = None
        if request.context:
            context = RecommendationContext(
                user_id=request.context.get("user_id"),
                session_id=request.context.get("session_id"),
                item_features=request.context.get("item_features"),
                user_features=request.context.get("user_features"), 
                context_features=request.context.get("context_features"),
                additional_info=request.context.get("additional_info")
            )
        
        # 构建推荐请求
        hybrid_request = HybridRecommendationRequest(
            agent_id=session_id,
            bandit_arm_id=request.bandit_arm_id,
            state=request.state,
            context=context,
            combination_mode=StrategyCombinationMode(request.combination_mode),
            q_learning_weight=request.q_learning_weight,
            bandit_weight=request.bandit_weight,
            fallback_to_random=request.fallback_to_random
        )
        
        response = await recommendation_service.hybrid_recommendation(hybrid_request)
        
        return {
            "success": True,
            "recommended_action": response.recommended_action,
            "action_name": response.action_name,
            "confidence_score": response.confidence_score,
            "decision_source": response.decision_source.value,
            "q_learning_result": response.q_learning_result,
            "bandit_result": response.bandit_result,
            "combination_details": response.combination_details,
            "inference_time_ms": response.inference_time_ms,
            "timestamp": response.timestamp.isoformat(),
            "decision_id": f"{response.timestamp.strftime('%Y%m%d%H%M%S')}_{response.recommended_action}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"混合推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{session_id}/adaptive_recommendation", summary="自适应推荐")
async def adaptive_recommendation(session_id: str, request: HybridRecommendationRequestModel):
    """自适应权重的混合推荐"""
    try:
        # 构建推荐上下文
        context = None
        if request.context:
            context = RecommendationContext(
                user_id=request.context.get("user_id"),
                session_id=request.context.get("session_id"),
                item_features=request.context.get("item_features"),
                user_features=request.context.get("user_features"),
                context_features=request.context.get("context_features"),
                additional_info=request.context.get("additional_info")
            )
        
        # 构建推荐请求（自适应模式会自动调整权重）
        hybrid_request = HybridRecommendationRequest(
            agent_id=session_id,
            bandit_arm_id=request.bandit_arm_id,
            state=request.state,
            context=context,
            fallback_to_random=request.fallback_to_random
        )
        
        response = await recommendation_service.adaptive_recommendation(hybrid_request)
        
        return {
            "success": True,
            "recommended_action": response.recommended_action,
            "action_name": response.action_name,
            "confidence_score": response.confidence_score,
            "decision_source": response.decision_source.value,
            "q_learning_result": response.q_learning_result,
            "bandit_result": response.bandit_result,
            "combination_details": response.combination_details,
            "inference_time_ms": response.inference_time_ms,
            "timestamp": response.timestamp.isoformat(),
            "decision_id": f"{response.timestamp.strftime('%Y%m%d%H%M%S')}_{response.recommended_action}",
            "adaptive_weights": recommendation_service.adaptive_weights
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"自适应推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", summary="策略反馈")
async def update_strategy_feedback(request: FeedbackRequest):
    """更新策略反馈以优化推荐性能"""
    try:
        await recommendation_service.update_strategy_feedback(
            request.decision_id,
            request.reward,
            request.success
        )
        
        return {
            "success": True,
            "message": "策略反馈更新成功",
            "decision_id": request.decision_id
        }
        
    except Exception as e:
        logger.error(f"更新策略反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategy_performance", summary="策略性能报告")
async def get_strategy_performance():
    """获取策略性能报告"""
    try:
        report = await recommendation_service.get_strategy_performance_report()
        
        return {
            "success": True,
            "performance_report": report
        }
        
    except Exception as e:
        logger.error(f"获取策略性能报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize_weights", summary="优化策略权重")
async def optimize_strategy_weights():
    """基于历史性能优化策略权重"""
    try:
        optimized_weights = await recommendation_service.optimize_strategy_weights()
        
        return {
            "success": True,
            "message": "策略权重优化完成",
            "optimized_weights": optimized_weights
        }
        
    except Exception as e:
        logger.error(f"优化策略权重失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/combination_modes", summary="获取组合模式")
async def get_combination_modes():
    """获取支持的策略组合模式"""
    return {
        "success": True,
        "combination_modes": [
            {
                "mode": "weighted_average",
                "name": "加权平均",
                "description": "基于权重和置信度的加权选择"
            },
            {
                "mode": "epsilon_switching", 
                "name": "随机切换",
                "description": "基于概率随机选择策略"
            },
            {
                "mode": "contextual_selection",
                "name": "上下文选择",
                "description": "基于上下文特征选择最佳策略"
            },
            {
                "mode": "hierarchical",
                "name": "分层决策",
                "description": "分层次的策略决策"
            },
            {
                "mode": "ensemble_voting",
                "name": "集成投票",
                "description": "多策略投票决策"
            }
        ]
    }


# =============================================================================
# 智能体监控API端点
# =============================================================================

@router.post("/monitoring/start", summary="启动智能体监控")
async def start_agent_monitoring():
    """启动全局智能体监控系统"""
    try:
        await initialize_agent_monitoring()
        
        return {
            "success": True,
            "message": "智能体监控系统已启动"
        }
        
    except Exception as e:
        logger.error(f"启动智能体监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop", summary="停止智能体监控")
async def stop_agent_monitoring():
    """停止全局智能体监控系统"""
    try:
        monitor = get_agent_monitor()
        await monitor.stop_monitoring()
        
        return {
            "success": True,
            "message": "智能体监控系统已停止"
        }
        
    except Exception as e:
        logger.error(f"停止智能体监控失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/status", summary="获取监控状态")
async def get_monitoring_status():
    """获取智能体监控系统状态"""
    try:
        monitor = get_agent_monitor()
        
        return {
            "success": True,
            "monitoring_status": {
                "is_monitoring": monitor.is_monitoring,
                "buffer_sizes": {
                    "actions": len(monitor.action_buffer),
                    "decisions": len(monitor.decision_buffer),
                    "performance": len(monitor.performance_buffer),
                    "events": len(monitor.event_buffer)
                },
                "active_agents": len(monitor.agent_stats),
                "last_flush_time": monitor.last_flush_time
            }
        }
        
    except Exception as e:
        logger.error(f"获取监控状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}/summary", summary="获取智能体监控摘要")
async def get_agent_monitoring_summary(session_id: str):
    """获取指定智能体的监控摘要"""
    try:
        monitor = get_agent_monitor()
        summary = monitor.get_agent_summary(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="未找到智能体监控数据")
        
        return {
            "success": True,
            "agent_id": session_id,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取智能体监控摘要失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}/actions", summary="获取智能体动作历史")
async def get_agent_actions(session_id: str, limit: int = 100):
    """获取智能体的最近动作历史"""
    try:
        monitor = get_agent_monitor()
        actions = monitor.get_recent_actions(session_id, limit)
        
        return {
            "success": True,
            "agent_id": session_id,
            "actions": actions,
            "count": len(actions)
        }
        
    except Exception as e:
        logger.error(f"获取智能体动作历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{session_id}/performance_trend", summary="获取性能趋势")
async def get_agent_performance_trend(session_id: str, metric_name: str, hours: int = 24):
    """获取智能体的性能趋势数据"""
    try:
        monitor = get_agent_monitor()
        trend_data = monitor.get_performance_trend(session_id, metric_name, hours)
        
        return {
            "success": True,
            "agent_id": session_id,
            "metric_name": metric_name,
            "hours": hours,
            "trend_data": trend_data,
            "data_points": len(trend_data)
        }
        
    except Exception as e:
        logger.error(f"获取性能趋势失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/agents", summary="获取所有被监控的智能体")
async def get_monitored_agents():
    """获取所有被监控的智能体列表和状态"""
    try:
        monitor = get_agent_monitor()
        
        agents_info = []
        for agent_id, stats in monitor.agent_stats.items():
            agents_info.append({
                "agent_id": agent_id,
                "is_active": stats.get("is_active", False),
                "total_actions": stats.get("total_actions", 0),
                "exploration_rate": stats.get("exploration_rate", 0.0),
                "average_reward": stats.get("average_reward", 0.0),
                "last_activity": stats.get("last_activity").isoformat() if stats.get("last_activity") else None,
                "event_counts": dict(stats.get("event_counts", {}))
            })
        
        # 按活跃度和最后活动时间排序
        agents_info.sort(key=lambda x: (x["is_active"], x["last_activity"]), reverse=True)
        
        return {
            "success": True,
            "monitored_agents": agents_info,
            "total_agents": len(agents_info),
            "active_agents": sum(1 for agent in agents_info if agent["is_active"])
        }
        
    except Exception as e:
        logger.error(f"获取被监控智能体列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AgentEventRequest(BaseModel):
    """智能体事件请求"""
    event_type: str = Field(..., description="事件类型")
    message: str = Field(..., description="事件消息")
    level: str = Field(default="info", description="事件级别")
    data: Optional[Dict[str, Any]] = Field(default=None, description="额外数据")


@router.post("/agents/{session_id}/log_event", summary="记录智能体事件")
async def log_agent_event(session_id: str, request: AgentEventRequest):
    """手动记录智能体事件"""
    try:
        from ...ai.qlearning.agent_monitor import AgentEvent, LogLevel
        from datetime import datetime
        
        # 验证日志级别
        try:
            level = LogLevel(request.level.lower())
        except ValueError:
            level = LogLevel.INFO
        
        monitor = get_agent_monitor()
        
        event = AgentEvent(
            timestamp=datetime.now(),
            agent_id=session_id,
            session_id=session_id,
            event_type=request.event_type,
            level=level,
            message=request.message,
            data=request.data
        )
        
        monitor.log_event(event)
        
        return {
            "success": True,
            "message": "事件记录成功",
            "event": {
                "agent_id": session_id,
                "event_type": request.event_type,
                "level": request.level,
                "timestamp": event.timestamp.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"记录智能体事件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/statistics", summary="获取全局监控统计")
async def get_monitoring_statistics():
    """获取全局监控统计信息"""
    try:
        monitor = get_agent_monitor()
        
        # 统计各类数据
        total_actions = len(monitor.action_buffer)
        total_decisions = len(monitor.decision_buffer)
        total_events = len(monitor.event_buffer)
        total_performance_records = len(monitor.performance_buffer)
        
        # 统计活跃智能体
        active_agents = sum(1 for stats in monitor.agent_stats.values() 
                          if stats.get("is_active", False))
        
        # 统计事件级别分布
        event_level_counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for event in monitor.event_buffer:
            level = event.level.value
            if level in event_level_counts:
                event_level_counts[level] += 1
        
        # 计算平均决策时间
        if monitor.decision_buffer:
            avg_decision_time = sum(d.decision_time for d in monitor.decision_buffer) / len(monitor.decision_buffer)
        else:
            avg_decision_time = 0
        
        return {
            "success": True,
            "global_statistics": {
                "monitoring_active": monitor.is_monitoring,
                "total_agents": len(monitor.agent_stats),
                "active_agents": active_agents,
                "data_counts": {
                    "actions": total_actions,
                    "decisions": total_decisions,
                    "events": total_events,
                    "performance_records": total_performance_records
                },
                "event_distribution": event_level_counts,
                "average_decision_time_ms": round(avg_decision_time, 2),
                "buffer_utilization": {
                    "actions": f"{(total_actions / monitor.max_memory_size * 100):.1f}%",
                    "decisions": f"{(total_decisions / monitor.max_memory_size * 100):.1f}%",
                    "events": f"{(total_events / monitor.max_memory_size * 100):.1f}%",
                    "performance": f"{(total_performance_records / monitor.max_memory_size * 100):.1f}%"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"获取全局监控统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/monitoring/clear/{session_id}", summary="清理智能体监控数据")
async def clear_agent_monitoring_data(session_id: str):
    """清理指定智能体的监控数据"""
    try:
        from collections import deque
        monitor = get_agent_monitor()
        
        # 从缓冲区中移除指定智能体的数据
        monitor.action_buffer = deque(
            [action for action in monitor.action_buffer if action.agent_id != session_id],
            maxlen=monitor.max_memory_size
        )
        
        monitor.decision_buffer = deque(
            [decision for decision in monitor.decision_buffer if decision.agent_id != session_id],
            maxlen=monitor.max_memory_size
        )
        
        monitor.performance_buffer = deque(
            [perf for perf in monitor.performance_buffer if perf.agent_id != session_id],
            maxlen=monitor.max_memory_size
        )
        
        monitor.event_buffer = deque(
            [event for event in monitor.event_buffer if event.agent_id != session_id],
            maxlen=monitor.max_memory_size
        )
        
        # 删除统计信息
        if session_id in monitor.agent_stats:
            del monitor.agent_stats[session_id]
        
        return {
            "success": True,
            "message": f"已清理智能体 {session_id} 的监控数据"
        }
        
    except Exception as e:
        logger.error(f"清理智能体监控数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/monitoring/clear_all", summary="清理所有监控数据")
async def clear_all_monitoring_data():
    """清理所有智能体的监控数据"""
    try:
        monitor = get_agent_monitor()
        
        # 清空所有缓冲区
        monitor.action_buffer.clear()
        monitor.decision_buffer.clear()
        monitor.performance_buffer.clear()
        monitor.event_buffer.clear()
        
        # 清空统计信息
        monitor.agent_stats.clear()
        monitor.session_stats.clear()
        
        return {
            "success": True,
            "message": "已清理所有监控数据"
        }
        
    except Exception as e:
        logger.error(f"清理所有监控数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))