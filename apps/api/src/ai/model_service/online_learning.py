"""在线学习引擎"""

import asyncio
import json
import math
from src.core.utils import secure_pickle as pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mean_squared_error
from .registry import ModelRegistry, ModelMetadata, ModelFormat
from .deployment import DeploymentManager

logger = get_logger(__name__)

class LearningSessionStatus(str, Enum):
    """学习会话状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class ABTestStatus(str, Enum):
    """A/B测试状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class FeedbackData:
    """用户反馈数据"""
    feedback_id: str
    session_id: str
    prediction_id: str
    inputs: Dict[str, Any]
    expected_output: Any
    actual_output: Any
    feedback_type: str  # 'classification', 'regression', 'ranking', 'rating'
    quality_score: float = 1.0  # 反馈质量分数
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningSession:
    """学习会话"""
    session_id: str
    model_id: str
    model_name: str
    model_version: str
    config: Dict[str, Any]
    status: LearningSessionStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_count: int = 0
    update_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class ABTestConfig:
    """A/B测试配置"""
    test_id: str
    name: str
    description: str
    control_model: str  # 对照组模型
    treatment_models: List[str]  # 实验组模型
    traffic_split: Dict[str, float]  # 流量分配
    success_metrics: List[str]  # 成功指标
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    max_duration_days: int = 30
    auto_winner_threshold: float = 0.05  # 显著性阈值

@dataclass
class ABTestResult:
    """A/B测试结果"""
    test_id: str
    model_id: str
    sample_count: int
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    is_winner: bool = False
    significance_level: float = 0.0

class IncrementalLearner:
    """增量学习器"""
    
    def __init__(self, model_format: ModelFormat, config: Dict[str, Any]):
        self.model_format = model_format
        self.config = config
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 32)
        self.buffer_size = config.get('buffer_size', 1000)
        self.update_frequency = config.get('update_frequency', 100)  # 每100个反馈更新一次
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
    def add_feedback(self, feedback: FeedbackData):
        """添加反馈到缓冲区"""
        training_example = self._convert_feedback_to_training_data(feedback)
        if training_example:
            self.replay_buffer.append(training_example)
    
    def _convert_feedback_to_training_data(self, feedback: FeedbackData) -> Optional[Dict[str, Any]]:
        """将反馈转换为训练数据"""
        try:
            if feedback.feedback_type == 'classification':
                return {
                    'inputs': feedback.inputs,
                    'targets': feedback.expected_output,
                    'weight': feedback.quality_score
                }
            elif feedback.feedback_type == 'regression':
                return {
                    'inputs': feedback.inputs,
                    'targets': feedback.expected_output,
                    'weight': feedback.quality_score
                }
            elif feedback.feedback_type == 'rating':
                # 将评分转换为回归任务
                return {
                    'inputs': feedback.inputs,
                    'targets': float(feedback.expected_output),
                    'weight': feedback.quality_score
                }
            else:
                logger.warning(f"不支持的反馈类型: {feedback.feedback_type}")
                return None
        except Exception as e:
            logger.error(f"转换反馈数据失败: {e}")
            return None
    
    def should_update(self) -> bool:
        """判断是否应该更新模型"""
        return len(self.replay_buffer) >= self.update_frequency
    
    def prepare_training_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """准备训练批次"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 从缓冲区随机采样
        batch_data = random.sample(list(self.replay_buffer), 
                                 min(self.batch_size, len(self.replay_buffer)))
        
        try:
            inputs = []
            targets = []
            weights = []
            
            for example in batch_data:
                # 简化处理，假设输入是数值列表
                input_data = example['inputs']
                if isinstance(input_data, dict) and 'data' in input_data:
                    inputs.append(input_data['data'])
                elif isinstance(input_data, list):
                    inputs.append(input_data)
                else:
                    continue
                
                targets.append(example['targets'])
                weights.append(example['weight'])
            
            if not inputs:
                return None
            
            # 转换为张量
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            
            return inputs_tensor, targets_tensor, weights_tensor
            
        except Exception as e:
            logger.error(f"准备训练批次失败: {e}")
            return None
    
    def update_model(self, model: nn.Module) -> Dict[str, float]:
        """更新模型"""
        if not self.should_update():
            return {"status": "no_update", "reason": "insufficient_data"}
        
        batch = self.prepare_training_batch()
        if batch is None:
            return {"status": "no_update", "reason": "batch_preparation_failed"}
        
        inputs, targets, weights = batch
        
        try:
            model.train()
            
            # 使用Adam优化器
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算加权损失
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1)
            if len(targets.shape) == 1:
                targets = targets.unsqueeze(1)
            
            loss_fn = nn.MSELoss(reduction='none')
            losses = loss_fn(outputs, targets.view_as(outputs))
            weighted_loss = (losses * weights.unsqueeze(1)).mean()
            
            # 反向传播
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            
            model.eval()
            
            # 计算性能指标
            with torch.no_grad():
                mse = losses.mean().item()
                mae = torch.abs(outputs - targets.view_as(outputs)).mean().item()
            
            return {
                "status": "updated",
                "loss": weighted_loss.item(),
                "mse": mse,
                "mae": mae,
                "samples_used": len(inputs)
            }
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            return {"status": "failed", "error": str(e)}

class ABTestEngine:
    """A/B测试引擎"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, ABTestResult]] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # {test_id: {user_id: model_id}}
        
    def create_ab_test(self, config: ABTestConfig) -> str:
        """创建A/B测试"""
        # 验证流量分配
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"流量分配总和必须为1.0，当前为{total_traffic}")
        
        # 验证模型存在
        all_models = [config.control_model] + config.treatment_models
        if not all(model in config.traffic_split for model in all_models):
            raise ValueError("所有模型都必须有流量分配")
        
        self.active_tests[config.test_id] = config
        self.test_results[config.test_id] = {}
        self.user_assignments[config.test_id] = {}
        
        logger.info(f"创建A/B测试: {config.test_id}")
        return config.test_id
    
    def assign_user_to_model(self, test_id: str, user_id: str) -> Optional[str]:
        """为用户分配模型"""
        if test_id not in self.active_tests:
            return None
        
        # 如果用户已有分配，返回原分配
        if user_id in self.user_assignments[test_id]:
            return self.user_assignments[test_id][user_id]
        
        config = self.active_tests[test_id]
        
        # 根据流量分配随机选择模型
        rand_val = random.random()
        cumulative = 0.0
        
        for model_id, traffic_ratio in config.traffic_split.items():
            cumulative += traffic_ratio
            if rand_val <= cumulative:
                self.user_assignments[test_id][user_id] = model_id
                return model_id
        
        # 兜底返回对照组
        self.user_assignments[test_id][user_id] = config.control_model
        return config.control_model
    
    def record_test_result(self, test_id: str, user_id: str, metrics: Dict[str, float]):
        """记录测试结果"""
        if test_id not in self.active_tests:
            return
        
        if user_id not in self.user_assignments[test_id]:
            return
        
        model_id = self.user_assignments[test_id][user_id]
        
        if model_id not in self.test_results[test_id]:
            self.test_results[test_id][model_id] = ABTestResult(
                test_id=test_id,
                model_id=model_id,
                sample_count=0,
                metrics={},
                confidence_intervals={}
            )
        
        result = self.test_results[test_id][model_id]
        result.sample_count += 1
        
        # 增量更新指标
        for metric, value in metrics.items():
            if metric not in result.metrics:
                result.metrics[metric] = value
            else:
                # 使用简单平均
                result.metrics[metric] = (
                    (result.metrics[metric] * (result.sample_count - 1) + value) / 
                    result.sample_count
                )
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """分析测试结果"""
        if test_id not in self.active_tests:
            return {"error": "测试不存在"}
        
        config = self.active_tests[test_id]
        results = self.test_results[test_id]
        
        if not results:
            return {"status": "no_data"}
        
        # 检查样本量是否足够
        min_samples = min(r.sample_count for r in results.values()) if results else 0
        if min_samples < config.minimum_sample_size:
            return {
                "status": "insufficient_samples",
                "min_samples": min_samples,
                "required_samples": config.minimum_sample_size
            }
        
        # 简化的显著性检测（实际应用中需要更严格的统计检验）
        analysis = {
            "status": "completed",
            "models": {},
            "winner": None,
            "significant_difference": False
        }
        
        control_result = results.get(config.control_model)
        if not control_result:
            return {"error": "对照组数据不足"}
        
        best_model = config.control_model
        best_score = 0.0
        
        for model_id, result in results.items():
            model_analysis = {
                "sample_count": result.sample_count,
                "metrics": result.metrics,
                "is_control": model_id == config.control_model
            }
            
            # 计算主要成功指标的改善
            if config.success_metrics and config.success_metrics[0] in result.metrics:
                primary_metric = config.success_metrics[0]
                score = result.metrics[primary_metric]
                
                if model_id != config.control_model:
                    improvement = (score - control_result.metrics.get(primary_metric, 0)) / max(control_result.metrics.get(primary_metric, 1), 1)
                    model_analysis["improvement"] = improvement
                    
                    if improvement > config.auto_winner_threshold:
                        model_analysis["significantly_better"] = True
                        if score > best_score:
                            best_model = model_id
                            best_score = score
                else:
                    model_analysis["improvement"] = 0.0
            
            analysis["models"][model_id] = model_analysis
        
        if best_model != config.control_model:
            analysis["winner"] = best_model
            analysis["significant_difference"] = True
        
        return analysis
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """获取活跃的测试列表"""
        result = []
        for test_id, config in self.active_tests.items():
            test_info = {
                "test_id": test_id,
                "name": config.name,
                "control_model": config.control_model,
                "treatment_models": config.treatment_models,
                "traffic_split": config.traffic_split,
                "total_users": len(self.user_assignments.get(test_id, {})),
                "sample_counts": {}
            }
            
            for model_id, result_obj in self.test_results.get(test_id, {}).items():
                test_info["sample_counts"][model_id] = result_obj.sample_count
            
            result.append(test_info)
        
        return result

class OnlineLearningEngine:
    """在线学习引擎"""
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        deployment_manager: DeploymentManager,
        storage_path: str = "/tmp/online_learning"
    ):
        self.model_registry = model_registry
        self.deployment_manager = deployment_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 学习会话管理
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.session_learners: Dict[str, IncrementalLearner] = {}
        
        # 反馈数据缓冲区
        self.feedback_buffer: Dict[str, List[FeedbackData]] = defaultdict(list)
        
        # A/B测试引擎
        self.ab_test_engine = ABTestEngine()
        
        # 性能监控
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # 后台任务
        self._running = True
        self._background_tasks = []
    
    async def start_online_learning(
        self, 
        model_name: str, 
        model_version: str,
        config: Dict[str, Any]
    ) -> str:
        """开始在线学习会话"""
        try:
            # 验证模型存在
            model_metadata = self.model_registry.get_model(model_name, model_version)
            if not model_metadata:
                raise ValueError(f"模型不存在: {model_name}:{model_version}")
            
            # 生成会话ID
            session_id = str(uuid.uuid4())
            
            # 创建学习会话
            session = LearningSession(
                session_id=session_id,
                model_id=model_metadata.model_id,
                model_name=model_name,
                model_version=model_metadata.version,
                config=config,
                status=LearningSessionStatus.ACTIVE
            )
            
            # 创建增量学习器
            learner = IncrementalLearner(model_metadata.format, config)
            
            # 保存会话
            self.learning_sessions[session_id] = session
            self.session_learners[session_id] = learner
            self.feedback_buffer[session_id] = []
            
            logger.info(f"开始在线学习会话: {session_id} for {model_name}:{model_version}")
            return session_id
            
        except Exception as e:
            logger.error(f"启动在线学习失败: {e}")
            raise
    
    async def collect_feedback(
        self, 
        session_id: str,
        prediction_id: str,
        feedback_data: Dict[str, Any]
    ):
        """收集用户反馈"""
        if session_id not in self.learning_sessions:
            raise ValueError(f"学习会话不存在: {session_id}")
        
        session = self.learning_sessions[session_id]
        if session.status != LearningSessionStatus.ACTIVE:
            raise ValueError(f"学习会话未激活: {session_id}")
        
        # 创建反馈对象
        feedback = FeedbackData(
            feedback_id=str(uuid.uuid4()),
            session_id=session_id,
            prediction_id=prediction_id,
            **feedback_data
        )
        
        # 添加到缓冲区
        self.feedback_buffer[session_id].append(feedback)
        
        # 添加到学习器
        if session_id in self.session_learners:
            self.session_learners[session_id].add_feedback(feedback)
        
        # 更新会话统计
        session.feedback_count += 1
        session.updated_at = datetime.now(timezone.utc)
        
        logger.debug(f"收集反馈: {feedback.feedback_id} for session {session_id}")
    
    async def update_model(self, session_id: str) -> Dict[str, Any]:
        """更新模型"""
        if session_id not in self.learning_sessions:
            return {"error": "学习会话不存在"}
        
        session = self.learning_sessions[session_id]
        learner = self.session_learners.get(session_id)
        
        if not learner or not learner.should_update():
            return {"status": "no_update", "reason": "条件不满足"}
        
        try:
            # 获取当前模型
            model_path = self.model_registry.get_model_path(
                session.model_name, 
                session.model_version
            )
            
            if not model_path:
                return {"error": "模型文件不存在"}
            
            # 加载模型
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            
            # 执行增量更新
            update_result = learner.update_model(model)
            
            if update_result.get("status") == "updated":
                # 保存更新后的模型
                updated_model_path = self.storage_path / f"{session_id}_updated_{session.update_count}.pt"
                torch.save(model, updated_model_path)
                
                # 更新会话统计
                session.update_count += 1
                session.updated_at = datetime.now(timezone.utc)
                session.performance_metrics.update(update_result)
                
                # 记录性能历史
                self.performance_history[session_id].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "update_count": session.update_count,
                    "metrics": update_result
                })
                
                logger.info(f"模型更新成功: session {session_id}, update #{session.update_count}")
            
            return update_result
            
        except Exception as e:
            logger.error(f"模型更新失败: {e}")
            return {"error": str(e)}
    
    def create_ab_test(
        self,
        name: str,
        description: str,
        control_model: str,
        treatment_models: List[str],
        traffic_split: Dict[str, float],
        success_metrics: List[str],
        **kwargs
    ) -> str:
        """创建A/B测试"""
        test_id = str(uuid.uuid4())
        
        config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            control_model=control_model,
            treatment_models=treatment_models,
            traffic_split=traffic_split,
            success_metrics=success_metrics,
            **kwargs
        )
        
        return self.ab_test_engine.create_ab_test(config)
    
    def assign_model_for_user(self, test_id: str, user_id: str) -> Optional[str]:
        """为用户分配A/B测试模型"""
        return self.ab_test_engine.assign_user_to_model(test_id, user_id)
    
    def record_ab_test_metrics(self, test_id: str, user_id: str, metrics: Dict[str, float]):
        """记录A/B测试指标"""
        self.ab_test_engine.record_test_result(test_id, user_id, metrics)
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """获取A/B测试结果"""
        return self.ab_test_engine.analyze_test_results(test_id)
    
    def get_learning_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取学习统计信息"""
        if session_id not in self.learning_sessions:
            return None
        
        session = self.learning_sessions[session_id]
        learner = self.session_learners.get(session_id)
        
        stats = {
            "session_id": session_id,
            "model_name": session.model_name,
            "model_version": session.model_version,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "config": session.config,
            "feedback_count": session.feedback_count,
            "update_count": session.update_count,
            "performance_metrics": session.performance_metrics,
            "pending_feedback": len(self.feedback_buffer.get(session_id, [])),
            "buffer_usage": len(learner.replay_buffer) if learner else 0,
            "buffer_capacity": learner.buffer_size if learner else 0
        }
        
        return stats
    
    def get_all_learning_sessions(self) -> List[Dict[str, Any]]:
        """获取所有学习会话"""
        result = []
        for session_id in self.learning_sessions:
            stats = self.get_learning_stats(session_id)
            if stats:
                result.append(stats)
        return result
    
    def get_performance_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取性能历史记录"""
        return self.performance_history.get(session_id, [])
    
    async def pause_learning_session(self, session_id: str) -> bool:
        """暂停学习会话"""
        if session_id not in self.learning_sessions:
            return False
        
        session = self.learning_sessions[session_id]
        session.status = LearningSessionStatus.PAUSED
        session.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"暂停学习会话: {session_id}")
        return True
    
    async def resume_learning_session(self, session_id: str) -> bool:
        """恢复学习会话"""
        if session_id not in self.learning_sessions:
            return False
        
        session = self.learning_sessions[session_id]
        session.status = LearningSessionStatus.ACTIVE
        session.updated_at = datetime.now(timezone.utc)
        
        logger.info(f"恢复学习会话: {session_id}")
        return True
    
    async def stop_learning_session(self, session_id: str) -> bool:
        """停止学习会话"""
        if session_id not in self.learning_sessions:
            return False
        
        session = self.learning_sessions[session_id]
        session.status = LearningSessionStatus.COMPLETED
        session.updated_at = datetime.now(timezone.utc)
        
        # 清理资源
        if session_id in self.session_learners:
            del self.session_learners[session_id]
        
        if session_id in self.feedback_buffer:
            del self.feedback_buffer[session_id]
        
        logger.info(f"停止学习会话: {session_id}")
        return True
    
    def get_ab_tests(self) -> List[Dict[str, Any]]:
        """获取所有A/B测试"""
        return self.ab_test_engine.get_active_tests()
    
    async def cleanup_expired_sessions(self, max_age_days: int = 30):
        """清理过期会话"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        expired_sessions = []
        
        for session_id, session in self.learning_sessions.items():
            if session.updated_at < cutoff_time and session.status != LearningSessionStatus.ACTIVE:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.stop_learning_session(session_id)
            del self.learning_sessions[session_id]
        
        logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        active_sessions = sum(1 for s in self.learning_sessions.values() 
                            if s.status == LearningSessionStatus.ACTIVE)
        total_feedback = sum(len(fb) for fb in self.feedback_buffer.values())
        active_tests = len(self.ab_test_engine.active_tests)
        
        return {
            "status": "healthy",
            "total_sessions": len(self.learning_sessions),
            "active_sessions": active_sessions,
            "total_feedback_pending": total_feedback,
            "active_ab_tests": active_tests,
            "performance_history_size": sum(len(h) for h in self.performance_history.values())
        }
from src.core.logging import get_logger
