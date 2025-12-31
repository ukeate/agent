"""
超参数优化数据模型

定义实验和试验的数据库模型，用于持久化存储优化历史和结果。
"""

from sqlalchemy import Column, String, Float, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid as uuid_lib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict
from enum import Enum

class Base(DeclarativeBase):
    ...

class TrialState(str, Enum):
    """试验状态枚举"""
    RUNNING = "running"
    COMPLETE = "complete"
    PRUNED = "pruned"
    FAILED = "failed"
    WAITING = "waiting"

class ExperimentState(str, Enum):
    """实验状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class OptimizationAlgorithm(str, Enum):
    """优化算法枚举"""
    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"
    GRID = "grid"
    HYPERBAND = "hyperband"

class Experiment:
    """实验对象"""
    def __init__(self, id, name, state, **kwargs):
        self.id = id
        self.name = name
        self.state = state
        self.description = kwargs.get('description')
        self.algorithm = kwargs.get('algorithm')
        self.created_at = kwargs.get('created_at', utc_now())
        self.parameter_ranges = kwargs.get('parameter_ranges', {})
        self.optimization_config = kwargs.get('optimization_config', {})
        self.checkpoint = kwargs.get('checkpoint', {})

class Trial:
    """试验对象"""
    def __init__(self, id, experiment_id, state, **kwargs):
        self.id = id
        self.experiment_id = experiment_id
        self.state = state
        self.parameters = kwargs.get('parameters', {})
        self.value = kwargs.get('value')
        self.metrics = kwargs.get('metrics', {})
        self.created_at = kwargs.get('created_at', utc_now())

class ExperimentModel(Base):
    """实验模型"""
    __tablename__ = "hyperparameter_experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="created")  # created, running, completed, failed, stopped
    algorithm = Column(String(50), nullable=False)
    objective = Column(String(20), nullable=False)  # maximize, minimize
    
    # 配置JSON
    config = Column(JSONB)
    parameters = Column(JSONB)
    
    # 结果
    best_value = Column(Float)
    best_params = Column(JSONB)
    
    # 统计
    total_trials = Column(Integer, default=0)
    successful_trials = Column(Integer, default=0)
    pruned_trials = Column(Integer, default=0)
    failed_trials = Column(Integer, default=0)
    
    # 资源使用
    max_concurrent_trials = Column(Integer, default=5)
    resource_timeout = Column(Integer, default=3600)
    
    # 时间戳
    created_at = Column(DateTime, default=utc_now)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    # 关系
    trials = relationship("TrialModel", back_populates="experiment", cascade="all, delete-orphan")

class TrialModel(Base):
    """试验模型"""
    __tablename__ = "hyperparameter_trials"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("hyperparameter_experiments.id"), nullable=False)
    trial_number = Column(Integer, nullable=False)
    
    # 参数和结果
    parameters = Column(JSONB)
    value = Column(Float)
    state = Column(String(20), nullable=False)  # COMPLETE, PRUNED, FAIL, RUNNING
    
    # 中间值 (用于剪枝)
    intermediate_values = Column(JSONB)
    
    # 时间统计
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)  # 秒
    
    # 错误和系统信息
    error_message = Column(Text)
    system_attrs = Column(JSONB)
    user_attrs = Column(JSONB)
    
    # 资源使用
    resource_usage = Column(JSONB)  # CPU, memory, GPU使用情况
    
    # 关系
    experiment = relationship("ExperimentModel", back_populates="trials")

class StudyMetadataModel(Base):
    """研究元数据模型"""
    __tablename__ = "hyperparameter_study_metadata"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    study_name = Column(String(255), unique=True, nullable=False)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("hyperparameter_experiments.id"))
    
    # Optuna study相关
    storage_url = Column(String(500))
    sampler_class = Column(String(100))
    pruner_class = Column(String(100))
    
    # 搜索空间
    search_space = Column(JSONB)
    
    # 创建时间
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

# Pydantic模型用于API

class HyperparameterRangeSchema(BaseModel):
    """超参数范围Schema"""
    name: str
    type: str  # float, int, categorical, boolean
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None

class ExperimentRequest(BaseModel):
    """创建实验请求"""
    name: str
    description: Optional[str] = None
    algorithm: str = "tpe"
    objective: str = "maximize"
    n_trials: int = 100
    timeout: Optional[int] = None
    early_stopping: bool = True
    patience: int = 20
    min_improvement: float = 0.001
    max_concurrent_trials: int = 5
    parameters: List[HyperparameterRangeSchema]

class ExperimentResponse(BaseModel):
    """实验响应"""
    id: str
    name: str
    status: str
    algorithm: str
    objective: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ExperimentDetail(BaseModel):
    """实验详情"""
    id: str
    name: str
    description: Optional[str]
    status: str
    algorithm: str
    objective: str
    config: Dict[str, Any]
    parameters: List[Dict[str, Any]]
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]
    total_trials: int
    successful_trials: int
    pruned_trials: int
    failed_trials: int
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    trials_count: int
    
    model_config = ConfigDict(from_attributes=True)

class TrialResponse(BaseModel):
    """试验响应"""
    id: str
    trial_number: int
    parameters: Dict[str, Any]
    value: Optional[float]
    state: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    intermediate_values: Optional[Dict[str, float]]
    resource_usage: Optional[Dict[str, float]]
    
    model_config = ConfigDict(from_attributes=True)

class OptimizationResult(BaseModel):
    """优化结果"""
    best_params: Dict[str, Any]
    best_value: float
    stats: Dict[str, Any]
    visualizations: List[str]

class AlgorithmComparison(BaseModel):
    """算法对比结果"""
    results: Dict[str, OptimizationResult]
    comparison: Dict[str, Any]

class ResourceStats(BaseModel):
    """资源统计"""
    current_trials: int
    max_concurrent: int
    resource_usage: Dict[str, float]
    active_trials: List[str]

class TaskInfo(BaseModel):
    """任务信息"""
    task_name: str
    algorithm: str
    pruning: str
    n_trials: int
    direction: str
    parameters: List[Dict[str, Any]]

class CustomTaskRequest(BaseModel):
    """自定义任务请求"""
    task_name: str
    parameters: List[HyperparameterRangeSchema]
    algorithm: str = "tpe"
    pruning: str = "hyperband"
    direction: str = "maximize"
    n_trials: int = 100
    early_stopping: bool = True
    patience: int = 20

class OptimizationProgress(BaseModel):
    """优化进度"""
    experiment_id: str
    current_trial: int
    total_trials: int
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]
    elapsed_time: float
    estimated_remaining_time: Optional[float]
    status: str
