"""
模型评估相关的数据库模型定义
"""

from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import uuid

class EvaluationStatus(str, Enum):
    """评估状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReportStatus(str, Enum):
    """报告状态枚举"""
    GENERATING = "generating"
    READY = "ready"
    ERROR = "error"
    EXPIRED = "expired"

class AlertSeverity(str, Enum):
    """告警严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BenchmarkDifficulty(str, Enum):
    """基准测试难度"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

@dataclass
class ModelInfo:
    """模型信息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: Optional[str] = None
    description: Optional[str] = None
    model_path: str = ""
    model_type: str = "text_generation"  # text_generation, classification, qa
    architecture: Optional[str] = None  # transformer, bert, gpt
    parameters_count: Optional[int] = None
    training_data: Optional[str] = None
    license_info: Optional[str] = None
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class EvaluationJob:
    """评估任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Optional[str] = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    created_by: str = "system"
    
    # 模型配置
    models: List[Dict[str, Any]] = field(default_factory=list)  # 评估的模型列表
    benchmarks: List[Dict[str, Any]] = field(default_factory=list)  # 基准测试配置
    
    # 执行信息
    progress: float = 0.0
    current_task: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 结果
    results: List[Dict[str, Any]] = field(default_factory=list)
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    # 资源使用
    estimated_duration_minutes: Optional[int] = None
    actual_duration_seconds: Optional[float] = None
    peak_memory_usage_gb: Optional[float] = None
    total_samples_processed: int = 0
    
    # 元数据
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass  
class EvaluationResult:
    """单次评估结果"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    model_name: str = ""
    benchmark_name: str = ""
    task_name: str = ""
    
    # 指标结果
    accuracy: float = 0.0
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    perplexity: Optional[float] = None
    
    # 性能指标
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0
    gpu_utilization_percent: Optional[float] = None
    
    # 执行信息
    samples_evaluated: int = 0
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # 详细结果
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)  # 样本输出示例
    
    created_at: datetime = field(default_factory=utc_factory)

@dataclass
class BenchmarkDefinition:
    """基准测试定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    display_name: str = ""
    description: str = ""
    category: str = "general"  # general, nlp, code, reasoning, knowledge
    difficulty: BenchmarkDifficulty = BenchmarkDifficulty.MEDIUM
    
    # 测试配置
    tasks: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    metrics: List[str] = field(default_factory=list)
    
    # 资源要求
    num_samples: Optional[int] = None
    estimated_runtime_minutes: Optional[int] = None
    memory_requirements_gb: Optional[float] = None
    requires_gpu: bool = False
    
    # 数据集信息
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    dataset_split: str = "test"
    
    # 元信息
    paper_url: Optional[str] = None
    homepage_url: Optional[str] = None
    citation: Optional[str] = None
    license_info: Optional[str] = None
    
    # 配置
    few_shot_examples: int = 0
    prompt_template: Optional[str] = None
    evaluation_script: Optional[str] = None
    
    is_active: bool = True
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class PerformanceMetric:
    """性能指标记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: Optional[str] = None
    model_name: Optional[str] = None
    benchmark_name: Optional[str] = None
    
    # 指标类型
    metric_type: str = "system"  # system, model, benchmark
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: str = ""
    
    # 时间戳和持续时间
    timestamp: datetime = field(default_factory=utc_factory)
    duration_seconds: Optional[float] = None
    
    # 上下文信息
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class PerformanceAlert:
    """性能告警"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = "performance"
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    # 告警内容
    title: str = ""
    description: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    
    # 关联信息
    job_id: Optional[str] = None
    model_name: Optional[str] = None
    benchmark_name: Optional[str] = None
    
    # 状态
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    
    # 自动处理
    is_auto_created: bool = True
    auto_resolve_after_minutes: Optional[int] = None
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class EvaluationReport:
    """评估报告"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    subtitle: Optional[str] = None
    status: ReportStatus = ReportStatus.GENERATING
    
    # 内容配置
    include_charts: bool = True
    include_detailed_metrics: bool = True
    include_recommendations: bool = True
    output_format: str = "html"  # html, pdf, json
    template_name: str = "default"
    
    # 关联的评估任务
    job_ids: List[str] = field(default_factory=list)
    model_names: List[str] = field(default_factory=list)
    benchmark_names: List[str] = field(default_factory=list)
    
    # 报告文件
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    download_count: int = 0
    
    # 生成信息
    generated_by: str = "system"
    generation_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    
    # 有效期
    expires_at: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class ComparisonStudy:
    """模型对比研究"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Optional[str] = None
    
    # 对比配置
    model_names: List[str] = field(default_factory=list)
    benchmark_names: List[str] = field(default_factory=list)
    comparison_metrics: List[str] = field(default_factory=list)
    
    # 研究结果
    winner_model: Optional[str] = None
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # 统计分析
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 关联数据
    job_ids: List[str] = field(default_factory=list)
    report_id: Optional[str] = None
    
    created_by: str = "system"
    is_published: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)

@dataclass
class BaselineModel:
    """基线模型记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    benchmark_name: str = ""
    
    # 基线性能
    baseline_accuracy: float = 0.0
    baseline_f1_score: Optional[float] = None
    baseline_inference_time_ms: float = 0.0
    baseline_memory_usage_mb: float = 0.0
    
    # 基线建立
    established_date: datetime = field(default_factory=utc_factory)
    established_by: str = "system"
    source_job_id: str = ""
    
    # 阈值设置
    performance_degradation_threshold: float = 0.05  # 5% degradation
    speed_degradation_threshold: float = 0.20  # 20% slower
    memory_increase_threshold: float = 0.30  # 30% more memory
    
    # 元数据
    notes: Optional[str] = None
    is_active: bool = True
    tags: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
