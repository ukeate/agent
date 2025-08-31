"""解释记录的持久化模型和缓存管理

本模块提供解释记录的持久化存储、历史版本控制和缓存管理功能。
"""

import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import Column, String, DateTime, Text, Float, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from src.core.config import get_settings
from src.models.schemas.explanation import (
    DecisionExplanation, 
    ExplanationHistory,
    ExplanationLevel,
    ExplanationType
)

Base = declarative_base()
settings = get_settings()


class ExplanationRecord(Base):
    """解释记录数据库模型"""
    __tablename__ = "explanation_records"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, index=True)
    decision_id = Column(String(255), nullable=False, index=True)
    explanation_type = Column(String(50), nullable=False)
    explanation_level = Column(String(20), nullable=False)
    
    # 决策信息
    decision_description = Column(Text, nullable=False)
    decision_outcome = Column(Text, nullable=False)
    decision_context = Column(Text, nullable=True)
    
    # 解释内容
    summary_explanation = Column(Text, nullable=False)
    detailed_explanation = Column(Text, nullable=True)
    technical_explanation = Column(Text, nullable=True)
    
    # 解释组件 (JSON存储)
    components = Column(JSON, nullable=False, default=list)
    
    # 置信度指标 (JSON存储)
    confidence_metrics = Column(JSON, nullable=False)
    
    # 反事实场景 (JSON存储)
    counterfactuals = Column(JSON, nullable=False, default=list)
    
    # 可视化数据 (JSON存储)
    visualization_data = Column(JSON, nullable=True)
    
    # 元数据 (JSON存储) - 使用不同名称避免与SQLAlchemy的metadata冲突
    model_metadata = Column(JSON, nullable=False, default=dict)
    
    # 时间戳
    created_at = Column(DateTime, default=lambda: utc_now(), nullable=False)
    updated_at = Column(DateTime, default=lambda: utc_now(), onupdate=lambda: utc_now())
    
    # 版本控制
    version = Column(Integer, default=1, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    def to_schema(self) -> DecisionExplanation:
        """转换为Pydantic模型"""
        return DecisionExplanation.model_validate({
            "id": self.id,
            "decision_id": self.decision_id,
            "explanation_type": self.explanation_type,
            "explanation_level": self.explanation_level,
            "decision_description": self.decision_description,
            "decision_outcome": self.decision_outcome,
            "decision_context": self.decision_context,
            "summary_explanation": self.summary_explanation,
            "detailed_explanation": self.detailed_explanation,
            "technical_explanation": self.technical_explanation,
            "components": self.components,
            "confidence_metrics": self.confidence_metrics,
            "counterfactuals": self.counterfactuals,
            "visualization_data": self.visualization_data,
            "metadata": self.model_metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        })
    
    @classmethod
    def from_schema(cls, explanation: DecisionExplanation) -> 'ExplanationRecord':
        """从Pydantic模型创建"""
        return cls(
            id=explanation.id,
            decision_id=explanation.decision_id,
            explanation_type=explanation.explanation_type.value,
            explanation_level=explanation.explanation_level.value,
            decision_description=explanation.decision_description,
            decision_outcome=explanation.decision_outcome,
            decision_context=explanation.decision_context,
            summary_explanation=explanation.summary_explanation,
            detailed_explanation=explanation.detailed_explanation,
            technical_explanation=explanation.technical_explanation,
            components=[comp.model_dump() for comp in explanation.components],
            confidence_metrics=explanation.confidence_metrics.model_dump(),
            counterfactuals=[cf.model_dump() for cf in explanation.counterfactuals],
            visualization_data=explanation.visualization_data,
            model_metadata=explanation.metadata,
            created_at=explanation.created_at,
            updated_at=explanation.updated_at
        )


class ExplanationHistoryRecord(Base):
    """解释历史记录数据库模型"""
    __tablename__ = "explanation_history"
    
    id = Column(PostgresUUID(as_uuid=True), primary_key=True, index=True)
    explanation_id = Column(PostgresUUID(as_uuid=True), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    changes = Column(JSON, nullable=False)
    change_reason = Column(Text, nullable=False)
    changed_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: utc_now(), nullable=False)
    
    def to_schema(self) -> ExplanationHistory:
        """转换为Pydantic模型"""
        return ExplanationHistory.model_validate({
            "id": self.id,
            "explanation_id": self.explanation_id,
            "version": self.version,
            "changes": self.changes,
            "change_reason": self.change_reason,
            "changed_by": self.changed_by,
            "created_at": self.created_at
        })


class ExplanationCache:
    """解释缓存管理类"""
    
    def __init__(self, redis_client=None, cache_ttl: int = 3600):
        """
        初始化缓存管理器
        
        Args:
            redis_client: Redis客户端实例
            cache_ttl: 缓存过期时间(秒)，默认1小时
        """
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.cache_prefix = "explanation:"
    
    def _get_cache_key(self, decision_id: str, explanation_type: str, level: str) -> str:
        """生成缓存键"""
        return f"{self.cache_prefix}{decision_id}:{explanation_type}:{level}"
    
    async def get(self, decision_id: str, explanation_type: ExplanationType, 
                  level: ExplanationLevel) -> Optional[DecisionExplanation]:
        """从缓存获取解释"""
        if not self.redis_client:
            return None
            
        cache_key = self._get_cache_key(decision_id, explanation_type.value, level.value)
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                explanation_dict = json.loads(cached_data)
                return DecisionExplanation.model_validate(explanation_dict)
        except Exception as e:
            # 缓存读取失败时静默处理
            pass
        
        return None
    
    async def set(self, explanation: DecisionExplanation) -> bool:
        """将解释存入缓存"""
        if not self.redis_client:
            return False
            
        cache_key = self._get_cache_key(
            explanation.decision_id, 
            explanation.explanation_type.value,
            explanation.explanation_level.value
        )
        
        try:
            explanation_json = explanation.model_dump_json()
            await self.redis_client.setex(cache_key, self.cache_ttl, explanation_json)
            return True
        except Exception as e:
            # 缓存写入失败时静默处理
            return False
    
    async def delete(self, decision_id: str, explanation_type: ExplanationType, 
                     level: ExplanationLevel) -> bool:
        """从缓存删除解释"""
        if not self.redis_client:
            return False
            
        cache_key = self._get_cache_key(decision_id, explanation_type.value, level.value)
        
        try:
            await self.redis_client.delete(cache_key)
            return True
        except Exception as e:
            return False
    
    async def delete_by_decision(self, decision_id: str) -> int:
        """删除决策相关的所有缓存"""
        if not self.redis_client:
            return 0
            
        pattern = f"{self.cache_prefix}{decision_id}:*"
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            pattern = f"{self.cache_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            stats = {
                "total_cached_explanations": len(keys),
                "cache_ttl": self.cache_ttl,
                "cache_prefix": self.cache_prefix,
                "status": "active"
            }
            
            # 统计不同类型的解释数量
            type_counts = {}
            level_counts = {}
            
            for key in keys:
                key_parts = key.decode().split(':')
                if len(key_parts) >= 4:
                    exp_type = key_parts[2]
                    exp_level = key_parts[3]
                    
                    type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
                    level_counts[exp_level] = level_counts.get(exp_level, 0) + 1
            
            stats["by_type"] = type_counts
            stats["by_level"] = level_counts
            
            return stats
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


class ExplanationTemplate:
    """解释模板管理类"""
    
    def __init__(self):
        """初始化模板管理器"""
        self.templates = {
            ExplanationType.DECISION: {
                ExplanationLevel.SUMMARY: "基于{factors}，系统做出了{decision}的决定，置信度为{confidence}。",
                ExplanationLevel.DETAILED: "系统分析了{num_factors}个关键因素：{factor_details}。通过权重计算，{primary_factor}是最重要的决策依据，最终得出{decision}的结论，置信度达到{confidence}。",
                ExplanationLevel.TECHNICAL: "决策过程采用{algorithm}算法，输入特征{input_features}，权重向量{weights}，计算得出决策分数{score}。不确定性分析显示{uncertainty_details}，最终输出{decision}。"
            },
            ExplanationType.REASONING: {
                ExplanationLevel.SUMMARY: "推理过程包含{step_count}个步骤，从{start_point}到{conclusion}。",
                ExplanationLevel.DETAILED: "推理链：{reasoning_steps}。每个步骤的置信度分别为{step_confidences}，整体推理置信度为{overall_confidence}。",
                ExplanationLevel.TECHNICAL: "推理引擎：{engine_type}，状态转换：{state_transitions}，约束条件：{constraints}，推理路径：{reasoning_path}。"
            },
            ExplanationType.WORKFLOW: {
                ExplanationLevel.SUMMARY: "工作流执行了{task_count}个任务，当前状态为{current_state}。",
                ExplanationLevel.DETAILED: "工作流决策：{workflow_decisions}。任务分配基于{allocation_strategy}，执行顺序为{execution_order}。",
                ExplanationLevel.TECHNICAL: "工作流引擎：{engine_details}，任务调度：{scheduling_details}，状态管理：{state_management}。"
            }
        }
    
    def get_template(self, explanation_type: ExplanationType, 
                     level: ExplanationLevel) -> str:
        """获取解释模板"""
        return self.templates.get(explanation_type, {}).get(
            level, 
            "无法为{explanation_type}和{level}提供模板。"
        )
    
    def render_template(self, explanation_type: ExplanationType, 
                       level: ExplanationLevel, **kwargs) -> str:
        """渲染解释模板"""
        template = self.get_template(explanation_type, level)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"模板渲染失败，缺少参数：{e}"
    
    def add_template(self, explanation_type: ExplanationType, 
                     level: ExplanationLevel, template: str) -> None:
        """添加自定义模板"""
        if explanation_type not in self.templates:
            self.templates[explanation_type] = {}
        self.templates[explanation_type][level] = template