"""推理链持久化模型"""

from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Optional
from uuid import UUID
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import ForeignKey

class Base(DeclarativeBase):
    ...

class ReasoningChainModel(Base):
    """推理链持久化模型"""
    __tablename__ = "reasoning_chains"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    session_id = Column(String(255), nullable=True, index=True)
    strategy = Column(String(50), nullable=False)
    problem = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    conclusion = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    total_duration_ms = Column(Integer, nullable=True)
    metadata = Column(JSON, default=dict)
    version = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # 关系
    steps = relationship("ThoughtStepModel", back_populates="chain", cascade="all, delete-orphan")
    branches = relationship("ReasoningBranchModel", back_populates="chain", cascade="all, delete-orphan")

class ThoughtStepModel(Base):
    """思考步骤持久化模型"""
    __tablename__ = "thought_steps"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    chain_id = Column(PG_UUID(as_uuid=True), ForeignKey("reasoning_chains.id"), nullable=False)
    branch_id = Column(PG_UUID(as_uuid=True), ForeignKey("reasoning_branches.id"), nullable=True)
    step_number = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    chain = relationship("ReasoningChainModel", back_populates="steps")
    branch = relationship("ReasoningBranchModel", back_populates="steps")
    validations = relationship("StepValidationModel", back_populates="step", cascade="all, delete-orphan")

class ReasoningBranchModel(Base):
    """推理分支持久化模型"""
    __tablename__ = "reasoning_branches"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    chain_id = Column(PG_UUID(as_uuid=True), ForeignKey("reasoning_chains.id"), nullable=False)
    parent_step_id = Column(PG_UUID(as_uuid=True), ForeignKey("thought_steps.id"), nullable=True)
    branch_reason = Column(Text, nullable=False)
    priority = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    chain = relationship("ReasoningChainModel", back_populates="branches")
    steps = relationship("ThoughtStepModel", back_populates="branch")

class StepValidationModel(Base):
    """步骤验证持久化模型"""
    __tablename__ = "step_validations"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    step_id = Column(PG_UUID(as_uuid=True), ForeignKey("thought_steps.id"), nullable=False)
    is_valid = Column(Boolean, nullable=False)
    consistency_score = Column(Float, nullable=False)
    issues = Column(JSON, default=list)
    suggestions = Column(JSON, default=list)
    validated_at = Column(DateTime(timezone=True), server_default=func.now())

    # 关系
    step = relationship("ThoughtStepModel", back_populates="validations")

class ReasoningExampleModel(Base):
    """推理示例模型（用于Few-shot）"""
    __tablename__ = "reasoning_examples"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    category = Column(String(100), nullable=False, index=True)
    problem = Column(Text, nullable=False)
    solution_steps = Column(JSON, nullable=False)
    conclusion = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ReasoningCacheModel(Base):
    """推理缓存模型"""
    __tablename__ = "reasoning_cache"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    cache_key = Column(String(512), unique=True, nullable=False, index=True)
    problem_hash = Column(String(128), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    chain_id = Column(PG_UUID(as_uuid=True), ForeignKey("reasoning_chains.id"))
    result = Column(JSON, nullable=False)
    hit_count = Column(Integer, default=0)
    ttl_seconds = Column(Integer, default=3600)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
