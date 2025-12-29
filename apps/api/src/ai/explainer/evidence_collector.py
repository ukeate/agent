"""证据收集器

本模块实现证据链的收集和组织功能，包括：
- 证据收集和分类
- 因果关系分析
- 证据权重计算
- 证据链构建
"""

import hashlib
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4
from src.models.schemas.explanation import (
    EvidenceType,
    ConfidenceSource,
    ExplanationComponent
)

class Evidence:
    """证据对象"""
    
    def __init__(
        self,
        evidence_id: str,
        evidence_type: EvidenceType,
        content: str,
        source: str,
        raw_data: Dict[str, Any],
        confidence: float = 1.0,
        timestamp: Optional[datetime] = None
    ):
        self.evidence_id = evidence_id
        self.evidence_type = evidence_type
        self.content = content
        self.source = source
        self.raw_data = raw_data
        self.confidence = confidence
        self.timestamp = timestamp or utc_now()
        
        # 证据属性
        self.weight = 0.0  # 在决策中的权重
        self.reliability_score = 1.0  # 可靠性分数
        self.relevance_score = 1.0  # 相关性分数
        self.freshness_score = 1.0  # 新鲜度分数
        
        # 关联关系
        self.related_evidence: Set[str] = set()
        self.contradictory_evidence: Set[str] = set()
        self.supporting_evidence: Set[str] = set()
        
        # 元数据
        self.metadata: Dict[str, Any] = {}
        self.validation_status = "pending"  # pending, validated, rejected
        
    def add_metadata(self, key: str, value: Any):
        """添加元数据"""
        self.metadata[key] = value
    
    def set_validation_status(self, status: str, reason: str = ""):
        """设置验证状态"""
        self.validation_status = status
        self.metadata["validation_reason"] = reason
        self.metadata["validation_timestamp"] = utc_now().isoformat()
    
    def calculate_overall_score(self) -> float:
        """计算总体分数"""
        return (
            self.confidence * 0.3 +
            self.reliability_score * 0.25 +
            self.relevance_score * 0.25 +
            self.freshness_score * 0.2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "content": self.content,
            "source": self.source,
            "raw_data": self.raw_data,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "weight": self.weight,
            "reliability_score": self.reliability_score,
            "relevance_score": self.relevance_score,
            "freshness_score": self.freshness_score,
            "related_evidence": list(self.related_evidence),
            "contradictory_evidence": list(self.contradictory_evidence),
            "supporting_evidence": list(self.supporting_evidence),
            "metadata": self.metadata,
            "validation_status": self.validation_status,
            "overall_score": self.calculate_overall_score()
        }

class CausalRelationship:
    """因果关系"""
    
    def __init__(
        self,
        cause_evidence_id: str,
        effect_evidence_id: str,
        relationship_type: str,  # "direct", "indirect", "contributing", "inhibiting"
        strength: float,  # 0.0 - 1.0
        confidence: float,  # 0.0 - 1.0
        description: str = ""
    ):
        self.cause_evidence_id = cause_evidence_id
        self.effect_evidence_id = effect_evidence_id
        self.relationship_type = relationship_type
        self.strength = strength
        self.confidence = confidence
        self.description = description
        self.discovered_at = utc_now()
        self.metadata: Dict[str, Any] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause_evidence_id": self.cause_evidence_id,
            "effect_evidence_id": self.effect_evidence_id,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "confidence": self.confidence,
            "description": self.description,
            "discovered_at": self.discovered_at.isoformat(),
            "metadata": self.metadata
        }

class EvidenceCollector:
    """证据收集器"""
    
    def __init__(self, decision_id: str):
        self.decision_id = decision_id
        self.evidence_store: Dict[str, Evidence] = {}
        self.causal_relationships: List[CausalRelationship] = []
        self.evidence_chains: List[List[str]] = []  # 证据链
        self.collection_started_at = utc_now()
        self.collection_completed_at: Optional[datetime] = None
        
        # 统计信息
        self.stats = {
            "total_evidence": 0,
            "by_type": {},
            "by_source": {},
            "validation_stats": {"validated": 0, "rejected": 0, "pending": 0}
        }
    
    def collect_evidence(
        self,
        evidence_type: EvidenceType,
        content: str,
        source: str,
        raw_data: Dict[str, Any],
        confidence: float = 1.0,
        evidence_id: Optional[str] = None
    ) -> str:
        """收集证据"""
        if evidence_id is None:
            # 基于内容生成唯一ID
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            evidence_id = f"{self.decision_id}_evidence_{content_hash}"
        
        evidence = Evidence(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            content=content,
            source=source,
            raw_data=raw_data,
            confidence=confidence
        )
        
        # 计算证据属性分数
        self._calculate_evidence_scores(evidence)
        
        self.evidence_store[evidence_id] = evidence
        self._update_stats()
        
        return evidence_id
    
    def collect_input_data_evidence(
        self,
        field_name: str,
        field_value: Any,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """收集输入数据证据"""
        return self.collect_evidence(
            evidence_type=EvidenceType.INPUT_DATA,
            content=f"输入字段 {field_name} 的值为 {field_value}",
            source=source,
            raw_data={
                "field_name": field_name,
                "field_value": field_value,
                "metadata": metadata or {}
            }
        )
    
    def collect_context_evidence(
        self,
        context_type: str,
        context_data: Dict[str, Any],
        source: str
    ) -> str:
        """收集上下文证据"""
        return self.collect_evidence(
            evidence_type=EvidenceType.CONTEXT,
            content=f"上下文信息：{context_type}",
            source=source,
            raw_data={
                "context_type": context_type,
                "context_data": context_data
            }
        )
    
    def collect_memory_evidence(
        self,
        memory_type: str,
        memory_content: Dict[str, Any],
        source: str,
        retrieval_score: float = 1.0
    ) -> str:
        """收集记忆证据"""
        return self.collect_evidence(
            evidence_type=EvidenceType.MEMORY,
            content=f"记忆信息：{memory_type}",
            source=source,
            raw_data={
                "memory_type": memory_type,
                "memory_content": memory_content,
                "retrieval_score": retrieval_score
            },
            confidence=retrieval_score
        )
    
    def collect_reasoning_step_evidence(
        self,
        step_description: str,
        step_input: Dict[str, Any],
        step_output: Dict[str, Any],
        reasoning_logic: str,
        source: str
    ) -> str:
        """收集推理步骤证据"""
        return self.collect_evidence(
            evidence_type=EvidenceType.REASONING_STEP,
            content=f"推理步骤：{step_description}",
            source=source,
            raw_data={
                "step_description": step_description,
                "step_input": step_input,
                "step_output": step_output,
                "reasoning_logic": reasoning_logic
            }
        )
    
    def add_causal_relationship(
        self,
        cause_evidence_id: str,
        effect_evidence_id: str,
        relationship_type: str,
        strength: float,
        confidence: float,
        description: str = ""
    ):
        """添加因果关系"""
        if cause_evidence_id not in self.evidence_store:
            raise ValueError(f"Cause evidence {cause_evidence_id} not found")
        if effect_evidence_id not in self.evidence_store:
            raise ValueError(f"Effect evidence {effect_evidence_id} not found")
        
        relationship = CausalRelationship(
            cause_evidence_id=cause_evidence_id,
            effect_evidence_id=effect_evidence_id,
            relationship_type=relationship_type,
            strength=strength,
            confidence=confidence,
            description=description
        )
        
        self.causal_relationships.append(relationship)
        
        # 更新证据关联关系
        if relationship_type in ["direct", "indirect", "contributing"]:
            self.evidence_store[cause_evidence_id].supporting_evidence.add(effect_evidence_id)
            self.evidence_store[effect_evidence_id].related_evidence.add(cause_evidence_id)
        elif relationship_type == "inhibiting":
            self.evidence_store[cause_evidence_id].contradictory_evidence.add(effect_evidence_id)
            self.evidence_store[effect_evidence_id].contradictory_evidence.add(cause_evidence_id)
    
    def build_evidence_chain(self, start_evidence_id: str) -> List[str]:
        """构建证据链"""
        visited = set()
        chain = []
        
        def dfs(evidence_id: str):
            if evidence_id in visited or evidence_id not in self.evidence_store:
                return
            
            visited.add(evidence_id)
            chain.append(evidence_id)
            
            # 跟踪支持证据
            evidence = self.evidence_store[evidence_id]
            for related_id in evidence.supporting_evidence:
                dfs(related_id)
        
        dfs(start_evidence_id)
        
        if len(chain) > 1:
            self.evidence_chains.append(chain)
        
        return chain
    
    def calculate_evidence_weights(self):
        """计算证据权重"""
        total_score = sum(evidence.calculate_overall_score() for evidence in self.evidence_store.values())
        
        if total_score == 0:
            return
        
        for evidence in self.evidence_store.values():
            evidence.weight = evidence.calculate_overall_score() / total_score
    
    def validate_evidence(self, evidence_id: str, is_valid: bool, reason: str = ""):
        """验证证据"""
        if evidence_id not in self.evidence_store:
            raise ValueError(f"Evidence {evidence_id} not found")
        
        evidence = self.evidence_store[evidence_id]
        status = "validated" if is_valid else "rejected"
        evidence.set_validation_status(status, reason)
        
        self._update_stats()
    
    def get_evidence_by_type(self, evidence_type: EvidenceType) -> List[Evidence]:
        """按类型获取证据"""
        return [
            evidence for evidence in self.evidence_store.values()
            if evidence.evidence_type == evidence_type
        ]
    
    def get_evidence_by_source(self, source: str) -> List[Evidence]:
        """按来源获取证据"""
        return [
            evidence for evidence in self.evidence_store.values()
            if evidence.source == source
        ]
    
    def get_validated_evidence(self) -> List[Evidence]:
        """获取已验证的证据"""
        return [
            evidence for evidence in self.evidence_store.values()
            if evidence.validation_status == "validated"
        ]
    
    def get_conflicting_evidence(self) -> List[Tuple[Evidence, Evidence]]:
        """获取冲突证据对"""
        conflicts = []
        
        for evidence1 in self.evidence_store.values():
            for contradictory_id in evidence1.contradictory_evidence:
                if contradictory_id in self.evidence_store:
                    evidence2 = self.evidence_store[contradictory_id]
                    if (evidence2, evidence1) not in conflicts:
                        conflicts.append((evidence1, evidence2))
        
        return conflicts
    
    def analyze_evidence_coverage(self) -> Dict[str, Any]:
        """分析证据覆盖情况"""
        coverage = {
            "total_evidence": len(self.evidence_store),
            "by_type": {},
            "by_source": {},
            "validation_coverage": 0,
            "causal_relationships": len(self.causal_relationships),
            "evidence_chains": len(self.evidence_chains)
        }
        
        # 按类型统计
        for evidence_type in EvidenceType:
            count = len(self.get_evidence_by_type(evidence_type))
            coverage["by_type"][evidence_type.value] = count
        
        # 按来源统计
        sources = set(evidence.source for evidence in self.evidence_store.values())
        for source in sources:
            count = len(self.get_evidence_by_source(source))
            coverage["by_source"][source] = count
        
        # 验证覆盖率
        validated_count = len(self.get_validated_evidence())
        if self.evidence_store:
            coverage["validation_coverage"] = validated_count / len(self.evidence_store)
        
        return coverage
    
    def generate_explanation_components(self) -> List[ExplanationComponent]:
        """生成解释组件"""
        components = []
        
        # 确保权重已计算
        self.calculate_evidence_weights()
        
        for evidence in self.evidence_store.values():
            # 只包含已验证的证据
            if evidence.validation_status != "validated":
                continue
            
            # 计算影响分数（基于权重和置信度）
            impact_score = evidence.weight * evidence.confidence
            
            component = ExplanationComponent(
                factor_name=f"evidence_{evidence.evidence_id}",
                factor_value=evidence.raw_data,
                weight=evidence.weight,
                impact_score=impact_score,
                evidence_type=evidence.evidence_type,
                evidence_source=evidence.source,
                evidence_content=evidence.content,
                metadata={
                    "reliability_score": evidence.reliability_score,
                    "relevance_score": evidence.relevance_score,
                    "freshness_score": evidence.freshness_score,
                    "overall_score": evidence.calculate_overall_score(),
                    "timestamp": evidence.timestamp.isoformat(),
                    "validation_status": evidence.validation_status,
                    "related_evidence_count": len(evidence.related_evidence),
                    "supporting_evidence_count": len(evidence.supporting_evidence),
                    "contradictory_evidence_count": len(evidence.contradictory_evidence)
                }
            )
            components.append(component)
        
        return components
    
    def finalize_collection(self):
        """完成证据收集"""
        self.collection_completed_at = utc_now()
        self.calculate_evidence_weights()
        self._update_stats()
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """获取收集摘要"""
        duration_ms = 0
        if self.collection_completed_at:
            duration_ms = int(
                (self.collection_completed_at - self.collection_started_at).total_seconds() * 1000
            )
        
        return {
            "decision_id": self.decision_id,
            "total_evidence": len(self.evidence_store),
            "evidence_types": {
                evidence_type.value: len(self.get_evidence_by_type(evidence_type))
                for evidence_type in EvidenceType
            },
            "causal_relationships": len(self.causal_relationships),
            "evidence_chains": len(self.evidence_chains),
            "validation_stats": self.stats["validation_stats"],
            "collection_duration_ms": duration_ms,
            "started_at": self.collection_started_at.isoformat(),
            "completed_at": self.collection_completed_at.isoformat() if self.collection_completed_at else None,
            "status": "completed" if self.collection_completed_at else "in_progress"
        }
    
    def _calculate_evidence_scores(self, evidence: Evidence):
        """计算证据各项分数"""
        # 可靠性分数 - 基于来源类型
        source_reliability = {
            "user_input": 0.8,
            "database": 0.9,
            "api": 0.7,
            "memory": 0.6,
            "external": 0.5,
            "calculated": 0.8
        }
        
        for source_type, score in source_reliability.items():
            if source_type in evidence.source.lower():
                evidence.reliability_score = score
                break
        
        # 相关性分数 - 基于证据类型
        type_relevance = {
            EvidenceType.INPUT_DATA: 1.0,
            EvidenceType.CONTEXT: 0.8,
            EvidenceType.MEMORY: 0.7,
            EvidenceType.REASONING_STEP: 0.9,
            EvidenceType.EXTERNAL_SOURCE: 0.6,
            EvidenceType.DOMAIN_KNOWLEDGE: 0.8
        }
        evidence.relevance_score = type_relevance.get(evidence.evidence_type, 0.5)
        
        # 新鲜度分数 - 基于时间
        age_minutes = (utc_now() - evidence.timestamp).total_seconds() / 60
        if age_minutes < 1:
            evidence.freshness_score = 1.0
        elif age_minutes < 60:
            evidence.freshness_score = 0.9
        elif age_minutes < 1440:  # 24小时
            evidence.freshness_score = 0.7
        else:
            evidence.freshness_score = 0.5
    
    def _update_stats(self):
        """更新统计信息"""
        self.stats["total_evidence"] = len(self.evidence_store)
        
        # 按类型统计
        self.stats["by_type"] = {}
        for evidence_type in EvidenceType:
            count = len(self.get_evidence_by_type(evidence_type))
            self.stats["by_type"][evidence_type.value] = count
        
        # 按来源统计
        self.stats["by_source"] = {}
        sources = set(evidence.source for evidence in self.evidence_store.values())
        for source in sources:
            count = len(self.get_evidence_by_source(source))
            self.stats["by_source"][source] = count
        
        # 验证统计
        validation_stats = {"validated": 0, "rejected": 0, "pending": 0}
        for evidence in self.evidence_store.values():
            validation_stats[evidence.validation_status] += 1
        self.stats["validation_stats"] = validation_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "decision_id": self.decision_id,
            "evidence_store": {
                eid: evidence.to_dict() for eid, evidence in self.evidence_store.items()
            },
            "causal_relationships": [rel.to_dict() for rel in self.causal_relationships],
            "evidence_chains": self.evidence_chains,
            "stats": self.stats,
            "collection_started_at": self.collection_started_at.isoformat(),
            "collection_completed_at": self.collection_completed_at.isoformat() if self.collection_completed_at else None
        }
