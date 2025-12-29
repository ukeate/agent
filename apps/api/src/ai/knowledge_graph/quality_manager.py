"""
图谱质量管理器
实现图谱完整性、一致性、准确性评估和质量分数计算
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from .graph_database import Neo4jGraphDatabase
from .graph_operations import GraphOperations

from src.core.logging import get_logger
logger = get_logger(__name__)

class QualityDimension(str, Enum):
    """质量维度"""
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    REDUNDANCY = "redundancy"
    FRESHNESS = "freshness"
    CONNECTIVITY = "connectivity"

@dataclass
class QualityMetrics:
    """图谱质量指标"""
    completeness_score: float      # 完整性分数
    consistency_score: float       # 一致性分数
    accuracy_score: float          # 准确性分数
    redundancy_score: float        # 冗余度分数
    freshness_score: float         # 时效性分数
    connectivity_score: float      # 连通性分数
    overall_quality_score: float   # 综合质量分数
    timestamp: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "accuracy_score": self.accuracy_score,
            "redundancy_score": self.redundancy_score,
            "freshness_score": self.freshness_score,
            "connectivity_score": self.connectivity_score,
            "overall_quality_score": self.overall_quality_score,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class QualityIssue:
    """质量问题"""
    issue_id: str
    issue_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_entities: List[str]
    recommended_actions: List[str]
    confidence: float
    detected_at: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "description": self.description,
            "affected_entities": self.affected_entities,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat()
        }

class QualityManager:
    """图谱质量管理器"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase, graph_ops: GraphOperations):
        self.graph_db = graph_db
        self.graph_ops = graph_ops
        self.quality_rules = self._load_quality_rules()
        self.quality_thresholds = {
            "completeness": 0.8,
            "consistency": 0.9,
            "accuracy": 0.85,
            "redundancy": 0.1,  # 低冗余更好
            "freshness": 0.7,
            "connectivity": 0.6
        }
    
    def _load_quality_rules(self) -> Dict[str, Any]:
        """加载质量评估规则"""
        return {
            "completeness_rules": [
                {
                    "rule": "entities_have_required_properties",
                    "required_properties": ["canonical_form", "type", "confidence"]
                },
                {
                    "rule": "relations_have_required_properties", 
                    "required_properties": ["type", "confidence"]
                }
            ],
            "consistency_rules": [
                {
                    "rule": "no_self_referential_relations",
                    "description": "实体不能与自己建立关系"
                },
                {
                    "rule": "confidence_in_valid_range",
                    "description": "置信度必须在0-1之间"
                },
                {
                    "rule": "no_contradictory_relations",
                    "description": "不能存在逻辑矛盾的关系"
                }
            ],
            "accuracy_rules": [
                {
                    "rule": "high_confidence_entities",
                    "threshold": 0.7,
                    "description": "高质量实体应有高置信度"
                }
            ]
        }
    
    async def calculate_quality_score(self) -> QualityMetrics:
        """计算图谱质量分数"""
        try:
            logger.info("开始计算图谱质量分数")
            
            # 并行执行各项质量检查
            tasks = [
                self._check_completeness(),
                self._check_consistency(), 
                self._check_accuracy(),
                self._check_redundancy(),
                self._check_freshness(),
                self._check_connectivity()
            ]
            
            results = await asyncio.gather(*tasks)
            
            completeness, consistency, accuracy, redundancy, freshness, connectivity = results
            
            # 加权计算综合分数
            overall_score = (
                completeness * 0.20 +      # 完整性 20%
                consistency * 0.25 +       # 一致性 25%
                accuracy * 0.20 +          # 准确性 20%
                (1 - redundancy) * 0.15 +  # 冗余度 15% (越低越好)
                freshness * 0.10 +         # 时效性 10%
                connectivity * 0.10        # 连通性 10%
            )
            
            metrics = QualityMetrics(
                completeness_score=completeness,
                consistency_score=consistency,
                accuracy_score=accuracy,
                redundancy_score=redundancy,
                freshness_score=freshness,
                connectivity_score=connectivity,
                overall_quality_score=overall_score
            )
            
            logger.info(f"质量分数计算完成: {overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"计算质量分数失败: {str(e)}")
            # 返回默认值
            return QualityMetrics(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                redundancy_score=1.0,
                freshness_score=0.0,
                connectivity_score=0.0,
                overall_quality_score=0.0
            )
    
    async def _check_completeness(self) -> float:
        """检查完整性"""
        try:
            # 检查实体必需属性完整性
            entity_completeness_query = """
            MATCH (e:Entity)
            WITH count(e) as total,
                 count(CASE WHEN e.canonical_form IS NOT NULL THEN 1 END) as has_canonical,
                 count(CASE WHEN e.type IS NOT NULL THEN 1 END) as has_type,
                 count(CASE WHEN e.confidence IS NOT NULL THEN 1 END) as has_confidence
            RETURN 
                CASE 
                    WHEN total = 0 THEN 0.0
                    ELSE (has_canonical + has_type + has_confidence) / (total * 3.0)
                END as completeness_score
            """
            
            result = await self.graph_db.execute_read_query(entity_completeness_query)
            entity_score = result[0]["completeness_score"] if result else 0.0
            
            # 检查关系必需属性完整性
            relation_completeness_query = """
            MATCH ()-[r:RELATION]->()
            WITH count(r) as total,
                 count(CASE WHEN r.type IS NOT NULL THEN 1 END) as has_type,
                 count(CASE WHEN r.confidence IS NOT NULL THEN 1 END) as has_confidence
            RETURN 
                CASE 
                    WHEN total = 0 THEN 1.0
                    ELSE (has_type + has_confidence) / (total * 2.0)
                END as completeness_score
            """
            
            result = await self.graph_db.execute_read_query(relation_completeness_query)
            relation_score = result[0]["completeness_score"] if result else 1.0
            
            # 综合完整性分数
            return (entity_score + relation_score) / 2.0
            
        except Exception as e:
            logger.error(f"完整性检查失败: {str(e)}")
            return 0.0
    
    async def _check_consistency(self) -> float:
        """检查一致性"""
        try:
            total_issues = 0
            total_checks = 0
            
            # 检查自引用关系
            self_ref_query = """
            MATCH (e:Entity)-[r:RELATION]->(e)
            RETURN count(r) as self_ref_count
            """
            result = await self.graph_db.execute_read_query(self_ref_query)
            self_ref_count = result[0]["self_ref_count"] if result else 0
            total_issues += self_ref_count
            total_checks += 1
            
            # 检查置信度范围
            invalid_confidence_query = """
            MATCH (e:Entity)
            WHERE e.confidence < 0 OR e.confidence > 1
            RETURN count(e) as invalid_count
            UNION ALL
            MATCH ()-[r:RELATION]->()
            WHERE r.confidence < 0 OR r.confidence > 1
            RETURN count(r) as invalid_count
            """
            result = await self.graph_db.execute_read_query(invalid_confidence_query)
            invalid_confidence_count = sum(record["invalid_count"] for record in result)
            total_issues += invalid_confidence_count
            total_checks += 1
            
            # 检查重复实体
            duplicate_query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.id < e2.id 
            AND e1.canonical_form = e2.canonical_form 
            AND e1.type = e2.type
            RETURN count(*) as duplicate_count
            """
            result = await self.graph_db.execute_read_query(duplicate_query)
            duplicate_count = result[0]["duplicate_count"] if result else 0
            total_issues += duplicate_count
            total_checks += 1
            
            # 计算一致性分数
            if total_checks == 0:
                return 1.0
            
            # 假设每个检查项最多有100个问题算作完全不一致
            max_issues_per_check = 100
            consistency_score = max(0.0, 1.0 - (total_issues / (total_checks * max_issues_per_check)))
            
            return consistency_score
            
        except Exception as e:
            logger.error(f"一致性检查失败: {str(e)}")
            return 0.0
    
    async def _check_accuracy(self) -> float:
        """检查准确性"""
        try:
            # 基于置信度分布评估准确性
            confidence_distribution_query = """
            MATCH (e:Entity)
            WITH e.confidence as conf
            RETURN 
                avg(conf) as avg_confidence,
                count(CASE WHEN conf >= 0.8 THEN 1 END) as high_conf_count,
                count(CASE WHEN conf >= 0.5 THEN 1 END) as medium_conf_count,
                count(e) as total_count
            """
            
            result = await self.graph_db.execute_read_query(confidence_distribution_query)
            
            if not result or result[0]["total_count"] == 0:
                return 0.0
            
            data = result[0]
            avg_confidence = data["avg_confidence"] or 0.0
            high_conf_ratio = data["high_conf_count"] / data["total_count"]
            medium_conf_ratio = data["medium_conf_count"] / data["total_count"]
            
            # 准确性分数综合考虑平均置信度和高置信度实体比例
            accuracy_score = (
                avg_confidence * 0.6 +          # 平均置信度权重60%
                high_conf_ratio * 0.3 +         # 高置信度实体比例权重30%
                medium_conf_ratio * 0.1          # 中等置信度实体比例权重10%
            )
            
            return min(1.0, accuracy_score)
            
        except Exception as e:
            logger.error(f"准确性检查失败: {str(e)}")
            return 0.0
    
    async def _check_redundancy(self) -> float:
        """检查冗余度"""
        try:
            # 检查重复实体的比例
            total_entities_query = "MATCH (e:Entity) RETURN count(e) as total"
            result = await self.graph_db.execute_read_query(total_entities_query)
            total_entities = result[0]["total"] if result else 0
            
            if total_entities == 0:
                return 0.0
            
            # 查找潜在重复实体
            duplicates_query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.id < e2.id 
            AND e1.canonical_form = e2.canonical_form 
            AND e1.type = e2.type
            RETURN count(DISTINCT e1.id) + count(DISTINCT e2.id) as duplicate_entities
            """
            
            result = await self.graph_db.execute_read_query(duplicates_query)
            duplicate_entities = result[0]["duplicate_entities"] if result else 0
            
            # 检查重复关系
            duplicate_relations_query = """
            MATCH (a)-[r1:RELATION]->(b), (a)-[r2:RELATION]->(b)
            WHERE r1.id < r2.id AND r1.type = r2.type
            RETURN count(*) as duplicate_relations
            """
            
            result = await self.graph_db.execute_read_query(duplicate_relations_query)
            duplicate_relations = result[0]["duplicate_relations"] if result else 0
            
            # 计算冗余度 (冗余实体数 + 冗余关系数) / 总实体数
            redundancy_score = (duplicate_entities + duplicate_relations) / max(total_entities, 1)
            
            return min(1.0, redundancy_score)  # 限制在0-1范围
            
        except Exception as e:
            logger.error(f"冗余度检查失败: {str(e)}")
            return 0.0
    
    async def _check_freshness(self) -> float:
        """检查时效性"""
        try:
            # 计算最近更新实体的比例
            seven_days_ago = (utc_now() - timedelta(days=7)).isoformat()
            thirty_days_ago = (utc_now() - timedelta(days=30)).isoformat()
            
            freshness_query = """
            MATCH (e:Entity)
            WITH count(e) as total,
                 count(CASE WHEN e.updated_at > $seven_days_ago THEN 1 END) as recent_7d,
                 count(CASE WHEN e.updated_at > $thirty_days_ago THEN 1 END) as recent_30d
            RETURN 
                CASE 
                    WHEN total = 0 THEN 0.0
                    ELSE (recent_7d * 1.0 + recent_30d * 0.5) / total
                END as freshness_score
            """
            
            result = await self.graph_db.execute_read_query(
                freshness_query,
                {"seven_days_ago": seven_days_ago, "thirty_days_ago": thirty_days_ago}
            )
            
            freshness_score = result[0]["freshness_score"] if result else 0.0
            return min(1.0, freshness_score)
            
        except Exception as e:
            logger.error(f"时效性检查失败: {str(e)}")
            return 0.0
    
    async def _check_connectivity(self) -> float:
        """检查连通性"""
        try:
            # 检查图的连通性
            connectivity_query = """
            MATCH (e:Entity)
            WITH count(e) as total_entities
            MATCH (connected:Entity)
            WHERE exists((connected)-[:RELATION]-()) OR exists(()-[:RELATION]->(connected))
            WITH total_entities, count(connected) as connected_entities
            RETURN 
                CASE 
                    WHEN total_entities = 0 THEN 0.0
                    ELSE connected_entities * 1.0 / total_entities
                END as connectivity_ratio
            """
            
            result = await self.graph_db.execute_read_query(connectivity_query)
            connectivity_ratio = result[0]["connectivity_ratio"] if result else 0.0
            
            # 计算平均度数（连接数）
            avg_degree_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r:RELATION]-()
            WITH e, count(r) as degree
            RETURN avg(degree) as avg_degree
            """
            
            result = await self.graph_db.execute_read_query(avg_degree_query)
            avg_degree = result[0]["avg_degree"] if result else 0.0
            
            # 综合连通性分数
            connectivity_score = (connectivity_ratio * 0.7 + min(1.0, avg_degree / 5.0) * 0.3)
            
            return connectivity_score
            
        except Exception as e:
            logger.error(f"连通性检查失败: {str(e)}")
            return 0.0
    
    async def detect_quality_issues(self) -> List[QualityIssue]:
        """检测质量问题"""
        issues = []
        
        try:
            # 检测重复实体
            duplicate_issues = await self._detect_duplicate_entities()
            issues.extend(duplicate_issues)
            
            # 检测低质量实体
            low_quality_issues = await self._detect_low_quality_entities()
            issues.extend(low_quality_issues)
            
            # 检测孤立节点
            isolated_issues = await self._detect_isolated_nodes()
            issues.extend(isolated_issues)
            
            # 检测无效数据
            invalid_issues = await self._detect_invalid_data()
            issues.extend(invalid_issues)
            
        except Exception as e:
            logger.error(f"检测质量问题失败: {str(e)}")
        
        return issues
    
    async def _detect_duplicate_entities(self) -> List[QualityIssue]:
        """检测重复实体"""
        issues = []
        
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id 
        AND e1.canonical_form = e2.canonical_form 
        AND e1.type = e2.type
        RETURN e1.id as entity1_id, e2.id as entity2_id, e1.canonical_form as canonical_form
        LIMIT 100
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="duplicate_entity",
                severity="medium",
                description=f"检测到重复实体: {record['canonical_form']}",
                affected_entities=[record["entity1_id"], record["entity2_id"]],
                recommended_actions=["merge_entities", "review_manually"],
                confidence=0.9
            )
            issues.append(issue)
        
        return issues
    
    async def _detect_low_quality_entities(self) -> List[QualityIssue]:
        """检测低质量实体"""
        issues = []
        
        query = """
        MATCH (e:Entity)
        WHERE e.confidence < 0.3
        RETURN e.id as entity_id, e.canonical_form as canonical_form, e.confidence as confidence
        LIMIT 50
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="low_quality_entity",
                severity="low",
                description=f"低质量实体 (置信度: {record['confidence']:.2f}): {record['canonical_form']}",
                affected_entities=[record["entity_id"]],
                recommended_actions=["review_manually", "improve_extraction"],
                confidence=1.0 - record["confidence"]
            )
            issues.append(issue)
        
        return issues
    
    async def _detect_isolated_nodes(self) -> List[QualityIssue]:
        """检测孤立节点"""
        issues = []
        
        query = """
        MATCH (e:Entity)
        WHERE NOT exists((e)-[:RELATION]-()) AND NOT exists(()-[:RELATION]->(e))
        RETURN e.id as entity_id, e.canonical_form as canonical_form
        LIMIT 50
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="isolated_node",
                severity="low",
                description=f"孤立节点: {record['canonical_form']}",
                affected_entities=[record["entity_id"]],
                recommended_actions=["find_relations", "remove_if_irrelevant"],
                confidence=0.8
            )
            issues.append(issue)
        
        return issues
    
    async def _detect_invalid_data(self) -> List[QualityIssue]:
        """检测无效数据"""
        issues = []
        
        # 检测无效置信度
        query = """
        MATCH (e:Entity)
        WHERE e.confidence < 0 OR e.confidence > 1
        RETURN e.id as entity_id, e.canonical_form as canonical_form, e.confidence as confidence
        LIMIT 20
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            issue = QualityIssue(
                issue_id=str(uuid.uuid4()),
                issue_type="invalid_confidence",
                severity="high",
                description=f"无效置信度 ({record['confidence']}): {record['canonical_form']}",
                affected_entities=[record["entity_id"]],
                recommended_actions=["fix_confidence_value"],
                confidence=1.0
            )
            issues.append(issue)
        
        return issues
    
    async def generate_quality_report(self) -> Dict[str, Any]:
        """生成质量报告"""
        try:
            # 计算质量分数
            quality_metrics = await self.calculate_quality_score()
            
            # 检测质量问题
            quality_issues = await self.detect_quality_issues()
            
            # 生成建议
            recommendations = self._generate_recommendations(quality_metrics, quality_issues)
            
            # 获取基础统计
            stats_result = await self.graph_ops.get_graph_statistics()
            stats = stats_result.data[0]["statistics"] if stats_result.success and stats_result.data else {}
            
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": utc_now().isoformat(),
                "quality_metrics": quality_metrics.to_dict(),
                "quality_issues": {
                    "total_issues": len(quality_issues),
                    "issues_by_severity": self._group_issues_by_severity(quality_issues),
                    "issues_by_type": self._group_issues_by_type(quality_issues),
                    "detailed_issues": [issue.to_dict() for issue in quality_issues[:20]]  # 只返回前20个
                },
                "graph_statistics": stats,
                "recommendations": recommendations,
                "quality_assessment": self._assess_quality_level(quality_metrics.overall_quality_score)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成质量报告失败: {str(e)}")
            return {
                "error": str(e),
                "generated_at": utc_now().isoformat()
            }
    
    def _group_issues_by_severity(self, issues: List[QualityIssue]) -> Dict[str, int]:
        """按严重程度分组问题"""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        return severity_counts
    
    def _group_issues_by_type(self, issues: List[QualityIssue]) -> Dict[str, int]:
        """按类型分组问题"""
        type_counts = {}
        for issue in issues:
            type_counts[issue.issue_type] = type_counts.get(issue.issue_type, 0) + 1
        return type_counts
    
    def _generate_recommendations(self, 
                                metrics: QualityMetrics, 
                                issues: List[QualityIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于质量分数生成建议
        if metrics.completeness_score < 0.8:
            recommendations.append("提高数据完整性：确保所有实体和关系都有必需的属性")
        
        if metrics.consistency_score < 0.9:
            recommendations.append("解决数据一致性问题：检查并修复重复实体和矛盾关系")
        
        if metrics.accuracy_score < 0.85:
            recommendations.append("提高数据准确性：改进实体识别和关系抽取的准确率")
        
        if metrics.redundancy_score > 0.1:
            recommendations.append("减少数据冗余：合并重复实体和关系")
        
        if metrics.freshness_score < 0.7:
            recommendations.append("更新过时数据：定期刷新和维护知识图谱内容")
        
        if metrics.connectivity_score < 0.6:
            recommendations.append("改善图连通性：增加实体间的关系连接")
        
        # 基于问题类型生成建议
        issue_types = set(issue.issue_type for issue in issues)
        
        if "duplicate_entity" in issue_types:
            recommendations.append("实施实体去重：使用相似度算法识别和合并重复实体")
        
        if "low_quality_entity" in issue_types:
            recommendations.append("提高抽取质量：使用更高质量的数据源和抽取模型")
        
        if "isolated_node" in issue_types:
            recommendations.append("建立关系连接：为孤立实体寻找适当的关系")
        
        return recommendations
    
    def _assess_quality_level(self, overall_score: float) -> Dict[str, str]:
        """评估质量等级"""
        if overall_score >= 0.9:
            return {"level": "excellent", "description": "图谱质量优秀"}
        elif overall_score >= 0.8:
            return {"level": "good", "description": "图谱质量良好"}
        elif overall_score >= 0.7:
            return {"level": "fair", "description": "图谱质量一般，需要改进"}
        elif overall_score >= 0.6:
            return {"level": "poor", "description": "图谱质量较差，需要重点改进"}
        else:
            return {"level": "very_poor", "description": "图谱质量很差，需要全面重构"}
