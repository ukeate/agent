"""
训练数据标注系统核心实现

这个模块提供完整的数据标注功能，包括：
- 标注任务管理
- 标注分配和进度跟踪
- 质量控制和一致性检查
- 标注者协作工具
"""

import asyncio
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import (
    AnnotationTaskModel, AnnotationModel, DataRecordModel,
    AnnotationTaskType, AnnotationStatus
)
from .core import (
    AnnotationTask, Annotation, DataRecord, TaskStatistics, 
    AnnotationProgress

)

class AssignmentStrategy(Enum):
    """标注分配策略"""
    ROUND_ROBIN = "round_robin"
    BALANCED = "balanced"
    EXPERTISE_BASED = "expertise_based"
    RANDOM = "random"

class QualityMetric(Enum):
    """质量评估指标"""
    AGREEMENT = "agreement"
    CONFIDENCE = "confidence"
    CONSISTENCY = "consistency"
    SPEED = "speed"

@dataclass
class AnnotatorProfile:
    """标注者档案"""
    annotator_id: str
    name: str
    expertise_areas: List[str]
    skill_level: float  # 0.0 - 1.0
    avg_time_per_task: float  # 秒
    consistency_score: float  # 0.0 - 1.0
    total_annotations: int
    accuracy_rate: float  # 0.0 - 1.0

@dataclass
class QualityReport:
    """质量报告"""
    task_id: str
    overall_score: float
    agreement_metrics: Dict[str, float]
    consistency_metrics: Dict[str, float]
    annotator_performance: List[Dict[str, Any]]
    recommendations: List[str]

class AnnotationManager:
    """标注管理器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.annotator_profiles: Dict[str, AnnotatorProfile] = {}
    
    async def create_task(
        self,
        task_data: AnnotationTask,
        assignment_strategy: AssignmentStrategy = AssignmentStrategy.BALANCED
    ) -> str:
        """创建标注任务"""
        task_model = AnnotationTaskModel(
            task_id=task_data.task_id,
            name=task_data.name,
            description=task_data.description,
            task_type=task_data.task_type.value,
            data_records=task_data.data_records,
            annotation_schema=task_data.annotation_schema,
            guidelines=task_data.guidelines,
            assignees=task_data.assignees,
            created_by=task_data.created_by,
            status=task_data.status.value,
            deadline=task_data.deadline
        )
        
        self.db.add(task_model)
        await self.db.commit()
        
        # 自动分配标注者
        await self._assign_annotators(task_data.task_id, assignment_strategy)
        
        return task_data.task_id
    
    async def _assign_annotators(
        self,
        task_id: str,
        strategy: AssignmentStrategy
    ) -> None:
        """根据策略分配标注者"""
        stmt = select(AnnotationTaskModel).where(AnnotationTaskModel.task_id == task_id)
        result = await self.db.execute(stmt)
        task = result.scalar_one()
        
        if strategy == AssignmentStrategy.BALANCED:
            assignments = await self._balanced_assignment(task)
        elif strategy == AssignmentStrategy.EXPERTISE_BASED:
            assignments = await self._expertise_based_assignment(task)
        elif strategy == AssignmentStrategy.RANDOM:
            assignments = await self._random_assignment(task)
        else:  # ROUND_ROBIN
            assignments = await self._round_robin_assignment(task)
        
        task.assignees = list(assignments.keys())
        await self.db.commit()
    
    async def _balanced_assignment(
        self,
        task: AnnotationTaskModel
    ) -> Dict[str, List[str]]:
        """平衡分配策略"""
        annotators = await self._get_available_annotators(task.task_type)
        if not annotators:
            return {}
        
        # 计算每个标注者的工作量
        workloads = {}
        for annotator_id in annotators:
            stmt = select(func.count(AnnotationModel.id)).where(
                and_(
                    AnnotationModel.annotator_id == annotator_id,
                    AnnotationModel.status.in_(['pending', 'in_progress'])
                )
            )
            result = await self.db.execute(stmt)
            workloads[annotator_id] = result.scalar() or 0
        
        # 按工作量排序，优先分配给工作量少的标注者
        sorted_annotators = sorted(workloads.keys(), key=lambda x: workloads[x])
        
        # 分配记录
        assignments = {aid: [] for aid in sorted_annotators}
        for i, record_id in enumerate(task.data_records):
            annotator_id = sorted_annotators[i % len(sorted_annotators)]
            assignments[annotator_id].append(record_id)
        
        return assignments
    
    async def _expertise_based_assignment(
        self,
        task: AnnotationTaskModel
    ) -> Dict[str, List[str]]:
        """基于专业程度的分配策略"""
        task_type = task.task_type
        suitable_annotators = []
        
        for annotator_id, profile in self.annotator_profiles.items():
            if task_type in profile.expertise_areas:
                suitable_annotators.append((annotator_id, profile.skill_level))
        
        if not suitable_annotators:
            return await self._balanced_assignment(task)
        
        # 按技能水平排序
        suitable_annotators.sort(key=lambda x: x[1], reverse=True)
        
        # 分配给技能水平最高的标注者
        assignments = {}
        records_per_annotator = len(task.data_records) // len(suitable_annotators)
        
        for i, (annotator_id, _) in enumerate(suitable_annotators):
            start_idx = i * records_per_annotator
            end_idx = start_idx + records_per_annotator
            if i == len(suitable_annotators) - 1:  # 最后一个标注者处理剩余记录
                end_idx = len(task.data_records)
            assignments[annotator_id] = task.data_records[start_idx:end_idx]
        
        return assignments
    
    async def _random_assignment(
        self,
        task: AnnotationTaskModel
    ) -> Dict[str, List[str]]:
        """随机分配策略"""
        annotators = await self._get_available_annotators(task.task_type)
        if not annotators:
            return {}
        
        import random
        random.shuffle(task.data_records)
        
        assignments = {aid: [] for aid in annotators}
        for i, record_id in enumerate(task.data_records):
            annotator_id = annotators[i % len(annotators)]
            assignments[annotator_id].append(record_id)
        
        return assignments
    
    async def _round_robin_assignment(
        self,
        task: AnnotationTaskModel
    ) -> Dict[str, List[str]]:
        """轮询分配策略"""
        annotators = await self._get_available_annotators(task.task_type)
        if not annotators:
            return {}
        
        assignments = {aid: [] for aid in annotators}
        for i, record_id in enumerate(task.data_records):
            annotator_id = annotators[i % len(annotators)]
            assignments[annotator_id].append(record_id)
        
        return assignments
    
    async def _get_available_annotators(self, task_type: str) -> List[str]:
        """获取可用的标注者列表"""
        # 这里可以根据标注者的可用性、专业程度等过滤
        return list(self.annotator_profiles.keys())
    
    async def submit_annotation(
        self,
        annotation_data: Annotation
    ) -> str:
        """提交标注结果"""
        annotation_model = AnnotationModel(
            annotation_id=annotation_data.annotation_id,
            task_id=annotation_data.task_id,
            record_id=annotation_data.record_id,
            annotator_id=annotation_data.annotator_id,
            annotation_data=annotation_data.annotation_data,
            confidence=annotation_data.confidence,
            time_spent=annotation_data.time_spent,
            status=annotation_data.status
        )
        
        self.db.add(annotation_model)
        await self.db.commit()
        
        # 更新任务进度
        await self._update_task_progress(annotation_data.task_id)
        
        return annotation_data.annotation_id
    
    async def _update_task_progress(self, task_id: str) -> None:
        """更新任务进度"""
        # 统计已完成的标注数量
        completed_count_stmt = select(func.count(AnnotationModel.id)).where(
            and_(
                AnnotationModel.task_id == task_id,
                AnnotationModel.status == 'submitted'
            )
        )
        completed_result = await self.db.execute(completed_count_stmt)
        completed_count = completed_result.scalar() or 0
        
        # 获取总记录数
        task_stmt = select(AnnotationTaskModel).where(AnnotationTaskModel.task_id == task_id)
        task_result = await self.db.execute(task_stmt)
        task = task_result.scalar_one()
        
        total_records = len(task.data_records)
        if total_records > 0:
            progress = completed_count / total_records
            if progress >= 1.0:
                task.status = AnnotationStatus.COMPLETED.value
                await self.db.commit()
    
    async def get_task_progress(self, task_id: str) -> AnnotationProgress:
        """获取任务进度"""
        task_stmt = select(AnnotationTaskModel).where(AnnotationTaskModel.task_id == task_id)
        task_result = await self.db.execute(task_stmt)
        task = task_result.scalar_one()
        
        total_records = len(task.data_records)
        
        # 统计各状态的标注数量
        status_counts_stmt = select(
            AnnotationModel.status,
            func.count(AnnotationModel.id)
        ).where(
            AnnotationModel.task_id == task_id
        ).group_by(AnnotationModel.status)
        
        status_result = await self.db.execute(status_counts_stmt)
        status_distribution = dict(status_result.fetchall())
        
        annotated_records = sum(status_distribution.values())
        progress_percentage = (annotated_records / total_records * 100) if total_records > 0 else 0
        
        # 标注者绩效统计
        annotator_performance = await self._get_annotator_performance(task_id)
        
        # 估算完成时间
        estimated_completion = await self._estimate_completion_time(task_id)
        
        return AnnotationProgress(
            task_id=task_id,
            total_records=total_records,
            annotated_records=annotated_records,
            progress_percentage=progress_percentage,
            status_distribution=status_distribution,
            annotator_performance=annotator_performance,
            estimated_completion=estimated_completion
        )
    
    async def _get_annotator_performance(
        self,
        task_id: str
    ) -> List[Dict[str, Any]]:
        """获取标注者绩效统计"""
        performance_stmt = select(
            AnnotationModel.annotator_id,
            func.count(AnnotationModel.id).label('count'),
            func.avg(AnnotationModel.time_spent).label('avg_time'),
            func.avg(AnnotationModel.confidence).label('avg_confidence')
        ).where(
            AnnotationModel.task_id == task_id
        ).group_by(AnnotationModel.annotator_id)
        
        result = await self.db.execute(performance_stmt)
        performance_data = []
        
        for row in result.fetchall():
            performance_data.append({
                'annotator_id': row.annotator_id,
                'annotations_count': row.count,
                'avg_time_per_annotation': row.avg_time or 0,
                'avg_confidence': row.avg_confidence or 0
            })
        
        return performance_data
    
    async def _estimate_completion_time(
        self,
        task_id: str
    ) -> Optional[datetime]:
        """估算任务完成时间"""
        task_stmt = select(AnnotationTaskModel).where(AnnotationTaskModel.task_id == task_id)
        task_result = await self.db.execute(task_stmt)
        task = task_result.scalar_one()
        
        if task.deadline:
            return task.deadline
        
        # 基于历史数据估算
        remaining_records = len(task.data_records)
        completed_stmt = select(func.count(AnnotationModel.id)).where(
            and_(
                AnnotationModel.task_id == task_id,
                AnnotationModel.status == 'submitted'
            )
        )
        completed_result = await self.db.execute(completed_stmt)
        completed_count = completed_result.scalar() or 0
        
        remaining_records -= completed_count
        
        if remaining_records <= 0:
            return utc_now()
        
        # 计算平均标注时间
        avg_time_stmt = select(func.avg(AnnotationModel.time_spent)).where(
            AnnotationModel.task_id == task_id
        )
        avg_time_result = await self.db.execute(avg_time_stmt)
        avg_time = avg_time_result.scalar() or 300  # 默认5分钟
        
        # 估算完成时间
        estimated_seconds = remaining_records * avg_time
        estimated_completion = utc_now()
        estimated_completion = estimated_completion.replace(
            second=estimated_completion.second + int(estimated_seconds)
        )
        
        return estimated_completion

class QualityController:
    """质量控制器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def calculate_inter_annotator_agreement(
        self,
        task_id: str
    ) -> Dict[str, float]:
        """计算标注者间一致性"""
        # 获取所有标注结果
        annotations_stmt = select(AnnotationModel).where(
            AnnotationModel.task_id == task_id
        )
        result = await self.db.execute(annotations_stmt)
        annotations = result.scalars().all()
        
        # 按记录ID分组
        record_annotations = {}
        for annotation in annotations:
            if annotation.record_id not in record_annotations:
                record_annotations[annotation.record_id] = []
            record_annotations[annotation.record_id].append(annotation)
        
        # 计算各种一致性指标
        agreements = {
            'fleiss_kappa': self._calculate_fleiss_kappa(record_annotations),
            'cohens_kappa': self._calculate_cohens_kappa(record_annotations),
            'percentage_agreement': self._calculate_percentage_agreement(record_annotations)
        }
        
        return agreements
    
    def _calculate_fleiss_kappa(
        self,
        record_annotations: Dict[str, List[AnnotationModel]]
    ) -> float:
        """计算Fleiss' Kappa"""
        if len(record_annotations) < 2:
            return 0.0
        
        # 简化实现，实际项目中需要更复杂的计算
        agreements = []
        for record_id, annotations in record_annotations.items():
            if len(annotations) >= 2:
                labels = [ann.annotation_data.get('label') for ann in annotations]
                agreement = len(set(labels)) == 1
                agreements.append(1.0 if agreement else 0.0)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _calculate_cohens_kappa(
        self,
        record_annotations: Dict[str, List[AnnotationModel]]
    ) -> float:
        """计算Cohen's Kappa"""
        if len(record_annotations) < 2:
            return 0.0
        
        # 简化实现
        agreements = []
        for record_id, annotations in record_annotations.items():
            if len(annotations) == 2:
                label1 = annotations[0].annotation_data.get('label')
                label2 = annotations[1].annotation_data.get('label')
                agreements.append(1.0 if label1 == label2 else 0.0)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _calculate_percentage_agreement(
        self,
        record_annotations: Dict[str, List[AnnotationModel]]
    ) -> float:
        """计算百分比一致性"""
        total_comparisons = 0
        agreements = 0
        
        for record_id, annotations in record_annotations.items():
            if len(annotations) >= 2:
                labels = [ann.annotation_data.get('label') for ann in annotations]
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        total_comparisons += 1
                        if labels[i] == labels[j]:
                            agreements += 1
        
        return agreements / total_comparisons if total_comparisons > 0 else 0.0
    
    async def generate_quality_report(self, task_id: str) -> QualityReport:
        """生成质量报告"""
        # 计算一致性指标
        agreement_metrics = await self.calculate_inter_annotator_agreement(task_id)
        
        # 计算一致性指标
        consistency_metrics = await self._calculate_consistency_metrics(task_id)
        
        # 标注者绩效分析
        annotator_performance = await self._analyze_annotator_performance(task_id)
        
        # 计算总体评分
        overall_score = self._calculate_overall_score(
            agreement_metrics, consistency_metrics, annotator_performance
        )
        
        # 生成建议
        recommendations = self._generate_recommendations(
            agreement_metrics, consistency_metrics, annotator_performance
        )
        
        return QualityReport(
            task_id=task_id,
            overall_score=overall_score,
            agreement_metrics=agreement_metrics,
            consistency_metrics=consistency_metrics,
            annotator_performance=annotator_performance,
            recommendations=recommendations
        )
    
    async def _calculate_consistency_metrics(self, task_id: str) -> Dict[str, float]:
        """计算一致性指标"""
        # 获取每个标注者的标注历史
        annotator_consistency = {}
        
        annotations_stmt = select(AnnotationModel).where(
            AnnotationModel.task_id == task_id
        ).order_by(AnnotationModel.annotator_id, AnnotationModel.created_at)
        
        result = await self.db.execute(annotations_stmt)
        annotations = result.scalars().all()
        
        # 按标注者分组
        annotator_annotations = {}
        for annotation in annotations:
            if annotation.annotator_id not in annotator_annotations:
                annotator_annotations[annotation.annotator_id] = []
            annotator_annotations[annotation.annotator_id].append(annotation)
        
        # 计算每个标注者的一致性
        for annotator_id, anns in annotator_annotations.items():
            labels = [ann.annotation_data.get('label') for ann in anns]
            confidences = [ann.confidence or 0.5 for ann in anns]
            
            # 计算标签分布的一致性
            label_consistency = 1.0 - (len(set(labels)) / len(labels)) if labels else 0.0
            
            # 计算置信度的一致性（标准差的倒数）
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
            confidence_consistency = 1.0 / (1.0 + confidence_std)
            
            annotator_consistency[annotator_id] = {
                'label_consistency': label_consistency,
                'confidence_consistency': confidence_consistency
            }
        
        # 计算平均一致性
        avg_label_consistency = np.mean([
            metrics['label_consistency'] 
            for metrics in annotator_consistency.values()
        ]) if annotator_consistency else 0.0
        
        avg_confidence_consistency = np.mean([
            metrics['confidence_consistency'] 
            for metrics in annotator_consistency.values()
        ]) if annotator_consistency else 0.0
        
        return {
            'avg_label_consistency': avg_label_consistency,
            'avg_confidence_consistency': avg_confidence_consistency,
            'annotator_level': annotator_consistency
        }
    
    async def _analyze_annotator_performance(
        self,
        task_id: str
    ) -> List[Dict[str, Any]]:
        """分析标注者绩效"""
        performance_stmt = select(
            AnnotationModel.annotator_id,
            func.count(AnnotationModel.id).label('total_annotations'),
            func.avg(AnnotationModel.confidence).label('avg_confidence'),
            func.avg(AnnotationModel.time_spent).label('avg_time'),
            func.min(AnnotationModel.created_at).label('first_annotation'),
            func.max(AnnotationModel.created_at).label('last_annotation')
        ).where(
            AnnotationModel.task_id == task_id
        ).group_by(AnnotationModel.annotator_id)
        
        result = await self.db.execute(performance_stmt)
        performance_data = []
        
        for row in result.fetchall():
            # 计算标注速度
            if row.first_annotation and row.last_annotation:
                time_span = (row.last_annotation - row.first_annotation).total_seconds()
                annotations_per_hour = (row.total_annotations / (time_span / 3600)) if time_span > 0 else 0
            else:
                annotations_per_hour = 0
            
            performance_data.append({
                'annotator_id': row.annotator_id,
                'total_annotations': row.total_annotations,
                'avg_confidence': row.avg_confidence or 0.0,
                'avg_time_per_annotation': row.avg_time or 0.0,
                'annotations_per_hour': annotations_per_hour,
                'quality_score': min(1.0, (row.avg_confidence or 0.5) * 1.5)  # 简化的质量评分
            })
        
        return performance_data
    
    def _calculate_overall_score(
        self,
        agreement_metrics: Dict[str, float],
        consistency_metrics: Dict[str, float],
        annotator_performance: List[Dict[str, Any]]
    ) -> float:
        """计算总体质量评分"""
        # 各指标权重
        weights = {
            'agreement': 0.4,
            'consistency': 0.3,
            'performance': 0.3
        }
        
        # 计算一致性评分
        agreement_score = np.mean(list(agreement_metrics.values()))
        
        # 计算一致性评分
        consistency_score = consistency_metrics.get('avg_label_consistency', 0.0)
        
        # 计算绩效评分
        if annotator_performance:
            performance_scores = [p['quality_score'] for p in annotator_performance]
            performance_score = np.mean(performance_scores)
        else:
            performance_score = 0.0
        
        # 加权平均
        overall_score = (
            agreement_score * weights['agreement'] +
            consistency_score * weights['consistency'] +
            performance_score * weights['performance']
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _generate_recommendations(
        self,
        agreement_metrics: Dict[str, float],
        consistency_metrics: Dict[str, float],
        annotator_performance: List[Dict[str, Any]]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于一致性指标的建议
        avg_agreement = np.mean(list(agreement_metrics.values()))
        if avg_agreement < 0.7:
            recommendations.append("标注者间一致性较低，建议加强培训或细化标注指南")
        
        # 基于一致性的建议
        avg_consistency = consistency_metrics.get('avg_label_consistency', 0.0)
        if avg_consistency < 0.8:
            recommendations.append("标注者内部一致性不足，建议进行标注质量复核")
        
        # 基于绩效的建议
        if annotator_performance:
            low_confidence_annotators = [
                p for p in annotator_performance 
                if p['avg_confidence'] < 0.6
            ]
            if low_confidence_annotators:
                recommendations.append(f"有{len(low_confidence_annotators)}名标注者置信度较低，建议额外培训")
            
            slow_annotators = [
                p for p in annotator_performance 
                if p['avg_time_per_annotation'] > 600  # 10分钟
            ]
            if slow_annotators:
                recommendations.append(f"有{len(slow_annotators)}名标注者速度较慢，建议优化标注流程")
        
        if not recommendations:
            recommendations.append("标注质量良好，请继续保持")
        
        return recommendations
