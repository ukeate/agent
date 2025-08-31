"""
数据标注管理系统

提供标注任务创建、分配、质量控制等功能
"""

import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Any, Optional, Set
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from .models import (
    Base, AnnotationTask, Annotation, AnnotationTaskType, AnnotationStatus,
    AnnotationTaskModel, AnnotationModel, DataRecordModel
)


class AnnotationManager:
    """标注管理器"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        self.logger = logging.getLogger(__name__)
    
    def create_annotation_task(self, task: AnnotationTask) -> str:
        """创建标注任务"""
        
        with self.SessionLocal() as db:
            # 验证数据记录存在
            existing_records = db.query(DataRecordModel.record_id).filter(
                DataRecordModel.record_id.in_(task.record_ids)
            ).all()
            
            existing_record_ids = {record.record_id for record in existing_records}
            missing_records = set(task.record_ids) - existing_record_ids
            
            if missing_records:
                self.logger.warning(f"Some records not found: {missing_records}")
            
            # 创建任务
            db_task = AnnotationTaskModel(
                task_id=task.task_id,
                name=task.name,
                description=task.description,
                task_type=task.task_type,  # 标注任务类型
                data_records=list(existing_record_ids),  # 只包含存在的记录
                annotation_schema=task.schema,  # 标注模式
                guidelines="",  # 标注指南，可后续更新
                assignees=task.annotators,  # 分配的标注员
                created_by="system",  # 创建者，可由调用者传入
                status=task.status.value if hasattr(task.status, 'value') else str(task.status),
                created_at=task.created_at,
                deadline=None  # 截止时间，可后续设置
            )
            
            db.add(db_task)
            db.commit()
            db.refresh(db_task)
            
            self.logger.info(f"Created annotation task: {task.name}")
            return str(db_task.id)
    
    def assign_task(self, task_id: str, user_ids: List[str]) -> bool:
        """分配标注任务给用户"""
        
        with self.SessionLocal() as db:
            db_task = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).first()
            
            if not db_task:
                return False
            
            current_assignees = set(db_task.assignees or [])
            new_assignees = set(user_ids)
            updated_assignees = list(current_assignees.union(new_assignees))
            
            db_task.assignees = updated_assignees
            db_task.updated_at = utc_now()
            
            db.commit()
            
            self.logger.info(f"Assigned task {task_id} to users: {user_ids}")
            return True
    
    def unassign_task(self, task_id: str, user_ids: List[str]) -> bool:
        """取消分配标注任务"""
        
        with self.SessionLocal() as db:
            db_task = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).first()
            
            if not db_task:
                return False
            
            current_assignees = set(db_task.assignees or [])
            users_to_remove = set(user_ids)
            updated_assignees = list(current_assignees - users_to_remove)
            
            db_task.assignees = updated_assignees
            db_task.updated_at = utc_now()
            
            db.commit()
            
            self.logger.info(f"Unassigned task {task_id} from users: {user_ids}")
            return True
    
    def submit_annotation(self, annotation: Annotation) -> str:
        """提交标注结果"""
        
        with self.SessionLocal() as db:
            # 验证任务和记录存在
            task_exists = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == annotation.task_id
            ).first()
            
            if not task_exists:
                raise ValueError(f"Annotation task {annotation.task_id} not found")
            
            record_exists = db.query(DataRecordModel).filter(
                DataRecordModel.record_id == annotation.record_id
            ).first()
            
            if not record_exists:
                raise ValueError(f"Data record {annotation.record_id} not found")
            
            # 检查是否已有标注
            existing = db.query(AnnotationModel).filter(
                AnnotationModel.task_id == annotation.task_id,
                AnnotationModel.record_id == annotation.record_id,
                AnnotationModel.annotator_id == annotation.annotator_id
            ).first()
            
            if existing:
                # 更新现有标注
                existing.annotation_data = annotation.annotation_data
                existing.confidence = annotation.confidence
                existing.time_spent = annotation.time_spent
                existing.status = annotation.status
                existing.updated_at = utc_now()
                db.commit()
                return str(existing.id)
            else:
                # 创建新标注
                db_annotation = AnnotationModel(
                    annotation_id=annotation.annotation_id,
                    task_id=annotation.task_id,
                    record_id=annotation.record_id,
                    annotator_id=annotation.annotator_id,
                    annotation_data=annotation.annotation_data,
                    confidence=annotation.confidence,
                    time_spent=annotation.time_spent,
                    status=annotation.status,
                    created_at=annotation.created_at
                )
                
                db.add(db_annotation)
                db.commit()
                db.refresh(db_annotation)
                
                self.logger.info(f"Submitted annotation: {annotation.annotation_id}")
                return str(db_annotation.id)
    
    def get_annotation_progress(self, task_id: str) -> Dict[str, Any]:
        """获取标注进度"""
        
        with self.SessionLocal() as db:
            task = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).first()
            
            if not task:
                return {}
            
            total_records = len(task.data_records or [])
            
            # 统计已标注记录数
            annotated_records = db.query(AnnotationModel.record_id.distinct()).filter(
                AnnotationModel.task_id == task_id
            ).count()
            
            # 统计各状态的标注数
            status_counts = {}
            for status in ['submitted', 'reviewed', 'approved', 'rejected']:
                count = db.query(AnnotationModel).filter(
                    AnnotationModel.task_id == task_id,
                    AnnotationModel.status == status
                ).count()
                status_counts[status] = count
            
            # 计算标注者统计
            annotator_stats = db.query(
                AnnotationModel.annotator_id,
                func.count(AnnotationModel.id).label('annotation_count'),
                func.avg(AnnotationModel.time_spent).label('avg_time_spent'),
                func.avg(AnnotationModel.confidence).label('avg_confidence')
            ).filter(
                AnnotationModel.task_id == task_id
            ).group_by(AnnotationModel.annotator_id).all()
            
            # 获取最近的标注活动
            recent_annotations = db.query(AnnotationModel).filter(
                AnnotationModel.task_id == task_id
            ).order_by(AnnotationModel.created_at.desc()).limit(5).all()
            
            return {
                'task_id': task_id,
                'task_name': task.name,
                'task_status': task.status,
                'total_records': total_records,
                'annotated_records': annotated_records,
                'progress_percentage': (annotated_records / total_records * 100) if total_records > 0 else 0,
                'status_counts': status_counts,
                'annotator_stats': [
                    {
                        'annotator_id': stat.annotator_id,
                        'annotation_count': stat.annotation_count,
                        'avg_time_spent': float(stat.avg_time_spent) if stat.avg_time_spent else 0,
                        'avg_confidence': float(stat.avg_confidence) if stat.avg_confidence else 0
                    }
                    for stat in annotator_stats
                ],
                'recent_activity': [
                    {
                        'annotation_id': ann.annotation_id,
                        'annotator_id': ann.annotator_id,
                        'record_id': ann.record_id,
                        'status': ann.status,
                        'confidence': ann.confidence,
                        'created_at': ann.created_at.isoformat() if ann.created_at else None
                    }
                    for ann in recent_annotations
                ],
                'created_by': task.created_by,
                'deadline': task.deadline.isoformat() if task.deadline else None
            }
    
    def calculate_inter_annotator_agreement(self, task_id: str) -> Dict[str, float]:
        """计算标注者间一致性"""
        
        with self.SessionLocal() as db:
            # 获取所有标注
            annotations = db.query(AnnotationModel).filter(
                AnnotationModel.task_id == task_id
            ).all()
            
            # 按记录分组
            records_annotations = {}
            for annotation in annotations:
                if annotation.record_id not in records_annotations:
                    records_annotations[annotation.record_id] = []
                records_annotations[annotation.record_id].append(annotation)
            
            # 计算一致性指标
            total_agreements = 0
            total_comparisons = 0
            agreement_details = []
            
            for record_id, record_annotations in records_annotations.items():
                if len(record_annotations) < 2:
                    continue
                
                record_agreements = 0
                record_comparisons = 0
                
                # 计算该记录的所有标注者对之间的一致性
                for i in range(len(record_annotations)):
                    for j in range(i + 1, len(record_annotations)):
                        total_comparisons += 1
                        record_comparisons += 1
                        
                        ann1 = record_annotations[i].annotation_data
                        ann2 = record_annotations[j].annotation_data
                        
                        if self._compare_annotations(ann1, ann2):
                            total_agreements += 1
                            record_agreements += 1
                
                agreement_details.append({
                    'record_id': record_id,
                    'annotator_count': len(record_annotations),
                    'agreement_rate': record_agreements / record_comparisons if record_comparisons > 0 else 0
                })
            
            overall_agreement_rate = total_agreements / total_comparisons if total_comparisons > 0 else 0
            
            # 计算Fleiss' Kappa近似值（简化版）
            kappa_score = self._calculate_simple_kappa(records_annotations)
            
            return {
                'overall_agreement_rate': overall_agreement_rate,
                'total_comparisons': total_comparisons,
                'total_agreements': total_agreements,
                'records_with_multiple_annotations': len([r for r in records_annotations.values() if len(r) >= 2]),
                'simple_kappa': kappa_score,
                'record_details': agreement_details
            }
    
    def _compare_annotations(self, ann1: Dict[str, Any], ann2: Dict[str, Any]) -> bool:
        """比较两个标注是否一致"""
        
        # 如果都是字典，比较关键字段
        if isinstance(ann1, dict) and isinstance(ann2, dict):
            key_fields = ['label', 'category', 'sentiment', 'classification', 'answer']
            
            for field in key_fields:
                if field in ann1 and field in ann2:
                    val1 = ann1[field]
                    val2 = ann2[field]
                    
                    # 对于列表类型，比较集合
                    if isinstance(val1, list) and isinstance(val2, list):
                        return set(val1) == set(val2)
                    # 对于字符串，忽略大小写
                    elif isinstance(val1, str) and isinstance(val2, str):
                        return val1.lower().strip() == val2.lower().strip()
                    # 其他类型直接比较
                    else:
                        return val1 == val2
            
            # 如果没有关键字段匹配，比较整个字典
            return ann1 == ann2
        
        # 非字典类型直接比较
        return ann1 == ann2
    
    def _calculate_simple_kappa(self, records_annotations: Dict[str, List]) -> float:
        """计算简化版Kappa系数"""
        
        if not records_annotations:
            return 0.0
        
        # 收集所有可能的标签
        all_labels = set()
        for annotations in records_annotations.values():
            for ann in annotations:
                if isinstance(ann.annotation_data, dict):
                    label = ann.annotation_data.get('label') or ann.annotation_data.get('category')
                    if label:
                        all_labels.add(str(label).lower())
        
        if len(all_labels) < 2:
            return 1.0  # 如果只有一个标签类别，完全一致
        
        # 计算观察到的一致性
        total_pairs = 0
        agreeing_pairs = 0
        
        for annotations in records_annotations.values():
            if len(annotations) < 2:
                continue
                
            for i in range(len(annotations)):
                for j in range(i + 1, len(annotations)):
                    total_pairs += 1
                    if self._compare_annotations(annotations[i].annotation_data, annotations[j].annotation_data):
                        agreeing_pairs += 1
        
        observed_agreement = agreeing_pairs / total_pairs if total_pairs > 0 else 0
        
        # 简化的期望一致性（假设随机分布）
        expected_agreement = 1.0 / len(all_labels)
        
        # Kappa系数
        if expected_agreement >= 1.0:
            return 0.0
        
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        return max(0.0, min(1.0, kappa))  # 限制在[0,1]范围内
    
    def get_annotation_tasks(
        self, 
        assignee_id: Optional[str] = None,
        status: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取标注任务列表"""
        
        with self.SessionLocal() as db:
            query = db.query(AnnotationTaskModel)
            
            if assignee_id:
                query = query.filter(AnnotationTaskModel.assignees.contains([assignee_id]))
            
            if status:
                query = query.filter(AnnotationTaskModel.status == status)
            
            if created_by:
                query = query.filter(AnnotationTaskModel.created_by == created_by)
            
            query = query.order_by(AnnotationTaskModel.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            tasks = query.all()
            
            return [
                {
                    'id': str(task.id),
                    'task_id': task.task_id,
                    'name': task.name,
                    'description': task.description,
                    'task_type': task.task_type,
                    'data_record_count': len(task.data_records or []),
                    'assignee_count': len(task.assignees or []),
                    'status': task.status,
                    'created_by': task.created_by,
                    'created_at': task.created_at,
                    'deadline': task.deadline,
                    'updated_at': task.updated_at
                }
                for task in tasks
            ]
    
    def get_annotation_task_details(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取标注任务详情"""
        
        with self.SessionLocal() as db:
            task = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).first()
            
            if not task:
                return None
            
            return {
                'id': str(task.id),
                'task_id': task.task_id,
                'name': task.name,
                'description': task.description,
                'task_type': task.task_type,
                'data_records': task.data_records,
                'annotation_schema': task.annotation_schema,
                'guidelines': task.guidelines,
                'assignees': task.assignees,
                'status': task.status,
                'created_by': task.created_by,
                'created_at': task.created_at,
                'deadline': task.deadline,
                'updated_at': task.updated_at
            }
    
    def get_user_annotations(
        self,
        annotator_id: str,
        task_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取用户的标注记录"""
        
        with self.SessionLocal() as db:
            query = db.query(AnnotationModel).filter(
                AnnotationModel.annotator_id == annotator_id
            )
            
            if task_id:
                query = query.filter(AnnotationModel.task_id == task_id)
            
            if status:
                query = query.filter(AnnotationModel.status == status)
            
            query = query.order_by(AnnotationModel.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            annotations = query.all()
            
            return [
                {
                    'id': str(ann.id),
                    'annotation_id': ann.annotation_id,
                    'task_id': ann.task_id,
                    'record_id': ann.record_id,
                    'annotation_data': ann.annotation_data,
                    'confidence': ann.confidence,
                    'time_spent': ann.time_spent,
                    'status': ann.status,
                    'created_at': ann.created_at,
                    'updated_at': ann.updated_at
                }
                for ann in annotations
            ]
    
    def update_task_status(self, task_id: str, new_status: str) -> bool:
        """更新任务状态"""
        
        valid_statuses = [status.value for status in AnnotationStatus]
        if new_status not in valid_statuses:
            self.logger.error(f"Invalid status: {new_status}")
            return False
        
        with self.SessionLocal() as db:
            task = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).first()
            
            if not task:
                return False
            
            task.status = new_status
            task.updated_at = utc_now()
            db.commit()
            
            self.logger.info(f"Updated task {task_id} status to {new_status}")
            return True
    
    def get_quality_control_report(self, task_id: str) -> Dict[str, Any]:
        """生成质量控制报告"""
        
        progress = self.get_annotation_progress(task_id)
        agreement = self.calculate_inter_annotator_agreement(task_id)
        
        with self.SessionLocal() as db:
            # 计算平均标注时间
            avg_time = db.query(func.avg(AnnotationModel.time_spent)).filter(
                AnnotationModel.task_id == task_id,
                AnnotationModel.time_spent.isnot(None)
            ).scalar()
            
            # 计算平均置信度
            avg_confidence = db.query(func.avg(AnnotationModel.confidence)).filter(
                AnnotationModel.task_id == task_id,
                AnnotationModel.confidence.isnot(None)
            ).scalar()
            
            # 异常检测：查找时间过短或过长的标注
            time_outliers = db.query(AnnotationModel).filter(
                AnnotationModel.task_id == task_id,
                AnnotationModel.time_spent.isnot(None)
            ).all()
            
            if time_outliers and avg_time:
                fast_annotations = [a for a in time_outliers if a.time_spent < avg_time * 0.3]
                slow_annotations = [a for a in time_outliers if a.time_spent > avg_time * 3]
            else:
                fast_annotations = []
                slow_annotations = []
            
            # 低置信度标注
            low_confidence_annotations = db.query(AnnotationModel).filter(
                AnnotationModel.task_id == task_id,
                AnnotationModel.confidence.isnot(None),
                AnnotationModel.confidence < 0.6
            ).all()
            
            return {
                'task_id': task_id,
                'progress_summary': progress,
                'agreement_analysis': agreement,
                'quality_metrics': {
                    'average_annotation_time': float(avg_time) if avg_time else 0,
                    'average_confidence': float(avg_confidence) if avg_confidence else 0,
                    'fast_annotations_count': len(fast_annotations),
                    'slow_annotations_count': len(slow_annotations),
                    'low_confidence_count': len(low_confidence_annotations)
                },
                'potential_issues': {
                    'very_fast_annotations': [
                        {
                            'annotation_id': a.annotation_id,
                            'annotator_id': a.annotator_id,
                            'time_spent': a.time_spent
                        }
                        for a in fast_annotations[:10]  # 只返回前10个
                    ],
                    'very_slow_annotations': [
                        {
                            'annotation_id': a.annotation_id,
                            'annotator_id': a.annotator_id,
                            'time_spent': a.time_spent
                        }
                        for a in slow_annotations[:10]
                    ],
                    'low_confidence_annotations': [
                        {
                            'annotation_id': a.annotation_id,
                            'annotator_id': a.annotator_id,
                            'confidence': a.confidence
                        }
                        for a in low_confidence_annotations[:10]
                    ]
                },
                'generated_at': utc_now().isoformat()
            }
    
    def delete_annotation_task(self, task_id: str) -> bool:
        """删除标注任务（包括相关标注）"""
        
        with self.SessionLocal() as db:
            # 删除相关标注
            db.query(AnnotationModel).filter(
                AnnotationModel.task_id == task_id
            ).delete()
            
            # 删除任务
            task_deleted = db.query(AnnotationTaskModel).filter(
                AnnotationTaskModel.task_id == task_id
            ).delete()
            
            db.commit()
            
            if task_deleted:
                self.logger.info(f"Deleted annotation task: {task_id}")
                return True
            else:
                return False