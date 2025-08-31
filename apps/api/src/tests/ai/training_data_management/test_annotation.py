"""
注解管理器测试
"""

import pytest
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.ai.training_data_management.models import (
    DataRecord, AnnotationTask, Annotation, AnnotationStatus, AnnotationTaskStatus
)
from src.ai.training_data_management.annotation import AnnotationManager


class TestAnnotationManager:
    """注解管理器测试"""
    
    @pytest.fixture
    def annotation_manager(self):
        # 创建一个mock的注解管理器，绕过数据库初始化
        with patch('src.ai.training_data_management.annotation.create_engine'):
            mock_db_session = AsyncMock()
            manager = AnnotationManager(mock_db_session)
            manager.db = mock_db_session
            return manager
    
    @pytest.fixture
    def sample_records(self):
        return [
            DataRecord(
                record_id="rec1",
                source_id="src1",
                raw_data={'text': 'Sample text 1'},
                processed_data={'content': 'Sample text 1'},
                status='processed'
            ),
            DataRecord(
                record_id="rec2",
                source_id="src1",
                raw_data={'text': 'Sample text 2'},
                processed_data={'content': 'Sample text 2'},
                status='processed'
            )
        ]
    
    @pytest.fixture
    def sample_task(self):
        return AnnotationTask(
            task_id="task1",
            name="Test Task",
            description="Test annotation task",
            task_type="classification",
            schema={
                'type': 'object',
                'properties': {
                    'label': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}
                },
                'required': ['label']
            },
            annotators=['user1', 'user2'],
            status=AnnotationTaskStatus.ACTIVE
        )
    
    @pytest.mark.asyncio
    async def test_create_annotation_task(self, annotation_manager, sample_records):
        """测试创建注解任务"""
        
        task_config = {
            'name': 'Text Classification',
            'description': 'Classify text sentiment',
            'task_type': 'classification',
            'schema': {
                'type': 'object',
                'properties': {
                    'label': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']}
                }
            },
            'annotators': ['user1', 'user2']
        }
        
        with patch.object(annotation_manager.db, 'add') as mock_add:
            with patch.object(annotation_manager.db, 'commit') as mock_commit:
                task = await annotation_manager.create_annotation_task(
                    record_ids=[r.record_id for r in sample_records],
                    **task_config
                )
                
                assert task.name == 'Text Classification'
                assert task.task_type == 'classification'
                assert task.annotators == ['user1', 'user2']
                assert task.status == AnnotationTaskStatus.ACTIVE
                assert len(task.record_ids) == 2
                
                mock_add.assert_called_once()
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assign_annotation_task(self, annotation_manager, sample_task):
        """测试分配注解任务"""
        
        with patch.object(annotation_manager, 'get_annotation_task') as mock_get_task:
            mock_get_task.return_value = sample_task
            
            with patch.object(annotation_manager.db, 'commit') as mock_commit:
                updated_task = await annotation_manager.assign_annotation_task(
                    task_id="task1",
                    annotators=['user3', 'user4']
                )
                
                assert updated_task.annotators == ['user3', 'user4']
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_submit_annotation(self, annotation_manager, sample_task):
        """测试提交注解"""
        
        annotation_data = {
            'label': 'positive',
            'confidence': 0.9
        }
        
        with patch.object(annotation_manager, 'get_annotation_task') as mock_get_task:
            mock_get_task.return_value = sample_task
            
            with patch.object(annotation_manager.db, 'add') as mock_add:
                with patch.object(annotation_manager.db, 'commit') as mock_commit:
                    annotation = await annotation_manager.submit_annotation(
                        task_id="task1",
                        record_id="rec1",
                        annotator_id="user1",
                        annotation_data=annotation_data
                    )
                    
                    assert annotation.task_id == "task1"
                    assert annotation.record_id == "rec1"
                    assert annotation.annotator_id == "user1"
                    assert annotation.annotation_data == annotation_data
                    assert annotation.status == AnnotationStatus.SUBMITTED
                    
                    mock_add.assert_called_once()
                    mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_annotation(self, annotation_manager):
        """测试注解验证"""
        
        schema = {
            'type': 'object',
            'properties': {
                'label': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}
            },
            'required': ['label']
        }
        
        # 有效注解
        valid_annotation = {
            'label': 'positive',
            'confidence': 0.8
        }
        assert annotation_manager.validate_annotation(valid_annotation, schema) is True
        
        # 缺少必需字段
        invalid_annotation1 = {
            'confidence': 0.8
        }
        assert annotation_manager.validate_annotation(invalid_annotation1, schema) is False
        
        # 无效枚举值
        invalid_annotation2 = {
            'label': 'invalid',
            'confidence': 0.8
        }
        assert annotation_manager.validate_annotation(invalid_annotation2, schema) is False
        
        # 无效数值范围
        invalid_annotation3 = {
            'label': 'positive',
            'confidence': 1.5
        }
        assert annotation_manager.validate_annotation(invalid_annotation3, schema) is False
    
    @pytest.mark.asyncio
    async def test_get_task_progress(self, annotation_manager):
        """测试获取任务进度"""
        
        # 模拟数据库查询
        mock_annotations = [
            MagicMock(
                task_id="task1",
                record_id="rec1",
                annotator_id="user1",
                status=AnnotationStatus.SUBMITTED
            ),
            MagicMock(
                task_id="task1",
                record_id="rec1",
                annotator_id="user2",
                status=AnnotationStatus.SUBMITTED
            ),
            MagicMock(
                task_id="task1",
                record_id="rec2",
                annotator_id="user1",
                status=AnnotationStatus.IN_PROGRESS
            )
        ]
        
        with patch.object(annotation_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_annotations
            
            progress = await annotation_manager.get_task_progress("task1")
            
            expected_progress = {
                'task_id': 'task1',
                'total_records': 2,
                'total_annotations': 3,
                'completed_annotations': 2,
                'in_progress_annotations': 1,
                'completion_rate': 1.0,  # rec1完成了所有注解，rec2部分完成
                'annotator_progress': {
                    'user1': {'completed': 1, 'in_progress': 1, 'total': 2},
                    'user2': {'completed': 1, 'in_progress': 0, 'total': 1}
                }
            }
            
            assert progress['task_id'] == expected_progress['task_id']
            assert progress['total_records'] == expected_progress['total_records']
            assert progress['total_annotations'] == expected_progress['total_annotations']
    
    @pytest.mark.asyncio
    async def test_calculate_inter_annotator_agreement(self, annotation_manager):
        """测试计算注解者间一致性"""
        
        # 模拟注解数据
        annotations = [
            # record 1
            {'record_id': 'rec1', 'annotator_id': 'user1', 'annotation_data': {'label': 'positive'}},
            {'record_id': 'rec1', 'annotator_id': 'user2', 'annotation_data': {'label': 'positive'}},
            {'record_id': 'rec1', 'annotator_id': 'user3', 'annotation_data': {'label': 'negative'}},
            
            # record 2
            {'record_id': 'rec2', 'annotator_id': 'user1', 'annotation_data': {'label': 'negative'}},
            {'record_id': 'rec2', 'annotator_id': 'user2', 'annotation_data': {'label': 'negative'}},
            {'record_id': 'rec2', 'annotator_id': 'user3', 'annotation_data': {'label': 'negative'}},
            
            # record 3
            {'record_id': 'rec3', 'annotator_id': 'user1', 'annotation_data': {'label': 'neutral'}},
            {'record_id': 'rec3', 'annotator_id': 'user2', 'annotation_data': {'label': 'positive'}},
            {'record_id': 'rec3', 'annotator_id': 'user3', 'annotation_data': {'label': 'neutral'}},
        ]
        
        agreement = annotation_manager.calculate_inter_annotator_agreement(
            annotations, 
            field='label'
        )
        
        # 验证Fleiss' Kappa计算
        assert 'fleiss_kappa' in agreement
        assert 'pairwise_agreements' in agreement
        assert 'overall_agreement_rate' in agreement
        
        # 验证整体一致性率
        expected_overall_rate = 2/3  # rec2完全一致，rec1和rec3部分一致
        assert abs(agreement['overall_agreement_rate'] - expected_overall_rate) < 0.1
    
    @pytest.mark.asyncio
    async def test_get_annotation_quality_report(self, annotation_manager):
        """测试获取注解质量报告"""
        
        mock_annotations = [
            MagicMock(
                annotator_id="user1",
                annotation_data={'label': 'positive', 'confidence': 0.9},
                created_at=utc_now(),
                review_score=0.9
            ),
            MagicMock(
                annotator_id="user1",
                annotation_data={'label': 'negative', 'confidence': 0.8},
                created_at=utc_now(),
                review_score=0.8
            ),
            MagicMock(
                annotator_id="user2",
                annotation_data={'label': 'positive', 'confidence': 0.7},
                created_at=utc_now(),
                review_score=0.7
            )
        ]
        
        with patch.object(annotation_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_annotations
            
            with patch.object(annotation_manager, 'calculate_inter_annotator_agreement') as mock_agreement:
                mock_agreement.return_value = {
                    'fleiss_kappa': 0.6,
                    'overall_agreement_rate': 0.8
                }
                
                report = await annotation_manager.get_annotation_quality_report("task1")
                
                assert 'task_id' in report
                assert 'annotator_statistics' in report
                assert 'quality_metrics' in report
                assert 'inter_annotator_agreement' in report
                
                # 验证注解者统计
                user1_stats = report['annotator_statistics']['user1']
                assert user1_stats['annotation_count'] == 2
                assert user1_stats['average_confidence'] == 0.85
                assert user1_stats['average_review_score'] == 0.85
    
    def test_calculate_fleiss_kappa(self, annotation_manager):
        """测试Fleiss' Kappa计算"""
        
        # 简单的测试数据
        # 3个注解者，2个记录，3个类别（positive, negative, neutral）
        annotations_matrix = [
            [2, 1, 0],  # record 1: 2个positive, 1个negative, 0个neutral
            [0, 3, 0],  # record 2: 0个positive, 3个negative, 0个neutral
        ]
        
        kappa = annotation_manager._calculate_fleiss_kappa(annotations_matrix)
        
        # Fleiss' Kappa应该在-1到1之间
        assert -1 <= kappa <= 1
        
        # 对于这个例子，应该有较高的一致性（record 2完全一致）
        assert kappa > 0
    
    def test_calculate_pairwise_agreement(self, annotation_manager):
        """测试成对一致性计算"""
        
        annotations_by_record = {
            'rec1': ['positive', 'positive', 'negative'],
            'rec2': ['negative', 'negative', 'negative'],
            'rec3': ['neutral', 'positive', 'neutral']
        }
        
        pairwise_agreements = annotation_manager._calculate_pairwise_agreement(
            annotations_by_record
        )
        
        assert 'user1_user2' in pairwise_agreements
        assert 'user1_user3' in pairwise_agreements
        assert 'user2_user3' in pairwise_agreements
        
        # 验证一致性率在0-1范围内
        for pair, rate in pairwise_agreements.items():
            assert 0 <= rate <= 1
    
    @pytest.mark.asyncio
    async def test_export_annotations(self, annotation_manager):
        """测试导出注解"""
        
        mock_annotations = [
            MagicMock(
                task_id="task1",
                record_id="rec1",
                annotator_id="user1",
                annotation_data={'label': 'positive'},
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                status=AnnotationStatus.SUBMITTED
            ),
            MagicMock(
                task_id="task1",
                record_id="rec2",
                annotator_id="user1",
                annotation_data={'label': 'negative'},
                created_at=datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc),
                status=AnnotationStatus.SUBMITTED
            )
        ]
        
        with patch.object(annotation_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_annotations
            
            # 导出为JSON
            json_export = await annotation_manager.export_annotations(
                task_id="task1",
                format='json'
            )
            
            assert isinstance(json_export, str)
            
            # 导出为CSV
            csv_export = await annotation_manager.export_annotations(
                task_id="task1",
                format='csv'
            )
            
            assert isinstance(csv_export, str)
            assert 'task_id,record_id,annotator_id' in csv_export
    
    @pytest.mark.asyncio
    async def test_batch_update_annotation_status(self, annotation_manager):
        """测试批量更新注解状态"""
        
        annotation_ids = ['ann1', 'ann2', 'ann3']
        new_status = AnnotationStatus.APPROVED
        
        with patch.object(annotation_manager.db, 'query') as mock_query:
            mock_filter = mock_query.return_value.filter
            mock_update = mock_filter.return_value.update
            
            with patch.object(annotation_manager.db, 'commit') as mock_commit:
                updated_count = await annotation_manager.batch_update_annotation_status(
                    annotation_ids=annotation_ids,
                    status=new_status
                )
                
                mock_update.assert_called_once_with({'status': new_status})
                mock_commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_annotation_conflicts(self, annotation_manager):
        """测试获取注解冲突"""
        
        # 模拟有冲突的注解数据
        mock_annotations = [
            MagicMock(
                record_id='rec1',
                annotator_id='user1',
                annotation_data={'label': 'positive'}
            ),
            MagicMock(
                record_id='rec1', 
                annotator_id='user2',
                annotation_data={'label': 'negative'}
            ),
            MagicMock(
                record_id='rec2',
                annotator_id='user1',
                annotation_data={'label': 'positive'}
            ),
            MagicMock(
                record_id='rec2',
                annotator_id='user2',
                annotation_data={'label': 'positive'}
            )
        ]
        
        with patch.object(annotation_manager.db, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_annotations
            
            conflicts = await annotation_manager.get_annotation_conflicts(
                task_id="task1",
                field='label'
            )
            
            # rec1应该有冲突，rec2没有冲突
            conflict_records = [c['record_id'] for c in conflicts]
            assert 'rec1' in conflict_records
            assert 'rec2' not in conflict_records
            
            # 验证冲突详情
            rec1_conflict = next(c for c in conflicts if c['record_id'] == 'rec1')
            assert len(rec1_conflict['conflicting_annotations']) == 2
            assert rec1_conflict['conflict_type'] == 'disagreement'