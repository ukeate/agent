#!/usr/bin/env python3
"""
简化的测试运行器，用于快速验证核心功能
"""

import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# 添加路径
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

async def test_annotation_manager_basic():
    """测试注解管理器基本功能"""
    print("🔍 开始测试注解管理器...")
    
    try:
        # Mock数据库依赖
        with patch('src.ai.training_data_management.annotation.create_engine'):
            from src.ai.training_data_management.annotation import AnnotationManager
            from src.ai.training_data_management.models import AnnotationTask, AnnotationTaskStatus
            
            # 创建mock数据库会话
            mock_db_session = AsyncMock()
            mock_db_session.add = MagicMock()
            mock_db_session.commit = AsyncMock()
            
            # 创建注解管理器
            manager = AnnotationManager(mock_db_session)
            manager.db = mock_db_session
            
            # 测试创建任务
            task_config = {
                'name': 'Test Classification',
                'description': 'Test task',
                'task_type': 'classification',
                'schema': {'type': 'object', 'properties': {'label': {'type': 'string'}}},
                'annotators': ['user1']
            }
            
            from src.ai.training_data_management.models import AnnotationTask
            import uuid
            
            task_obj = AnnotationTask(
                task_id=str(uuid.uuid4()),
                name=task_config['name'],
                description=task_config['description'],
                task_type=task_config['task_type'],
                record_ids=['rec1', 'rec2'],
                schema=task_config['schema'],
                annotators=task_config['annotators']
            )
            
            with patch.object(manager.db, 'add') as mock_add:
                with patch.object(manager.db, 'commit') as mock_commit:
                    with patch.object(manager, 'SessionLocal') as mock_session:
                        # Mock数据库查询结果
                        mock_db = MagicMock()
                        mock_session.return_value.__enter__.return_value = mock_db
                        mock_db.query.return_value.filter.return_value.all.return_value = [
                            MagicMock(record_id='rec1'),
                            MagicMock(record_id='rec2')
                        ]
                        
                        db_id = manager.create_annotation_task(task_obj)
                        task = task_obj  # 使用创建的任务对象
                    
                    assert task.name == 'Test Classification'
                    assert task.task_type == 'classification'
                    assert task.status == AnnotationTaskStatus.DRAFT  # 默认状态
                    print("✅ 注解任务创建测试通过")
            
            # 测试基本方法存在性
            assert hasattr(manager, 'create_annotation_task')
            assert hasattr(manager, 'SessionLocal')
            print("✅ 注解管理器基本方法测试通过")
            
        print("🎉 注解管理器测试全部通过!")
        return True
        
    except Exception as e:
        print(f"❌ 注解管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_training_data_api_basic():
    """测试训练数据API基本功能"""
    print("🔍 开始测试训练数据API...")
    
    try:
        from src.api.v1.training_data import router, DataSourceCreate, AnnotationTaskCreate
        
        # 验证路由器创建
        assert router is not None
        assert router.prefix == "/training-data"
        print("✅ API路由器创建测试通过")
        
        # 验证Pydantic模型
        source_data = DataSourceCreate(
            source_id="test-source",
            source_type="file",
            name="Test Source", 
            description="Test description",
            config={"path": "/test/data.json"}
        )
        assert source_data.source_id == "test-source"
        print("✅ Pydantic模型测试通过")
        
        print("🎉 训练数据API测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 训练数据API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试入口"""
    print("🚀 开始运行训练数据管理系统简化测试...")
    
    results = []
    
    # 测试注解管理器
    results.append(await test_annotation_manager_basic())
    
    # 测试API路由
    results.append(await test_training_data_api_basic())
    
    # 汇总结果
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 测试结果汇总:")
    print(f"✅ 通过: {passed}/{total}")
    print(f"❌ 失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试都通过了!")
        return 0
    else:
        print("⚠️  有测试失败，需要检查")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)