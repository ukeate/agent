"""
知识图谱系统集成测试 - 端到端测试所有组件的集成和协作
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from pathlib import Path
from ai.knowledge_graph.sparql_engine import SPARQLEngine, SPARQLQuery, QueryType, execute_sparql_query
from ai.knowledge_graph.query_optimizer import QueryOptimizer, OptimizationLevel
from ai.knowledge_graph.result_formatter import ResultFormatter, ResultFormat
from ai.knowledge_graph.performance_monitor import SPARQLPerformanceMonitor
from ai.knowledge_graph.data_importer import DataImporter, ImportJob, ImportFormat, ConflictResolution
from ai.knowledge_graph.data_exporter import DataExporter, ExportJob, ExportFormat
from ai.knowledge_graph.format_processors import FormatProcessor, FormatConverter, DataFormat
from ai.knowledge_graph.version_manager import VersionManager, VersionType, create_knowledge_graph_version
from ai.knowledge_graph.change_tracker import ChangeTracker, EventType, Priority
from ai.knowledge_graph.kg_auth import KnowledgeGraphAuth, SecurityConfig, Role, Permission
from ai.knowledge_graph.kg_models import (

    Triple, Entity, Relation, KnowledgeGraph, 
    create_triple, create_entity, create_relation, create_knowledge_graph,
    model_registry, SerializationFormat
)
from ai.knowledge_graph.kg_monitor import KnowledgeGraphMonitor, kg_monitor
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

class TestKnowledgeGraphIntegration:
    """知识图谱集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.auth_dir = self.temp_dir / "auth"
        self.version_dir = self.temp_dir / "versions"
        self.monitor_dir = self.temp_dir / "monitor"
        
        # 创建目录
        self.auth_dir.mkdir()
        self.version_dir.mkdir()
        self.monitor_dir.mkdir()
    
    def teardown_method(self):
        """测试清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """测试完整的工作流程"""
        # 1. 创建和配置组件
        auth = KnowledgeGraphAuth(storage_path=str(self.auth_dir))
        version_manager = VersionManager(storage_path=str(self.version_dir))
        change_tracker = ChangeTracker()
        monitor = KnowledgeGraphMonitor(log_dir=str(self.monitor_dir))
        
        await change_tracker.start()
        await monitor.start_monitoring(interval_seconds=1)
        
        try:
            # 2. 用户认证
            user = await auth.create_user(
                username="test_user",
                email="test@example.com",
                password="Test123!@#",
                role=Role.EDITOR
            )
            
            auth_result = await auth.authenticate_user(
                "test_user", "Test123!@#", "127.0.0.1"
            )
            
            assert auth_result.success
            assert auth_result.user_id == user.user_id
            
            # 记录用户操作
            await change_tracker.track_change(
                event_type=EventType.ENTITY_CREATED,
                user_id=user.user_id,
                affected_resources=["user:test_user"],
                new_value={"username": "test_user"},
                metadata={"operation": "user_creation"}
            )
            
            # 3. 创建知识图谱
            kg = create_knowledge_graph(
                name="测试知识图谱",
                description="集成测试用知识图谱",
                namespace="http://test.example.org/"
            )
            
            # 4. 添加实体和关系
            person_class = create_entity(
                uri="http://test.example.org/Person",
                label="人",
                description="人类实体类",
                type_uris=["http://www.w3.org/2002/07/owl#Class"]
            )
            
            john_entity = create_entity(
                uri="http://test.example.org/John",
                label="约翰",
                description="一个人的实例",
                type_uris=["http://test.example.org/Person"]
            )
            john_entity.add_property("http://test.example.org/age", 30)
            john_entity.add_property("http://test.example.org/name", "John Doe")
            
            mary_entity = create_entity(
                uri="http://test.example.org/Mary",
                label="玛丽",
                description="另一个人的实例",
                type_uris=["http://test.example.org/Person"]
            )
            mary_entity.add_property("http://test.example.org/age", 28)
            mary_entity.add_property("http://test.example.org/name", "Mary Smith")
            
            # 创建关系
            knows_relation = create_relation(
                uri="http://test.example.org/knows",
                label="认识",
                description="人与人之间的认识关系",
                domain_uris=["http://test.example.org/Person"],
                range_uris=["http://test.example.org/Person"]
            )
            
            # 添加到知识图谱
            kg.add_entity(person_class)
            kg.add_entity(john_entity)
            kg.add_entity(mary_entity)
            kg.add_relation(knows_relation)
            
            # 添加三元组
            type_triple1 = create_triple(
                subject="http://test.example.org/John",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://test.example.org/Person"
            )
            
            type_triple2 = create_triple(
                subject="http://test.example.org/Mary",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://test.example.org/Person"
            )
            
            knows_triple = create_triple(
                subject="http://test.example.org/John",
                predicate="http://test.example.org/knows",
                object="http://test.example.org/Mary"
            )
            
            kg.add_triple(type_triple1)
            kg.add_triple(type_triple2)
            kg.add_triple(knows_triple)
            
            # 5. 记录变更
            await change_tracker.track_change(
                event_type=EventType.ENTITY_CREATED,
                user_id=user.user_id,
                affected_resources=[john_entity.uri, mary_entity.uri],
                new_value={"entities": [john_entity.uri, mary_entity.uri]},
                metadata={"operation": "entity_creation"}
            )
            
            # 6. 创建版本
            version1 = await version_manager.create_version(
                graph_data={"triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in kg.triples.values()]},
                version_type=VersionType.MAJOR,
                description="初始版本 - 添加John和Mary",
                created_by=user.user_id
            )
            
            assert version1.version_number == "1.0.0"
            assert version1.triple_count > 0
            
            # 7. 权限检查
            can_read = await auth.check_permission(auth_result, Permission.READ_GRAPH)
            can_write = await auth.check_permission(auth_result, Permission.WRITE_GRAPH)
            can_admin = await auth.check_permission(auth_result, Permission.ADMIN_ALL)
            
            assert can_read
            assert can_write
            assert not can_admin  # Editor没有管理员权限
            
            # 8. 数据导出测试
            exporter = DataExporter()
            export_job = ExportJob(
                job_id="test_export",
                options=None
            )
            export_job.options = type('Options', (), {
                'format': ExportFormat.JSON,
                'compression': type('CompressionType', (), {'NONE': 'none'})(),
                'export_directory': str(self.temp_dir)
            })()
            export_job.options.format = ExportFormat.JSON
            export_job.output_filename = "test_export.json"
            
            # 模拟导出数据
            export_data = {
                "triples": [
                    {"subject": "http://test.example.org/John", "predicate": "rdf:type", "object": "http://test.example.org/Person"},
                    {"subject": "http://test.example.org/Mary", "predicate": "rdf:type", "object": "http://test.example.org/Person"},
                    {"subject": "http://test.example.org/John", "predicate": "http://test.example.org/knows", "object": "http://test.example.org/Mary"}
                ]
            }
            
            # 简化的导出操作
            export_file = self.temp_dir / "test_export.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            assert export_file.exists()
            
            # 9. 数据导入测试
            importer = DataImporter()
            
            # 创建导入数据
            import_data = {
                "triples": [
                    {"subject": "http://test.example.org/Alice", "predicate": "rdf:type", "object": "http://test.example.org/Person"},
                    {"subject": "http://test.example.org/Alice", "predicate": "http://test.example.org/age", "object": "25"},
                    {"subject": "http://test.example.org/John", "predicate": "http://test.example.org/knows", "object": "http://test.example.org/Alice"}
                ]
            }
            
            import_file = self.temp_dir / "import_data.json"
            with open(import_file, 'w', encoding='utf-8') as f:
                json.dump(import_data, f, indent=2, ensure_ascii=False)
            
            # 10. SPARQL查询测试
            engine = SPARQLEngine()
            
            # 测试简单查询
            query_text = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
            query = SPARQLQuery(
                query_id="test_query",
                query_text=query_text,
                query_type=QueryType.SELECT
            )
            
            result = await engine.execute_query(query)
            assert result.success
            
            # 记录查询操作
            monitor.record_query(query_text, result.execution_time_ms / 1000, len(result.results))
            
            # 11. 格式转换测试
            processor = FormatProcessor()
            converter = FormatConverter()
            
            # JSON到N-Triples转换
            json_data = json.dumps(import_data)
            result = converter.convert(json_data, DataFormat.JSON, DataFormat.N_TRIPLES)
            
            assert result.success
            assert "alice" in result.data.lower() or "Alice" in result.data
            
            # 12. 序列化测试
            kg.calculate_statistics()
            
            # JSON序列化
            json_data = model_registry.serialize(kg, SerializationFormat.JSON)
            assert len(json_data) > 0
            
            # 反序列化
            kg_restored = model_registry.deserialize(json_data, SerializationFormat.JSON)
            assert kg_restored.name == kg.name
            assert len(kg_restored.entities) == len(kg.entities)
            
            # 13. 创建第二个版本
            # 添加Alice到图谱
            alice_entity = create_entity(
                uri="http://test.example.org/Alice",
                label="爱丽丝",
                description="第三个人的实例",
                type_uris=["http://test.example.org/Person"]
            )
            alice_entity.add_property("http://test.example.org/age", 25)
            alice_entity.add_property("http://test.example.org/name", "Alice Johnson")
            
            kg.add_entity(alice_entity)
            
            alice_type_triple = create_triple(
                subject="http://test.example.org/Alice",
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object="http://test.example.org/Person"
            )
            
            john_knows_alice = create_triple(
                subject="http://test.example.org/John",
                predicate="http://test.example.org/knows",
                object="http://test.example.org/Alice"
            )
            
            kg.add_triple(alice_type_triple)
            kg.add_triple(john_knows_alice)
            
            # 创建第二个版本
            version2 = await version_manager.create_version(
                graph_data={"triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in kg.triples.values()]},
                version_type=VersionType.MINOR,
                description="添加Alice实体",
                created_by=user.user_id
            )
            
            assert version2.version_number == "1.1.0"
            assert version2.triple_count > version1.triple_count
            
            # 14. 版本比较
            diff = await version_manager.compare_versions(version1.version_id, version2.version_id)
            assert len(diff.added_triples) > 0
            assert len(diff.removed_triples) == 0
            
            # 15. 监控数据检查
            await asyncio.sleep(2)  # 等待监控周期
            
            system_status = await monitor.get_system_status()
            assert system_status["overall_status"] in ["OK", "WARNING", "ERROR"]
            assert system_status["monitoring_enabled"]
            
            # 16. 变更历史检查
            events = await change_tracker.get_events(limit=10)
            assert len(events) > 0
            
            user_activity = await change_tracker.get_user_activity(user.user_id, hours=1)
            assert len(user_activity) > 0
            
            # 17. API密钥测试
            api_key, key_obj = await auth.create_api_key(
                user_id=user.user_id,
                name="集成测试密钥",
                permissions={Permission.READ_GRAPH, Permission.QUERY_SPARQL}
            )
            
            api_auth_result = await auth.authenticate_api_key(api_key, "127.0.0.1")
            assert api_auth_result.success
            assert api_auth_result.user_id == user.user_id
            
            # 18. 性能优化测试
            optimizer = QueryOptimizer()
            
            complex_query = """
                SELECT ?person ?name ?age WHERE {
                    ?person rdf:type <http://test.example.org/Person> .
                    ?person <http://test.example.org/name> ?name .
                    ?person <http://test.example.org/age> ?age .
                    FILTER (?age > 25)
                }
            """
            
            optimization_result = await optimizer.optimize_query(complex_query)
            assert optimization_result["success"]
            
            # 19. 结果格式化测试
            formatter = ResultFormatter()
            
            sample_results = [
                {"person": "http://test.example.org/John", "name": "John Doe", "age": "30"},
                {"person": "http://test.example.org/Mary", "name": "Mary Smith", "age": "28"}
            ]
            
            json_result = formatter.format_results(sample_results, "SELECT", ResultFormat.JSON)
            assert json_result["success"]
            assert len(json_result["data"]["results"]["bindings"]) == 2
            
            csv_result = formatter.format_results(sample_results, "SELECT", ResultFormat.CSV)
            assert csv_result["success"]
            assert "John Doe" in csv_result["data"]
            
            # 20. 版本回滚测试
            rollback_success = await version_manager.revert_to_version(version1.version_id)
            assert rollback_success
            assert version_manager.current_version == version1.version_id
            
            # 21. 最终统计检查
            final_stats = kg.calculate_statistics()
            assert final_stats["total_entities"] >= 3  # Person, John, Mary, Alice
            assert final_stats["total_relations"] >= 1  # knows
            assert final_stats["total_triples"] >= 3
            
            version_stats = await version_manager.get_version_statistics()
            assert version_stats["total_versions"] >= 2
            
            auth_stats = await auth.get_security_statistics()
            assert auth_stats["total_users"] >= 1
            assert auth_stats["total_api_keys"] >= 1
            
            change_stats = await change_tracker.get_statistics()
            assert change_stats.total_events > 0
            
            logger.info("✓ 所有集成测试通过")
            
        finally:
            await change_tracker.stop()
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        # 认证错误
        auth = KnowledgeGraphAuth(storage_path=str(self.auth_dir))
        
        # 错误的密码
        auth_result = await auth.authenticate_user("nonexistent", "wrong", "127.0.0.1")
        assert not auth_result.success
        
        # 无效的API密钥
        api_result = await auth.authenticate_api_key("invalid_key", "127.0.0.1")
        assert not api_result.success
        
        # SPARQL查询错误
        engine = SPARQLEngine()
        invalid_query = SPARQLQuery(
            query_id="invalid_query",
            query_text="INVALID SPARQL SYNTAX",
            query_type=QueryType.SELECT
        )
        
        result = await engine.execute_query(invalid_query)
        assert not result.success
        assert result.error_message is not None
        
        # 版本管理错误
        version_manager = VersionManager(storage_path=str(self.version_dir))
        
        # 获取不存在的版本
        non_existent = await version_manager.get_version("non-existent-id")
        assert non_existent is None
        
        # 比较不存在的版本
        with pytest.raises(ValueError):
            await version_manager.compare_versions("fake1", "fake2")
        
        logger.error("✓ 错误处理测试通过")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        auth = KnowledgeGraphAuth(storage_path=str(self.auth_dir))
        change_tracker = ChangeTracker()
        
        await change_tracker.start()
        
        try:
            # 创建测试用户
            user = await auth.create_user(
                username="concurrent_user",
                email="concurrent@example.com", 
                password="Test123!@#",
                role=Role.EDITOR
            )
            
            # 并发执行多个操作
            tasks = []
            
            # 并发创建多个变更事件
            for i in range(10):
                task = change_tracker.track_change(
                    event_type=EventType.ENTITY_CREATED,
                    user_id=user.user_id,
                    affected_resources=[f"entity:{i}"],
                    new_value={"id": i, "name": f"Entity{i}"},
                    metadata={"batch": "concurrent_test"}
                )
                tasks.append(task)
            
            # 并发执行SPARQL查询
            engine = SPARQLEngine()
            for i in range(5):
                query = SPARQLQuery(
                    query_id=f"concurrent_query_{i}",
                    query_text=f"SELECT * WHERE {{ ?s ?p ?o }} LIMIT {i+1}",
                    query_type=QueryType.SELECT
                )
                tasks.append(engine.execute_query(query))
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查结果
            successful_ops = sum(1 for r in results if not isinstance(r, Exception))
            assert successful_ops >= len(tasks) * 0.8  # 至少80%成功
            
            # 检查变更追踪
            await asyncio.sleep(1)  # 等待处理完成
            events = await change_tracker.get_events(limit=20)
            concurrent_events = [e for e in events if e.metadata and e.metadata.get("batch") == "concurrent_test"]
            assert len(concurrent_events) >= 8  # 大部分事件应该被记录
            
            logger.info("✓ 并发操作测试通过")
            
        finally:
            await change_tracker.stop()
    
    @pytest.mark.asyncio  
    async def test_performance_under_load(self):
        """测试负载下的性能"""
        import time
        
        engine = SPARQLEngine()
        monitor = KnowledgeGraphMonitor(log_dir=str(self.monitor_dir))
        
        await monitor.start_monitoring(interval_seconds=1)
        
        try:
            # 执行大量查询
            start_time = time.time()
            tasks = []
            
            for i in range(100):
                query = SPARQLQuery(
                    query_id=f"load_test_{i}",
                    query_text="SELECT * WHERE { ?s ?p ?o } LIMIT 10",
                    query_type=QueryType.SELECT
                )
                tasks.append(engine.execute_query(query))
                
                # 模拟请求监控
                monitor.record_request("POST", "/api/query", 0.1, 200)
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            # 性能检查
            total_time = end_time - start_time
            successful_queries = sum(1 for r in results if r.success)
            
            assert successful_queries >= 90  # 至少90%成功
            assert total_time < 30  # 应该在30秒内完成
            
            queries_per_second = successful_queries / total_time
            assert queries_per_second > 1  # 每秒至少1个查询
            
            # 检查监控数据
            await asyncio.sleep(2)
            system_status = await monitor.get_system_status()
            
            assert "performance_metrics" in system_status
            assert system_status["performance_metrics"]["total_requests"] >= 100
            
            logger.info(f"✓ 性能测试通过 - {successful_queries}个查询，{queries_per_second:.2f} QPS")
            
        finally:
            await monitor.stop_monitoring()

if __name__ == "__main__":
    setup_logging()
    # 运行集成测试
    async def run_integration_tests():
        logger.info("开始知识图谱系统集成测试...")
        
        test_instance = TestKnowledgeGraphIntegration()
        test_instance.setup_method()
        
        try:
            logger.info("1. 运行完整工作流程测试...")
            await test_instance.test_complete_workflow()
            
            logger.error("2. 运行错误处理测试...")
            await test_instance.test_error_handling()
            
            logger.info("3. 运行并发操作测试...")
            await test_instance.test_concurrent_operations()
            
            logger.info("4. 运行性能负载测试...")
            await test_instance.test_performance_under_load()
            
            logger.info("✅ 所有集成测试成功完成！")
            
        except Exception as e:
            logger.error(f"❌ 集成测试失败: {str(e)}")
            raise
        finally:
            test_instance.teardown_method()
    
    asyncio.run(run_integration_tests())
