"""
智能体通信协议测试
"""

import pytest
from datetime import datetime
from src.ai.distributed_message.protocol import MessageProtocol
from src.ai.distributed_message.models import MessageHeader, MessagePriority, DeliveryMode

class TestMessageProtocol:
    """消息协议测试"""
    
    def test_create_message_header(self):
        """测试创建消息头"""
        header = MessageProtocol.create_message_header(
            priority=MessagePriority.HIGH,
            ttl=300
        )
        
        assert header.priority == MessagePriority.HIGH
        assert header.ttl == 300
        assert header.delivery_mode == DeliveryMode.AT_LEAST_ONCE
        assert header.max_retries == 3
        assert header.message_id is not None
        assert isinstance(header.timestamp, datetime)
    
    def test_create_ping_message(self):
        """测试创建ping消息"""
        payload = MessageProtocol.create_ping_message("agent-123")
        
        assert payload["agent_id"] == "agent-123"
        assert "timestamp" in payload
        assert "ping_seq" in payload
    
    def test_create_pong_message(self):
        """测试创建pong消息"""
        ping_time = "2025-08-26T12:00:00"
        payload = MessageProtocol.create_pong_message(ping_time, "agent-456")
        
        assert payload["ping_timestamp"] == ping_time
        assert payload["agent_id"] == "agent-456"
        assert "pong_timestamp" in payload
    
    def test_create_heartbeat_message(self):
        """测试创建心跳消息"""
        metrics = {"cpu": 50.0, "memory": 75.0}
        payload = MessageProtocol.create_heartbeat_message(
            agent_id="agent-789",
            status="active",
            metrics=metrics
        )
        
        assert payload["agent_id"] == "agent-789"
        assert payload["status"] == "active"
        assert payload["metrics"] == metrics
        assert "timestamp" in payload
    
    def test_create_task_request(self):
        """测试创建任务请求"""
        task_data = {"input": "test_data"}
        requirements = {"cpu": 2, "memory": "4GB"}
        
        payload = MessageProtocol.create_task_request(
            task_id="task-001",
            task_type="data_analysis",
            task_data=task_data,
            requirements=requirements,
            priority=8,
            deadline="2025-08-26T15:00:00"
        )
        
        assert payload["task_id"] == "task-001"
        assert payload["task_type"] == "data_analysis"
        assert payload["task_data"] == task_data
        assert payload["requirements"] == requirements
        assert payload["priority"] == 8
        assert payload["deadline"] == "2025-08-26T15:00:00"
        assert "created_at" in payload
    
    def test_create_task_response(self):
        """测试创建任务响应"""
        payload = MessageProtocol.create_task_response(
            task_id="task-001",
            status="accept",
            agent_id="worker-agent",
            estimated_duration=300,
            required_resources=["gpu", "storage"]
        )
        
        assert payload["task_id"] == "task-001"
        assert payload["status"] == "accept"
        assert payload["agent_id"] == "worker-agent"
        assert payload["estimated_duration"] == 300
        assert payload["required_resources"] == ["gpu", "storage"]
        assert "timestamp" in payload
    
    def test_create_task_result(self):
        """测试创建任务结果"""
        result = {"output": "processed_data", "score": 0.95}
        artifacts = [{"type": "file", "path": "/tmp/result.json"}]
        
        payload = MessageProtocol.create_task_result(
            task_id="task-001",
            result=result,
            status="completed",
            execution_time=120.5,
            output_artifacts=artifacts
        )
        
        assert payload["task_id"] == "task-001"
        assert payload["result"] == result
        assert payload["status"] == "completed"
        assert payload["execution_time"] == 120.5
        assert payload["output_artifacts"] == artifacts
        assert "completed_at" in payload
    
    def test_create_collaboration_invite(self):
        """测试创建协作邀请"""
        capabilities = ["nlp", "vision", "reasoning"]
        sharing_rules = {"data_retention": 24, "access_level": "read_only"}
        
        payload = MessageProtocol.create_collaboration_invite(
            collaboration_id="collab-001",
            collaboration_type="research_project",
            description="Multi-agent research collaboration",
            initiator_id="lead-agent",
            required_capabilities=capabilities,
            max_participants=5,
            duration_minutes=120,
            data_sharing_rules=sharing_rules
        )
        
        assert payload["collaboration_id"] == "collab-001"
        assert payload["collaboration_type"] == "research_project"
        assert payload["initiator_id"] == "lead-agent"
        assert payload["required_capabilities"] == capabilities
        assert payload["max_participants"] == 5
        assert payload["data_sharing_rules"] == sharing_rules
        assert "created_at" in payload
    
    def test_create_resource_request(self):
        """测试创建资源请求"""
        requirements = {"cpu_cores": 4, "memory_gb": 8, "gpu": "optional"}
        
        payload = MessageProtocol.create_resource_request(
            resource_type="compute",
            resource_requirements=requirements,
            duration_minutes=60,
            requester_id="compute-agent",
            priority=7,
            exclusive=True
        )
        
        assert payload["resource_type"] == "compute"
        assert payload["requirements"] == requirements
        assert payload["duration_minutes"] == 60
        assert payload["requester_id"] == "compute-agent"
        assert payload["priority"] == 7
        assert payload["exclusive"] is True
        assert "requested_at" in payload
    
    def test_create_resource_offer(self):
        """测试创建资源提供"""
        resources = {"cpu_cores": 8, "memory_gb": 16, "available_gpus": 2}
        availability = {"start": "2025-08-26T14:00:00", "end": "2025-08-26T18:00:00"}
        
        payload = MessageProtocol.create_resource_offer(
            resource_type="compute",
            available_resources=resources,
            provider_id="resource-provider",
            cost=0.50,
            availability_window=availability
        )
        
        assert payload["resource_type"] == "compute"
        assert payload["available_resources"] == resources
        assert payload["provider_id"] == "resource-provider"
        assert payload["cost"] == 0.50
        assert payload["availability_window"] == availability
        assert "offered_at" in payload
    
    def test_create_agent_joined_message(self):
        """测试创建智能体加入消息"""
        capabilities = ["text_processing", "data_analysis"]
        metadata = {"version": "1.2.0", "region": "us-west"}
        
        payload = MessageProtocol.create_agent_joined_message(
            agent_id="new-agent",
            agent_type="worker",
            capabilities=capabilities,
            group="processing_group",
            metadata=metadata
        )
        
        assert payload["agent_id"] == "new-agent"
        assert payload["agent_type"] == "worker"
        assert payload["capabilities"] == capabilities
        assert payload["group"] == "processing_group"
        assert payload["metadata"] == metadata
        assert "joined_at" in payload
    
    def test_create_data_stream_messages(self):
        """测试创建数据流相关消息"""
        # 数据流开始
        start_payload = MessageProtocol.create_data_stream_start_message(
            stream_id="stream-001",
            sender_id="data-agent",
            data_type="csv",
            metadata={"rows": 1000, "columns": 10},
            expected_chunks=5
        )
        
        assert start_payload["stream_id"] == "stream-001"
        assert start_payload["sender_id"] == "data-agent"
        assert start_payload["data_type"] == "csv"
        assert start_payload["expected_chunks"] == 5
        assert "started_at" in start_payload
        
        # 数据块
        chunk_payload = MessageProtocol.create_data_chunk_message(
            stream_id="stream-001",
            chunk_index=0,
            data="base64_encoded_data",
            checksum="abc123",
            chunk_size=1024
        )
        
        assert chunk_payload["stream_id"] == "stream-001"
        assert chunk_payload["chunk_index"] == 0
        assert chunk_payload["data"] == "base64_encoded_data"
        assert chunk_payload["checksum"] == "abc123"
        assert chunk_payload["size"] == 1024
        
        # 数据流结束
        end_payload = MessageProtocol.create_data_stream_end_message(
            stream_id="stream-001",
            total_chunks=5,
            total_size=5120,
            final_checksum="final_abc123"
        )
        
        assert end_payload["stream_id"] == "stream-001"
        assert end_payload["total_chunks"] == 5
        assert end_payload["total_size"] == 5120
        assert end_payload["final_checksum"] == "final_abc123"
        assert end_payload["success"] is True
        assert "completed_at" in end_payload
    
    def test_create_system_messages(self):
        """测试创建系统消息"""
        # 系统关闭消息
        shutdown_payload = MessageProtocol.create_system_shutdown_message(
            shutdown_type="planned",
            reason="Scheduled maintenance",
            grace_period_seconds=60,
            affected_services=["task_queue", "message_bus"]
        )
        
        assert shutdown_payload["shutdown_type"] == "planned"
        assert shutdown_payload["reason"] == "Scheduled maintenance"
        assert shutdown_payload["grace_period_seconds"] == 60
        assert shutdown_payload["affected_services"] == ["task_queue", "message_bus"]
        assert "scheduled_at" in shutdown_payload
        
        # 配置更新消息
        config_data = {"max_connections": 1000, "timeout": 30}
        config_payload = MessageProtocol.create_config_update_message(
            config_type="connection_limits",
            config_data=config_data,
            version="2.1.0",
            applies_to=["worker_agents"],
            restart_required=True
        )
        
        assert config_payload["config_type"] == "connection_limits"
        assert config_payload["config_data"] == config_data
        assert config_payload["version"] == "2.1.0"
        assert config_payload["applies_to"] == ["worker_agents"]
        assert config_payload["restart_required"] is True
        assert "updated_at" in config_payload
    
    def test_create_ack_nack_messages(self):
        """测试创建确认和否认消息"""
        # ACK消息
        ack_payload = MessageProtocol.create_ack_message(
            original_message_id="msg-123",
            status="processed",
            details="Successfully processed request"
        )
        
        assert ack_payload["original_message_id"] == "msg-123"
        assert ack_payload["status"] == "processed"
        assert ack_payload["details"] == "Successfully processed request"
        assert "ack_timestamp" in ack_payload
        
        # NACK消息
        nack_payload = MessageProtocol.create_nack_message(
            original_message_id="msg-456",
            error_code="INVALID_FORMAT",
            error_message="Message format is invalid",
            retry_after=30
        )
        
        assert nack_payload["original_message_id"] == "msg-456"
        assert nack_payload["error_code"] == "INVALID_FORMAT"
        assert nack_payload["error_message"] == "Message format is invalid"
        assert nack_payload["retry_after"] == 30
        assert "nack_timestamp" in nack_payload
