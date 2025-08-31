"""
智能体通信协议工厂
定义标准化的智能体通信语言(ACL)消息格式
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import MessageType, MessageHeader, MessagePriority, DeliveryMode


class MessageProtocol:
    """智能体通信协议工厂类"""
    
    @staticmethod
    def create_message_header(
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        ttl: Optional[int] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE,
        max_retries: int = 3
    ) -> MessageHeader:
        """创建标准消息头"""
        return MessageHeader(
            message_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            reply_to=reply_to,
            timestamp=datetime.now(),
            ttl=ttl,
            priority=priority,
            delivery_mode=delivery_mode,
            retry_count=0,
            max_retries=max_retries
        )
    
    @staticmethod
    def create_ping_message(agent_id: str) -> Dict[str, Any]:
        """创建ping消息"""
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "ping_seq": 0
        }
    
    @staticmethod
    def create_pong_message(ping_timestamp: str, agent_id: str) -> Dict[str, Any]:
        """创建pong消息"""
        return {
            "ping_timestamp": ping_timestamp,
            "pong_timestamp": datetime.now().isoformat(),
            "agent_id": agent_id
        }
    
    @staticmethod
    def create_heartbeat_message(
        agent_id: str,
        status: str = "active",
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建心跳消息"""
        return {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "metrics": metrics or {},
            "uptime": 0  # 可以在实际实现中计算
        }
    
    @staticmethod
    def create_task_request(
        task_id: str,
        task_type: str,
        task_data: Dict[str, Any],
        requirements: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        deadline: Optional[str] = None,
        estimated_duration: Optional[int] = None
    ) -> Dict[str, Any]:
        """创建任务请求消息"""
        return {
            "task_id": task_id,
            "task_type": task_type,
            "task_data": task_data,
            "requirements": requirements or {},
            "priority": priority,
            "deadline": deadline,
            "estimated_duration": estimated_duration,
            "created_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_task_response(
        task_id: str,
        status: str,  # accept, reject
        agent_id: str,
        reason: Optional[str] = None,
        estimated_duration: Optional[int] = None,
        required_resources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """创建任务响应消息"""
        return {
            "task_id": task_id,
            "status": status,
            "agent_id": agent_id,
            "reason": reason,
            "estimated_duration": estimated_duration,
            "required_resources": required_resources or [],
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_task_result(
        task_id: str,
        result: Dict[str, Any],
        status: str = "completed",  # completed, failed, cancelled
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None,
        output_artifacts: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """创建任务结果消息"""
        return {
            "task_id": task_id,
            "result": result,
            "status": status,
            "error_message": error_message,
            "execution_time": execution_time,
            "output_artifacts": output_artifacts or [],
            "completed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_task_status(
        task_id: str,
        status: str,  # running, paused, cancelled
        progress: float = 0.0,
        current_step: Optional[str] = None,
        estimated_remaining: Optional[int] = None
    ) -> Dict[str, Any]:
        """创建任务状态消息"""
        return {
            "task_id": task_id,
            "status": status,
            "progress": progress,  # 0.0 - 1.0
            "current_step": current_step,
            "estimated_remaining": estimated_remaining,
            "updated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_collaboration_invite(
        collaboration_id: str,
        collaboration_type: str,
        description: str,
        initiator_id: str,
        required_capabilities: List[str],
        max_participants: int = 10,
        duration_minutes: Optional[int] = None,
        data_sharing_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建协作邀请消息"""
        return {
            "collaboration_id": collaboration_id,
            "collaboration_type": collaboration_type,
            "description": description,
            "initiator_id": initiator_id,
            "required_capabilities": required_capabilities,
            "max_participants": max_participants,
            "duration_minutes": duration_minutes,
            "data_sharing_rules": data_sharing_rules or {},
            "created_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_collaboration_response(
        collaboration_id: str,
        agent_id: str,
        response: str,  # accept, reject, leave
        capabilities: Optional[List[str]] = None,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建协作响应消息"""
        return {
            "collaboration_id": collaboration_id,
            "agent_id": agent_id,
            "response": response,
            "capabilities": capabilities or [],
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_collaboration_update(
        collaboration_id: str,
        update_type: str,  # participant_joined, participant_left, status_changed
        data: Dict[str, Any],
        sender_id: str
    ) -> Dict[str, Any]:
        """创建协作更新消息"""
        return {
            "collaboration_id": collaboration_id,
            "update_type": update_type,
            "data": data,
            "sender_id": sender_id,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_resource_request(
        resource_type: str,
        resource_requirements: Dict[str, Any],
        duration_minutes: int,
        requester_id: str,
        priority: int = 5,
        exclusive: bool = False
    ) -> Dict[str, Any]:
        """创建资源请求消息"""
        return {
            "resource_type": resource_type,
            "requirements": resource_requirements,
            "duration_minutes": duration_minutes,
            "requester_id": requester_id,
            "priority": priority,
            "exclusive": exclusive,
            "requested_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_resource_offer(
        resource_type: str,
        available_resources: Dict[str, Any],
        provider_id: str,
        cost: Optional[float] = None,
        availability_window: Optional[Dict[str, str]] = None,
        terms: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建资源提供消息"""
        return {
            "resource_type": resource_type,
            "available_resources": available_resources,
            "provider_id": provider_id,
            "cost": cost,
            "availability_window": availability_window or {},
            "terms": terms or {},
            "offered_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_resource_status(
        resource_id: str,
        status: str,  # available, allocated, maintenance, offline
        current_usage: Optional[Dict[str, Any]] = None,
        next_available: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建资源状态消息"""
        return {
            "resource_id": resource_id,
            "status": status,
            "current_usage": current_usage or {},
            "next_available": next_available,
            "updated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_agent_joined_message(
        agent_id: str,
        agent_type: str,
        capabilities: List[str],
        group: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建智能体加入消息"""
        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "capabilities": capabilities,
            "group": group,
            "metadata": metadata or {},
            "joined_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_agent_left_message(
        agent_id: str,
        reason: str = "normal_shutdown",
        group: Optional[str] = None,
        final_status: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """创建智能体离开消息"""
        return {
            "agent_id": agent_id,
            "reason": reason,
            "group": group,
            "final_status": final_status or {},
            "left_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_system_shutdown_message(
        shutdown_type: str,  # planned, emergency
        reason: str,
        grace_period_seconds: int = 30,
        affected_services: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """创建系统关闭消息"""
        return {
            "shutdown_type": shutdown_type,
            "reason": reason,
            "grace_period_seconds": grace_period_seconds,
            "affected_services": affected_services or [],
            "scheduled_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_config_update_message(
        config_type: str,
        config_data: Dict[str, Any],
        version: str,
        applies_to: Optional[List[str]] = None,
        restart_required: bool = False
    ) -> Dict[str, Any]:
        """创建配置更新消息"""
        return {
            "config_type": config_type,
            "config_data": config_data,
            "version": version,
            "applies_to": applies_to or [],
            "restart_required": restart_required,
            "updated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_data_stream_start_message(
        stream_id: str,
        sender_id: str,
        data_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        expected_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """创建数据流开始消息"""
        return {
            "stream_id": stream_id,
            "sender_id": sender_id,
            "data_type": data_type,
            "metadata": metadata or {},
            "expected_chunks": expected_chunks,
            "started_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_data_chunk_message(
        stream_id: str,
        chunk_index: int,
        data: str,  # base64 encoded
        checksum: str,
        is_final: bool = False,
        chunk_size: int = 0
    ) -> Dict[str, Any]:
        """创建数据块消息"""
        return {
            "stream_id": stream_id,
            "chunk_index": chunk_index,
            "data": data,
            "size": chunk_size,
            "checksum": checksum,
            "is_final": is_final,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_data_stream_end_message(
        stream_id: str,
        total_chunks: int,
        total_size: int,
        final_checksum: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建数据流结束消息"""
        return {
            "stream_id": stream_id,
            "total_chunks": total_chunks,
            "total_size": total_size,
            "final_checksum": final_checksum,
            "success": success,
            "error_message": error_message,
            "completed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_data_sync_message(
        sync_id: str,
        sync_type: str,  # full, incremental
        data_version: str,
        data_hash: str,
        sync_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建数据同步消息"""
        return {
            "sync_id": sync_id,
            "sync_type": sync_type,
            "data_version": data_version,
            "data_hash": data_hash,
            "sync_data": sync_data,
            "sync_timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_ack_message(
        original_message_id: str,
        status: str = "received",  # received, processed, failed
        details: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建确认消息"""
        return {
            "original_message_id": original_message_id,
            "status": status,
            "details": details,
            "ack_timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_nack_message(
        original_message_id: str,
        error_code: str,
        error_message: str,
        retry_after: Optional[int] = None
    ) -> Dict[str, Any]:
        """创建否认消息"""
        return {
            "original_message_id": original_message_id,
            "error_code": error_code,
            "error_message": error_message,
            "retry_after": retry_after,
            "nack_timestamp": datetime.now().isoformat()
        }