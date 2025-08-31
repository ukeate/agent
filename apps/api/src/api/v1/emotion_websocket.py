"""
情感智能系统WebSocket实时通信API
支持多模态情感数据的实时传输和处理
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ValidationError

from ai.emotion_modeling.core_interfaces import (
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    EmotionState, EmotionType, ModalityType
)
from ai.emotion_modeling.realtime_stream_processor import (
    MultiUserStreamManager, RealtimeStreamProcessor, ProcessingMode
)
# from ai.emotion_modeling.emotion_recognition_integration import (
#     EmotionRecognitionEngineImpl
# )
from ai.emotion_modeling.data_flow_manager import EmotionalDataFlowManagerImpl
from ai.emotion_modeling.communication_protocol import CommunicationProtocol
from ai.emotion_modeling.quality_monitor import quality_monitor
from core.dependencies import get_current_user
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ws", tags=["emotion-websocket"])

# WebSocket消息类型
class MessageType(str, Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    EMOTION_INPUT = "emotion_input"
    EMOTION_RESULT = "emotion_result"
    STREAM_STATUS = "stream_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SYSTEM_STATUS = "system_status"


# WebSocket消息格式
class WebSocketMessage(BaseModel):
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime
    message_id: Optional[str] = None
    user_id: Optional[str] = None


class EmotionInputMessage(BaseModel):
    text: Optional[str] = None
    audio_data: Optional[str] = None  # Base64编码
    video_data: Optional[str] = None  # Base64编码
    image_data: Optional[str] = None  # Base64编码
    physiological_data: Optional[Dict[str, Any]] = None
    modalities: List[ModalityType]
    timestamp: datetime


class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        # 活跃连接映射 {user_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # 用户会话信息 {user_id: session_info}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        # 连接统计
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
        
        # 系统组件
        self.stream_manager: Optional[MultiUserStreamManager] = None
        self.communication_protocol: Optional[CommunicationProtocol] = None
        
        # 心跳检查
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = 30  # 30秒心跳间隔
    
    async def initialize(self):
        """初始化连接管理器"""
        try:
            # 初始化通信协议
            self.communication_protocol = CommunicationProtocol()
            await self.communication_protocol.start()
            
            # 初始化识别引擎（这里使用模拟实现）
            recognition_engine = EmotionRecognitionEngineImpl()
            
            # 初始化数据流管理器
            data_flow_manager = EmotionalDataFlowManagerImpl(self.communication_protocol)
            await data_flow_manager.initialize()
            
            # 初始化流管理器
            self.stream_manager = MultiUserStreamManager(
                recognition_engine=recognition_engine,
                data_flow_manager=data_flow_manager,
                communication_protocol=self.communication_protocol
            )
            await self.stream_manager.start_manager()
            
            # 启动心跳检查
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("WebSocket connection manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection manager: {e}")
            raise
    
    async def shutdown(self):
        """关闭连接管理器"""
        try:
            # 停止心跳检查
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # 关闭所有连接
            for user_id in list(self.active_connections.keys()):
                await self.disconnect(user_id)
            
            # 停止系统组件
            if self.stream_manager:
                await self.stream_manager.stop_manager()
            
            if self.communication_protocol:
                await self.communication_protocol.stop()
            
            logger.info("WebSocket connection manager shutdown")
            
        except Exception as e:
            logger.error(f"Error during connection manager shutdown: {e}")
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """建立WebSocket连接"""
        try:
            await websocket.accept()
            
            # 如果用户已有连接，先关闭旧连接
            if user_id in self.active_connections:
                old_websocket = self.active_connections[user_id]
                try:
                    await old_websocket.close()
                except:
                    pass
            
            self.active_connections[user_id] = websocket
            self.user_sessions[user_id] = {
                "connected_at": datetime.now(),
                "last_activity": datetime.now(),
                "message_count": 0
            }
            
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] = len(self.active_connections)
            
            # 发送连接确认消息
            await self.send_message(user_id, {
                "type": MessageType.CONNECT,
                "data": {
                    "status": "connected",
                    "user_id": user_id,
                    "server_time": datetime.now().isoformat(),
                    "capabilities": [
                        "text_emotion_analysis",
                        "audio_emotion_analysis", 
                        "video_emotion_analysis",
                        "realtime_streaming",
                        "multi_modal_fusion"
                    ]
                },
                "timestamp": datetime.now(),
                "message_id": str(uuid.uuid4())
            })
            
            logger.info(f"WebSocket connected for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket for user {user_id}: {e}")
            self.connection_stats["errors"] += 1
            raise
    
    async def disconnect(self, user_id: str):
        """断开WebSocket连接"""
        try:
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]
                
                # 发送断开连接消息
                try:
                    await self.send_message(user_id, {
                        "type": MessageType.DISCONNECT,
                        "data": {
                            "status": "disconnecting",
                            "reason": "client_disconnect"
                        },
                        "timestamp": datetime.now(),
                        "message_id": str(uuid.uuid4())
                    })
                except:
                    pass
                
                # 关闭连接
                try:
                    await websocket.close()
                except:
                    pass
                
                # 清理资源
                del self.active_connections[user_id]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
                
                # 停止用户的流处理
                if self.stream_manager:
                    await self.stream_manager.remove_user(user_id)
                
                self.connection_stats["active_connections"] = len(self.active_connections)
                
                logger.info(f"WebSocket disconnected for user {user_id}")
        
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket for user {user_id}: {e}")
            self.connection_stats["errors"] += 1
    
    async def send_message(self, user_id: str, message: Dict[str, Any]):
        """向指定用户发送消息"""
        try:
            if user_id not in self.active_connections:
                logger.warning(f"No active connection for user {user_id}")
                return False
            
            websocket = self.active_connections[user_id]
            
            # 确保消息格式正确
            if "timestamp" not in message:
                message["timestamp"] = datetime.now()
            if "message_id" not in message:
                message["message_id"] = str(uuid.uuid4())
            
            # 序列化时间戳
            if isinstance(message["timestamp"], datetime):
                message["timestamp"] = message["timestamp"].isoformat()
            
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            
            self.connection_stats["messages_sent"] += 1
            
            # 更新用户活动时间
            if user_id in self.user_sessions:
                self.user_sessions[user_id]["last_activity"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")
            self.connection_stats["errors"] += 1
            # 连接可能已断开，清理资源
            await self.disconnect(user_id)
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_users: Set[str] = None):
        """广播消息给所有连接的用户"""
        exclude_users = exclude_users or set()
        
        tasks = []
        for user_id in self.active_connections:
            if user_id not in exclude_users:
                tasks.append(self.send_message(user_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_emotion_input(self, user_id: str, emotion_input: EmotionInputMessage):
        """处理情感输入数据"""
        try:
            if not self.stream_manager:
                raise RuntimeError("Stream manager not initialized")
            
            # 处理各种模态的数据
            for modality in emotion_input.modalities:
                data = None
                
                if modality == ModalityType.TEXT and emotion_input.text:
                    data = emotion_input.text
                elif modality == ModalityType.AUDIO and emotion_input.audio_data:
                    # 这里应该解码Base64音频数据
                    data = emotion_input.audio_data
                elif modality == ModalityType.VIDEO and emotion_input.video_data:
                    # 这里应该解码Base64视频数据
                    data = emotion_input.video_data
                elif modality == ModalityType.IMAGE and emotion_input.image_data:
                    # 这里应该解码Base64图像数据
                    data = emotion_input.image_data
                elif modality == ModalityType.PHYSIOLOGICAL and emotion_input.physiological_data:
                    data = emotion_input.physiological_data
                
                if data:
                    # 添加到流处理器
                    success = await self.stream_manager.add_user_data(user_id, modality, data)
                    if not success:
                        logger.warning(f"Failed to add {modality.value} data for user {user_id}")
            
            # 记录处理统计
            if user_id in self.user_sessions:
                self.user_sessions[user_id]["message_count"] += 1
                self.user_sessions[user_id]["last_activity"] = datetime.now()
            
            self.connection_stats["messages_received"] += 1
            
        except Exception as e:
            logger.error(f"Error processing emotion input for user {user_id}: {e}")
            
            # 发送错误消息
            await self.send_message(user_id, {
                "type": MessageType.ERROR,
                "data": {
                    "error": "processing_failed",
                    "message": "Failed to process emotion input",
                    "details": str(e)
                },
                "timestamp": datetime.now(),
                "message_id": str(uuid.uuid4())
            })
            
            self.connection_stats["errors"] += 1
    
    async def _heartbeat_loop(self):
        """心跳检查循环"""
        while True:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                
                current_time = datetime.now()
                disconnected_users = []
                
                # 检查每个用户的连接状态
                for user_id, session in self.user_sessions.items():
                    last_activity = session.get("last_activity", current_time)
                    if (current_time - last_activity).total_seconds() > self._heartbeat_interval * 2:
                        # 连接可能已断开
                        disconnected_users.append(user_id)
                    else:
                        # 发送心跳消息
                        await self.send_message(user_id, {
                            "type": MessageType.HEARTBEAT,
                            "data": {
                                "server_time": current_time.isoformat(),
                                "active_connections": len(self.active_connections)
                            },
                            "timestamp": current_time,
                            "message_id": str(uuid.uuid4())
                        })
                
                # 清理断开的连接
                for user_id in disconnected_users:
                    await self.disconnect(user_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self._heartbeat_interval)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            **self.connection_stats,
            "user_sessions": {
                user_id: {
                    "connected_at": session["connected_at"].isoformat(),
                    "last_activity": session["last_activity"].isoformat(),
                    "message_count": session["message_count"]
                }
                for user_id, session in self.user_sessions.items()
            }
        }


# 全局连接管理器实例
connection_manager = ConnectionManager()

# 启动时初始化
@router.on_event("startup")
async def startup_event():
    await connection_manager.initialize()

# 关闭时清理
@router.on_event("shutdown")
async def shutdown_event():
    await connection_manager.shutdown()


@router.websocket("/emotion/{user_id}")
async def emotion_websocket_endpoint(websocket: WebSocket, user_id: str):
    """情感分析WebSocket端点"""
    await connection_manager.connect(websocket, user_id)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                message = WebSocketMessage(**message_data)
                
                if message.type == MessageType.EMOTION_INPUT:
                    # 处理情感输入
                    emotion_input = EmotionInputMessage(**message.data)
                    await connection_manager.process_emotion_input(user_id, emotion_input)
                    
                elif message.type == MessageType.HEARTBEAT:
                    # 响应心跳
                    await connection_manager.send_message(user_id, {
                        "type": MessageType.HEARTBEAT,
                        "data": {"status": "alive"},
                        "timestamp": datetime.now(),
                        "message_id": str(uuid.uuid4())
                    })
                
                elif message.type == MessageType.STREAM_STATUS:
                    # 获取流状态
                    if connection_manager.stream_manager:
                        status = connection_manager.stream_manager.get_manager_status()
                        await connection_manager.send_message(user_id, {
                            "type": MessageType.STREAM_STATUS,
                            "data": status,
                            "timestamp": datetime.now(),
                            "message_id": str(uuid.uuid4())
                        })
                
            except ValidationError as e:
                # 消息格式错误
                await connection_manager.send_message(user_id, {
                    "type": MessageType.ERROR,
                    "data": {
                        "error": "invalid_message_format",
                        "message": "Invalid message format",
                        "details": str(e)
                    },
                    "timestamp": datetime.now(),
                    "message_id": str(uuid.uuid4())
                })
            
            except json.JSONDecodeError:
                # JSON解析错误
                await connection_manager.send_message(user_id, {
                    "type": MessageType.ERROR,
                    "data": {
                        "error": "json_parse_error",
                        "message": "Failed to parse JSON message"
                    },
                    "timestamp": datetime.now(),
                    "message_id": str(uuid.uuid4())
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        await connection_manager.disconnect(user_id)


# 情感结果回调处理器
class EmotionResultHandler:
    """处理情感分析结果并发送给WebSocket客户端"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def handle_emotion_result(self, user_id: str, result: UnifiedEmotionalData):
        """处理情感分析结果"""
        try:
            # 将结果转换为客户端格式
            result_data = {
                "user_id": result.user_id,
                "timestamp": result.timestamp.isoformat(),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "data_quality": result.data_quality
            }
            
            # 添加识别结果
            if result.recognition_result:
                result_data["recognition_result"] = {
                    "fused_emotion": {
                        "emotion": result.recognition_result.fused_emotion.emotion.value,
                        "intensity": result.recognition_result.fused_emotion.intensity,
                        "valence": result.recognition_result.fused_emotion.valence,
                        "arousal": result.recognition_result.fused_emotion.arousal,
                        "dominance": result.recognition_result.fused_emotion.dominance,
                        "confidence": result.recognition_result.fused_emotion.confidence
                    },
                    "emotions": {
                        modality.value: {
                            "emotion": emotion.emotion.value,
                            "intensity": emotion.intensity,
                            "valence": emotion.valence,
                            "arousal": emotion.arousal,
                            "dominance": emotion.dominance,
                            "confidence": emotion.confidence
                        }
                        for modality, emotion in result.recognition_result.emotions.items()
                    },
                    "confidence": result.recognition_result.confidence,
                    "processing_time": result.recognition_result.processing_time
                }
            
            # 添加其他组件的结果
            if result.empathy_response:
                result_data["empathy_response"] = {
                    "message": result.empathy_response.message,
                    "response_type": result.empathy_response.response_type,
                    "confidence": result.empathy_response.confidence,
                    "generation_strategy": result.empathy_response.generation_strategy
                }
            
            if result.personality_profile:
                result_data["personality_profile"] = {
                    "openness": result.personality_profile.openness,
                    "conscientiousness": result.personality_profile.conscientiousness,
                    "extraversion": result.personality_profile.extraversion,
                    "agreeableness": result.personality_profile.agreeableness,
                    "neuroticism": result.personality_profile.neuroticism,
                    "updated_at": result.personality_profile.updated_at.isoformat()
                }
            
            # 发送结果给客户端
            await self.connection_manager.send_message(user_id, {
                "type": MessageType.EMOTION_RESULT,
                "data": result_data,
                "timestamp": datetime.now(),
                "message_id": str(uuid.uuid4())
            })
            
            # 记录质量监控数据
            if result.emotional_state:
                await quality_monitor.record_prediction(
                    user_id=user_id,
                    predicted_emotion=result.emotional_state,
                    modality=ModalityType.TEXT,  # 默认模态，实际应该从结果中获取
                    processing_time=result.processing_time,
                    confidence=result.confidence,
                    data_quality=result.data_quality
                )
            
        except Exception as e:
            logger.error(f"Error handling emotion result for user {user_id}: {e}")


# 创建结果处理器实例
emotion_result_handler = EmotionResultHandler(connection_manager)


# REST API端点用于管理和监控
@router.get("/stats")
async def get_websocket_stats():
    """获取WebSocket连接统计"""
    return connection_manager.get_connection_stats()


@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """向所有连接的用户广播消息"""
    await connection_manager.broadcast_message({
        "type": MessageType.SYSTEM_STATUS,
        "data": message,
        "timestamp": datetime.now(),
        "message_id": str(uuid.uuid4())
    })
    return {"status": "broadcasted", "active_connections": len(connection_manager.active_connections)}