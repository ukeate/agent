#!/usr/bin/env python3
"""
调试WebSocket多智能体对话问题的测试脚本
"""
import asyncio
import json
import websockets
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_multi_agent_websocket():
    """测试多智能体WebSocket对话"""
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    logger.info(f"连接WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info("WebSocket连接成功")
            
            # 等待连接确认消息
            message = await websocket.recv()
            logger.info(f"收到连接确认: {message}")
            
            # 发送ping测试
            ping_message = {
                "type": "ping",
                "data": {"test": True},
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(ping_message))
            logger.info(f"发送ping: {ping_message}")
            
            # 等待pong响应
            pong_response = await websocket.recv()
            logger.info(f"收到pong响应: {pong_response}")
            
            # 发送开始对话消息
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "请各位智能体介绍一下自己的专业领域",
                    "participants": ["code_expert-1", "architect-1", "doc_expert-1"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"发送开始对话消息: {start_message}")
            await websocket.send(json.dumps(start_message))
            
            # 监听响应消息
            message_count = 0
            timeout_count = 0
            max_messages = 10  # 最多接收10条消息
            max_timeout = 60   # 最多等待60秒
            
            while message_count < max_messages and timeout_count < max_timeout:
                try:
                    # 等待消息，设置1秒超时
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    
                    try:
                        parsed_message = json.loads(message)
                        msg_type = parsed_message.get("type", "unknown")
                        
                        logger.info(f"[消息 {message_count}] 类型: {msg_type}")
                        logger.info(f"[消息 {message_count}] 内容: {json.dumps(parsed_message, ensure_ascii=False, indent=2)}")
                        
                        # 检查是否是对话完成或错误
                        if msg_type in ["conversation_completed", "conversation_error"]:
                            logger.info("对话已完成或出错，结束监听")
                            break
                            
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析JSON消息: {message}")
                        
                except asyncio.TimeoutError:
                    timeout_count += 1
                    logger.info(f"等待消息超时 ({timeout_count}/{max_timeout})")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket连接已关闭")
                    break
                    
            logger.info(f"测试完成 - 收到 {message_count} 条消息，超时 {timeout_count} 次")
            
    except Exception as e:
        logger.error(f"WebSocket测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_multi_agent_websocket())