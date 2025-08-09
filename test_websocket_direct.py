#!/usr/bin/env python3
"""
直接测试WebSocket连接和消息发送
"""
import asyncio
import json
import websockets
import uuid

async def test_websocket():
    session_id = f"test-{int(asyncio.get_event_loop().time() * 1000)}"
    uri = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"连接到: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("WebSocket连接成功")
            
            # 等待连接确认消息
            welcome_msg = await websocket.recv()
            print(f"收到欢迎消息: {welcome_msg}")
            
            # 发送ping测试
            ping_msg = {
                "type": "ping",
                "data": {"test": True},
                "timestamp": "2025-08-05T12:51:00Z"
            }
            
            await websocket.send(json.dumps(ping_msg))
            print(f"发送ping: {ping_msg}")
            
            # 等待pong响应
            pong_response = await websocket.recv()
            print(f"收到pong: {pong_response}")
            
            # 发送启动对话消息
            start_msg = {
                "type": "start_conversation",
                "data": {
                    "message": "测试WebSocket消息发送功能",
                    "participants": ["code_expert-1"]
                },
                "timestamp": "2025-08-05T12:51:00Z"
            }
            
            print(f"发送start_conversation: {start_msg}")
            await websocket.send(json.dumps(start_msg))
            
            # 等待响应消息
            print("等待响应...")
            for i in range(5):  # 等待最多5个消息
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    print(f"收到响应 {i+1}: {response}")
                except asyncio.TimeoutError:
                    print(f"等待响应超时 (第{i+1}次尝试)")
                    break
            
    except Exception as e:
        print(f"WebSocket测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())