#!/usr/bin/env python3
"""
简单的WebSocket连接测试
"""
import asyncio
import websockets
import json

async def test_websocket_connection():
    """测试基本WebSocket连接"""
    session_id = "test-session-simple"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"尝试连接: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送ping消息测试连接
            ping_message = {"type": "ping"}
            print(f"发送ping: {ping_message}")
            await websocket.send(json.dumps(ping_message))
            
            # 等待pong响应
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            parsed = json.loads(response)
            print(f"收到响应: {parsed}")
            
            if parsed.get('type') == 'pong':
                print("✅ ping-pong测试成功")
                return True
            else:
                print(f"❌ 意外的响应类型: {parsed.get('type')}")
                return False
                
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_websocket_connection())
    if success:
        print("\n✅ WebSocket基础连接测试通过")
    else:
        print("\n❌ WebSocket基础连接测试失败")