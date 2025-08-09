#!/usr/bin/env python3
"""
测试具体conversation ID的streaming token接收
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_specific_conversation():
    """测试具体conversation ID的streaming token接收"""
    conversation_id = "04afa408-5a00-4a95-9c38-9a6930460315"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 测试连接到具体会话: {conversation_id}")
    
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{conversation_id}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket连接已建立")
            
            token_count = 0
            start_time = datetime.now()
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if msg_type == "connection_established":
                        print(f"[{timestamp}] ✅ 连接已确认")
                        
                    elif msg_type == "streaming_token":
                        token_count += 1
                        token = data.get("data", {}).get("token", "")
                        agent_name = data.get("data", {}).get("agent_name", "")
                        is_complete = data.get("data", {}).get("is_complete", False)
                        
                        if token_count <= 5 or token_count % 10 == 0:
                            print(f"[{timestamp}] 🎯 Token #{token_count} - {agent_name}: '{token}' (完成: {is_complete})")
                        
                    elif msg_type == "streaming_complete":
                        agent_name = data.get("data", {}).get("agent_name", "")
                        print(f"[{timestamp}] ✅ 流式完成 - {agent_name}, 总tokens: {token_count}")
                        break
                        
                    else:
                        print(f"[{timestamp}] 📝 消息: {msg_type}")
                        
                    # 超时检测
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 30:  # 30秒超时
                        print(f"[{timestamp}] ⏰ 30秒超时，停止测试")
                        break
                        
                except json.JSONDecodeError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 无法解析: {message}")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 异常: {e}")
            
            print(f"\n测试结果: 收到 {token_count} 个streaming tokens")
            if token_count == 0:
                print("❌ 没有收到任何streaming tokens - 后端可能没有向此connection发送消息")
            else:
                print("✅ 成功接收到streaming tokens")
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接异常: {e}")

if __name__ == "__main__":
    asyncio.run(test_specific_conversation())