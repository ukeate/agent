#!/usr/bin/env python3
"""
测试前端当前使用的conversation ID是否能接收streaming tokens
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_frontend_conversation_id():
    """测试前端当前使用的conversation ID"""
    conversation_id = "d924710b-1cfb-4b42-98f7-4713d9b67d89"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 测试前端conversation ID: {conversation_id}")
    
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{conversation_id}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 连接: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ WebSocket连接已建立")
            
            token_count = 0
            start_time = datetime.now()
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 👀 监控streaming tokens...")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if msg_type == "connection_established":
                        print(f"[{timestamp}] ✅ 连接确认")
                        
                    elif msg_type == "streaming_token":
                        token_count += 1
                        token = data.get("data", {}).get("token", "")
                        agent_name = data.get("data", {}).get("agent_name", "")
                        
                        if token_count <= 3 or token_count % 5 == 0:
                            print(f"[{timestamp}] 🎯 Token #{token_count} - {agent_name}: '{token}'")
                        
                    elif msg_type == "speaker_change":
                        speaker = data.get("data", {}).get("current_speaker", "")
                        print(f"[{timestamp}] 🎤 发言者: {speaker}")
                        
                    elif msg_type == "streaming_complete":
                        agent_name = data.get("data", {}).get("agent_name", "")
                        print(f"[{timestamp}] ✅ 流式完成 - {agent_name}, 收到 {token_count} tokens")
                        break
                        
                    elif msg_type == "conversation_started":
                        print(f"[{timestamp}] 🚀 对话已启动")
                        
                    else:
                        print(f"[{timestamp}] 📨 消息: {msg_type}")
                        
                    # 超时检测  
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 20:  # 20秒超时
                        print(f"[{timestamp}] ⏰ 超时，停止监控")
                        break
                        
                    # 成功接收到一些tokens就算测试通过
                    if token_count >= 10:
                        print(f"[{timestamp}] 🎉 成功收到 {token_count} tokens！")
                        break
                        
                except json.JSONDecodeError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ JSON解析失败: {message}")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 处理异常: {e}")
            
            print(f"\n🎯 测试结果:")
            print(f"📝 Conversation ID: {conversation_id}")
            print(f"🎯 接收到的tokens: {token_count}")
            
            if token_count > 0:
                print("✅ 这个conversation ID能接收到streaming tokens")
                print("❌ 问题在于前端WebSocket消息处理或显示逻辑")
            else:
                print("❌ 这个conversation ID没有接收到streaming tokens")
                print("❌ 问题在于后端WebSocket回调路由")
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接异常: {e}")

if __name__ == "__main__":
    asyncio.run(test_frontend_conversation_id())