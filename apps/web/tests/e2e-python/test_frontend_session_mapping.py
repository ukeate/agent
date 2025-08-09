#!/usr/bin/env python3
"""
测试前端会话ID映射修复
模拟前端完整流程：临时ID -> WebSocket连接 -> start_conversation -> 接收streaming tokens
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_frontend_session_mapping():
    """测试前端会话ID映射修复"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始测试前端会话ID映射修复...")
    
    # 1. 模拟前端生成临时session ID
    temp_session_id = f"session-{int(datetime.now().timestamp() * 1000)}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 生成临时会话ID: {temp_session_id}")
    
    # 2. 连接WebSocket使用临时ID
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{temp_session_id}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket连接已建立")
            
            # 3. 等待connection_established消息
            await wait_for_message_type(websocket, "connection_established")
            
            # 4. 发送start_conversation消息
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "请简单介绍一下Python的异步编程特性",
                    "participants": ["code_expert-1"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送start_conversation消息...")
            await websocket.send(json.dumps(start_message))
            
            # 5. 监控消息，特别关注conversation_created和streaming_token
            token_count = 0
            conversation_id = None
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始监控消息...")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if msg_type == "conversation_created":
                        conversation_id = data.get("data", {}).get("conversation_id", "")
                        print(f"[{timestamp}] ✅ 对话创建成功: {conversation_id}")
                        print(f"[{timestamp}] 🔄 会话ID映射: {temp_session_id} -> {conversation_id}")
                        
                    elif msg_type == "conversation_started":
                        print(f"[{timestamp}] 🚀 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        speaker = data.get("data", {}).get("current_speaker", "")
                        round_num = data.get("data", {}).get("round", "")
                        print(f"[{timestamp}] 🎤 发言者切换: {speaker} (第{round_num}轮)")
                        
                    elif msg_type == "streaming_token":
                        token_count += 1
                        token = data.get("data", {}).get("token", "")
                        agent_name = data.get("data", {}).get("agent_name", "")
                        is_complete = data.get("data", {}).get("is_complete", False)
                        
                        if token_count <= 5 or token_count % 10 == 0 or is_complete:
                            print(f"[{timestamp}] 🎯 流式token #{token_count} - {agent_name}: '{token}' (完成: {is_complete})")
                        
                    elif msg_type == "streaming_complete":
                        agent_name = data.get("data", {}).get("agent_name", "")
                        print(f"[{timestamp}] ✅ 流式响应完成 - {agent_name}, 总token数: {token_count}")
                        
                    elif msg_type == "new_message":
                        sender = data.get("data", {}).get("sender", "")
                        content_preview = data.get("data", {}).get("content", "")[:50]
                        print(f"[{timestamp}] 💬 新消息 - {sender}: {content_preview}...")
                        
                    elif msg_type == "conversation_completed":
                        print(f"[{timestamp}] ✅ 对话已完成，最终token数: {token_count}")
                        break
                        
                    elif msg_type == "error":
                        error_msg = data.get("data", {}).get("message", "")
                        print(f"[{timestamp}] ❌ 错误: {error_msg}")
                        break
                        
                    else:
                        print(f"[{timestamp}] 📝 其他消息: {msg_type}")
                        
                    # 设置超时防止无限等待
                    if token_count > 100:  # 如果收到超过100个token就认为测试成功
                        print(f"[{timestamp}] 🎉 测试成功！已收到{token_count}个streaming tokens")
                        break
                        
                except json.JSONDecodeError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 无法解析消息: {message}")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 处理消息异常: {e}")
            
            # 总结测试结果
            print("\n" + "="*60)
            print("测试结果总结:")
            print(f"临时会话ID: {temp_session_id}")
            print(f"真实对话ID: {conversation_id}")
            print(f"接收到的streaming tokens数量: {token_count}")
            
            if token_count > 0:
                print("🎉 修复成功！前端现在可以接收到streaming tokens了！")
            else:
                print("❌ 修复失败，前端仍然无法接收到streaming tokens")
            print("="*60)
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 测试异常: {e}")

async def wait_for_message_type(websocket, expected_type, timeout=10):
    """等待特定类型的消息"""
    try:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            data = json.loads(message)
            if data.get("type") == expected_type:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 收到 {expected_type} 消息")
                return data
    except asyncio.TimeoutError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 等待 {expected_type} 消息超时")
        return None

if __name__ == "__main__":
    print("="*60)
    print("前端会话ID映射修复测试")
    print("="*60)
    asyncio.run(test_frontend_session_mapping())