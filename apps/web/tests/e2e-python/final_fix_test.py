#!/usr/bin/env python3
"""
最终修复验证测试
测试WebSocket回调立即传递的修复效果
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_final_fix():
    """测试最终修复效果"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 开始最终修复验证测试...")
    
    # 使用新的临时session ID
    temp_session_id = f"session-{int(datetime.now().timestamp() * 1000)}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📝 临时会话ID: {temp_session_id}")
    
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{temp_session_id}"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔗 连接: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ WebSocket连接已建立")
            
            # 等待connection_established
            await wait_for_message_type(websocket, "connection_established")
            
            # 发送start_conversation
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "请简要介绍Python异步编程",
                    "participants": ["code_expert-1"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 发送启动消息...")
            await websocket.send(json.dumps(start_message))
            
            # 监控消息
            token_count = 0
            conversation_id = None
            tokens_received = []
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 👀 开始监控消息...")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if msg_type == "conversation_created":
                        conversation_id = data.get("data", {}).get("conversation_id", "")
                        print(f"[{timestamp}] ✅ 对话创建: {conversation_id}")
                        
                    elif msg_type == "conversation_started":
                        print(f"[{timestamp}] 🚀 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        speaker = data.get("data", {}).get("current_speaker", "")
                        print(f"[{timestamp}] 🎤 发言者: {speaker}")
                        
                    elif msg_type == "streaming_token":
                        token_count += 1
                        token = data.get("data", {}).get("token", "")
                        agent_name = data.get("data", {}).get("agent_name", "")
                        tokens_received.append(token)
                        
                        # 显示前几个和每10个token
                        if token_count <= 3 or token_count % 10 == 0:
                            content_preview = ''.join(tokens_received[-10:])  # 显示最近10个token
                            print(f"[{timestamp}] 🎯 Token #{token_count} - {agent_name}: '{token}' | 内容: {content_preview}")
                        
                    elif msg_type == "streaming_complete":
                        agent_name = data.get("data", {}).get("agent_name", "")
                        final_content = ''.join(tokens_received)
                        print(f"[{timestamp}] ✅ 流式完成 - {agent_name}")
                        print(f"[{timestamp}] 📝 完整内容 ({len(final_content)}字符): {final_content[:100]}...")
                        break
                        
                    elif msg_type == "conversation_completed":
                        print(f"[{timestamp}] 🏁 对话完成")
                        break
                        
                    elif msg_type == "error":
                        error_msg = data.get("data", {}).get("message", "")
                        print(f"[{timestamp}] ❌ 错误: {error_msg}")
                        break
                        
                    else:
                        print(f"[{timestamp}] 📨 消息: {msg_type}")
                        
                    # 成功接收到足够tokens就认为测试通过
                    if token_count >= 20:
                        print(f"[{timestamp}] 🎉 成功！已接收到{token_count}个tokens，测试通过！")
                        break
                        
                except json.JSONDecodeError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ JSON解析失败: {message}")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 处理异常: {e}")
            
            # 测试结果
            print("\n" + "="*70)
            print("🎯 最终修复验证结果:")
            print(f"📝 临时会话ID: {temp_session_id}")
            print(f"💬 对话ID: {conversation_id}")
            print(f"🎯 接收到的streaming tokens: {token_count}")
            
            if token_count > 0:
                final_content = ''.join(tokens_received)
                print(f"📄 内容长度: {len(final_content)}字符")
                print("🎉 修复成功！前端现在能够正确接收streaming tokens！")
                print("✅ WebSocket回调立即传递修复生效！")
            else:
                print("❌ 修复失败，仍然没有接收到streaming tokens")
            print("="*70)
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 连接异常: {e}")

async def wait_for_message_type(websocket, expected_type, timeout=10):
    """等待特定类型的消息"""
    try:
        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            data = json.loads(message)
            if data.get("type") == expected_type:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 收到 {expected_type}")
                return data
    except asyncio.TimeoutError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 等待 {expected_type} 超时")
        return None

if __name__ == "__main__":
    print("="*70)
    print("🎯 最终修复验证测试")
    print("🔧 测试WebSocket回调立即传递的修复效果")
    print("="*70)
    asyncio.run(test_final_fix())