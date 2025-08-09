#!/usr/bin/env python3
"""
测试WebSocket流式多智能体对话功能

这个脚本测试：
1. WebSocket连接建立
2. 多智能体对话创建
3. 流式响应token接收
4. 消息完成通知
"""

import asyncio
import json
import websockets
import uuid
from datetime import datetime

async def test_streaming_websocket():
    """测试流式WebSocket功能"""
    
    # 生成测试session ID
    session_id = f"test-session-{int(datetime.now().timestamp())}"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"连接WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送启动对话消息
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "请讨论如何实现WebSocket流式响应功能",
                    "participants": ["supervisor-1", "code_expert-1"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send(json.dumps(start_message))
            print("📤 已发送启动对话消息")
            
            # 监听消息
            token_count = 0
            message_count = 0
            streaming_messages = {}
            
            async for message_raw in websocket:
                try:
                    message = json.loads(message_raw)
                    msg_type = message.get("type")
                    msg_data = message.get("data", {})
                    
                    if msg_type == "connection_established":
                        print("🔗 WebSocket连接确认")
                        
                    elif msg_type == "conversation_created":
                        print(f"🎯 对话创建成功: {msg_data.get('conversation_id')}")
                        
                    elif msg_type == "conversation_started":
                        print("🚀 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        current_speaker = msg_data.get("current_speaker")
                        round_num = msg_data.get("round")
                        print(f"🎤 发言者变更: {current_speaker} (第{round_num}轮)")
                        
                    elif msg_type == "streaming_token":
                        # 流式token
                        message_id = msg_data.get("message_id")
                        agent_name = msg_data.get("agent_name")
                        token = msg_data.get("token")
                        full_content = msg_data.get("full_content", "")
                        
                        if message_id not in streaming_messages:
                            streaming_messages[message_id] = {
                                "agent_name": agent_name,
                                "content": "",
                                "token_count": 0
                            }
                            print(f"\n🔄 开始接收流式响应 - {agent_name}")
                        
                        streaming_messages[message_id]["content"] = full_content
                        streaming_messages[message_id]["token_count"] += 1
                        token_count += 1
                        
                        # 实时显示token (只显示最后50个字符避免刷屏)
                        display_content = full_content[-50:] if len(full_content) > 50 else full_content
                        print(f"📝 Token #{token_count}: ...{display_content}", end="\r")
                        
                    elif msg_type == "streaming_complete":
                        # 流式响应完成
                        message_id = msg_data.get("message_id")
                        agent_name = msg_data.get("agent_name")
                        full_content = msg_data.get("full_content", "")
                        
                        if message_id in streaming_messages:
                            token_count_for_msg = streaming_messages[message_id]["token_count"]
                            print(f"\n✅ {agent_name} 流式响应完成 - 共{token_count_for_msg}个token")
                            print(f"📄 完整内容: {full_content[:100]}{'...' if len(full_content) > 100 else ''}")
                        
                    elif msg_type == "streaming_error":
                        # 流式响应错误
                        message_id = msg_data.get("message_id")
                        agent_name = msg_data.get("agent_name")
                        error = msg_data.get("error")
                        print(f"\n❌ {agent_name} 流式响应错误: {error}")
                        
                    elif msg_type == "new_message":
                        # 完整消息通知
                        message_data = msg_data
                        sender = message_data.get("sender")
                        content = message_data.get("content", "")
                        message_count += 1
                        print(f"\n💬 消息#{message_count} - {sender}: {content[:50]}{'...' if len(content) > 50 else ''}")
                        
                    elif msg_type == "conversation_completed":
                        print(f"\n🎉 对话完成 - 总消息数: {msg_data.get('total_messages')}, 总轮数: {msg_data.get('total_rounds')}")
                        break
                        
                    elif msg_type == "conversation_error":
                        error = msg_data.get("error")
                        print(f"\n❌ 对话错误: {error}")
                        break
                        
                    else:
                        print(f"\n📡 其他消息: {msg_type}")
                        
                except json.JSONDecodeError:
                    print(f"⚠️  无法解析消息: {message_raw}")
                except Exception as e:
                    print(f"⚠️  处理消息时出错: {e}")
                    
                # 限制测试时间，避免无限等待
                if token_count > 100:  # 接收100个token后停止
                    print(f"\n⏱️  已接收{token_count}个token，测试完成")
                    break
                    
    except websockets.exceptions.ConnectionClosed:
        print("🔌 WebSocket连接已关闭")
    except Exception as e:
        print(f"❌ WebSocket连接错误: {e}")

if __name__ == "__main__":
    print("🧪 开始测试WebSocket流式多智能体对话功能")
    print("=" * 60)
    asyncio.run(test_streaming_websocket())
    print("=" * 60)
    print("🏁 测试完成")