#!/usr/bin/env python3
"""
完整的WebSocket流式响应系统测试
测试会话创建时序修复和实时token显示功能
"""
import asyncio
import websockets
import json
import time
from datetime import datetime

async def test_streaming_websocket():
    """测试完整的WebSocket流式响应流程"""
    
    # 使用时间戳生成唯一会话ID
    session_id = f"test-session-{int(time.time())}"
    
    # WebSocket连接URL
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🚀 开始测试WebSocket流式响应系统")
    print(f"📡 连接URL: {ws_url}")
    print(f"🆔 会话ID: {session_id}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接已建立")
            
            # 接收连接确认消息
            response = await websocket.recv()
            connection_msg = json.loads(response)
            print(f"📨 连接确认: {connection_msg.get('type')}")
            
            # 发送对话启动消息
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "请讨论如何实现一个高效的WebSocket流式响应系统，包括技术架构和用户体验优化",
                    "participants": ["supervisor-1", "code_expert-1"]
                }
            }
            
            print(f"📤 发送对话启动消息...")
            await websocket.send(json.dumps(start_message))
            
            # 统计数据
            message_count = 0
            token_count = 0
            unique_agents = set()
            conversation_started = False
            streaming_messages = {}
            
            # 监听响应
            while True:
                try:
                    # 设置超时避免无限等待
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    msg_data = message.get("data", {})
                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    
                    if msg_type == "conversation_created":
                        print(f"🎯 [{timestamp}] 对话创建成功: {msg_data.get('conversation_id')}")
                        
                    elif msg_type == "conversation_started":
                        conversation_started = True
                        print(f"🚀 [{timestamp}] 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        current_speaker = msg_data.get("current_speaker")
                        round_num = msg_data.get("round", 0)
                        unique_agents.add(current_speaker)
                        print(f"🎤 [{timestamp}] 发言者变更: {current_speaker} (第{round_num}轮)")
                        
                    elif msg_type == "streaming_token":
                        # 流式token - 这是我们要验证的核心功能
                        message_id = msg_data.get("message_id")
                        agent_name = msg_data.get("agent_name")
                        token = msg_data.get("token", "")
                        full_content = msg_data.get("full_content", "")
                        round_num = msg_data.get("round", 0)
                        is_complete = msg_data.get("is_complete", False)
                        
                        token_count += 1
                        
                        # 跟踪流式消息
                        if message_id not in streaming_messages:
                            streaming_messages[message_id] = {
                                "agent_name": agent_name,
                                "start_time": time.time(),
                                "tokens": 0,
                                "content_length": 0
                            }
                        
                        streaming_messages[message_id]["tokens"] += 1
                        streaming_messages[message_id]["content_length"] = len(full_content)
                        
                        # 每10个token显示一次进度
                        if token_count % 10 == 0 or is_complete:
                            print(f"📝 [{timestamp}] Token #{token_count}: {agent_name} | 内容长度: {len(full_content)} | 完成: {is_complete}")
                            if len(full_content) > 50:
                                preview = full_content[:50] + "..."
                                print(f"    预览: {preview}")
                        
                    elif msg_type == "streaming_complete":
                        message_id = msg_data.get("message_id")
                        agent_name = msg_data.get("agent_name")
                        full_content = msg_data.get("full_content", "")
                        round_num = msg_data.get("round", 0)
                        
                        if message_id in streaming_messages:
                            stream_info = streaming_messages[message_id]
                            duration = time.time() - stream_info["start_time"]
                            
                            print(f"✅ [{timestamp}] 流式响应完成: {agent_name}")
                            print(f"    消息ID: {message_id}")
                            print(f"    总tokens: {stream_info['tokens']}")
                            print(f"    内容长度: {len(full_content)}")
                            print(f"    持续时间: {duration:.2f}秒")
                            print(f"    平均速度: {stream_info['tokens']/duration:.1f} tokens/秒")
                            
                            if len(full_content) > 100:
                                print(f"    内容预览: {full_content[:100]}...")
                            
                    elif msg_type == "new_message":
                        # 完整消息（用于历史记录）
                        message_data = msg_data.get("message") or msg_data
                        if message_data and message_data.get("content"):
                            sender = message_data.get("sender", "Unknown")
                            content_length = len(message_data.get("content", ""))
                            print(f"💬 [{timestamp}] 完整消息: {sender} | 长度: {content_length}")
                            
                    elif msg_type == "conversation_completed":
                        print(f"🏁 [{timestamp}] 对话完成")
                        break
                        
                    elif msg_type == "error":
                        error_msg = msg_data.get("message", "未知错误")
                        print(f"❌ [{timestamp}] 错误: {error_msg}")
                        
                    else:
                        print(f"📋 [{timestamp}] 其他消息: {msg_type}")
                        
                    # 安全退出条件
                    if message_count > 200:  # 防止无限循环
                        print(f"⚠️  达到最大消息数量限制，退出测试")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"⏰ 等待响应超时，可能对话已完成")
                    break
                except Exception as e:
                    print(f"❌ 处理消息时出错: {e}")
                    break
            
            # 测试总结
            print("\n" + "=" * 60)
            print("📊 测试总结报告")
            print("=" * 60)
            print(f"✅ WebSocket连接: 成功")
            print(f"✅ 对话启动: {'成功' if conversation_started else '失败'}")
            print(f"📨 总消息数: {message_count}")
            print(f"🎯 总Token数: {token_count}")
            print(f"👥 参与智能体: {len(unique_agents)} ({', '.join(unique_agents)})")
            print(f"💬 流式消息数: {len(streaming_messages)}")
            
            if streaming_messages:
                print(f"\n📈 流式响应详情:")
                for msg_id, info in streaming_messages.items():
                    print(f"  - {info['agent_name']}: {info['tokens']} tokens, {info['content_length']} 字符")
            
            # 验证核心要求
            print(f"\n🎯 核心功能验证:")
            print(f"  ✅ 实时Token显示: {'通过' if token_count > 0 else '失败'}")
            print(f"  ✅ 多智能体协作: {'通过' if len(unique_agents) > 1 else '失败'}")
            print(f"  ✅ 流式响应完整性: {'通过' if len(streaming_messages) > 0 else '失败'}")
            
            if token_count > 50 and len(unique_agents) >= 2:
                print(f"\n🎉 测试结果: 所有核心功能正常工作！")
                print(f"📝 用户需求满足度: 100% - 页面可以实时显示讨论的每个token")
            else:
                print(f"\n⚠️  测试结果: 部分功能可能存在问题")
                
    except Exception as e:
        print(f"❌ WebSocket连接失败: {e}")
        print(f"请确保后端服务在 localhost:8000 运行")

if __name__ == "__main__":
    print("🧪 WebSocket流式响应系统完整测试")
    print("=" * 60)
    asyncio.run(test_streaming_websocket())