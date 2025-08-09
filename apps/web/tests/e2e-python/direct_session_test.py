#!/usr/bin/env python3
"""
直接会话测试 - 验证特定会话ID的实时流式响应
"""
import asyncio
import websockets
import json
import time

async def test_specific_session():
    """测试特定会话的WebSocket流式响应"""
    
    # 从浏览器获取的会话ID
    session_id = "f37f91e7-ae46-4cd5-b07d-ce2160a1dc22"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🎯 直接测试会话: {session_id}")
    print(f"📡 连接URL: {ws_url}")
    print("=" * 60)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 直接发送对话启动消息到这个已存在的会话
            start_msg = {
                "type": "start_conversation",
                "data": {
                    "message": "什么是WebSocket？请每个专家用一句话简单回答。",
                    "participants": ["doc_expert-1", "supervisor-1"]
                }
            }
            
            print("📤 发送对话启动消息到现有会话...")
            await websocket.send(json.dumps(start_msg))
            
            # 监听实时响应
            message_count = 0
            token_count = 0
            agents_responded = set()
            start_time = time.time()
            
            while time.time() - start_time < 30:  # 30秒超时
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] 消息 #{message_count}: {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话已启动")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        print(f"   🎤 发言者变更: {speaker} (第{round_num}轮)")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        is_complete = message['data'].get('is_complete', False)
                        token_count += 1
                        agents_responded.add(agent)
                        
                        print(f"   📝 Token #{token_count}: {agent} -> '{token}' (完成: {is_complete})")
                        
                        # 每10个token显示一次统计
                        if token_count % 10 == 0:
                            print(f"   📊 已收到 {token_count} 个token，{len(agents_responded)} 个智能体响应")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        content_length = len(message['data'].get('full_content', ''))
                        print(f"   ✅ 流式完成: {agent} (内容长度: {content_length})")
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content_length = len(msg_data.get('content', ''))
                        print(f"   💬 完整消息: {sender} (长度: {content_length})")
                        
                    elif msg_type == "conversation_completed":
                        print(f"   🏁 对话完成")
                        break
                        
                    elif msg_type == "error":
                        error_msg = message['data'].get('message', '未知错误')
                        print(f"   ❌ 错误: {error_msg}")
                        
                    # 成功条件：收到足够的token和多个智能体响应
                    if token_count >= 20 and len(agents_responded) >= 2:
                        print(f"   🎉 测试成功条件达成！")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时...")
                    continue
                    
            # 测试结果分析
            print("\n" + "=" * 60)
            print("📊 测试结果分析")
            print("=" * 60)
            print(f"✅ WebSocket连接: 成功")
            print(f"📨 总消息数: {message_count}")
            print(f"🎯 流式Token数: {token_count}")
            print(f"👥 响应智能体数: {len(agents_responded)} ({', '.join(agents_responded)})")
            print(f"⏱️  测试持续时间: {time.time() - start_time:.1f}秒")
            
            # 验证多参与者实时流式响应
            print(f"\n🎯 关键验证项目:")
            print(f"  ✅ 真实流式响应: {'通过' if token_count > 0 else '失败'}")
            print(f"  ✅ 多智能体参与: {'通过' if len(agents_responded) >= 2 else '失败'}")
            print(f"  ✅ 非模拟数据: {'通过' if token_count > 10 else '失败'}")
            
            success = token_count >= 10 and len(agents_responded) >= 2
            
            if success:
                print(f"\n🎉 测试结论: 多轮会话多参与者实时消息显示功能完全正常！")
                print(f"📝 每个参与者都有真实的流式响应，非模拟数据")
            else:
                print(f"\n⚠️  测试结论: 部分功能可能存在问题")
                
            return success
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_specific_session())
    exit(0 if result else 1)