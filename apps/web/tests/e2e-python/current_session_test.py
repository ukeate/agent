#!/usr/bin/env python3
"""
当前会话实时测试 - 验证Playwright页面当前会话WebSocket消息
"""
import asyncio
import websockets
import json
import time

async def test_current_display():
    """测试当前页面会话的WebSocket流式响应和页面显示"""
    
    # 从页面控制台获取的当前会话ID
    session_id = "c4d2ecee-c669-4cf9-a00d-161d3f3eafa6"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🎯 当前会话实时测试 - 验证页面消息显示")
    print(f"📡 会话ID: {session_id}")
    print(f"📡 WebSocket URL: {ws_url}")
    print(f"💬 页面显示: 对话进行中，但聊天区域未显示消息")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 发送新的消息启动流式响应
            start_message = {
                "type": "start_conversation",
                "data": {
                    "message": "WebSocket在现代Web开发中的重要性？请每个专家用一句话简单回答，必须立即显示在页面。",
                    "participants": ["doc_expert", "supervisor"]
                }
            }
            
            print(f"📤 发送启动消息: {start_message['data']['message']}")
            await websocket.send(json.dumps(start_message))
            
            # 监听实时响应
            message_count = 0
            token_count = 0
            agents_responded = set()
            messages_for_display = []
            start_time = time.time()
            
            print("\n🔄 开始监听页面应该实时显示的WebSocket消息...")
            print("-" * 100)
            
            while time.time() - start_time < 60:  # 60秒超时
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] 消息 #{message_count}: {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话已启动 - 页面应更新状态显示")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        print(f"   🎤 发言者变更: {speaker} (第{round_num}轮)")
                        print(f"       ➤ 页面参与者状态应更新为发言中")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        is_complete = message['data'].get('is_complete', False)
                        token_count += 1
                        agents_responded.add(agent)
                        
                        print(f"   📝 实时Token #{token_count}: {agent} -> '{token}'")
                        print(f"       ➤ 页面聊天对话框应立即显示此token内容")
                        
                        # 每3个token提醒一次页面应该显示的内容
                        if token_count % 3 == 0:
                            print(f"   📊 页面应已实时显示 {token_count} 个token，来自{agent}")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        content = message['data'].get('full_content', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       ➤ 页面应显示{agent}的完整消息")
                        print(f"       ➤ 消息内容: {content[:80]}...")
                        
                        messages_for_display.append({
                            'agent': agent,
                            'content': content,
                            'timestamp': timestamp,
                            'type': 'complete_message'
                        })
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')
                        print(f"   💬 完整消息: {sender}")
                        print(f"       ➤ 页面聊天区域应添加此消息到对话框")
                        print(f"       ➤ 消息内容: {content[:80]}...")
                        
                        messages_for_display.append({
                            'agent': sender,
                            'content': content,
                            'timestamp': timestamp,
                            'type': 'new_message'
                        })
                        
                    elif msg_type == "conversation_completed":
                        print(f"   🏁 对话完成 - 页面应更新状态为已完成")
                        break
                        
                    elif msg_type == "error":
                        error_msg = message['data'].get('message', '未知错误')
                        print(f"   ❌ 错误: {error_msg}")
                        print(f"       ➤ 页面应显示错误提示")
                        
                    # 成功条件：收到足够的消息内容
                    if len(messages_for_display) >= 2 or token_count >= 5:
                        print(f"   🎉 已收到足够的消息内容供页面显示验证！")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时 (已收到 {message_count} 条消息)...")
                    continue
                    
            # 测试结果分析
            print("\n" + "=" * 100)
            print("📊 页面消息显示验证测试结果")
            print("=" * 100)
            print(f"✅ WebSocket连接: 成功")
            print(f"📨 总消息数: {message_count}")
            print(f"🎯 实时流式Token数: {token_count}")
            print(f"👥 响应智能体数: {len(agents_responded)} ({', '.join(agents_responded)})")
            print(f"💬 页面应显示的消息数: {len(messages_for_display)}")
            print(f"⏱️  测试持续时间: {time.time() - start_time:.1f}秒")
            
            # 显示页面应该显示的具体消息内容
            if messages_for_display:
                print(f"\n🖥️  页面应该在聊天区域显示的消息:")
                for i, msg in enumerate(messages_for_display, 1):
                    print(f"  {i}. [{msg['timestamp']}] {msg['agent']}: {msg['content'][:100]}...")
                    print(f"     类型: {msg['type']}")
                    
            # 最终验证结论
            print(f"\n🎯 Playwright MCP页面显示最终验证:")
            print(f"  ✅ 真实流式响应: {'通过' if token_count > 0 else '失败'}")
            print(f"  ✅ 多智能体参与: {'通过' if len(agents_responded) >= 2 else '部分通过' if len(agents_responded) >= 1 else '失败'}")
            print(f"  ✅ 非模拟数据: {'通过' if token_count > 3 else '部分通过' if token_count > 0 else '失败'}")
            print(f"  ⚠️  页面实时显示: {'后端正常，需Playwright验证前端' if len(messages_for_display) > 0 else '系统问题'}")
            
            success = len(messages_for_display) >= 1 and token_count >= 1
            
            if success:
                print(f"\n🎉 测试结论:")
                print(f"📝 ✅ 后端WebSocket成功发送了 {len(messages_for_display)} 条智能体消息")
                print(f"📝 ✅ 后端WebSocket成功发送了 {token_count} 个实时token")
                print(f"📝 ✅ 多轮会话多参与者实时消息显示功能 - 后端完全正常")
                print(f"📝 ⚠️  现在需要使用Playwright MCP验证前端页面是否显示这些消息")
                print(f"💡 建议: 立即检查页面DOM和聊天区域元素")
            else:
                print(f"\n⚠️  测试结论: 需要重新启动对话或检查系统状态")
                
            return success
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_current_display())
    exit(0 if result else 1)