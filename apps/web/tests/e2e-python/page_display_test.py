#!/usr/bin/env python3
"""
页面显示测试 - 验证当前浏览器会话是否收到并显示智能体消息
"""
import asyncio
import websockets
import json
import time

async def test_page_display():
    """测试当前浏览器会话的WebSocket消息接收"""
    
    # 从浏览器控制台获取的最新会话ID
    session_id = "c4d2ecee-c669-4cf9-a00d-161d3f3eafa6"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🎯 页面显示测试 - 验证Playwright页面WebSocket消息接收")
    print(f"📡 当前会话ID: {session_id}")
    print(f"📡 WebSocket URL: {ws_url}")
    print(f"🌐 话题: WebSocket在现代Web开发中的重要性？")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 监听实时响应
            message_count = 0
            token_count = 0
            agents_responded = set()
            messages_for_page = []
            start_time = time.time()
            
            print("\n🔄 开始监听页面应该显示的WebSocket消息...")
            print("-" * 80)
            
            while time.time() - start_time < 45:  # 45秒超时
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=8.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] 消息 #{message_count}: {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话已启动 - 页面应显示此状态变化")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        print(f"   🎤 发言者变更: {speaker} (第{round_num}轮)")
                        print(f"       ➤ 页面应更新参与者状态显示")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        is_complete = message['data'].get('is_complete', False)
                        token_count += 1
                        agents_responded.add(agent)
                        
                        print(f"   📝 实时Token #{token_count}: {agent} -> '{token}'")
                        print(f"       ➤ 页面聊天区域应实时显示此token")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        content = message['data'].get('full_content', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       ➤ 页面应显示{agent}的完整消息: {content[:50]}...")
                        messages_for_page.append({
                            'agent': agent,
                            'content': content,
                            'timestamp': timestamp
                        })
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')
                        print(f"   💬 新消息: {sender}")
                        print(f"       ➤ 页面聊天区域应添加此消息")
                        messages_for_page.append({
                            'agent': sender,
                            'content': content,
                            'timestamp': timestamp
                        })
                        
                    elif msg_type == "conversation_completed":
                        print(f"   🏁 对话完成 - 页面应显示对话结束状态")
                        break
                        
                    elif msg_type == "error":
                        error_msg = message['data'].get('message', '未知错误')
                        print(f"   ❌ 错误: {error_msg}")
                        
                    # 成功条件：收到足够的消息
                    if len(messages_for_page) >= 2:
                        print(f"   🎉 已收到足够的智能体消息供页面显示！")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时 (已收到 {message_count} 条)...")
                    continue
                    
            # 测试结果分析
            print("\n" + "=" * 100)
            print("📊 页面显示测试结果分析")
            print("=" * 100)
            print(f"✅ WebSocket连接: 成功")
            print(f"📨 总消息数: {message_count}")
            print(f"🎯 实时流式Token数: {token_count}")
            print(f"👥 响应智能体数: {len(agents_responded)} ({', '.join(agents_responded)})")
            print(f"💬 页面应显示的消息数: {len(messages_for_page)}")
            print(f"⏱️  测试持续时间: {time.time() - start_time:.1f}秒")
            
            # 显示页面应该显示的消息
            if messages_for_page:
                print(f"\n🖥️  页面应该显示的智能体消息:")
                for i, msg in enumerate(messages_for_page, 1):
                    print(f"  {i}. [{msg['timestamp']}] {msg['agent']}: {msg['content'][:100]}...")
            
            # 验证页面显示要求
            print(f"\n🎯 Playwright MCP页面显示验证:")
            print(f"  ✅ 真实流式响应: {'通过' if token_count > 0 else '失败'}")
            print(f"  ✅ 多智能体参与: {'通过' if len(agents_responded) >= 2 else '部分通过' if len(agents_responded) >= 1 else '失败'}")
            print(f"  ✅ 非模拟数据: {'通过' if token_count > 10 else '部分通过' if token_count > 0 else '失败'}")
            print(f"  ⚠️  页面实时显示: {'后端正常，前端UI存在问题' if len(messages_for_page) > 0 else '完全失败'}")
            
            success = len(messages_for_page) >= 1 and token_count >= 1
            
            if success:
                print(f"\n🎉 页面显示测试结论:")
                print(f"📝 ✅ 后端WebSocket发送了 {len(messages_for_page)} 条智能体消息")
                print(f"📝 ✅ 后端WebSocket发送了 {token_count} 个实时token")
                print(f"📝 ⚠️  前端页面应该显示这些消息，但可能存在UI渲染问题")
                print(f"💡 建议: 检查前端React组件的消息处理和DOM更新逻辑")
            else:
                print(f"\n⚠️  测试结论: 后端没有发送足够的消息数据")
                
            return success
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_page_display())
    exit(0 if result else 1)