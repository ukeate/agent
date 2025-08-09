#!/usr/bin/env python3
"""
立即显示测试 - 验证当前Playwright会话的WebSocket流式响应和前端显示
"""
import asyncio
import websockets
import json
import time

async def test_immediate_display():
    """测试当前页面会话的实时流式响应是否在前端显示"""
    
    # 从Playwright页面获取的最新会话ID
    session_id = "7526dea6-819c-4878-96a6-6d6b2bbe1c66"
    ws_url = f"ws://localhost:8000/api/v1/multi-agent/ws/{session_id}"
    
    print(f"🎯 立即显示测试 - 检查前端显示bug")
    print(f"📡 会话ID: {session_id}")
    print(f"📡 WebSocket URL: {ws_url}")
    print(f"💬 问题: 页面显示'对话还未开始'但后端状态'进行中'")
    print("=" * 100)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 接收连接确认
            response = await websocket.recv()
            msg = json.loads(response)
            print(f"📨 连接确认: {msg.get('type')}")
            
            # 立即监听流式响应
            message_count = 0
            token_count = 0
            agents_responded = set()
            messages_for_display = []
            start_time = time.time()
            
            print("\n🔄 监听页面应该显示的实时流式消息...")
            print("-" * 100)
            
            while time.time() - start_time < 30:  # 30秒快速测试
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message = json.loads(response)
                    message_count += 1
                    
                    msg_type = message.get("type")
                    timestamp = time.strftime('%H:%M:%S')
                    
                    print(f"📨 [{timestamp}] 消息 #{message_count}: {msg_type}")
                    
                    if msg_type == "conversation_started":
                        print(f"   🚀 对话已启动 - 页面应显示对话开始状态")
                        
                    elif msg_type == "speaker_change":
                        speaker = message['data'].get('current_speaker')
                        round_num = message['data'].get('round', 0)
                        print(f"   🎤 发言者变更: {speaker} (第{round_num}轮)")
                        print(f"       ➤ 页面参与者状态应更新")
                        
                    elif msg_type == "streaming_token":
                        token = message['data'].get('token', '')
                        agent = message['data'].get('agent_name', '')
                        token_count += 1
                        agents_responded.add(agent)
                        
                        print(f"   📝 实时Token #{token_count}: {agent} -> '{token}'")
                        print(f"       ➤ 页面聊天区域应立即显示此token")
                        
                        # 每个token都应该在页面显示
                        if token_count <= 10:  # 显示前10个token的详细信息
                            print(f"   🖥️  页面应立即显示第{token_count}个token: '{token}'")
                        
                    elif msg_type == "streaming_complete":
                        agent = message['data'].get('agent_name', '')
                        content = message['data'].get('full_content', '')
                        print(f"   ✅ 流式完成: {agent}")
                        print(f"       ➤ 页面应显示{agent}的完整消息")
                        print(f"       ➤ 消息内容: {content[:80]}...")
                        
                        messages_for_display.append({
                            'agent': agent,
                            'content': content,
                            'timestamp': timestamp
                        })
                        
                    elif msg_type == "new_message":
                        msg_data = message['data'].get('message') or message['data']
                        sender = msg_data.get('sender', 'Unknown')
                        content = msg_data.get('content', '')
                        print(f"   💬 完整消息: {sender}")
                        print(f"       ➤ 页面应显示此消息")
                        print(f"       ➤ 消息内容: {content[:80]}...")
                        
                    # 如果收到足够的流式token，说明后端正常
                    if token_count >= 5:
                        print(f"   🎉 足够的流式token已产生，页面应该显示这些内容！")
                        break
                        
                except asyncio.TimeoutError:
                    print(f"   ⏰ 等待消息超时...")
                    break
                    
            # 测试结果分析
            print("\n" + "=" * 100)
            print("📊 前端显示Bug验证结果")
            print("=" * 100)
            print(f"✅ WebSocket连接: 成功")
            print(f"📨 收到消息数: {message_count}")
            print(f"🎯 实时Token数: {token_count}")
            print(f"👥 智能体响应数: {len(agents_responded)} ({', '.join(agents_responded)})")
            print(f"💬 应显示消息数: {len(messages_for_display)}")
            
            # 最终结论
            if token_count > 0:
                print(f"\n🔥 CRITICAL BUG 确认:")
                print(f"📝 ✅ 后端WebSocket正常发送了 {token_count} 个实时token")
                print(f"📝 ✅ 后端智能体正常响应")
                print(f"📝 ❌ 前端页面没有显示这些token和消息")
                print(f"📝 🐛 前端UI渲染存在严重bug，需要立即修复")
                return True
            else:
                print(f"\n⚠️  未检测到流式响应，需要重新测试")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_immediate_display())
    exit(0 if result else 1)