#!/usr/bin/env python3
"""
调试对话流程，查找为什么智能体没有生成响应
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

async def debug_conversation_flow():
    """调试对话流程"""
    print("🚀 开始调试对话流程...")
    
    try:
        # 1. 测试智能体创建
        print("\n📋 1. 测试智能体创建...")
        from ai.autogen.agents import create_default_agents
        agents = create_default_agents()
        print(f"✅ 创建了 {len(agents)} 个智能体")
        
        # 2. 测试单个智能体响应
        print("\n📋 2. 测试单个智能体响应...")
        test_agent = agents[0]  # 使用第一个智能体
        print(f"测试智能体: {test_agent.config.name}")
        
        try:
            print("正在生成响应...")
            response = await test_agent.generate_response("你好，请简单介绍一下你自己")
            print(f"✅ 智能体响应成功，长度: {len(response)} 字符")
            print(f"响应内容: {response[:200]}...")
        except Exception as e:
            print(f"❌ 智能体响应失败: {e}")
            return
        
        # 3. 测试对话会话创建
        print("\n📋 3. 测试对话会话创建...")
        from ai.autogen.group_chat import GroupChatManager, ConversationConfig
        from core.constants import ConversationConstants
        
        manager = GroupChatManager()
        session = await manager.create_session(
            participants=agents[:2],  # 使用前两个智能体
            config=ConversationConfig(
                max_rounds=2,
                timeout_seconds=ConversationConstants.DEFAULT_TIMEOUT_SECONDS
            ),
            initial_topic="测试对话"
        )
        print(f"✅ 对话会话创建成功: {session.session_id}")
        
        # 4. 测试对话启动（带回调）
        print("\n📋 4. 测试对话启动...")
        messages_received = []
        
        async def test_callback(data):
            """测试回调函数"""
            message_type = data.get('type', 'unknown')
            print(f"📨 收到回调消息: {message_type}")
            messages_received.append(data)
            
            if message_type == 'new_message':
                message_data = data.get('data', {})
                if 'message' in message_data:
                    msg = message_data['message']
                else:
                    msg = message_data
                sender = msg.get('sender', 'Unknown')
                content = msg.get('content', '')
                print(f"💬 {sender}: {content[:100]}...")
        
        # 启动对话
        print("启动对话...")
        result = await session.start_conversation("请各位简单介绍一下自己的专业领域", test_callback)
        print(f"✅ 对话启动结果: {result}")
        
        # 等待一段时间让对话进行
        print("\n⏳ 等待对话进行 (30秒)...")
        await asyncio.sleep(30)
        
        # 5. 检查结果
        print(f"\n📊 对话结果统计:")
        print(f"  收到回调消息数: {len(messages_received)}")
        print(f"  对话消息数: {len(session.messages)}")
        print(f"  当前轮次: {session.round_count}")
        print(f"  对话状态: {session.status}")
        
        if messages_received:
            print(f"\n📝 收到的回调消息类型:")
            for msg in messages_received:
                print(f"  - {msg.get('type', 'unknown')}")
        
        if session.messages:
            print(f"\n💬 对话消息:")
            for i, msg in enumerate(session.messages):
                print(f"  {i+1}. {msg['sender']}: {msg['content'][:100]}...")
        else:
            print("\n❌ 没有收到任何对话消息")
        
    except Exception as e:
        print(f"\n❌ 调试过程中出错: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await debug_conversation_flow()

if __name__ == "__main__":
    asyncio.run(main())