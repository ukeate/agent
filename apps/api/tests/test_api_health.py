"""
API健康检查测试
"""

import sys
import os
import asyncio
from unittest.mock import AsyncMock, patch

# 添加src目录到Python路径
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)

import pytest

@pytest.mark.asyncio
async def test_api_structure():
    """测试API结构"""
    try:
        # 导入主应用
        from main import app
        print("✅ FastAPI应用导入成功")
        
        # 检查路由注册
        routes = [route.path for route in app.routes]
        print(f"✅ 发现 {len(routes)} 个路由")
        
        # 检查智能体相关路由
        agent_routes = [route for route in routes if 'agent' in route.lower()]
        print(f"✅ 智能体相关路由: {len(agent_routes)} 个")
        
        for route in agent_routes:
            print(f"  - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ API结构测试失败: {e}")
        return False

@pytest.mark.asyncio
async def test_agent_service_creation():
    """测试智能体服务创建"""
    try:
        with patch('ai.openai_client.get_openai_client') as mock_openai, \
             patch('ai.mcp.client.get_mcp_client_manager') as mock_mcp:
            
            # 配置Mock
            mock_openai_client = AsyncMock()
            mock_openai.return_value = mock_openai_client
            
            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_available_tools.return_value = {}
            mock_mcp.return_value = mock_mcp_client
            
            # 测试服务创建
            from services.agent_service import get_agent_service
            
            service = await get_agent_service()
            await service.initialize()
            print("✅ 智能体服务创建成功")
            
            # 测试会话创建
            session_result = await service.create_agent_session(
                user_id="test_user",
                agent_type="react"
            )
            print(f"✅ 智能体会话创建成功: {session_result['conversation_id']}")
            
            return True
            
    except Exception as e:
        print(f"❌ 智能体服务测试失败: {e}")
        return False

@pytest.mark.asyncio
async def test_conversation_service():
    """测试对话服务"""
    try:
        from services.conversation_service import get_conversation_service
        
        service = await get_conversation_service()
        print("✅ 对话服务创建成功")
        
        # 创建测试对话
        conversation_id = await service.create_conversation(
            user_id="test_user",
            title="健康检查对话"
        )
        print(f"✅ 对话创建成功: {conversation_id}")
        
        # 添加测试消息
        message_id = await service.add_message(
            conversation_id=conversation_id,
            content="测试消息",
            sender_type="user"
        )
        print(f"✅ 消息添加成功: {message_id}")
        
        # 获取对话历史
        history = await service.get_conversation_history(conversation_id)
        print(f"✅ 对话历史获取成功: {len(history)} 条消息")
        
        return True
        
    except Exception as e:
        print(f"❌ 对话服务测试失败: {e}")
        return False

@pytest.mark.asyncio
async def test_react_agent():
    """测试ReAct智能体"""
    try:
        with patch('ai.openai_client.get_openai_client') as mock_openai, \
             patch('ai.mcp.client.get_mcp_client_manager') as mock_mcp:
            
            # 配置Mock
            mock_openai_client = AsyncMock()
            mock_openai_client.create_completion.return_value = {
                "content": "Final Answer: 测试回答完成"
            }
            mock_openai.return_value = mock_openai_client
            
            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_available_tools.return_value = {}
            mock_mcp.return_value = mock_mcp_client
            
            # 创建智能体
            from ai.agents.react_agent import ReActAgent
            
            agent = ReActAgent()
            await agent.initialize()
            print("✅ ReAct智能体初始化成功")
            
            # 测试会话运行
            session = await agent.run_session(
                user_input="测试用户输入",
                session_id="test_session"
            )
            print(f"✅ ReAct会话运行成功: {len(session.steps)} 个步骤")
            
            # 检查会话摘要
            summary = agent.get_session_summary("test_session")
            print(f"✅ 会话摘要生成成功: {summary['completed']}")
            
            return True
            
    except Exception as e:
        print(f"❌ ReAct智能体测试失败: {e}")
        return False

async def run_health_checks():
    """运行所有健康检查"""
    print("=== ReAct智能体系统健康检查 ===\n")
    
    tests = [
        ("API结构检查", test_api_structure),
        ("对话服务检查", test_conversation_service),
        ("ReAct智能体检查", test_react_agent),
        ("智能体服务检查", test_agent_service_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"--- {name} ---")
        try:
            if await test_func():
                print(f"✅ {name}通过\n")
                passed += 1
            else:
                print(f"❌ {name}失败\n")
        except Exception as e:
            print(f"❌ {name}异常: {e}\n")
    
    print(f"=== 健康检查结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 ReAct智能体系统健康检查全部通过！")
        print("系统已准备就绪，可以处理用户请求。")
        return True
    else:
        print("⚠️ 部分检查失败，系统可能存在问题。")
        return False

if __name__ == "__main__":
    asyncio.run(run_health_checks())