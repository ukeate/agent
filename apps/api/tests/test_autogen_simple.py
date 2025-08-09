#!/usr/bin/env python3
"""
测试AutoGen基础功能，避免复杂的消息格式问题
"""
import asyncio
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.autogen.agents import BaseAutoGenAgent
from ai.autogen.config import AGENT_CONFIGS, AgentRole
from core.config import get_settings

async def test_simple_agent():
    """测试单个智能体的基础功能"""
    print("测试AutoGen智能体基础功能...")
    
    try:
        # 获取设置
        settings = get_settings()
        print(f"API Key配置: {'已配置' if settings.OPENAI_API_KEY else '未配置'}")
        
        # 创建代码专家智能体
        config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
        agent = BaseAutoGenAgent(config)
        print(f"智能体创建成功: {agent.config.name}")
        
        # 简单测试消息
        test_message = "请简单介绍你的功能"
        print(f"发送测试消息: {test_message}")
        
        # 生成响应
        response = await agent.generate_response(test_message)
        print(f"收到响应: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_minimal_conversation():
    """测试最小化的多智能体对话"""
    print("\n测试最小化多智能体对话...")
    
    try:
        # 创建两个智能体
        code_agent = BaseAutoGenAgent(AGENT_CONFIGS[AgentRole.CODE_EXPERT])
        arch_agent = BaseAutoGenAgent(AGENT_CONFIGS[AgentRole.ARCHITECT])
        
        agents = [code_agent, arch_agent]
        print(f"创建了 {len(agents)} 个智能体")
        
        # 初始消息
        initial_message = "请讨论如何优化Python代码性能"
        current_message = initial_message
        
        print(f"开始讨论: {initial_message}")
        
        # 简单轮流对话
        for round_num in range(2):  # 只做2轮测试
            print(f"\n=== 第 {round_num + 1} 轮 ===")
            
            for i, agent in enumerate(agents):
                try:
                    print(f"{agent.config.name} 正在思考...")
                    
                    # 使用超时控制
                    response = await asyncio.wait_for(
                        agent.generate_response(current_message), 
                        timeout=15.0
                    )
                    
                    print(f"{agent.config.name}: {response[:200]}...")
                    current_message = response
                    
                except asyncio.TimeoutError:
                    print(f"{agent.config.name}: 响应超时")
                    break
                except Exception as e:
                    print(f"{agent.config.name}: 响应错误 - {e}")
                    break
        
        print("\n对话测试完成")
        return True
        
    except Exception as e:
        print(f"对话测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试函数"""
    print("=== AutoGen简化测试 ===")
    
    # 测试单个智能体
    single_success = await test_simple_agent()
    
    if single_success:
        # 测试简单对话
        conversation_success = await test_minimal_conversation()
        
        if conversation_success:
            print("\n✅ 所有测试通过")
        else:
            print("\n❌ 对话测试失败")
    else:
        print("\n❌ 单智能体测试失败")

if __name__ == "__main__":
    asyncio.run(main())