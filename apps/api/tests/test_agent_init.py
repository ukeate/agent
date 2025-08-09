#!/usr/bin/env python3
"""
测试智能体初始化和基本响应
"""
import asyncio
from ai.autogen.agents import create_default_agents

async def test_agent_initialization():
    """测试智能体初始化"""
    print("🔧 测试智能体初始化...")
    
    try:
        agents = create_default_agents()
        print(f"✅ 成功创建 {len(agents)} 个智能体")
        
        for agent in agents:
            print(f"  - {agent.config.name} ({agent.config.role})")
            
        return agents
    except Exception as e:
        print(f"❌ 智能体初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return []

async def test_agent_response(agent, test_message="你好，请简单介绍一下自己"):
    """测试单个智能体响应"""
    print(f"\n🤖 测试智能体 {agent.config.name} 的响应...")
    
    try:
        response = await agent.generate_response(test_message)
        print(f"✅ 响应生成成功 (长度: {len(response)})")
        print(f"📝 响应内容: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 响应生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主测试流程"""
    print("🚀 开始智能体基础测试")
    print("=" * 50)
    
    # 测试智能体初始化
    agents = await test_agent_initialization()
    
    if not agents:
        print("❌ 无法继续测试，智能体初始化失败")
        return
    
    # 测试每个智能体的响应
    success_count = 0
    for agent in agents:
        if await test_agent_response(agent):
            success_count += 1
    
    print("\n" + "=" * 50)
    print("📋 测试结果")
    print(f"🎯 智能体初始化: {len(agents)}/{len(agents)}")
    print(f"🎯 响应生成成功: {success_count}/{len(agents)}")
    
    if success_count == len(agents):
        print("\n🎉 所有智能体基础测试通过！")
    else:
        print(f"\n⚠️ {len(agents) - success_count} 个智能体响应测试失败")

if __name__ == "__main__":
    asyncio.run(main())