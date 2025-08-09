"""
基础功能验证测试
"""

import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
sys.path.insert(0, src_dir)
print(f"添加路径: {src_dir}")

def test_basic_imports():
    """测试基本模块导入"""
    try:
        # 测试核心配置导入
        from core.config import get_settings
        settings = get_settings()
        print(f"✅ 配置加载成功: {type(settings)}")
        
        # 测试ReAct智能体导入
        from ai.agents.react_agent import ReActAgent, ReActStep, ReActStepType
        print(f"✅ ReAct智能体导入成功: {ReActAgent}")
        
        # 测试数据结构
        step = ReActStep(content="测试")
        print(f"✅ ReAct数据结构创建成功: {step.step_type}")
        
        # 测试服务导入
        from services.conversation_service import ConversationService
        from services.agent_service import AgentService
        print(f"✅ 服务层导入成功")
        
        # 测试API导入
        from api.v1.agents import CreateAgentSessionRequest
        print(f"✅ API模型导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_basic_data_structures():
    """测试基本数据结构"""
    try:
        from ai.agents.react_agent import ReActStep, ReActStepType, ReActSession
        
        # 测试步骤创建
        step = ReActStep(
            step_type=ReActStepType.THOUGHT,
            content="这是一个测试思考"
        )
        assert step.step_type == ReActStepType.THOUGHT
        assert step.content == "这是一个测试思考"
        print(f"✅ ReAct步骤创建成功")
        
        # 测试会话创建
        session = ReActSession(max_steps=5)
        assert session.max_steps == 5
        assert len(session.steps) == 0
        print(f"✅ ReAct会话创建成功")
        
        # 测试步骤添加
        session.steps.append(step)
        assert len(session.steps) == 1
        print(f"✅ 会话步骤管理成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据结构测试失败: {e}")
        return False

def test_basic_parsing():
    """测试基本解析功能"""
    try:
        from ai.agents.react_agent import ReActAgent, ReActStepType
        
        agent = ReActAgent()
        
        # 测试思考解析
        thought_response = "Thought: 我需要分析这个问题"
        step = agent._parse_response(thought_response)
        assert step.step_type == ReActStepType.THOUGHT
        assert step.content == "我需要分析这个问题"
        print(f"✅ 思考解析成功")
        
        # 测试行动解析
        action_response = '''Action: read_file
Action Input: {"file_path": "test.txt"}'''
        step = agent._parse_response(action_response)
        assert step.step_type == ReActStepType.ACTION
        assert step.tool_name == "read_file"
        assert step.tool_args == {"file_path": "test.txt"}
        print(f"✅ 行动解析成功")
        
        # 测试最终答案解析
        final_response = "Final Answer: 这是最终答案"
        step = agent._parse_response(final_response)
        assert step.step_type == ReActStepType.FINAL_ANSWER
        assert step.content == "这是最终答案"
        print(f"✅ 最终答案解析成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 解析功能测试失败: {e}")
        return False

def test_configuration():
    """测试配置加载"""
    try:
        from core.config import get_settings
        
        settings = get_settings()
        
        # 验证基本配置存在
        assert hasattr(settings, 'DEBUG')
        assert hasattr(settings, 'HOST')
        assert hasattr(settings, 'PORT')
        print(f"✅ 基本配置验证成功")
        
        # 验证AI配置
        assert hasattr(settings, 'OPENAI_API_KEY')
        assert hasattr(settings, 'MAX_CONTEXT_LENGTH')
        assert hasattr(settings, 'SESSION_TIMEOUT_MINUTES')
        print(f"✅ AI配置验证成功")
        
        # 验证默认值
        assert settings.MAX_CONTEXT_LENGTH == 100000
        assert settings.SESSION_TIMEOUT_MINUTES == 60
        print(f"✅ 配置默认值验证成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=== ReAct智能体基础功能验证 ===")
    
    tests = [
        ("模块导入", test_basic_imports),
        ("数据结构", test_basic_data_structures),
        ("解析功能", test_basic_parsing),
        ("配置加载", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n--- {name}测试 ---")
        try:
            if test_func():
                print(f"✅ {name}测试通过")
                passed += 1
            else:
                print(f"❌ {name}测试失败")
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")
    
    print(f"\n=== 测试结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 所有基础功能测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，需要检查")
        return False

if __name__ == "__main__":
    run_all_tests()