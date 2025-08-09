#!/usr/bin/env python3
"""测试单个智能体响应生成"""

import asyncio
import time
import sys
import os

# 添加项目路径
sys.path.append('/Users/runout/awork/code/my_git/agent/apps/api/src')

from ai.autogen.agents import BaseAutoGenAgent, CodeExpertAgent, ArchitectAgent, DocExpertAgent
from ai.autogen.config import AGENT_CONFIGS, AgentRole

class AgentGenerationTest:
    """测试智能体响应生成"""
    
    def __init__(self):
        self.test_message = "请从你的专业角度简要分析一个用户认证系统的设计要点"
    
    async def test_single_agent_response(self, agent: BaseAutoGenAgent, agent_name: str):
        """测试单个智能体响应"""
        print(f"\n🤖 开始测试 {agent_name}...")
        start_time = time.time()
        
        try:
            print(f"  📝 发送消息: {self.test_message}")
            print(f"  ⏰ 开始时间: {time.strftime('%H:%M:%S')}")
            
            # 设置更短的超时时间进行测试
            response = await asyncio.wait_for(
                agent.generate_response(self.test_message),
                timeout=20.0
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"  ✅ 响应成功!")
            print(f"  ⏱️  耗时: {duration:.2f}秒")
            print(f"  📄 响应长度: {len(response)}字符")
            print(f"  📝 响应预览: {response[:200]}...")
            
            # 检查响应的专业性
            self._check_response_quality(agent_name, response)
            
            return True, response
            
        except asyncio.TimeoutError:
            end_time = time.time()
            duration = end_time - start_time
            print(f"  ❌ 响应超时! 耗时: {duration:.2f}秒")
            return False, "超时"
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"  ❌ 响应失败! 耗时: {duration:.2f}秒, 错误: {e}")
            return False, str(e)
    
    def _check_response_quality(self, agent_name: str, response: str):
        """检查响应质量"""
        response_lower = response.lower()
        
        if 'code_expert' in agent_name.lower() or '代码专家' in agent_name:
            keywords = ['代码', 'code', '实现', '安全', '加密', '验证', '数据库', '接口']
            if any(keyword in response_lower for keyword in keywords):
                print(f"  ✅ {agent_name}: 体现代码专业性")
            else:
                print(f"  ❓ {agent_name}: 专业性待确认")
                
        elif 'architect' in agent_name.lower() or '架构师' in agent_name:
            keywords = ['架构', '设计', '模块', '系统', '微服务', '分层', '组件']
            if any(keyword in response_lower for keyword in keywords):
                print(f"  ✅ {agent_name}: 体现架构专业性")
            else:
                print(f"  ❓ {agent_name}: 专业性待确认")
                
        elif 'doc_expert' in agent_name.lower() or '文档专家' in agent_name:
            keywords = ['文档', '说明', '规范', '手册', '指南', '流程', '步骤']
            if any(keyword in response_lower for keyword in keywords):
                print(f"  ✅ {agent_name}: 体现文档专业性")
            else:
                print(f"  ❓ {agent_name}: 专业性待确认")
    
    async def test_all_agents(self):
        """测试所有智能体"""
        print("🚀 开始测试所有智能体的响应生成能力...")
        print(f"📝 测试消息: {self.test_message}")
        print("="*80)
        
        results = {}
        
        # 测试代码专家
        try:
            code_expert = CodeExpertAgent()
            success, response = await self.test_single_agent_response(code_expert, "代码专家")
            results['code_expert'] = {'success': success, 'response': response}
        except Exception as e:
            print(f"❌ 代码专家初始化失败: {e}")
            results['code_expert'] = {'success': False, 'response': f"初始化失败: {e}"}
        
        # 测试架构师
        try:
            architect = ArchitectAgent()
            success, response = await self.test_single_agent_response(architect, "架构师")
            results['architect'] = {'success': success, 'response': response}
        except Exception as e:
            print(f"❌ 架构师初始化失败: {e}")
            results['architect'] = {'success': False, 'response': f"初始化失败: {e}"}
        
        # 测试文档专家
        try:
            doc_expert = DocExpertAgent()
            success, response = await self.test_single_agent_response(doc_expert, "文档专家")
            results['doc_expert'] = {'success': success, 'response': response}
        except Exception as e:
            print(f"❌ 文档专家初始化失败: {e}")
            results['doc_expert'] = {'success': False, 'response': f"初始化失败: {e}"}
        
        # 汇总结果
        print("\n" + "="*80)
        print("📊 测试结果汇总")
        print("="*80)
        
        success_count = 0
        total_count = len(results)
        
        for agent_name, result in results.items():
            status = "✅ 成功" if result['success'] else "❌ 失败"
            print(f"{agent_name}: {status}")
            if result['success']:
                success_count += 1
        
        print(f"\n🎯 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count == total_count:
            print("🎉 所有智能体响应正常!")
            print("💡 多智能体对话应该能正常工作")
        elif success_count > 0:
            print("⚠️  部分智能体响应正常，部分有问题")
            print("💡 需要检查失败的智能体配置")
        else:
            print("❌ 所有智能体都无法正常响应")
            print("💡 需要检查OpenAI API配置和网络连接")
        
        return results

async def main():
    """主测试函数"""
    tester = AgentGenerationTest()
    await tester.test_all_agents()

if __name__ == "__main__":
    asyncio.run(main())