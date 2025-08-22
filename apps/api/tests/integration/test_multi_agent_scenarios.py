"""多智能体协作场景测试"""

import pytest
import asyncio
from typing import List, Dict, Any
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json


class MockAgent:
    """模拟智能体"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.messages = []
        self.state = "idle"
    
    async def receive_message(self, message: Dict[str, Any]):
        """接收消息"""
        self.messages.append(message)
        self.state = "processing"
        
        # 模拟处理
        await asyncio.sleep(0.01)
        
        response = {
            "from": self.name,
            "role": self.role,
            "content": f"{self.role} response to: {message.get('content')}",
            "timestamp": time.time()
        }
        
        self.state = "idle"
        return response
    
    async def collaborate(self, other_agent: 'MockAgent', task: str):
        """与其他智能体协作"""
        # 发送任务给其他智能体
        message = {
            "from": self.name,
            "to": other_agent.name,
            "content": task,
            "type": "collaboration_request"
        }
        
        response = await other_agent.receive_message(message)
        self.messages.append(response)
        
        return {
            "task": task,
            "collaborator": other_agent.name,
            "result": response["content"]
        }


class TestMultiAgentScenarios:
    """多智能体场景测试"""
    
    @pytest.fixture
    def create_agent_team(self):
        """创建智能体团队"""
        def _create_team(roles: List[str]) -> List[MockAgent]:
            agents = []
            for i, role in enumerate(roles):
                agent = MockAgent(f"Agent_{i}_{role}", role)
                agents.append(agent)
            return agents
        return _create_team
    
    @pytest.mark.asyncio
    async def test_sequential_collaboration(self, create_agent_team):
        """测试顺序协作场景"""
        # 创建智能体链
        agents = create_agent_team(["Analyzer", "Designer", "Implementer", "Reviewer"])
        
        # 模拟顺序处理流程
        task = "Create a new feature"
        current_result = task
        
        for i in range(len(agents) - 1):
            message = {
                "from": f"Coordinator",
                "content": current_result,
                "step": i + 1
            }
            
            response = await agents[i].receive_message(message)
            current_result = response["content"]
            
            # 传递给下一个智能体
            if i < len(agents) - 1:
                await agents[i + 1].receive_message({
                    "from": agents[i].name,
                    "content": current_result,
                    "previous_step": response
                })
        
        # 验证顺序处理
        for i, agent in enumerate(agents):
            assert len(agent.messages) > 0
            if i > 0:
                # 验证收到了前一个智能体的消息
                assert any(
                    msg.get("from") == agents[i-1].name 
                    for msg in agent.messages
                )
    
    @pytest.mark.asyncio
    async def test_parallel_task_distribution(self, create_agent_team):
        """测试并行任务分发场景"""
        # 创建工作智能体
        workers = create_agent_team(["Worker"] * 5)
        
        # 创建任务列表
        tasks = [f"Task_{i}" for i in range(20)]
        
        # 并行分发任务
        async def assign_task(worker: MockAgent, task: str):
            return await worker.receive_message({
                "from": "TaskDistributor",
                "content": task,
                "type": "task_assignment"
            })
        
        # 使用轮询分配任务
        results = []
        for i, task in enumerate(tasks):
            worker = workers[i % len(workers)]
            result = await assign_task(worker, task)
            results.append(result)
        
        # 验证任务分配
        assert len(results) == 20
        
        # 验证负载均衡
        for worker in workers:
            assert len(worker.messages) == 4  # 20个任务，5个工作者，每个应该处理4个
    
    @pytest.mark.asyncio
    async def test_hierarchical_delegation(self, create_agent_team):
        """测试层级委派场景"""
        # 创建层级结构
        manager = MockAgent("Manager", "Manager")
        team_leads = create_agent_team(["TeamLead"] * 2)
        developers = create_agent_team(["Developer"] * 4)
        
        # 经理分配给团队领导
        main_task = "Develop new system module"
        
        subtasks = []
        for i, lead in enumerate(team_leads):
            response = await lead.receive_message({
                "from": manager.name,
                "content": f"Subtask {i+1} of {main_task}",
                "type": "delegation"
            })
            subtasks.append(response)
        
        # 团队领导分配给开发者
        dev_tasks = []
        for i, lead in enumerate(team_leads):
            # 每个领导管理2个开发者
            lead_devs = developers[i*2:(i+1)*2]
            
            for j, dev in enumerate(lead_devs):
                response = await dev.receive_message({
                    "from": lead.name,
                    "content": f"Development task {j+1} from {lead.name}",
                    "type": "assignment"
                })
                dev_tasks.append(response)
        
        # 验证层级委派
        assert len(subtasks) == 2
        assert len(dev_tasks) == 4
        
        # 验证消息链
        for dev in developers:
            assert len(dev.messages) > 0
            # 每个开发者应该收到来自团队领导的消息
            assert any("TeamLead" in msg.get("from", "") for msg in dev.messages)
    
    @pytest.mark.asyncio
    async def test_consensus_decision_making(self, create_agent_team):
        """测试共识决策场景"""
        # 创建决策团队
        voters = create_agent_team(["Expert"] * 5)
        
        # 提出决策问题
        proposal = "Should we adopt new technology X?"
        
        votes = []
        for voter in voters:
            # 模拟投票（随机决定）
            import random
            vote_value = random.choice(["approve", "reject", "abstain"])
            
            response = await voter.receive_message({
                "from": "Facilitator",
                "content": proposal,
                "type": "vote_request",
                "vote": vote_value  # 模拟投票
            })
            
            votes.append({
                "voter": voter.name,
                "vote": vote_value,
                "reasoning": response["content"]
            })
        
        # 计算共识
        approve_count = sum(1 for v in votes if v["vote"] == "approve")
        reject_count = sum(1 for v in votes if v["vote"] == "reject")
        abstain_count = sum(1 for v in votes if v["vote"] == "abstain")
        
        consensus = {
            "proposal": proposal,
            "total_votes": len(votes),
            "approve": approve_count,
            "reject": reject_count,
            "abstain": abstain_count,
            "decision": "approved" if approve_count > len(votes) / 2 else "rejected"
        }
        
        # 验证投票过程
        assert len(votes) == 5
        assert consensus["total_votes"] == 5
        assert approve_count + reject_count + abstain_count == 5
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing_scenario(self, create_agent_team):
        """测试知识共享场景"""
        # 创建专家智能体
        experts = create_agent_team(["DataExpert", "DomainExpert", "TechExpert"])
        learner = MockAgent("Learner", "Learner")
        
        # 知识库
        knowledge_base = {}
        
        # 专家分享知识
        for expert in experts:
            knowledge = {
                "domain": expert.role,
                "facts": [f"Fact_{i} from {expert.name}" for i in range(3)],
                "insights": f"Key insight from {expert.role}"
            }
            
            # 存储到知识库
            knowledge_base[expert.name] = knowledge
            
            # 分享给学习者
            await learner.receive_message({
                "from": expert.name,
                "content": json.dumps(knowledge),
                "type": "knowledge_transfer"
            })
        
        # 验证知识共享
        assert len(knowledge_base) == 3
        assert len(learner.messages) == 3
        
        # 验证学习者接收到所有专家的知识
        for expert in experts:
            assert any(
                expert.name == msg.get("from") 
                for msg in learner.messages
            )
    
    @pytest.mark.asyncio
    async def test_competitive_bidding_scenario(self, create_agent_team):
        """测试竞争投标场景"""
        # 创建投标者
        bidders = create_agent_team(["Bidder"] * 4)
        
        # 任务详情
        task = {
            "id": "task_001",
            "description": "Complex computation task",
            "max_time": 10,
            "max_cost": 100
        }
        
        # 收集投标
        bids = []
        for i, bidder in enumerate(bidders):
            # 模拟投标
            bid = {
                "bidder": bidder.name,
                "time_estimate": 5 + i,  # 不同的时间估计
                "cost_estimate": 50 + i * 10,  # 不同的成本估计
                "confidence": 0.8 - i * 0.1  # 不同的置信度
            }
            
            response = await bidder.receive_message({
                "from": "AuctionManager",
                "content": json.dumps(task),
                "type": "bid_request",
                "bid": bid
            })
            
            bids.append(bid)
        
        # 选择最佳投标
        best_bid = min(bids, key=lambda x: x["cost_estimate"] / x["confidence"])
        
        # 通知获胜者
        winner = next(b for b in bidders if b.name == best_bid["bidder"])
        await winner.receive_message({
            "from": "AuctionManager",
            "content": "You won the bid!",
            "type": "bid_result",
            "task": task
        })
        
        # 验证投标过程
        assert len(bids) == 4
        assert best_bid is not None
        assert any(msg.get("type") == "bid_result" for msg in winner.messages)
    
    @pytest.mark.asyncio
    async def test_crisis_response_scenario(self, create_agent_team):
        """测试危机响应场景"""
        # 创建应急团队
        crisis_team = create_agent_team([
            "Coordinator",
            "Analyst", 
            "Responder",
            "Monitor"
        ])
        
        # 模拟危机事件
        crisis_event = {
            "type": "system_failure",
            "severity": "high",
            "affected_components": ["Database", "API"],
            "timestamp": time.time()
        }
        
        # 危机响应流程
        response_plan = []
        
        # 1. 协调员接收警报
        coordinator = crisis_team[0]
        await coordinator.receive_message({
            "from": "MonitoringSystem",
            "content": json.dumps(crisis_event),
            "type": "alert"
        })
        
        # 2. 分析员评估情况
        analyst = crisis_team[1]
        analysis = await analyst.receive_message({
            "from": coordinator.name,
            "content": "Analyze crisis situation",
            "crisis_data": crisis_event
        })
        response_plan.append({"step": "analysis", "result": analysis})
        
        # 3. 响应者执行修复
        responder = crisis_team[2]
        response = await responder.receive_message({
            "from": coordinator.name,
            "content": "Execute recovery plan",
            "analysis": analysis
        })
        response_plan.append({"step": "response", "result": response})
        
        # 4. 监控员验证恢复
        monitor = crisis_team[3]
        verification = await monitor.receive_message({
            "from": coordinator.name,
            "content": "Verify system recovery",
            "response": response
        })
        response_plan.append({"step": "verification", "result": verification})
        
        # 验证危机响应
        assert len(response_plan) == 3
        assert all(step["result"] is not None for step in response_plan)
        
        # 验证团队协作
        for agent in crisis_team:
            assert len(agent.messages) > 0
    
    @pytest.mark.asyncio
    async def test_negotiation_scenario(self, create_agent_team):
        """测试协商场景"""
        # 创建协商双方
        buyer = MockAgent("Buyer", "Buyer")
        seller = MockAgent("Seller", "Seller")
        mediator = MockAgent("Mediator", "Mediator")
        
        # 初始条件
        initial_offer = {"price": 1000, "quantity": 10, "delivery": 7}
        target_deal = {"price": 800, "quantity": 10, "delivery": 5}
        
        negotiation_rounds = []
        current_offer = initial_offer.copy()
        
        # 协商回合
        for round_num in range(5):
            # 买方提出条件
            buyer_response = await buyer.receive_message({
                "from": mediator.name,
                "content": f"Current offer: {current_offer}",
                "type": "negotiation",
                "round": round_num
            })
            
            # 卖方响应
            seller_response = await seller.receive_message({
                "from": mediator.name,
                "content": f"Buyer's response: {buyer_response['content']}",
                "type": "negotiation",
                "round": round_num
            })
            
            # 更新报价（模拟让步）
            current_offer["price"] -= 50
            current_offer["delivery"] -= 1
            
            negotiation_rounds.append({
                "round": round_num,
                "offer": current_offer.copy(),
                "buyer": buyer_response,
                "seller": seller_response
            })
            
            # 检查是否达成协议
            if current_offer["price"] <= target_deal["price"]:
                break
        
        # 验证协商过程
        assert len(negotiation_rounds) > 0
        assert negotiation_rounds[-1]["offer"]["price"] <= initial_offer["price"]
        
        # 验证参与方都参与了协商
        assert len(buyer.messages) > 0
        assert len(seller.messages) > 0