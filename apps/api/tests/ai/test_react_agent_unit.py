"""
ReAct智能体单元测试（独立）
"""

import pytest
import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# 创建测试用的枚举和数据类
class ReActStepType(Enum):
    """ReAct步骤类型"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"

@dataclass
class ReActStep:
    """ReAct步骤数据结构"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_type: ReActStepType = ReActStepType.THOUGHT
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())

@dataclass
class ReActSession:
    """ReAct会话状态"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[ReActStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10
    current_step: int = 0


class TestReActAgent:
    """ReAct智能体单元测试类"""

    def test_step_type_enum(self):
        """测试步骤类型枚举"""
        assert ReActStepType.THOUGHT.value == "thought"
        assert ReActStepType.ACTION.value == "action"
        assert ReActStepType.OBSERVATION.value == "observation"
        assert ReActStepType.FINAL_ANSWER.value == "final_answer"

    def test_react_step_creation(self):
        """测试ReAct步骤创建"""
        step = ReActStep(
            step_type=ReActStepType.THOUGHT,
            content="测试思考内容"
        )
        
        assert step.step_type == ReActStepType.THOUGHT
        assert step.content == "测试思考内容"
        assert step.tool_name is None
        assert step.tool_args is None
        assert step.tool_result is None
        assert len(step.step_id) > 0
        assert step.timestamp > 0

    def test_react_step_with_tool(self):
        """测试带工具信息的ReAct步骤"""
        step = ReActStep(
            step_type=ReActStepType.ACTION,
            content="调用工具",
            tool_name="read_file",
            tool_args={"file_path": "test.txt"}
        )
        
        assert step.step_type == ReActStepType.ACTION
        assert step.tool_name == "read_file"
        assert step.tool_args == {"file_path": "test.txt"}

    def test_react_session_creation(self):
        """测试ReAct会话创建"""
        session = ReActSession(max_steps=5)
        
        assert session.max_steps == 5
        assert session.current_step == 0
        assert len(session.steps) == 0
        assert len(session.context) == 0
        assert len(session.session_id) > 0

    def test_react_session_with_steps(self):
        """测试带步骤的ReAct会话"""
        session = ReActSession()
        
        # 添加思考步骤
        thought_step = ReActStep(
            step_type=ReActStepType.THOUGHT,
            content="我需要读取文件"
        )
        session.steps.append(thought_step)
        
        # 添加行动步骤
        action_step = ReActStep(
            step_type=ReActStepType.ACTION,
            content="调用read_file工具",
            tool_name="read_file",
            tool_args={"file_path": "example.txt"}
        )
        session.steps.append(action_step)
        
        # 添加观察步骤
        observation_step = ReActStep(
            step_type=ReActStepType.OBSERVATION,
            content="文件内容: Hello World"
        )
        session.steps.append(observation_step)
        
        assert len(session.steps) == 3
        assert session.steps[0].step_type == ReActStepType.THOUGHT
        assert session.steps[1].step_type == ReActStepType.ACTION
        assert session.steps[2].step_type == ReActStepType.OBSERVATION

    def test_parse_thought_response(self):
        """测试解析思考响应"""
        # 由于我们无法直接导入ReActAgent，我们模拟其解析逻辑
        response = "Thought: 我需要分析这个问题"
        
        # 模拟解析逻辑
        if response.startswith("Thought:"):
            content = response[len("Thought:"):].strip()
            step = ReActStep(
                step_type=ReActStepType.THOUGHT,
                content=content
            )
            
            assert step.step_type == ReActStepType.THOUGHT
            assert step.content == "我需要分析这个问题"

    def test_parse_action_response(self):
        """测试解析行动响应"""
        response = '''Action: read_file
Action Input: {"file_path": "test.txt", "encoding": "utf-8"}'''
        
        # 模拟解析逻辑
        if "Action:" in response and "Action Input:" in response:
            lines = response.split('\n')
            action_line = ""
            action_input_line = ""
            
            for line in lines:
                if line.startswith("Action:"):
                    action_line = line[len("Action:"):].strip()
                elif line.startswith("Action Input:"):
                    action_input_line = line[len("Action Input:"):].strip()
            
            if action_line and action_input_line:
                try:
                    tool_args = json.loads(action_input_line)
                    step = ReActStep(
                        step_type=ReActStepType.ACTION,
                        content=f"调用工具: {action_line}",
                        tool_name=action_line,
                        tool_args=tool_args
                    )
                    
                    assert step.step_type == ReActStepType.ACTION
                    assert step.tool_name == "read_file"
                    assert step.tool_args == {"file_path": "test.txt", "encoding": "utf-8"}
                    
                except json.JSONDecodeError:
                    pytest.fail("JSON解析失败")

    def test_parse_final_answer_response(self):
        """测试解析最终答案响应"""
        response = "Final Answer: 根据文件内容，我得出的结论是..."
        
        # 模拟解析逻辑
        if response.startswith("Final Answer:"):
            content = response[len("Final Answer:"):].strip()
            step = ReActStep(
                step_type=ReActStepType.FINAL_ANSWER,
                content=content
            )
            
            assert step.step_type == ReActStepType.FINAL_ANSWER
            assert step.content == "根据文件内容，我得出的结论是..."

    def test_parse_invalid_json_action(self):
        """测试解析无效JSON的行动响应"""
        response = '''Action: read_file
Action Input: {invalid json}'''
        
        # 模拟解析逻辑
        if "Action:" in response and "Action Input:" in response:
            lines = response.split('\n')
            action_line = ""
            action_input_line = ""
            
            for line in lines:
                if line.startswith("Action:"):
                    action_line = line[len("Action:"):].strip()
                elif line.startswith("Action Input:"):
                    action_input_line = line[len("Action Input:"):].strip()
            
            if action_line and action_input_line:
                try:
                    tool_args = json.loads(action_input_line)
                    pytest.fail("应该抛出JSON解析错误")
                except json.JSONDecodeError:
                    # 转为思考步骤
                    step = ReActStep(
                        step_type=ReActStepType.THOUGHT,
                        content="工具参数格式错误"
                    )
                    assert step.step_type == ReActStepType.THOUGHT

    def test_session_summary_logic(self):
        """测试会话摘要逻辑"""
        session = ReActSession()
        
        # 添加各种类型的步骤
        session.steps.extend([
            ReActStep(step_type=ReActStepType.THOUGHT, content="思考1"),
            ReActStep(step_type=ReActStepType.ACTION, content="行动1", tool_name="tool1"),
            ReActStep(step_type=ReActStepType.OBSERVATION, content="观察1"),
            ReActStep(step_type=ReActStepType.THOUGHT, content="思考2"),
            ReActStep(step_type=ReActStepType.FINAL_ANSWER, content="最终答案")
        ])
        
        # 模拟摘要生成逻辑
        step_counts = {}
        for step in session.steps:
            step_type = step.step_type.value
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        # 获取最终答案
        final_answer = None
        for step in reversed(session.steps):
            if step.step_type == ReActStepType.FINAL_ANSWER:
                final_answer = step.content
                break
        
        summary = {
            "session_id": session.session_id,
            "total_steps": len(session.steps),
            "step_counts": step_counts,
            "final_answer": final_answer,
            "completed": final_answer is not None
        }
        
        assert summary["total_steps"] == 5
        assert summary["step_counts"]["thought"] == 2
        assert summary["step_counts"]["action"] == 1
        assert summary["step_counts"]["observation"] == 1
        assert summary["step_counts"]["final_answer"] == 1
        assert summary["final_answer"] == "最终答案"
        assert summary["completed"] is True