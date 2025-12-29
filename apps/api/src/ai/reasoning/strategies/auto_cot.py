"""Auto-CoT自动链式思考生成策略"""

import re
from typing import Optional, Tuple, List, Dict, Any
from src.ai.reasoning.cot_engine import BaseCoTEngine
from src.models.schemas.reasoning import ThoughtStepType

logger = get_logger(__name__)

class AutoCoTEngine(BaseCoTEngine):
    """Auto-CoT自动链式思考推理引擎"""

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.problem_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """初始化问题模式库"""
        return {
            "数学": [
                "计算", "求解", "证明", "几何", "代数", "概率"
            ],
            "逻辑": [
                "推理", "分析", "判断", "归纳", "演绎", "假设"
            ],
            "编程": [
                "算法", "代码", "调试", "优化", "设计", "实现"
            ],
            "科学": [
                "实验", "观察", "假说", "验证", "结论", "原理"
            ]
        }

    async def generate_prompt(self, problem: str, context: Optional[str] = None, **kwargs) -> str:
        """生成Auto-CoT提示词"""
        history = kwargs.get('history', '')
        
        # 自动识别问题类型
        problem_type = self._identify_problem_type(problem)
        
        # 根据问题类型生成适合的推理策略
        strategy_steps = self._generate_strategy(problem_type)
        
        prompt_parts = [
            f"这是一个{problem_type}类型的问题。我将使用以下策略进行链式思考:",
            ""
        ]
        
        # 添加自动生成的策略步骤
        for i, step in enumerate(strategy_steps, 1):
            prompt_parts.append(f"{i}. {step}")
        
        prompt_parts.extend([
            "",
            f"问题: {problem}"
        ])
        
        if context:
            prompt_parts.append(f"背景信息: {context}")
        
        if history:
            prompt_parts.append(f"\n{history}")
        
        prompt_parts.extend([
            "",
            "现在按照上述策略进行推理:",
            "步骤类型: [OBSERVATION/ANALYSIS/HYPOTHESIS/VALIDATION/REFLECTION/CONCLUSION]",
            "内容: [这一步的主要发现或分析]",
            "推理: [为什么这样思考的详细解释]",
            "置信度: [0.0-1.0之间的数值]",
            "",
            "继续下一步推理:"
        ])
        
        return "\n".join(prompt_parts)

    async def parse_response(self, response: str) -> Tuple[ThoughtStepType, str, str, float]:
        """解析模型响应"""
        try:
            # 使用正则表达式提取各部分
            step_type_match = re.search(r'步骤类型[：:]\s*(\w+)', response, re.IGNORECASE)
            content_match = re.search(r'内容[：:]\s*(.+?)(?=推理|$)', response, re.DOTALL)
            reasoning_match = re.search(r'推理[：:]\s*(.+?)(?=置信度|$)', response, re.DOTALL)
            confidence_match = re.search(r'置信度[：:]\s*([0-9.]+)', response)
            
            # 解析步骤类型
            step_type_str = step_type_match.group(1).upper() if step_type_match else 'ANALYSIS'
            step_type = self._parse_step_type(step_type_str)
            
            # 解析内容和推理
            content = content_match.group(1).strip() if content_match else response[:200]
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "自动生成的推理"
            
            # 解析置信度
            confidence = float(confidence_match.group(1)) if confidence_match else 0.75
            confidence = max(0.0, min(1.0, confidence))
            
            return step_type, content, reasoning, confidence
            
        except Exception as e:
            logger.warning(f"解析响应失败，使用默认值: {e}")
            return ThoughtStepType.ANALYSIS, response[:200], "Auto-CoT推理", 0.65

    def _identify_problem_type(self, problem: str) -> str:
        """识别问题类型"""
        problem_lower = problem.lower()
        
        # 检查每个类型的关键词
        for problem_type, keywords in self.problem_patterns.items():
            for keyword in keywords:
                if keyword in problem_lower or keyword in problem:
                    return problem_type
        
        # 默认为通用问题
        return "通用"

    def _generate_strategy(self, problem_type: str) -> List[str]:
        """根据问题类型生成推理策略"""
        strategies = {
            "数学": [
                "识别问题中的已知条件和未知量",
                "确定适用的数学公式或定理",
                "逐步进行计算或推导",
                "验证结果的合理性",
                "得出最终答案"
            ],
            "逻辑": [
                "分析问题的前提和条件",
                "识别逻辑关系和推理规则",
                "构建推理链条",
                "检查推理的一致性",
                "得出逻辑结论"
            ],
            "编程": [
                "理解问题需求和约束",
                "设计算法思路",
                "分析时间和空间复杂度",
                "考虑边界条件和特殊情况",
                "提出实现方案"
            ],
            "科学": [
                "观察现象和收集数据",
                "提出假设",
                "设计验证方法",
                "分析实验结果",
                "得出科学结论"
            ],
            "通用": [
                "理解问题的核心内容",
                "分解问题为子任务",
                "分析每个子任务",
                "综合各部分结果",
                "形成最终答案"
            ]
        }
        
        return strategies.get(problem_type, strategies["通用"])

    def _parse_step_type(self, type_str: str) -> ThoughtStepType:
        """解析步骤类型字符串"""
        type_map = {
            'OBSERVATION': ThoughtStepType.OBSERVATION,
            'ANALYSIS': ThoughtStepType.ANALYSIS,
            'HYPOTHESIS': ThoughtStepType.HYPOTHESIS,
            'VALIDATION': ThoughtStepType.VALIDATION,
            'REFLECTION': ThoughtStepType.REFLECTION,
            'CONCLUSION': ThoughtStepType.CONCLUSION,
        }
        
        if type_str in type_map:
            return type_map[type_str]
        
        for key, value in type_map.items():
            if key in type_str or type_str in key:
                return value
        
        return ThoughtStepType.ANALYSIS
from src.core.logging import get_logger
