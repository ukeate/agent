"""Zero-shot CoT推理策略实现"""

import re
from typing import Optional, Tuple

from src.ai.reasoning.cot_engine import BaseCoTEngine
from models.schemas.reasoning import ThoughtStepType
from src.core.logging import get_logger

logger = get_logger(__name__)


class ZeroShotCoTEngine(BaseCoTEngine):
    """Zero-shot链式思考推理引擎"""

    async def generate_prompt(self, problem: str, context: Optional[str] = None, **kwargs) -> str:
        """生成Zero-shot CoT提示词"""
        history = kwargs.get('history', '')
        
        prompt_parts = [
            "让我们一步一步地思考这个问题。",
            "",
            f"问题: {problem}"
        ]
        
        if context:
            prompt_parts.append(f"\n背景信息: {context}")
        
        if history:
            prompt_parts.append(f"\n{history}")
        
        prompt_parts.extend([
            "",
            "请按照以下格式进行推理:",
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
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "基于问题进行分析"
            
            # 解析置信度
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            confidence = max(0.0, min(1.0, confidence))  # 确保在范围内
            
            return step_type, content, reasoning, confidence
            
        except Exception as e:
            logger.warning(f"解析响应失败，使用默认值: {e}")
            # 返回默认值
            return ThoughtStepType.ANALYSIS, response[:200], "自动推理", 0.5

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
        
        # 尝试精确匹配
        if type_str in type_map:
            return type_map[type_str]
        
        # 尝试模糊匹配
        for key, value in type_map.items():
            if key in type_str or type_str in key:
                return value
        
        # 默认返回分析类型
        return ThoughtStepType.ANALYSIS