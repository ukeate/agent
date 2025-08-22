"""Few-shot CoT推理策略实现"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.ai.reasoning.cot_engine import BaseCoTEngine
from models.schemas.reasoning import ThoughtStepType
from src.core.logging import logger


class FewShotCoTEngine(BaseCoTEngine):
    """Few-shot链式思考推理引擎"""

    async def generate_prompt(self, problem: str, context: Optional[str] = None, **kwargs) -> str:
        """生成Few-shot CoT提示词"""
        history = kwargs.get('history', '')
        examples = kwargs.get('examples', [])
        
        prompt_parts = [
            "以下是一些链式思考推理的示例:",
            ""
        ]
        
        # 添加示例
        if examples:
            for i, example in enumerate(examples[:3], 1):  # 最多3个示例
                prompt_parts.append(f"示例 {i}:")
                prompt_parts.append(f"问题: {example.get('problem', '')}")
                
                # 如果示例包含步骤
                if 'steps' in example:
                    for step in example['steps']:
                        prompt_parts.append(
                            f"  步骤{step.get('number', '')}: {step.get('content', '')}"
                        )
                        if 'reasoning' in step:
                            prompt_parts.append(f"    推理: {step['reasoning']}")
                
                if 'answer' in example:
                    prompt_parts.append(f"答案: {example['answer']}")
                prompt_parts.append("")
        else:
            # 没有提供示例时，使用默认示例
            prompt_parts.extend(self._get_default_examples())
        
        prompt_parts.extend([
            "现在，请用同样的方式解决下面的问题:",
            "",
            f"问题: {problem}"
        ])
        
        if context:
            prompt_parts.append(f"背景信息: {context}")
        
        if history:
            prompt_parts.append(f"\n{history}")
        
        prompt_parts.extend([
            "",
            "请按照示例的格式进行推理:",
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
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "基于示例进行分析"
            
            # 解析置信度
            confidence = float(confidence_match.group(1)) if confidence_match else 0.8
            confidence = max(0.0, min(1.0, confidence))  # 确保在范围内
            
            return step_type, content, reasoning, confidence
            
        except Exception as e:
            logger.warning(f"解析响应失败，使用默认值: {e}")
            return ThoughtStepType.ANALYSIS, response[:200], "基于示例推理", 0.6

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

    def _get_default_examples(self) -> List[str]:
        """获取默认示例"""
        return [
            "示例 1:",
            "问题: 如果一个盒子里有 5 个红球和 3 个蓝球，随机抽取一个球是红球的概率是多少？",
            "  步骤1: 观察 - 盒子里总共有 5 + 3 = 8 个球",
            "    推理: 首先需要确定总数量",
            "  步骤2: 分析 - 红球有 5 个，总球数是 8 个",
            "    推理: 概率 = 有利结果数 / 总结果数",
            "  步骤3: 结论 - 概率是 5/8 = 0.625",
            "    推理: 直接计算得出结果",
            "答案: 0.625 或 62.5%",
            ""
        ]