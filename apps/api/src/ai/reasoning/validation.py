"""推理质量控制和验证模块"""

import re
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from src.models.schemas.reasoning import (
    ThoughtStep,
    ThoughtStepType,
    ReasoningChain,
    ReasoningValidation
)

class BaseValidator(ABC):
    """验证器基类"""

    @abstractmethod
    async def validate(self, step: ThoughtStep, chain: ReasoningChain) -> ReasoningValidation:
        """验证推理步骤"""
        raise NotImplementedError

class ConsistencyValidator(BaseValidator):
    """一致性验证器"""
    
    async def validate(self, step: ThoughtStep, chain: ReasoningChain) -> ReasoningValidation:
        """验证推理步骤的一致性"""
        issues = []
        suggestions = []
        consistency_score = 1.0
        
        # 检查与前续步骤的一致性
        if chain.steps:
            prev_steps = chain.steps[-3:]  # 检查最近3个步骤
            
            # 检查逻辑连贯性
            if not self._check_logical_flow(step, prev_steps):
                issues.append("推理逻辑不连贯")
                suggestions.append("重新梳理上一步的结论")
                consistency_score -= 0.3
            
            # 检查矛盾
            contradictions = self._find_contradictions(step, prev_steps)
            if contradictions:
                for contradiction in contradictions:
                    issues.append(f"与步骤{contradiction['step_num']}存在矛盾")
                    suggestions.append(f"解决与{contradiction['content']}的冲突")
                consistency_score -= 0.2 * len(contradictions)
        
        # 检查步骤类型合理性
        if not self._check_step_type_sequence(step, chain):
            issues.append("步骤类型序列不合理")
            suggestions.append("调整推理步骤的类型")
            consistency_score -= 0.1
        
        consistency_score = max(0.0, consistency_score)
        is_valid = consistency_score >= 0.5
        
        return ReasoningValidation(
            step_id=step.id,
            is_valid=is_valid,
            consistency_score=consistency_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _check_logical_flow(self, current_step: ThoughtStep, prev_steps: List[ThoughtStep]) -> bool:
        """检查逻辑流程"""
        if not prev_steps:
            return True
        
        # 简单检查: 当前步骤应该基于前面的步骤
        last_step = prev_steps[-1]
        
        # 如果前一步是结论，当前步骤不应该是观察或分析
        if last_step.step_type == ThoughtStepType.CONCLUSION:
            if current_step.step_type in [ThoughtStepType.OBSERVATION, ThoughtStepType.ANALYSIS]:
                return False
        
        return True
    
    def _find_contradictions(
        self, 
        current_step: ThoughtStep, 
        prev_steps: List[ThoughtStep]
    ) -> List[Dict[str, Any]]:
        """查找矛盾"""
        contradictions = []
        
        # 简单的矛盾检测: 检查相反的关键词
        negative_patterns = [
            (r'不是', r'是'),
            (r'不能', r'能'),
            (r'错误', r'正确'),
            (r'失败', r'成功')
        ]
        
        for prev_step in prev_steps:
            for neg_pattern, pos_pattern in negative_patterns:
                if (re.search(neg_pattern, current_step.content) and 
                    re.search(pos_pattern, prev_step.content)):
                    contradictions.append({
                        'step_num': prev_step.step_number,
                        'content': prev_step.content[:50]
                    })
                    break
        
        return contradictions
    
    def _check_step_type_sequence(self, step: ThoughtStep, chain: ReasoningChain) -> bool:
        """检查步骤类型序列的合理性"""
        if not chain.steps:
            # 第一步通常应该是观察或分析
            return step.step_type in [ThoughtStepType.OBSERVATION, ThoughtStepType.ANALYSIS]
        
        # 结论通常应该在最后
        if step.step_type == ThoughtStepType.CONCLUSION:
            # 至少有一些分析或验证
            analysis_count = sum(
                1 for s in chain.steps 
                if s.step_type in [ThoughtStepType.ANALYSIS, ThoughtStepType.VALIDATION]
            )
            return analysis_count > 0
        
        return True

class ConfidenceValidator(BaseValidator):
    """置信度验证器"""
    
    def __init__(self, min_confidence: float = 0.3, warning_threshold: float = 0.5):
        self.min_confidence = min_confidence
        self.warning_threshold = warning_threshold
    
    async def validate(self, step: ThoughtStep, chain: ReasoningChain) -> ReasoningValidation:
        """验证置信度"""
        issues = []
        suggestions = []
        
        # 检查置信度水平
        if step.confidence < self.min_confidence:
            issues.append(f"置信度过低 ({step.confidence:.2f})")
            suggestions.append("重新分析或收集更多信息")
        elif step.confidence < self.warning_threshold:
            issues.append(f"置信度偏低 ({step.confidence:.2f})")
            suggestions.append("考虑增加验证步骤")
        
        # 检查置信度趋势
        if chain.steps:
            trend = self._analyze_confidence_trend(step, chain.steps[-3:])
            if trend == "declining":
                issues.append("置信度持续下降")
                suggestions.append("可能需要重新评估推理路径")
        
        is_valid = step.confidence >= self.min_confidence
        
        return ReasoningValidation(
            step_id=step.id,
            is_valid=is_valid,
            consistency_score=step.confidence,
            issues=issues,
            suggestions=suggestions
        )
    
    def _analyze_confidence_trend(self, current_step: ThoughtStep, prev_steps: List[ThoughtStep]) -> str:
        """分析置信度趋势"""
        if len(prev_steps) < 2:
            return "stable"
        
        confidences = [s.confidence for s in prev_steps] + [current_step.confidence]
        
        # 计算趋势
        declining_count = sum(
            1 for i in range(1, len(confidences)) 
            if confidences[i] < confidences[i-1]
        )
        
        if declining_count >= len(confidences) - 1:
            return "declining"
        elif declining_count == 0:
            return "increasing"
        else:
            return "stable"

class SelfCheckValidator(BaseValidator):
    """自我检查验证器"""
    
    async def validate(self, step: ThoughtStep, chain: ReasoningChain) -> ReasoningValidation:
        """执行自我检查"""
        issues = []
        suggestions = []
        
        # 检查推理是否支持内容
        if not self._reasoning_supports_content(step):
            issues.append("推理与内容不匹配")
            suggestions.append("确保推理解释支持主要内容")
        
        # 检查是否有循环推理
        if self._has_circular_reasoning(step, chain):
            issues.append("存在循环推理")
            suggestions.append("提供独立的证据或推理")
        
        # 检查是否有未支持的假设
        if step.step_type == ThoughtStepType.HYPOTHESIS:
            if not self._hypothesis_has_basis(step, chain):
                issues.append("假设缺乏依据")
                suggestions.append("基于前面的观察和分析提出假设")
        
        consistency_score = 1.0 - (len(issues) * 0.2)
        consistency_score = max(0.0, consistency_score)
        is_valid = len(issues) == 0
        
        return ReasoningValidation(
            step_id=step.id,
            is_valid=is_valid,
            consistency_score=consistency_score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _reasoning_supports_content(self, step: ThoughtStep) -> bool:
        """检查推理是否支持内容"""
        # 简单检查: 推理和内容应该有一些共同关键词
        content_words = set(re.findall(r'\w+', step.content.lower()))
        reasoning_words = set(re.findall(r'\w+', step.reasoning.lower()))
        
        # 至少有10%的词汇重叠
        overlap = content_words & reasoning_words
        min_overlap = min(len(content_words), len(reasoning_words)) * 0.1
        
        return len(overlap) >= min_overlap
    
    def _has_circular_reasoning(self, step: ThoughtStep, chain: ReasoningChain) -> bool:
        """检查循环推理"""
        if not chain.steps:
            return False
        
        # 检查当前步骤是否重复之前的内容
        for prev_step in chain.steps[-5:]:
            similarity = self._calculate_similarity(step.content, prev_step.content)
            if similarity > 0.8:  # 高度相似
                return True
        
        return False
    
    def _hypothesis_has_basis(self, step: ThoughtStep, chain: ReasoningChain) -> bool:
        """检查假设是否有依据"""
        if not chain.steps:
            return False
        
        # 假设应该基于之前的观察或分析
        has_observation = any(
            s.step_type in [ThoughtStepType.OBSERVATION, ThoughtStepType.ANALYSIS]
            for s in chain.steps
        )
        
        return has_observation
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0

class CompositeValidator(BaseValidator):
    """组合验证器"""
    
    def __init__(self, validators: Optional[List[BaseValidator]] = None):
        self.validators = validators or [
            ConsistencyValidator(),
            ConfidenceValidator(),
            SelfCheckValidator()
        ]
    
    async def validate(self, step: ThoughtStep, chain: ReasoningChain) -> ReasoningValidation:
        """执行所有验证器"""
        all_issues = []
        all_suggestions = []
        total_score = 0.0
        valid_count = 0
        
        for validator in self.validators:
            validation = await validator.validate(step, chain)
            
            all_issues.extend(validation.issues)
            all_suggestions.extend(validation.suggestions)
            total_score += validation.consistency_score
            
            if validation.is_valid:
                valid_count += 1
        
        # 去重
        all_issues = list(set(all_issues))
        all_suggestions = list(set(all_suggestions))
        
        # 计算平均分
        avg_score = total_score / len(self.validators) if self.validators else 0.0
        
        # 大多数验证器通过才算有效
        is_valid = valid_count > len(self.validators) / 2
        
        return ReasoningValidation(
            step_id=step.id,
            is_valid=is_valid,
            consistency_score=avg_score,
            issues=all_issues,
            suggestions=all_suggestions
        )

def calculate_chain_quality_score(chain: ReasoningChain) -> float:
    """计算推理链的总体质量分数"""
    if not chain.steps:
        return 0.0
    
    # 基础分数: 平均置信度
    avg_confidence = sum(s.confidence for s in chain.steps) / len(chain.steps)
    
    # 完整性分数
    has_observation = any(s.step_type == ThoughtStepType.OBSERVATION for s in chain.steps)
    has_analysis = any(s.step_type == ThoughtStepType.ANALYSIS for s in chain.steps)
    has_conclusion = any(s.step_type == ThoughtStepType.CONCLUSION for s in chain.steps)
    
    completeness_score = sum([
        0.2 if has_observation else 0,
        0.3 if has_analysis else 0,
        0.5 if has_conclusion else 0
    ])
    
    # 最终分数
    quality_score = avg_confidence * 0.6 + completeness_score * 0.4
    
    return min(1.0, quality_score)
