"""推理失败恢复机制"""

import asyncio
from typing import List, Optional, Dict, Any
from enum import Enum
from src.models.schemas.reasoning import (
    ReasoningChain,
    ThoughtStep,
    ThoughtStepType,
    ReasoningBranch,
    ReasoningStrategy
)
from src.ai.reasoning.validation import CompositeValidator, calculate_chain_quality_score

logger = get_logger(__name__)

class RecoveryStrategy(Enum):
    """恢复策略类型"""
    BACKTRACK = "backtrack"  # 回溯
    BRANCH = "branch"  # 分支
    RESTART = "restart"  # 重启
    REFINE = "refine"  # 细化
    ALTERNATIVE = "alternative"  # 替代路径

class FailureDetector:
    """失败检测器"""
    
    def __init__(self):
        self.validator = CompositeValidator()
        self.failure_threshold = 0.3
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
    
    async def detect_failure(self, step: ThoughtStep, chain: ReasoningChain) -> Optional[Dict[str, Any]]:
        """检测推理失败"""
        # 验证当前步骤
        validation = await self.validator.validate(step, chain)
        
        if not validation.is_valid or validation.consistency_score < self.failure_threshold:
            self.consecutive_failures += 1
            
            failure_info = {
                "step_id": step.id,
                "step_number": step.step_number,
                "failure_type": self._classify_failure(validation),
                "severity": self._calculate_severity(validation),
                "issues": validation.issues,
                "suggestions": validation.suggestions,
                "consecutive_failures": self.consecutive_failures
            }
            
            logger.warning(
                f"检测到推理失败: 步骤{step.step_number}, "
                f"类型: {failure_info['failure_type']}, "
                f"严重程度: {failure_info['severity']}"
            )
            
            return failure_info
        else:
            # 重置连续失败计数
            self.consecutive_failures = 0
            return None
    
    def _classify_failure(self, validation) -> str:
        """分类失败类型"""
        if "逻辑不连贯" in validation.issues:
            return "logical_inconsistency"
        elif "置信度过低" in validation.issues:
            return "low_confidence"
        elif "矛盾" in ' '.join(validation.issues):
            return "contradiction"
        elif "循环推理" in validation.issues:
            return "circular_reasoning"
        else:
            return "general_failure"
    
    def _calculate_severity(self, validation) -> str:
        """计算失败严重程度"""
        if validation.consistency_score < 0.2:
            return "critical"
        elif validation.consistency_score < 0.5:
            return "high"
        elif validation.consistency_score < 0.7:
            return "medium"
        else:
            return "low"
    
    def should_abort(self) -> bool:
        """判断是否应该中止推理"""
        return self.consecutive_failures >= self.max_consecutive_failures

class BacktrackMechanism:
    """回溯机制"""
    
    def __init__(self):
        self.checkpoints = []  # 保存检查点
        self.max_backtrack_depth = 3
    
    def create_checkpoint(self, chain: ReasoningChain, step_number: int) -> None:
        """创建检查点"""
        checkpoint = {
            "step_number": step_number,
            "steps": chain.steps.copy(),
            "branches": chain.branches.copy(),
            "quality_score": calculate_chain_quality_score(chain)
        }
        self.checkpoints.append(checkpoint)
        
        # 保留最近的N个检查点
        if len(self.checkpoints) > self.max_backtrack_depth:
            self.checkpoints.pop(0)
    
    def find_backtrack_point(self, chain: ReasoningChain) -> Optional[int]:
        """找到回溯点"""
        if not self.checkpoints:
            return None
        
        # 找到质量最高的检查点
        best_checkpoint = max(self.checkpoints, key=lambda c: c["quality_score"])
        
        # 如果最佳检查点的质量比当前好
        current_score = calculate_chain_quality_score(chain)
        if best_checkpoint["quality_score"] > current_score:
            return best_checkpoint["step_number"]
        
        return None
    
    def backtrack_to(self, chain: ReasoningChain, step_number: int) -> bool:
        """回溯到指定步骤"""
        checkpoint = next(
            (c for c in self.checkpoints if c["step_number"] == step_number),
            None
        )
        
        if not checkpoint:
            return False
        
        # 恢复到检查点状态
        chain.steps = checkpoint["steps"].copy()
        chain.branches = checkpoint["branches"].copy()
        
        logger.info(f"回溯到步骤 {step_number}")
        return True

class AlternativePathGenerator:
    """替代路径生成器"""
    
    def __init__(self):
        self.alternative_strategies = {
            ThoughtStepType.OBSERVATION: [
                "从不同角度观察",
                "收集更多数据",
                "细化观察细节"
            ],
            ThoughtStepType.ANALYSIS: [
                "使用不同的分析方法",
                "分解问题",
                "寻找类比"
            ],
            ThoughtStepType.HYPOTHESIS: [
                "提出替代假设",
                "细化假设",
                "结合多个假设"
            ],
            ThoughtStepType.VALIDATION: [
                "使用不同的验证方法",
                "增加验证样本",
                "交叉验证"
            ]
        }
    
    async def generate_alternative(
        self,
        failed_step: ThoughtStep,
        chain: ReasoningChain,
        failure_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """生成替代路径"""
        strategies = self.alternative_strategies.get(
            failed_step.step_type,
            ["重新思考问题"]
        )
        
        # 根据失败类型选择策略
        if failure_info["failure_type"] == "low_confidence":
            strategy = strategies[1] if len(strategies) > 1 else strategies[0]
        elif failure_info["failure_type"] == "contradiction":
            strategy = strategies[0]
        else:
            strategy = strategies[2] if len(strategies) > 2 else strategies[0]
        
        alternative = {
            "strategy": strategy,
            "step_type": failed_step.step_type,
            "prompt_modifier": self._generate_prompt_modifier(strategy),
            "confidence_boost": 0.1,  # 替代路径的置信度加成
            "context": failure_info.get("suggestions", [])
        }
        
        return alternative
    
    def _generate_prompt_modifier(self, strategy: str) -> str:
        """生成提示词修饰符"""
        modifiers = {
            "从不同角度观察": "请从另一个角度重新审视这个问题",
            "收集更多数据": "请提供更详细的信息和证据",
            "使用不同的分析方法": "请尝试使用另一种分析方法",
            "分解问题": "请将问题分解为更小的子问题",
            "提出替代假设": "请考虑其他可能的解释"
        }
        
        return modifiers.get(strategy, "请重新思考")

class RecoveryManager:
    """恢复管理器"""
    
    def __init__(self):
        self.detector = FailureDetector()
        self.backtrack = BacktrackMechanism()
        self.path_generator = AlternativePathGenerator()
        self.recovery_history = []
    
    async def handle_failure(
        self,
        step: ThoughtStep,
        chain: ReasoningChain
    ) -> Optional[RecoveryStrategy]:
        """处理推理失败"""
        # 检测失败
        failure_info = await self.detector.detect_failure(step, chain)
        
        if not failure_info:
            # 没有失败，创建检查点
            self.backtrack.create_checkpoint(chain, step.step_number)
            return None
        
        # 记录失败
        self.recovery_history.append(failure_info)
        
        # 如果连续失败太多，中止
        if self.detector.should_abort():
            logger.error("连续失败过多，中止推理")
            return RecoveryStrategy.RESTART
        
        # 选择恢复策略
        strategy = await self._select_recovery_strategy(failure_info, chain)
        
        logger.info(f"选择恢复策略: {strategy.value}")
        return strategy
    
    async def _select_recovery_strategy(
        self,
        failure_info: Dict[str, Any],
        chain: ReasoningChain
    ) -> RecoveryStrategy:
        """选择恢复策略"""
        severity = failure_info["severity"]
        failure_type = failure_info["failure_type"]
        
        # 严重失败: 回溯或重启
        if severity == "critical":
            backtrack_point = self.backtrack.find_backtrack_point(chain)
            if backtrack_point:
                return RecoveryStrategy.BACKTRACK
            else:
                return RecoveryStrategy.RESTART
        
        # 逻辑问题: 回溯
        if failure_type in ["logical_inconsistency", "contradiction"]:
            return RecoveryStrategy.BACKTRACK
        
        # 循环推理: 分支
        if failure_type == "circular_reasoning":
            return RecoveryStrategy.BRANCH
        
        # 低置信度: 细化或替代路径
        if failure_type == "low_confidence":
            if len(chain.steps) < 3:
                return RecoveryStrategy.REFINE
            else:
                return RecoveryStrategy.ALTERNATIVE
        
        # 默认: 替代路径
        return RecoveryStrategy.ALTERNATIVE
    
    async def execute_recovery(
        self,
        strategy: RecoveryStrategy,
        chain: ReasoningChain,
        failed_step: Optional[ThoughtStep] = None
    ) -> bool:
        """执行恢复策略"""
        try:
            if strategy == RecoveryStrategy.BACKTRACK:
                backtrack_point = self.backtrack.find_backtrack_point(chain)
                if backtrack_point:
                    return self.backtrack.backtrack_to(chain, backtrack_point)
            
            elif strategy == RecoveryStrategy.BRANCH:
                if failed_step:
                    branch = chain.create_branch(
                        parent_step_id=failed_step.id,
                        reason=f"从步骤{failed_step.step_number}开始的替代路径"
                    )
                    chain.current_branch_id = branch.id
                    return True
            
            elif strategy == RecoveryStrategy.RESTART:
                # 清空步骤，保留问题和上下文
                chain.steps.clear()
                chain.branches.clear()
                chain.conclusion = None
                chain.confidence_score = None
                self.detector.consecutive_failures = 0
                return True
            
            elif strategy == RecoveryStrategy.REFINE:
                # 细化最后一个步骤
                if chain.steps:
                    last_step = chain.steps[-1]
                    last_step.confidence *= 0.8  # 降低置信度，促使重新思考
                    return True
            
            elif strategy == RecoveryStrategy.ALTERNATIVE:
                if failed_step:
                    # 生成替代路径
                    failure_info = self.recovery_history[-1] if self.recovery_history else {}
                    alternative = await self.path_generator.generate_alternative(
                        failed_step, chain, failure_info
                    )
                    if alternative:
                        # 将替代信息存储在metadata中
                        failed_step.metadata["alternative"] = alternative
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"执行恢复策略失败: {e}")
            return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """获取恢复统计信息"""
        if not self.recovery_history:
            return {
                "total_failures": 0,
                "recovery_attempts": 0,
                "success_rate": 0.0
            }
        
        failure_types = {}
        for failure in self.recovery_history:
            ft = failure["failure_type"]
            failure_types[ft] = failure_types.get(ft, 0) + 1
        
        return {
            "total_failures": len(self.recovery_history),
            "recovery_attempts": len(self.recovery_history),
            "failure_types": failure_types,
            "consecutive_failures": self.detector.consecutive_failures,
            "checkpoints_created": len(self.backtrack.checkpoints)
        }
from src.core.logging import get_logger
