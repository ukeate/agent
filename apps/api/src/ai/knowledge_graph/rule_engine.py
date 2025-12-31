"""
规则推理引擎 - 基于SWRL语义Web规则语言的推理系统

实现功能:
- SWRL规则解析和验证
- 前向链式推理算法
- 规则冲突检测和解决
- 推理结果缓存和优化
- 规则执行统计和监控

技术栈:
- SWRL规则语法支持
- 图数据库集成
- 异步推理处理
- 置信度计算
"""

import re
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.core.logging import get_logger
logger = get_logger(__name__)

class RuleStatus(str, Enum):
    """规则状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"

class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略"""
    PRIORITY_BASED = "priority"
    CONFIDENCE_BASED = "confidence"
    RECENT_FIRST = "recent"
    MERGE = "merge"

@dataclass
class RuleCondition:
    """规则条件"""
    subject: str
    predicate: str
    object: str
    negated: bool = False
    
    def __str__(self) -> str:
        neg_prefix = "NOT " if self.negated else ""
        return f"{neg_prefix}{self.subject}({self.predicate}, {self.object})"

@dataclass
class RuleConclusion:
    """规则结论"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.subject}({self.predicate}, {self.object}) [conf: {self.confidence}]"

@dataclass
class ReasoningRule:
    """推理规则数据结构"""
    id: str
    name: str
    rule_text: str
    conditions: List[RuleCondition] = field(default_factory=list)
    conclusions: List[RuleConclusion] = field(default_factory=list)
    confidence: float = 1.0
    priority: int = 1
    status: RuleStatus = RuleStatus.ACTIVE
    created_by: str = "system"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    execution_count: int = 0
    success_count: int = 0
    last_executed: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

@dataclass
class InferenceResult:
    """推理结果"""
    fact: str
    confidence: float
    source_rules: List[str]
    derivation_path: List[str]
    timestamp: datetime = field(default_factory=utc_now)

class SWRLParser:
    """SWRL规则解析器"""
    
    def __init__(self):
        # SWRL语法模式
        self.atom_pattern = re.compile(r'(\w+)\s*\(\s*([^,\)]+)\s*,\s*([^,\)]+)\s*\)')
        self.rule_pattern = re.compile(r'(.+?)\s*->\s*(.+)')
        
    def parse_rule(self, rule_text: str) -> Tuple[List[RuleCondition], List[RuleConclusion]]:
        """解析SWRL规则文本"""
        try:
            # 移除注释和多余空格
            cleaned_rule = re.sub(r'//.*$', '', rule_text, flags=re.MULTILINE).strip()
            
            # 分离前提和结论
            rule_match = self.rule_pattern.match(cleaned_rule)
            if not rule_match:
                raise ValueError(f"Invalid rule syntax: {rule_text}")
            
            antecedent_text, consequent_text = rule_match.groups()
            
            # 解析前提条件
            conditions = self._parse_conditions(antecedent_text.strip())
            
            # 解析结论
            conclusions = self._parse_conclusions(consequent_text.strip())
            
            return conditions, conclusions
            
        except Exception as e:
            logger.error(f"Failed to parse rule '{rule_text}': {str(e)}")
            raise ValueError(f"Rule parsing error: {str(e)}")
    
    def _parse_conditions(self, conditions_text: str) -> List[RuleCondition]:
        """解析规则条件"""
        conditions = []
        
        # 分割多个条件（用逗号或AND分隔）
        condition_parts = re.split(r'\s*[,∧&]\s*', conditions_text)
        
        for part in condition_parts:
            part = part.strip()
            if not part:
                continue
                
            # 检查是否为否定条件
            negated = False
            if part.startswith('NOT ') or part.startswith('¬'):
                negated = True
                part = re.sub(r'^(NOT\s+|¬)', '', part).strip()
            
            # 解析原子条件
            atom_match = self.atom_pattern.match(part)
            if atom_match:
                predicate, subject, obj = atom_match.groups()
                conditions.append(RuleCondition(
                    subject=subject.strip(),
                    predicate=predicate.strip(), 
                    object=obj.strip(),
                    negated=negated
                ))
            else:
                logger.warning(f"Could not parse condition: {part}")
        
        return conditions
    
    def _parse_conclusions(self, conclusions_text: str) -> List[RuleConclusion]:
        """解析规则结论"""
        conclusions = []
        
        # 分割多个结论
        conclusion_parts = re.split(r'\s*[,∧&]\s*', conclusions_text)
        
        for part in conclusion_parts:
            part = part.strip()
            if not part:
                continue
            
            # 提取置信度（如果存在）
            confidence = 1.0
            conf_match = re.search(r'\[conf:\s*([\d.]+)\]', part)
            if conf_match:
                confidence = float(conf_match.group(1))
                part = re.sub(r'\[conf:\s*[\d.]+\]', '', part).strip()
            
            # 解析原子结论
            atom_match = self.atom_pattern.match(part)
            if atom_match:
                predicate, subject, obj = atom_match.groups()
                conclusions.append(RuleConclusion(
                    subject=subject.strip(),
                    predicate=predicate.strip(),
                    object=obj.strip(),
                    confidence=confidence
                ))
            else:
                logger.warning(f"Could not parse conclusion: {part}")
        
        return conclusions

class RuleEngine:
    """基于规则的推理引擎"""
    
    def __init__(self, graph_db=None, max_iterations: int = 10):
        self.graph_db = graph_db
        self.max_iterations = max_iterations
        self.rules: Dict[str, ReasoningRule] = {}
        self.parser = SWRLParser()
        self.inference_cache: Dict[str, List[InferenceResult]] = {}
        self.fact_store: Set[str] = set()
        self.conflict_resolution = ConflictResolutionStrategy.CONFIDENCE_BASED
        
        # 统计信息
        self.total_inferences = 0
        self.cache_hits = 0
        
    async def add_rule(self, 
                      rule_text: str, 
                      name: str = None,
                      confidence: float = 1.0,
                      priority: int = 1) -> ReasoningRule:
        """添加推理规则"""
        try:
            # 解析规则
            conditions, conclusions = self.parser.parse_rule(rule_text)
            
            # 验证规则
            self._validate_rule(conditions, conclusions)
            
            # 创建规则对象
            rule_id = str(uuid.uuid4())
            rule = ReasoningRule(
                id=rule_id,
                name=name or f"rule_{len(self.rules)}",
                rule_text=rule_text,
                conditions=conditions,
                conclusions=conclusions,
                confidence=confidence,
                priority=priority
            )
            
            self.rules[rule_id] = rule
            logger.info(f"Added rule '{rule.name}': {rule_text}")
            
            return rule
            
        except Exception as e:
            logger.error(f"Failed to add rule: {str(e)}")
            raise
    
    def _validate_rule(self, 
                      conditions: List[RuleCondition], 
                      conclusions: List[RuleConclusion]) -> None:
        """验证规则的有效性"""
        if not conditions:
            raise ValueError("Rule must have at least one condition")
        
        if not conclusions:
            raise ValueError("Rule must have at least one conclusion")
        
        # 检查变量一致性
        condition_vars = set()
        for cond in conditions:
            condition_vars.update([cond.subject, cond.object])
        
        conclusion_vars = set()
        for concl in conclusions:
            conclusion_vars.update([concl.subject, concl.object])
        
        # 结论中的变量必须在条件中出现过
        unbound_vars = conclusion_vars - condition_vars
        if unbound_vars:
            logger.warning(f"Unbound variables in conclusion: {unbound_vars}")
    
    async def forward_chaining(self, 
                             initial_facts: List[str], 
                             max_iterations: int = None) -> List[InferenceResult]:
        """前向链式推理"""
        max_iter = max_iterations or self.max_iterations
        
        # 初始化事实集合
        facts = set(initial_facts)
        inferred_results = []
        
        logger.info(f"Starting forward chaining with {len(facts)} initial facts")
        
        for iteration in range(max_iter):
            iteration_results = []
            new_facts_found = False
            
            # 按优先级和置信度排序规则
            sorted_rules = sorted(
                [rule for rule in self.rules.values() if rule.status == RuleStatus.ACTIVE],
                key=lambda r: (-r.priority, -r.confidence)
            )
            
            for rule in sorted_rules:
                try:
                    # 检查规则是否可以触发
                    bindings_list = await self._match_rule_conditions(rule, facts)
                    
                    if bindings_list:
                        rule.execution_count += 1
                        rule.last_executed = utc_now()
                        
                        for bindings in bindings_list:
                            # 应用规则得出结论
                            new_facts = await self._apply_rule_conclusions(rule, bindings)
                            
                            for new_fact in new_facts:
                                if new_fact.fact not in facts:
                                    facts.add(new_fact.fact)
                                    iteration_results.append(new_fact)
                                    new_facts_found = True
                                    rule.success_count += 1
                                    
                except Exception as e:
                    logger.error(f"Error executing rule '{rule.name}': {str(e)}")
                    continue
            
            inferred_results.extend(iteration_results)
            
            # 如果没有新事实推导出来，停止推理
            if not new_facts_found:
                logger.info(f"Forward chaining converged after {iteration + 1} iterations")
                break
                
            logger.debug(f"Iteration {iteration + 1}: Found {len(iteration_results)} new facts")
        
        self.total_inferences += len(inferred_results)
        logger.info(f"Forward chaining completed: {len(inferred_results)} new facts inferred")
        
        return inferred_results
    
    async def _match_rule_conditions(self, 
                                   rule: ReasoningRule, 
                                   facts: Set[str]) -> List[Dict[str, str]]:
        """匹配规则条件并返回变量绑定"""
        if not rule.conditions:
            return []
        
        # 从图数据库获取相关事实（如果可用）
        if self.graph_db:
            graph_facts = await self._query_graph_facts(rule.conditions)
            facts = facts.union(graph_facts)
        
        # 使用回溯算法找到所有可能的变量绑定
        return self._find_variable_bindings(rule.conditions, list(facts))
    
    def _find_variable_bindings(self, 
                               conditions: List[RuleCondition], 
                               facts: List[str]) -> List[Dict[str, str]]:
        """查找满足条件的变量绑定"""
        if not conditions:
            return [{}]
        
        valid_bindings = []
        
        def backtrack(cond_index: int, current_bindings: Dict[str, str]):
            if cond_index >= len(conditions):
                valid_bindings.append(current_bindings.copy())
                return
            
            condition = conditions[cond_index]
            
            for fact in facts:
                binding = self._try_match_fact(condition, fact, current_bindings)
                if binding is not None:
                    # 检查绑定一致性
                    if self._is_consistent_binding(binding, current_bindings):
                        merged_bindings = {**current_bindings, **binding}
                        backtrack(cond_index + 1, merged_bindings)
        
        backtrack(0, {})
        return valid_bindings
    
    def _try_match_fact(self, 
                       condition: RuleCondition, 
                       fact: str,
                       existing_bindings: Dict[str, str]) -> Optional[Dict[str, str]]:
        """尝试将条件与事实匹配"""
        # 简化的事实格式: "predicate(subject, object)"
        fact_pattern = re.match(r'(\w+)\s*\(\s*([^,\)]+)\s*,\s*([^,\)]+)\s*\)', fact)
        if not fact_pattern:
            return None
        
        fact_pred, fact_subj, fact_obj = fact_pattern.groups()
        fact_pred, fact_subj, fact_obj = fact_pred.strip(), fact_subj.strip(), fact_obj.strip()
        
        # 检查谓词是否匹配
        if condition.predicate != fact_pred:
            return None
        
        # 检查否定条件
        if condition.negated:
            return None  # 否定条件的处理需要完整的闭世界假设
        
        binding = {}
        
        # 匹配主语
        if condition.subject.startswith('?'):  # 变量
            var_name = condition.subject
            if var_name in existing_bindings:
                if existing_bindings[var_name] != fact_subj:
                    return None  # 绑定冲突
            else:
                binding[var_name] = fact_subj
        elif condition.subject != fact_subj:
            return None  # 常量不匹配
        
        # 匹配宾语
        if condition.object.startswith('?'):  # 变量
            var_name = condition.object
            if var_name in existing_bindings:
                if existing_bindings[var_name] != fact_obj:
                    return None  # 绑定冲突
            else:
                binding[var_name] = fact_obj
        elif condition.object != fact_obj:
            return None  # 常量不匹配
        
        return binding
    
    def _is_consistent_binding(self, 
                              new_binding: Dict[str, str], 
                              existing_binding: Dict[str, str]) -> bool:
        """检查新绑定与现有绑定是否一致"""
        for var, value in new_binding.items():
            if var in existing_binding and existing_binding[var] != value:
                return False
        return True
    
    async def _apply_rule_conclusions(self, 
                                    rule: ReasoningRule, 
                                    bindings: Dict[str, str]) -> List[InferenceResult]:
        """应用规则结论生成新事实"""
        results = []
        
        for conclusion in rule.conclusions:
            # 替换变量绑定
            subject = self._substitute_variables(conclusion.subject, bindings)
            predicate = conclusion.predicate
            obj = self._substitute_variables(conclusion.object, bindings)
            
            # 构造新事实
            new_fact = f"{predicate}({subject}, {obj})"
            
            # 计算置信度
            confidence = rule.confidence * conclusion.confidence
            
            result = InferenceResult(
                fact=new_fact,
                confidence=confidence,
                source_rules=[rule.id],
                derivation_path=[rule.name],
                timestamp=utc_now()
            )
            
            results.append(result)
            logger.debug(f"Inferred fact: {new_fact} [confidence: {confidence:.3f}]")
        
        return results
    
    def _substitute_variables(self, term: str, bindings: Dict[str, str]) -> str:
        """替换项中的变量"""
        if term.startswith('?') and term in bindings:
            return bindings[term]
        return term
    
    async def _query_graph_facts(self, conditions: List[RuleCondition]) -> Set[str]:
        """从图数据库查询相关事实"""
        if not self.graph_db:
            return set()
        
        facts = set()
        
        try:
            for condition in conditions:
                if not condition.negated:
                    # 构造图查询
                    query = f"""
                    MATCH (s)-[:{condition.predicate}]->(o)
                    RETURN s.name as subject, o.name as object
                    """
                    
                    results = await self.graph_db.execute_query(query)
                    
                    for record in results:
                        fact = f"{condition.predicate}({record['subject']}, {record['object']})"
                        facts.add(fact)
                        
        except Exception as e:
            logger.error(f"Error querying graph database: {str(e)}")
        
        return facts

    async def test_rule(self, rule_id: str, facts: List[str]) -> List[InferenceResult]:
        """测试单条规则"""
        rule = self.get_rule_by_id(rule_id)
        if not rule:
            raise ValueError("规则不存在")
        bindings_list = await self._match_rule_conditions(rule, set(facts))
        results: List[InferenceResult] = []
        for bindings in bindings_list:
            results.extend(await self._apply_rule_conclusions(rule, bindings))
        return results
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则引擎统计信息"""
        active_rules = len([r for r in self.rules.values() if r.status == RuleStatus.ACTIVE])
        total_executions = sum(r.execution_count for r in self.rules.values())
        total_successes = sum(r.success_count for r in self.rules.values())
        
        return {
            "total_rules": len(self.rules),
            "active_rules": active_rules,
            "total_executions": total_executions,
            "total_successes": total_successes,
            "success_rate": total_successes / total_executions if total_executions > 0 else 0.0,
            "total_inferences": self.total_inferences,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.total_inferences if self.total_inferences > 0 else 0.0
        }
    
    def clear_cache(self):
        """清除推理缓存"""
        self.inference_cache.clear()
        logger.info("Rule engine cache cleared")
    
    async def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            rule.status = RuleStatus.DEPRECATED
            logger.info(f"Rule '{rule.name}' marked as deprecated")
            return True
        return False
    
    def get_rule_by_id(self, rule_id: str) -> Optional[ReasoningRule]:
        """根据ID获取规则"""
        return self.rules.get(rule_id)
    
    def list_rules(self, status: RuleStatus = None) -> List[ReasoningRule]:
        """列出规则"""
        if status is None:
            return list(self.rules.values())
        return [rule for rule in self.rules.values() if rule.status == status]
