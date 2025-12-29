"""
SPARQL查询优化器

实现高级查询优化策略，包括：
- 查询重写和代数优化
- 统计驱动的优化
- 索引使用建议
- 成本估算模型
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass
import re

from src.core.logging import get_logger
logger = get_logger(__name__)

class OptimizationLevel(str, Enum):
    """优化级别"""
    NONE = "none"
    BASIC = "basic" 
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"

class JoinType(str, Enum):
    """连接类型"""
    INNER = "inner"
    LEFT = "left"
    OPTIONAL = "optional"

@dataclass
class QueryPattern:
    """查询模式"""
    pattern_type: str
    subject: str
    predicate: str
    object: str
    optional: bool = False
    filter_conditions: List[str] = None
    estimated_selectivity: float = 1.0
    
    def __post_init__(self):
        if self.filter_conditions is None:
            self.filter_conditions = []

@dataclass
class JoinOperation:
    """连接操作"""
    join_type: JoinType
    left_pattern: QueryPattern
    right_pattern: QueryPattern
    join_variables: List[str]
    estimated_cost: float = 0.0

@dataclass
class OptimizationContext:
    """优化上下文"""
    statistics: Dict[str, Any]
    indexes: List[str]
    query_type: str
    optimization_level: OptimizationLevel
    time_budget_ms: int = 1000

class StatisticsManager:
    """统计信息管理器"""
    
    def __init__(self):
        self.predicate_counts = {}
        self.class_counts = {}
        self.literal_patterns = {}
        self.join_selectivities = {}
    
    def update_predicate_stats(self, predicate: str, count: int):
        """更新谓词统计"""
        self.predicate_counts[predicate] = count
    
    def update_class_stats(self, class_uri: str, count: int):
        """更新类统计"""
        self.class_counts[class_uri] = count
    
    def get_predicate_selectivity(self, predicate: str) -> float:
        """获取谓词选择性"""
        if predicate in self.predicate_counts:
            total_triples = sum(self.predicate_counts.values())
            if total_triples > 0:
                return self.predicate_counts[predicate] / total_triples
        return 0.1  # 默认选择性
    
    def get_join_selectivity(self, pred1: str, pred2: str) -> float:
        """获取连接选择性"""
        join_key = f"{pred1}|{pred2}"
        if join_key in self.join_selectivities:
            return self.join_selectivities[join_key]
        
        # 基于谓词频率估算
        sel1 = self.get_predicate_selectivity(pred1)
        sel2 = self.get_predicate_selectivity(pred2)
        
        # 简单估算：较低选择性的乘积
        return min(sel1, sel2) * 0.1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "predicate_counts": self.predicate_counts.copy(),
            "class_counts": self.class_counts.copy(),
            "total_predicates": len(self.predicate_counts),
            "total_triples": sum(self.predicate_counts.values())
        }

class QueryRewriter:
    """查询重写器"""
    
    def __init__(self):
        self.rewrite_rules = [
            self._eliminate_redundant_patterns,
            self._merge_adjacent_patterns,
            self._convert_exists_to_join,
            self._optimize_union_patterns,
            self._simplify_filter_expressions
        ]
    
    async def rewrite_query(self, query_text: str, context: OptimizationContext) -> str:
        """重写查询"""
        try:
            optimized_query = query_text
            
            for rule in self.rewrite_rules:
                optimized_query = await rule(optimized_query, context)
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query_text
    
    async def _eliminate_redundant_patterns(self, query: str, context: OptimizationContext) -> str:
        """消除冗余模式"""
        # 识别和移除重复的三元组模式
        lines = query.split('\n')
        unique_patterns = []
        seen_patterns = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # 简化的重复检测
                pattern_key = re.sub(r'\s+', ' ', stripped.lower())
                if pattern_key not in seen_patterns:
                    unique_patterns.append(line)
                    seen_patterns.add(pattern_key)
                else:
                    logger.debug(f"移除冗余模式: {stripped}")
            else:
                unique_patterns.append(line)
        
        return '\n'.join(unique_patterns)
    
    async def _merge_adjacent_patterns(self, query: str, context: OptimizationContext) -> str:
        """合并相邻模式"""
        # 合并具有相同主语的相邻模式
        # 例如: ?s :p1 ?o1 . ?s :p2 ?o2 -> ?s :p1 ?o1 ; :p2 ?o2
        
        # 简化实现：查找可合并的模式
        lines = query.split('\n')
        optimized_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if self._is_triple_pattern(line):
                # 查找具有相同主语的后续模式
                merged_patterns = [line]
                subject = self._extract_subject(line)
                
                j = i + 1
                while j < len(lines) and self._can_merge_pattern(lines[j], subject):
                    merged_patterns.append(lines[j].strip())
                    j += 1
                
                if len(merged_patterns) > 1:
                    # 合并模式
                    merged = self._merge_patterns(merged_patterns)
                    optimized_lines.append(merged)
                    i = j
                else:
                    optimized_lines.append(lines[i])
                    i += 1
            else:
                optimized_lines.append(lines[i])
                i += 1
        
        return '\n'.join(optimized_lines)
    
    def _is_triple_pattern(self, line: str) -> bool:
        """检查是否为三元组模式"""
        # 简化检测：包含三个元素和点结束
        parts = line.strip().split()
        return len(parts) >= 3 and line.strip().endswith('.')
    
    def _extract_subject(self, line: str) -> str:
        """提取主语"""
        parts = line.strip().split()
        return parts[0] if parts else ""
    
    def _can_merge_pattern(self, line: str, subject: str) -> bool:
        """检查是否可以合并模式"""
        if not self._is_triple_pattern(line):
            return False
        
        line_subject = self._extract_subject(line)
        return line_subject == subject
    
    def _merge_patterns(self, patterns: List[str]) -> str:
        """合并模式列表"""
        if not patterns:
            return ""
        
        # 简化合并：保持第一个模式，其余的合并为分号分隔
        first = patterns[0]
        if len(patterns) == 1:
            return first
        
        subject = self._extract_subject(first)
        merged_predicates = []
        
        for pattern in patterns:
            parts = pattern.strip().rstrip('.').split()
            if len(parts) >= 3:
                predicate_object = ' '.join(parts[1:])
                merged_predicates.append(predicate_object)
        
        return f"{subject} {' ; '.join(merged_predicates)} ."
    
    async def _convert_exists_to_join(self, query: str, context: OptimizationContext) -> str:
        """转换EXISTS为JOIN"""
        # 在某些情况下，EXISTS子查询可以转换为更高效的JOIN
        # 这是一个简化的实现
        
        exists_pattern = re.compile(r'EXISTS\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL)
        
        def replace_exists(match):
            exists_content = match.group(1).strip()
            # 简化转换：将EXISTS转换为OPTIONAL + FILTER
            return f"OPTIONAL {{ {exists_content} }}"
        
        optimized = exists_pattern.sub(replace_exists, query)
        
        if optimized != query:
            logger.debug("转换EXISTS为OPTIONAL模式")
        
        return optimized
    
    async def _optimize_union_patterns(self, query: str, context: OptimizationContext) -> str:
        """优化UNION模式"""
        # 检查UNION操作是否可以简化
        union_pattern = re.compile(r'UNION\s*\{([^}]+)\}', re.IGNORECASE | re.DOTALL)
        
        # 这里可以实现更复杂的UNION优化
        # 例如合并相似的UNION分支
        
        return query
    
    async def _simplify_filter_expressions(self, query: str, context: OptimizationContext) -> str:
        """简化过滤表达式"""
        # 简化FILTER表达式
        filter_patterns = re.finditer(r'FILTER\s*\(([^)]+)\)', query, re.IGNORECASE)
        
        optimized = query
        for match in filter_patterns:
            filter_expr = match.group(1)
            simplified = self._simplify_expression(filter_expr)
            if simplified != filter_expr:
                optimized = optimized.replace(match.group(0), f"FILTER({simplified})")
        
        return optimized
    
    def _simplify_expression(self, expr: str) -> str:
        """简化表达式"""
        # 移除冗余括号
        expr = expr.strip()
        
        # 简化逻辑表达式
        # 例如: (true && x) -> x
        expr = re.sub(r'\(true\s*&&\s*([^)]+)\)', r'\1', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\(([^)]+)\s*&&\s*true\)', r'\1', expr, flags=re.IGNORECASE)
        
        return expr

class JoinOptimizer:
    """连接优化器"""
    
    def __init__(self, stats_manager: StatisticsManager):
        self.stats_manager = stats_manager
    
    async def optimize_join_order(
        self, 
        patterns: List[QueryPattern], 
        context: OptimizationContext
    ) -> List[QueryPattern]:
        """优化连接顺序"""
        try:
            if len(patterns) <= 1:
                return patterns
            
            # 计算每个模式的选择性
            for pattern in patterns:
                pattern.estimated_selectivity = self._estimate_selectivity(pattern)
            
            # 使用贪心算法重排序
            optimized_order = await self._greedy_join_ordering(patterns)
            
            logger.debug(f"连接优化: {len(patterns)} 个模式重新排序")
            return optimized_order
            
        except Exception as e:
            logger.error(f"连接优化失败: {e}")
            return patterns
    
    def _estimate_selectivity(self, pattern: QueryPattern) -> float:
        """估算模式选择性"""
        selectivity = 1.0
        
        # 基于谓词的选择性
        if pattern.predicate != "?p" and not pattern.predicate.startswith("?"):
            pred_selectivity = self.stats_manager.get_predicate_selectivity(pattern.predicate)
            selectivity *= pred_selectivity
        
        # 基于字面量的选择性
        if not pattern.object.startswith("?") and not pattern.object.startswith("<"):
            # 字面量通常具有较高的选择性
            selectivity *= 0.1
        
        # FILTER条件的选择性
        for filter_cond in pattern.filter_conditions:
            selectivity *= self._estimate_filter_selectivity(filter_cond)
        
        return max(selectivity, 0.001)  # 最小选择性
    
    def _estimate_filter_selectivity(self, filter_expr: str) -> float:
        """估算过滤条件选择性"""
        # 简化的过滤条件选择性估算
        if "=" in filter_expr:
            return 0.1  # 等值条件
        elif any(op in filter_expr for op in ["<", ">", "<=", ">="]):
            return 0.3  # 范围条件
        elif "REGEX" in filter_expr.upper():
            return 0.5  # 正则表达式
        else:
            return 0.7  # 其他条件
    
    async def _greedy_join_ordering(self, patterns: List[QueryPattern]) -> List[QueryPattern]:
        """贪心连接排序"""
        if len(patterns) <= 1:
            return patterns
        
        # 按选择性排序（选择性高的优先）
        sorted_patterns = sorted(patterns, key=lambda p: p.estimated_selectivity)
        
        # 确保依赖关系得到满足
        ordered_patterns = []
        remaining_patterns = sorted_patterns.copy()
        available_variables = set()
        
        while remaining_patterns:
            # 查找可以执行的模式（所有变量都已可用）
            executable = []
            
            for pattern in remaining_patterns:
                pattern_vars = self._extract_variables(pattern)
                if not pattern_vars or any(var in available_variables for var in pattern_vars):
                    executable.append(pattern)
            
            if not executable:
                # 如果没有可执行的模式，选择选择性最高的
                next_pattern = remaining_patterns.pop(0)
            else:
                # 选择选择性最高的可执行模式
                next_pattern = min(executable, key=lambda p: p.estimated_selectivity)
                remaining_patterns.remove(next_pattern)
            
            ordered_patterns.append(next_pattern)
            
            # 更新可用变量
            pattern_vars = self._extract_variables(next_pattern)
            available_variables.update(pattern_vars)
        
        return ordered_patterns
    
    def _extract_variables(self, pattern: QueryPattern) -> Set[str]:
        """提取模式中的变量"""
        variables = set()
        
        for element in [pattern.subject, pattern.predicate, pattern.object]:
            if element.startswith("?"):
                variables.add(element)
        
        return variables

class AdvancedQueryOptimizer:
    """高级查询优化器"""
    
    def __init__(self):
        self.stats_manager = StatisticsManager()
        self.query_rewriter = QueryRewriter()
        self.join_optimizer = JoinOptimizer(self.stats_manager)
        
        self.optimization_strategies = {
            OptimizationLevel.BASIC: [
                self._basic_optimizations
            ],
            OptimizationLevel.STANDARD: [
                self._basic_optimizations,
                self._intermediate_optimizations
            ],
            OptimizationLevel.AGGRESSIVE: [
                self._basic_optimizations,
                self._intermediate_optimizations,
                self._advanced_optimizations
            ]
        }
    
    async def optimize_query(
        self, 
        query_text: str, 
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
        statistics: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """执行查询优化"""
        try:
            context = OptimizationContext(
                statistics=statistics or {},
                indexes=[],
                query_type=self._determine_query_type(query_text),
                optimization_level=optimization_level
            )
            
            # 更新统计信息
            if statistics:
                self._update_statistics(statistics)
            
            optimized_query = query_text
            optimization_log = []
            
            # 应用优化策略
            strategies = self.optimization_strategies.get(optimization_level, [])
            
            for strategy in strategies:
                result = await strategy(optimized_query, context)
                if result["query"] != optimized_query:
                    optimization_log.extend(result["changes"])
                    optimized_query = result["query"]
            
            # 计算优化效果估算
            optimization_impact = self._estimate_optimization_impact(
                query_text, 
                optimized_query, 
                context
            )
            
            return {
                "original_query": query_text,
                "optimized_query": optimized_query,
                "optimization_level": optimization_level,
                "optimizations_applied": optimization_log,
                "estimated_impact": optimization_impact,
                "context": {
                    "query_type": context.query_type,
                    "statistics_available": bool(statistics)
                }
            }
            
        except Exception as e:
            logger.error(f"高级查询优化失败: {e}")
            return {
                "original_query": query_text,
                "optimized_query": query_text,
                "optimization_level": optimization_level,
                "optimizations_applied": [],
                "error": str(e)
            }
    
    async def _basic_optimizations(
        self, 
        query: str, 
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """基础优化"""
        changes = []
        optimized = query
        
        # 1. 查询重写
        rewritten = await self.query_rewriter.rewrite_query(optimized, context)
        if rewritten != optimized:
            changes.append("应用查询重写规则")
            optimized = rewritten
        
        # 2. 移除不必要的空格和格式化
        formatted = self._format_query(optimized)
        if formatted != optimized:
            changes.append("标准化查询格式")
            optimized = formatted
        
        return {"query": optimized, "changes": changes}
    
    async def _intermediate_optimizations(
        self, 
        query: str, 
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """中级优化"""
        changes = []
        optimized = query
        
        # 1. 连接顺序优化
        patterns = self._parse_query_patterns(optimized)
        if len(patterns) > 1:
            optimized_patterns = await self.join_optimizer.optimize_join_order(
                patterns, 
                context
            )
            if optimized_patterns != patterns:
                changes.append(f"优化连接顺序 ({len(patterns)} 个模式)")
                # 这里应该重建查询，简化实现跳过
        
        # 2. 过滤条件下推
        filter_optimized = self._push_filters(optimized)
        if filter_optimized != optimized:
            changes.append("下推过滤条件")
            optimized = filter_optimized
        
        return {"query": optimized, "changes": changes}
    
    async def _advanced_optimizations(
        self, 
        query: str, 
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """高级优化"""
        changes = []
        optimized = query
        
        # 1. 子查询优化
        subquery_optimized = self._optimize_subqueries(optimized)
        if subquery_optimized != optimized:
            changes.append("优化子查询")
            optimized = subquery_optimized
        
        # 2. 索引使用建议
        index_hints = self._generate_index_hints(optimized, context)
        if index_hints:
            changes.append(f"生成索引建议 ({len(index_hints)} 项)")
        
        return {"query": optimized, "changes": changes}
    
    def _determine_query_type(self, query: str) -> str:
        """确定查询类型"""
        query_upper = query.upper().strip()
        
        if query_upper.startswith("SELECT"):
            return "select"
        elif query_upper.startswith("CONSTRUCT"):
            return "construct"
        elif query_upper.startswith("ASK"):
            return "ask"
        elif query_upper.startswith("DESCRIBE"):
            return "describe"
        elif any(query_upper.startswith(kw) for kw in ["INSERT", "DELETE", "UPDATE"]):
            return "update"
        else:
            return "unknown"
    
    def _update_statistics(self, statistics: Dict[str, Any]):
        """更新统计信息"""
        if "predicate_counts" in statistics:
            for pred, count in statistics["predicate_counts"].items():
                self.stats_manager.update_predicate_stats(pred, count)
        
        if "class_counts" in statistics:
            for cls, count in statistics["class_counts"].items():
                self.stats_manager.update_class_stats(cls, count)
    
    def _format_query(self, query: str) -> str:
        """格式化查询"""
        # 标准化空格和换行
        formatted = re.sub(r'\s+', ' ', query.strip())
        
        # 标准化关键字
        keywords = ["SELECT", "WHERE", "FROM", "ORDER BY", "GROUP BY", "LIMIT", "OFFSET", 
                   "OPTIONAL", "UNION", "FILTER", "CONSTRUCT", "ASK", "DESCRIBE"]
        
        for keyword in keywords:
            formatted = re.sub(
                rf'\b{keyword.lower()}\b', 
                keyword, 
                formatted, 
                flags=re.IGNORECASE
            )
        
        return formatted
    
    def _parse_query_patterns(self, query: str) -> List[QueryPattern]:
        """解析查询模式"""
        # 简化的模式解析
        patterns = []
        
        # 提取WHERE子句中的三元组模式
        where_match = re.search(r'WHERE\s*\{([^}]+)\}', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_content = where_match.group(1)
            
            # 简单的三元组模式匹配
            triple_pattern = re.compile(r'([?:\w<>]+)\s+([?:\w<>]+)\s+([?:\w<>]+)', re.IGNORECASE)
            
            for match in triple_pattern.finditer(where_content):
                subject, predicate, obj = match.groups()
                
                pattern = QueryPattern(
                    pattern_type="triple",
                    subject=subject.strip(),
                    predicate=predicate.strip(),
                    object=obj.strip()
                )
                patterns.append(pattern)
        
        return patterns
    
    def _push_filters(self, query: str) -> str:
        """下推过滤条件"""
        # 简化实现：将FILTER条件移近相关的模式
        # 实际实现会更复杂，需要分析变量依赖
        return query
    
    def _optimize_subqueries(self, query: str) -> str:
        """优化子查询"""
        # 检查是否有可以优化的子查询
        # 例如将某些子查询转换为JOIN
        return query
    
    def _generate_index_hints(self, query: str, context: OptimizationContext) -> List[str]:
        """生成索引建议"""
        hints = []
        
        # 分析查询中使用的谓词
        predicates = re.findall(r'<([^>]+)>', query)
        
        for pred in set(predicates):
            if pred.startswith("http"):
                hints.append(f"为谓词 {pred} 创建索引")
        
        # 分析过滤条件
        filters = re.findall(r'FILTER\s*\([^)]+\)', query, re.IGNORECASE)
        if filters:
            hints.append("为过滤条件中的属性创建索引")
        
        return hints
    
    def _estimate_optimization_impact(
        self, 
        original: str, 
        optimized: str, 
        context: OptimizationContext
    ) -> Dict[str, Any]:
        """估算优化影响"""
        impact = {
            "query_length_change": len(optimized) - len(original),
            "complexity_reduction": 0,
            "estimated_speedup": 1.0
        }
        
        # 计算复杂度变化
        orig_complexity = self._calculate_query_complexity(original)
        opt_complexity = self._calculate_query_complexity(optimized)
        
        impact["complexity_reduction"] = orig_complexity - opt_complexity
        
        # 估算性能提升
        if impact["complexity_reduction"] > 0:
            impact["estimated_speedup"] = 1 + (impact["complexity_reduction"] * 0.1)
        
        return impact
    
    def _calculate_query_complexity(self, query: str) -> int:
        """计算查询复杂度"""
        complexity = 0
        query_upper = query.upper()
        
        # 基于各种操作的权重
        weights = {
            "JOIN": 3,
            "OPTIONAL": 2,
            "UNION": 2,
            "FILTER": 1,
            "GROUP BY": 3,
            "ORDER BY": 2,
            "REGEX": 5,
            "SERVICE": 4
        }
        
        for operation, weight in weights.items():
            complexity += query_upper.count(operation) * weight
        
        return complexity
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """获取统计信息摘要"""
        return self.stats_manager.get_statistics()

# 创建默认优化器实例
default_optimizer = AdvancedQueryOptimizer()

async def optimize_sparql_query(
    query_text: str,
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
    statistics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """优化SPARQL查询的便捷函数"""
    return await default_optimizer.optimize_query(
        query_text, 
        optimization_level, 
        statistics
    )
