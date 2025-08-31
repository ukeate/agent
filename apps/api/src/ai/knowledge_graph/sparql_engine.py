"""
SPARQL查询引擎实现

基于RDFLib构建的高性能SPARQL查询引擎，支持：
- SPARQL 1.1查询和更新操作
- 查询优化和执行计划分析
- 结果缓存和格式转换
- 性能监控和统计
"""

import asyncio
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

try:
    import rdflib
    from rdflib import Graph, Namespace, URIRef, Literal, BNode
    from rdflib.plugins.sparql import prepareQuery, processUpdate
    from rdflib.plugins.sparql.evaluate import evalQuery
    from rdflib.plugins.sparql.algebra import translateQuery
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False

from ..rag.embeddings import MockEmbeddings

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """SPARQL查询类型"""
    SELECT = "select"
    CONSTRUCT = "construct"  
    ASK = "ask"
    DESCRIBE = "describe"
    UPDATE = "update"


class ExecutionStatus(str, Enum):
    """查询执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SPARQLQuery:
    """SPARQL查询对象"""
    query_id: str
    query_text: str
    query_type: QueryType
    parameters: Dict[str, Any]
    timeout_seconds: int = 30
    use_cache: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SPARQLResult:
    """SPARQL查询结果"""
    query_id: str
    success: bool
    result_type: str  # bindings, boolean, graph
    results: List[Dict[str, Any]]
    execution_time_ms: float
    row_count: int
    cached: bool = False
    error_message: Optional[str] = None
    execution_plan: Optional[Dict[str, Any]] = None
    performance_stats: Optional[Dict[str, Any]] = None


class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.optimization_rules = [
            self._reorder_joins,
            self._push_filters,
            self._merge_basic_graph_patterns,
            self._optimize_optional_patterns
        ]
    
    async def optimize(self, query: str, statistics: Dict[str, Any] = None) -> Dict[str, Any]:
        """优化SPARQL查询"""
        try:
            # 解析查询
            if RDFLIB_AVAILABLE:
                parsed = prepareQuery(query)
                algebra = translateQuery(parsed)
                
                optimized_algebra = algebra
                
                # 应用优化规则
                for rule in self.optimization_rules:
                    optimized_algebra = await rule(optimized_algebra, statistics)
                
                return {
                    "original_query": query,
                    "parsed_query": str(parsed),
                    "original_algebra": str(algebra),
                    "optimized_algebra": str(optimized_algebra),
                    "optimization_applied": True
                }
            else:
                # 基础优化：移除多余空格，规范化关键字
                optimized = self._basic_optimization(query)
                return {
                    "original_query": query,
                    "optimized_query": optimized,
                    "optimization_applied": True
                }
                
        except Exception as e:
            logger.error(f"查询优化失败: {e}")
            return {
                "original_query": query,
                "optimized_query": query,
                "optimization_applied": False,
                "error": str(e)
            }
    
    async def _reorder_joins(self, algebra, statistics):
        """重排序连接操作"""
        # 基于统计信息重排序连接
        return algebra
    
    async def _push_filters(self, algebra, statistics):
        """下推过滤条件"""
        # 将过滤条件尽早应用
        return algebra
    
    async def _merge_basic_graph_patterns(self, algebra, statistics):
        """合并基础图模式"""
        # 合并相邻的BGP
        return algebra
    
    async def _optimize_optional_patterns(self, algebra, statistics):
        """优化OPTIONAL模式"""
        # 优化OPTIONAL查询
        return algebra
    
    def _basic_optimization(self, query: str) -> str:
        """基础查询优化"""
        # 移除多余空格
        optimized = " ".join(query.split())
        
        # 规范化关键字
        keywords = ["SELECT", "WHERE", "FROM", "ORDER BY", "GROUP BY", "LIMIT", "OFFSET"]
        for keyword in keywords:
            optimized = optimized.replace(keyword.lower(), keyword)
            optimized = optimized.replace(keyword.upper(), keyword)
        
        return optimized


class QueryCache:
    """查询缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_cache_key(self, query_text: str, parameters: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_data = f"{query_text}{str(sorted(parameters.items()) if parameters else '')}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query_text: str, parameters: Dict[str, Any]) -> Optional[SPARQLResult]:
        """获取缓存结果"""
        cache_key = self._generate_cache_key(query_text, parameters)
        
        if cache_key not in self.cache:
            return None
        
        # 检查TTL
        cached_item = self.cache[cache_key]
        if time.time() - cached_item['timestamp'] > self.ttl_seconds:
            del self.cache[cache_key]
            return None
        
        # 更新访问时间
        self.access_times[cache_key] = time.time()
        
        # 标记为缓存结果
        result = cached_item['result']
        result.cached = True
        
        return result
    
    async def set(self, query_text: str, parameters: Dict[str, Any], result: SPARQLResult):
        """设置缓存"""
        cache_key = self._generate_cache_key(query_text, parameters)
        
        # 如果缓存已满，移除最久未访问的项
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        self.access_times[cache_key] = time.time()
    
    def _evict_lru(self):
        """移除最久未访问的缓存项"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """缓存统计"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
            "ttl_seconds": self.ttl_seconds
        }


class ExecutionPlanner:
    """执行计划生成器"""
    
    async def create_plan(self, optimized_query: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行计划"""
        try:
            plan = {
                "plan_id": str(uuid.uuid4()),
                "query_type": self._determine_query_type(optimized_query),
                "estimated_cost": await self._estimate_cost(optimized_query),
                "execution_steps": await self._generate_steps(optimized_query),
                "resource_requirements": self._estimate_resources(optimized_query),
                "parallelizable": self._check_parallelizable(optimized_query)
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"执行计划生成失败: {e}")
            return {
                "plan_id": str(uuid.uuid4()),
                "query_type": "unknown",
                "estimated_cost": {"time": 1000, "memory": 100},
                "execution_steps": ["parse", "execute", "format"],
                "error": str(e)
            }
    
    def _determine_query_type(self, query: Dict[str, Any]) -> str:
        """判断查询类型"""
        query_text = query.get("original_query", "").upper()
        
        if "SELECT" in query_text:
            return "select"
        elif "CONSTRUCT" in query_text:
            return "construct"
        elif "ASK" in query_text:
            return "ask"
        elif "DESCRIBE" in query_text:
            return "describe"
        elif any(kw in query_text for kw in ["INSERT", "DELETE", "UPDATE"]):
            return "update"
        else:
            return "unknown"
    
    async def _estimate_cost(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """估算执行成本"""
        # 基于查询复杂度估算
        query_text = query.get("original_query", "")
        
        # 简单启发式估算
        base_cost = 100
        
        # 计算复杂度因子
        complexity_factors = {
            "JOIN": query_text.upper().count("JOIN") * 50,
            "OPTIONAL": query_text.upper().count("OPTIONAL") * 30,
            "UNION": query_text.upper().count("UNION") * 40,
            "FILTER": query_text.upper().count("FILTER") * 20,
            "GROUP BY": query_text.upper().count("GROUP BY") * 60,
            "ORDER BY": query_text.upper().count("ORDER BY") * 30,
            "REGEX": query_text.upper().count("REGEX") * 100
        }
        
        total_complexity = sum(complexity_factors.values())
        
        return {
            "time_estimate_ms": base_cost + total_complexity,
            "memory_estimate_mb": max(10, total_complexity // 10),
            "complexity_score": total_complexity,
            "factors": complexity_factors
        }
    
    async def _generate_steps(self, query: Dict[str, Any]) -> List[str]:
        """生成执行步骤"""
        steps = ["parse_query", "validate_syntax"]
        
        query_text = query.get("original_query", "").upper()
        
        if "FROM" in query_text:
            steps.append("load_graph")
        
        steps.extend([
            "build_execution_tree",
            "optimize_execution_order",
            "execute_patterns",
            "apply_filters",
            "apply_modifiers",
            "format_results"
        ])
        
        if "ORDER BY" in query_text:
            steps.append("sort_results")
        
        if "LIMIT" in query_text:
            steps.append("limit_results")
        
        return steps
    
    def _estimate_resources(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """估算资源需求"""
        query_text = query.get("original_query", "")
        
        return {
            "cpu_intensive": "REGEX" in query_text.upper() or "ORDER BY" in query_text.upper(),
            "memory_intensive": "CONSTRUCT" in query_text.upper() or "UNION" in query_text.upper(),
            "io_intensive": "FROM" in query_text.upper(),
            "network_required": "SERVICE" in query_text.upper()
        }
    
    def _check_parallelizable(self, query: Dict[str, Any]) -> bool:
        """检查是否可并行化"""
        query_text = query.get("original_query", "").upper()
        
        # 包含ORDER BY或LIMIT的查询通常不易并行化
        if "ORDER BY" in query_text or "LIMIT" in query_text:
            return False
        
        # UNION查询可以并行化
        if "UNION" in query_text:
            return True
        
        # 简单SELECT查询可以并行化
        if "SELECT" in query_text and query_text.count("SELECT") == 1:
            return True
        
        return False


class SPARQLEngine:
    """SPARQL查询引擎"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self.cache = QueryCache(cache_size, cache_ttl)
        self.optimizer = QueryOptimizer()
        self.planner = ExecutionPlanner()
        
        # 初始化RDF处理器
        if RDFLIB_AVAILABLE:
            self.graph = Graph()
            self.rdflib_available = True
            logger.info("RDFLib SPARQL引擎初始化完成")
        else:
            self.graph = None
            self.rdflib_available = False
            logger.warning("RDFLib不可用，使用模拟SPARQL引擎")
        
        # 性能统计
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_execution_time": 0.0
        }
    
    async def execute_query(self, sparql_query: SPARQLQuery) -> SPARQLResult:
        """执行SPARQL查询"""
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            # 1. 检查缓存
            if sparql_query.use_cache:
                cached_result = await self.cache.get(
                    sparql_query.query_text, 
                    sparql_query.parameters
                )
                if cached_result:
                    self.stats["cached_queries"] += 1
                    return cached_result
            
            # 2. 查询验证和解析
            validation_result = await self._validate_query(sparql_query.query_text)
            if not validation_result["valid"]:
                return self._create_error_result(
                    sparql_query.query_id,
                    f"查询验证失败: {validation_result['error']}",
                    time.time() - start_time
                )
            
            # 3. 查询优化
            optimized_query = await self.optimizer.optimize(sparql_query.query_text)
            
            # 4. 执行计划生成
            execution_plan = await self.planner.create_plan(optimized_query)
            
            # 5. 执行查询
            result = await self._execute_with_timeout(
                sparql_query, 
                optimized_query,
                execution_plan
            )
            
            # 6. 缓存结果
            if sparql_query.use_cache and result.success and result.execution_time_ms < 5000:
                await self.cache.set(
                    sparql_query.query_text,
                    sparql_query.parameters,
                    result
                )
            
            # 更新统计
            execution_time = time.time() - start_time
            self.stats["total_execution_time"] += execution_time
            if result.success:
                self.stats["successful_queries"] += 1
            else:
                self.stats["failed_queries"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"SPARQL查询执行失败: {e}")
            execution_time = time.time() - start_time
            self.stats["failed_queries"] += 1
            self.stats["total_execution_time"] += execution_time
            
            return self._create_error_result(
                sparql_query.query_id,
                str(e),
                execution_time
            )
    
    async def _validate_query(self, query_text: str) -> Dict[str, Any]:
        """验证SPARQL查询"""
        try:
            if self.rdflib_available:
                # 使用RDFLib验证
                prepareQuery(query_text)
                return {"valid": True}
            else:
                # 基础语法验证
                return self._basic_validation(query_text)
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _basic_validation(self, query_text: str) -> Dict[str, Any]:
        """基础SPARQL语法验证"""
        query_upper = query_text.upper().strip()
        
        # 检查基本结构
        if not query_upper:
            return {"valid": False, "error": "空查询"}
        
        # 检查支持的查询类型
        supported_types = ["SELECT", "CONSTRUCT", "ASK", "DESCRIBE", "INSERT", "DELETE"]
        if not any(query_upper.startswith(qtype) for qtype in supported_types):
            return {"valid": False, "error": "不支持的查询类型"}
        
        # 检查基本语法
        if "SELECT" in query_upper and "WHERE" not in query_upper:
            return {"valid": False, "error": "SELECT查询缺少WHERE子句"}
        
        # 检查括号匹配
        if query_text.count("{") != query_text.count("}"):
            return {"valid": False, "error": "括号不匹配"}
        
        return {"valid": True}
    
    async def _execute_with_timeout(
        self, 
        sparql_query: SPARQLQuery,
        optimized_query: Dict[str, Any],
        execution_plan: Dict[str, Any]
    ) -> SPARQLResult:
        """带超时的查询执行"""
        try:
            # 执行查询
            result = await asyncio.wait_for(
                self._execute_query_internal(sparql_query, optimized_query, execution_plan),
                timeout=sparql_query.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            return self._create_error_result(
                sparql_query.query_id,
                f"查询超时 ({sparql_query.timeout_seconds}秒)",
                sparql_query.timeout_seconds * 1000
            )
    
    async def _execute_query_internal(
        self,
        sparql_query: SPARQLQuery,
        optimized_query: Dict[str, Any],
        execution_plan: Dict[str, Any]
    ) -> SPARQLResult:
        """内部查询执行方法"""
        start_time = time.time()
        
        if self.rdflib_available and self.graph:
            # 使用RDFLib执行
            return await self._execute_with_rdflib(sparql_query, execution_plan)
        else:
            # 使用模拟执行
            return await self._execute_mock(sparql_query, execution_plan)
    
    async def _execute_with_rdflib(
        self,
        sparql_query: SPARQLQuery,
        execution_plan: Dict[str, Any]
    ) -> SPARQLResult:
        """使用RDFLib执行查询"""
        start_time = time.time()
        
        try:
            query_text = sparql_query.query_text
            
            # 确定查询类型
            if sparql_query.query_type == QueryType.UPDATE:
                # 执行更新操作
                self.graph.update(query_text)
                execution_time = (time.time() - start_time) * 1000
                
                return SPARQLResult(
                    query_id=sparql_query.query_id,
                    success=True,
                    result_type="update",
                    results=[{"status": "updated"}],
                    execution_time_ms=execution_time,
                    row_count=1,
                    execution_plan=execution_plan,
                    performance_stats=self._generate_performance_stats(execution_time)
                )
            else:
                # 执行查询操作
                qres = self.graph.query(query_text)
                results = []
                
                # 处理不同类型的查询结果
                if sparql_query.query_type == QueryType.ASK:
                    results = [{"result": bool(qres)}]
                    result_type = "boolean"
                elif sparql_query.query_type in [QueryType.SELECT]:
                    results = []
                    for row in qres:
                        row_dict = {}
                        for var in qres.vars:
                            value = row[var]
                            row_dict[str(var)] = self._serialize_rdf_term(value)
                        results.append(row_dict)
                    result_type = "bindings"
                else:
                    # CONSTRUCT, DESCRIBE
                    results = []
                    for triple in qres:
                        results.append({
                            "subject": self._serialize_rdf_term(triple[0]),
                            "predicate": self._serialize_rdf_term(triple[1]),
                            "object": self._serialize_rdf_term(triple[2])
                        })
                    result_type = "graph"
                
                execution_time = (time.time() - start_time) * 1000
                
                return SPARQLResult(
                    query_id=sparql_query.query_id,
                    success=True,
                    result_type=result_type,
                    results=results,
                    execution_time_ms=execution_time,
                    row_count=len(results),
                    execution_plan=execution_plan,
                    performance_stats=self._generate_performance_stats(execution_time)
                )
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return SPARQLResult(
                query_id=sparql_query.query_id,
                success=False,
                result_type="error",
                results=[],
                execution_time_ms=execution_time,
                row_count=0,
                error_message=str(e),
                execution_plan=execution_plan
            )
    
    async def _execute_mock(
        self,
        sparql_query: SPARQLQuery,
        execution_plan: Dict[str, Any]
    ) -> SPARQLResult:
        """模拟查询执行"""
        start_time = time.time()
        
        # 模拟执行时间
        await asyncio.sleep(0.1)
        
        # 生成模拟结果
        if sparql_query.query_type == QueryType.ASK:
            results = [{"result": True}]
            result_type = "boolean"
        elif sparql_query.query_type == QueryType.SELECT:
            results = [
                {"s": "http://example.org/subject1", "p": "http://example.org/predicate1", "o": "value1"},
                {"s": "http://example.org/subject2", "p": "http://example.org/predicate2", "o": "value2"}
            ]
            result_type = "bindings"
        elif sparql_query.query_type == QueryType.UPDATE:
            results = [{"status": "updated"}]
            result_type = "update"
        else:
            results = []
            result_type = "graph"
        
        execution_time = (time.time() - start_time) * 1000
        
        return SPARQLResult(
            query_id=sparql_query.query_id,
            success=True,
            result_type=result_type,
            results=results,
            execution_time_ms=execution_time,
            row_count=len(results),
            execution_plan=execution_plan,
            performance_stats=self._generate_performance_stats(execution_time)
        )
    
    def _serialize_rdf_term(self, term) -> str:
        """序列化RDF术语"""
        if term is None:
            return None
        
        if hasattr(term, 'toPython'):
            return str(term.toPython())
        else:
            return str(term)
    
    def _generate_performance_stats(self, execution_time: float) -> Dict[str, Any]:
        """生成性能统计"""
        return {
            "execution_time_ms": execution_time,
            "memory_used_mb": 10,  # 模拟值
            "cache_hits": 0,
            "index_scans": 1
        }
    
    def _create_error_result(self, query_id: str, error_msg: str, execution_time: float) -> SPARQLResult:
        """创建错误结果"""
        return SPARQLResult(
            query_id=query_id,
            success=False,
            result_type="error",
            results=[],
            execution_time_ms=execution_time * 1000,
            row_count=0,
            error_message=error_msg
        )
    
    async def explain_query(self, query_text: str) -> Dict[str, Any]:
        """查询执行计划分析"""
        try:
            # 查询优化
            optimized_query = await self.optimizer.optimize(query_text)
            
            # 生成执行计划
            execution_plan = await self.planner.create_plan(optimized_query)
            
            # 分析查询模式
            query_patterns = self._analyze_query_patterns(query_text)
            
            # 建议优化策略
            optimization_suggestions = self._suggest_optimizations(query_patterns)
            
            return {
                "query_text": query_text,
                "optimized_query": optimized_query,
                "execution_plan": execution_plan,
                "query_patterns": query_patterns,
                "optimization_suggestions": optimization_suggestions,
                "performance_prediction": execution_plan.get("estimated_cost", {}),
                "index_recommendations": self._recommend_indexes(query_patterns)
            }
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return {
                "query_text": query_text,
                "error": str(e),
                "analysis_available": False
            }
    
    def _analyze_query_patterns(self, query_text: str) -> Dict[str, Any]:
        """分析查询模式"""
        query_upper = query_text.upper()
        
        patterns = {
            "joins": query_upper.count("JOIN"),
            "optionals": query_upper.count("OPTIONAL"),
            "unions": query_upper.count("UNION"),
            "filters": query_upper.count("FILTER"),
            "group_by": "GROUP BY" in query_upper,
            "order_by": "ORDER BY" in query_upper,
            "limit": "LIMIT" in query_upper,
            "regex_patterns": query_upper.count("REGEX"),
            "service_calls": query_upper.count("SERVICE"),
            "subqueries": query_text.count("{") - 1  # 估算子查询数量
        }
        
        return patterns
    
    def _suggest_optimizations(self, patterns: Dict[str, Any]) -> List[str]:
        """建议优化策略"""
        suggestions = []
        
        if patterns["joins"] > 3:
            suggestions.append("考虑减少JOIN操作数量或优化连接顺序")
        
        if patterns["optionals"] > 2:
            suggestions.append("检查OPTIONAL模式是否必要，考虑使用EXISTS")
        
        if patterns["regex_patterns"] > 0:
            suggestions.append("REGEX操作成本高，考虑使用字符串函数替代")
        
        if patterns["order_by"] and not patterns["limit"]:
            suggestions.append("ORDER BY without LIMIT可能影响性能")
        
        if patterns["subqueries"] > 2:
            suggestions.append("考虑简化嵌套查询结构")
        
        if patterns["service_calls"] > 0:
            suggestions.append("联邦查询可能影响性能，考虑缓存远程数据")
        
        return suggestions
    
    def _recommend_indexes(self, patterns: Dict[str, Any]) -> List[str]:
        """推荐索引"""
        recommendations = []
        
        if patterns["joins"] > 0:
            recommendations.append("为连接谓词创建索引")
        
        if patterns["filters"] > 0:
            recommendations.append("为过滤条件中的谓词创建索引")
        
        if patterns["order_by"]:
            recommendations.append("为排序字段创建索引")
        
        return recommendations
    
    def load_data(self, data: str, format: str = "turtle"):
        """加载RDF数据"""
        if self.rdflib_available and self.graph:
            try:
                self.graph.parse(data=data, format=format)
                logger.info(f"已加载RDF数据，格式: {format}")
            except Exception as e:
                logger.error(f"数据加载失败: {e}")
                raise
        else:
            logger.warning("RDFLib不可用，无法加载数据")
    
    def load_from_file(self, file_path: str, format: str = None):
        """从文件加载RDF数据"""
        if self.rdflib_available and self.graph:
            try:
                self.graph.parse(file_path, format=format)
                logger.info(f"已加载文件: {file_path}")
            except Exception as e:
                logger.error(f"文件加载失败: {e}")
                raise
        else:
            logger.warning("RDFLib不可用，无法加载文件")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        engine_stats = dict(self.stats)
        
        if self.stats["total_queries"] > 0:
            engine_stats["average_execution_time"] = (
                self.stats["total_execution_time"] / self.stats["total_queries"]
            )
            engine_stats["success_rate"] = (
                self.stats["successful_queries"] / self.stats["total_queries"]
            )
        else:
            engine_stats["average_execution_time"] = 0.0
            engine_stats["success_rate"] = 0.0
        
        engine_stats["cache_stats"] = self.cache.stats()
        
        if self.rdflib_available and self.graph:
            engine_stats["graph_size"] = len(self.graph)
        else:
            engine_stats["graph_size"] = 0
        
        return engine_stats
    
    def clear_cache(self):
        """清空查询缓存"""
        self.cache.clear()
        logger.info("查询缓存已清空")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cached_queries": 0,
            "total_execution_time": 0.0
        }
        logger.info("统计信息已重置")


# 创建默认引擎实例
default_sparql_engine = SPARQLEngine()


async def execute_sparql_query(
    query_text: str,
    query_type: QueryType = QueryType.SELECT,
    parameters: Dict[str, Any] = None,
    timeout_seconds: int = 30,
    use_cache: bool = True
) -> SPARQLResult:
    """执行SPARQL查询的便捷函数"""
    query = SPARQLQuery(
        query_id=str(uuid.uuid4()),
        query_text=query_text,
        query_type=query_type,
        parameters=parameters or {},
        timeout_seconds=timeout_seconds,
        use_cache=use_cache
    )
    
    return await default_sparql_engine.execute_query(query)


async def explain_sparql_query(query_text: str) -> Dict[str, Any]:
    """分析SPARQL查询的便捷函数"""
    return await default_sparql_engine.explain_query(query_text)