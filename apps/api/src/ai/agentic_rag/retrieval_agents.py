"""
多策略检索代理协作系统

实现多个专业化检索代理的协调与协作，包括：
- 语义检索专家代理（基于Story 3.1）
- 关键词匹配专家代理（BM25算法）
- 结构化数据检索代理（数据库查询）
- 代理协作调度和结果融合机制
- 检索策略动态选择算法
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
from sqlalchemy import select, or_
from src.ai.rag.retriever import SemanticRetriever
from src.ai.rag.hybrid_search import get_hybrid_search_engine, SearchStrategy
from src.core.config import get_settings
from src.core.database import get_db_session
from src.models.database.session import Session
from src.models.database.experiment import Experiment, ExperimentVariant
from src.models.database.workflow import WorkflowModel
from src.models.database.user import AuthUser
from src.models.database.api_key import APIKey
from src.models.database.event_tracking import EventStream
from src.models.database.supervisor import SupervisorAgent, SupervisorTask, SupervisorDecision
from .query_analyzer import QueryAnalysis, QueryIntent

from src.core.logging import get_logger
logger = get_logger(__name__)

def _openai_key_configured() -> bool:
    key = (get_settings().OPENAI_API_KEY or "").strip()
    return bool(key) and not key.startswith("sk-test")

class RetrievalStrategy(str, Enum):
    """检索策略类型"""
    SEMANTIC = "semantic"        # 语义检索
    KEYWORD = "keyword"          # 关键词匹配
    STRUCTURED = "structured"    # 结构化查询
    HYBRID = "hybrid"           # 混合检索

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    agent_type: RetrievalStrategy
    query: str
    results: List[Dict[str, Any]]
    score: float  # 整体检索质量评分 0-1
    confidence: float  # 结果置信度 0-1
    processing_time: float  # 处理时间（秒）
    metadata: Dict[str, Any] = None
    explanation: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseRetrievalAgent(ABC):
    """检索代理基类"""
    
    def __init__(self, name: str, strategy: RetrievalStrategy):
        self.name = name
        self.strategy = strategy
        self.performance_stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "avg_score": 0.0,
            "success_rate": 0.0
        }
    
    @abstractmethod
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行检索"""
        ...
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估代理对查询的适用性 (0-1)"""
        return 0.5  # 默认适用性
    
    def update_performance_stats(self, result: RetrievalResult):
        """更新性能统计"""
        self.performance_stats["total_queries"] += 1
        
        # 更新平均响应时间
        prev_avg = self.performance_stats["avg_response_time"]
        n = self.performance_stats["total_queries"]
        self.performance_stats["avg_response_time"] = (
            (prev_avg * (n - 1) + result.processing_time) / n
        )
        
        # 更新平均分数
        prev_score = self.performance_stats["avg_score"]
        self.performance_stats["avg_score"] = (
            (prev_score * (n - 1) + result.score) / n
        )
        
        # 更新成功率（分数 > 0.3 视为成功）
        success_count = self.performance_stats["success_rate"] * (n - 1)
        if result.score > 0.3:
            success_count += 1
        self.performance_stats["success_rate"] = success_count / n

class SemanticRetrievalAgent(BaseRetrievalAgent):
    """语义检索专家代理（继承Story 3.1基础能力）"""
    
    def __init__(self):
        super().__init__("SemanticExpert", RetrievalStrategy.SEMANTIC)
        self.semantic_retriever = SemanticRetriever()
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估语义检索的适用性"""
        suitability = 0.6  # 基础适用性
        
        # 基于意图类型调整
        if query_analysis.intent_type in [QueryIntent.FACTUAL, QueryIntent.EXPLORATORY]:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.CREATIVE:
            suitability += 0.2
        
        # 基于查询复杂度调整
        if query_analysis.complexity_score > 0.5:
            suitability += 0.1
        
        # 基于实体数量调整
        if len(query_analysis.entities) > 0:
            suitability += 0.1
        
        return min(suitability, 1.0)
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行语义检索"""
        import time
        start_time = time.time()
        
        try:
            if not _openai_key_configured():
                processing_time = time.time() - start_time
                result = RetrievalResult(
                    agent_type=RetrievalStrategy.SEMANTIC,
                    query=query_analysis.query_text,
                    results=[],
                    score=0.0,
                    confidence=0.0,
                    processing_time=processing_time,
                    explanation="未配置OpenAI嵌入服务，语义检索不可用"
                )
                self.update_performance_stats(result)
                return result

            # 选择集合
            collection = "code" if query_analysis.intent_type == QueryIntent.CODE else "documents"
            
            # 执行语义搜索
            results = await self.semantic_retriever.search(
                query=query_analysis.query_text,
                collection=collection,
                limit=limit,
                score_threshold=0.3,
                filter_dict=filters
            )
            
            # 计算整体分数
            if results:
                avg_score = sum(r["score"] for r in results) / len(results)
                # 语义检索置信度基于平均分数和结果数量
                confidence = min(avg_score * (1 + 0.1 * len(results) / limit), 1.0)
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "collection": collection,
                    "total_results": len(results),
                    "intent_type": query_analysis.intent_type.value
                },
                explanation=f"使用语义向量搜索在{collection}集合中找到{len(results)}个相关结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Semantic retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"语义检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result

class KeywordRetrievalAgent(BaseRetrievalAgent):
    """关键词匹配专家代理（BM25算法）"""
    
    def __init__(self):
        super().__init__("KeywordExpert", RetrievalStrategy.KEYWORD)
        self.bm42_engine = None
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估关键词检索的适用性"""
        suitability = 0.5  # 基础适用性
        
        # 基于意图类型调整
        if query_analysis.intent_type in [QueryIntent.PROCEDURAL, QueryIntent.CODE]:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.FACTUAL:
            suitability += 0.2
        
        # 基于关键词数量调整
        if len(query_analysis.keywords) >= 3:
            suitability += 0.2
        elif len(query_analysis.keywords) >= 2:
            suitability += 0.1
        
        # 基于查询长度调整（较短查询更适合关键词匹配）
        if len(query_analysis.query_text.split()) <= 5:
            suitability += 0.1
        
        return min(suitability, 1.0)
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行基于BM25的关键词检索"""
        import time
        start_time = time.time()
        
        try:
            # 使用关键词和查询文本进行搜索
            search_terms = query_analysis.keywords + query_analysis.query_text.split()
            search_terms = [term.lower() for term in search_terms if len(term) > 1]
            
            # 这里实现一个简化的BM25搜索
            # 在实际应用中，可以集成Elasticsearch或其他全文搜索引擎
            results = await self._perform_keyword_search(
                search_terms, limit, filters, query_analysis
            )
            
            # 计算分数
            if results:
                avg_score = sum(r.get("score", 0) for r in results) / len(results)
                confidence = min(avg_score + 0.1 * len(results) / limit, 1.0)
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.KEYWORD,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "search_terms": search_terms,
                    "total_results": len(results)
                },
                explanation=f"使用BM25关键词匹配找到{len(results)}个相关结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Keyword retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.KEYWORD,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"关键词检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result
    
    async def _perform_keyword_search(self, 
                                    search_terms: List[str], 
                                    limit: int,
                                    filters: Optional[Dict[str, Any]],
                                    query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """执行关键词搜索"""
        if not search_terms:
            return []

        if self.bm42_engine is None:
            self.bm42_engine = get_hybrid_search_engine()

        collection = "code" if query_analysis.intent_type == QueryIntent.CODE else "documents"
        search_results = await self.bm42_engine.search(
            query=query_analysis.query_text,
            collection=collection,
            limit=limit,
            filters=filters,
            strategy=SearchStrategy.BM25_ONLY
        )

        if not search_results:
            return []

        max_score = max((r.score for r in search_results), default=0.0)
        if max_score <= 0:
            max_score = 1.0

        results = []
        for result in search_results:
            normalized_score = min(result.score / max_score, 1.0)
            results.append({
                "id": str(result.id),
                "score": normalized_score,
                "bm25_score": result.score,
                "content": result.content,
                "file_path": result.file_path,
                "file_type": result.file_type,
                "chunk_index": result.chunk_index,
                "metadata": result.metadata,
            })

        return results

class StructuredRetrievalAgent(BaseRetrievalAgent):
    """结构化数据检索代理（数据库查询）"""
    
    def __init__(self):
        super().__init__("StructuredExpert", RetrievalStrategy.STRUCTURED)
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估结构化检索的适用性"""
        suitability = 0.3  # 基础适用性较低
        
        # 基于实体数量调整（结构化查询通常涉及具体实体）
        if len(query_analysis.entities) >= 2:
            suitability += 0.4
        elif len(query_analysis.entities) >= 1:
            suitability += 0.2
        
        # 基于意图类型调整
        if query_analysis.intent_type == QueryIntent.FACTUAL:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.PROCEDURAL:
            suitability += 0.1
        
        # 基于领域调整
        if query_analysis.domain in ["技术", "业务"]:
            suitability += 0.2
        
        return min(suitability, 1.0)
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行结构化数据检索"""
        import time
        start_time = time.time()
        
        try:
            # 执行数据库查询
            results = await self._perform_structured_search(
                query_analysis, limit, filters
            )
            
            # 计算分数
            if results:
                avg_score = sum(r.get("score", 0.0) for r in results) / len(results)
                confidence = min(avg_score + 0.1 * len(results) / limit, 1.0)
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.STRUCTURED,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "entities_found": len(query_analysis.entities),
                    "total_results": len(results)
                },
                explanation=f"通过结构化数据库查询找到{len(results)}个精确匹配结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Structured retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.STRUCTURED,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"结构化检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result
    
    async def _perform_structured_search(self, 
                                       query_analysis: QueryAnalysis,
                                       limit: int,
                                       filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行结构化搜索（基于数据库模型）"""
        results = []

        terms = []
        for term in query_analysis.entities + query_analysis.keywords:
            term = term.strip()
            if term:
                terms.append(term)
        if not terms:
            query_text = query_analysis.query_text.strip()
            if query_text:
                terms.append(query_text)

        terms = list(dict.fromkeys(terms))[:8]
        if not terms:
            return []

        def score_text(text: str) -> float:
            if not text:
                return 0.0
            lower_text = text.lower()
            matched = sum(1 for term in terms if term.lower() in lower_text)
            return min(matched / len(terms), 1.0) if terms else 0.0

        def build_clause(columns):
            clauses = []
            for term in terms:
                pattern = f"%{term}%"
                for col in columns:
                    clauses.append(col.ilike(pattern))
            if not clauses:
                return None
            return or_(*clauses)

        targets = [
            (
                Session,
                [Session.title, Session.user_id, Session.status],
                lambda row: (
                    f"会话: {row.title}",
                    {"model": "session", "id": str(row.id), "status": row.status, "user_id": row.user_id}
                ),
            ),
            (
                Experiment,
                [Experiment.name, Experiment.description, Experiment.hypothesis, Experiment.owner, Experiment.status],
                lambda row: (
                    f"实验: {row.name}\n{row.description}",
                    {"model": "experiment", "id": row.id, "status": row.status, "owner": row.owner}
                ),
            ),
            (
                ExperimentVariant,
                [ExperimentVariant.name, ExperimentVariant.description, ExperimentVariant.variant_id],
                lambda row: (
                    f"实验变体: {row.name}\n{row.description or ''}",
                    {"model": "experiment_variant", "id": row.id, "experiment_id": row.experiment_id, "variant_id": row.variant_id}
                ),
            ),
            (
                WorkflowModel,
                [WorkflowModel.name, WorkflowModel.description, WorkflowModel.workflow_type, WorkflowModel.status],
                lambda row: (
                    f"工作流: {row.name}\n{row.description or ''}",
                    {"model": "workflow", "id": str(row.id), "status": row.status, "workflow_type": row.workflow_type}
                ),
            ),
            (
                AuthUser,
                [AuthUser.username, AuthUser.email, AuthUser.full_name],
                lambda row: (
                    f"用户: {row.username}\n{row.full_name or ''}",
                    {"model": "user", "id": str(row.id), "email": row.email}
                ),
            ),
            (
                APIKey,
                [APIKey.name, APIKey.description, APIKey.key_prefix],
                lambda row: (
                    f"API Key: {row.name}\n{row.description or ''}",
                    {"model": "api_key", "id": str(row.id), "key_prefix": row.key_prefix}
                ),
            ),
            (
                EventStream,
                [EventStream.event_name, EventStream.event_type, EventStream.event_category, EventStream.user_id, EventStream.experiment_id, EventStream.variant_id, EventStream.session_id],
                lambda row: (
                    f"事件: {row.event_name}\n类型: {row.event_type}",
                    {
                        "model": "event_stream",
                        "id": row.id,
                        "event_type": row.event_type,
                        "event_name": row.event_name,
                        "experiment_id": row.experiment_id,
                        "variant_id": row.variant_id,
                        "user_id": row.user_id,
                    }
                ),
            ),
            (
                SupervisorAgent,
                [SupervisorAgent.name, SupervisorAgent.role, SupervisorAgent.status],
                lambda row: (
                    f"Supervisor智能体: {row.name}\n角色: {row.role}",
                    {"model": "supervisor_agent", "id": row.id, "status": row.status}
                ),
            ),
            (
                SupervisorTask,
                [SupervisorTask.name, SupervisorTask.description, SupervisorTask.task_type, SupervisorTask.status],
                lambda row: (
                    f"Supervisor任务: {row.name}\n{row.description}",
                    {"model": "supervisor_task", "id": row.id, "status": row.status, "task_type": row.task_type}
                ),
            ),
            (
                SupervisorDecision,
                [SupervisorDecision.decision_id, SupervisorDecision.task_description, SupervisorDecision.assigned_agent],
                lambda row: (
                    f"Supervisor决策: {row.decision_id}\n{row.task_description}",
                    {"model": "supervisor_decision", "id": row.id, "assigned_agent": row.assigned_agent}
                ),
            ),
        ]

        async with get_db_session() as session:
            for model, columns, formatter in targets:
                if len(results) >= limit:
                    break
                clause = build_clause(columns)
                if clause is None:
                    continue
                stmt = select(model).where(clause).limit(limit - len(results))
                query_result = await session.execute(stmt)
                rows = query_result.scalars().all()
                for row in rows:
                    content, metadata = formatter(row)
                    result_score = score_text(content)
                    results.append({
                        "id": f"{metadata.get('model')}:{metadata.get('id')}",
                        "score": result_score,
                        "content": content,
                        "metadata": metadata,
                        "source_type": "database",
                    })

        return results[:limit]

class MultiAgentRetriever:
    """多代理检索协调器"""
    
    def __init__(self):
        # 初始化各个专家代理
        self.agents = {
            RetrievalStrategy.SEMANTIC: SemanticRetrievalAgent(),
            RetrievalStrategy.KEYWORD: KeywordRetrievalAgent(),
            RetrievalStrategy.STRUCTURED: StructuredRetrievalAgent(),
        }
        
        # 策略选择权重
        self.strategy_weights = {
            RetrievalStrategy.SEMANTIC: 1.0,
            RetrievalStrategy.KEYWORD: 1.0,
            RetrievalStrategy.STRUCTURED: 0.8,
        }
    
    def select_strategies(self, query_analysis: QueryAnalysis) -> List[Tuple[RetrievalStrategy, float]]:
        """动态选择检索策略和权重"""
        strategy_scores = []
        
        for strategy, agent in self.agents.items():
            # 计算基础适用性
            suitability = agent.is_suitable_for_query(query_analysis)
            
            # 考虑代理历史性能
            performance_score = agent.performance_stats.get("success_rate", 0.5)
            
            # 综合评分
            final_score = (
                suitability * 0.7 +  # 适用性权重70%
                performance_score * 0.2 +  # 历史性能权重20%
                self.strategy_weights[strategy] * 0.1  # 全局权重10%
            )
            
            strategy_scores.append((strategy, final_score))
        
        # 排序并返回
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 根据分数阈值过滤
        filtered_strategies = [(s, score) for s, score in strategy_scores if score > 0.3]
        
        # 至少保留一个策略
        if not filtered_strategies:
            filtered_strategies = [strategy_scores[0]]
        
        return filtered_strategies
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      strategies: Optional[List[RetrievalStrategy]] = None,
                      enable_parallel: bool = True,
                      filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """执行多代理协作检索"""
        
        # 策略选择
        if strategies is None:
            selected_strategies = self.select_strategies(query_analysis)
            strategies = [s for s, _ in selected_strategies]
        else:
            expanded_strategies = []
            for strategy in strategies:
                if strategy == RetrievalStrategy.HYBRID:
                    if _openai_key_configured():
                        expanded_strategies.append(RetrievalStrategy.SEMANTIC)
                    expanded_strategies.append(RetrievalStrategy.KEYWORD)
                    expanded_strategies.append(RetrievalStrategy.STRUCTURED)
                else:
                    expanded_strategies.append(strategy)

            if not _openai_key_configured():
                expanded_strategies = [
                    s for s in expanded_strategies if s != RetrievalStrategy.SEMANTIC
                ]

            seen = set()
            strategies = []
            for strategy in expanded_strategies:
                if strategy not in seen:
                    seen.add(strategy)
                    strategies.append(strategy)

            selected_strategies = [(s, 1.0) for s in strategies]
        
        # 执行检索
        if enable_parallel:
            # 并行执行
            tasks = []
            for strategy in strategies:
                if strategy in self.agents:
                    agent = self.agents[strategy]
                    task = agent.retrieve(query_analysis, limit, filters)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤异常结果
            valid_results = []
            for result in results:
                if isinstance(result, RetrievalResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Agent retrieval failed: {result}")
            
            return valid_results
            
        else:
            # 串行执行
            results = []
            for strategy in strategies:
                if strategy in self.agents:
                    agent = self.agents[strategy]
                    try:
                        result = await agent.retrieve(query_analysis, limit, filters)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Agent {strategy} failed: {e}")
                        continue
            
            return results
    
    def fuse_results(self, 
                    results: List[RetrievalResult],
                    fusion_method: str = "weighted_score",
                    max_results: int = 20) -> List[Dict[str, Any]]:
        """融合多个检索结果"""
        
        if not results:
            return []
        
        if fusion_method == "weighted_score":
            return self._weighted_score_fusion(results, max_results)
        elif fusion_method == "rank_fusion":
            return self._rank_fusion(results, max_results)
        elif fusion_method == "confidence_weighted":
            return self._confidence_weighted_fusion(results, max_results)
        else:
            # 默认使用加权分数融合
            return self._weighted_score_fusion(results, max_results)
    
    def _weighted_score_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于加权分数的结果融合"""
        all_items = []
        
        for result in results:
            agent_weight = result.confidence  # 使用置信度作为代理权重
            
            for item in result.results:
                # 计算融合分数
                original_score = item.get("score", 0.0)
                fused_score = original_score * agent_weight
                
                fused_item = item.copy()
                fused_item["fused_score"] = fused_score
                fused_item["agent_type"] = result.agent_type.value
                fused_item["agent_confidence"] = result.confidence
                
                all_items.append(fused_item)
        
        # 去重（基于content或id）
        unique_items = self._deduplicate_results(all_items)
        
        # 按融合分数排序
        unique_items.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return unique_items[:max_results]
    
    def _rank_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于排名的结果融合（Reciprocal Rank Fusion）"""
        item_scores = defaultdict(float)
        item_data = {}
        
        for result in results:
            for rank, item in enumerate(result.results):
                item_id = item.get("id", item.get("content", str(rank)))
                
                # RRF分数计算
                rrf_score = 1.0 / (rank + 60)  # 60是常用的k值
                item_scores[item_id] += rrf_score * result.confidence
                
                # 保存物品数据（第一次遇到时）
                if item_id not in item_data:
                    fused_item = item.copy()
                    fused_item["agent_type"] = result.agent_type.value
                    fused_item["agent_confidence"] = result.confidence
                    item_data[item_id] = fused_item
        
        # 排序并构建最终结果
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for item_id, score in sorted_items[:max_results]:
            item = item_data[item_id]
            item["fused_score"] = score
            fused_results.append(item)
        
        return fused_results
    
    def _confidence_weighted_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于置信度加权的结果融合"""
        # 计算总置信度用于归一化
        total_confidence = sum(result.confidence for result in results)
        if total_confidence == 0:
            return []
        
        all_items = []
        
        for result in results:
            normalized_weight = result.confidence / total_confidence
            
            for item in result.results:
                fused_item = item.copy()
                fused_item["fused_score"] = item.get("score", 0.0) * normalized_weight
                fused_item["agent_type"] = result.agent_type.value
                fused_item["agent_confidence"] = result.confidence
                fused_item["normalized_weight"] = normalized_weight
                
                all_items.append(fused_item)
        
        # 去重和排序
        unique_items = self._deduplicate_results(all_items)
        unique_items.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return unique_items[:max_results]
    
    def _deduplicate_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """结果去重"""
        seen = set()
        unique_items = []
        
        for item in items:
            # 使用多个字段组合作为唯一标识
            identifier = (
                item.get("id", ""),
                item.get("content", "")[:100],  # 使用内容前100字符
                item.get("file_path", "")
            )
            
            if identifier not in seen:
                seen.add(identifier)
                unique_items.append(item)
        
        return unique_items
    
    def get_retrieval_explanation(self, 
                                query_analysis: QueryAnalysis,
                                results: List[RetrievalResult],
                                fused_results: List[Dict[str, Any]]) -> str:
        """生成检索过程解释"""
        explanation_parts = []
        
        # 查询分析摘要
        explanation_parts.append(f"查询分析: 意图={query_analysis.intent_type.value}, 复杂度={query_analysis.complexity_score:.2f}")
        
        # 策略使用情况
        strategies_used = [r.agent_type.value for r in results]
        explanation_parts.append(f"使用策略: {', '.join(strategies_used)}")
        
        # 各代理结果摘要
        for result in results:
            explanation_parts.append(
                f"{result.agent_type.value}: {len(result.results)}个结果, "
                f"置信度={result.confidence:.2f}, 用时={result.processing_time:.3f}s"
            )
        
        # 融合结果摘要
        explanation_parts.append(f"最终融合: {len(fused_results)}个结果")
        
        return " | ".join(explanation_parts)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for strategy, agent in self.agents.items():
            summary[strategy.value] = {
                "name": agent.name,
                "stats": agent.performance_stats.copy()
            }
        
        return summary

# 全局多代理检索器实例
multi_agent_retriever = MultiAgentRetriever()
