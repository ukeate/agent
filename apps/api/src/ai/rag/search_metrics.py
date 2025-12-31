"""搜索质量监控和评估模块"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from src.ai.rag.hybrid_search import HybridSearchEngine, SearchResult, SearchStrategy
from src.core.config import get_settings

logger = get_logger(__name__)

@dataclass
class SearchMetrics:
    """搜索质量指标"""
    query_count: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    search_strategy_distribution: Dict[str, int] = None
    quality_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.search_strategy_distribution is None:
            self.search_strategy_distribution = {}
        if self.quality_scores is None:
            self.quality_scores = {}

@dataclass
class BenchmarkQuery:
    """基准测试查询"""
    query: str
    language: str
    expected_results: List[str]  # 期望的文档ID列表
    relevance_scores: Dict[str, float]  # 相关性分数
    category: str  # 查询类别

class SearchQualityEvaluator:
    """搜索质量评估器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.metrics_collector = MetricsCollector()
        self.benchmark_queries = self._load_benchmark_queries()
    
    def _load_benchmark_queries(self) -> List[BenchmarkQuery]:
        """加载基准测试查询"""
        # 简化的基准查询集合
        return [
            BenchmarkQuery(
                query="Python函数实现",
                language="zh",
                expected_results=["python_func_1", "python_func_2"],
                relevance_scores={"python_func_1": 1.0, "python_func_2": 0.8},
                category="code"
            ),
            BenchmarkQuery(
                query="How to implement authentication",
                language="en",
                expected_results=["auth_doc_1", "auth_doc_2"],
                relevance_scores={"auth_doc_1": 1.0, "auth_doc_2": 0.9},
                category="documentation"
            ),
            BenchmarkQuery(
                query="错误处理最佳实践",
                language="zh",
                expected_results=["error_handle_1", "error_handle_2"],
                relevance_scores={"error_handle_1": 1.0, "error_handle_2": 0.7},
                category="best_practices"
            ),
        ]
    
    async def evaluate_search_quality(
        self, 
        search_engine: HybridSearchEngine,
        strategies: List[SearchStrategy] = None
    ) -> Dict[str, float]:
        """评估搜索质量"""
        if strategies is None:
            strategies = [
                SearchStrategy.VECTOR_ONLY,
                SearchStrategy.BM25_ONLY,
                SearchStrategy.HYBRID_RRF,
                SearchStrategy.HYBRID_WEIGHTED,
                SearchStrategy.ADAPTIVE
            ]
        
        results = {}
        
        for strategy in strategies:
            strategy_results = {}
            
            for k in [5, 10, 20]:
                precision_scores = []
                recall_scores = []
                ndcg_scores = []
                
                for query_set in self.benchmark_queries:
                    try:
                        # 执行搜索
                        search_results = await search_engine.search(
                            query=query_set.query,
                            collection="documents",
                            limit=k,
                            strategy=strategy
                        )
                        
                        # 计算评估指标
                        precision = self._calculate_precision_at_k(search_results, query_set, k)
                        recall = self._calculate_recall_at_k(search_results, query_set, k)
                        ndcg = self._calculate_ndcg(search_results, query_set, k)
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        ndcg_scores.append(ndcg)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating query {query_set.query}: {e}")
                        precision_scores.append(0.0)
                        recall_scores.append(0.0)
                        ndcg_scores.append(0.0)
                
                # 计算平均分数
                strategy_results.update({
                    f"precision@{k}": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
                    f"recall@{k}": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
                    f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
                })
            
            results[strategy.value] = strategy_results
        
        return results
    
    def _calculate_precision_at_k(
        self, 
        search_results: List[SearchResult], 
        benchmark: BenchmarkQuery, 
        k: int
    ) -> float:
        """计算Precision@K"""
        if not search_results or k <= 0:
            return 0.0
        
        relevant_found = 0
        for i, result in enumerate(search_results[:k]):
            if result.id in benchmark.expected_results:
                relevant_found += 1
        
        return relevant_found / min(k, len(search_results))
    
    def _calculate_recall_at_k(
        self, 
        search_results: List[SearchResult], 
        benchmark: BenchmarkQuery, 
        k: int
    ) -> float:
        """计算Recall@K"""
        if not benchmark.expected_results:
            return 0.0
        
        relevant_found = 0
        for result in search_results[:k]:
            if result.id in benchmark.expected_results:
                relevant_found += 1
        
        return relevant_found / len(benchmark.expected_results)
    
    def _calculate_ndcg(
        self, 
        search_results: List[SearchResult], 
        benchmark: BenchmarkQuery, 
        k: int
    ) -> float:
        """计算NDCG@K"""
        if not search_results or not benchmark.relevance_scores:
            return 0.0
        
        # 计算DCG
        dcg = 0.0
        for i, result in enumerate(search_results[:k]):
            relevance = benchmark.relevance_scores.get(result.id, 0.0)
            if relevance > 0:
                dcg += relevance / (1 + i)  # 简化的DCG计算
        
        # 计算IDCG（理想DCG）
        sorted_relevances = sorted(benchmark.relevance_scores.values(), reverse=True)
        idcg = sum(rel / (1 + i) for i, rel in enumerate(sorted_relevances[:k]))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    async def benchmark_performance(
        self, 
        search_engine: HybridSearchEngine,
        num_queries: int = 100
    ) -> Dict[str, Any]:
        """性能基准测试"""
        strategies = [
            SearchStrategy.VECTOR_ONLY,
            SearchStrategy.BM25_ONLY,
            SearchStrategy.HYBRID_RRF,
            SearchStrategy.HYBRID_WEIGHTED
        ]
        
        results = {}
        
        for strategy in strategies:
            response_times = []
            success_count = 0
            
            for i in range(num_queries):
                # 使用基准查询或生成测试查询
                query = self.benchmark_queries[i % len(self.benchmark_queries)].query
                
                start_time = time.time()
                try:
                    await search_engine.search(
                        query=query,
                        collection="documents",
                        limit=10,
                        strategy=strategy
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(f"Search failed for strategy {strategy}: {e}")
                
                end_time = time.time()
                response_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # 计算统计信息
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))] if response_times else 0
            success_rate = success_count / num_queries
            
            results[strategy.value] = {
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time,
                "success_rate": success_rate,
                "total_queries": num_queries
            }
        
        return results

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.query_metrics = defaultdict(list)
        self.response_times = []
        self.strategy_usage = defaultdict(int)
        self.cache_stats = {"hits": 0, "misses": 0}
    
    def record_query(
        self, 
        query: str, 
        strategy: SearchStrategy, 
        response_time: float,
        results_count: int,
        cache_hit: bool = False
    ):
        """记录查询指标"""
        self.query_metrics[strategy.value].append({
            "query": query,
            "response_time": response_time,
            "results_count": results_count,
            "timestamp": time.time()
        })
        
        self.response_times.append(response_time)
        self.strategy_usage[strategy.value] += 1
        
        if cache_hit:
            self.cache_stats["hits"] += 1
        else:
            self.cache_stats["misses"] += 1
    
    def get_metrics_summary(self) -> SearchMetrics:
        """获取指标摘要"""
        total_queries = sum(self.strategy_usage.values())
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        cache_total = self.cache_stats["hits"] + self.cache_stats["misses"]
        cache_hit_rate = self.cache_stats["hits"] / cache_total if cache_total > 0 else 0
        
        # 计算各策略使用分布
        strategy_distribution = {}
        for strategy, count in self.strategy_usage.items():
            strategy_distribution[strategy] = count / total_queries if total_queries > 0 else 0
        
        # 计算质量分数（简化）
        quality_scores = {}
        for strategy, metrics_list in self.query_metrics.items():
            if metrics_list:
                avg_results = sum(m["results_count"] for m in metrics_list) / len(metrics_list)
                quality_scores[f"{strategy}_avg_results"] = avg_results
        
        return SearchMetrics(
            query_count=total_queries,
            avg_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            search_strategy_distribution=strategy_distribution,
            quality_scores=quality_scores
        )
    
    def reset_metrics(self):
        """重置指标"""
        self.query_metrics.clear()
        self.response_times.clear()
        self.strategy_usage.clear()
        self.cache_stats = {"hits": 0, "misses": 0}

class ABTestFramework:
    """A/B测试框架"""
    
    def __init__(self):
        self.experiments = {}
        self.results = {}
    
    def create_experiment(
        self, 
        name: str, 
        control_strategy: SearchStrategy,
        treatment_strategy: SearchStrategy,
        traffic_split: float = 0.5
    ):
        """创建A/B测试实验"""
        self.experiments[name] = {
            "control": control_strategy,
            "treatment": treatment_strategy,
            "traffic_split": traffic_split,
            "control_metrics": MetricsCollector(),
            "treatment_metrics": MetricsCollector(),
            "start_time": time.time()
        }
    
    def assign_strategy(self, experiment_name: str, user_id: str) -> SearchStrategy:
        """为用户分配搜索策略"""
        if experiment_name not in self.experiments:
            return SearchStrategy.HYBRID_RRF  # 默认策略
        
        experiment = self.experiments[experiment_name]
        
        # 简单的哈希分配
        user_hash = hash(user_id) % 100
        
        if user_hash < experiment["traffic_split"] * 100:
            return experiment["treatment"]
        else:
            return experiment["control"]
    
    def record_experiment_result(
        self, 
        experiment_name: str, 
        strategy: SearchStrategy,
        query: str,
        response_time: float,
        results_count: int
    ):
        """记录实验结果"""
        if experiment_name not in self.experiments:
            return
        
        experiment = self.experiments[experiment_name]
        
        if strategy == experiment["control"]:
            experiment["control_metrics"].record_query(
                query, strategy, response_time, results_count
            )
        elif strategy == experiment["treatment"]:
            experiment["treatment_metrics"].record_query(
                query, strategy, response_time, results_count
            )
    
    def get_experiment_results(self, experiment_name: str) -> Dict[str, Any]:
        """获取实验结果"""
        if experiment_name not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_name]
        
        control_metrics = experiment["control_metrics"].get_metrics_summary()
        treatment_metrics = experiment["treatment_metrics"].get_metrics_summary()
        
        return {
            "experiment_name": experiment_name,
            "control": {
                "strategy": experiment["control"].value,
                "metrics": control_metrics
            },
            "treatment": {
                "strategy": experiment["treatment"].value,
                "metrics": treatment_metrics
            },
            "duration_seconds": time.time() - experiment["start_time"]
        }
    
    def analyze_significance(self, experiment_name: str) -> Dict[str, Any]:
        """分析实验结果的统计显著性（简化版）"""
        results = self.get_experiment_results(experiment_name)
        
        if not results:
            return {}
        
        control_metrics = results["control"]["metrics"]
        treatment_metrics = results["treatment"]["metrics"]
        
        # 简化的统计分析
        response_time_improvement = (
            (control_metrics.avg_response_time - treatment_metrics.avg_response_time) 
            / control_metrics.avg_response_time * 100
            if control_metrics.avg_response_time > 0 else 0
        )
        
        cache_hit_improvement = (
            treatment_metrics.cache_hit_rate - control_metrics.cache_hit_rate
        ) * 100
        
        return {
            "response_time_improvement_pct": response_time_improvement,
            "cache_hit_improvement_pct": cache_hit_improvement,
            "control_sample_size": control_metrics.query_count,
            "treatment_sample_size": treatment_metrics.query_count,
            "statistical_significance": "需要更复杂的统计检验"  # 简化提示
        }

# 全局实例
global_metrics_collector = MetricsCollector()
global_ab_framework = ABTestFramework()

def get_search_evaluator() -> SearchQualityEvaluator:
    """获取搜索质量评估器"""
    return SearchQualityEvaluator()

def get_ab_framework() -> ABTestFramework:
    """获取A/B测试框架"""
    return global_ab_framework

def get_metrics_collector() -> MetricsCollector:
    """获取指标收集器"""
    return global_metrics_collector
from src.core.logging import get_logger
