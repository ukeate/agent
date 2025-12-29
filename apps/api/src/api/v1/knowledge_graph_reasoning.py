"""
知识图推理引擎API端点

提供统一的推理服务接口，支持多种推理策略和方法
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import Field
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from src.ai.knowledge_graph.hybrid_reasoner import (
    HybridReasoner, ReasoningRequest, HybridReasoningResult,
    ReasoningStrategy, ConfidenceWeights
)
from src.api.base_model import ApiBaseModel
from src.ai.knowledge_graph.rule_engine import RuleEngine, RuleStatus
from src.ai.knowledge_graph.embedding_engine import EmbeddingConfig, EmbeddingEngine, EmbeddingModel
from src.ai.knowledge_graph.path_reasoning import PathReasoner
from src.ai.knowledge_graph.uncertainty_reasoning import UncertaintyReasoner

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/kg-reasoning", tags=["知识图推理"])

# 全局推理引擎实例
_hybrid_reasoner: Optional[HybridReasoner] = None

# Pydantic模型定义
class ReasoningQueryRequest(ApiBaseModel):
    """推理查询请求"""
    query: str = Field(..., description="推理查询")
    query_type: str = Field("general", description="查询类型")
    entities: List[str] = Field(default_factory=list, description="相关实体")
    relations: List[str] = Field(default_factory=list, description="相关关系")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="约束条件")
    strategy: str = Field("adaptive", description="推理策略")
    max_depth: int = Field(3, ge=1, le=10, description="最大推理深度")
    top_k: int = Field(10, ge=1, le=100, description="返回结果数量")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="置信度阈值")
    timeout: int = Field(30, ge=5, le=300, description="超时时间（秒）")
    context: Dict[str, Any] = Field(default_factory=dict, description="上下文信息")

class ReasoningQueryResponse(ApiBaseModel):
    """推理查询响应"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    confidence: float
    strategy_used: str
    execution_time: float
    method_contributions: Dict[str, float]
    evidences_count: int
    explanation: str
    uncertainty_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BatchReasoningRequest(ApiBaseModel):
    """批量推理请求"""
    queries: List[ReasoningQueryRequest] = Field(..., max_items=50, description="推理查询列表")
    parallel: bool = Field(True, description="是否并行处理")
    priority: str = Field("medium", description="处理优先级")

class BatchReasoningResponse(ApiBaseModel):
    """批量推理响应"""
    success: bool
    total_queries: int
    successful_queries: int
    failed_queries: int
    results: List[ReasoningQueryResponse]
    total_execution_time: float

class StrategyPerformanceResponse(ApiBaseModel):
    """策略性能响应"""
    success: bool
    strategies: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]

class ReasoningConfigRequest(ApiBaseModel):
    """推理配置请求"""
    confidence_weights: Optional[Dict[str, float]] = None
    adaptive_thresholds: Optional[Dict[str, float]] = None
    cache_settings: Optional[Dict[str, Any]] = None

class RuleCreateRequest(ApiBaseModel):
    """创建规则请求"""
    name: Optional[str] = Field(default=None, description="规则名称")
    rule_text: str = Field(..., description="规则文本")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="置信度")
    priority: int = Field(1, ge=1, le=100, description="优先级")

class RuleUpdateRequest(ApiBaseModel):
    """更新规则请求"""
    name: Optional[str] = Field(default=None, description="规则名称")
    rule_text: Optional[str] = Field(default=None, description="规则文本")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="置信度")
    priority: Optional[int] = Field(default=None, ge=1, le=100, description="优先级")
    status: Optional[str] = Field(default=None, description="规则状态")

class RuleTestRequest(ApiBaseModel):
    """规则测试请求"""
    facts: List[str] = Field(default_factory=list, description="事实列表")

def _serialize_rule(rule) -> Dict[str, Any]:
    return {
        "id": rule.id,
        "name": rule.name,
        "rule_text": rule.rule_text,
        "confidence": rule.confidence,
        "priority": rule.priority,
        "status": rule.status.value if isinstance(rule.status, RuleStatus) else str(rule.status),
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
        "execution_count": rule.execution_count,
        "success_count": rule.success_count,
        "success_rate": rule.success_rate,
        "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
    }

async def get_hybrid_reasoner() -> HybridReasoner:
    """获取混合推理引擎实例"""
    global _hybrid_reasoner
    
    if _hybrid_reasoner is None:
        try:
            # 初始化推理引擎组件
            rule_engine = RuleEngine()
            embedding_engine = EmbeddingEngine(
                EmbeddingConfig(model_type=EmbeddingModel.TRANSE, embedding_dim=256)
            )
            path_reasoner = PathReasoner()
            uncertainty_reasoner = UncertaintyReasoner()
            
            # 创建混合推理引擎
            _hybrid_reasoner = HybridReasoner(
                rule_engine=rule_engine,
                embedding_engine=embedding_engine,
                path_reasoner=path_reasoner,
                uncertainty_reasoner=uncertainty_reasoner
            )
            
            logger.info("混合推理引擎实例创建成功")
            
        except Exception as e:
            logger.error(f"创建混合推理引擎失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"推理引擎初始化失败: {str(e)}")
    
    return _hybrid_reasoner

def _convert_strategy_string(strategy_str: str) -> ReasoningStrategy:
    """转换策略字符串为枚举"""
    strategy_map = {
        "rule_only": ReasoningStrategy.RULE_ONLY,
        "embedding_only": ReasoningStrategy.EMBEDDING_ONLY,
        "path_only": ReasoningStrategy.PATH_ONLY,
        "uncertainty_only": ReasoningStrategy.UNCERTAINTY_ONLY,
        "ensemble": ReasoningStrategy.ENSEMBLE,
        "adaptive": ReasoningStrategy.ADAPTIVE,
        "cascading": ReasoningStrategy.CASCADING,
        "voting": ReasoningStrategy.VOTING
    }
    
    return strategy_map.get(strategy_str.lower(), ReasoningStrategy.ADAPTIVE)

@router.post("/query", response_model=ReasoningQueryResponse)
async def query_reasoning(
    request: ReasoningQueryRequest,
    reasoner: HybridReasoner = Depends(get_hybrid_reasoner)
):
    """
    执行推理查询
    
    支持多种推理策略：
    - rule_only: 仅规则推理
    - embedding_only: 仅嵌入推理
    - path_only: 仅路径推理
    - uncertainty_only: 仅不确定性推理
    - ensemble: 集成所有方法
    - adaptive: 自适应策略选择
    - cascading: 级联推理
    - voting: 投票机制
    """
    try:
        start_time = asyncio.get_running_loop().time()
        
        # 转换请求格式
        reasoning_request = ReasoningRequest(
            query=request.query,
            query_type=request.query_type,
            entities=request.entities,
            relations=request.relations,
            constraints=request.constraints,
            strategy=_convert_strategy_string(request.strategy),
            max_depth=request.max_depth,
            top_k=request.top_k,
            confidence_threshold=request.confidence_threshold,
            timeout=request.timeout,
            metadata=request.context
        )
        
        # 执行推理
        result = await reasoner.reason(reasoning_request)
        
        # 转换响应格式
        response = ReasoningQueryResponse(
            success=True,
            query=result.query,
            results=result.results,
            confidence=result.confidence,
            strategy_used=result.strategy_used.value,
            execution_time=result.execution_time,
            method_contributions=result.method_contributions,
            evidences_count=len(result.evidences),
            explanation=result.explanation,
            uncertainty_analysis=result.uncertainty_analysis.__dict__ if result.uncertainty_analysis else None,
            metadata=result.metadata
        )
        
        logger.info(f"推理查询完成: {request.query[:50]}..., 置信度: {result.confidence:.3f}, 耗时: {result.execution_time:.3f}s")
        
        return response
        
    except asyncio.TimeoutError:
        logger.warning(f"推理查询超时: {request.query[:50]}...")
        raise HTTPException(status_code=408, detail="推理查询超时")
    
    except Exception as e:
        logger.error(f"推理查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理查询失败: {str(e)}")

@router.post("/batch", response_model=BatchReasoningResponse)
async def batch_reasoning(
    request: BatchReasoningRequest,
    background_tasks: BackgroundTasks,
    reasoner: HybridReasoner = Depends(get_hybrid_reasoner)
):
    """
    批量推理查询
    
    支持并行和串行处理模式
    """
    try:
        start_time = asyncio.get_running_loop().time()
        
        if not request.queries:
            raise HTTPException(status_code=400, detail="查询列表不能为空")
        
        results = []
        successful_queries = 0
        failed_queries = 0
        
        if request.parallel:
            # 并行处理
            tasks = []
            for query_req in request.queries:
                reasoning_request = ReasoningRequest(
                    query=query_req.query,
                    query_type=query_req.query_type,
                    entities=query_req.entities,
                    relations=query_req.relations,
                    constraints=query_req.constraints,
                    strategy=_convert_strategy_string(query_req.strategy),
                    max_depth=query_req.max_depth,
                    top_k=query_req.top_k,
                    confidence_threshold=query_req.confidence_threshold,
                    timeout=query_req.timeout,
                    metadata=query_req.context
                )
                tasks.append(reasoner.reason(reasoning_request))
            
            # 等待所有任务完成
            reasoning_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(reasoning_results):
                if isinstance(result, Exception):
                    failed_queries += 1
                    results.append(ReasoningQueryResponse(
                        success=False,
                        query=request.queries[i].query,
                        results=[],
                        confidence=0.0,
                        strategy_used="none",
                        execution_time=0.0,
                        method_contributions={},
                        evidences_count=0,
                        explanation=f"推理失败: {str(result)}"
                    ))
                else:
                    successful_queries += 1
                    results.append(ReasoningQueryResponse(
                        success=True,
                        query=result.query,
                        results=result.results,
                        confidence=result.confidence,
                        strategy_used=result.strategy_used.value,
                        execution_time=result.execution_time,
                        method_contributions=result.method_contributions,
                        evidences_count=len(result.evidences),
                        explanation=result.explanation,
                        uncertainty_analysis=result.uncertainty_analysis.__dict__ if result.uncertainty_analysis else None,
                        metadata=result.metadata
                    ))
        else:
            # 串行处理
            for query_req in request.queries:
                try:
                    reasoning_request = ReasoningRequest(
                        query=query_req.query,
                        query_type=query_req.query_type,
                        entities=query_req.entities,
                        relations=query_req.relations,
                        constraints=query_req.constraints,
                        strategy=_convert_strategy_string(query_req.strategy),
                        max_depth=query_req.max_depth,
                        top_k=query_req.top_k,
                        confidence_threshold=query_req.confidence_threshold,
                        timeout=query_req.timeout,
                        metadata=query_req.context
                    )
                    
                    result = await reasoner.reason(reasoning_request)
                    successful_queries += 1
                    
                    results.append(ReasoningQueryResponse(
                        success=True,
                        query=result.query,
                        results=result.results,
                        confidence=result.confidence,
                        strategy_used=result.strategy_used.value,
                        execution_time=result.execution_time,
                        method_contributions=result.method_contributions,
                        evidences_count=len(result.evidences),
                        explanation=result.explanation,
                        uncertainty_analysis=result.uncertainty_analysis.__dict__ if result.uncertainty_analysis else None,
                        metadata=result.metadata
                    ))
                    
                except Exception as e:
                    failed_queries += 1
                    results.append(ReasoningQueryResponse(
                        success=False,
                        query=query_req.query,
                        results=[],
                        confidence=0.0,
                        strategy_used="none",
                        execution_time=0.0,
                        method_contributions={},
                        evidences_count=0,
                        explanation=f"推理失败: {str(e)}"
                    ))
        
        total_execution_time = asyncio.get_running_loop().time() - start_time
        
        response = BatchReasoningResponse(
            success=True,
            total_queries=len(request.queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            results=results,
            total_execution_time=total_execution_time
        )
        
        logger.info(f"批量推理完成: 总计{len(request.queries)}个查询, 成功{successful_queries}个, 失败{failed_queries}个, 耗时{total_execution_time:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"批量推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")

@router.get("/strategies/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    reasoner: HybridReasoner = Depends(get_hybrid_reasoner)
):
    """获取推理策略性能统计"""
    try:
        stats = await reasoner.get_strategy_performance_stats()
        
        # 计算总体统计
        summary = {
            "total_strategies": len(stats),
            "total_queries": sum(s["total_queries"] for s in stats.values()),
            "avg_success_rate": sum(s["success_rate"] for s in stats.values()) / max(len(stats), 1),
            "avg_confidence": sum(s["avg_confidence"] for s in stats.values()) / max(len(stats), 1),
            "avg_execution_time": sum(s["avg_execution_time"] for s in stats.values()) / max(len(stats), 1)
        }
        
        return StrategyPerformanceResponse(
            success=True,
            strategies=stats,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"获取策略性能失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取策略性能失败: {str(e)}")

@router.post("/config")
async def update_reasoning_config(
    request: ReasoningConfigRequest,
    reasoner: HybridReasoner = Depends(get_hybrid_reasoner)
):
    """更新推理配置"""
    try:
        updated_configs = []
        
        if request.confidence_weights:
            await reasoner.update_confidence_weights(request.confidence_weights)
            updated_configs.append("confidence_weights")
        
        if request.adaptive_thresholds:
            reasoner.adaptive_thresholds.update(request.adaptive_thresholds)
            updated_configs.append("adaptive_thresholds")
        
        if request.cache_settings:
            # 这里可以更新优化器的缓存设置
            updated_configs.append("cache_settings")
        
        logger.info(f"推理配置已更新: {updated_configs}")
        
        return {
            "success": True,
            "message": f"配置已更新: {', '.join(updated_configs)}",
            "updated_configs": updated_configs
        }
        
    except Exception as e:
        logger.error(f"更新推理配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新推理配置失败: {str(e)}")

@router.post("/explain")
async def explain_reasoning_result(
    result_data: Dict[str, Any],
    reasoner: HybridReasoner = Depends(get_hybrid_reasoner)
):
    """生成推理结果解释"""
    try:
        # 重建推理结果对象
        result = HybridReasoningResult(
            query=result_data.get("query", ""),
            results=result_data.get("results", []),
            confidence=result_data.get("confidence", 0.0),
            evidences=[],  # 简化处理
            strategy_used=ReasoningStrategy(result_data.get("strategy_used", "adaptive")),
            execution_time=result_data.get("execution_time", 0.0),
            method_contributions=result_data.get("method_contributions", {}),
            explanation=result_data.get("explanation", "")
        )
        
        explanation = await reasoner.explain_reasoning(result)
        
        return {
            "success": True,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"生成推理解释失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成推理解释失败: {str(e)}")

@router.get("/rules")
async def list_rules(
    status: Optional[str] = Query(None, description="规则状态过滤")
):
    """获取规则列表"""
    try:
        reasoner = await get_hybrid_reasoner()
        status_enum = None
        if status:
            try:
                status_enum = RuleStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail="无效的规则状态")
        rules = reasoner.rule_engine.list_rules(status_enum)
        return {"total": len(rules), "rules": [_serialize_rule(rule) for rule in rules]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取规则列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取规则列表失败: {str(e)}")

@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """获取规则详情"""
    try:
        reasoner = await get_hybrid_reasoner()
        rule = reasoner.rule_engine.get_rule_by_id(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="规则不存在")
        return _serialize_rule(rule)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取规则失败: {str(e)}")

@router.post("/rules")
async def create_rule(request: RuleCreateRequest):
    """创建新规则"""
    try:
        reasoner = await get_hybrid_reasoner()
        rule = await reasoner.rule_engine.add_rule(
            rule_text=request.rule_text,
            name=request.name,
            confidence=request.confidence,
            priority=request.priority
        )
        return _serialize_rule(rule)
    except Exception as e:
        logger.error(f"创建规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建规则失败: {str(e)}")

@router.patch("/rules/{rule_id}")
async def update_rule(rule_id: str, request: RuleUpdateRequest):
    """更新规则"""
    try:
        reasoner = await get_hybrid_reasoner()
        rule = reasoner.rule_engine.get_rule_by_id(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="规则不存在")

        if request.rule_text is not None:
            conditions, conclusions = reasoner.rule_engine.parser.parse_rule(request.rule_text)
            reasoner.rule_engine._validate_rule(conditions, conclusions)
            rule.rule_text = request.rule_text
            rule.conditions = conditions
            rule.conclusions = conclusions

        if request.name is not None:
            rule.name = request.name
        if request.confidence is not None:
            rule.confidence = request.confidence
        if request.priority is not None:
            rule.priority = request.priority
        if request.status is not None:
            try:
                rule.status = RuleStatus(request.status)
            except ValueError:
                raise HTTPException(status_code=400, detail="无效的规则状态")

        rule.updated_at = utc_now()
        return _serialize_rule(rule)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新规则失败: {str(e)}")

@router.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """删除规则"""
    try:
        reasoner = await get_hybrid_reasoner()
        success = await reasoner.rule_engine.remove_rule(rule_id)
        if not success:
            raise HTTPException(status_code=404, detail="规则不存在")
        return {"success": True, "rule_id": rule_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除规则失败: {str(e)}")

@router.post("/rules/{rule_id}/test")
async def test_rule(rule_id: str, request: RuleTestRequest):
    """测试规则"""
    try:
        reasoner = await get_hybrid_reasoner()
        results = await reasoner.rule_engine.test_rule(rule_id, request.facts)
        return {
            "success": True,
            "inferences": [
                {
                    "fact": r.fact,
                    "confidence": r.confidence,
                    "source_rules": r.source_rules,
                    "derivation_path": r.derivation_path,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in results
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"测试规则失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"测试规则失败: {str(e)}")

@router.get("/health")
async def health_check():
    """健康检查"""
    try:
        global _hybrid_reasoner
        
        health_status = {
            "status": "healthy",
            "reasoner_initialized": _hybrid_reasoner is not None,
            "timestamp": utc_now().isoformat()
        }
        
        if _hybrid_reasoner:
            # 执行简单的推理测试
            test_request = ReasoningRequest(
                query="test query",
                query_type="health_check",
                entities=["test_entity"],
                strategy=ReasoningStrategy.RULE_ONLY,
                timeout=5
            )
            
            try:
                result = await _hybrid_reasoner.reason(test_request)
                health_status["test_reasoning"] = "passed"
                health_status["test_confidence"] = result.confidence
            except Exception as e:
                health_status["test_reasoning"] = "failed"
                health_status["test_error"] = str(e)
        
        return health_status
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康检查失败: {str(e)}")

# 路由元数据
router.tags = ["知识图推理"]
router.responses = {
    400: {"description": "请求参数错误"},
    408: {"description": "请求超时"},
    500: {"description": "服务器内部错误"}
}
