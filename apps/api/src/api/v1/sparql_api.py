"""
SPARQL查询API接口

提供标准的SPARQL查询和更新API：
- SPARQL SELECT/CONSTRUCT/ASK/DESCRIBE查询
- SPARQL UPDATE操作
- 查询计划分析和优化建议
- 多种结果格式支持
"""

import asyncio
import json
import io
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import Field, field_validator
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel
from src.ai.knowledge_graph.sparql_engine import (
    ResultFormat,
    default_formatter,
    format_sparql_results,
)
from src.ai.knowledge_graph.query_optimizer import (
    default_optimizer,
    OptimizationLevel,
    optimize_sparql_query
)
from src.ai.knowledge_graph.performance_monitor import default_performance_monitor

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/kg/sparql", tags=["SPARQL Query API"])

_SPARQL_HISTORY_KEY = "sparql:history"

async def _push_query_history(entry: Dict[str, Any]) -> None:
    redis_client = get_redis()
    if not redis_client:
        return
    await redis_client.lpush(_SPARQL_HISTORY_KEY, json.dumps(entry, ensure_ascii=False))
    await redis_client.ltrim(_SPARQL_HISTORY_KEY, 0, 999)

class SPARQLQueryRequest(ApiBaseModel):
    """SPARQL查询请求"""
    query: str = Field(..., description="SPARQL查询语句")
    default_graph_uri: Optional[str] = Field(None, description="默认图URI")
    named_graph_uri: Optional[List[str]] = Field(None, description="命名图URI列表")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="超时时间（秒）")
    format: Optional[ResultFormat] = Field(ResultFormat.JSON, description="结果格式")
    use_cache: Optional[bool] = Field(True, description="是否使用缓存")
    optimization_level: Optional[OptimizationLevel] = Field(
        OptimizationLevel.STANDARD, 
        description="查询优化级别"
    )
    
    @field_validator('optimization_level', mode='before')
    def normalize_optimization_level(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('查询不能为空')
        return v.strip()

class SPARQLUpdateRequest(ApiBaseModel):
    """SPARQL更新请求"""
    update: str = Field(..., description="SPARQL更新语句")
    default_graph_uri: Optional[str] = Field(None, description="默认图URI")
    named_graph_uri: Optional[List[str]] = Field(None, description="命名图URI列表")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="超时时间（秒）")
    
    @field_validator('update')
    def validate_update(cls, v):
        if not v.strip():
            raise ValueError('更新语句不能为空')
        update_upper = v.upper().strip()
        if not any(kw in update_upper for kw in ['INSERT', 'DELETE', 'CLEAR', 'DROP', 'CREATE', 'LOAD']):
            raise ValueError('无效的更新操作')
        return v.strip()

class QueryExplanationRequest(ApiBaseModel):
    """查询解释请求"""
    query: str = Field(..., description="SPARQL查询语句")
    include_optimization: Optional[bool] = Field(True, description="是否包含优化建议")
    include_statistics: Optional[bool] = Field(True, description="是否包含统计信息")

class QueryOptimizationRequest(ApiBaseModel):
    """查询优化请求"""
    query: str = Field(..., description="SPARQL查询语句")
    optimization_level: OptimizationLevel = Field(
        OptimizationLevel.STANDARD,
        description="查询优化级别"
    )

    @field_validator('optimization_level', mode='before')
    def normalize_optimization_level(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('查询不能为空')
        return v.strip()

class QueryValidationRequest(ApiBaseModel):
    """查询校验请求"""
    query: str = Field(..., description="SPARQL查询语句")

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('查询不能为空')
        return v.strip()

class SPARQLQueryResponse(ApiBaseModel):
    """SPARQL查询响应"""
    query_id: str = Field(..., description="查询ID")
    success: bool = Field(..., description="执行是否成功")
    result_type: str = Field(..., description="结果类型")
    results: Any = Field(..., description="查询结果")
    format: str = Field(..., description="结果格式")
    execution_time_ms: float = Field(..., description="执行时间（毫秒）")
    row_count: int = Field(..., description="结果行数")
    cached: bool = Field(False, description="是否来自缓存")
    error_message: Optional[str] = Field(None, description="错误信息")
    performance_stats: Optional[Dict[str, Any]] = Field(None, description="性能统计")
    optimization_applied: Optional[bool] = Field(None, description="是否应用了优化")

class SPARQLUpdateResponse(ApiBaseModel):
    """SPARQL更新响应"""
    update_id: str = Field(..., description="更新ID")
    success: bool = Field(..., description="执行是否成功")
    execution_time_ms: float = Field(..., description="执行时间（毫秒）")
    affected_triples: Optional[int] = Field(None, description="影响的三元组数")
    error_message: Optional[str] = Field(None, description="错误信息")

class QueryExplanationResponse(ApiBaseModel):
    """查询解释响应"""
    query_text: str = Field(..., description="原始查询")
    query_type: str = Field(..., description="查询类型")
    complexity_score: float = Field(..., description="复杂度评分")
    estimated_execution_time_ms: float = Field(..., description="预估执行时间")
    optimization_suggestions: List[str] = Field(default_factory=list, description="优化建议")
    execution_plan: Optional[Dict[str, Any]] = Field(None, description="执行计划")
    statistics: Optional[Dict[str, Any]] = Field(None, description="统计信息")

# SPARQL查询API
@router.post(
    "/query",
    response_model=SPARQLQueryResponse,
    summary="执行SPARQL查询",
    description="执行SPARQL SELECT/CONSTRUCT/ASK/DESCRIBE查询"
)
async def execute_query(request: SPARQLQueryRequest):
    """执行SPARQL查询"""
    try:
        query_id = str(uuid.uuid4())
        
        # 开始性能分析
        profile = default_performance_monitor.start_query_profile(
            query_id, 
            request.query
        )
        
        # 确定查询类型
        query_text_upper = request.query.upper().strip()
        if query_text_upper.startswith('SELECT'):
            query_type = QueryType.SELECT
        elif query_text_upper.startswith('CONSTRUCT'):
            query_type = QueryType.CONSTRUCT
        elif query_text_upper.startswith('ASK'):
            query_type = QueryType.ASK
        elif query_text_upper.startswith('DESCRIBE'):
            query_type = QueryType.DESCRIBE
        else:
            raise HTTPException(
                status_code=400,
                detail="不支持的查询类型。仅支持SELECT、CONSTRUCT、ASK、DESCRIBE"
            )
        
        # 查询优化（如果启用）
        optimized_query = request.query
        optimization_applied = False
        
        if request.optimization_level != OptimizationLevel.NONE:
            try:
                optimization_result = await optimize_sparql_query(
                    request.query,
                    request.optimization_level
                )
                
                if optimization_result.get('optimizations_applied'):
                    optimized_query = optimization_result.get('optimized_query', request.query)
                    optimization_applied = True
                    
            except Exception as e:
                logger.warning(f"查询优化失败: {e}")
        
        # 创建查询对象
        sparql_query = SPARQLQuery(
            query_id=query_id,
            query_text=optimized_query,
            query_type=query_type,
            parameters={
                'default_graph_uri': request.default_graph_uri,
                'named_graph_uri': request.named_graph_uri or []
            },
            timeout_seconds=request.timeout,
            use_cache=request.use_cache
        )
        
        # 执行查询
        result = await default_sparql_engine.execute_query(sparql_query)
        
        # 结束性能分析
        default_performance_monitor.end_query_profile(
            query_id,
            success=result.success,
            result_count=result.row_count,
            cache_hit=result.cached,
            error=result.error_message
        )
        
        # 格式化结果
        formatted_result = None
        if result.success and request.format != ResultFormat.JSON:
            try:
                format_result = format_sparql_results(
                    result.results,
                    result.result_type,
                    request.format
                )
                formatted_result = format_result["data"]
            except Exception as e:
                logger.warning(f"结果格式化失败: {e}")
                formatted_result = result.results
        else:
            formatted_result = result.results

        await _push_query_history(
            {
                "query_id": query_id,
                "query_type": query_type.value,
                "query_text": request.query,
                "success": result.success,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
                "cached": result.cached,
                "error_message": result.error_message,
                "timestamp": utc_now().isoformat(),
            }
        )
        
        return SPARQLQueryResponse(
            query_id=query_id,
            success=result.success,
            result_type=result.result_type,
            results=formatted_result,
            format=request.format.value,
            execution_time_ms=result.execution_time_ms,
            row_count=result.row_count,
            cached=result.cached,
            error_message=result.error_message,
            performance_stats=result.performance_stats,
            optimization_applied=optimization_applied
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"执行SPARQL查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/query",
    summary="GET方式执行SPARQL查询",
    description="通过GET请求执行SPARQL查询（符合SPARQL Protocol标准）"
)
async def execute_query_get(
    query: str = Query(..., description="SPARQL查询语句"),
    default_graph_uri: Optional[str] = Query(None, alias="default-graph-uri", description="默认图URI"),
    named_graph_uri: Optional[List[str]] = Query(None, alias="named-graph-uri", description="命名图URI列表"),
    timeout: Optional[int] = Query(30, ge=1, le=300, description="超时时间（秒）"),
    format: Optional[ResultFormat] = Query(ResultFormat.JSON, description="结果格式")
):
    """GET方式执行SPARQL查询"""
    request = SPARQLQueryRequest(
        query=query,
        default_graph_uri=default_graph_uri,
        named_graph_uri=named_graph_uri,
        timeout=timeout,
        format=format
    )
    
    response = await execute_query(request)
    
    # 根据format返回适当的响应
    if format == ResultFormat.JSON:
        return response
    else:
        # 返回格式化的内容
        content_type = default_formatter._get_content_type(format)
        
        if isinstance(response.results, str):
            content = response.results
        else:
            content = str(response.results)
        
        return Response(
            content=content,
            media_type=content_type,
            headers={
                "X-Query-ID": response.query_id,
                "X-Execution-Time": str(response.execution_time_ms),
                "X-Row-Count": str(response.row_count)
            }
        )

# SPARQL更新API
@router.post(
    "/update",
    response_model=SPARQLUpdateResponse,
    summary="执行SPARQL更新",
    description="执行SPARQL INSERT/DELETE/UPDATE操作"
)
async def execute_update(request: SPARQLUpdateRequest):
    """执行SPARQL更新"""
    try:
        update_id = str(uuid.uuid4())
        
        # 开始性能分析
        profile = default_performance_monitor.start_query_profile(
            update_id,
            request.update
        )
        
        # 创建查询对象
        sparql_query = SPARQLQuery(
            query_id=update_id,
            query_text=request.update,
            query_type=QueryType.UPDATE,
            parameters={
                'default_graph_uri': request.default_graph_uri,
                'named_graph_uri': request.named_graph_uri or []
            },
            timeout_seconds=request.timeout,
            use_cache=False  # 更新操作不使用缓存
        )
        
        # 执行更新
        result = await default_sparql_engine.execute_query(sparql_query)
        
        # 结束性能分析
        default_performance_monitor.end_query_profile(
            update_id,
            success=result.success,
            result_count=0,  # 更新操作不返回结果计数
            cache_hit=False,
            error=result.error_message
        )

        await _push_query_history(
            {
                "query_id": update_id,
                "query_type": QueryType.UPDATE.value,
                "query_text": request.update,
                "success": result.success,
                "row_count": 0,
                "execution_time_ms": result.execution_time_ms,
                "cached": False,
                "error_message": result.error_message,
                "timestamp": utc_now().isoformat(),
            }
        )
        
        return SPARQLUpdateResponse(
            update_id=update_id,
            success=result.success,
            execution_time_ms=result.execution_time_ms,
            affected_triples=None,  # 这里可以实现三元组计数逻辑
            error_message=result.error_message
        )
        
    except Exception as e:
        logger.error(f"执行SPARQL更新失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/update/form",
    summary="Form方式执行SPARQL更新",
    description="通过Form提交执行SPARQL更新（符合SPARQL Protocol标准）"
)
async def execute_update_form(
    update: str = Form(..., description="SPARQL更新语句"),
    default_graph_uri: Optional[str] = Form(None, alias="default-graph-uri", description="默认图URI"),
    named_graph_uri: Optional[str] = Form(None, alias="named-graph-uri", description="命名图URI"),
    timeout: Optional[int] = Form(30, description="超时时间（秒）")
):
    """Form方式执行SPARQL更新"""
    named_graph_list = None
    if named_graph_uri:
        named_graph_list = [uri.strip() for uri in named_graph_uri.split(',')]
    
    request = SPARQLUpdateRequest(
        update=update,
        default_graph_uri=default_graph_uri,
        named_graph_uri=named_graph_list,
        timeout=timeout
    )
    
    return await execute_update(request)

# 查询解释API
@router.post(
    "/explain",
    response_model=QueryExplanationResponse,
    summary="解释SPARQL查询",
    description="分析SPARQL查询的执行计划和性能特征"
)
async def explain_query(request: QueryExplanationRequest):
    """解释SPARQL查询"""
    try:
        # 获取查询解释
        explanation = await explain_sparql_query(request.query)
        
        if not explanation:
            raise HTTPException(
                status_code=500,
                detail="查询解释失败"
            )
        
        # 确定查询类型
        query_text_upper = request.query.upper().strip()
        if query_text_upper.startswith('SELECT'):
            query_type = "SELECT"
        elif query_text_upper.startswith('CONSTRUCT'):
            query_type = "CONSTRUCT"
        elif query_text_upper.startswith('ASK'):
            query_type = "ASK"
        elif query_text_upper.startswith('DESCRIBE'):
            query_type = "DESCRIBE"
        else:
            query_type = "UNKNOWN"
        
        # 计算复杂度评分
        complexity_score = explanation.get('performance_prediction', {}).get('complexity_score', 0)
        
        # 预估执行时间
        estimated_time = explanation.get('performance_prediction', {}).get('time_estimate_ms', 1000)
        
        return QueryExplanationResponse(
            query_text=request.query,
            query_type=query_type,
            complexity_score=float(complexity_score),
            estimated_execution_time_ms=float(estimated_time),
            optimization_suggestions=explanation.get('optimization_suggestions', []),
            execution_plan=explanation.get('execution_plan'),
            statistics=explanation.get('query_patterns') if request.include_statistics else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询解释失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询优化API
@router.post(
    "/optimize",
    summary="优化SPARQL查询",
    description="返回优化后的查询与性能提升估计"
)
async def optimize_query(request: QueryOptimizationRequest):
    """优化SPARQL查询"""
    try:
        optimization_result = await optimize_sparql_query(
            request.query,
            request.optimization_level
        )

        improvements = optimization_result.get("optimizations_applied", [])
        impact = optimization_result.get("estimated_impact", {}) or {}
        estimated_speedup = float(impact.get("estimated_speedup", 1.0) or 1.0)

        return {
            "original_query": optimization_result.get("original_query", request.query),
            "optimized_query": optimization_result.get("optimized_query", request.query),
            "improvements": improvements,
            "estimated_speedup": estimated_speedup
        }
    except Exception as e:
        logger.error(f"查询优化失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询校验API
@router.post(
    "/validate",
    summary="校验SPARQL查询",
    description="只校验语法和类型，不执行查询"
)
async def validate_query(request: QueryValidationRequest):
    """校验SPARQL查询"""
    try:
        query_text = request.query.strip()
        query_upper = query_text.upper()
        update_keywords = ("INSERT", "DELETE", "CLEAR", "DROP", "CREATE", "LOAD", "ADD", "MOVE", "COPY", "WITH")
        query_type = QueryType.UPDATE if query_upper.startswith(update_keywords) else QueryType.SELECT

        validation = await default_sparql_engine._validate_query(query_text, query_type)
        if validation.get("valid"):
            return {"valid": True}
        return {"valid": False, "errors": [validation.get("error", "查询语法错误")]}
    except Exception as e:
        logger.error(f"查询校验失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询结果流式传输API
@router.get(
    "/stream",
    summary="流式查询结果",
    description="以流的方式返回大量查询结果"
)
async def stream_query_results(
    query: str = Query(..., description="SPARQL查询语句"),
    format: ResultFormat = Query(ResultFormat.JSON, description="结果格式"),
    chunk_size: int = Query(100, ge=1, le=1000, description="分块大小")
):
    """流式查询结果"""
    try:
        # 执行查询
        result = await execute_sparql_query(
            query,
            QueryType.SELECT,
            timeout_seconds=300  # 流式查询使用更长的超时时间
        )
        
        if not result.success:
            await _push_query_history(
                {
                    "query_id": result.query_id,
                    "query_type": QueryType.SELECT.value,
                    "query_text": query,
                    "success": False,
                    "row_count": 0,
                    "execution_time_ms": result.execution_time_ms,
                    "cached": result.cached,
                    "error_message": result.error_message,
                    "timestamp": utc_now().isoformat(),
                }
            )
            raise HTTPException(
                status_code=500,
                detail=f"查询失败: {result.error_message}"
            )
        
        # 确定内容类型
        content_type = default_formatter._get_content_type(format)
        
        async def generate_chunks():
            """生成数据块"""
            results = result.results
            
            if format == ResultFormat.JSON:
                # JSON格式的流式传输
                yield '{"results": [\n'
                
                for i in range(0, len(results), chunk_size):
                    chunk = results[i:i + chunk_size]
                    
                    for j, item in enumerate(chunk):
                        if i > 0 or j > 0:
                            yield ',\n'
                        yield json.dumps(item, ensure_ascii=False)
                
                yield '\n], "metadata": {'
                yield f'"total_count": {len(results)}, '
                yield f'"execution_time_ms": {result.execution_time_ms}'
                yield '}}'
                
            elif format == ResultFormat.CSV:
                # CSV格式的流式传输
                if results:
                    # 表头
                    headers = list(results[0].keys())
                    yield ','.join(headers) + '\n'
                    
                    # 数据行
                    for i in range(0, len(results), chunk_size):
                        chunk = results[i:i + chunk_size]
                        
                        for item in chunk:
                            row_values = []
                            for header in headers:
                                value = str(item.get(header, ''))
                                # 简单的CSV转义
                                if ',' in value or '"' in value or '\n' in value:
                                    value = '"' + value.replace('"', '""') + '"'
                                row_values.append(value)
                            
                            yield ','.join(row_values) + '\n'
            
            else:
                # 其他格式一次性返回
                formatted_result = format_sparql_results(
                    results,
                    result.result_type,
                    format
                )
                yield formatted_result["data"]

        await _push_query_history(
            {
                "query_id": result.query_id,
                "query_type": QueryType.SELECT.value,
                "query_text": query,
                "success": True,
                "row_count": result.row_count,
                "execution_time_ms": result.execution_time_ms,
                "cached": result.cached,
                "error_message": result.error_message,
                "timestamp": utc_now().isoformat(),
            }
        )
        
        return StreamingResponse(
            generate_chunks(),
            media_type=content_type,
            headers={
                "X-Row-Count": str(result.row_count),
                "X-Execution-Time": str(result.execution_time_ms)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"流式查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 查询历史API
@router.get(
    "/history",
    response_model=Dict[str, Any],
    summary="查询历史",
    description="获取最近的查询执行历史"
)
async def get_query_history(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    include_errors: bool = Query(True, description="是否包含失败的查询")
):
    """获取查询历史"""
    redis_client = get_redis()
    if not redis_client:
        return {"history": [], "total": 0, "timestamp": utc_now().isoformat()}
    raw_items = await redis_client.lrange(_SPARQL_HISTORY_KEY, 0, 999)
    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if not include_errors and not item.get("success"):
            continue
        items.append(item)
        if len(items) >= limit:
            break
    return {"history": items, "total": len(items), "timestamp": utc_now().isoformat()}

# 查询统计API
@router.get(
    "/statistics",
    summary="查询统计",
    description="获取查询统计与趋势数据"
)
async def get_query_statistics(
    period: Optional[str] = Query(None, description="统计周期：day|week|month")
):
    """获取查询统计信息"""
    now = utc_now()
    period_map = {"day": 1, "week": 7, "month": 30}
    days = period_map.get((period or "").lower())
    cutoff = now - timedelta(days=days) if days else None

    redis_client = get_redis()
    raw_items = []
    if redis_client:
        raw_items = await redis_client.lrange(_SPARQL_HISTORY_KEY, 0, 999)

    items: List[Dict[str, Any]] = []
    for raw in raw_items:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        timestamp = item.get("timestamp")
        if cutoff and timestamp:
            try:
                ts = datetime.fromisoformat(timestamp)
            except Exception:
                ts = None
            if ts and ts < cutoff:
                continue
        items.append(item)

    total_queries = len(items)
    success_count = len([i for i in items if i.get("success")])
    cache_hits = len([i for i in items if i.get("cached")])
    average_execution_time = 0.0
    if total_queries:
        average_execution_time = sum(float(i.get("execution_time_ms") or 0) for i in items) / total_queries
    success_rate = (success_count / total_queries) if total_queries else 0.0
    cache_hit_rate = (cache_hits / total_queries) if total_queries else 0.0

    pattern_counts: Dict[str, int] = {}
    for item in items:
        pattern = item.get("query_type") or ""
        if not pattern:
            query_text = (item.get("query_text") or "").strip().upper()
            pattern = query_text.split(" ", 1)[0] if query_text else "UNKNOWN"
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    most_frequent_patterns = [
        {"pattern": key, "count": value}
        for key, value in sorted(pattern_counts.items(), key=lambda kv: kv[1], reverse=True)
    ]

    trend_map: Dict[str, Dict[str, Any]] = {}
    for item in items:
        timestamp = item.get("timestamp")
        if not timestamp:
            continue
        date_key = timestamp.split("T")[0]
        entry = trend_map.setdefault(date_key, {"total_time": 0.0, "count": 0})
        entry["total_time"] += float(item.get("execution_time_ms") or 0)
        entry["count"] += 1

    performance_trend = []
    for date_key, entry in sorted(trend_map.items()):
        count = entry["count"]
        avg_time = (entry["total_time"] / count) if count else 0.0
        performance_trend.append(
            {"date": date_key, "avg_time": avg_time, "query_count": count}
        )

    return {
        "total_queries": total_queries,
        "success_rate": success_rate,
        "average_execution_time": average_execution_time,
        "cache_hit_rate": cache_hit_rate,
        "most_frequent_patterns": most_frequent_patterns,
        "performance_trend": performance_trend
    }

# 查询性能分析API
@router.get(
    "/performance",
    response_model=Dict[str, Any],
    summary="查询性能分析",
    description="获取SPARQL查询的性能分析报告"
)
async def get_query_performance(
    window_minutes: int = Query(60, ge=1, le=1440, description="时间窗口（分钟）")
):
    """获取查询性能分析"""
    try:
        # 获取性能报告
        performance_report = default_performance_monitor.get_detailed_report(window_minutes)
        
        return {
            "performance_report": performance_report,
            "sparql_engine_stats": default_sparql_engine.get_statistics(),
            "recommendations": performance_report.get("recommendations", [])
        }
        
    except Exception as e:
        logger.error(f"获取查询性能分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 缓存管理API
@router.delete(
    "/cache",
    summary="清空查询缓存",
    description="清空SPARQL查询结果缓存"
)
async def clear_query_cache():
    """清空查询缓存"""
    try:
        stats = default_sparql_engine.get_statistics()
        cleared = int((stats.get("cache_stats") or {}).get("size") or 0)
        default_sparql_engine.clear_cache()
        
        return {
            "cleared": cleared,
            "message": "查询缓存已清空",
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"清空查询缓存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/cache/clear",
    summary="清空查询缓存",
    description="清空SPARQL查询结果缓存"
)
async def clear_query_cache_alias():
    """清空查询缓存（兼容前端）"""
    return await clear_query_cache()

@router.get(
    "/cache/stats",
    summary="缓存统计",
    description="获取查询缓存的统计信息"
)
async def get_cache_stats():
    """获取缓存统计"""
    try:
        stats = default_sparql_engine.get_statistics()
        
        return {
            "cache_stats": stats.get("cache_stats", {}),
            "total_queries": stats.get("total_queries", 0),
            "cached_queries": stats.get("cached_queries", 0),
            "cache_hit_rate": (
                stats.get("cached_queries", 0) / max(stats.get("total_queries", 1), 1)
            ),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 批量查询API
@router.post(
    "/batch",
    summary="批量查询",
    description="批量执行多个SPARQL查询"
)
async def batch_queries(
    queries: List[SPARQLQueryRequest],
    max_concurrent: int = Query(5, ge=1, le=10, description="最大并发数")
):
    """批量查询"""
    try:
        if len(queries) > 20:
            raise HTTPException(
                status_code=400,
                detail="批量查询数量不能超过20个"
            )
        
        # 限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_query(query_request: SPARQLQueryRequest):
            async with semaphore:
                try:
                    return await execute_query(query_request)
                except Exception as e:
                    return {
                        "query_id": str(uuid.uuid4()),
                        "success": False,
                        "error": str(e),
                        "query": query_request.query[:100] + "..." if len(query_request.query) > 100 else query_request.query
                    }
        
        # 并发执行所有查询
        start_time = asyncio.get_running_loop().time()
        results = await asyncio.gather(
            *[execute_single_query(query) for query in queries]
        )
        total_time = (asyncio.get_running_loop().time() - start_time) * 1000
        
        # 统计结果
        successful_count = sum(1 for result in results if result.get("success", False))
        failed_count = len(results) - successful_count
        
        return {
            "batch_id": str(uuid.uuid4()),
            "total_queries": len(queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "total_execution_time_ms": total_time,
            "results": results,
            "timestamp": utc_now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
