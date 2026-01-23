"""
Story 1.5: 标准化智能体API接口实现
提供清晰的API接口来与智能体交互，符合架构文档规范
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse
import json
import time
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from typing import AsyncIterator, Dict
from src.models.schemas import (
    APIResponse, SuccessResponse, ErrorResponse,
    ChatRequest, ChatResponse, ToolCall,
    TaskRequest, TaskResponse, TaskStatus, TaskResult,
    AgentStatusResponse, AgentHealth, AgentInfo, 
    SystemResource, PerformanceMetrics
)
from src.services.agent_service import get_agent_service, AgentService
from src.core.dependencies import get_current_user

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent-interface"])

# ===== POST /api/v1/agent/chat =====

@router.post("/chat", response_model=SuccessResponse[ChatResponse])
async def chat_with_agent(
    request: ChatRequest,
    http_request: Request,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
) -> SuccessResponse[ChatResponse]:
    """
    单轮对话接口
    
    实现AC1: POST /api/v1/agent/chat接口实现，支持单轮对话
    实现AC4: 所有接口都有完整的请求/响应数据模型定义
    实现AC5: API接口的输入验证和错误响应标准化
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    
    logger.info(
        "聊天请求开始",
        request_id=request_id,
        user_id=current_user,
        message_length=len(request.message),
        stream=request.stream
    )
    
    try:
        # 如果启用流式响应
        if request.stream:
            return await _handle_stream_chat(
                request, request_id, message_id, agent_service, current_user
            )
        
        # 处理普通聊天请求
        # 创建临时会话用于单轮对话
        session_result = await agent_service.create_agent_session(
            user_id=current_user,
            agent_type="react",
            agent_config={}
        )
        
        conversation_id = session_result["conversation_id"]
        
        try:
            # 执行对话
            chat_result = await agent_service.chat_with_agent(
                conversation_id=conversation_id,
                user_input=request.message,
                user_id=current_user,
                stream=False
            )
            
            # 构建标准化响应
            response_time = time.time() - start_time
            
            # 转换工具调用格式
            tool_calls = []
            if "tool_calls" in chat_result:
                for tool_call in chat_result["tool_calls"]:
                    tool_calls.append(ToolCall(
                        tool_name=tool_call.get("tool_name", "unknown"),
                        arguments=tool_call.get("arguments", {}),
                        result=tool_call.get("result"),
                        execution_time=tool_call.get("execution_time"),
                        status=tool_call.get("status", "success")
                    ))
            
            chat_response = ChatResponse(
                message=chat_result["response"],
                conversation_id=conversation_id,
                message_id=message_id,
                reasoning_steps=chat_result.get("reasoning_steps", []),
                tool_calls=tool_calls,
                metadata={
                    "session_temporary": True,
                    "agent_type": "react",
                    "steps": chat_result.get("steps", 0)
                },
                response_time=response_time,
                token_usage=chat_result.get("token_usage")
            )
            
            logger.info(
                "聊天请求完成",
                request_id=request_id,
                conversation_id=conversation_id,
                response_time=response_time,
                tool_calls_count=len(tool_calls)
            )
            
            return SuccessResponse(
                data=chat_response,
                request_id=request_id
            )
            
        finally:
            # 清理临时会话
            try:
                await agent_service.close_agent_session(conversation_id, current_user)
            except Exception as cleanup_error:
                logger.warning(
                    "清理临时会话失败",
                    conversation_id=conversation_id,
                    error=str(cleanup_error)
                )
        
    except ValueError as e:
        logger.error(
            "聊天请求验证失败",
            request_id=request_id,
            error=str(e),
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请求参数无效: {str(e)}"
        )
    
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(
            "聊天请求失败",
            request_id=request_id,
            error=str(e),
            user_id=current_user,
            response_time=response_time
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"智能体对话失败: {str(e)}"
        )

async def _handle_stream_chat(
    request: ChatRequest,
    request_id: str,
    message_id: str,
    agent_service: AgentService,
    current_user: str
) -> StreamingResponse:
    """处理流式聊天响应，使用OpenAI标准格式"""
    
    async def generate_stream() -> AsyncIterator[str]:
        conversation_id = None
        created_timestamp = int(time.time())
        
        try:
            # 创建临时会话
            session_result = await agent_service.create_agent_session(
                user_id=current_user,
                agent_type="react",
                agent_config={}
            )
            conversation_id = session_result["conversation_id"]
            
            # 流式处理智能体响应，直接使用OpenAI标准格式
            async for step_data in agent_service.chat_with_agent(
                conversation_id=conversation_id,
                user_input=request.message,
                user_id=current_user,
                stream=True
            ):
                # 处理所有streaming_token类型，包括推理过程和最终答案
                if step_data.get("step_type") == "streaming_token":
                    openai_chunk = {
                        "id": f"chatcmpl-{message_id}",
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": step_data.get("content", "")
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                
                # 处理推理步骤 - 不显示内部推理过程
                elif step_data.get("step_type") in ["thought", "action", "observation"]:
                    # 静默处理推理步骤，不向用户显示内部推理过程
                    continue
                
                elif step_data.get("step_type") == "final_answer":
                    # 发送完成标记，使用OpenAI标准格式
                    openai_finish_chunk = {
                        "id": f"chatcmpl-{message_id}",
                        "object": "chat.completion.chunk", 
                        "created": created_timestamp,
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(openai_finish_chunk, ensure_ascii=False)}\n\n"
                    break
            
        except Exception as e:
            # 发送错误信息，使用OpenAI标准错误格式
            error_chunk = {
                "id": f"chatcmpl-{message_id}",
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": "gpt-4o-mini",
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
            
        finally:
            # 清理临时会话
            if conversation_id:
                try:
                    await agent_service.close_agent_session(conversation_id, current_user)
                except Exception:
                    logger.exception("关闭会话失败", exc_info=True)
            
            # 发送标准结束标记
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id
        }
    )

# ===== POST /api/v1/agent/task =====

@router.post("/task", response_model=SuccessResponse[TaskResponse])
async def execute_agent_task(
    request: TaskRequest,
    http_request: Request,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: str = Depends(get_current_user)
) -> SuccessResponse[TaskResponse]:
    """
    任务执行接口
    
    实现AC2: POST /api/v1/agent/task接口实现，支持任务执行
    实现AC4: 所有接口都有完整的请求/响应数据模型定义
    实现AC5: API接口的输入验证和错误响应标准化
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    logger.info(
        "任务执行请求开始",
        request_id=request_id,
        task_id=task_id,
        user_id=current_user,
        task_type=request.task_type,
        priority=request.priority
    )
    
    try:
        # 创建任务专用会话
        session_result = await agent_service.create_agent_session(
            user_id=current_user,
            agent_type="react",
            agent_config={
                "task_mode": True,
                "task_type": request.task_type,
                "priority": request.priority
            }
        )
        
        conversation_id = session_result["conversation_id"]
        
        try:
            # 构建任务消息
            task_message = f"任务: {request.description}"
            
            if request.requirements:
                task_message += f"\n要求: {'; '.join(request.requirements)}"
            
            if request.constraints:
                constraints_str = "; ".join([f"{k}: {v}" for k, v in request.constraints.items()])
                task_message += f"\n约束: {constraints_str}"
            
            if request.expected_output:
                task_message += f"\n期望输出: {request.expected_output}"
            
            if request.context:
                task_message += f"\n上下文: {json.dumps(request.context, ensure_ascii=False)}"
            
            # 执行任务
            execution_result = await agent_service.chat_with_agent(
                conversation_id=conversation_id,
                user_input=task_message,
                user_id=current_user,
                stream=False
            )
            
            execution_time = time.time() - start_time
            
            # 构建任务结果
            task_result = TaskResult(
                output=execution_result["response"],
                artifacts=[],  # 可以扩展为从结果中提取的文件等
                metrics={
                    "steps": execution_result.get("steps", 0),
                    "tool_calls": len(execution_result.get("tool_calls", [])),
                    "response_time": execution_time
                }
            )
            
            # 构建任务响应
            task_response = TaskResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                progress=100.0,
                result=task_result,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=utc_now(),
                execution_time=execution_time,
                steps_completed=[
                    "任务解析",
                    "智能体推理",
                    "工具调用执行",
                    "结果生成"
                ]
            )
            
            logger.info(
                "任务执行完成",
                request_id=request_id,
                task_id=task_id,
                conversation_id=conversation_id,
                execution_time=execution_time,
                steps=execution_result.get("steps", 0)
            )
            
            return SuccessResponse(
                data=task_response,
                request_id=request_id
            )
            
        finally:
            # 清理任务会话
            try:
                await agent_service.close_agent_session(conversation_id, current_user)
            except Exception as cleanup_error:
                logger.warning(
                    "清理任务会话失败",
                    conversation_id=conversation_id,
                    error=str(cleanup_error)
                )
        
    except ValueError as e:
        logger.error(
            "任务请求验证失败",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
            user_id=current_user
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"任务参数无效: {str(e)}"
        )
    
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            "任务执行失败",
            request_id=request_id,
            task_id=task_id,
            error=str(e),
            user_id=current_user,
            execution_time=execution_time
        )
        
        # 返回失败的任务响应
        failed_response = TaskResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            error_message=str(e),
            started_at=datetime.fromtimestamp(start_time),
            execution_time=execution_time,
            steps_completed=["任务解析"]
        )
        
        return SuccessResponse(
            data=failed_response,
            request_id=request_id
        )

# ===== GET /api/v1/agent/status =====

@router.get("/status", response_model=SuccessResponse[AgentStatusResponse])
async def get_agent_status(
    http_request: Request,
    agent_service: AgentService = Depends(get_agent_service)
) -> SuccessResponse[AgentStatusResponse]:
    """
    智能体状态查询接口
    
    实现AC3: GET /api/v1/agent/status接口实现，查询智能体状态
    实现AC4: 所有接口都有完整的请求/响应数据模型定义
    实现AC6: 接口响应时间监控和性能日志记录
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    logger.info("智能体状态查询开始", request_id=request_id)
    
    try:
        import psutil

        active_conversations = len(getattr(getattr(agent_service, 'conversation_service', None), 'active_sessions', {}) or {})
        total_conversations = active_conversations

        system_resources = SystemResource(
            cpu_usage=psutil.cpu_percent(interval=0.1),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            active_connections=active_conversations
        )

        performance_metrics = PerformanceMetrics(
            average_response_time=0.0,
            requests_per_minute=0.0,
            success_rate=100.0,
            error_rate=0.0,
            uptime=0.0
        )

        agent_info = AgentInfo(
            agent_id="react-agent-main",
            agent_type="react",
            version="1.0.0",
            capabilities=[],
            active_conversations=active_conversations,
            total_conversations=total_conversations
        )

        status_response = AgentStatusResponse(
            health=AgentHealth.HEALTHY,
            agent_info=agent_info,
            system_resources=system_resources,
            performance_metrics=performance_metrics,
            last_activity=utc_now(),
            diagnostics={}
        )

        response_time = time.time() - start_time
        logger.info(
            "智能体状态查询完成",
            request_id=request_id,
            health=status_response.health,
            response_time=response_time,
            active_conversations=active_conversations
        )

        return SuccessResponse(
            data=status_response,
            request_id=request_id
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(
            "智能体状态查询失败",
            request_id=request_id,
            error=str(e),
            response_time=response_time
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取智能体状态失败: {str(e)}"
        )

# ===== 性能指标接口 =====

@router.get("/metrics", response_model=SuccessResponse[Dict])
async def get_performance_metrics(
    http_request: Request
) -> SuccessResponse[Dict]:
    """
    获取API性能指标
    
    实现AC6: 接口响应时间监控和性能日志记录
    """
    request_id = str(uuid.uuid4())
    
    try:
        from src.core.monitoring import monitoring_service

        stats = await monitoring_service.performance_monitor.get_stats()
        durations = [
            item["duration"] * 1000
            for item in monitoring_service.performance_monitor.request_times
            if "duration" in item
        ]

        def _percentile(values: list[float], percentile: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = int(round((len(ordered) - 1) * percentile))
            return ordered[min(max(idx, 0), len(ordered) - 1)]

        metrics = {
            "total_requests": stats.get("total_requests", 0),
            "error_rate": stats.get("error_rate", 0.0),
            "average_response_time_ms": stats.get("average_response_time_ms", 0.0),
            "requests_per_minute": stats.get("requests_per_minute", 0),
            "active_requests": stats.get("active_requests", 0),
            "error_counts": stats.get("error_counts", {}),
            "latency_ms_p50": _percentile(durations, 0.5),
            "latency_ms_p95": _percentile(durations, 0.95),
            "requests": [],
        }
        
        metrics["timestamp"] = utc_now().isoformat()
        metrics["api_version"] = "1.0"
        
        logger.info(
            "性能指标查询完成",
            request_id=request_id,
            total_requests=metrics.get("total_requests", 0)
        )
        
        return SuccessResponse(
            data=metrics,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "获取性能指标失败",
            request_id=request_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取性能指标失败: {str(e)}"
        )
