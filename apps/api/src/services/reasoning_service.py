"""推理服务层"""

import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from uuid import UUID, uuid4
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from models.schemas.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningChain,
    ReasoningStreamChunk,
    ReasoningValidation,
    ReasoningStrategy,
    ThoughtStep
)
from src.ai.reasoning.cot_engine import BaseCoTEngine
from src.ai.reasoning.strategies.zero_shot import ZeroShotCoTEngine
from src.ai.reasoning.strategies.few_shot import FewShotCoTEngine
from src.ai.reasoning.strategies.auto_cot import AutoCoTEngine
from src.ai.reasoning.state_integration import ReasoningStateMachine
from src.ai.reasoning.validation import CompositeValidator, calculate_chain_quality_score
from src.ai.reasoning.recovery import RecoveryManager, RecoveryStrategy
from src.ai.reasoning.models import ReasoningChainModel, ThoughtStepModel
from src.core.database import get_session
from src.core.redis import get_redis_client
from src.core.logging import get_logger

# 离线推理支持
from ..offline.reasoning_engine import (
    OfflineReasoningEngine, ReasoningStrategy as OfflineReasoningStrategy
)
from ..offline.local_inference import LocalInferenceEngine
from ..offline.memory_manager import OfflineMemoryManager
from ..offline.model_cache import ModelCacheManager
from ..models.schemas.offline import OfflineMode, NetworkStatus

logger = get_logger(__name__)


class ReasoningService:
    """推理服务"""
    
    def __init__(self):
        self.engines = {
            ReasoningStrategy.ZERO_SHOT: ZeroShotCoTEngine(),
            ReasoningStrategy.FEW_SHOT: FewShotCoTEngine(),
            ReasoningStrategy.AUTO_COT: AutoCoTEngine()
        }
        self.state_machine = ReasoningStateMachine()
        self.validator = CompositeValidator()
        self.recovery_manager = RecoveryManager()
        self.redis_client = None
        
        # 离线推理组件
        self.offline_mode = OfflineMode.ONLINE
        self.network_status = NetworkStatus.CONNECTED
        self._offline_engine = None
        
    def _get_offline_engine(self) -> OfflineReasoningEngine:
        """获取离线推理引擎"""
        if not self._offline_engine:
            model_cache = ModelCacheManager()
            inference_engine = LocalInferenceEngine(model_cache)
            memory_manager = OfflineMemoryManager()
            self._offline_engine = OfflineReasoningEngine(inference_engine, memory_manager)
        return self._offline_engine
    
    def set_offline_mode(self, mode: OfflineMode, network_status: NetworkStatus = NetworkStatus.UNKNOWN):
        """设置离线模式"""
        self.offline_mode = mode
        self.network_status = network_status
        
        # 同步设置推理引擎的网络状态
        if self._offline_engine:
            self._offline_engine.inference_engine.set_network_status(network_status)
    
    async def _get_redis(self):
        """获取Redis客户端"""
        if not self.redis_client:
            self.redis_client = await get_redis_client()
        return self.redis_client
    
    async def execute_reasoning(
        self,
        request: ReasoningRequest,
        user: Dict[str, Any]
    ) -> ReasoningResponse:
        """执行推理"""
        try:
            # 根据离线模式选择执行策略
            if self.offline_mode == OfflineMode.OFFLINE or self.network_status == NetworkStatus.DISCONNECTED:
                return await self._execute_offline_reasoning(request, user)
            elif self.offline_mode == OfflineMode.ONLINE and self.network_status == NetworkStatus.CONNECTED:
                return await self._execute_online_reasoning(request, user)
            else:
                # 混合模式：优先在线，降级到离线
                try:
                    return await self._execute_online_reasoning(request, user)
                except Exception as e:
                    logger.warning(f"在线推理失败，降级到离线模式: {e}")
                    return await self._execute_offline_reasoning(request, user)
                    
        except Exception as e:
            logger.error(f"执行推理失败: {e}")
            return ReasoningResponse(
                chain_id=uuid4(),
                problem=request.problem,
                strategy=request.strategy,
                steps=[],
                success=False,
                error=str(e)
            )
    
    async def _execute_online_reasoning(
        self,
        request: ReasoningRequest,
        user: Dict[str, Any]
    ) -> ReasoningResponse:
        """执行在线推理（原有逻辑）"""
        # 选择引擎
        engine = self.engines.get(request.strategy)
        if not engine:
            raise ValueError(f"不支持的推理策略: {request.strategy}")
        
        # 检查缓存
        cache_key = await self._get_cache_key(request)
        cached_result = await self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"使用缓存结果: {cache_key}")
            return cached_result
        
        # 执行推理
        chain = await engine.execute_chain(request)
        
        # 保存到数据库
        await self._save_chain(chain, user.get('id'))
        
        # 缓存结果
        response = self._chain_to_response(chain)
        await self._cache_result(cache_key, response)
        
        return response
    
    async def _execute_offline_reasoning(
        self,
        request: ReasoningRequest,
        user: Dict[str, Any]
    ) -> ReasoningResponse:
        """执行离线推理"""
        offline_engine = self._get_offline_engine()
        
        # 转换推理策略
        strategy_mapping = {
            ReasoningStrategy.ZERO_SHOT: OfflineReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningStrategy.FEW_SHOT: OfflineReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningStrategy.AUTO_COT: OfflineReasoningStrategy.TREE_OF_THOUGHT
        }
        
        offline_strategy = strategy_mapping.get(request.strategy, OfflineReasoningStrategy.CHAIN_OF_THOUGHT)
        
        # 创建离线推理工作流
        session_id = user.get('id', 'anonymous')
        workflow_id = await offline_engine.create_reasoning_workflow(
            name=f"离线推理-{request.strategy.value}",
            problem=request.problem,
            session_id=session_id,
            strategy=offline_strategy
        )
        
        # 执行工作流
        reasoning_chain = await offline_engine.execute_workflow(workflow_id)
        
        # 转换为在线格式
        return self._convert_offline_to_online_response(reasoning_chain, request)
    
    async def stream_reasoning(
        self,
        request: ReasoningRequest,
        user: Dict[str, Any]
    ) -> AsyncGenerator[ReasoningStreamChunk, None]:
        """流式推理"""
        try:
            engine = self.engines.get(request.strategy)
            if not engine:
                raise ValueError(f"不支持的推理策略: {request.strategy}")
            
            chain_id = uuid4()
            
            # 流式回调
            async def stream_callback(chunk: ReasoningStreamChunk):
                yield chunk
            
            # 创建推理链
            chain = ReasoningChain(
                id=chain_id,
                strategy=request.strategy,
                problem=request.problem,
                context=request.context
            )
            
            # 逐步执行
            for step_num in range(1, request.max_steps + 1):
                step = await engine.execute_step(
                    chain,
                    step_num,
                    request.problem,
                    request.context,
                    examples=request.examples
                )
                
                chain.add_step(step)
                
                # 验证步骤
                validation = await self.validator.validate(step, chain)
                
                # 如果验证失败，尝试恢复
                if not validation.is_valid:
                    recovery_strategy = await self.recovery_manager.handle_failure(
                        step, chain
                    )
                    if recovery_strategy:
                        await self.recovery_manager.execute_recovery(
                            recovery_strategy, chain, step
                        )
                
                # 生成流式响应
                chunk = ReasoningStreamChunk(
                    chain_id=chain_id,
                    step_number=step_num,
                    step_type=step.step_type,
                    content=step.content,
                    reasoning=step.reasoning,
                    confidence=step.confidence,
                    is_final=(step.step_type == "conclusion")
                )
                
                yield chunk
                
                # 如果到达结论，结束
                if step.step_type == "conclusion":
                    chain.complete(step.content)
                    break
            
            # 保存推理链
            await self._save_chain(chain, user.get('id'))
            
        except Exception as e:
            logger.error(f"流式推理失败: {e}")
            # 生成错误块
            yield ReasoningStreamChunk(
                chain_id=uuid4(),
                step_number=0,
                step_type="error",
                content=str(e),
                reasoning="推理失败",
                confidence=0.0,
                is_final=True
            )
    
    async def get_chain(
        self,
        chain_id: UUID,
        user_id: Optional[str] = None
    ) -> Optional[ReasoningChain]:
        """获取推理链"""
        async with get_session() as session:
            query = select(ReasoningChainModel).where(
                ReasoningChainModel.id == chain_id
            )
            
            if user_id:
                query = query.where(
                    ReasoningChainModel.session_id == user_id
                )
            
            result = await session.execute(query)
            chain_model = result.scalar_one_or_none()
            
            if not chain_model:
                return None
            
            return self._model_to_chain(chain_model)
    
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[ReasoningChain]:
        """获取用户推理历史"""
        async with get_session() as session:
            query = (
                select(ReasoningChainModel)
                .where(ReasoningChainModel.session_id == user_id)
                .order_by(ReasoningChainModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            
            result = await session.execute(query)
            chains = result.scalars().all()
            
            return [self._model_to_chain(c) for c in chains]
    
    async def validate_chain(
        self,
        chain_id: UUID,
        step_number: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Optional[ReasoningValidation]:
        """验证推理链"""
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return None
        
        if step_number:
            # 验证特定步骤
            step = next(
                (s for s in chain.steps if s.step_number == step_number),
                None
            )
            if step:
                return await self.validator.validate(step, chain)
        else:
            # 验证最后一步
            if chain.steps:
                return await self.validator.validate(chain.steps[-1], chain)
        
        return None
    
    async def create_branch(
        self,
        chain_id: UUID,
        parent_step_number: int,
        reason: str,
        user_id: Optional[str] = None
    ) -> Optional[UUID]:
        """创建分支"""
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return None
        
        parent_step = next(
            (s for s in chain.steps if s.step_number == parent_step_number),
            None
        )
        
        if parent_step:
            branch = chain.create_branch(parent_step.id, reason)
            await self._save_chain(chain, user_id)
            return branch.id
        
        return None
    
    async def recover_chain(
        self,
        chain_id: UUID,
        strategy: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """恢复推理链"""
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return False
        
        recovery_strategy = None
        if strategy:
            try:
                recovery_strategy = RecoveryStrategy(strategy)
            except ValueError:
                logger.warning(f"无效的恢复策略: {strategy}")
        
        if not recovery_strategy:
            # 自动选择策略
            if chain.steps:
                last_step = chain.steps[-1]
                recovery_strategy = await self.recovery_manager.handle_failure(
                    last_step, chain
                )
        
        if recovery_strategy:
            success = await self.recovery_manager.execute_recovery(
                recovery_strategy, chain
            )
            if success:
                await self._save_chain(chain, user_id)
            return success
        
        return False
    
    async def delete_chain(
        self,
        chain_id: UUID,
        user_id: Optional[str] = None
    ) -> bool:
        """删除推理链"""
        async with get_session() as session:
            query = select(ReasoningChainModel).where(
                ReasoningChainModel.id == chain_id
            )
            
            if user_id:
                query = query.where(
                    ReasoningChainModel.session_id == user_id
                )
            
            result = await session.execute(query)
            chain_model = result.scalar_one_or_none()
            
            if chain_model:
                await session.delete(chain_model)
                await session.commit()
                
                # 清除缓存
                await self._clear_cache(chain_id)
                
                return True
            
            return False
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        async with get_session() as session:
            # 统计总数
            count_query = (
                select(ReasoningChainModel)
                .where(ReasoningChainModel.session_id == user_id)
            )
            count_result = await session.execute(count_query)
            total_chains = len(count_result.scalars().all())
            
            # 统计完成数
            completed_query = (
                select(ReasoningChainModel)
                .where(
                    and_(
                        ReasoningChainModel.session_id == user_id,
                        ReasoningChainModel.completed_at.isnot(None)
                    )
                )
            )
            completed_result = await session.execute(completed_query)
            completed_chains = len(completed_result.scalars().all())
            
            # 平均置信度
            avg_confidence = 0.0
            if total_chains > 0:
                all_chains = await self.get_user_history(user_id, limit=100)
                confidences = [
                    c.confidence_score for c in all_chains 
                    if c.confidence_score is not None
                ]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            return {
                "total_chains": total_chains,
                "completed_chains": completed_chains,
                "completion_rate": completed_chains / total_chains if total_chains > 0 else 0,
                "average_confidence": avg_confidence,
                "recovery_stats": self.recovery_manager.get_recovery_stats()
            }
    
    async def _save_chain(self, chain: ReasoningChain, user_id: Optional[str]) -> None:
        """保存推理链到数据库"""
        # 简化实现，实际应该保存到数据库
        logger.info(f"保存推理链: {chain.id}")
    
    def _chain_to_response(self, chain: ReasoningChain) -> ReasoningResponse:
        """转换推理链为响应"""
        return ReasoningResponse(
            chain_id=chain.id,
            problem=chain.problem,
            strategy=chain.strategy,
            steps=chain.steps,
            conclusion=chain.conclusion,
            confidence=chain.confidence_score,
            duration_ms=chain.total_duration_ms,
            success=bool(chain.conclusion)
        )
    
    def _model_to_chain(self, model: ReasoningChainModel) -> ReasoningChain:
        """转换数据库模型为推理链"""
        chain = ReasoningChain(
            id=model.id,
            strategy=ReasoningStrategy(model.strategy),
            problem=model.problem,
            context=model.context,
            conclusion=model.conclusion,
            confidence_score=model.confidence_score,
            total_duration_ms=model.total_duration_ms,
            created_at=model.created_at,
            updated_at=model.updated_at,
            completed_at=model.completed_at
        )
        
        # 恢复步骤
        for step_model in model.steps:
            step = ThoughtStep(
                id=step_model.id,
                step_number=step_model.step_number,
                step_type=step_model.step_type,
                content=step_model.content,
                reasoning=step_model.reasoning,
                confidence=step_model.confidence,
                duration_ms=step_model.duration_ms,
                created_at=step_model.created_at
            )
            chain.steps.append(step)
        
        return chain
    
    async def _get_cache_key(self, request: ReasoningRequest) -> str:
        """生成缓存键"""
        from ai.reasoning.cot_engine import generate_cache_key
        return generate_cache_key(
            request.problem,
            request.strategy,
            request.context
        )
    
    async def _get_cached_result(self, cache_key: str) -> Optional[ReasoningResponse]:
        """获取缓存结果"""
        try:
            redis = await self._get_redis()
            if redis:
                cached = await redis.get(f"reasoning:{cache_key}")
                if cached:
                    import json
                    data = json.loads(cached)
                    return ReasoningResponse(**data)
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, response: ReasoningResponse) -> None:
        """缓存结果"""
        try:
            redis = await self._get_redis()
            if redis:
                import json
                data = json.dumps(response.dict(), default=str)
                await redis.setex(
                    f"reasoning:{cache_key}",
                    3600,  # 1小时TTL
                    data
                )
        except Exception as e:
            logger.warning(f"缓存结果失败: {e}")
    
    async def _clear_cache(self, chain_id: UUID) -> None:
        """清除缓存"""
        try:
            redis = await self._get_redis()
            if redis:
                # 清除相关缓存
                pattern = f"reasoning:*{chain_id}*"
                async for key in redis.scan_iter(match=pattern):
                    await redis.delete(key)
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")
    
    def _convert_offline_to_online_response(
        self, 
        offline_chain,
        request: ReasoningRequest
    ) -> ReasoningResponse:
        """将离线推理链转换为在线响应格式"""
        # 转换步骤
        steps = []
        for i, step_result in enumerate(offline_chain.steps):
            thought_step = ThoughtStep(
                id=uuid4(),
                step_number=i + 1,
                step_type=step_result.step_type.value,
                content=step_result.reasoning_text,
                reasoning=step_result.reasoning_text,
                confidence=step_result.confidence_score,
                duration_ms=step_result.execution_time_ms,
                created_at=step_result.timestamp
            )
            steps.append(thought_step)
        
        return ReasoningResponse(
            chain_id=UUID(offline_chain.chain_id),
            problem=request.problem,
            strategy=request.strategy,
            steps=steps,
            conclusion=offline_chain.final_conclusion,
            confidence=offline_chain.overall_confidence,
            duration_ms=offline_chain.total_execution_time_ms,
            success=offline_chain.status.value == "completed",
            metadata={
                "offline_mode": True,
                "offline_strategy": offline_chain.strategy.value,
                "total_tokens": offline_chain.total_tokens_used
            }
        )
    
    # 离线推理API方法
    
    async def get_offline_workflow_status(self, workflow_id: str) -> Optional[str]:
        """获取离线工作流状态"""
        if self._offline_engine:
            status = self._offline_engine.get_workflow_status(workflow_id)
            return status.value if status else None
        return None
    
    async def list_offline_workflows(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出离线工作流"""
        if self._offline_engine:
            return self._offline_engine.list_workflows(session_id)
        return []
    
    async def pause_offline_workflow(self, workflow_id: str) -> bool:
        """暂停离线工作流"""
        if self._offline_engine:
            return await self._offline_engine.pause_workflow(workflow_id)
        return False
    
    async def resume_offline_workflow(self, workflow_id: str) -> bool:
        """恢复离线工作流"""
        if self._offline_engine:
            return await self._offline_engine.resume_workflow(workflow_id)
        return False
    
    async def cancel_offline_workflow(self, workflow_id: str) -> bool:
        """取消离线工作流"""
        if self._offline_engine:
            return await self._offline_engine.cancel_workflow(workflow_id)
        return False
    
    def get_offline_reasoning_stats(self) -> Dict[str, Any]:
        """获取离线推理统计"""
        if self._offline_engine:
            offline_stats = self._offline_engine.get_reasoning_statistics()
            inference_stats = self._offline_engine.inference_engine.get_inference_stats()
            memory_stats = self._offline_engine.memory_manager.get_memory_stats()
            
            return {
                "offline_reasoning": offline_stats,
                "inference_engine": inference_stats,
                "memory_system": memory_stats,
                "mode": self.offline_mode.value,
                "network_status": self.network_status.value
            }
        return {
            "mode": self.offline_mode.value,
            "network_status": self.network_status.value,
            "offline_engine_initialized": False
        }