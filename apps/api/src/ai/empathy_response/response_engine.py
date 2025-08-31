"""
共情响应生成引擎

统筹协调整个共情响应生成流程
"""
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .models import (
    EmpathyRequest, EmpathyResponse, DialogueContext, 
    EmpathyType, CulturalContext, ResponseTone
)
from .strategy_selector import StrategySelector
from .quality_assessor import QualityAssessor
from .context_manager import ContextManager
from .personalization_engine import PersonalizationEngine
from ..emotion_modeling.models import EmotionState, PersonalityProfile
from ..emotion_recognition.models.emotion_models import MultiModalEmotion

logger = logging.getLogger(__name__)


class EmpathyResponseEngine:
    """共情响应生成引擎"""
    
    def __init__(self):
        """初始化响应生成引擎"""
        self.strategy_selector = StrategySelector()
        self.quality_assessor = QualityAssessor()
        self.context_manager = ContextManager()
        self.personalization_engine = PersonalizationEngine()
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "strategy_usage": {
                "cognitive": 0,
                "affective": 0,
                "compassionate": 0
            }
        }
        
        # 配置参数
        self.config = {
            "max_response_length": 300,
            "min_response_length": 20,
            "target_response_time_ms": 300,
            "quality_threshold": 0.7,
            "max_retries": 2
        }
    
    def generate_response(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext] = None
    ) -> EmpathyResponse:
        """
        生成共情响应
        
        Args:
            request: 共情请求
            context: 对话上下文
            
        Returns:
            EmpathyResponse: 生成的共情响应
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # 1. 预处理和验证
            request = self._preprocess_request(request)
            context = self._prepare_context(request, context)
            
            # 2. 选择策略
            selected_strategy = self.strategy_selector.select_best_strategy(
                emotion_state=request.emotion_state,
                personality=request.personality_profile,
                context=context,
                preferred_type=request.preferred_empathy_type
            )
            
            # 3. 生成初始响应
            initial_response = selected_strategy.generate_response(request, context)
            
            # 4. 个性化优化
            personalized_response = self.personalization_engine.personalize_response(
                initial_response, request, context
            )
            
            # 5. 质量评估和优化
            final_response = self._optimize_response(
                personalized_response, request, context
            )
            
            # 6. 更新上下文
            if context:
                context.add_response(final_response)
                self.context_manager.update_context(context)
            
            # 7. 更新统计信息
            self._update_stats(final_response, time.time() - start_time)
            
            self.stats["successful_responses"] += 1
            logger.info(
                f"Successfully generated empathy response. Strategy: {final_response.empathy_type.value}, "
                f"Time: {final_response.generation_time_ms:.2f}ms"
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating empathy response: {e}", exc_info=True)
            # 返回错误恢复响应
            return self._create_fallback_response(request, str(e))
    
    def batch_generate_responses(
        self,
        requests: List[EmpathyRequest]
    ) -> List[EmpathyResponse]:
        """
        批量生成共情响应
        
        Args:
            requests: 响应请求列表
            
        Returns:
            List[EmpathyResponse]: 生成的响应列表
        """
        responses = []
        
        for request in requests:
            try:
                # 获取或创建上下文
                context = self.context_manager.get_or_create_context(
                    request.user_id, request.dialogue_context
                )
                
                response = self.generate_response(request, context)
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                fallback_response = self._create_fallback_response(request, str(e))
                responses.append(fallback_response)
        
        return responses
    
    def _preprocess_request(self, request: EmpathyRequest) -> EmpathyRequest:
        """预处理请求"""
        # 确保有基本的情感状态
        if not request.emotion_state:
            if request.multimodal_emotion:
                # 从多模态情感创建情感状态
                request.emotion_state = EmotionState(
                    user_id=request.user_id,
                    emotion=request.multimodal_emotion.primary_emotion,
                    intensity=request.multimodal_emotion.intensity_level,
                    valence=request.multimodal_emotion.valence,
                    arousal=request.multimodal_emotion.arousal,
                    dominance=request.multimodal_emotion.dominance,
                    confidence=request.multimodal_emotion.overall_confidence,
                    source="multimodal"
                )
            else:
                # 创建默认的中性情感状态
                request.emotion_state = EmotionState(
                    user_id=request.user_id,
                    emotion="neutral",
                    intensity=0.5,
                    source="default"
                )
        
        # 设置默认配置
        if request.max_response_length <= 0:
            request.max_response_length = self.config["max_response_length"]
            
        if request.max_generation_time_ms <= 0:
            request.max_generation_time_ms = self.config["target_response_time_ms"]
        
        return request
    
    def _prepare_context(
        self,
        request: EmpathyRequest,
        context: Optional[DialogueContext]
    ) -> Optional[DialogueContext]:
        """准备对话上下文"""
        if not context:
            # 尝试从上下文管理器获取
            context = self.context_manager.get_context(request.user_id)
        
        if context:
            # 更新上下文中的情感状态
            if request.emotion_state:
                context.add_emotion(request.emotion_state)
        else:
            # 创建新的上下文
            context = self.context_manager.create_context(
                user_id=request.user_id,
                conversation_id=getattr(request, 'conversation_id', f"conv_{int(time.time())}")
            )
            
            if request.emotion_state:
                context.add_emotion(request.emotion_state)
        
        return context
    
    def _optimize_response(
        self,
        response: EmpathyResponse,
        request: EmpathyRequest,
        context: Optional[DialogueContext]
    ) -> EmpathyResponse:
        """优化响应质量"""
        # 质量评估
        quality_score = self.quality_assessor.assess_response(response, request, context)
        
        # 如果质量不达标，尝试优化
        if quality_score < self.config["quality_threshold"]:
            logger.warning(f"Response quality below threshold: {quality_score}")
            
            # 尝试不同的策略或重新生成
            for retry in range(self.config["max_retries"]):
                try:
                    # 获取次优策略
                    rankings = self.strategy_selector.get_strategy_rankings(
                        request.emotion_state,
                        request.personality_profile,
                        context
                    )
                    
                    if len(rankings) > retry + 1:
                        fallback_strategy = self.strategy_selector.get_strategy(rankings[retry + 1][0])
                        improved_response = fallback_strategy.generate_response(request, context)
                        
                        improved_quality = self.quality_assessor.assess_response(
                            improved_response, request, context
                        )
                        
                        if improved_quality > quality_score:
                            logger.info(f"Improved response quality: {improved_quality}")
                            response = improved_response
                            break
                            
                except Exception as e:
                    logger.error(f"Error in response optimization retry {retry}: {e}")
                    continue
        
        # 最终格式化和验证
        response = self._finalize_response(response, request)
        
        return response
    
    def _finalize_response(self, response: EmpathyResponse, request: EmpathyRequest) -> EmpathyResponse:
        """最终化响应"""
        # 长度控制
        max_length = min(request.max_response_length, self.config["max_response_length"])
        if len(response.response_text) > max_length:
            # 智能截断，保持语句完整
            truncated_text = response.response_text[:max_length]
            last_period = truncated_text.rfind('。')
            last_exclamation = truncated_text.rfind('！')
            last_question = truncated_text.rfind('？')
            
            last_punct = max(last_period, last_exclamation, last_question)
            
            if last_punct > max_length * 0.7:  # 如果截断点合理
                response.response_text = truncated_text[:last_punct + 1]
            else:
                response.response_text = truncated_text + "..."
        
        # 确保最小长度
        if len(response.response_text) < self.config["min_response_length"]:
            response.response_text += " 我在这里陪伴和支持你。"
        
        # 添加元数据
        response.metadata.update({
            "engine_version": "1.0.0",
            "processing_timestamp": datetime.now().isoformat(),
            "request_id": getattr(request, 'request_id', None)
        })
        
        return response
    
    def _create_fallback_response(self, request: EmpathyRequest, error_msg: str) -> EmpathyResponse:
        """创建错误恢复响应"""
        fallback_text = "我能感受到你想要分享的情感，虽然现在我可能无法完全理解，但我在这里支持你。"
        
        return EmpathyResponse(
            response_text=fallback_text,
            empathy_type=EmpathyType.COGNITIVE,
            emotion_addressed="unknown",
            comfort_level=0.6,
            personalization_score=0.0,
            confidence=0.5,
            tone=ResponseTone.SUPPORTIVE,
            metadata={
                "fallback": True,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _update_stats(self, response: EmpathyResponse, processing_time: float):
        """更新统计信息"""
        # 更新平均响应时间
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_response_time"]
        new_avg = ((current_avg * (total_requests - 1)) + (processing_time * 1000)) / total_requests
        self.stats["average_response_time"] = new_avg
        
        # 更新策略使用统计
        strategy_name = response.empathy_type.value
        self.stats["strategy_usage"][strategy_name] += 1
        
        # 记录响应时间
        response.generation_time_ms = processing_time * 1000
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        success_rate = (
            self.stats["successful_responses"] / self.stats["total_requests"] 
            if self.stats["total_requests"] > 0 else 0
        )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "average_response_time_ms": round(self.stats["average_response_time"], 2)
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        logger.info(f"Updated engine configuration: {new_config}")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "strategy_usage": {
                "cognitive": 0,
                "affective": 0,
                "compassionate": 0
            }
        }
        logger.info("Performance statistics reset")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基本功能
            test_request = EmpathyRequest(
                user_id="health_check",
                message="测试消息",
                emotion_state=EmotionState(emotion="neutral", intensity=0.5)
            )
            
            start_time = time.time()
            test_response = self.generate_response(test_request)
            response_time = (time.time() - start_time) * 1000
            
            is_healthy = (
                test_response is not None and
                response_time < self.config["target_response_time_ms"] * 2 and
                len(test_response.response_text) > 0
            )
            
            return {
                "healthy": is_healthy,
                "response_time_ms": round(response_time, 2),
                "test_response_length": len(test_response.response_text),
                "timestamp": datetime.now().isoformat(),
                "stats": self.get_performance_stats()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }