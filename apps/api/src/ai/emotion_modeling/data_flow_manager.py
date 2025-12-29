"""
统一情感数据流管理器
负责协调各个情感智能模块之间的数据流转和状态同步
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import timedelta
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from .core_interfaces import (
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    EmotionalDataFlowManager, EmotionRecognitionEngine,
    EmotionStateModeler, EmpathyResponseGenerator,
    EmotionalMemoryManager, EmotionalIntelligenceDecisionEngine,
    SocialEmotionalAnalyzer, EmotionState, MultiModalEmotion,
    PersonalityProfile, EmpathyResponse, EmotionalMemory,
    DecisionContext, RiskAssessment, SocialContext,
    GroupEmotionalState, ModalityType
)
from .communication_protocol import (
    CommunicationProtocol, ModuleType, MessageType, Priority
)

from src.core.logging import get_logger
class DataFlowState:
    """数据流状态"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class DataFlowContext:
    """数据流上下文"""
    
    def __init__(self, request_id: str, user_id: str, input_data: Dict[str, Any]):
        self.request_id = request_id
        self.user_id = user_id
        self.input_data = input_data
        self.created_at = utc_now()
        self.state = DataFlowState.IDLE
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.processing_times: Dict[str, float] = {}

class EmotionalDataFlowManagerImpl(EmotionalDataFlowManager):
    """情感数据流管理器实现"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._protocol = CommunicationProtocol(ModuleType.DATA_FLOW_MANAGER)
        self._executor = ThreadPoolExecutor(max_workers=20)
        
        # 模块引用
        self._emotion_recognition: Optional[EmotionRecognitionEngine] = None
        self._state_modeler: Optional[EmotionStateModeler] = None
        self._empathy_generator: Optional[EmpathyResponseGenerator] = None
        self._memory_manager: Optional[EmotionalMemoryManager] = None
        self._decision_engine: Optional[EmotionalIntelligenceDecisionEngine] = None
        self._social_analyzer: Optional[SocialEmotionalAnalyzer] = None
        
        # 数据流管理
        self._active_flows: Dict[str, DataFlowContext] = {}
        self._flow_dependencies: Dict[str, List[str]] = {
            "emotion_recognition": [],
            "state_modeling": ["emotion_recognition"],
            "empathy_generation": ["emotion_recognition", "state_modeling"],
            "memory_management": ["emotion_recognition", "state_modeling"],
            "decision_engine": ["state_modeling", "memory_management"],
            "social_analysis": ["emotion_recognition", "state_modeling"]
        }
        
        # 性能指标
        self._metrics = {
            "total_flows": 0,
            "successful_flows": 0,
            "failed_flows": 0,
            "average_processing_time": 0.0,
            "active_flows_count": 0,
            "throughput_per_second": 0.0
        }
        
        self._last_throughput_calculation = time.time()
        self._throughput_counter = 0
        self._is_running = False
    
    def set_modules(
        self,
        emotion_recognition: Optional[EmotionRecognitionEngine] = None,
        state_modeler: Optional[EmotionStateModeler] = None,
        empathy_generator: Optional[EmpathyResponseGenerator] = None,
        memory_manager: Optional[EmotionalMemoryManager] = None,
        decision_engine: Optional[EmotionalIntelligenceDecisionEngine] = None,
        social_analyzer: Optional[SocialEmotionalAnalyzer] = None
    ):
        """设置模块引用"""
        self._emotion_recognition = emotion_recognition
        self._state_modeler = state_modeler
        self._empathy_generator = empathy_generator
        self._memory_manager = memory_manager
        self._decision_engine = decision_engine
        self._social_analyzer = social_analyzer
    
    async def start(self):
        """启动数据流管理器"""
        self._is_running = True
        self._logger.info("Starting emotional data flow manager")
        
        # 启动协议服务
        asyncio.create_task(self._protocol.start())
        
        # 启动监控任务
        asyncio.create_task(self._monitor_flows())
        asyncio.create_task(self._calculate_throughput())
    
    async def stop(self):
        """停止数据流管理器"""
        self._is_running = False
        await self._protocol.stop()
        self._executor.shutdown(wait=True)
        self._logger.info("Emotional data flow manager stopped")
    
    async def route_data(self, data: UnifiedEmotionalData) -> Dict[str, Any]:
        """路由数据到相应模块"""
        request_id = f"{data.user_id}_{utc_now().timestamp()}"
        context = DataFlowContext(request_id, data.user_id, asdict(data))
        
        self._active_flows[request_id] = context
        self._metrics["total_flows"] += 1
        self._metrics["active_flows_count"] = len(self._active_flows)
        
        try:
            context.state = DataFlowState.PROCESSING
            start_time = time.time()
            
            # 执行数据流处理管道
            await self._execute_processing_pipeline(context, data)
            
            # 构建最终响应
            response_data = await self._build_unified_response(context)
            
            processing_time = time.time() - start_time
            context.processing_times["total"] = processing_time
            context.state = DataFlowState.COMPLETED
            
            # 更新指标
            self._metrics["successful_flows"] += 1
            self._update_average_processing_time(processing_time)
            self._throughput_counter += 1
            
            return response_data
            
        except Exception as e:
            self._logger.error(f"Data routing error: {e}")
            context.errors.append(str(e))
            context.state = DataFlowState.ERROR
            self._metrics["failed_flows"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }
        finally:
            # 清理完成的流
            self._active_flows.pop(request_id, None)
            self._metrics["active_flows_count"] = len(self._active_flows)
    
    async def _execute_processing_pipeline(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """执行处理管道"""
        
        # 第一阶段：多模态情感识别
        if self._emotion_recognition:
            await self._process_emotion_recognition(context, data)
        
        # 第二阶段：情感状态建模 (依赖情感识别结果)
        if self._state_modeler and "emotion_recognition" in context.results:
            await self._process_state_modeling(context, data)
        
        # 第三阶段：并行处理多个模块
        parallel_tasks = []
        
        # 共情响应生成
        if (self._empathy_generator and 
            "emotion_recognition" in context.results and 
            "state_modeling" in context.results):
            parallel_tasks.append(
                self._process_empathy_generation(context, data)
            )
        
        # 情感记忆管理
        if (self._memory_manager and 
            "emotion_recognition" in context.results and 
            "state_modeling" in context.results):
            parallel_tasks.append(
                self._process_memory_management(context, data)
            )
        
        # 社交情感分析
        if (self._social_analyzer and 
            "emotion_recognition" in context.results and 
            "state_modeling" in context.results):
            parallel_tasks.append(
                self._process_social_analysis(context, data)
            )
        
        # 执行并行任务
        if parallel_tasks:
            await asyncio.gather(*parallel_tasks, return_exceptions=True)
        
        # 第四阶段：智能决策 (依赖前面的结果)
        if (self._decision_engine and 
            "state_modeling" in context.results and 
            "memory_management" in context.results):
            await self._process_decision_engine(context, data)
    
    async def _process_emotion_recognition(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理情感识别"""
        try:
            start_time = time.time()
            
            # 构建输入数据
            input_data = {}
            if hasattr(data, 'input_data') and data.input_data:
                for modality, content in data.input_data.items():
                    if modality in ModalityType.__members__.values():
                        input_data[ModalityType(modality)] = content
            
            # 调用情感识别引擎
            recognition_result = await self._emotion_recognition.recognize_emotion(input_data)
            
            context.results["emotion_recognition"] = recognition_result
            context.processing_times["emotion_recognition"] = time.time() - start_time
            
            self._logger.debug(f"Emotion recognition completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"Emotion recognition error: {e}")
            context.errors.append(f"emotion_recognition: {str(e)}")
    
    async def _process_state_modeling(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理情感状态建模"""
        try:
            start_time = time.time()
            
            recognition_result = context.results.get("emotion_recognition")
            if recognition_result and hasattr(recognition_result, 'fused_emotion'):
                new_emotion = recognition_result.fused_emotion
                
                # 更新情感状态
                updated_state = await self._state_modeler.update_emotional_state(
                    context.user_id, new_emotion
                )
                
                # 获取个性画像
                personality = await self._state_modeler.get_personality_profile(
                    context.user_id
                )
                
                context.results["state_modeling"] = {
                    "emotional_state": updated_state,
                    "personality_profile": personality
                }
                
                context.processing_times["state_modeling"] = time.time() - start_time
                
                self._logger.debug(f"State modeling completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"State modeling error: {e}")
            context.errors.append(f"state_modeling: {str(e)}")
    
    async def _process_empathy_generation(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理共情响应生成"""
        try:
            start_time = time.time()
            
            state_result = context.results.get("state_modeling", {})
            emotional_state = state_result.get("emotional_state")
            
            if emotional_state:
                empathy_response = await self._empathy_generator.generate_empathy_response(
                    emotional_state,
                    context.input_data.get("context")
                )
                
                context.results["empathy_generation"] = empathy_response
                context.processing_times["empathy_generation"] = time.time() - start_time
                
                self._logger.debug(f"Empathy generation completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"Empathy generation error: {e}")
            context.errors.append(f"empathy_generation: {str(e)}")
    
    async def _process_memory_management(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理情感记忆管理"""
        try:
            start_time = time.time()
            
            state_result = context.results.get("state_modeling", {})
            emotional_state = state_result.get("emotional_state")
            
            if emotional_state:
                # 检索相关记忆
                relevant_memories = await self._memory_manager.retrieve_relevant_memories(
                    context.user_id, emotional_state
                )
                
                # 如果有新的情感体验，存储为记忆
                if hasattr(data, 'content') and data.content:
                    memory = EmotionalMemory(
                        memory_id=f"{context.user_id}_{utc_now().timestamp()}",
                        content=data.content,
                        emotional_context=emotional_state,
                        importance=emotional_state.intensity,
                        created_at=utc_now(),
                        last_accessed=utc_now()
                    )
                    memory_id = await self._memory_manager.store_emotional_memory(
                        context.user_id, memory
                    )
                
                context.results["memory_management"] = {
                    "relevant_memories": relevant_memories,
                    "new_memory_id": memory_id if 'memory_id' in locals() else None
                }
                
                context.processing_times["memory_management"] = time.time() - start_time
                
                self._logger.debug(f"Memory management completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"Memory management error: {e}")
            context.errors.append(f"memory_management: {str(e)}")
    
    async def _process_social_analysis(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理社交情感分析"""
        try:
            start_time = time.time()
            
            # 这里假设输入数据包含社交上下文信息
            social_context = context.input_data.get("social_context")
            if social_context:
                participants = social_context.get("participants", [context.user_id])
                conversation_data = social_context.get("conversation_data", {})
                
                # 分析社交动态
                social_dynamics = await self._social_analyzer.analyze_social_dynamics(
                    participants, conversation_data
                )
                
                # 检测群体情感
                individual_emotions = social_context.get("individual_emotions", {})
                if individual_emotions:
                    group_emotion = await self._social_analyzer.detect_group_emotion(
                        individual_emotions
                    )
                    
                    context.results["social_analysis"] = {
                        "social_dynamics": social_dynamics,
                        "group_emotion": group_emotion
                    }
                
                context.processing_times["social_analysis"] = time.time() - start_time
                
                self._logger.debug(f"Social analysis completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"Social analysis error: {e}")
            context.errors.append(f"social_analysis: {str(e)}")
    
    async def _process_decision_engine(
        self, 
        context: DataFlowContext, 
        data: UnifiedEmotionalData
    ):
        """处理智能决策"""
        try:
            start_time = time.time()
            
            state_result = context.results.get("state_modeling", {})
            emotional_state = state_result.get("emotional_state")
            
            if emotional_state:
                # 构建决策上下文
                decision_context = DecisionContext(
                    decision_type="emotional_response",
                    factors=context.results,
                    emotional_weight=0.7,
                    rational_weight=0.3
                )
                
                # 做出决策
                decision = await self._decision_engine.make_decision(
                    decision_context, emotional_state
                )
                
                # 评估风险
                risk_assessment = await self._decision_engine.assess_risk(
                    context.user_id, emotional_state, context.results
                )
                
                context.results["decision_engine"] = {
                    "decision": decision,
                    "risk_assessment": risk_assessment
                }
                
                context.processing_times["decision_engine"] = time.time() - start_time
                
                self._logger.debug(f"Decision engine completed for {context.request_id}")
            
        except Exception as e:
            self._logger.error(f"Decision engine error: {e}")
            context.errors.append(f"decision_engine: {str(e)}")
    
    async def _build_unified_response(self, context: DataFlowContext) -> Dict[str, Any]:
        """构建统一响应"""
        unified_data = UnifiedEmotionalData(
            user_id=context.user_id,
            timestamp=utc_now(),
            confidence=self._calculate_overall_confidence(context),
            processing_time=context.processing_times.get("total", 0.0),
            data_quality=self._calculate_data_quality(context)
        )
        
        # 填充各模块结果
        if "emotion_recognition" in context.results:
            unified_data.recognition_result = context.results["emotion_recognition"]
        
        if "state_modeling" in context.results:
            state_result = context.results["state_modeling"]
            unified_data.emotional_state = state_result.get("emotional_state")
            unified_data.personality_profile = state_result.get("personality_profile")
        
        if "empathy_generation" in context.results:
            unified_data.empathy_response = context.results["empathy_generation"]
        
        if "memory_management" in context.results:
            memory_result = context.results["memory_management"]
            if memory_result.get("relevant_memories"):
                unified_data.emotional_memory = memory_result["relevant_memories"][0]
                unified_data.memory_relevance = 1.0
        
        if "decision_engine" in context.results:
            decision_result = context.results["decision_engine"]
            unified_data.decision_context = DecisionContext(
                decision_type="emotional_response",
                factors=decision_result.get("decision", {}),
                emotional_weight=0.7,
                rational_weight=0.3
            )
            unified_data.risk_assessment = decision_result.get("risk_assessment")
        
        if "social_analysis" in context.results:
            social_result = context.results["social_analysis"]
            unified_data.social_context = social_result.get("social_dynamics")
            unified_data.group_emotion = social_result.get("group_emotion")
        
        response = EmotionalIntelligenceResponse(
            success=len(context.errors) == 0,
            data=unified_data,
            error={"errors": context.errors} if context.errors else None,
            metadata={
                "request_id": context.request_id,
                "processing_times": context.processing_times,
                "modules_processed": list(context.results.keys())
            }
        )
        
        return response.model_dump(mode="json")
    
    async def validate_data_integrity(self, data: UnifiedEmotionalData) -> bool:
        """验证数据完整性"""
        try:
            # 基本字段验证
            if not data.user_id:
                return False
            
            if not data.timestamp:
                return False
            
            if not (0.0 <= data.confidence <= 1.0):
                return False
            
            if not (0.0 <= data.data_quality <= 1.0):
                return False
            
            # 模块结果一致性验证
            if data.emotional_state and data.recognition_result:
                # 验证情感识别结果与情感状态的一致性
                if data.emotional_state.emotion != data.recognition_result.fused_emotion.emotion:
                    return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Data integrity validation error: {e}")
            return False
    
    async def synchronize_modules(self) -> bool:
        """同步各模块状态"""
        try:
            # 通过协议发送同步请求到各模块
            sync_tasks = []
            
            for module_type in ModuleType:
                if module_type != ModuleType.DATA_FLOW_MANAGER:
                    task = self._protocol.send_request(
                        target_module=module_type,
                        payload={"action": "sync_status"},
                        priority=Priority.HIGH,
                        timeout=10.0
                    )
                    sync_tasks.append(task)
            
            # 等待所有同步完成
            results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            
            # 检查同步结果
            success_count = sum(1 for result in results if result and not isinstance(result, Exception))
            total_modules = len(ModuleType) - 1  # 排除自己
            
            sync_rate = success_count / total_modules if total_modules > 0 else 0.0
            self._logger.info(f"Module synchronization completed: {success_count}/{total_modules} ({sync_rate:.1%})")
            
            return sync_rate >= 0.8  # 80%以上同步成功认为成功
            
        except Exception as e:
            self._logger.error(f"Module synchronization error: {e}")
            return False
    
    def _calculate_overall_confidence(self, context: DataFlowContext) -> float:
        """计算整体置信度"""
        confidences = []
        
        for result in context.results.values():
            if hasattr(result, 'confidence'):
                confidences.append(result.confidence)
            elif isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _calculate_data_quality(self, context: DataFlowContext) -> float:
        """计算数据质量"""
        total_modules = len(self._flow_dependencies)
        processed_modules = len(context.results)
        error_count = len(context.errors)
        
        module_completion_score = processed_modules / total_modules
        error_penalty = max(0, 1.0 - (error_count * 0.2))
        
        return module_completion_score * error_penalty
    
    def _update_average_processing_time(self, processing_time: float):
        """更新平均处理时间"""
        current_avg = self._metrics["average_processing_time"]
        total_flows = self._metrics["total_flows"]
        
        # 使用加权平均
        self._metrics["average_processing_time"] = (
            (current_avg * (total_flows - 1) + processing_time) / total_flows
        )
    
    async def _monitor_flows(self):
        """监控活跃数据流"""
        while self._is_running:
            try:
                current_time = utc_now()
                expired_flows = []
                
                # 检查超时的流
                for request_id, context in self._active_flows.items():
                    if (current_time - context.created_at).total_seconds() > 300:  # 5分钟超时
                        expired_flows.append(request_id)
                
                # 清理超时的流
                for request_id in expired_flows:
                    context = self._active_flows.pop(request_id, None)
                    if context:
                        self._logger.warning(f"Flow {request_id} expired and removed")
                        context.errors.append("Flow timeout")
                        self._metrics["failed_flows"] += 1
                
                self._metrics["active_flows_count"] = len(self._active_flows)
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self._logger.error(f"Flow monitoring error: {e}")
    
    async def _calculate_throughput(self):
        """计算吞吐量"""
        while self._is_running:
            try:
                await asyncio.sleep(10)  # 每10秒计算一次
                current_time = time.time()
                time_diff = current_time - self._last_throughput_calculation
                
                if time_diff > 0:
                    throughput = self._throughput_counter / time_diff
                    self._metrics["throughput_per_second"] = throughput
                    
                    # 重置计数器
                    self._throughput_counter = 0
                    self._last_throughput_calculation = current_time
                    
            except Exception as e:
                self._logger.error(f"Throughput calculation error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self._metrics,
            "protocol_metrics": self._protocol.get_metrics(),
            "active_flows": list(self._active_flows.keys()),
            "flow_states": {
                request_id: context.state 
                for request_id, context in self._active_flows.items()
            }
        }
