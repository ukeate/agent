"""
API v1路由模块
"""

from fastapi import APIRouter

from .auth import router as auth_router
from .security import router as security_router
from .test import router as test_router
from .mcp import router as mcp_router
from .agents import router as agents_router
from .agent_interface import router as agent_interface_router
from .multi_agents import router as multi_agents_router
from .async_agents import router as async_agents_router
from .workflows import router as workflows_router
from .supervisor import router as supervisor_router
from .rag import router as rag_router
from .cache import router as cache_router
from .events import router as events_router
from .streaming import router as streaming_router
from .batch import router as batch_router
# from .health import router as health_router  # 临时禁用
from .pgvector import router as pgvector_router
# from .multimodal import router as multimodal_router  # 临时禁用
# from .multimodal_rag import router as multimodal_rag_router  # 临时禁用，有依赖问题
from .memory_management import router as memory_management_router
# from .memory_analytics import router as memory_analytics_router  # 重复路由，已合并到memory_management
from .multi_step_reasoning import router as multi_step_reasoning_router
from .explainable_ai import router as explainable_ai_router
# from .documents import router as documents_router  # Temporarily disabled due to missing dependencies
# from .knowledge_extraction import router as knowledge_extraction_router  # 暂时禁用，缺少spacy依赖
# from .knowledge_graph import router as knowledge_graph_router  # 暂时禁用，缺少spacy依赖
# from .knowledge_graph_reasoning import router as knowledge_graph_reasoning_router  # 暂时禁用，缺少spacy依赖  
# from .knowledge_management import router as knowledge_management_router  # 暂时禁用，缺少spacy依赖
# from .graphrag import router as graphrag_router  # 暂时禁用，缺少openai_client依赖
# 暂时禁用有问题的模块导入
from .files import router as files_router
# from .offline import router as offline_router
# from .unified import router as unified_router
from .langgraph_features import router as langgraph_features_router
from .bandit_recommendations import router as bandit_recommendations_router
# from .qlearning import router as qlearning_router  # 暂时禁用，缺少tensorflow依赖
from .feedback import router as feedback_router
from .event_tracking import router as event_tracking_router
from .event_batch import router as event_batch_router
from .statistical_analysis import router as statistical_analysis_router
from .hypothesis_testing import router as hypothesis_testing_router
from .power_analysis import router as power_analysis_router
from .multiple_testing_correction import router as multiple_testing_correction_router
from .realtime_metrics import router as realtime_metrics_router
from .report_generation import router as report_generation_router
from .anomaly_detection import router as anomaly_detection_router
from .alert_rules import router as alert_rules_router
from .traffic_ramp import router as traffic_ramp_router
from .auto_scaling import router as auto_scaling_router
from .risk_assessment import router as risk_assessment_router
from .release_strategy import router as release_strategy_router
from .model_compression import router as model_compression_router
from .hyperparameter_optimization import router as hyperparameter_optimization_router
from .training_data import router as training_data_router
from .platform_integration import router as platform_integration_router
from .service_discovery import router as service_discovery_router
from .distributed_task import router as distributed_task_router
from .cluster_management import router as cluster_management_router
from .fault_tolerance import router as fault_tolerance_router
from .emotion_recognition import router as emotion_recognition_router
from .empathy_response import router as empathy_response_router
from .emotional_memory import router as emotional_memory_router
from .emotional_intelligence import router as emotional_intelligence_router
# from .emotion_websocket import router as emotion_websocket_router  # 临时禁用，有导入问题
# from .social_emotional_understanding import router as social_emotional_understanding_router  # 临时禁用，有导入问题
# from .social_emotion_api import router as social_emotion_api_router  # 临时禁用，有导入问题
# from .enterprise import router as enterprise_router  # 暂时禁用，有循环导入问题
# from .analytics import router as analytics_router  # 暂时禁用，导入太慢

# 创建v1 API路由器
v1_router = APIRouter(prefix="/api/v1")

# 注册子路由
v1_router.include_router(auth_router)
v1_router.include_router(security_router)
v1_router.include_router(test_router)
v1_router.include_router(mcp_router)
v1_router.include_router(agents_router)
v1_router.include_router(agent_interface_router)
v1_router.include_router(multi_agents_router)
v1_router.include_router(async_agents_router)
v1_router.include_router(workflows_router)
v1_router.include_router(supervisor_router)
v1_router.include_router(rag_router)
v1_router.include_router(cache_router)
v1_router.include_router(events_router)
v1_router.include_router(streaming_router)
v1_router.include_router(batch_router)
# v1_router.include_router(health_router)  # 临时禁用，变量未定义
v1_router.include_router(pgvector_router)
# v1_router.include_router(multimodal_router)  # 临时禁用
# v1_router.include_router(multimodal_rag_router)  # 临时禁用
v1_router.include_router(memory_management_router)
# v1_router.include_router(memory_analytics_router)  # 重复路由，已合并到memory_management
v1_router.include_router(multi_step_reasoning_router)
v1_router.include_router(explainable_ai_router)
# v1_router.include_router(documents_router)  # Temporarily disabled due to missing dependencies
# v1_router.include_router(knowledge_extraction_router)  # 暂时禁用，缺少spacy依赖
# v1_router.include_router(knowledge_graph_router)  # 暂时禁用，缺少spacy依赖
# v1_router.include_router(knowledge_graph_reasoning_router)  # 暂时禁用，缺少spacy依赖
# v1_router.include_router(knowledge_management_router)  # 暂时禁用，缺少spacy依赖
# v1_router.include_router(graphrag_router)  # 暂时禁用，缺少openai_client依赖
# 暂时禁用有问题的路由注册
v1_router.include_router(files_router)
# v1_router.include_router(offline_router)
# v1_router.include_router(unified_router)
v1_router.include_router(langgraph_features_router)
v1_router.include_router(bandit_recommendations_router, prefix="/bandit", tags=["Bandit Recommendations"])
# v1_router.include_router(qlearning_router)  # 暂时禁用，缺少tensorflow依赖
v1_router.include_router(feedback_router)
v1_router.include_router(event_tracking_router)
v1_router.include_router(event_batch_router)
v1_router.include_router(statistical_analysis_router)
v1_router.include_router(hypothesis_testing_router)
v1_router.include_router(power_analysis_router)
v1_router.include_router(multiple_testing_correction_router)
v1_router.include_router(realtime_metrics_router)
v1_router.include_router(report_generation_router)
v1_router.include_router(anomaly_detection_router)
v1_router.include_router(alert_rules_router)
v1_router.include_router(traffic_ramp_router)
v1_router.include_router(auto_scaling_router)
v1_router.include_router(risk_assessment_router)
v1_router.include_router(release_strategy_router)
v1_router.include_router(model_compression_router)
v1_router.include_router(hyperparameter_optimization_router)
v1_router.include_router(training_data_router)
v1_router.include_router(platform_integration_router)
v1_router.include_router(service_discovery_router)
v1_router.include_router(distributed_task_router)
v1_router.include_router(cluster_management_router)
v1_router.include_router(fault_tolerance_router)
v1_router.include_router(emotion_recognition_router)
v1_router.include_router(empathy_response_router)
v1_router.include_router(emotional_memory_router)
v1_router.include_router(emotional_intelligence_router)
# v1_router.include_router(emotion_websocket_router)  # 临时禁用
# v1_router.include_router(social_emotional_understanding_router)  # 临时禁用
# v1_router.include_router(social_emotion_api_router)  # 临时禁用
# v1_router.include_router(enterprise_router)  # 暂时禁用
# v1_router.include_router(analytics_router)  # 暂时禁用

__all__ = ["v1_router"]
