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
from .multimodal import router as multimodal_router
from .multimodal_rag import router as multimodal_rag_router
from .memory_management import router as memory_management_router
from .memory_analytics import router as memory_analytics_router
from .multi_step_reasoning import router as multi_step_reasoning_router
from .explainable_ai import router as explainable_ai_router
from .documents import router as documents_router
# 暂时禁用有问题的模块导入
# from .files import router as files_router
# from .offline import router as offline_router
# from .unified import router as unified_router
from .langgraph_features import router as langgraph_features_router
from .bandit_recommendations import router as bandit_recommendations_router
from .qlearning import router as qlearning_router
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
from .fastapi_features import router as fastapi_features_router
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
v1_router.include_router(multimodal_router)
v1_router.include_router(multimodal_rag_router)
v1_router.include_router(memory_management_router)
v1_router.include_router(memory_analytics_router)
v1_router.include_router(multi_step_reasoning_router)
v1_router.include_router(explainable_ai_router)
v1_router.include_router(documents_router)
# 暂时禁用有问题的路由注册
# v1_router.include_router(files_router)
# v1_router.include_router(offline_router)
# v1_router.include_router(unified_router)
v1_router.include_router(langgraph_features_router)
v1_router.include_router(bandit_recommendations_router, prefix="/bandit", tags=["Bandit Recommendations"])
v1_router.include_router(qlearning_router)
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
v1_router.include_router(fastapi_features_router)
# v1_router.include_router(enterprise_router)  # 暂时禁用
# v1_router.include_router(analytics_router)  # 暂时禁用

__all__ = ["v1_router"]
