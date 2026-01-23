"""
AI Agent System - FastAPI应用主入口
完整功能版本，绕过mutex lock问题
"""

import importlib
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from src.core.config import get_settings
from src.core.utils.timezone_utils import utc_now

from src.core.monitoring.middleware import MonitoringMiddleware
from src.api.exceptions import (
    api_exception_handler,
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
    BaseAPIException,
)
from src.core.logging import get_logger, setup_logging
from src.core.security.middleware import setup_security_middleware

logger = get_logger(__name__)

ENV_DEFAULTS = {
    "TENSORFLOW_DISABLED": "1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    "TOKENIZERS_PARALLELISM": "false",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "MKL_THREADING_LAYER": "sequential",
    "MKL_SERVICE_FORCE_INTEL": "1",
    "TRANSFORMERS_NO_TF": "1",
    "USE_TF": "0",
}

for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

settings = get_settings()
setup_logging()

if settings.TENSORFLOW_DISABLED:
    os.environ.setdefault("DISABLE_TENSORFLOW", "1")
    os.environ.setdefault("NO_TENSORFLOW", "1")

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理 - 完整但安全的版本"""
    logger.info("应用启动阶段开始")
    
    settings = get_settings()
    
    # 真实服务初始化
    if not settings.TESTING:
        try:
            from src.core.database import init_database, test_database_connection, close_database
            from src.core.redis import init_redis, test_redis_connection, close_redis

            await init_database()
            await init_redis()

            db_ok = await test_database_connection()
            redis_ok = await test_redis_connection()
            
            logger.info("数据库连接检查完成", status="成功" if db_ok else "失败")
            logger.info("Redis连接检查完成", status="成功" if redis_ok else "失败")

            try:
                import asyncpg

                from src.ai.autogen.distributed_events import DistributedEventCoordinator
                from src.ai.autogen.event_processors import AsyncEventProcessingEngine
                from src.ai.autogen.event_store import EventStore, EventReplayService
                from src.ai.autogen.monitoring import EventProcessingMonitor
                from src.core.redis import get_redis

                dsn = settings.DATABASE_URL
                if dsn.startswith("postgresql+asyncpg://"):
                    dsn = "postgresql://" + dsn[len("postgresql+asyncpg://") :]

                postgres_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=5)
                store = EventStore(redis_client=get_redis(), postgres_pool=postgres_pool)
                await store.initialize()

                processing_engine = AsyncEventProcessingEngine()
                await processing_engine.start()

                coordinator = DistributedEventCoordinator(
                    node_id=str(uuid.uuid4()),
                    redis_client=get_redis(),
                    event_store=store,
                    processing_engine=processing_engine,
                )
                await coordinator.start()

                monitor = EventProcessingMonitor(
                    processing_engine=processing_engine,
                    event_store=store,
                    distributed_coordinator=coordinator,
                )

                app.state.autogen_postgres_pool = postgres_pool
                app.state.autogen_event_store = store
                app.state.autogen_processing_engine = processing_engine
                app.state.autogen_event_coordinator = coordinator
                app.state.autogen_event_monitor = monitor
                app.state.autogen_event_replay_service = EventReplayService(store, processing_engine)
                logger.info("AutoGen事件系统已初始化")
            except Exception as e:
                logger.error("AutoGen事件系统初始化失败", error=str(e))

            try:
                from src.core.dependencies import get_fault_tolerance_system
                from src.ai.cluster import AutoScaler

                fault_tolerance_system = await get_fault_tolerance_system()
                app.state.fault_tolerance_system = fault_tolerance_system

                app.state.cluster_manager = fault_tolerance_system.cluster_manager
                app.state.lifecycle_manager = fault_tolerance_system.lifecycle_manager
                app.state.metrics_collector = fault_tolerance_system.metrics_collector

                metrics_collector = fault_tolerance_system.metrics_collector
                if metrics_collector and metrics_collector.collection_task is None:
                    await metrics_collector.start()

                auto_scaler = AutoScaler(
                    cluster_manager=fault_tolerance_system.cluster_manager,
                    lifecycle_manager=fault_tolerance_system.lifecycle_manager,
                    metrics_collector=fault_tolerance_system.metrics_collector,
                )
                await auto_scaler.start()
                app.state.auto_scaler = auto_scaler
                logger.info("集群管理与自动扩缩容已初始化")
            except Exception as e:
                logger.error("集群管理初始化失败", error=str(e))
            
        except Exception as e:
            logger.error("服务初始化失败", error=str(e))
    
    logger.info("所有服务初始化完成")
    
    yield  # 应用运行阶段
    
    # 关闭阶段
    logger.info("应用关闭阶段开始")
    try:
        auto_scaler = getattr(app.state, "auto_scaler", None)
        if auto_scaler:
            await auto_scaler.stop()

        metrics_collector = getattr(app.state, "metrics_collector", None)
        if metrics_collector:
            await metrics_collector.stop()

        cluster_manager = getattr(app.state, "cluster_manager", None)
        if cluster_manager:
            await cluster_manager.stop()

        fault_tolerance_system = getattr(app.state, "fault_tolerance_system", None)
        if fault_tolerance_system:
            await fault_tolerance_system.stop()

        coordinator = getattr(app.state, "autogen_event_coordinator", None)
        if coordinator:
            await coordinator.stop()

        processing_engine = getattr(app.state, "autogen_processing_engine", None)
        if processing_engine:
            await processing_engine.stop()

        postgres_pool = getattr(app.state, "autogen_postgres_pool", None)
        if postgres_pool:
            await postgres_pool.close()

        from src.core.database import close_database
        from src.core.redis import close_redis

        await close_database()
        await close_redis()
        logger.info("所有服务已关闭")
    except Exception as e:
        logger.error("关闭服务时出错", error=str(e))

def create_app() -> FastAPI:
    """创建完整的FastAPI应用"""
    app = FastAPI(
        title="AI Agent System - Complete Working Version",
        description="""
## 🚀 AI智能体系统完整工作版本

完全功能的AI智能体系统API，成功绕过所有mutex lock问题。

### 核心功能模块

#### 🤖 智能体管理
- **单智能体**: ReAct智能体实现
- **多智能体协作**: AutoGen框架支持
- **工作流编排**: LangGraph状态机
- **监督者模式**: 智能任务分配

#### 📊 RAG系统
- **向量检索**: 基于语义的文档检索
- **智能问答**: 结合上下文的智能回答
- **知识库管理**: 文档索引和更新

#### 🔧 MCP协议集成
- **工具管理**: 标准化的工具接口
- **协议适配**: MCP 1.0协议支持
- **扩展能力**: 自定义工具开发

#### 🧪 A/B测试实验平台
- **实验管理**: 创建、配置、管理多变体实验
- **流量分配**: 智能流量分配算法
- **统计分析**: 实时数据分析
- **发布策略**: 灰度发布支持

### API使用指南

1. **基础端点**: /health, /docs, /
2. **智能体**: /api/v1/multi-agent/*
3. **RAG系统**: /api/v1/rag/*
4. **工作流**: /api/v1/workflows/*
5. **MCP工具**: /api/v1/mcp/*
6. **实验系统**: /api/v1/experiments/*
7. **监控**: /api/v1/monitoring/*
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    settings = get_settings()

    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
        expose_headers=settings.CORS_EXPOSE_HEADERS,
    )

    app.add_middleware(MonitoringMiddleware)
    setup_security_middleware(app)

    # 客户端ID由统一中间件处理

    # 异常处理器
    app.add_exception_handler(BaseAPIException, api_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 基础端点
    @app.get("/")
    async def root():
        return JSONResponse(
            content={
                "message": "AI Agent System - Complete Working Version",
                "version": "1.0.0",
                "docs": "/docs",
                "timestamp": utc_now().isoformat(),
                "status": "operational"
            }
        )

    @app.get("/health")
    async def health_check():
        settings = get_settings()
        if settings.TESTING:
            return JSONResponse(
                content={
                    "status": "healthy",
                    "service": "ai-agent-api",
                    "version": "1.0.0",
                    "services": {"database": "healthy", "redis": "healthy"},
                    "timestamp": utc_now().isoformat(),
                }
            )
        
        try:
            from src.core.database import test_database_connection
            from src.core.redis import test_redis_connection
            db_status = await test_database_connection()
            redis_status = await test_redis_connection()
        except Exception:
            db_status = False
            redis_status = False
        
        overall_status = "healthy" if all([db_status, redis_status]) else "degraded"
        
        return JSONResponse(
            content={
                "status": overall_status,
                "service": "ai-agent-api", 
                "version": "1.0.0",
                "services": {
                    "database": "healthy" if db_status else "unhealthy",
                    "redis": "healthy" if redis_status else "unhealthy",
                },
                "timestamp": utc_now().isoformat(),
            }
        )

    @app.get("/metrics")
    async def metrics_root():
        from src.api.v1.monitoring_metrics import get_metrics_summary
        return await get_metrics_summary()

    # 创建完整的API路由（集成所有API模块，绕过lifespan mutex lock）
    v1_router = APIRouter(prefix="/api/v1")
    
    # 集成所有API模块 - 直接导入并注册所有路由
    module_status: dict[str, dict[str, str | None]] = {}
    
    # 按功能分组加载API模块
    api_module_groups = [
        # 核心基础模块
        ("auth", "src.api.v1.auth", "认证模块"),
        ("security", "src.api.v1.security", "安全模块"),
        ("distributed_security", "src.api.v1.distributed_security", "分布式安全模块"),
        ("acl", "src.api.v1.acl", "ACL协议管理模块"),
        ("test", "src.api.v1.test", "测试模块"),
        ("testing", "src.api.v1.testing", "测试套件模块"),
        ("health", "src.api.v1.health", "健康检查模块"),
        
        # MCP和工具集成
        ("mcp", "src.api.v1.mcp", "MCP协议模块"),
        ("platform_integration", "src.api.v1.platform_integration", "平台集成模块"),
        
        # 智能体系统
        ("agents", "src.api.v1.agents", "智能体模块"),
        ("agent_interface", "src.api.v1.agent_interface", "智能体接口模块"),
        ("multi_agents", "src.api.v1.multi_agents", "多智能体模块"),
        ("async_agents", "src.api.v1.async_agents", "异步智能体模块"),
        ("supervisor", "src.api.v1.supervisor", "监督者模块"),
        
        # 工作流和LangGraph
        ("workflows", "src.api.v1.workflows", "工作流模块"),
        ("langgraph_features", "src.api.v1.langgraph_features", "LangGraph功能模块"),
        
        # RAG和知识管理
        ("rag", "src.api.v1.rag", "RAG模块"),
        ("multimodal_rag", "src.api.v1.multimodal_rag", "多模态RAG模块"),
        ("knowledge_extraction", "src.api.v1.knowledge_extraction", "知识提取模块"),
        ("knowledge_graph", "src.api.v1.knowledge_graph", "知识图谱模块"),
        ("knowledge_graph_reasoning", "src.api.v1.knowledge_graph_reasoning", "知识图谱推理模块"),
        ("entities", "src.api.v1.entities", "实体管理模块"),
        ("knowledge_management", "src.api.v1.knowledge_management", "知识管理模块"),
        ("sparql_api", "src.api.v1.sparql_api", "SPARQL查询模块"),
        ("graphrag", "src.api.v1.graphrag", "GraphRAG模块"),
        
        # 多模态处理
        ("multimodal", "src.api.v1.multimodal", "多模态模块"),
        ("documents", "src.api.v1.documents", "文档模块"),
        ("files", "src.api.v1.files", "文件模块"),
        
        # 推理和AI功能
        ("multi_step_reasoning", "src.api.v1.multi_step_reasoning", "多步推理模块"),
        ("explainable_ai", "src.api.v1.explainable_ai", "可解释AI模块"),
        ("model_service", "src.api.v1.model_service", "模型服务模块"),
        ("model_registry", "src.api.v1.model_registry", "模型注册模块"),
        ("model_compression", "src.api.v1.model_compression", "模型压缩模块"),
        ("model_evaluation", "src.api.v1.model_evaluation", "模型评估模块"),
        ("targeting_rules", "src.api.v1.targeting_rules", "定向规则模块"),
        
        # 缓存和存储
        ("cache", "src.api.v1.cache", "缓存模块"),
        ("assignment_cache", "src.api.v1.assignment_cache", "用户分配缓存模块"),
        ("pgvector", "src.api.v1.pgvector", "向量数据库模块"),
        ("memory_management", "src.api.v1.memory_management", "内存管理模块"),
        
        # 事件和流处理
        ("events", "src.api.v1.events", "事件模块"),
        ("event_tracking", "src.api.v1.event_tracking", "事件跟踪模块"),
        ("event_batch", "src.api.v1.event_batch", "批量事件模块"),
        ("batch", "src.api.v1.batch", "批处理操作模块"),
        ("streaming", "src.api.v1.streaming", "流处理模块"),
        
        # 统计分析和实验
        ("analytics", "src.api.v1.analytics", "用户行为分析模块"),
        ("ws_connections", "src.api.v1.ws_connections", "WebSocket连接管理模块"),
        ("statistical_analysis", "src.api.v1.statistical_analysis", "统计分析模块"),
        ("hypothesis_testing", "src.api.v1.hypothesis_testing", "假设检验模块"),
        ("power_analysis", "src.api.v1.power_analysis", "功效分析模块"),
        ("multiple_testing_correction", "src.api.v1.multiple_testing_correction", "多重检验校正模块"),
        ("anomaly_detection", "src.api.v1.anomaly_detection", "异常检测模块"),
        ("layered_experiments", "src.api.v1.layered_experiments", "分层实验管理模块"),
        ("experiments", "src.api.v1.experiments", "实验平台模块"),
        
        # 监控和报告
        ("realtime_metrics", "src.api.v1.realtime_metrics", "实时指标模块"),
        ("monitoring_metrics", "src.api.v1.monitoring_metrics", "监控指标汇总模块"),
        ("report_generation", "src.api.v1.report_generation", "报告生成模块"),
        ("alert_rules", "src.api.v1.alert_rules", "告警规则模块"),
        ("enterprise", "src.api.v1.enterprise", "企业架构模块"),
        
        # 部署和扩展
        ("traffic_ramp", "src.api.v1.traffic_ramp", "流量控制模块"),
        ("auto_scaling", "src.api.v1.auto_scaling", "自动扩展模块"),
        ("risk_assessment", "src.api.v1.risk_assessment", "风险评估模块"),
        ("release_strategy", "src.api.v1.release_strategy", "发布策略模块"),
        
        # ML训练和优化
        ("hyperparameter_optimization", "src.api.v1.hyperparameter_optimization", "超参数优化模块"),
        ("training_data", "src.api.v1.training_data", "训练数据模块"),
        ("fine_tuning", "src.api.v1.fine_tuning", "微调模块"),
        ("qlearning", "src.api.v1.qlearning_tensorflow_backup", "Q-Learning模块"),
        ("tensorflow_qlearning_ui", "src.api.v1.tensorflow_qlearning_ui", "TensorFlow Q-Learning UI模块（可选）"),
        
        # 分布式和集群
        ("service_discovery", "src.api.v1.service_discovery", "服务发现模块"),
        ("service_config", "src.api.v1.service_config", "服务配置模块"),
        ("service_routing", "src.api.v1.service_routing", "服务路由模块"),
        ("load_balancer", "src.api.v1.load_balancer", "负载均衡模块"),
        ("distributed_task", "src.api.v1.distributed_task", "分布式任务模块"),
        ("cluster_management", "src.api.v1.cluster_management", "集群管理模块"),
        ("fault_tolerance", "src.api.v1.fault_tolerance", "容错模块"),
        
        # 情感智能
        ("empathy_response", "src.api.v1.empathy_response", "共情响应模块"),
        ("emotional_memory", "src.api.v1.emotional_memory", "情感记忆模块"),
        ("emotion_intelligence", "src.api.v1.emotion_intelligence", "情感智能系统模块"),
        ("emotional_intelligence", "src.api.v1.emotional_intelligence", "情感智能决策引擎模块"),
        ("emotion_modeling", "src.api.v1.emotion_modeling", "情感状态建模系统模块"),
        ("emotion_recognition", "src.api.v1.emotion_recognition", "情感识别模块"),
        ("social_emotion_api", "src.api.v1.social_emotion_api", "社会情绪模块"),
        ("social_emotional_understanding", "src.api.v1.social_emotional_understanding", "社交情感理解模块"),
        ("emotion_websocket", "src.api.v1.emotion_websocket", "情感WebSocket模块"),
        
        # 其他功能
        ("feedback", "src.api.v1.feedback", "反馈模块"),
        ("bandit_recommendations", "src.api.v1.bandit_recommendations", "多臂老虎机推荐模块"),
        ("personalization", "src.api.v1.personalization", "个性化推荐模块"),
        ("offline", "src.api.v1.offline", "离线模块"),
        ("unified", "src.api.v1.unified", "统一模块"),
    ]
    
    # 加载API模块
    for module_name, import_path, description in api_module_groups:
        base_info = {
            "name": description,
            "import_path": import_path,
        }
        try:
            module = importlib.import_module(import_path)
            router = getattr(module, 'router', None)
            if router is None:
                raise AttributeError("router is None")
            v1_router.include_router(router)
            module_status[module_name] = {
                **base_info,
                "status": "active",
                "health": "healthy",
                "error": None,
            }
            logger.info("API模块加载成功", module=module_name, description=description)
        except (ImportError, AttributeError) as e:
            module_status[module_name] = {
                **base_info,
                "status": "inactive",
                "health": "unhealthy",
                "error": f"{e.__class__.__name__}: {str(e)}",
            }
            logger.warning("API模块导入失败", module=module_name, description=description, error=str(e))
            continue
        except Exception as e:
            module_status[module_name] = {
                **base_info,
                "status": "inactive",
                "health": "unhealthy",
                "error": f"{type(e).__name__}: {str(e)}",
            }
            logger.error("API模块加载未知错误", module=module_name, description=description, error=str(e))
            continue
    
    # 添加API模块状态端点
    @v1_router.get("/modules/status")
    async def get_modules_status():
        timestamp = utc_now().isoformat()
        modules = {
            module_key: {
                **info,
                "version": app.version,
                "last_check": timestamp,
            }
            for module_key, info in module_status.items()
        }
        total = len(modules)
        active = sum(1 for item in modules.values() if item["status"] == "active")
        failed = total - active
        success_rate = f"{active}/{total} ({(active / total * 100) if total else 0:.1f}%)"
        return {
            "success": True,
            "data": {
                "modules": modules,
                "summary": {
                    "total_attempted": total,
                    "loaded": active,
                    "failed": failed,
                    "success_rate": success_rate,
                },
                "timestamp": timestamp,
            }
        }
    
    # 注册API路由
    app.include_router(v1_router)
    
    # 可选：尝试加载TensorFlow路由（如果可用）
    # 注释掉TensorFlow路由以避免死锁问题
    # try:
    #     from api.v1.tensorflow import tensorflow_router
    #     app.include_router(tensorflow_router, prefix="/api/v1")
    # except ImportError as e:

    return app

# 创建应用实例
app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    logger.info("启动AI Agent System API服务器")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
    )
