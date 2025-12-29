# 后端 API 模块概览

根据 `apps/api/src/api/v1` 目录梳理的全部 FastAPI 模块，便于后续映射前端页面与测试覆盖。

| 模块 | 路由前缀 | 业务概述 | 示例端点 |
| --- | --- | --- | --- |

| `__init__` | `/` | API v1路由模块 | （查看源码获取端点详情） |
| `acl` | `/acl` | ACL (Access Control List) 协议管理 API | GET /acl/rules; GET /acl/rules/{rule_id}; POST /acl/rules; PUT /acl/rules/{rule_id} |
| `agent_interface` | `/agent` | Story 1.5: 标准化智能体API接口实现 提供清晰的API接口来与智能体交互，符合架构文档规范 | POST /agent/chat; POST /agent/task; GET /agent/status; GET /agent/metrics |
| `agents` | `/agents` | 智能体API端点 提供ReAct智能体的REST API接口 | POST /agents/sessions; POST /agents/react/chat/{conversation_id}; POST /agents/react/task/{conversation_id}; GET /agents/conversations/{conversation_id}/history |
| `alert_rules` | `/alert-rules` | 告警规则API端点 | POST /alert-rules/rules; PUT /alert-rules/rules/{rule_id}; DELETE /alert-rules/rules/{rule_id}; GET /alert-rules/rules |
| `analytics` | `/analytics` | 行为分析API端点（智能版本）  集成了真实的机器学习算法进行用户行为分析 - Isolation Forest异常检测 - 用户行为特征工程 - 统计异常检测 - 多算法融合决策 | POST /analytics/events; GET /analytics/events; GET /analytics/sessions; POST /analytics/analyze |
| `anomaly_detection` | `/anomalies` | 异常检测API端点 | POST /anomalies/detect; POST /anomalies/check-srm; POST /anomalies/check-data-quality; GET /anomalies/summary/{experiment_id} |
| `assignment_cache` | `/cache` | 用户分配缓存管理API端点 | ON_EVENT /cachestartup; ON_EVENT /cacheshutdown; GET /cache/assignments/{user_id}/{experiment_id}; POST /cache/assignments |
| `async_agents` | `/async-agents` | 异步多智能体系统API路由 集成AutoGen v0.7.x异步事件驱动架构 | POST /async-agents/agents; GET /async-agents/agents; GET /async-agents/agents/{agent_id}; DELETE /async-agents/agents/{agent_id} |
| `auth` | `/auth` | 身份认证API端点 | POST /auth/token; POST /auth/refresh; POST /auth/logout; GET /auth/me |
| `auto_scaling` | `/auto-scaling` | 自动扩量API端点 | POST /auto-scaling/rules; POST /auto-scaling/rules/{rule_id}/conditions; POST /auto-scaling/start; POST /auto-scaling/stop/{rule_id} |
| `bandit_recommendations` | `/` | 多臂老虎机推荐引擎API路由  提供推荐请求、反馈处理、算法管理和性能监控的REST API接口。 | POST /initialize; POST /recommend; POST /feedback; GET /statistics |
| `batch` | `/batch` | 批处理操作API端点 用于处理批量数据操作和任务调度 | POST /batch/jobs/create; GET /batch/jobs/{job_id}; GET /batch/jobs; POST /batch/jobs/{job_id}/start |
| `cache` | `/cache` | 缓存管理API端点 提供缓存状态查询、清理和监控功能 | GET /cache/stats; GET /cache/health; GET /cache/performance; DELETE /cache/clear |
| `cluster_management` | `/cluster` | 智能体集群管理API  提供集群管理的RESTful API接口，包括智能体生命周期管理、 监控数据查询、扩缩容控制等功能。 | GET /cluster/status; GET /cluster/topology; GET /cluster/health; POST /cluster/agents |
| `core_init` | `/` | API v1核心路由模块 - 仅包含基础功能 | （查看源码获取端点详情） |
| `distributed_security` | `/api/v1/distributed-security` | 分布式安全框架 API接口 提供身份认证、访问控制、安全审计等服务的REST API | POST /api/v1/distributed-security/authenticate; POST /api/v1/distributed-security/authorize; POST /api/v1/distributed-security/events; GET /api/v1/distributed-security/dashboard |
| `distributed_task` | `/distributed-task` | 分布式任务协调API端点 | POST /distributed-task/initialize; POST /distributed-task/submit; GET /distributed-task/status/{task_id}; POST /distributed-task/cancel/{task_id} |
| `documents` | `/documents` | 文档处理API接口 | POST /documents/upload; POST /documents/batch-upload; GET /documents/supported-formats; POST /documents/{doc_id}/analyze-relationships |
| `emotion_intelligence` | `/emotion-intelligence` | 情感智能系统主API 整合所有情感智能模块 | ON_EVENT /emotion-intelligencestartup; ON_EVENT /emotion-intelligenceshutdown; POST /emotion-intelligence/analyze; GET /emotion-intelligence/status |
| `emotion_modeling` | `/emotion` | 情感状态建模系统API接口 | POST /emotion/state; GET /emotion/state/latest; GET /emotion/state/history; POST /emotion/predict |
| `emotion_recognition` | `/api/v1/emotion` | 情感识别API端点 Story 11.1: 多模态情感识别引擎 | POST /api/v1/emotion/analyze/text; POST /api/v1/emotion/analyze/audio; POST /api/v1/emotion/analyze/visual; POST /api/v1/emotion/analyze/multimodal |
| `emotion_websocket` | `/ws` | 情感智能系统WebSocket实时通信API 支持多模态情感数据的实时传输和处理 | ON_EVENT /wsstartup; ON_EVENT /wsshutdown; WEBSOCKET /ws/emotion/{user_id}; GET /ws/stats |
| `emotional_intelligence` | `/emotional-intelligence` | 情感智能决策引擎 API 端点 | POST /emotional-intelligence/decide; POST /emotional-intelligence/risk-assessment; POST /emotional-intelligence/crisis-detection; POST /emotional-intelligence/intervention-plan |
| `emotional_memory` | `/emotional-memory` | 情感记忆管理系统 API Story 11.4 - 长期情感记忆存储、检索和模式分析 | POST /emotional-memory/memories; GET /emotional-memory/memories; POST /emotional-memory/memories/search; POST /emotional-memory/events/detect |
| `empathy_response` | `/empathy` | 共情响应生成API端点 | POST /empathy/generate; POST /empathy/batch-generate; GET /empathy/strategies; GET /empathy/analytics |
| `enterprise` | `/enterprise` | 企业级架构管理API 提供企业级架构的监控、配置和管理接口 | GET /enterprise/overview; GET /enterprise/health; GET /enterprise/security; GET /enterprise/performance |
| `event_batch` | `/event-batch` | 事件批处理API端点 - 管理事件批处理任务和缓冲 | POST /event-batch/events/submit; POST /event-batch/batches/submit; GET /event-batch/jobs/{job_id}/status; GET /event-batch/jobs |
| `event_tracking` | `/event-tracking` | 事件追踪API端点 - 提供事件收集、查询和管理功能 | POST /event-tracking/events; POST /event-tracking/events/batch; GET /event-tracking/events; GET /event-tracking/events/{event_id} |
| `events` | `/events` | 事件处理系统API端点 提供事件查询、监控和管理接口 | GET /events/list; GET /events/stats; POST /events/replay; GET /events/cluster/status |
| `experiments` | `/experiments` | A/B测试实验平台API端点 | POST /experiments/; GET /experiments/; GET /experiments/{experiment_id}; PUT /experiments/{experiment_id} |
| `explainable_ai` | `/explainable-ai` | 可解释AI API端点 | POST /explainable-ai/generate-explanation; POST /explainable-ai/cot-reasoning; POST /explainable-ai/workflow-explanation; POST /explainable-ai/format-explanation |
| `fault_tolerance` | `/fault-tolerance` | （缺少模块说明） | GET /fault-tolerance/status; GET /fault-tolerance/health; GET /fault-tolerance/health/{component_id}; GET /fault-tolerance/metrics |
| `feedback` | `/feedback` | 用户反馈系统API端点实现  提供隐式和显式反馈收集、处理、分析的REST API接口 | POST /feedback/implicit; POST /feedback/explicit; GET /feedback/user/{user_id}; GET /feedback/analytics/user/{user_id} |
| `files` | `/files` | 文件管理API路由 | POST /files/upload; POST /files/upload/batch; GET /files/list; GET /files/{file_id} |
| `fine_tuning` | `/api/v1/fine-tuning` | 微调API接口 提供LoRA/QLoRA微调的RESTful API | POST /api/v1/fine-tuning/jobs; GET /api/v1/fine-tuning/jobs; GET /api/v1/fine-tuning/jobs/{job_id}; PUT /api/v1/fine-tuning/jobs/{job_id}/cancel |
| `graphrag` | `/graphrag` | GraphRAG API端点  提供GraphRAG系统的HTTP API接口： - GraphRAG增强查询 - 查询分析和分解 - 推理路径查询 - 知识融合和冲突解决 - 性能监控和调试 | POST /graphrag/query; POST /graphrag/query/analyze; POST /graphrag/query/reasoning; GET /graphrag/query/{query_id} |
| `health` | `/health` | 健康检查API路由 | GET /health; GET /health/live; GET /health/ready; GET /health/metrics |
| `hyperparameter_optimization` | `/hyperparameter-optimization` | 超参数优化API接口  提供RESTful API用于管理超参数优化实验，包括创建、启动、监控、停止实验等功能。 | POST /hyperparameter-optimization/experiments; GET /hyperparameter-optimization/experiments; GET /hyperparameter-optimization/experiments/{experiment_id}; POST /hyperparameter-optimization/experiments/{experiment_id}/start |
| `hypothesis_testing` | `/hypothesis-testing` | 假设检验API端点 - 提供t检验、卡方检验等统计推断功能 | POST /hypothesis-testing/t-test/one-sample; POST /hypothesis-testing/t-test/two-sample; POST /hypothesis-testing/t-test/paired; POST /hypothesis-testing/chi-square/goodness-of-fit |
| `knowledge_extraction` | `/knowledge` | 知识图谱抽取API端点  提供知识图谱实体识别、关系抽取、批量处理等RESTful API接口 | GET /knowledge/health; GET /knowledge/metrics; POST /knowledge/extract; POST /knowledge/batch/submit |
| `knowledge_graph` | `/knowledge-graph` | 知识图谱存储系统API 提供图谱CRUD、查询、质量管理、性能监控等功能的REST API接口 | POST /knowledge-graph/entities; GET /knowledge-graph/entities/{entity_id}; PUT /knowledge-graph/entities/{entity_id}; DELETE /knowledge-graph/entities/{entity_id} |
| `knowledge_graph_reasoning` | `/kg-reasoning` | 知识图推理引擎API端点  提供统一的推理服务接口，支持多种推理策略和方法 | POST /kg-reasoning/query; POST /kg-reasoning/batch; GET /kg-reasoning/strategies/performance; POST /kg-reasoning/config |
| `knowledge_management` | `/api/v1/kg` | 知识图谱管理API  提供完整的知识图谱管理RESTful API接口： - 实体和关系CRUD操作 - 批量操作和事务支持 - 图谱结构验证和一致性检查 - API文档和OpenAPI规范集成 | GET /api/v1/kg/entities; POST /api/v1/kg/entities; GET /api/v1/kg/entities/{entity_uri:path}; PUT /api/v1/kg/entities/{entity_uri:path} |
| `langgraph_features` | `/langgraph` | LangGraph 0.6.5新特性演示API端点 提供Context API、durability控制、Node Caching和Pre/Post Hooks演示 | POST /langgraph/context-api/demo; POST /langgraph/durability/demo; POST /langgraph/caching/demo; POST /langgraph/hooks/demo |
| `layered_experiments` | `/experiments/layers` | 分层实验管理API端点 | POST /experiments/layers/; GET /experiments/layers/; GET /experiments/layers/{layer_id}; POST /experiments/layers/{layer_id}/groups |
| `mcp` | `/mcp` | MCP API路由 | POST /mcp/tools/call; GET /mcp/tools; GET /mcp/health; GET /mcp/metrics |
| `memory_analytics` | `/memories/analytics` | 记忆分析API端点 | GET /memories/analytics/; GET /memories/analytics/patterns; GET /memories/analytics/graph/stats; GET /memories/analytics/trends |
| `memory_management` | `/memories` | 记忆管理API端点 | POST /memories/; PUT /memories/{memory_id}; DELETE /memories/{memory_id}; GET /memories/search |
| `model_compression` | `/model-compression` | 模型压缩API接口  实现压缩任务提交和管理接口 添加压缩进度监控和结果查询 实现压缩模型下载和部署 提供压缩配置和模板管理 | POST /model-compression/jobs; GET /model-compression/jobs; GET /model-compression/jobs/{job_id}; PUT /model-compression/jobs/{job_id}/cancel |
| `model_evaluation` | `/model-evaluation` | （缺少模块说明） | GET /model-evaluation/; GET /model-evaluation/benchmarks; GET /model-evaluation/benchmark-suites; POST /model-evaluation/evaluate |
| `model_registry` | `/model-registry` | 模型注册表API端点  提供RESTful API接口用于管理AI模型注册表 | GET /model-registry/models; GET /model-registry/models/{name}; POST /model-registry/models/upload; POST /model-registry/models/{name}/register-from-hub |
| `model_service` | `/api/v1/model-service` | 模型服务平台API接口 | POST /api/v1/model-service/models/upload; POST /api/v1/model-service/models/register-from-hub; GET /api/v1/model-service/models; GET /api/v1/model-service/models/{model_name}/versions/{version} |
| `multi_agents` | `/multi-agent` | 多智能体协作API路由 | POST /multi-agent/conversation; GET /multi-agent/conversation/{conversation_id}/status; POST /multi-agent/conversation/{conversation_id}/pause; POST /multi-agent/conversation/{conversation_id}/resume |
| `multi_step_reasoning` | `/multi-step-reasoning` | 多步推理工作流API接口 对应技术架构的API端点 | POST /multi-step-reasoning/decompose; POST /multi-step-reasoning/execute; GET /multi-step-reasoning/executions/{execution_id}; POST /multi-step-reasoning/executions/control |
| `multimodal` | `/multimodal` | 多模态处理API路由 | POST /multimodal/upload; POST /multimodal/process; POST /multimodal/process/batch; GET /multimodal/status/{content_id} |
| `multimodal_rag` | `/multimodal-rag` | 多模态RAG API端点 | POST /multimodal-rag/query; POST /multimodal-rag/query-with-files; POST /multimodal-rag/stream-query; POST /multimodal-rag/upload-document |
| `multiple_testing_correction` | `/multiple-testing` | 多重检验校正API端点 | POST /multiple-testing/apply-correction; POST /multiple-testing/compare-methods; POST /multiple-testing/recommend-method; POST /multiple-testing/adjust-power |
| `offline` | `/offline` | 离线能力API端点  提供离线状态管理、同步控制等功能 | GET /offline/status; POST /offline/sync; GET /offline/conflicts; POST /offline/resolve |
| `personalization` | `/personalization` | （缺少模块说明） | POST /personalization/recommend; GET /personalization/user/{user_id}/profile; PUT /personalization/user/{user_id}/profile; POST /personalization/feedback |
| `pgvector` | `/pgvector` | pgvector 0.8.0 API端点 提供向量数据库管理、性能监控、量化配置等功能 | POST /pgvector/collections; POST /pgvector/vectors; GET /pgvector/collections/{collection_name}/stats; POST /pgvector/search/similarity |
| `platform_integration` | `/platform` | 平台集成API端点 | POST /platform/components/register; DELETE /platform/components/{component_id}; GET /platform/components; GET /platform/components/{component_id} |
| `power_analysis` | `/power-analysis` | 统计功效和样本量计算API端点 | POST /power-analysis/calculate-power; POST /power-analysis/calculate-sample-size; POST /power-analysis/calculate-effect-size; POST /power-analysis/proportion-power |
| `qlearning_tensorflow_backup` | `/qlearning` | Q-Learning API端点实现  提供Q-Learning智能体的REST API接口 | GET /qlearning/info; POST /qlearning/agents; GET /qlearning/agents; GET /qlearning/agents/{session_id} |
| `rag` | `/rag` | RAG系统 API 路由  包含基础RAG功能和Agentic RAG智能检索功能 | POST /rag/documents; POST /rag/search; POST /rag/query; POST /rag/index/file |
| `realtime_metrics` | `/realtime-metrics` | 实时指标API端点 | POST /realtime-metrics/register-metric; POST /realtime-metrics/calculate; POST /realtime-metrics/compare-groups; POST /realtime-metrics/trends |
| `reasoning` | `/reasoning` | 链式思考(CoT)推理API端点 | POST /reasoning/chain; POST /reasoning/stream; GET /reasoning/chain/{chain_id}; GET /reasoning/history |
| `release_strategy` | `/release-strategy` | 发布策略API端点 | POST /release-strategy/strategies; POST /release-strategy/strategies/from-template; GET /release-strategy/strategies; GET /release-strategy/strategies/{strategy_id} |
| `report_generation` | `/reports` | 实验报告生成API端点 | POST /reports/generate; POST /reports/generate-summary; POST /reports/schedule; GET /reports/export/{experiment_id} |
| `risk_assessment` | `/risk-assessment` | 风险评估和回滚API端点 | POST /risk-assessment/assess; GET /risk-assessment/history/{experiment_id}; POST /risk-assessment/rollback-plan; POST /risk-assessment/rollback/execute |
| `security` | `/security` | 安全管理API端点 | GET /security/config; PUT /security/config; GET /security/api-keys; POST /security/api-keys |
| `service_discovery` | `/service-discovery` | Service Discovery API Endpoints  FastAPI endpoints for agent service discovery and registration. | POST /service-discovery/agents; GET /service-discovery/agents; GET /service-discovery/agents/{agent_id}; PUT /service-discovery/agents/{agent_id}/status |
| `social_emotion_api` | `/api/v1/social-emotion` | 社交情感理解系统API - Story 11.6 Task 8 提供完整的社交情感理解系统REST API和WebSocket接口 | POST /api/v1/social-emotion/initialize; POST /api/v1/social-emotion/analyze; POST /api/v1/social-emotion/analyze/batch; POST /api/v1/social-emotion/session/create |
| `social_emotional_understanding` | `/social-emotional-understanding` | 社交情感理解系统API端点 - Story 11.6 提供完整的社交情感理解功能API接口 | POST /social-emotional-understanding/analyze/group-emotion; POST /social-emotional-understanding/analyze/relationships; POST /social-emotional-understanding/analyze/social-context; POST /social-emotional-understanding/analyze/cultural-adaptation |
| `sparql_api` | `/api/v1/kg/sparql` | SPARQL查询API接口  提供标准的SPARQL查询和更新API： - SPARQL SELECT/CONSTRUCT/ASK/DESCRIBE查询 - SPARQL UPDATE操作 - 查询计划分析和优化建议 - 多种结果格式支持 | POST /api/v1/kg/sparql/query; GET /api/v1/kg/sparql/query; POST /api/v1/kg/sparql/update; POST /api/v1/kg/sparql/update/form |
| `statistical_analysis` | `/statistical-analysis` | 统计分析API端点 - 提供基础统计计算功能 | POST /statistical-analysis/basic-stats; POST /statistical-analysis/conversion-stats; POST /statistical-analysis/percentiles; POST /statistical-analysis/multiple-groups-stats |
| `streaming` | `/streaming` | 流式处理API路由  提供SSE和WebSocket流式响应接口。 | POST /streaming/start; GET /streaming/sse/{session_id}; WEBSOCKET /streaming/ws/{session_id}; GET /streaming/sessions/{session_id}/metrics |
| `supervisor` | `/supervisor` | Supervisor管理API路由 提供Supervisor智能体的HTTP接口 | POST /supervisor/tasks; GET /supervisor/status; GET /supervisor/decisions; PUT /supervisor/config |
| `targeting_rules` | `/targeting` | 定向规则管理API端点 | POST /targeting/rules; GET /targeting/rules; GET /targeting/rules/{rule_id}; PUT /targeting/rules/{rule_id} |
| `test` | `/test` | 测试端点 - 用于验证异步处理能力 | GET /test/async-db; GET /test/async-redis; GET /test/concurrent; GET /test/mixed-async |
| `testing` | `/api/v1/testing` | 测试和验证API端点 | GET /api/v1/testing/integration/suites; POST /api/v1/testing/integration/run; GET /api/v1/testing/integration/results/{test_id}; GET /api/v1/testing/integration/reports |
| `traffic_ramp` | `/traffic-ramp` | 流量渐进调整API端点 | POST /traffic-ramp/plans; POST /traffic-ramp/start; POST /traffic-ramp/pause; POST /traffic-ramp/resume |
| `training_data` | `/training-data` | 训练数据管理API接口 | POST /training-data/sources; GET /training-data/sources; PUT /training-data/sources/{source_id}; DELETE /training-data/sources/{source_id} |
| `unified` | `/unified` | 统一处理引擎API  提供流批一体化处理接口，支持智能模式切换和混合处理。 | POST /unified/process; GET /unified/status/{session_id}; GET /unified/history; GET /unified/metrics |
| `workflows` | `/workflows` | 工作流管理API路由 | GET /workflows/health/check; POST /workflows/; GET /workflows/; GET /workflows/{workflow_id} |
