# API端点与前端页面对应关系分析报告

## 总览统计
- 总API端点数: 702
- 已使用端点数: 156
- 未使用端点数: 546
- 使用率: 22.2%

## 详细分析表格

| API端点 | HTTP方法 | 路径 | 所属模块 | 是否使用 | 使用次数 | 被使用的文件 |
|---------|----------|------|----------|----------|----------|--------------|
| DELETE /cache | DELETE | /cache | qlearning_tensorflow_backup | ✅ | 1 | services/knowledgeGraphService.ts |
| DELETE /clear | DELETE | /clear | cache | ✅ | 1 | services/knowledgeGraphService.ts |
| DELETE /data | DELETE | /data | emotion_modeling | ✅ | 2 | services/trainingDataService.ts, pages/SecurityAuditPage.tsx |
| DELETE /experiments/{experiment_id} | DELETE | /experiments/{experiment_id} | hyperparameter_optimization | ✅ | 1 | services/reportService.ts |
| DELETE /{experiment_id} | DELETE | /{experiment_id} | experiments | ✅ | 1 | services/reportService.ts |
| GET / | GET | / | workflows | ✅ | 44 | pages/DocumentProcessingAdvancedPage.tsx, services/filesService.ts, services/entityService.ts (+41个) |
| GET /abtest/list | GET | /abtest/list | model_service | ✅ | 1 | services/modelService.ts |
| GET /agents | GET | /agents | service_discovery | ✅ | 3 | pages/SecurityAuditPage.tsx, pages/SupervisorApiManagementPage.tsx, pages/ArchitectureDebugPage.tsx |
| GET /alerts | GET | /alerts | distributed_security | ✅ | 2 | services/modelService.ts, services/unifiedService.ts |
| GET /annotation-tasks | GET | /annotation-tasks | training_data | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| GET /annotation-tasks/{task_id} | GET | /annotation-tasks/{task_id} | training_data | ✅ | 1 | services/reportService.ts |
| GET /annotation-tasks/{task_id}/quality-report | GET | /annotation-tasks/{task_id}/quality-report | training_data | ✅ | 1 | services/reportService.ts |
| GET /benchmarks | GET | /benchmarks | model_evaluation | ✅ | 1 | pages/EvaluationApiManagementPage.tsx |
| GET /cluster/status | GET | /cluster/status | events | ✅ | 1 | services/eventService.ts |
| GET /compliance | GET | /compliance | enterprise | ✅ | 1 | services/reportService.ts |
| GET /compliance-report | GET | /compliance-report | security | ✅ | 1 | services/reportService.ts |
| GET /compliance/report | GET | /compliance/report | social_emotion_api | ✅ | 1 | services/reportService.ts |
| GET /components | GET | /components | platform_integration | ✅ | 1 | pages/PlatformApiManagementPage.tsx |
| GET /config | GET | /config | service_discovery | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| GET /dashboard | GET | /dashboard | distributed_security | ✅ | 1 | services/modelService.ts |
| GET /datasets | GET | /datasets | fine_tuning | ✅ | 1 | services/trainingDataService.ts |
| GET /deployment/list | GET | /deployment/list | model_service | ✅ | 1 | services/modelService.ts |
| GET /events | GET | /events | analytics | ✅ | 2 | pages/DistributedEventsPage.tsx, services/eventService.ts |
| GET /experiments | GET | /experiments | bandit_recommendations | ✅ | 2 | services/reportService.ts, components/hyperparameter/ExperimentList.tsx |
| GET /experiments/{experiment_id} | GET | /experiments/{experiment_id} | hyperparameter_optimization | ✅ | 1 | services/reportService.ts |
| GET /export | GET | /export | acl | ✅ | 4 | pages/behavior-analytics/DataExportPage.tsx, services/trainingDataService.ts, services/tensorflowService.ts (+1个) |
| GET /export/{experiment_id} | GET | /export/{experiment_id} | report_generation | ✅ | 1 | services/reportService.ts |
| GET /fusion/consistency | GET | /fusion/consistency | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| GET /health | GET | /health | distributed_security | ✅ | 5 | pages/ServiceHealthMonitorPage.tsx, services/reportService.ts, services/streamingService.ts (+2个) |
| GET /history | GET | /history | reasoning | ✅ | 1 | services/tensorflowService.ts |
| GET /inference/models/loaded | GET | /inference/models/loaded | model_service | ✅ | 1 | services/modelService.ts |
| GET /learning/sessions | GET | /learning/sessions | model_service | ✅ | 1 | services/modelService.ts |
| GET /list | GET | /list | events | ✅ | 2 | services/modelService.ts, services/eventService.ts |
| GET /me | GET | /me | auth | ✅ | 6 | services/authService.ts, pages/AuthManagementPage.tsx, services/streamingService.ts (+3个) |
| GET /metrics | GET | /metrics | distributed_security | ✅ | 4 | services/streamingService.ts, services/modelService.ts, pages/SupervisorApiManagementPage.tsx (+1个) |
| GET /models | GET | /models | model_registry | ✅ | 4 | pages/SecurityAuditPage.tsx, services/modelService.ts, services/tensorflowService.ts (+1个) |
| GET /monitoring/alerts | GET | /monitoring/alerts | model_service | ✅ | 2 | services/modelService.ts, services/unifiedService.ts |
| GET /monitoring/dashboard | GET | /monitoring/dashboard | model_service | ✅ | 1 | services/modelService.ts |
| GET /monitoring/metrics | GET | /monitoring/metrics | events | ✅ | 2 | services/modelService.ts, services/eventService.ts |
| GET /monitoring/overview | GET | /monitoring/overview | model_service | ✅ | 1 | services/modelService.ts |
| GET /monitoring/performance-report | GET | /monitoring/performance-report | pgvector | ✅ | 1 | services/reportService.ts |
| GET /monitoring/recommendations | GET | /monitoring/recommendations | model_service | ✅ | 1 | services/modelService.ts |
| GET /monitoring/report | GET | /monitoring/report | platform_integration | ✅ | 1 | services/reportService.ts |
| GET /optimization/report | GET | /optimization/report | platform_integration | ✅ | 1 | services/reportService.ts |
| GET /overview | GET | /overview | enterprise | ✅ | 1 | services/modelService.ts |
| GET /performance | GET | /performance | enterprise | ✅ | 2 | services/reportService.ts, services/knowledgeGraphService.ts |
| GET /performance/comparison | GET | /performance/comparison | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| GET /permissions | GET | /permissions | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| GET /policies | GET | /policies | distributed_security | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| GET /preview/{experiment_id} | GET | /preview/{experiment_id} | report_generation | ✅ | 1 | services/reportService.ts |
| GET /quality-report | GET | /quality-report | emotion_intelligence | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| GET /quality/report | GET | /quality/report | knowledge_graph | ✅ | 1 | services/reportService.ts |
| GET /rag/health | GET | /rag/health | mock_endpoints | ✅ | 1 | pages/HealthComprehensivePage.tsx |
| GET /rag/stats | GET | /rag/stats | mock_endpoints | ✅ | 1 | pages/BasicRagManagementPage.tsx |
| GET /report | GET | /report | fault_tolerance | ✅ | 1 | services/reportService.ts |
| GET /results/{job_id} | GET | /results/{job_id} | model_compression | ✅ | 1 | services/reportService.ts |
| GET /results/{job_id}/report | GET | /results/{job_id}/report | model_compression | ✅ | 1 | services/reportService.ts |
| GET /search | GET | /search | memory_management | ✅ | 1 | pages/BasicRagManagementPage.tsx |
| GET /security | GET | /security | enterprise | ✅ | 1 | services/reportService.ts |
| GET /sessions | GET | /sessions | streaming | ✅ | 2 | services/streamingService.ts, services/modelService.ts |
| GET /sources | GET | /sources | training_data | ✅ | 1 | services/trainingDataService.ts |
| GET /statistics | GET | /statistics | bandit_recommendations | ✅ | 2 | services/modelService.ts, services/trainingDataService.ts |
| GET /stats | GET | /stats | service_discovery | ✅ | 4 | services/modelService.ts, pages/BasicRagManagementPage.tsx, pages/SupervisorApiManagementPage.tsx (+1个) |
| GET /status | GET | /status | model_compression | ✅ | 6 | services/trainingDataService.ts, services/reportService.ts, services/unifiedService.ts (+3个) |
| GET /strategies | GET | /strategies | model_compression | ✅ | 1 | services/knowledgeGraphService.ts |
| GET /strategies/performance | GET | /strategies/performance | knowledge_graph_reasoning | ✅ | 1 | services/knowledgeGraphService.ts |
| GET /summary | GET | /summary | statistical_analysis | ✅ | 1 | services/unifiedService.ts |
| GET /tasks | GET | /tasks | hyperparameter_optimization | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| GET /templates | GET | /templates | alert_rules | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| GET /validate | GET | /validate | model_registry | ✅ | 2 | services/modelService.ts, services/reportService.ts |
| GET /verify | GET | /verify | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| GET /workflows | GET | /workflows | multi_step_reasoning | ✅ | 5 | services/workflowService.ts, pages/ArchitectureDebugPage.tsx, pages/HealthComprehensivePage.tsx (+2个) |
| GET /{experiment_id} | GET | /{experiment_id} | experiments | ✅ | 1 | services/reportService.ts |
| GET /{experiment_id}/report | GET | /{experiment_id}/report | experiments | ✅ | 1 | services/reportService.ts |
| POST / | POST | / | workflows | ✅ | 44 | pages/DocumentProcessingAdvancedPage.tsx, services/filesService.ts, services/entityService.ts (+41个) |
| POST /abtest/create | POST | /abtest/create | model_service | ✅ | 1 | services/modelService.ts |
| POST /agents | POST | /agents | service_discovery | ✅ | 3 | pages/SecurityAuditPage.tsx, pages/SupervisorApiManagementPage.tsx, pages/ArchitectureDebugPage.tsx |
| POST /annotation-tasks | POST | /annotation-tasks | training_data | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| POST /batch | POST | /batch | knowledge_graph_reasoning | ✅ | 5 | services/reportService.ts, services/knowledgeGraphService.ts, services/modelService.ts (+2个) |
| POST /batch-generate | POST | /batch-generate | empathy_response | ✅ | 1 | services/reportService.ts |
| POST /benchmark | POST | /benchmark | model_compression | ✅ | 2 | pages/EvaluationApiManagementPage.tsx, services/knowledgeGraphService.ts |
| POST /cache/clear | POST | /cache/clear | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /caching/demo | POST | /caching/demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /change-password | POST | /change-password | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| POST /chat | POST | /chat | agent_interface | ✅ | 1 | pages/AgentInterfacePage.tsx |
| POST /cleanup | POST | /cleanup | tensorflow | ✅ | 1 | services/streamingService.ts |
| POST /clear | POST | /clear | assignment_cache | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /compare | POST | /compare | report_generation | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| POST /complete-demo | POST | /complete-demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /components/register | POST | /components/register | platform_integration | ✅ | 1 | pages/PlatformApiManagementPage.tsx |
| POST /config | POST | /config | knowledge_graph_reasoning | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| POST /context-api/demo | POST | /context-api/demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /datasets | POST | /datasets | fine_tuning | ✅ | 1 | services/trainingDataService.ts |
| POST /debug/explain | POST | /debug/explain | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /deployment/deploy | POST | /deployment/deploy | model_service | ✅ | 1 | services/modelService.ts |
| POST /documents | POST | /documents | rag | ✅ | 2 | pages/DocumentProcessingAdvancedPage.tsx, pages/BasicRagManagementPage.tsx |
| POST /durability/demo | POST | /durability/demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /entities | POST | /entities | knowledge_graph | ✅ | 2 | services/entityService.ts, pages/EntityApiPage.tsx |
| POST /evaluate | POST | /evaluate | alert_rules | ✅ | 1 | services/tensorflowService.ts |
| POST /events | POST | /events | distributed_security | ✅ | 2 | pages/DistributedEventsPage.tsx, services/eventService.ts |
| POST /events/submit | POST | /events/submit | event_batch | ✅ | 1 | services/eventService.ts |
| POST /execute | POST | /execute | multi_step_reasoning | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /execute-demo | POST | /execute-demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /experiments | POST | /experiments | bandit_recommendations | ✅ | 2 | services/reportService.ts, components/hyperparameter/ExperimentList.tsx |
| POST /explain | POST | /explain | knowledge_graph_reasoning | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /export | POST | /export | social_emotion_api | ✅ | 4 | pages/behavior-analytics/DataExportPage.tsx, services/trainingDataService.ts, services/tensorflowService.ts (+1个) |
| POST /feedback | POST | /feedback | qlearning_tensorflow_backup | ✅ | 2 | services/modelService.ts, services/feedbackTracker.ts |
| POST /fusion/conflict-resolution | POST | /fusion/conflict-resolution | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /fusion/multi-source | POST | /fusion/multi-source | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /generate | POST | /generate | empathy_response | ✅ | 1 | services/reportService.ts |
| POST /generate-report | POST | /generate-report | model_evaluation | ✅ | 1 | services/reportService.ts |
| POST /generate-summary | POST | /generate-summary | report_generation | ✅ | 1 | services/reportService.ts |
| POST /graphrag/query | POST | /graphrag/query | rag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /hooks/demo | POST | /hooks/demo | langgraph_features | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /implicit | POST | /implicit | feedback | ✅ | 1 | services/feedbackTracker.ts |
| POST /import | POST | /import | acl | ✅ | 1 | services/trainingDataService.ts |
| POST /inference/batch-predict | POST | /inference/batch-predict | model_service | ✅ | 1 | services/modelService.ts |
| POST /inference/predict | POST | /inference/predict | model_service | ✅ | 1 | services/modelService.ts |
| POST /learning/start | POST | /learning/start | model_service | ✅ | 1 | services/modelService.ts |
| POST /logout | POST | /logout | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| POST /models | POST | /models | tensorflow | ✅ | 4 | pages/SecurityAuditPage.tsx, services/modelService.ts, services/tensorflowService.ts (+1个) |
| POST /models/load | POST | /models/load | tensorflow | ✅ | 1 | services/modelService.ts |
| POST /models/register-from-hub | POST | /models/register-from-hub | model_service | ✅ | 1 | services/modelService.ts |
| POST /models/upload | POST | /models/upload | model_registry | ✅ | 1 | services/modelService.ts |
| POST /monitor | POST | /monitor | risk_assessment | ✅ | 4 | services/modelService.ts, services/reportService.ts, services/eventService.ts (+1个) |
| POST /performance/benchmark | POST | /performance/benchmark | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /policies | POST | /policies | distributed_security | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| POST /predict | POST | /predict | emotion_modeling | ✅ | 2 | services/modelService.ts, services/tensorflowService.ts |
| POST /query | POST | /query | rag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /query/reasoning | POST | /query/reasoning | graphrag | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /recommend | POST | /recommend | personalization | ✅ | 2 | services/modelService.ts, pages/PersonalizationEnginePage.tsx |
| POST /refresh | POST | /refresh | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| POST /replay | POST | /replay | events | ✅ | 1 | services/eventService.ts |
| POST /reports/generate | POST | /reports/generate | analytics | ✅ | 1 | services/reportService.ts |
| POST /schedule | POST | /schedule | report_generation | ✅ | 1 | services/reportService.ts |
| POST /search | POST | /search | rag | ✅ | 1 | pages/BasicRagManagementPage.tsx |
| POST /sessions | POST | /sessions | agents | ✅ | 2 | services/streamingService.ts, services/modelService.ts |
| POST /sources | POST | /sources | training_data | ✅ | 1 | services/trainingDataService.ts |
| POST /start | POST | /start | streaming | ✅ | 4 | services/streamingService.ts, services/modelService.ts, services/trainingDataService.ts (+1个) |
| POST /strategies | POST | /strategies | release_strategy | ✅ | 1 | services/knowledgeGraphService.ts |
| POST /stream | POST | /stream | reasoning | ✅ | 2 | services/eventService.ts, pages/PersonalizationWebSocketPage.tsx |
| POST /submit | POST | /submit | distributed_task | ✅ | 1 | services/eventService.ts |
| POST /sync | POST | /sync | offline | ✅ | 1 | services/trainingDataService.ts |
| POST /task | POST | /task | agent_interface | ✅ | 2 | pages/SupervisorApiManagementPage.tsx, pages/AgentInterfacePage.tsx |
| POST /tasks | POST | /tasks | hyperparameter_optimization | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| POST /templates | POST | /templates | alert_rules | ✅ | 2 | services/trainingDataService.ts, services/reportService.ts |
| POST /test | POST | /test | alert_rules | ✅ | 3 | services/trainingDataService.ts, pages/TestCoveragePage.tsx, services/reportService.ts |
| POST /token | POST | /token | auth | ✅ | 2 | services/authService.ts, pages/AuthManagementPage.tsx |
| POST /upload | POST | /upload | multimodal | ✅ | 5 | pages/DocumentProcessingAdvancedPage.tsx, services/modelService.ts, pages/KGReasoningBatchPage.tsx (+2个) |
| POST /upload/batch | POST | /upload/batch | files | ✅ | 1 | pages/FileManagementAdvancedPage.tsx |
| POST /validate | POST | /validate | acl | ✅ | 2 | services/modelService.ts, services/reportService.ts |
| POST /versions | POST | /versions | training_data | ✅ | 2 | services/modelService.ts, services/trainingDataService.ts |
| POST /workflows/run | POST | /workflows/run | platform_integration | ✅ | 1 | pages/PlatformApiManagementPage.tsx |
| PUT /config | PUT | /config | security | ✅ | 1 | pages/SupervisorApiManagementPage.tsx |
| PUT /me | PUT | /me | auth | ✅ | 6 | services/authService.ts, pages/AuthManagementPage.tsx, services/streamingService.ts (+3个) |
| PUT /{experiment_id} | PUT | /{experiment_id} | experiments | ✅ | 1 | services/reportService.ts |
| DELETE /agents/{agent_id} | DELETE | /agents/{agent_id} | service_discovery | ❌ | 0 |  |
| DELETE /agents/{session_id} | DELETE | /agents/{session_id} | qlearning_tensorflow_backup | ❌ | 0 |  |
| DELETE /agents/{session_id}/cache | DELETE | /agents/{session_id}/cache | qlearning_tensorflow_backup | ❌ | 0 |  |
| DELETE /api-keys/{key_id} | DELETE | /api-keys/{key_id} | security | ❌ | 0 |  |
| DELETE /assignments/{user_id}/{experiment_id} | DELETE | /assignments/{user_id}/{experiment_id} | assignment_cache | ❌ | 0 |  |
| DELETE /batch/cleanup | DELETE | /batch/cleanup | qlearning_tensorflow_backup | ❌ | 0 |  |
| DELETE /batch/{batch_id} | DELETE | /batch/{batch_id} | knowledge_extraction | ❌ | 0 |  |
| DELETE /chain/{chain_id} | DELETE | /chain/{chain_id} | reasoning | ❌ | 0 |  |
| DELETE /components/{component_id} | DELETE | /components/{component_id} | platform_integration | ❌ | 0 |  |
| DELETE /context/{user_id} | DELETE | /context/{user_id} | empathy_response | ❌ | 0 |  |
| DELETE /conversations/{conversation_id} | DELETE | /conversations/{conversation_id} | agents | ❌ | 0 |  |
| DELETE /deployment/{deployment_id} | DELETE | /deployment/{deployment_id} | model_service | ❌ | 0 |  |
| DELETE /entities/{entity_id} | DELETE | /entities/{entity_id} | knowledge_graph | ❌ | 0 |  |
| DELETE /executions/{execution_id} | DELETE | /executions/{execution_id} | multi_step_reasoning | ❌ | 0 |  |
| DELETE /file/{content_id} | DELETE | /file/{content_id} | multimodal | ❌ | 0 |  |
| DELETE /index/reset | DELETE | /index/reset | rag | ❌ | 0 |  |
| DELETE /inference/models/{model_name}/unload | DELETE | /inference/models/{model_name}/unload | model_service | ❌ | 0 |  |
| DELETE /invalidate/{node_name} | DELETE | /invalidate/{node_name} | cache | ❌ | 0 |  |
| DELETE /jobs/{job_id} | DELETE | /jobs/{job_id} | model_compression | ❌ | 0 |  |
| DELETE /memories/{user_id}/{memory_id} | DELETE | /memories/{user_id}/{memory_id} | emotional_memory | ❌ | 0 |  |
| DELETE /models/{model_name} | DELETE | /models/{model_name} | tensorflow | ❌ | 0 |  |
| DELETE /models/{model_name}/versions/{version} | DELETE | /models/{model_name}/versions/{version} | model_service | ❌ | 0 |  |
| DELETE /models/{name} | DELETE | /models/{name} | model_registry | ❌ | 0 |  |
| DELETE /monitoring/clear/{session_id} | DELETE | /monitoring/clear/{session_id} | qlearning_tensorflow_backup | ❌ | 0 |  |
| DELETE /monitoring/clear_all | DELETE | /monitoring/clear_all | qlearning_tensorflow_backup | ❌ | 0 |  |
| DELETE /performance/cache | DELETE | /performance/cache | knowledge_graph | ❌ | 0 |  |
| DELETE /privacy/consent/{user_id} | DELETE | /privacy/consent/{user_id} | social_emotion_api | ❌ | 0 |  |
| DELETE /rules | DELETE | /rules | targeting_rules | ❌ | 0 |  |
| DELETE /rules/{rule_id} | DELETE | /rules/{rule_id} | alert_rules | ❌ | 0 |  |
| DELETE /session/{session_id} | DELETE | /session/{session_id} | social_emotion_api | ❌ | 0 |  |
| DELETE /sessions/{session_id} | DELETE | /sessions/{session_id} | streaming | ❌ | 0 |  |
| DELETE /sources/{source_id} | DELETE | /sources/{source_id} | training_data | ❌ | 0 |  |
| DELETE /user/{user_id}/assignments | DELETE | /user/{user_id}/assignments | layered_experiments | ❌ | 0 |  |
| DELETE /users/{user_id}/assignments | DELETE | /users/{user_id}/assignments | assignment_cache | ❌ | 0 |  |
| DELETE /versions/{version_id} | DELETE | /versions/{version_id} | training_data | ❌ | 0 |  |
| DELETE /{experiment_id}/allocation/cache | DELETE | /{experiment_id}/allocation/cache | experiments | ❌ | 0 |  |
| DELETE /{file_id} | DELETE | /{file_id} | files | ❌ | 0 |  |
| DELETE /{memory_id} | DELETE | /{memory_id} | memory_management | ❌ | 0 |  |
| DELETE /{workflow_id} | DELETE | /{workflow_id} | workflows | ❌ | 0 |  |
| GET /abtest/{test_id}/assign | GET | /abtest/{test_id}/assign | model_service | ❌ | 0 |  |
| GET /abtest/{test_id}/results | GET | /abtest/{test_id}/results | model_service | ❌ | 0 |  |
| GET /access-logs | GET | /access-logs | distributed_security | ❌ | 0 |  |
| GET /active-experiments | GET | /active-experiments | hyperparameter_optimization | ❌ | 0 |  |
| GET /admin/schema/status | GET | /admin/schema/status | knowledge_graph | ❌ | 0 |  |
| GET /agentic/explain | GET | /agentic/explain | rag | ❌ | 0 |  |
| GET /agentic/health | GET | /agentic/health | rag | ❌ | 0 |  |
| GET /agentic/stats | GET | /agentic/stats | rag | ❌ | 0 |  |
| GET /agents/{agent_id} | GET | /agents/{agent_id} | service_discovery | ❌ | 0 |  |
| GET /agents/{session_id} | GET | /agents/{session_id} | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/actions | GET | /agents/{session_id}/actions | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/insights | GET | /agents/{session_id}/insights | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/performance | GET | /agents/{session_id}/performance | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/performance_trend | GET | /agents/{session_id}/performance_trend | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/progress | GET | /agents/{session_id}/progress | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/statistics | GET | /agents/{session_id}/statistics | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /agents/{session_id}/summary | GET | /agents/{session_id}/summary | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /algorithms | GET | /algorithms | bandit_recommendations | ❌ | 0 |  |
| GET /allocation/stats | GET | /allocation/stats | experiments | ❌ | 0 |  |
| GET /analytics | GET | /analytics | empathy_response | ❌ | 0 |  |
| GET /analytics/graph/stats | GET | /analytics/graph/stats | memory_management | ❌ | 0 |  |
| GET /analytics/item/{item_id} | GET | /analytics/item/{item_id} | feedback | ❌ | 0 |  |
| GET /analytics/patterns | GET | /analytics/patterns | memory_management | ❌ | 0 |  |
| GET /analytics/trends | GET | /analytics/trends | memory_management | ❌ | 0 |  |
| GET /analytics/user/{user_id} | GET | /analytics/user/{user_id} | feedback | ❌ | 0 |  |
| GET /annotation-tasks/{task_id}/agreement | GET | /annotation-tasks/{task_id}/agreement | training_data | ❌ | 0 |  |
| GET /annotation-tasks/{task_id}/progress | GET | /annotation-tasks/{task_id}/progress | training_data | ❌ | 0 |  |
| GET /annotations | GET | /annotations | training_data | ❌ | 0 |  |
| GET /anomalies | GET | /anomalies | analytics | ❌ | 0 |  |
| GET /api-keys | GET | /api-keys | security | ❌ | 0 |  |
| GET /approval-levels | GET | /approval-levels | release_strategy | ❌ | 0 |  |
| GET /assignments/{user_id}/{experiment_id} | GET | /assignments/{user_id}/{experiment_id} | assignment_cache | ❌ | 0 |  |
| GET /audit/logs | GET | /audit/logs | enterprise | ❌ | 0 |  |
| GET /backpressure/status | GET | /backpressure/status | streaming | ❌ | 0 |  |
| GET /backup/statistics | GET | /backup/statistics | fault_tolerance | ❌ | 0 |  |
| GET /batch/{batch_id}/results | GET | /batch/{batch_id}/results | knowledge_extraction | ❌ | 0 |  |
| GET /batch/{batch_id}/status | GET | /batch/{batch_id}/status | knowledge_extraction | ❌ | 0 |  |
| GET /benchmark-suites | GET | /benchmark-suites | model_evaluation | ❌ | 0 |  |
| GET /buffer/metrics | GET | /buffer/metrics | event_batch | ❌ | 0 |  |
| GET /cache/stats | GET | /cache/stats | personalization | ❌ | 0 |  |
| GET /categories | GET | /categories | risk_assessment | ❌ | 0 |  |
| GET /chain/{chain_id} | GET | /chain/{chain_id} | reasoning | ❌ | 0 |  |
| GET /clusters | GET | /clusters | emotion_modeling | ❌ | 0 |  |
| GET /collections/{collection_name}/stats | GET | /collections/{collection_name}/stats | pgvector | ❌ | 0 |  |
| GET /combination_modes | GET | /combination_modes | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /components/{component_id} | GET | /components/{component_id} | platform_integration | ❌ | 0 |  |
| GET /configs/templates | GET | /configs/templates | fine_tuning | ❌ | 0 |  |
| GET /configuration | GET | /configuration | enterprise | ❌ | 0 |  |
| GET /conflicts | GET | /conflicts | distributed_task | ❌ | 0 |  |
| GET /consistency/statistics | GET | /consistency/statistics | fault_tolerance | ❌ | 0 |  |
| GET /conversations | GET | /conversations | agents | ❌ | 0 |  |
| GET /conversations/{conversation_id}/history | GET | /conversations/{conversation_id}/history | agents | ❌ | 0 |  |
| GET /conversations/{conversation_id}/status | GET | /conversations/{conversation_id}/status | agents | ❌ | 0 |  |
| GET /crisis-prediction/{user_id} | GET | /crisis-prediction/{user_id} | emotional_intelligence | ❌ | 0 |  |
| GET /current-phase/{exec_id} | GET | /current-phase/{exec_id} | traffic_ramp | ❌ | 0 |  |
| GET /dashboard/stats | GET | /dashboard/stats | analytics | ❌ | 0 |  |
| GET /datasets/{dataset_id} | GET | /datasets/{dataset_id} | fine_tuning | ❌ | 0 |  |
| GET /datasets/{dataset_name}/versions | GET | /datasets/{dataset_name}/versions | training_data | ❌ | 0 |  |
| GET /dead-letter | GET | /dead-letter | events | ❌ | 0 |  |
| GET /debug/trace/{query_id} | GET | /debug/trace/{query_id} | graphrag | ❌ | 0 |  |
| GET /demo-scenarios | GET | /demo-scenarios | explainable_ai | ❌ | 0 |  |
| GET /deployment/{deployment_id} | GET | /deployment/{deployment_id} | model_service | ❌ | 0 |  |
| GET /documentation/status | GET | /documentation/status | platform_integration | ❌ | 0 |  |
| GET /effect-size-guidelines | GET | /effect-size-guidelines | power_analysis | ❌ | 0 |  |
| GET /emotional-patterns/{user_id} | GET | /emotional-patterns/{user_id} | emotional_intelligence | ❌ | 0 |  |
| GET /entities/{entity_id} | GET | /entities/{entity_id} | knowledge_graph | ❌ | 0 |  |
| GET /entities/{entity_id}/relations | GET | /entities/{entity_id}/relations | knowledge_graph | ❌ | 0 |  |
| GET /enums/backup-types | GET | /enums/backup-types | fault_tolerance | ❌ | 0 |  |
| GET /enums/fault-types | GET | /enums/fault-types | fault_tolerance | ❌ | 0 |  |
| GET /enums/recovery-strategies | GET | /enums/recovery-strategies | fault_tolerance | ❌ | 0 |  |
| GET /enums/severities | GET | /enums/severities | fault_tolerance | ❌ | 0 |  |
| GET /environments | GET | /environments | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /events/{event_id} | GET | /events/{event_id} | event_tracking | ❌ | 0 |  |
| GET /events/{user_id} | GET | /events/{user_id} | emotional_memory | ❌ | 0 |  |
| GET /examples/simple-regression | GET | /examples/simple-regression | tensorflow | ❌ | 0 |  |
| GET /executions | GET | /executions | multi_step_reasoning | ❌ | 0 |  |
| GET /executions/{exec_id} | GET | /executions/{exec_id} | release_strategy | ❌ | 0 |  |
| GET /executions/{execution_id} | GET | /executions/{execution_id} | multi_step_reasoning | ❌ | 0 |  |
| GET /executions/{execution_id}/results | GET | /executions/{execution_id}/results | multi_step_reasoning | ❌ | 0 |  |
| GET /experiment/{experiment_id}/summary | GET | /experiment/{experiment_id}/summary | realtime_metrics | ❌ | 0 |  |
| GET /experiments/{experiment_id}/progress | GET | /experiments/{experiment_id}/progress | hyperparameter_optimization | ❌ | 0 |  |
| GET /experiments/{experiment_id}/results | GET | /experiments/{experiment_id}/results | bandit_recommendations | ❌ | 0 |  |
| GET /experiments/{experiment_id}/trials | GET | /experiments/{experiment_id}/trials | hyperparameter_optimization | ❌ | 0 |  |
| GET /experiments/{experiment_id}/visualizations | GET | /experiments/{experiment_id}/visualizations | hyperparameter_optimization | ❌ | 0 |  |
| GET /explanation-types | GET | /explanation-types | explainable_ai | ❌ | 0 |  |
| GET /exploration_strategies | GET | /exploration_strategies | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /export/events | GET | /export/events | analytics | ❌ | 0 |  |
| GET /export/metrics | GET | /export/metrics | model_evaluation | ❌ | 0 |  |
| GET /faults | GET | /faults | fault_tolerance | ❌ | 0 |  |
| GET /features/realtime/{user_id} | GET | /features/realtime/{user_id} | personalization | ❌ | 0 |  |
| GET /flow-control/metrics | GET | /flow-control/metrics | streaming | ❌ | 0 |  |
| GET /graph/stats | GET | /graph/stats | memory_analytics | ❌ | 0 |  |
| GET /graphrag/health | GET | /graphrag/health | rag | ❌ | 0 |  |
| GET /groups | GET | /groups | cluster_management | ❌ | 0 |  |
| GET /health-dashboard/{user_id} | GET | /health-dashboard/{user_id} | emotional_intelligence | ❌ | 0 |  |
| GET /health-insights/{user_id} | GET | /health-insights/{user_id} | emotional_intelligence | ❌ | 0 |  |
| GET /health/all | GET | /health/all | mock_endpoints | ❌ | 0 |  |
| GET /health/check | GET | /health/check | workflows | ❌ | 0 |  |
| GET /health/{component_id} | GET | /health/{component_id} | fault_tolerance | ❌ | 0 |  |
| GET /history/{experiment_id} | GET | /history/{experiment_id} | risk_assessment | ❌ | 0 |  |
| GET /hooks/status | GET | /hooks/status | langgraph_features | ❌ | 0 |  |
| GET /index/stats | GET | /index/stats | rag | ❌ | 0 |  |
| GET /info | GET | /info | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /jobs | GET | /jobs | event_batch | ❌ | 0 |  |
| GET /jobs/{job_id} | GET | /jobs/{job_id} | model_compression | ❌ | 0 |  |
| GET /jobs/{job_id}/logs | GET | /jobs/{job_id}/logs | fine_tuning | ❌ | 0 |  |
| GET /jobs/{job_id}/metrics | GET | /jobs/{job_id}/metrics | fine_tuning | ❌ | 0 |  |
| GET /jobs/{job_id}/progress | GET | /jobs/{job_id}/progress | fine_tuning | ❌ | 0 |  |
| GET /jobs/{job_id}/status | GET | /jobs/{job_id}/status | event_batch | ❌ | 0 |  |
| GET /learning/{session_id}/stats | GET | /learning/{session_id}/stats | model_service | ❌ | 0 |  |
| GET /live | GET | /live | health | ❌ | 0 |  |
| GET /mcp-tools/audit | GET | /mcp-tools/audit | security | ❌ | 0 |  |
| GET /mcp-tools/pending-approvals | GET | /mcp-tools/pending-approvals | security | ❌ | 0 |  |
| GET /mcp-tools/permissions | GET | /mcp-tools/permissions | security | ❌ | 0 |  |
| GET /mean | GET | /mean | statistical_analysis | ❌ | 0 |  |
| GET /memories | GET | /memories | emotional_memory | ❌ | 0 |  |
| GET /methods | GET | /methods | model_compression | ❌ | 0 |  |
| GET /metric-definition/{metric_name} | GET | /metric-definition/{metric_name} | realtime_metrics | ❌ | 0 |  |
| GET /metrics-catalog | GET | /metrics-catalog | realtime_metrics | ❌ | 0 |  |
| GET /metrics/realtime | GET | /metrics/realtime | feedback | ❌ | 0 |  |
| GET /metrics/summary | GET | /metrics/summary | cluster_management | ❌ | 0 |  |
| GET /metrics/trends/{metric_name} | GET | /metrics/trends/{metric_name} | cluster_management | ❌ | 0 |  |
| GET /models/status | GET | /models/status | personalization | ❌ | 0 |  |
| GET /models/{model_name} | GET | /models/{model_name} | tensorflow | ❌ | 0 |  |
| GET /models/{model_name}/versions/{version} | GET | /models/{model_name}/versions/{version} | model_service | ❌ | 0 |  |
| GET /models/{name} | GET | /models/{name} | model_registry | ❌ | 0 |  |
| GET /models/{name}/download | GET | /models/{name}/download | model_registry | ❌ | 0 |  |
| GET /models/{name}/export | GET | /models/{name}/export | model_registry | ❌ | 0 |  |
| GET /modes | GET | /modes | auto_scaling | ❌ | 0 |  |
| GET /monitoring/agents | GET | /monitoring/agents | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /monitoring/metrics/{metric_name} | GET | /monitoring/metrics/{metric_name} | model_service | ❌ | 0 |  |
| GET /monitoring/statistics | GET | /monitoring/statistics | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /monitoring/status | GET | /monitoring/status | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /multi-agent/agents | GET | /multi-agent/agents | mock_endpoints | ❌ | 0 |  |
| GET /network | GET | /network | offline | ❌ | 0 |  |
| GET /operations | GET | /operations | offline | ❌ | 0 |  |
| GET /operations/history | GET | /operations/history | cluster_management | ❌ | 0 |  |
| GET /operators | GET | /operators | targeting_rules | ❌ | 0 |  |
| GET /optimization/metrics | GET | /optimization/metrics | platform_integration | ❌ | 0 |  |
| GET /parameter-types | GET | /parameter-types | hyperparameter_optimization | ❌ | 0 |  |
| GET /path | GET | /path | knowledge_graph | ❌ | 0 |  |
| GET /patterns | GET | /patterns | analytics | ❌ | 0 |  |
| GET /patterns/triggers | GET | /patterns/triggers | emotional_memory | ❌ | 0 |  |
| GET /patterns/{user_id} | GET | /patterns/{user_id} | emotional_memory | ❌ | 0 |  |
| GET /performance/metrics | GET | /performance/metrics | pgvector | ❌ | 0 |  |
| GET /performance/models/{model_name} | GET | /performance/models/{model_name} | model_evaluation | ❌ | 0 |  |
| GET /performance/slow-queries | GET | /performance/slow-queries | knowledge_graph | ❌ | 0 |  |
| GET /performance/stats | GET | /performance/stats | graphrag | ❌ | 0 |  |
| GET /performance/system | GET | /performance/system | model_evaluation | ❌ | 0 |  |
| GET /performance/targets | GET | /performance/targets | pgvector | ❌ | 0 |  |
| GET /phases | GET | /phases | traffic_ramp | ❌ | 0 |  |
| GET /plans | GET | /plans | traffic_ramp | ❌ | 0 |  |
| GET /preferences/{user_id} | GET | /preferences/{user_id} | emotional_memory | ❌ | 0 |  |
| GET /profile | GET | /profile | emotion_modeling | ❌ | 0 |  |
| GET /pruning-strategies | GET | /pruning-strategies | hyperparameter_optimization | ❌ | 0 |  |
| GET /quality/issues | GET | /quality/issues | knowledge_graph | ❌ | 0 |  |
| GET /quality/metrics | GET | /quality/metrics | knowledge_graph | ❌ | 0 |  |
| GET /quantization/config | GET | /quantization/config | pgvector | ❌ | 0 |  |
| GET /query/{query_id} | GET | /query/{query_id} | graphrag | ❌ | 0 |  |
| GET /queue-status | GET | /queue-status | training_data | ❌ | 0 |  |
| GET /queue/status | GET | /queue/status | streaming | ❌ | 0 |  |
| GET /ready | GET | /ready | health | ❌ | 0 |  |
| GET /realtime/events | GET | /realtime/events | analytics | ❌ | 0 |  |
| GET /recommendations/{experiment_id} | GET | /recommendations/{experiment_id} | auto_scaling | ❌ | 0 |  |
| GET /records | GET | /records | training_data | ❌ | 0 |  |
| GET /recovery/statistics | GET | /recovery/statistics | fault_tolerance | ❌ | 0 |  |
| GET /release-types | GET | /release-types | release_strategy | ❌ | 0 |  |
| GET /reports/{report_id} | GET | /reports/{report_id} | analytics | ❌ | 0 |  |
| GET /reports/{report_id}/download | GET | /reports/{report_id}/download | analytics | ❌ | 0 |  |
| GET /resource-status | GET | /resource-status | hyperparameter_optimization | ❌ | 0 |  |
| GET /reward_functions | GET | /reward_functions | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /risk-assessment | GET | /risk-assessment | security | ❌ | 0 |  |
| GET /risk-levels | GET | /risk-levels | risk_assessment | ❌ | 0 |  |
| GET /risk-trends/{user_id} | GET | /risk-trends/{user_id} | emotional_intelligence | ❌ | 0 |  |
| GET /rollback/status/{exec_id} | GET | /rollback/status/{exec_id} | risk_assessment | ❌ | 0 |  |
| GET /rules | GET | /rules | alert_rules | ❌ | 0 |  |
| GET /rules/{rule_id} | GET | /rules/{rule_id} | alert_rules | ❌ | 0 |  |
| GET /sample-size-calculator | GET | /sample-size-calculator | power_analysis | ❌ | 0 |  |
| GET /scaling/history | GET | /scaling/history | cluster_management | ❌ | 0 |  |
| GET /scaling/recommendations | GET | /scaling/recommendations | cluster_management | ❌ | 0 |  |
| GET /selector/stats | GET | /selector/stats | unified | ❌ | 0 |  |
| GET /session/{session_id} | GET | /session/{session_id} | memory_management | ❌ | 0 |  |
| GET /sessions/{session_id}/metrics | GET | /sessions/{session_id}/metrics | streaming | ❌ | 0 |  |
| GET /sse/{session_id} | GET | /sse/{session_id} | streaming | ❌ | 0 |  |
| GET /state/history | GET | /state/history | emotion_modeling | ❌ | 0 |  |
| GET /state/latest | GET | /state/latest | emotion_modeling | ❌ | 0 |  |
| GET /statistics/overview | GET | /statistics/overview | layered_experiments | ❌ | 0 |  |
| GET /statistics/{user_id} | GET | /statistics/{user_id} | emotional_memory | ❌ | 0 |  |
| GET /stats/summary | GET | /stats/summary | files | ❌ | 0 |  |
| GET /status/{content_id} | GET | /status/{content_id} | multimodal | ❌ | 0 |  |
| GET /status/{exec_id} | GET | /status/{exec_id} | traffic_ramp | ❌ | 0 |  |
| GET /status/{session_id} | GET | /status/{session_id} | unified | ❌ | 0 |  |
| GET /status/{task_id} | GET | /status/{task_id} | distributed_task | ❌ | 0 |  |
| GET /strategies/{strategy_id} | GET | /strategies/{strategy_id} | release_strategy | ❌ | 0 |  |
| GET /strategy_performance | GET | /strategy_performance | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /subgraph/{entity_id} | GET | /subgraph/{entity_id} | knowledge_graph | ❌ | 0 |  |
| GET /summary/{experiment_id} | GET | /summary/{experiment_id} | anomaly_detection | ❌ | 0 |  |
| GET /supported-formats | GET | /supported-formats | documents | ❌ | 0 |  |
| GET /system-status | GET | /system-status | emotional_intelligence | ❌ | 0 |  |
| GET /system/metrics | GET | /system/metrics | multi_step_reasoning | ❌ | 0 |  |
| GET /tasks/{task_name} | GET | /tasks/{task_name} | hyperparameter_optimization | ❌ | 0 |  |
| GET /templates/configs | GET | /templates/configs | qlearning_tensorflow_backup | ❌ | 0 |  |
| GET /thresholds | GET | /thresholds | risk_assessment | ❌ | 0 |  |
| GET /tools | GET | /tools | mcp | ❌ | 0 |  |
| GET /tools/filesystem/list | GET | /tools/filesystem/list | mcp | ❌ | 0 |  |
| GET /topology | GET | /topology | enterprise | ❌ | 0 |  |
| GET /transitions | GET | /transitions | emotion_modeling | ❌ | 0 |  |
| GET /trends | GET | /trends | memory_analytics | ❌ | 0 |  |
| GET /triggers | GET | /triggers | auto_scaling | ❌ | 0 |  |
| GET /types | GET | /types | anomaly_detection | ❌ | 0 |  |
| GET /user/{user_id} | GET | /user/{user_id} | experiments | ❌ | 0 |  |
| GET /user/{user_id}/experiments | GET | /user/{user_id}/experiments | layered_experiments | ❌ | 0 |  |
| GET /user/{user_id}/profile | GET | /user/{user_id}/profile | personalization | ❌ | 0 |  |
| GET /users/{user_id}/assignments | GET | /users/{user_id}/assignments | assignment_cache | ❌ | 0 |  |
| GET /variance | GET | /variance | statistical_analysis | ❌ | 0 |  |
| GET /version-statistics | GET | /version-statistics | training_data | ❌ | 0 |  |
| GET /versions/{version_id} | GET | /versions/{version_id} | training_data | ❌ | 0 |  |
| GET /versions/{version_id}/history | GET | /versions/{version_id}/history | training_data | ❌ | 0 |  |
| GET /workers | GET | /workers | batch | ❌ | 0 |  |
| GET /workflows/{workflow_id}/status | GET | /workflows/{workflow_id}/status | platform_integration | ❌ | 0 |  |
| GET /ws/stats | GET | /ws/stats | analytics | ❌ | 0 |  |
| GET /{doc_id}/versions | GET | /{doc_id}/versions | documents | ❌ | 0 |  |
| GET /{experiment_id}/allocation/distribution | GET | /{experiment_id}/allocation/distribution | experiments | ❌ | 0 |  |
| GET /{experiment_id}/metrics/{metric_name} | GET | /{experiment_id}/metrics/{metric_name} | experiments | ❌ | 0 |  |
| GET /{experiment_id}/monitor | GET | /{experiment_id}/monitor | experiments | ❌ | 0 |  |
| GET /{experiment_id}/results | GET | /{experiment_id}/results | experiments | ❌ | 0 |  |
| GET /{file_id} | GET | /{file_id} | files | ❌ | 0 |  |
| GET /{file_id}/download | GET | /{file_id}/download | files | ❌ | 0 |  |
| GET /{layer_id} | GET | /{layer_id} | layered_experiments | ❌ | 0 |  |
| GET /{memory_id} | GET | /{memory_id} | memory_management | ❌ | 0 |  |
| GET /{memory_id}/related | GET | /{memory_id}/related | memory_management | ❌ | 0 |  |
| GET /{workflow_id} | GET | /{workflow_id} | workflows | ❌ | 0 |  |
| GET /{workflow_id}/checkpoints | GET | /{workflow_id}/checkpoints | workflows | ❌ | 0 |  |
| GET /{workflow_id}/status | GET | /{workflow_id}/status | workflows | ❌ | 0 |  |
| POST /ab-test-sample-size | POST | /ab-test-sample-size | power_analysis | ❌ | 0 |  |
| POST /abtest/{test_id}/record | POST | /abtest/{test_id}/record | model_service | ❌ | 0 |  |
| POST /admin/schema/initialize | POST | /admin/schema/initialize | knowledge_graph | ❌ | 0 |  |
| POST /agentic/feedback | POST | /agentic/feedback | rag | ❌ | 0 |  |
| POST /agentic/query | POST | /agentic/query | rag | ❌ | 0 |  |
| POST /agentic/query/stream | POST | /agentic/query/stream | rag | ❌ | 0 |  |
| POST /agents/batch/start | POST | /agents/batch/start | cluster_management | ❌ | 0 |  |
| POST /agents/batch/stop | POST | /agents/batch/stop | cluster_management | ❌ | 0 |  |
| POST /agents/{agent_id}/restart | POST | /agents/{agent_id}/restart | cluster_management | ❌ | 0 |  |
| POST /agents/{agent_id}/start | POST | /agents/{agent_id}/start | cluster_management | ❌ | 0 |  |
| POST /agents/{agent_id}/stop | POST | /agents/{agent_id}/stop | cluster_management | ❌ | 0 |  |
| POST /agents/{session_id}/adaptive_recommendation | POST | /agents/{session_id}/adaptive_recommendation | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/batch_inference | POST | /agents/{session_id}/batch_inference | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/evaluate | POST | /agents/{session_id}/evaluate | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/hybrid_recommendation | POST | /agents/{session_id}/hybrid_recommendation | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/inference | POST | /agents/{session_id}/inference | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/log_event | POST | /agents/{session_id}/log_event | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/predict | POST | /agents/{session_id}/predict | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/stop | POST | /agents/{session_id}/stop | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/train | POST | /agents/{session_id}/train | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /agents/{session_id}/warmup | POST | /agents/{session_id}/warmup | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /alerts/{alert_id}/acknowledge | POST | /alerts/{alert_id}/acknowledge | alert_rules | ❌ | 0 |  |
| POST /alerts/{alert_id}/resolve | POST | /alerts/{alert_id}/resolve | distributed_security | ❌ | 0 |  |
| POST /analytics | POST | /analytics | emotion_modeling | ❌ | 0 |  |
| POST /analyze | POST | /analyze | analytics | ❌ | 0 |  |
| POST /analyze/audio | POST | /analyze/audio | emotion_recognition | ❌ | 0 |  |
| POST /analyze/batch | POST | /analyze/batch | social_emotion_api | ❌ | 0 |  |
| POST /analyze/cultural-adaptation | POST | /analyze/cultural-adaptation | social_emotional_understanding | ❌ | 0 |  |
| POST /analyze/group-emotion | POST | /analyze/group-emotion | social_emotional_understanding | ❌ | 0 |  |
| POST /analyze/image | POST | /analyze/image | multimodal | ❌ | 0 |  |
| POST /analyze/multimodal | POST | /analyze/multimodal | emotion_recognition | ❌ | 0 |  |
| POST /analyze/relationships | POST | /analyze/relationships | social_emotional_understanding | ❌ | 0 |  |
| POST /analyze/social-context | POST | /analyze/social-context | social_emotional_understanding | ❌ | 0 |  |
| POST /analyze/text | POST | /analyze/text | emotion_recognition | ❌ | 0 |  |
| POST /analyze/visual | POST | /analyze/visual | emotion_recognition | ❌ | 0 |  |
| POST /annotation-tasks/{task_id}/assign | POST | /annotation-tasks/{task_id}/assign | training_data | ❌ | 0 |  |
| POST /annotations | POST | /annotations | training_data | ❌ | 0 |  |
| POST /api-keys | POST | /api-keys | security | ❌ | 0 |  |
| POST /approve | POST | /approve | release_strategy | ❌ | 0 |  |
| POST /assess | POST | /assess | risk_assessment | ❌ | 0 |  |
| POST /assign/{user_id} | POST | /assign/{user_id} | layered_experiments | ❌ | 0 |  |
| POST /assignments | POST | /assignments | assignment_cache | ❌ | 0 |  |
| POST /assignments/batch | POST | /assignments/batch | assignment_cache | ❌ | 0 |  |
| POST /assignments/batch-get | POST | /assignments/batch-get | assignment_cache | ❌ | 0 |  |
| POST /authenticate | POST | /authenticate | distributed_security | ❌ | 0 |  |
| POST /authorize | POST | /authorize | distributed_security | ❌ | 0 |  |
| POST /backup/manual | POST | /backup/manual | fault_tolerance | ❌ | 0 |  |
| POST /backup/restore | POST | /backup/restore | fault_tolerance | ❌ | 0 |  |
| POST /backup/validate | POST | /backup/validate | fault_tolerance | ❌ | 0 |  |
| POST /basic-stats | POST | /basic-stats | statistical_analysis | ❌ | 0 |  |
| POST /batch-assign | POST | /batch-assign | layered_experiments | ❌ | 0 |  |
| POST /batch-detect | POST | /batch-detect | anomaly_detection | ❌ | 0 |  |
| POST /batch-evaluate | POST | /batch-evaluate | model_evaluation | ❌ | 0 |  |
| POST /batch-upload | POST | /batch-upload | documents | ❌ | 0 |  |
| POST /batch-upsert | POST | /batch-upsert | knowledge_graph | ❌ | 0 |  |
| POST /batch/create | POST | /batch/create | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /batch/submit | POST | /batch/submit | knowledge_extraction | ❌ | 0 |  |
| POST /batches/submit | POST | /batches/submit | event_batch | ❌ | 0 |  |
| POST /broadcast | POST | /broadcast | emotion_websocket | ❌ | 0 |  |
| POST /buffer/flush | POST | /buffer/flush | event_batch | ❌ | 0 |  |
| POST /cache/invalidate/{user_id} | POST | /cache/invalidate/{user_id} | personalization | ❌ | 0 |  |
| POST /calculate | POST | /calculate | realtime_metrics | ❌ | 0 |  |
| POST /calculate-effect-size | POST | /calculate-effect-size | power_analysis | ❌ | 0 |  |
| POST /calculate-power | POST | /calculate-power | power_analysis | ❌ | 0 |  |
| POST /calculate-sample-size | POST | /calculate-sample-size | power_analysis | ❌ | 0 |  |
| POST /cancel/{task_id} | POST | /cancel/{task_id} | distributed_task | ❌ | 0 |  |
| POST /chain | POST | /chain | reasoning | ❌ | 0 |  |
| POST /chain/{chain_id}/branch | POST | /chain/{chain_id}/branch | reasoning | ❌ | 0 |  |
| POST /chain/{chain_id}/recover | POST | /chain/{chain_id}/recover | reasoning | ❌ | 0 |  |
| POST /chain/{chain_id}/validate | POST | /chain/{chain_id}/validate | reasoning | ❌ | 0 |  |
| POST /check-data-quality | POST | /check-data-quality | anomaly_detection | ❌ | 0 |  |
| POST /check-eligibility | POST | /check-eligibility | targeting_rules | ❌ | 0 |  |
| POST /check-srm | POST | /check-srm | anomaly_detection | ❌ | 0 |  |
| POST /checkpoint/create | POST | /checkpoint/create | distributed_task | ❌ | 0 |  |
| POST /checkpoint/rollback | POST | /checkpoint/rollback | distributed_task | ❌ | 0 |  |
| POST /clear-history | POST | /clear-history | unified | ❌ | 0 |  |
| POST /collect | POST | /collect | training_data | ❌ | 0 |  |
| POST /collections | POST | /collections | pgvector | ❌ | 0 |  |
| POST /communication/encrypt | POST | /communication/encrypt | distributed_security | ❌ | 0 |  |
| POST /compare-algorithms/{task_name} | POST | /compare-algorithms/{task_name} | hyperparameter_optimization | ❌ | 0 |  |
| POST /compare-groups | POST | /compare-groups | realtime_metrics | ❌ | 0 |  |
| POST /compliance/assess | POST | /compliance/assess | enterprise | ❌ | 0 |  |
| POST /comprehensive-analysis | POST | /comprehensive-analysis | social_emotional_understanding | ❌ | 0 |  |
| POST /config/concurrent-jobs | POST | /config/concurrent-jobs | event_batch | ❌ | 0 |  |
| POST /configs/validate | POST | /configs/validate | fine_tuning | ❌ | 0 |  |
| POST /configure | POST | /configure | anomaly_detection | ❌ | 0 |  |
| POST /conflicts/resolve/{conflict_id} | POST | /conflicts/resolve/{conflict_id} | distributed_task | ❌ | 0 |  |
| POST /consistency/check | POST | /consistency/check | fault_tolerance | ❌ | 0 |  |
| POST /consistency/force-repair | POST | /consistency/force-repair | fault_tolerance | ❌ | 0 |  |
| POST /consistency/{check_id}/repair | POST | /consistency/{check_id}/repair | fault_tolerance | ❌ | 0 |  |
| POST /consolidate/{session_id} | POST | /consolidate/{session_id} | memory_management | ❌ | 0 |  |
| POST /conversion-stats | POST | /conversion-stats | statistical_analysis | ❌ | 0 |  |
| POST /cot-reasoning | POST | /cot-reasoning | explainable_ai | ❌ | 0 |  |
| POST /crisis-detection | POST | /crisis-detection | emotional_intelligence | ❌ | 0 |  |
| POST /datasets/{dataset_id}/validate | POST | /datasets/{dataset_id}/validate | fine_tuning | ❌ | 0 |  |
| POST /datasets/{dataset_name}/rollback | POST | /datasets/{dataset_name}/rollback | training_data | ❌ | 0 |  |
| POST /decide | POST | /decide | emotional_intelligence | ❌ | 0 |  |
| POST /decisions/generate | POST | /decisions/generate | social_emotional_understanding | ❌ | 0 |  |
| POST /decompose | POST | /decompose | multi_step_reasoning | ❌ | 0 |  |
| POST /demo-scenario | POST | /demo-scenario | explainable_ai | ❌ | 0 |  |
| POST /detect | POST | /detect | anomaly_detection | ❌ | 0 |  |
| POST /documentation/generate | POST | /documentation/generate | platform_integration | ❌ | 0 |  |
| POST /documentation/training-materials | POST | /documentation/training-materials | platform_integration | ❌ | 0 |  |
| POST /entities/search | POST | /entities/search | knowledge_graph | ❌ | 0 |  |
| POST /evaluate/batch | POST | /evaluate/batch | targeting_rules | ❌ | 0 |  |
| POST /events/batch | POST | /events/batch | experiments | ❌ | 0 |  |
| POST /events/detect | POST | /events/detect | emotional_memory | ❌ | 0 |  |
| POST /events/{event_id}/reprocess | POST | /events/{event_id}/reprocess | event_tracking | ❌ | 0 |  |
| POST /execute/{strategy_id} | POST | /execute/{strategy_id} | release_strategy | ❌ | 0 |  |
| POST /executions/control | POST | /executions/control | multi_step_reasoning | ❌ | 0 |  |
| POST /experiments/{experiment_id}/end | POST | /experiments/{experiment_id}/end | bandit_recommendations | ❌ | 0 |  |
| POST /experiments/{experiment_id}/start | POST | /experiments/{experiment_id}/start | hyperparameter_optimization | ❌ | 0 |  |
| POST /experiments/{experiment_id}/stop | POST | /experiments/{experiment_id}/stop | hyperparameter_optimization | ❌ | 0 |  |
| POST /explicit | POST | /explicit | feedback | ❌ | 0 |  |
| POST /extract | POST | /extract | knowledge_extraction | ❌ | 0 |  |
| POST /features/compute | POST | /features/compute | personalization | ❌ | 0 |  |
| POST /format-explanation | POST | /format-explanation | explainable_ai | ❌ | 0 |  |
| POST /generate-explanation | POST | /generate-explanation | explainable_ai | ❌ | 0 |  |
| POST /groups | POST | /groups | cluster_management | ❌ | 0 |  |
| POST /groups/{group_id}/agents/{agent_id} | POST | /groups/{group_id}/agents/{agent_id} | cluster_management | ❌ | 0 |  |
| POST /hooks/configure | POST | /hooks/configure | langgraph_features | ❌ | 0 |  |
| POST /index/directory | POST | /index/directory | rag | ❌ | 0 |  |
| POST /index/file | POST | /index/file | rag | ❌ | 0 |  |
| POST /index/update | POST | /index/update | rag | ❌ | 0 |  |
| POST /inference/models/{model_name}/load | POST | /inference/models/{model_name}/load | model_service | ❌ | 0 |  |
| POST /initialize | POST | /initialize | distributed_task | ❌ | 0 |  |
| POST /intervention-effectiveness | POST | /intervention-effectiveness | emotional_intelligence | ❌ | 0 |  |
| POST /intervention-plan | POST | /intervention-plan | emotional_intelligence | ❌ | 0 |  |
| POST /jobs | POST | /jobs | model_compression | ❌ | 0 |  |
| POST /jobs/{job_id}/cancel | POST | /jobs/{job_id}/cancel | event_batch | ❌ | 0 |  |
| POST /jobs/{job_id}/pause | POST | /jobs/{job_id}/pause | batch | ❌ | 0 |  |
| POST /jobs/{job_id}/resume | POST | /jobs/{job_id}/resume | batch | ❌ | 0 |  |
| POST /jobs/{job_id}/retry | POST | /jobs/{job_id}/retry | batch | ❌ | 0 |  |
| POST /learning/{session_id}/feedback | POST | /learning/{session_id}/feedback | model_service | ❌ | 0 |  |
| POST /learning/{session_id}/update | POST | /learning/{session_id}/update | model_service | ❌ | 0 |  |
| POST /load-balancer/select | POST | /load-balancer/select | service_discovery | ❌ | 0 |  |
| POST /mcp-tools/approve/{request_id} | POST | /mcp-tools/approve/{request_id} | security | ❌ | 0 |  |
| POST /mcp-tools/whitelist | POST | /mcp-tools/whitelist | security | ❌ | 0 |  |
| POST /memories | POST | /memories | emotional_memory | ❌ | 0 |  |
| POST /memories/export/{user_id} | POST | /memories/export/{user_id} | emotional_memory | ❌ | 0 |  |
| POST /memories/search | POST | /memories/search | emotional_memory | ❌ | 0 |  |
| POST /methods/validate | POST | /methods/validate | model_compression | ❌ | 0 |  |
| POST /metrics/query | POST | /metrics/query | cluster_management | ❌ | 0 |  |
| POST /mode/recommendations | POST | /mode/recommendations | unified | ❌ | 0 |  |
| POST /mode/set-default | POST | /mode/set-default | unified | ❌ | 0 |  |
| POST /mode/{mode} | POST | /mode/{mode} | offline | ❌ | 0 |  |
| POST /models/predict | POST | /models/predict | personalization | ❌ | 0 |  |
| POST /models/save | POST | /models/save | tensorflow | ❌ | 0 |  |
| POST /models/train | POST | /models/train | tensorflow | ❌ | 0 |  |
| POST /models/validate | POST | /models/validate | fine_tuning | ❌ | 0 |  |
| POST /models/{model_name}/versions/{version}/validate | POST | /models/{model_name}/versions/{version}/validate | model_service | ❌ | 0 |  |
| POST /models/{name}/register-from-hub | POST | /models/{name}/register-from-hub | model_registry | ❌ | 0 |  |
| POST /monitoring/start | POST | /monitoring/start | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /monitoring/stop | POST | /monitoring/stop | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /multiple-groups-stats | POST | /multiple-groups-stats | statistical_analysis | ❌ | 0 |  |
| POST /optimization/profile/{profile_name} | POST | /optimization/profile/{profile_name} | platform_integration | ❌ | 0 |  |
| POST /optimization/run | POST | /optimization/run | platform_integration | ❌ | 0 |  |
| POST /optimize/{task_name} | POST | /optimize/{task_name} | hyperparameter_optimization | ❌ | 0 |  |
| POST /optimize_weights | POST | /optimize_weights | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /patterns/identify | POST | /patterns/identify | emotional_memory | ❌ | 0 |  |
| POST /patterns/predict | POST | /patterns/predict | emotional_memory | ❌ | 0 |  |
| POST /pause | POST | /pause | traffic_ramp | ❌ | 0 |  |
| POST /percentiles | POST | /percentiles | statistical_analysis | ❌ | 0 |  |
| POST /plans | POST | /plans | traffic_ramp | ❌ | 0 |  |
| POST /preferences/learn | POST | /preferences/learn | emotional_memory | ❌ | 0 |  |
| POST /privacy/consent | POST | /privacy/consent | social_emotion_api | ❌ | 0 |  |
| POST /privacy/policy | POST | /privacy/policy | social_emotion_api | ❌ | 0 |  |
| POST /process | POST | /process | multimodal | ❌ | 0 |  |
| POST /process/batch | POST | /process/batch | multimodal | ❌ | 0 |  |
| POST /proportion-power | POST | /proportion-power | power_analysis | ❌ | 0 |  |
| POST /proportion-sample-size | POST | /proportion-sample-size | power_analysis | ❌ | 0 |  |
| POST /quality/ground-truth | POST | /quality/ground-truth | emotion_intelligence | ❌ | 0 |  |
| POST /quality/score | POST | /quality/score | feedback | ❌ | 0 |  |
| POST /quantization/configure | POST | /quantization/configure | pgvector | ❌ | 0 |  |
| POST /quantization/test | POST | /quantization/test | pgvector | ❌ | 0 |  |
| POST /query-with-files | POST | /query-with-files | multimodal_rag | ❌ | 0 |  |
| POST /query/analyze | POST | /query/analyze | graphrag | ❌ | 0 |  |
| POST /quick-ramp | POST | /quick-ramp | traffic_ramp | ❌ | 0 |  |
| POST /rag/query | POST | /rag/query | mock_endpoints | ❌ | 0 |  |
| POST /react/chat/{conversation_id} | POST | /react/chat/{conversation_id} | agents | ❌ | 0 |  |
| POST /react/task/{conversation_id} | POST | /react/task/{conversation_id} | agents | ❌ | 0 |  |
| POST /real-time-monitor | POST | /real-time-monitor | anomaly_detection | ❌ | 0 |  |
| POST /realtime/broadcast | POST | /realtime/broadcast | analytics | ❌ | 0 |  |
| POST /recommended-plan | POST | /recommended-plan | traffic_ramp | ❌ | 0 |  |
| POST /register-metric | POST | /register-metric | realtime_metrics | ❌ | 0 |  |
| POST /relations | POST | /relations | knowledge_graph | ❌ | 0 |  |
| POST /reprocess | POST | /reprocess | training_data | ❌ | 0 |  |
| POST /resolve | POST | /resolve | offline | ❌ | 0 |  |
| POST /results/{job_id}/download | POST | /results/{job_id}/download | model_compression | ❌ | 0 |  |
| POST /resume | POST | /resume | traffic_ramp | ❌ | 0 |  |
| POST /revoke-access | POST | /revoke-access | distributed_security | ❌ | 0 |  |
| POST /reward/calculate | POST | /reward/calculate | feedback | ❌ | 0 |  |
| POST /risk-assessment | POST | /risk-assessment | emotional_intelligence | ❌ | 0 |  |
| POST /rollback | POST | /rollback | traffic_ramp | ❌ | 0 |  |
| POST /rollback-plan | POST | /rollback-plan | risk_assessment | ❌ | 0 |  |
| POST /rollback/execute | POST | /rollback/execute | risk_assessment | ❌ | 0 |  |
| POST /rules | POST | /rules | alert_rules | ❌ | 0 |  |
| POST /rules/templates | POST | /rules/templates | targeting_rules | ❌ | 0 |  |
| POST /rules/{rule_id}/conditions | POST | /rules/{rule_id}/conditions | auto_scaling | ❌ | 0 |  |
| POST /rules/{rule_id}/test | POST | /rules/{rule_id}/test | acl | ❌ | 0 |  |
| POST /scaling/groups/{group_id}/manual | POST | /scaling/groups/{group_id}/manual | cluster_management | ❌ | 0 |  |
| POST /scaling/policies | POST | /scaling/policies | cluster_management | ❌ | 0 |  |
| POST /search/hybrid | POST | /search/hybrid | pgvector | ❌ | 0 |  |
| POST /search/similarity | POST | /search/similarity | pgvector | ❌ | 0 |  |
| POST /security/scan | POST | /security/scan | enterprise | ❌ | 0 |  |
| POST /selector/set-strategy | POST | /selector/set-strategy | unified | ❌ | 0 |  |
| POST /session/create | POST | /session/create | social_emotion_api | ❌ | 0 |  |
| POST /sessions/{session_id}/cleanup | POST | /sessions/{session_id}/cleanup | streaming | ❌ | 0 |  |
| POST /shutdown | POST | /shutdown | distributed_task | ❌ | 0 |  |
| POST /simulate | POST | /simulate | auto_scaling | ❌ | 0 |  |
| POST /simulate/assignment | POST | /simulate/assignment | layered_experiments | ❌ | 0 |  |
| POST /start-monitoring/{experiment_id} | POST | /start-monitoring/{experiment_id} | realtime_metrics | ❌ | 0 |  |
| POST /state | POST | /state | emotion_modeling | ❌ | 0 |  |
| POST /stop-monitoring | POST | /stop-monitoring | realtime_metrics | ❌ | 0 |  |
| POST /stop/{rule_id} | POST | /stop/{rule_id} | auto_scaling | ❌ | 0 |  |
| POST /storage/optimize | POST | /storage/optimize | emotional_memory | ❌ | 0 |  |
| POST /strategies/compare | POST | /strategies/compare | qlearning_tensorflow_backup | ❌ | 0 |  |
| POST /strategies/from-template | POST | /strategies/from-template | release_strategy | ❌ | 0 |  |
| POST /strategies/recommend | POST | /strategies/recommend | model_compression | ❌ | 0 |  |
| POST /stream-query | POST | /stream-query | multimodal_rag | ❌ | 0 |  |
| POST /suicide-risk-assessment | POST | /suicide-risk-assessment | emotional_intelligence | ❌ | 0 |  |
| POST /system/start | POST | /system/start | fault_tolerance | ❌ | 0 |  |
| POST /system/stop | POST | /system/stop | fault_tolerance | ❌ | 0 |  |
| POST /templates/aggressive | POST | /templates/aggressive | auto_scaling | ❌ | 0 |  |
| POST /templates/safe | POST | /templates/safe | auto_scaling | ❌ | 0 |  |
| POST /test/connections | POST | /test/connections | enterprise | ❌ | 0 |  |
| POST /testing/inject-fault | POST | /testing/inject-fault | fault_tolerance | ❌ | 0 |  |
| POST /tools/call | POST | /tools/call | mcp | ❌ | 0 |  |
| POST /tools/database/query | POST | /tools/database/query | mcp | ❌ | 0 |  |
| POST /tools/filesystem/read | POST | /tools/filesystem/read | mcp | ❌ | 0 |  |
| POST /tools/filesystem/write | POST | /tools/filesystem/write | mcp | ❌ | 0 |  |
| POST /tools/system/command | POST | /tools/system/command | mcp | ❌ | 0 |  |
| POST /trends | POST | /trends | realtime_metrics | ❌ | 0 |  |
| POST /upload-document | POST | /upload-document | multimodal_rag | ❌ | 0 |  |
| POST /upsert-entity | POST | /upsert-entity | knowledge_graph | ❌ | 0 |  |
| POST /user/{user_id}/check-conflicts | POST | /user/{user_id}/check-conflicts | layered_experiments | ❌ | 0 |  |
| POST /vectors | POST | /vectors | pgvector | ❌ | 0 |  |
| POST /versions/{version_id1}/compare/{version_id2} | POST | /versions/{version_id1}/compare/{version_id2} | training_data | ❌ | 0 |  |
| POST /versions/{version_id}/export | POST | /versions/{version_id}/export | training_data | ❌ | 0 |  |
| POST /warmup | POST | /warmup | cache | ❌ | 0 |  |
| POST /workflow-explanation | POST | /workflow-explanation | explainable_ai | ❌ | 0 |  |
| POST /workflows/{workflow_id}/cancel | POST | /workflows/{workflow_id}/cancel | platform_integration | ❌ | 0 |  |
| POST /{doc_id}/analyze-relationships | POST | /{doc_id}/analyze-relationships | documents | ❌ | 0 |  |
| POST /{doc_id}/generate-tags | POST | /{doc_id}/generate-tags | documents | ❌ | 0 |  |
| POST /{doc_id}/rollback | POST | /{doc_id}/rollback | documents | ❌ | 0 |  |
| POST /{experiment_id}/alerts | POST | /{experiment_id}/alerts | experiments | ❌ | 0 |  |
| POST /{experiment_id}/allocate | POST | /{experiment_id}/allocate | experiments | ❌ | 0 |  |
| POST /{experiment_id}/allocate/batch | POST | /{experiment_id}/allocate/batch | experiments | ❌ | 0 |  |
| POST /{experiment_id}/allocation/rules | POST | /{experiment_id}/allocation/rules | experiments | ❌ | 0 |  |
| POST /{experiment_id}/allocation/simulate | POST | /{experiment_id}/allocation/simulate | experiments | ❌ | 0 |  |
| POST /{experiment_id}/allocation/stages | POST | /{experiment_id}/allocation/stages | experiments | ❌ | 0 |  |
| POST /{experiment_id}/assign | POST | /{experiment_id}/assign | experiments | ❌ | 0 |  |
| POST /{experiment_id}/pause | POST | /{experiment_id}/pause | experiments | ❌ | 0 |  |
| POST /{experiment_id}/start | POST | /{experiment_id}/start | experiments | ❌ | 0 |  |
| POST /{experiment_id}/stop | POST | /{experiment_id}/stop | experiments | ❌ | 0 |  |
| POST /{layer_id}/groups | POST | /{layer_id}/groups | layered_experiments | ❌ | 0 |  |
| POST /{memory_id}/associate | POST | /{memory_id}/associate | memory_management | ❌ | 0 |  |
| POST /{workflow_id}/start | POST | /{workflow_id}/start | workflows | ❌ | 0 |  |
| PUT /agents/{agent_id}/metrics | PUT | /agents/{agent_id}/metrics | service_discovery | ❌ | 0 |  |
| PUT /agents/{agent_id}/status | PUT | /agents/{agent_id}/status | service_discovery | ❌ | 0 |  |
| PUT /configuration | PUT | /configuration | enterprise | ❌ | 0 |  |
| PUT /entities/{entity_id} | PUT | /entities/{entity_id} | knowledge_graph | ❌ | 0 |  |
| PUT /item/{item_id}/features | PUT | /item/{item_id}/features | bandit_recommendations | ❌ | 0 |  |
| PUT /jobs/{job_id}/cancel | PUT | /jobs/{job_id}/cancel | model_compression | ❌ | 0 |  |
| PUT /mcp-tools/permissions | PUT | /mcp-tools/permissions | security | ❌ | 0 |  |
| PUT /models/update | PUT | /models/update | personalization | ❌ | 0 |  |
| PUT /rules/{rule_id} | PUT | /rules/{rule_id} | alert_rules | ❌ | 0 |  |
| PUT /sources/{source_id} | PUT | /sources/{source_id} | training_data | ❌ | 0 |  |
| PUT /thresholds | PUT | /thresholds | risk_assessment | ❌ | 0 |  |
| PUT /user/{user_id}/context | PUT | /user/{user_id}/context | bandit_recommendations | ❌ | 0 |  |
| PUT /user/{user_id}/profile | PUT | /user/{user_id}/profile | personalization | ❌ | 0 |  |
| PUT /{memory_id} | PUT | /{memory_id} | memory_management | ❌ | 0 |  |
| PUT /{workflow_id}/control | PUT | /{workflow_id}/control | workflows | ❌ | 0 |  |

## 未使用的API端点

- `POST /query-with-files` (模块: multimodal_rag, 文件: multimodal_rag.py)
- `POST /stream-query` (模块: multimodal_rag, 文件: multimodal_rag.py)
- `POST /upload-document` (模块: multimodal_rag, 文件: multimodal_rag.py)
- `POST /batch-upload` (模块: documents, 文件: documents.py)
- `GET /health/{component_id}` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /faults` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /recovery/statistics` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /backup/manual` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /backup/statistics` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /backup/restore` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /backup/validate` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /consistency/check` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /consistency/statistics` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /consistency/{check_id}/repair` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /consistency/force-repair` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /testing/inject-fault` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /enums/fault-types` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /enums/severities` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /enums/backup-types` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `GET /enums/recovery-strategies` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /system/start` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /system/stop` (模块: fault_tolerance, 文件: fault_tolerance.py)
- `POST /rules` (模块: alert_rules, 文件: alert_rules.py)
- `GET /rules` (模块: alert_rules, 文件: alert_rules.py)
- `GET /rules/{rule_id}` (模块: alert_rules, 文件: alert_rules.py)
- `PUT /rules/{rule_id}` (模块: alert_rules, 文件: alert_rules.py)
- `DELETE /rules/{rule_id}` (模块: alert_rules, 文件: alert_rules.py)
- `POST /evaluate/batch` (模块: targeting_rules, 文件: targeting_rules.py)
- `POST /check-eligibility` (模块: targeting_rules, 文件: targeting_rules.py)
- `DELETE /rules` (模块: targeting_rules, 文件: targeting_rules.py)
- `POST /rules/templates` (模块: targeting_rules, 文件: targeting_rules.py)
- `GET /operators` (模块: targeting_rules, 文件: targeting_rules.py)
- `GET /benchmark-suites` (模块: model_evaluation, 文件: model_evaluation.py)
- `POST /batch-evaluate` (模块: model_evaluation, 文件: model_evaluation.py)
- `GET /jobs` (模块: event_batch, 文件: event_batch.py)
- `GET /jobs/{job_id}` (模块: model_compression, 文件: model_compression.py)
- `DELETE /jobs/{job_id}` (模块: model_compression, 文件: model_compression.py)
- `GET /reports/{report_id}` (模块: analytics, 文件: analytics.py)
- `GET /performance/system` (模块: model_evaluation, 文件: model_evaluation.py)
- `GET /performance/models/{model_name}` (模块: model_evaluation, 文件: model_evaluation.py)
- `POST /alerts/{alert_id}/resolve` (模块: distributed_security, 文件: distributed_security.py)
- `GET /export/metrics` (模块: model_evaluation, 文件: model_evaluation.py)
- `GET /{file_id}` (模块: files, 文件: files.py)
- `GET /{file_id}/download` (模块: files, 文件: files.py)
- `DELETE /{file_id}` (模块: files, 文件: files.py)
- `GET /stats/summary` (模块: files, 文件: files.py)
- `PUT /{memory_id}` (模块: memory_management, 文件: memory_management.py)
- `DELETE /{memory_id}` (模块: memory_management, 文件: memory_management.py)
- `GET /session/{session_id}` (模块: memory_management, 文件: memory_management.py)
- `POST /{memory_id}/associate` (模块: memory_management, 文件: memory_management.py)
- `GET /{memory_id}/related` (模块: memory_management, 文件: memory_management.py)
- `POST /consolidate/{session_id}` (模块: memory_management, 文件: memory_management.py)
- `GET /analytics` (模块: empathy_response, 文件: empathy_response.py)
- `GET /analytics/patterns` (模块: memory_management, 文件: memory_management.py)
- `GET /analytics/trends` (模块: memory_management, 文件: memory_management.py)
- `GET /analytics/graph/stats` (模块: memory_management, 文件: memory_management.py)
- `GET /{memory_id}` (模块: memory_management, 文件: memory_management.py)
- `POST /decide` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /risk-assessment` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /crisis-detection` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /intervention-plan` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /health-dashboard/{user_id}` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /emotional-patterns/{user_id}` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /suicide-risk-assessment` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /risk-trends/{user_id}` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /crisis-prediction/{user_id}` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /intervention-effectiveness` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /health-insights/{user_id}` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `GET /system-status` (模块: emotional_intelligence, 文件: emotional_intelligence.py)
- `POST /state` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `GET /state/latest` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `GET /state/history` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `POST /analytics` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `GET /profile` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `GET /patterns` (模块: analytics, 文件: analytics.py)
- `GET /clusters` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `GET /transitions` (模块: emotion_modeling, 文件: emotion_modeling.py)
- `POST /generate-explanation` (模块: explainable_ai, 文件: explainable_ai.py)
- `POST /cot-reasoning` (模块: explainable_ai, 文件: explainable_ai.py)
- `POST /workflow-explanation` (模块: explainable_ai, 文件: explainable_ai.py)
- `POST /format-explanation` (模块: explainable_ai, 文件: explainable_ai.py)
- `POST /demo-scenario` (模块: explainable_ai, 文件: explainable_ai.py)
- `GET /explanation-types` (模块: explainable_ai, 文件: explainable_ai.py)
- `GET /demo-scenarios` (模块: explainable_ai, 文件: explainable_ai.py)
- `GET /assignments/{user_id}/{experiment_id}` (模块: assignment_cache, 文件: assignment_cache.py)
- `POST /assignments` (模块: assignment_cache, 文件: assignment_cache.py)
- `POST /assignments/batch` (模块: assignment_cache, 文件: assignment_cache.py)
- `GET /users/{user_id}/assignments` (模块: assignment_cache, 文件: assignment_cache.py)
- `POST /assignments/batch-get` (模块: assignment_cache, 文件: assignment_cache.py)
- `DELETE /assignments/{user_id}/{experiment_id}` (模块: assignment_cache, 文件: assignment_cache.py)
- `DELETE /users/{user_id}/assignments` (模块: assignment_cache, 文件: assignment_cache.py)
- `GET /info` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /warmup` (模块: cache, 文件: cache.py)
- `POST /register-metric` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /calculate` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /compare-groups` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /trends` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /start-monitoring/{experiment_id}` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /stop-monitoring` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `GET /metrics-catalog` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `GET /metric-definition/{metric_name}` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `GET /experiment/{experiment_id}/summary` (模块: realtime_metrics, 文件: realtime_metrics.py)
- `POST /analyze/group-emotion` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /analyze/relationships` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /analyze/social-context` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /analyze/cultural-adaptation` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /decisions/generate` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /comprehensive-analysis` (模块: social_emotional_understanding, 文件: social_emotional_understanding.py)
- `POST /initialize` (模块: distributed_task, 文件: distributed_task.py)
- `POST /analyze` (模块: analytics, 文件: analytics.py)
- `POST /analyze/batch` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `POST /session/create` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `DELETE /session/{session_id}` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `POST /privacy/consent` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `DELETE /privacy/consent/{user_id}` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `POST /privacy/policy` (模块: social_emotion_api, 文件: social_emotion_api.py)
- `PUT /sources/{source_id}` (模块: training_data, 文件: training_data.py)
- `DELETE /sources/{source_id}` (模块: training_data, 文件: training_data.py)
- `POST /collect` (模块: training_data, 文件: training_data.py)
- `GET /records` (模块: training_data, 文件: training_data.py)
- `POST /reprocess` (模块: training_data, 文件: training_data.py)
- `GET /annotation-tasks/{task_id}/progress` (模块: training_data, 文件: training_data.py)
- `POST /annotation-tasks/{task_id}/assign` (模块: training_data, 文件: training_data.py)
- `POST /annotations` (模块: training_data, 文件: training_data.py)
- `GET /annotation-tasks/{task_id}/agreement` (模块: training_data, 文件: training_data.py)
- `GET /annotations` (模块: training_data, 文件: training_data.py)
- `GET /datasets/{dataset_name}/versions` (模块: training_data, 文件: training_data.py)
- `GET /versions/{version_id}` (模块: training_data, 文件: training_data.py)
- `GET /versions/{version_id}/history` (模块: training_data, 文件: training_data.py)
- `POST /versions/{version_id1}/compare/{version_id2}` (模块: training_data, 文件: training_data.py)
- `POST /versions/{version_id}/export` (模块: training_data, 文件: training_data.py)
- `POST /datasets/{dataset_name}/rollback` (模块: training_data, 文件: training_data.py)
- `GET /version-statistics` (模块: training_data, 文件: training_data.py)
- `DELETE /versions/{version_id}` (模块: training_data, 文件: training_data.py)
- `GET /queue-status` (模块: training_data, 文件: training_data.py)
- `POST /jobs` (模块: model_compression, 文件: model_compression.py)
- `PUT /jobs/{job_id}/cancel` (模块: model_compression, 文件: model_compression.py)
- `GET /jobs/{job_id}/logs` (模块: fine_tuning, 文件: fine_tuning.py)
- `GET /jobs/{job_id}/metrics` (模块: fine_tuning, 文件: fine_tuning.py)
- `GET /jobs/{job_id}/progress` (模块: fine_tuning, 文件: fine_tuning.py)
- `POST /jobs/{job_id}/pause` (模块: batch, 文件: batch.py)
- `POST /jobs/{job_id}/resume` (模块: batch, 文件: batch.py)
- `POST /models/validate` (模块: fine_tuning, 文件: fine_tuning.py)
- `GET /configs/templates` (模块: fine_tuning, 文件: fine_tuning.py)
- `POST /configs/validate` (模块: fine_tuning, 文件: fine_tuning.py)
- `GET /datasets/{dataset_id}` (模块: fine_tuning, 文件: fine_tuning.py)
- `POST /datasets/{dataset_id}/validate` (模块: fine_tuning, 文件: fine_tuning.py)
- `POST /process` (模块: multimodal, 文件: multimodal.py)
- `GET /status/{session_id}` (模块: unified, 文件: unified.py)
- `POST /mode/recommendations` (模块: unified, 文件: unified.py)
- `GET /selector/stats` (模块: unified, 文件: unified.py)
- `POST /mode/set-default` (模块: unified, 文件: unified.py)
- `POST /selector/set-strategy` (模块: unified, 文件: unified.py)
- `POST /clear-history` (模块: unified, 文件: unified.py)
- `GET /modes` (模块: auto_scaling, 文件: auto_scaling.py)
- `POST /strategies/from-template` (模块: release_strategy, 文件: release_strategy.py)
- `GET /strategies/{strategy_id}` (模块: release_strategy, 文件: release_strategy.py)
- `POST /execute/{strategy_id}` (模块: release_strategy, 文件: release_strategy.py)
- `POST /approve` (模块: release_strategy, 文件: release_strategy.py)
- `GET /executions/{exec_id}` (模块: release_strategy, 文件: release_strategy.py)
- `GET /release-types` (模块: release_strategy, 文件: release_strategy.py)
- `GET /approval-levels` (模块: release_strategy, 文件: release_strategy.py)
- `GET /environments` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /live` (模块: health, 文件: health.py)
- `GET /ready` (模块: health, 文件: health.py)
- `GET /dead-letter` (模块: events, 文件: events.py)
- `GET /entities/{entity_id}` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `PUT /entities/{entity_id}` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `DELETE /entities/{entity_id}` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /entities/search` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /relations` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /entities/{entity_id}/relations` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /path` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /subgraph/{entity_id}` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /upsert-entity` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /batch-upsert` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /quality/metrics` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /quality/issues` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /performance/stats` (模块: graphrag, 文件: graphrag.py)
- `GET /performance/slow-queries` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `DELETE /performance/cache` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /admin/schema/initialize` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `GET /admin/schema/status` (模块: knowledge_graph, 文件: knowledge_graph.py)
- `POST /explicit` (模块: feedback, 文件: feedback.py)
- `GET /user/{user_id}` (模块: experiments, 文件: experiments.py)
- `GET /analytics/user/{user_id}` (模块: feedback, 文件: feedback.py)
- `GET /analytics/item/{item_id}` (模块: feedback, 文件: feedback.py)
- `POST /quality/score` (模块: feedback, 文件: feedback.py)
- `POST /process/batch` (模块: multimodal, 文件: multimodal.py)
- `POST /reward/calculate` (模块: feedback, 文件: feedback.py)
- `GET /metrics/realtime` (模块: feedback, 文件: feedback.py)
- `GET /{layer_id}` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /{layer_id}/groups` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /assign/{user_id}` (模块: layered_experiments, 文件: layered_experiments.py)
- `GET /user/{user_id}/experiments` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /user/{user_id}/check-conflicts` (模块: layered_experiments, 文件: layered_experiments.py)
- `DELETE /user/{user_id}/assignments` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /batch-assign` (模块: layered_experiments, 文件: layered_experiments.py)
- `GET /statistics/overview` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /simulate/assignment` (模块: layered_experiments, 文件: layered_experiments.py)
- `POST /query/analyze` (模块: graphrag, 文件: graphrag.py)
- `GET /query/{query_id}` (模块: graphrag, 文件: graphrag.py)
- `GET /debug/trace/{query_id}` (模块: graphrag, 文件: graphrag.py)
- `GET /agents/{session_id}` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `DELETE /agents/{session_id}` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/train` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/stop` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/progress` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/predict` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/evaluate` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/statistics` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /templates/configs` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /supported-formats` (模块: documents, 文件: documents.py)
- `POST /{doc_id}/analyze-relationships` (模块: documents, 文件: documents.py)
- `POST /{doc_id}/generate-tags` (模块: documents, 文件: documents.py)
- `GET /{doc_id}/versions` (模块: documents, 文件: documents.py)
- `POST /{doc_id}/rollback` (模块: documents, 文件: documents.py)
- `POST /extract` (模块: knowledge_extraction, 文件: knowledge_extraction.py)
- `POST /batch/submit` (模块: knowledge_extraction, 文件: knowledge_extraction.py)
- `GET /batch/{batch_id}/status` (模块: knowledge_extraction, 文件: knowledge_extraction.py)
- `GET /batch/{batch_id}/results` (模块: knowledge_extraction, 文件: knowledge_extraction.py)
- `DELETE /batch/{batch_id}` (模块: knowledge_extraction, 文件: knowledge_extraction.py)
- `POST /jobs/{job_id}/cancel` (模块: event_batch, 文件: event_batch.py)
- `POST /jobs/{job_id}/retry` (模块: batch, 文件: batch.py)
- `GET /workers` (模块: batch, 文件: batch.py)
- `POST /plans` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `POST /pause` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `POST /resume` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `POST /rollback` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /status/{exec_id}` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /plans` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /executions` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `POST /recommended-plan` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `POST /quick-ramp` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /phases` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /current-phase/{exec_id}` (模块: traffic_ramp, 文件: traffic_ramp.py)
- `GET /api-keys` (模块: security, 文件: security.py)
- `POST /api-keys` (模块: security, 文件: security.py)
- `DELETE /api-keys/{key_id}` (模块: security, 文件: security.py)
- `GET /mcp-tools/audit` (模块: security, 文件: security.py)
- `POST /mcp-tools/whitelist` (模块: security, 文件: security.py)
- `GET /mcp-tools/permissions` (模块: security, 文件: security.py)
- `PUT /mcp-tools/permissions` (模块: security, 文件: security.py)
- `GET /risk-assessment` (模块: security, 文件: security.py)
- `GET /mcp-tools/pending-approvals` (模块: security, 文件: security.py)
- `POST /mcp-tools/approve/{request_id}` (模块: security, 文件: security.py)
- `POST /chain` (模块: reasoning, 文件: reasoning.py)
- `GET /chain/{chain_id}` (模块: reasoning, 文件: reasoning.py)
- `POST /chain/{chain_id}/validate` (模块: reasoning, 文件: reasoning.py)
- `POST /chain/{chain_id}/branch` (模块: reasoning, 文件: reasoning.py)
- `POST /chain/{chain_id}/recover` (模块: reasoning, 文件: reasoning.py)
- `DELETE /chain/{chain_id}` (模块: reasoning, 文件: reasoning.py)
- `POST /experiments/{experiment_id}/start` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `POST /experiments/{experiment_id}/stop` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /experiments/{experiment_id}/trials` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /experiments/{experiment_id}/visualizations` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /experiments/{experiment_id}/progress` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /tasks/{task_name}` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `POST /optimize/{task_name}` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `POST /compare-algorithms/{task_name}` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /resource-status` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /active-experiments` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /algorithms` (模块: bandit_recommendations, 文件: bandit_recommendations.py)
- `GET /pruning-strategies` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `GET /parameter-types` (模块: hyperparameter_optimization, 文件: hyperparameter_optimization.py)
- `POST /quality/ground-truth` (模块: emotion_intelligence, 文件: emotion_intelligence.py)
- `GET /status/{content_id}` (模块: multimodal, 文件: multimodal.py)
- `GET /queue/status` (模块: streaming, 文件: streaming.py)
- `POST /analyze/image` (模块: multimodal, 文件: multimodal.py)
- `DELETE /file/{content_id}` (模块: multimodal, 文件: multimodal.py)
- `DELETE /invalidate/{node_name}` (模块: cache, 文件: cache.py)
- `GET /hooks/status` (模块: langgraph_features, 文件: langgraph_features.py)
- `POST /hooks/configure` (模块: langgraph_features, 文件: langgraph_features.py)
- `GET /cache/stats` (模块: personalization, 文件: personalization.py)
- `GET /conflicts` (模块: distributed_task, 文件: distributed_task.py)
- `POST /resolve` (模块: offline, 文件: offline.py)
- `GET /operations` (模块: offline, 文件: offline.py)
- `POST /mode/{mode}` (模块: offline, 文件: offline.py)
- `GET /network` (模块: offline, 文件: offline.py)
- `POST /rules/{rule_id}/conditions` (模块: auto_scaling, 文件: auto_scaling.py)
- `POST /stop/{rule_id}` (模块: auto_scaling, 文件: auto_scaling.py)
- `GET /history/{experiment_id}` (模块: risk_assessment, 文件: risk_assessment.py)
- `GET /recommendations/{experiment_id}` (模块: auto_scaling, 文件: auto_scaling.py)
- `POST /simulate` (模块: auto_scaling, 文件: auto_scaling.py)
- `POST /templates/safe` (模块: auto_scaling, 文件: auto_scaling.py)
- `POST /templates/aggressive` (模块: auto_scaling, 文件: auto_scaling.py)
- `GET /triggers` (模块: auto_scaling, 文件: auto_scaling.py)
- `GET /multi-agent/agents` (模块: mock_endpoints, 文件: mock_endpoints.py)
- `POST /rag/query` (模块: mock_endpoints, 文件: mock_endpoints.py)
- `GET /health/all` (模块: mock_endpoints, 文件: mock_endpoints.py)
- `GET /models/{model_name}` (模块: tensorflow, 文件: tensorflow.py)
- `POST /models/train` (模块: tensorflow, 文件: tensorflow.py)
- `POST /models/predict` (模块: personalization, 文件: personalization.py)
- `POST /models/save` (模块: tensorflow, 文件: tensorflow.py)
- `DELETE /models/{model_name}` (模块: tensorflow, 文件: tensorflow.py)
- `GET /examples/simple-regression` (模块: tensorflow, 文件: tensorflow.py)
- `POST /react/chat/{conversation_id}` (模块: agents, 文件: agents.py)
- `POST /react/task/{conversation_id}` (模块: agents, 文件: agents.py)
- `GET /conversations/{conversation_id}/history` (模块: agents, 文件: agents.py)
- `GET /conversations/{conversation_id}/status` (模块: agents, 文件: agents.py)
- `DELETE /conversations/{conversation_id}` (模块: agents, 文件: agents.py)
- `GET /conversations` (模块: agents, 文件: agents.py)
- `DELETE /components/{component_id}` (模块: platform_integration, 文件: platform_integration.py)
- `GET /components/{component_id}` (模块: platform_integration, 文件: platform_integration.py)
- `GET /workflows/{workflow_id}/status` (模块: platform_integration, 文件: platform_integration.py)
- `POST /workflows/{workflow_id}/cancel` (模块: platform_integration, 文件: platform_integration.py)
- `POST /optimization/run` (模块: platform_integration, 文件: platform_integration.py)
- `GET /optimization/metrics` (模块: platform_integration, 文件: platform_integration.py)
- `POST /optimization/profile/{profile_name}` (模块: platform_integration, 文件: platform_integration.py)
- `POST /documentation/generate` (模块: platform_integration, 文件: platform_integration.py)
- `GET /documentation/status` (模块: platform_integration, 文件: platform_integration.py)
- `POST /documentation/training-materials` (模块: platform_integration, 文件: platform_integration.py)
- `POST /memories` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /memories` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /memories/search` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /events/detect` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /events/{user_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /preferences/learn` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /patterns/triggers` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /storage/optimize` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /preferences/{user_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /patterns/identify` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /patterns/{user_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /patterns/predict` (模块: emotional_memory, 文件: emotional_memory.py)
- `DELETE /memories/{user_id}/{memory_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /memories/export/{user_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `GET /statistics/{user_id}` (模块: emotional_memory, 文件: emotional_memory.py)
- `POST /events/batch` (模块: experiments, 文件: experiments.py)
- `GET /events/{event_id}` (模块: event_tracking, 文件: event_tracking.py)
- `POST /events/{event_id}/reprocess` (模块: event_tracking, 文件: event_tracking.py)
- `GET /models/{model_name}/versions/{version}` (模块: model_service, 文件: model_service.py)
- `DELETE /models/{model_name}/versions/{version}` (模块: model_service, 文件: model_service.py)
- `POST /models/{model_name}/versions/{version}/validate` (模块: model_service, 文件: model_service.py)
- `POST /inference/models/{model_name}/load` (模块: model_service, 文件: model_service.py)
- `DELETE /inference/models/{model_name}/unload` (模块: model_service, 文件: model_service.py)
- `GET /deployment/{deployment_id}` (模块: model_service, 文件: model_service.py)
- `DELETE /deployment/{deployment_id}` (模块: model_service, 文件: model_service.py)
- `POST /learning/{session_id}/feedback` (模块: model_service, 文件: model_service.py)
- `POST /learning/{session_id}/update` (模块: model_service, 文件: model_service.py)
- `GET /learning/{session_id}/stats` (模块: model_service, 文件: model_service.py)
- `GET /abtest/{test_id}/assign` (模块: model_service, 文件: model_service.py)
- `POST /abtest/{test_id}/record` (模块: model_service, 文件: model_service.py)
- `GET /abtest/{test_id}/results` (模块: model_service, 文件: model_service.py)
- `GET /monitoring/metrics/{metric_name}` (模块: model_service, 文件: model_service.py)
- `POST /rules/{rule_id}/test` (模块: acl, 文件: acl.py)
- `POST /{experiment_id}/start` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/pause` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/stop` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/assign` (模块: experiments, 文件: experiments.py)
- `GET /{experiment_id}/results` (模块: experiments, 文件: experiments.py)
- `GET /{experiment_id}/metrics/{metric_name}` (模块: experiments, 文件: experiments.py)
- `GET /{experiment_id}/monitor` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/alerts` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/allocate` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/allocate/batch` (模块: experiments, 文件: experiments.py)
- `GET /{experiment_id}/allocation/distribution` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/allocation/simulate` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/allocation/rules` (模块: experiments, 文件: experiments.py)
- `POST /{experiment_id}/allocation/stages` (模块: experiments, 文件: experiments.py)
- `DELETE /{experiment_id}/allocation/cache` (模块: experiments, 文件: experiments.py)
- `GET /allocation/stats` (模块: experiments, 文件: experiments.py)
- `POST /analyze/text` (模块: emotion_recognition, 文件: emotion_recognition.py)
- `POST /analyze/audio` (模块: emotion_recognition, 文件: emotion_recognition.py)
- `POST /analyze/visual` (模块: emotion_recognition, 文件: emotion_recognition.py)
- `POST /analyze/multimodal` (模块: emotion_recognition, 文件: emotion_recognition.py)
- `POST /collections` (模块: pgvector, 文件: pgvector.py)
- `POST /vectors` (模块: pgvector, 文件: pgvector.py)
- `GET /collections/{collection_name}/stats` (模块: pgvector, 文件: pgvector.py)
- `POST /search/similarity` (模块: pgvector, 文件: pgvector.py)
- `POST /search/hybrid` (模块: pgvector, 文件: pgvector.py)
- `POST /quantization/test` (模块: pgvector, 文件: pgvector.py)
- `GET /quantization/config` (模块: pgvector, 文件: pgvector.py)
- `POST /quantization/configure` (模块: pgvector, 文件: pgvector.py)
- `GET /performance/metrics` (模块: pgvector, 文件: pgvector.py)
- `GET /performance/targets` (模块: pgvector, 文件: pgvector.py)
- `POST /detect` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `POST /check-srm` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `POST /check-data-quality` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `GET /summary/{experiment_id}` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `POST /configure` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `POST /real-time-monitor` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `GET /types` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `GET /methods` (模块: model_compression, 文件: model_compression.py)
- `POST /batch-detect` (模块: anomaly_detection, 文件: anomaly_detection.py)
- `POST /tools/call` (模块: mcp, 文件: mcp.py)
- `GET /tools` (模块: mcp, 文件: mcp.py)
- `POST /tools/filesystem/read` (模块: mcp, 文件: mcp.py)
- `POST /tools/filesystem/write` (模块: mcp, 文件: mcp.py)
- `GET /tools/filesystem/list` (模块: mcp, 文件: mcp.py)
- `POST /tools/database/query` (模块: mcp, 文件: mcp.py)
- `POST /tools/system/command` (模块: mcp, 文件: mcp.py)
- `DELETE /context/{user_id}` (模块: empathy_response, 文件: empathy_response.py)
- `POST /index/file` (模块: rag, 文件: rag.py)
- `POST /index/directory` (模块: rag, 文件: rag.py)
- `POST /index/update` (模块: rag, 文件: rag.py)
- `GET /index/stats` (模块: rag, 文件: rag.py)
- `DELETE /index/reset` (模块: rag, 文件: rag.py)
- `POST /agentic/query` (模块: rag, 文件: rag.py)
- `POST /agentic/query/stream` (模块: rag, 文件: rag.py)
- `GET /agentic/explain` (模块: rag, 文件: rag.py)
- `POST /agentic/feedback` (模块: rag, 文件: rag.py)
- `GET /agentic/stats` (模块: rag, 文件: rag.py)
- `GET /agentic/health` (模块: rag, 文件: rag.py)
- `GET /graphrag/health` (模块: rag, 文件: rag.py)
- `POST /alerts/{alert_id}/acknowledge` (模块: alert_rules, 文件: alert_rules.py)
- `POST /assess` (模块: risk_assessment, 文件: risk_assessment.py)
- `POST /rollback-plan` (模块: risk_assessment, 文件: risk_assessment.py)
- `POST /rollback/execute` (模块: risk_assessment, 文件: risk_assessment.py)
- `GET /rollback/status/{exec_id}` (模块: risk_assessment, 文件: risk_assessment.py)
- `GET /thresholds` (模块: risk_assessment, 文件: risk_assessment.py)
- `PUT /thresholds` (模块: risk_assessment, 文件: risk_assessment.py)
- `GET /risk-levels` (模块: risk_assessment, 文件: risk_assessment.py)
- `GET /categories` (模块: risk_assessment, 文件: risk_assessment.py)
- `POST /basic-stats` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `POST /conversion-stats` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `POST /percentiles` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `POST /multiple-groups-stats` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `GET /mean` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `GET /variance` (模块: statistical_analysis, 文件: statistical_analysis.py)
- `GET /graph/stats` (模块: memory_analytics, 文件: memory_analytics.py)
- `GET /trends` (模块: memory_analytics, 文件: memory_analytics.py)
- `GET /topology` (模块: enterprise, 文件: enterprise.py)
- `GET /agents/{agent_id}` (模块: service_discovery, 文件: service_discovery.py)
- `POST /agents/{agent_id}/start` (模块: cluster_management, 文件: cluster_management.py)
- `POST /agents/{agent_id}/stop` (模块: cluster_management, 文件: cluster_management.py)
- `POST /agents/{agent_id}/restart` (模块: cluster_management, 文件: cluster_management.py)
- `DELETE /agents/{agent_id}` (模块: service_discovery, 文件: service_discovery.py)
- `POST /groups` (模块: cluster_management, 文件: cluster_management.py)
- `GET /groups` (模块: cluster_management, 文件: cluster_management.py)
- `POST /groups/{group_id}/agents/{agent_id}` (模块: cluster_management, 文件: cluster_management.py)
- `POST /metrics/query` (模块: cluster_management, 文件: cluster_management.py)
- `GET /metrics/summary` (模块: cluster_management, 文件: cluster_management.py)
- `GET /metrics/trends/{metric_name}` (模块: cluster_management, 文件: cluster_management.py)
- `POST /scaling/policies` (模块: cluster_management, 文件: cluster_management.py)
- `GET /scaling/recommendations` (模块: cluster_management, 文件: cluster_management.py)
- `POST /scaling/groups/{group_id}/manual` (模块: cluster_management, 文件: cluster_management.py)
- `GET /scaling/history` (模块: cluster_management, 文件: cluster_management.py)
- `POST /agents/batch/start` (模块: cluster_management, 文件: cluster_management.py)
- `POST /agents/batch/stop` (模块: cluster_management, 文件: cluster_management.py)
- `GET /operations/history` (模块: cluster_management, 文件: cluster_management.py)
- `PUT /user/{user_id}/context` (模块: bandit_recommendations, 文件: bandit_recommendations.py)
- `PUT /item/{item_id}/features` (模块: bandit_recommendations, 文件: bandit_recommendations.py)
- `GET /experiments/{experiment_id}/results` (模块: bandit_recommendations, 文件: bandit_recommendations.py)
- `POST /experiments/{experiment_id}/end` (模块: bandit_recommendations, 文件: bandit_recommendations.py)
- `GET /user/{user_id}/profile` (模块: personalization, 文件: personalization.py)
- `PUT /user/{user_id}/profile` (模块: personalization, 文件: personalization.py)
- `GET /features/realtime/{user_id}` (模块: personalization, 文件: personalization.py)
- `POST /features/compute` (模块: personalization, 文件: personalization.py)
- `GET /models/status` (模块: personalization, 文件: personalization.py)
- `PUT /models/update` (模块: personalization, 文件: personalization.py)
- `POST /cache/invalidate/{user_id}` (模块: personalization, 文件: personalization.py)
- `GET /health/check` (模块: workflows, 文件: workflows.py)
- `GET /{workflow_id}` (模块: workflows, 文件: workflows.py)
- `POST /{workflow_id}/start` (模块: workflows, 文件: workflows.py)
- `GET /{workflow_id}/status` (模块: workflows, 文件: workflows.py)
- `PUT /{workflow_id}/control` (模块: workflows, 文件: workflows.py)
- `GET /{workflow_id}/checkpoints` (模块: workflows, 文件: workflows.py)
- `DELETE /{workflow_id}` (模块: workflows, 文件: workflows.py)
- `GET /status/{task_id}` (模块: distributed_task, 文件: distributed_task.py)
- `POST /cancel/{task_id}` (模块: distributed_task, 文件: distributed_task.py)
- `POST /conflicts/resolve/{conflict_id}` (模块: distributed_task, 文件: distributed_task.py)
- `POST /checkpoint/create` (模块: distributed_task, 文件: distributed_task.py)
- `POST /checkpoint/rollback` (模块: distributed_task, 文件: distributed_task.py)
- `POST /shutdown` (模块: distributed_task, 文件: distributed_task.py)
- `POST /broadcast` (模块: emotion_websocket, 文件: emotion_websocket.py)
- `GET /configuration` (模块: enterprise, 文件: enterprise.py)
- `PUT /configuration` (模块: enterprise, 文件: enterprise.py)
- `POST /compliance/assess` (模块: enterprise, 文件: enterprise.py)
- `POST /security/scan` (模块: enterprise, 文件: enterprise.py)
- `GET /audit/logs` (模块: enterprise, 文件: enterprise.py)
- `POST /test/connections` (模块: enterprise, 文件: enterprise.py)
- `GET /anomalies` (模块: analytics, 文件: analytics.py)
- `GET /dashboard/stats` (模块: analytics, 文件: analytics.py)
- `GET /ws/stats` (模块: analytics, 文件: analytics.py)
- `GET /realtime/events` (模块: analytics, 文件: analytics.py)
- `POST /realtime/broadcast` (模块: analytics, 文件: analytics.py)
- `GET /export/events` (模块: analytics, 文件: analytics.py)
- `GET /reports/{report_id}/download` (模块: analytics, 文件: analytics.py)
- `POST /results/{job_id}/download` (模块: model_compression, 文件: model_compression.py)
- `POST /methods/validate` (模块: model_compression, 文件: model_compression.py)
- `POST /strategies/recommend` (模块: model_compression, 文件: model_compression.py)
- `GET /sse/{session_id}` (模块: streaming, 文件: streaming.py)
- `GET /sessions/{session_id}/metrics` (模块: streaming, 文件: streaming.py)
- `GET /backpressure/status` (模块: streaming, 文件: streaming.py)
- `GET /flow-control/metrics` (模块: streaming, 文件: streaming.py)
- `DELETE /sessions/{session_id}` (模块: streaming, 文件: streaming.py)
- `POST /sessions/{session_id}/cleanup` (模块: streaming, 文件: streaming.py)
- `POST /batches/submit` (模块: event_batch, 文件: event_batch.py)
- `GET /jobs/{job_id}/status` (模块: event_batch, 文件: event_batch.py)
- `GET /buffer/metrics` (模块: event_batch, 文件: event_batch.py)
- `POST /buffer/flush` (模块: event_batch, 文件: event_batch.py)
- `POST /config/concurrent-jobs` (模块: event_batch, 文件: event_batch.py)
- `POST /calculate-power` (模块: power_analysis, 文件: power_analysis.py)
- `POST /calculate-sample-size` (模块: power_analysis, 文件: power_analysis.py)
- `POST /calculate-effect-size` (模块: power_analysis, 文件: power_analysis.py)
- `POST /proportion-power` (模块: power_analysis, 文件: power_analysis.py)
- `POST /proportion-sample-size` (模块: power_analysis, 文件: power_analysis.py)
- `POST /ab-test-sample-size` (模块: power_analysis, 文件: power_analysis.py)
- `GET /effect-size-guidelines` (模块: power_analysis, 文件: power_analysis.py)
- `GET /sample-size-calculator` (模块: power_analysis, 文件: power_analysis.py)
- `POST /batch/create` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `DELETE /batch/cleanup` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /exploration_strategies` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /reward_functions` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/inference` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/batch_inference` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /strategies/compare` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/insights` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/performance` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/warmup` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `DELETE /agents/{session_id}/cache` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/hybrid_recommendation` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/adaptive_recommendation` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /strategy_performance` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /optimize_weights` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /combination_modes` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /monitoring/start` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /monitoring/stop` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /monitoring/status` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/summary` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/actions` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /agents/{session_id}/performance_trend` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /monitoring/agents` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `POST /agents/{session_id}/log_event` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /monitoring/statistics` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `DELETE /monitoring/clear/{session_id}` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `DELETE /monitoring/clear_all` (模块: qlearning_tensorflow_backup, 文件: qlearning_tensorflow_backup.py)
- `GET /models/{name}` (模块: model_registry, 文件: model_registry.py)
- `POST /models/{name}/register-from-hub` (模块: model_registry, 文件: model_registry.py)
- `DELETE /models/{name}` (模块: model_registry, 文件: model_registry.py)
- `GET /models/{name}/download` (模块: model_registry, 文件: model_registry.py)
- `GET /models/{name}/export` (模块: model_registry, 文件: model_registry.py)
- `POST /decompose` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `GET /executions/{execution_id}` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `POST /executions/control` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `GET /system/metrics` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `DELETE /executions/{execution_id}` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `GET /executions/{execution_id}/results` (模块: multi_step_reasoning, 文件: multi_step_reasoning.py)
- `PUT /agents/{agent_id}/status` (模块: service_discovery, 文件: service_discovery.py)
- `PUT /agents/{agent_id}/metrics` (模块: service_discovery, 文件: service_discovery.py)
- `POST /load-balancer/select` (模块: service_discovery, 文件: service_discovery.py)
- `POST /authenticate` (模块: distributed_security, 文件: distributed_security.py)
- `POST /authorize` (模块: distributed_security, 文件: distributed_security.py)
- `POST /communication/encrypt` (模块: distributed_security, 文件: distributed_security.py)
- `GET /access-logs` (模块: distributed_security, 文件: distributed_security.py)
- `POST /revoke-access` (模块: distributed_security, 文件: distributed_security.py)

## 按模块统计

| 模块 | 总端点数 | 已使用 | 使用率 |
|------|----------|--------|--------|
| acl | 4 | 3 | 75.0% |
| agent_interface | 2 | 2 | 100.0% |
| agents | 7 | 1 | 14.3% |
| alert_rules | 10 | 4 | 40.0% |
| analytics | 12 | 2 | 16.7% |
| anomaly_detection | 8 | 0 | 0.0% |
| assignment_cache | 8 | 1 | 12.5% |
| auth | 8 | 8 | 100.0% |
| auto_scaling | 8 | 0 | 0.0% |
| bandit_recommendations | 8 | 3 | 37.5% |
| batch | 4 | 0 | 0.0% |
| cache | 3 | 1 | 33.3% |
| cluster_management | 16 | 0 | 0.0% |
| distributed_security | 13 | 7 | 53.8% |
| distributed_task | 9 | 1 | 11.1% |
| documents | 6 | 0 | 0.0% |
| emotion_intelligence | 2 | 1 | 50.0% |
| emotion_modeling | 9 | 2 | 22.2% |
| emotion_recognition | 4 | 0 | 0.0% |
| emotion_websocket | 1 | 0 | 0.0% |
| emotional_intelligence | 12 | 0 | 0.0% |
| emotional_memory | 15 | 0 | 0.0% |
| empathy_response | 4 | 2 | 50.0% |
| enterprise | 11 | 4 | 36.4% |
| event_batch | 8 | 1 | 12.5% |
| event_tracking | 2 | 0 | 0.0% |
| events | 5 | 4 | 80.0% |
| experiments | 22 | 4 | 18.2% |
| explainable_ai | 7 | 0 | 0.0% |
| fault_tolerance | 19 | 1 | 5.3% |
| feedback | 7 | 1 | 14.3% |
| files | 5 | 1 | 20.0% |
| fine_tuning | 10 | 2 | 20.0% |
| graphrag | 11 | 7 | 63.6% |
| health | 2 | 0 | 0.0% |
| hyperparameter_optimization | 16 | 4 | 25.0% |
| knowledge_extraction | 5 | 0 | 0.0% |
| knowledge_graph | 18 | 2 | 11.1% |
| knowledge_graph_reasoning | 4 | 4 | 100.0% |
| langgraph_features | 9 | 7 | 77.8% |
| layered_experiments | 9 | 0 | 0.0% |
| mcp | 7 | 0 | 0.0% |
| memory_analytics | 2 | 0 | 0.0% |
| memory_management | 11 | 1 | 9.1% |
| mock_endpoints | 5 | 2 | 40.0% |
| model_compression | 13 | 5 | 38.5% |
| model_evaluation | 7 | 2 | 28.6% |
| model_registry | 8 | 3 | 37.5% |
| model_service | 28 | 14 | 50.0% |
| multi_step_reasoning | 9 | 2 | 22.2% |
| multimodal | 6 | 1 | 16.7% |
| multimodal_rag | 3 | 0 | 0.0% |
| offline | 5 | 1 | 20.0% |
| personalization | 10 | 1 | 10.0% |
| pgvector | 11 | 1 | 9.1% |
| platform_integration | 15 | 5 | 33.3% |
| power_analysis | 8 | 0 | 0.0% |
| qlearning_tensorflow_backup | 40 | 2 | 5.0% |
| rag | 16 | 4 | 25.0% |
| realtime_metrics | 9 | 0 | 0.0% |
| reasoning | 8 | 2 | 25.0% |
| release_strategy | 8 | 1 | 12.5% |
| report_generation | 5 | 5 | 100.0% |
| risk_assessment | 10 | 1 | 10.0% |
| security | 12 | 2 | 16.7% |
| service_discovery | 9 | 4 | 44.4% |
| social_emotion_api | 8 | 2 | 25.0% |
| social_emotional_understanding | 6 | 0 | 0.0% |
| statistical_analysis | 7 | 1 | 14.3% |
| streaming | 9 | 2 | 22.2% |
| targeting_rules | 5 | 0 | 0.0% |
| tensorflow | 8 | 3 | 37.5% |
| traffic_ramp | 10 | 0 | 0.0% |
| training_data | 26 | 7 | 26.9% |
| unified | 6 | 0 | 0.0% |
| workflows | 9 | 2 | 22.2% |