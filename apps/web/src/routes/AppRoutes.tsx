import { lazy } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { MENU_INDEX, MENU_KEY_SET } from './menuIndex'
import { getMenuLabelText, resolveMenuKey } from './menuConfig'
import { readStoredLastRoute } from './navigationStorage'
import FeatureComingSoon from '../components/FeatureComingSoon'

// 懒加载所有页面组件
const ChatPage = lazy(() => import('../pages/ChatPage'))
const WorkspacePage = lazy(() => import('../pages/WorkspacePage'))
const ConversationHistoryPage = lazy(
  () => import('../pages/ConversationHistoryPage')
)
const MultiAgentPage = lazy(() => import('../pages/MultiAgentPage'))
const TensorFlowQLearningPage = lazy(
  () => import('../pages/TensorFlowQLearningManagementPage')
)
const TestingManagementPage = lazy(
  () => import('../pages/TestingManagementPage')
)
const HypothesisTestingPage = lazy(
  () => import('../pages/HypothesisTestingPage')
)
const EnhancedExperimentAnalysisPage = lazy(
  () => import('../pages/experiments/EnhancedExperimentAnalysisPage')
)
const SupervisorPage = lazy(() => import('../pages/SupervisorPage'))
const RagPage = lazy(() => import('../pages/RagPage'))
const WorkflowPage = lazy(() => import('../pages/WorkflowPage'))
const AsyncAgentPage = lazy(() => import('../pages/AsyncAgentPage'))
const AgenticRagPage = lazy(() => import('../pages/AgenticRagPage'))
const GraphRAGPage = lazy(() => import('../pages/GraphRAGPage'))
const GraphRAGPageEnhanced = lazy(() => import('../pages/GraphRAGPageEnhanced'))
const MultimodalPageComplete = lazy(
  () => import('../pages/MultimodalPageComplete')
)
const FlowControlPage = lazy(() => import('../pages/FlowControlPage'))
const DistributedMessageOverviewPage = lazy(
  () => import('../pages/DistributedMessageOverviewPage')
)
const MCPToolsPage = lazy(() => import('../pages/MCPToolsPage'))
const PgVectorPage = lazy(() => import('../pages/PgVectorPage'))
const CacheMonitorPage = lazy(() => import('../pages/CacheMonitorPage'))
const BatchJobsPageFixed = lazy(() => import('../pages/BatchJobsPageFixed'))
const BatchProcessingPage = lazy(() => import('../pages/BatchProcessingPage'))
const IntelligentSchedulingPage = lazy(
  () => import('../pages/IntelligentSchedulingPage')
)
const HealthMonitorPage = lazy(() => import('../pages/HealthMonitorPage'))
const PerformanceMonitorPage = lazy(
  () => import('../pages/PerformanceMonitorPage')
)
const StreamingMonitorPage = lazy(() => import('../pages/StreamingMonitorPage'))
const MonitoringDashboardPage = lazy(
  () => import('../pages/MonitoringDashboardPage')
)
const EnterpriseArchitecturePage = lazy(
  () => import('../pages/EnterpriseArchitecturePageSimple')
)
const EnterpriseConfigPage = lazy(() => import('../pages/EnterpriseConfigPage'))
const ArchitectureDebugPage = lazy(
  () => import('../pages/ArchitectureDebugPage')
)
const StructuredErrorPage = lazy(() => import('../pages/StructuredErrorPage'))
const TestCoveragePage = lazy(() => import('../pages/TestCoveragePage'))
const IntegrationTestPage = lazy(() => import('../pages/IntegrationTestPage'))
const TestingSuitePage = lazy(() => import('../pages/TestingSuitePage'))
const DocumentProcessingAdvancedPage = lazy(
  () => import('../pages/DocumentProcessingAdvancedPage')
)
const QLearningPerformanceOptimizationPage = lazy(
  () => import('../pages/QLearningPerformanceOptimizationPage')
)
const AuthManagementPage = lazy(() => import('../pages/AuthManagementPage'))
const LangGraphFeaturesPage = lazy(
  () => import('../pages/LangGraphFeaturesPage')
)
const AgentInterfacePage = lazy(() => import('../pages/AgentInterfacePage'))
const NotFoundPage = lazy(() => import('../pages/NotFoundPage'))

// 增强版页面
const MultiAgentEnhancedPage = lazy(
  () => import('../pages/MultiAgentEnhancedPage')
)
const RAGEnhancedPage = lazy(() => import('../pages/RAGEnhancedPage'))
const ExperimentsPlatformPage = lazy(
  () => import('../pages/ExperimentsPlatformPage')
)
const WorkflowManagementPage = lazy(
  () => import('../pages/WorkflowManagementPage')
)
const AgentClusterManagementPage = lazy(
  () => import('../pages/AgentClusterManagementPage')
)
const AgentClusterManagementPageEnhanced = lazy(
  () => import('../pages/AgentClusterManagementPageEnhanced')
)
const MemoryHierarchyPage = lazy(() => import('../pages/MemoryHierarchyPage'))
const MemoryRecallTestPage = lazy(() => import('../pages/MemoryRecallTestPage'))
const MemoryAnalyticsDashboard = lazy(
  () => import('../pages/MemoryAnalyticsDashboard')
)
const ReasoningPage = lazy(() => import('../pages/ReasoningPage'))
const MultiStepReasoningPage = lazy(
  () => import('../pages/MultiStepReasoningPage')
)
const ExplainableAiPage = lazy(() => import('../pages/ExplainableAiPage'))
const TargetingRulesManagementPage = lazy(
  () => import('../pages/TargetingRulesManagementPage')
)
const HybridSearchAdvancedPage = lazy(
  () => import('../pages/HybridSearchAdvancedPage')
)
const FileManagementPageComplete = lazy(
  () => import('../pages/FileManagementPageComplete')
)
const AnomalyDetectionPage = lazy(() => import('../pages/AnomalyDetectionPage'))
const AutoScalingManagementPage = lazy(
  () => import('../pages/AutoScalingManagementPage')
)
const BatchOperationsPage = lazy(() => import('../pages/BatchOperationsPage'))
const AssignmentCacheManagementPage = lazy(
  () => import('../pages/AssignmentCacheManagementPage')
)

// 智能代理服务发现系统页面 (Story 10.1)
const ServiceDiscoveryOverviewPage = lazy(
  () => import('../pages/ServiceDiscoveryOverviewPage')
)
const AgentRegistryManagementPage = lazy(
  () => import('../pages/AgentRegistryManagementPage')
)
const ServiceRoutingManagementPage = lazy(
  () => import('../pages/ServiceRoutingManagementPage')
)
const LoadBalancerConfigPage = lazy(
  () => import('../pages/LoadBalancerConfigPage')
)
const ServiceHealthMonitorPage = lazy(
  () => import('../pages/ServiceHealthMonitorPage')
)
const ServiceClusterManagementPage = lazy(
  () => import('../pages/ServiceClusterManagementPage')
)
const ServicePerformanceDashboardPage = lazy(
  () => import('../pages/ServicePerformanceDashboardPage')
)
const ServiceConfigManagementPage = lazy(
  () => import('../pages/ServiceConfigManagementPage')
)

// 平台集成优化系统页面
const PlatformIntegrationOverviewPage = lazy(
  () => import('../pages/PlatformIntegrationOverviewPage')
)
const ComponentManagementPage = lazy(
  () => import('../pages/ComponentManagementPage')
)
const WorkflowOrchestrationPage = lazy(
  () => import('../pages/WorkflowOrchestrationPage')
)
const PerformanceOptimizationPage = lazy(
  () => import('../pages/PerformanceOptimizationPage')
)
const SystemMonitoringPage = lazy(() => import('../pages/SystemMonitoringPage'))
const DocumentationManagementPage = lazy(
  () => import('../pages/DocumentationManagementPage')
)

// 故障容错和恢复系统页面 (Story 10.5)
const FaultToleranceSystemPage = lazy(
  () => import('../pages/FaultToleranceSystemPage')
)
const RealtimeCommunicationPage = lazy(
  () => import('../pages/RealtimeCommunicationPage')
)
const FaultDetectionPage = lazy(() => import('../pages/FaultDetectionPage'))
const RecoveryManagementPage = lazy(
  () => import('../pages/RecoveryManagementPage')
)
const BackupManagementPage = lazy(() => import('../pages/BackupManagementPage'))
const ConsistencyManagementPage = lazy(
  () => import('../pages/ConsistencyManagementPage')
)
const FaultTestingPage = lazy(() => import('../pages/FaultTestingPage'))

// LoRA/QLoRA微调框架页面
const FineTuningJobsPage = lazy(() => import('../pages/FineTuningJobsPage'))
const FineTuningConfigPage = lazy(() => import('../pages/FineTuningConfigPage'))
const FineTuningMonitorPage = lazy(
  () => import('../pages/FineTuningMonitorPage')
)
const FineTuningModelsPage = lazy(() => import('../pages/FineTuningModelsPage'))
const FineTuningDatasetsPage = lazy(
  () => import('../pages/FineTuningDatasetsPage')
)
const FineTuningCheckpointsPage = lazy(
  () => import('../pages/FineTuningCheckpointsPage')
)
const FineTuningPage = lazy(() => import('../pages/FineTuningPage'))
const FineTuningPageEnhanced = lazy(
  () => import('../pages/FineTuningPageEnhanced')
)
const LoRATrainingPage = lazy(() => import('../pages/LoRATrainingPage'))
const QLoRATrainingPage = lazy(() => import('../pages/QLoRATrainingPage'))
const DistributedTrainingPage = lazy(
  () => import('../pages/DistributedTrainingPage')
)
const RiskAssessmentDashboard = lazy(
  () => import('../pages/RiskAssessmentDashboard')
)
const StatisticalAnalysisDashboard = lazy(
  () => import('../pages/StatisticalAnalysisDashboard')
)
const ModelAdaptersPage = lazy(() => import('../pages/ModelAdaptersPage'))
const TrainingMonitorDashboard = lazy(
  () => import('../pages/TrainingMonitorDashboard')
)
const ModelPerformanceComparison = lazy(
  () => import('../pages/ModelPerformanceComparison')
)

// 分布式任务协调引擎页面 (Story 10.3)
const DistributedTaskCoordinationPage = lazy(
  () => import('../pages/DistributedTaskCoordinationPage')
)
const TaskDecomposerPage = lazy(() => import('../pages/TaskDecomposerPage'))
const IntelligentAssignerPage = lazy(
  () => import('../pages/IntelligentAssignerPage')
)
const RaftConsensusPage = lazy(() => import('../pages/RaftConsensusPage'))
const DistributedStateManagerPage = lazy(
  () => import('../pages/DistributedStateManagerPage')
)
const ConflictResolverPage = lazy(() => import('../pages/ConflictResolverPage'))
const DistributedTaskMonitorPage = lazy(
  () => import('../pages/DistributedTaskMonitorPage')
)
const DistributedTaskSystemStatusPage = lazy(
  () => import('../pages/DistributedTaskSystemStatusPage')
)
const DistributedTaskManagementPageEnhanced = lazy(
  () => import('../pages/DistributedTaskManagementPageEnhanced')
)

// 知识图谱引擎
const KnowledgeExtractionOverviewPage = lazy(
  () => import('../pages/KnowledgeExtractionOverviewPage')
)
const EntityRecognitionPage = lazy(
  () => import('../pages/EntityRecognitionPage')
)
const RelationExtractionPage = lazy(
  () => import('../pages/RelationExtractionPage')
)
const EntityLinkingPage = lazy(() => import('../pages/EntityLinkingPage'))
const MultilingualProcessingPage = lazy(
  () => import('../pages/MultilingualProcessingPage')
)
const KnowledgeGraphVisualizationPage = lazy(
  () => import('../pages/KnowledgeGraphVisualizationPage')
)
const KnowledgeGraphPage = lazy(
  () => import('../pages/KnowledgeGraphManagementPage')
)
const GraphQueryEnginePage = lazy(() => import('../pages/GraphQueryEnginePage'))
const GraphAnalyticsPage = lazy(() => import('../pages/GraphAnalyticsPage'))
const KnowledgeBatchJobsPage = lazy(
  () => import('../pages/KnowledgeBatchJobsPage')
)
const KnowledgeBatchMonitorPage = lazy(
  () => import('../pages/KnowledgeBatchMonitorPage')
)
const KnowledgePerformanceOptimizationPage = lazy(
  () => import('../pages/KnowledgePerformanceOptimizationPage')
)
const KnowledgeCacheManagementPage = lazy(
  () => import('../pages/KnowledgeCacheManagementPage')
)
const KnowledgeValidationPage = lazy(
  () => import('../pages/KnowledgeValidationPage')
)
const KnowledgeConfidenceAnalysisPage = lazy(
  () => import('../pages/KnowledgeConfidenceAnalysisPage')
)
const KnowledgeErrorAnalysisPage = lazy(
  () => import('../pages/KnowledgeErrorAnalysisPage')
)
const KnowledgeModelComparisonPage = lazy(
  () => import('../pages/KnowledgeModelComparisonPage')
)
const ACLProtocolManagementPage = lazy(
  () => import('../pages/ACLProtocolManagementPage')
)

// 动态知识图谱存储系统 (Story 8.2)
const KnowledgeGraphEntityManagement = lazy(
  () => import('../pages/KnowledgeGraphEntityManagement')
)
const KnowledgeGraphRelationManagement = lazy(
  () => import('../pages/KnowledgeGraphRelationManagement')
)
const KnowledgeGraphQueryEngine = lazy(
  () => import('../pages/KnowledgeGraphQueryEngine')
)
const KnowledgeGraphIncrementalUpdate = lazy(
  () => import('../pages/KnowledgeGraphIncrementalUpdate')
)
const KnowledgeGraphQualityAssessment = lazy(
  () => import('../pages/KnowledgeGraphQualityAssessment')
)
const KnowledgeGraphPerformanceMonitor = lazy(
  () => import('../pages/KnowledgeGraphPerformanceMonitorWorkingMinimal')
)
const KnowledgeGraphSchemaManagement = lazy(
  () => import('../pages/KnowledgeGraphSchemaManagement')
)
const KnowledgeGraphDataMigration = lazy(
  () => import('../pages/KnowledgeGraphDataMigration')
)

// 知识管理API接口 (Story 8.6)
// SPARQL查询引擎
const SparqlQueryInterface = lazy(() => import('../pages/SparqlQueryInterface'))
const SparqlOptimization = lazy(() => import('../pages/SparqlOptimization'))
const SparqlPerformance = lazy(() => import('../pages/SparqlPerformance'))
const SparqlCache = lazy(() => import('../pages/SparqlCache'))

// 知识管理REST API
const EntityApiPage = lazy(() => import('../pages/EntityApiPage'))
const RelationApiPage = lazy(() => import('../pages/RelationApiPage'))
const GraphValidationPage = lazy(() => import('../pages/GraphValidationPage'))
const BasicRagManagementPage = lazy(
  () => import('../pages/BasicRagManagementPage')
)
const SupervisorApiManagementPage = lazy(
  () => import('../pages/SupervisorApiManagementPage')
)
const PlatformApiManagementPage = lazy(
  () => import('../pages/PlatformApiManagementPage')
)

// 数据导入导出
const RdfImportExportPage = lazy(() => import('../pages/RdfImportExportPage'))
const CsvExcelImportPage = lazy(() => import('../pages/CsvExcelImportPage'))
const BatchImportJobsPage = lazy(() => import('../pages/BatchImportJobsPage'))
const ExportFormatsPage = lazy(() => import('../pages/ExportFormatsPage'))

// 版本控制系统
const GraphSnapshotsPage = lazy(() => import('../pages/GraphSnapshotsPage'))
const VersionComparisonPage = lazy(
  () => import('../pages/VersionComparisonPage')
)
const RollbackOperationsPage = lazy(
  () => import('../pages/RollbackOperationsPage')
)
const ChangeTrackingPage = lazy(() => import('../pages/ChangeTrackingPage'))

// 认证与安全
const JwtAuthPage = lazy(() => import('../pages/JwtAuthPage'))
const ApiKeyManagementPage = lazy(() => import('../pages/ApiKeyManagementPage'))
const RolePermissionsPage = lazy(() => import('../pages/RolePermissionsPage'))
const SecurityAuditPage = lazy(() => import('../pages/SecurityAuditPage'))
const SecurityPage = lazy(() => import('../pages/SecurityPage'))
const SecurityManagementPage = lazy(
  () => import('../pages/SecurityManagementPage')
)
const SecurityManagementEnhancedPage = lazy(
  () => import('../pages/SecurityManagementEnhancedPage')
)
const DistributedSecurityMonitorPage = lazy(
  () => import('../pages/DistributedSecurityMonitorPage')
)

// 监控与日志
const PerformanceMetricsPage = lazy(
  () => import('../pages/PerformanceMetricsPage')
)
const SystemHealthPage = lazy(() => import('../pages/SystemHealthPage'))
const AlertManagementPage = lazy(() => import('../pages/AlertManagementPage'))
const AuditLogsPage = lazy(() => import('../pages/AuditLogsPage'))

// 知识图推理引擎 (Story 8.3)
// 混合推理引擎
const KGReasoningDashboardPage = lazy(
  () => import('../pages/KGReasoningDashboardPage')
)
const KGReasoningQueryPage = lazy(() => import('../pages/KGReasoningQueryPage'))
const KGReasoningBatchPage = lazy(() => import('../pages/KGReasoningBatchPage'))
const KGReasoningOptimizationPage = lazy(
  () => import('../pages/KGReasoningOptimizationPage')
)
const KGReasoningConfigPage = lazy(
  () => import('../pages/KGReasoningConfigPage')
)
const KGReasoningAnalysisPage = lazy(
  () => import('../pages/KGReasoningAnalysisPage')
)

// 规则推理引擎
const KGRuleManagementPage = lazy(() => import('../pages/KGRuleManagementPage'))
const KGRuleExecutionPage = lazy(() => import('../pages/KGRuleExecutionPage'))
const KGRuleValidationPage = lazy(() => import('../pages/KGRuleValidationPage'))
const KGRuleConflictPage = lazy(() => import('../pages/KGRuleConflictPage'))

// 嵌入推理引擎
const KGEmbeddingModelsPage = lazy(
  () => import('../pages/KGEmbeddingModelsPage')
)
const KGEmbeddingTrainingPage = lazy(
  () => import('../pages/KGEmbeddingTrainingPage')
)
const KGEmbeddingSimilarityPage = lazy(
  () => import('../pages/KGEmbeddingSimilarityPage')
)
const KGEmbeddingIndexPage = lazy(() => import('../pages/KGEmbeddingIndexPage'))

// 路径推理引擎
const KGPathDiscoveryPage = lazy(() => import('../pages/KGPathDiscoveryPage'))
const KGPathAnalysisPage = lazy(() => import('../pages/KGPathAnalysisPage'))
const KGPathOptimizationPage = lazy(
  () => import('../pages/KGPathOptimizationPage')
)
const KGPathConfidencePage = lazy(() => import('../pages/KGPathConfidencePage'))

// 不确定性推理
const KGUncertaintyAnalysisPage = lazy(
  () => import('../pages/KGUncertaintyAnalysisPage')
)
const KGBayesianNetworkPage = lazy(
  () => import('../pages/KGBayesianNetworkPage')
)
const KGProbabilityCalculationPage = lazy(
  () => import('../pages/KGProbabilityCalculationPage')
)
const KGConfidenceIntervalPage = lazy(
  () => import('../pages/KGConfidenceIntervalPage')
)

// 用户反馈学习系统
const FeedbackSystemPage = lazy(() => import('../pages/FeedbackSystemPage'))
const FeedbackAnalyticsPage = lazy(
  () => import('../pages/FeedbackAnalyticsPage')
)
const UserFeedbackProfilesPage = lazy(
  () => import('../pages/UserFeedbackProfilesPage')
)
const ItemFeedbackAnalysisPage = lazy(
  () => import('../pages/ItemFeedbackAnalysisPage')
)
const FeedbackQualityMonitorPage = lazy(
  () => import('../pages/FeedbackQualityMonitorPage')
)

// Story 4.11 离线能力与同步机制
const OfflineCapabilityPage = lazy(
  () => import('../pages/OfflineCapabilityPage')
)
const SyncManagementPage = lazy(() => import('../pages/SyncManagementPage'))
const ConflictResolutionPage = lazy(
  () => import('../pages/ConflictResolutionPage')
)
const VectorClockVisualizationPage = lazy(
  () => import('../pages/VectorClockVisualizationPage')
)
const NetworkMonitorDetailPage = lazy(
  () => import('../pages/NetworkMonitorDetailPage')
)
const SyncEngineInternalPage = lazy(
  () => import('../pages/SyncEngineInternalPage')
)
const ModelCacheMonitorPage = lazy(
  () => import('../pages/ModelCacheMonitorPage')
)
const DagOrchestratorPage = lazy(() => import('../pages/DagOrchestratorPage'))
const UnifiedMonitorPage = lazy(() => import('../pages/UnifiedMonitorPage'))
const AiTrismPage = lazy(() => import('../pages/AiTrismPage'))
const EventDashboardPage = lazy(() => import('../pages/EventDashboardPage'))
const BanditRecommendationPage = lazy(
  () => import('../pages/BanditRecommendationPage')
)
const QLearningPage = lazy(() => import('../pages/QLearningPage'))
const QLearningTrainingPage = lazy(
  () => import('../pages/QLearningTrainingPage')
)
const QLearningStrategyPage = lazy(
  () => import('../pages/QLearningStrategyPage')
)
const QLearningRecommendationPage = lazy(
  () => import('../pages/QLearningRecommendationPage')
)
const QLearningPerformancePage = lazy(
  () => import('../pages/QLearningPerformancePage')
)
const TabularQLearningPage = lazy(
  () => import('../pages/qlearning/TabularQLearningPage')
)
const DQNPage = lazy(() => import('../pages/qlearning/DQNPage'))
const DQNVariantsPage = lazy(() => import('../pages/qlearning/DQNVariantsPage'))
const ExplorationStrategiesPage = lazy(
  () => import('../pages/qlearning/ExplorationStrategiesPage')
)
const UCBStrategiesPage = lazy(
  () => import('../pages/qlearning/UCBStrategiesPage')
)
const ThompsonSamplingPage = lazy(
  () => import('../pages/qlearning/ThompsonSamplingPage')
)
const BasicRewardsPage = lazy(
  () => import('../pages/qlearning/BasicRewardsPage')
)
const CompositeRewardsPage = lazy(
  () => import('../pages/qlearning/CompositeRewardsPage')
)
const AdaptiveRewardsPage = lazy(
  () => import('../pages/qlearning/AdaptiveRewardsPage')
)
const RewardShapingPage = lazy(
  () => import('../pages/qlearning/RewardShapingPage')
)
const StateSpacePage = lazy(() => import('../pages/qlearning/StateSpacePage'))
const ActionSpacePage = lazy(() => import('../pages/qlearning/ActionSpacePage'))
const GridWorldPage = lazy(() => import('../pages/qlearning/GridWorldPage'))
const EnvironmentSimulatorPage = lazy(
  () => import('../pages/qlearning/EnvironmentSimulatorPage')
)
const TrainingManagerPage = lazy(
  () => import('../pages/qlearning/TrainingManagerPage')
)
const LearningRateSchedulerPage = lazy(
  () => import('../pages/qlearning/LearningRateSchedulerPage')
)
const EarlyStoppingPage = lazy(
  () => import('../pages/qlearning/EarlyStoppingPage')
)
const PerformanceTrackerPage = lazy(
  () => import('../pages/qlearning/PerformanceTrackerPage')
)

// TensorFlow模型管理页面
const TensorFlowManagementPage = lazy(
  () => import('../pages/TensorFlowManagementPage')
)

// 新增缺失页面（去重后）
const ConflictResolutionLearningPage = lazy(
  () => import('../pages/ConflictResolutionLearningPage')
)
const SyncEngineLearningPage = lazy(
  () => import('../pages/SyncEngineLearningPage')
)
const HealthComprehensivePage = lazy(
  () => import('../pages/HealthComprehensivePage')
)
// 已移除不存在的MultimodalPageSimple
const VectorAdvancedPage = lazy(() => import('../pages/VectorAdvancedPage'))
// 已移除不存在的VectorAdvancedPageSimple
// 已移除不存在的VectorAdvancedTestPage
const VectorClockAlgorithmPage = lazy(
  () => import('../pages/VectorClockAlgorithmPage')
)
const UnifiedEnginePageComplete = lazy(
  () => import('../pages/UnifiedEnginePageComplete')
)
// 已移除不存在的BatchJobsPage，使用BatchJobsPageFixed
const DocumentProcessingPage = lazy(
  () => import('../pages/DocumentProcessingPage')
)
// 已移除不存在的MultimodalPage，使用MultimodalPageComplete
const FileManagementAdvancedPage = lazy(
  () => import('../pages/FileManagementAdvancedPage')
)
const DistributedEventsPage = lazy(
  () => import('../pages/DistributedEventsPage')
)
const LangGraph065Page = lazy(() => import('../pages/LangGraph065Page'))
const MultimodalRagPage = lazy(() => import('../pages/MultimodalRagPage'))
const MultimodalRagManagementPage = lazy(
  () => import('../pages/MultimodalRagManagementPage')
)
const DocumentManagementPageComplete = lazy(
  () => import('../pages/DocumentManagementPageComplete')
)
const RealtimeMetricsManagementPage = lazy(
  () => import('../pages/RealtimeMetricsManagementPage')
)

// 个性化引擎页面
const PersonalizationEnginePage = lazy(
  () => import('../pages/PersonalizationEnginePage')
)
const PersonalizationMonitorPage = lazy(
  () => import('../pages/PersonalizationMonitorPage')
)
const PersonalizationFeaturePage = lazy(
  () => import('../pages/PersonalizationFeaturePage')
)
const PersonalizationLearningPage = lazy(
  () => import('../pages/PersonalizationLearningPage')
)
const PersonalizationApiPage = lazy(
  () => import('../pages/PersonalizationApiPage')
)
const PersonalizationAlertsPage = lazy(
  () => import('../pages/PersonalizationAlertsPage')
)
const PersonalizationProductionPage = lazy(
  () => import('../pages/PersonalizationProductionPage')
)
const PersonalizationWebSocketPage = lazy(
  () => import('../pages/PersonalizationWebSocketPage')
)
const WebSocketManagementPage = lazy(
  () => import('../pages/WebSocketManagementPage')
)

// 高级情感智能系统页面 (Story 11.1 & 11.2)
const EmotionRecognitionOverviewPage = lazy(
  () => import('../pages/EmotionRecognitionOverviewPage')
)
const TextEmotionAnalysisPage = lazy(
  () => import('../pages/TextEmotionAnalysisPage')
)
const AudioEmotionRecognitionPage = lazy(
  () => import('../pages/AudioEmotionRecognitionPage')
)
const VisualEmotionAnalysisPage = lazy(
  () => import('../pages/VisualEmotionAnalysisPage')
)
const MultiModalEmotionFusionPage = lazy(
  () => import('../pages/MultiModalEmotionFusionPage')
)
const EmotionModelingPage = lazy(() => import('../pages/EmotionModelingPage'))

// 情感记忆管理系统 (Story 11.4)
const EmotionalMemoryPage = lazy(() => import('../pages/EmotionalMemoryPage'))
const EmotionalMemoryManagementPage = lazy(
  () => import('../pages/EmotionalMemoryManagementPage')
)
const EmotionalEventAnalysisPage = lazy(
  () => import('../pages/EmotionalEventAnalysisPage')
)

// 情感智能决策引擎 (Story 11.5)
const EmotionalIntelligenceDecisionEnginePage = lazy(
  () => import('../pages/EmotionalIntelligenceDecisionEnginePage')
)
const EmotionalRiskAssessmentDashboardPage = lazy(
  () => import('../pages/EmotionalRiskAssessmentDashboardPage')
)
const CrisisDetectionSupportPage = lazy(
  () => import('../pages/CrisisDetectionSupportPage')
)
const InterventionStrategyManagementPage = lazy(
  () => import('../pages/InterventionStrategyManagementPage')
)
const EmotionalHealthMonitoringDashboardPage = lazy(
  () => import('../pages/EmotionalHealthMonitoringDashboardPage')
)
const DecisionHistoryAnalysisPage = lazy(
  () => import('../pages/DecisionHistoryAnalysisPage')
)
const EmpathyResponseGeneratorPage = lazy(
  () => import('../pages/EmpathyResponseGeneratorPage')
)

// 社交情感理解系统 (Story 11.6)
const GroupEmotionAnalysisPage = lazy(
  () => import('../pages/GroupEmotionAnalysisPage')
)
const RelationshipDynamicsPage = lazy(
  () => import('../pages/RelationshipDynamicsPage')
)
const SocialContextAdaptationPage = lazy(
  () => import('../pages/SocialContextAdaptationPage')
)
const SocialEmotionalUnderstandingPage = lazy(
  () => import('../pages/SocialEmotionalUnderstandingPage')
)
const CulturalAdaptationPage = lazy(
  () => import('../pages/CulturalAdaptationPage')
)
const SocialIntelligenceDecisionPage = lazy(
  () => import('../pages/SocialIntelligenceDecisionPage')
)

// A/B测试实验平台页面
const ExperimentListPage = lazy(
  () => import('../pages/experiments/ExperimentListPage')
)

// 新增未使用API模块页面
const ServiceDiscoveryManagementPage = lazy(
  () => import('../pages/ServiceDiscoveryManagementPage')
)
const OfflineManagementPage = lazy(
  () => import('../pages/OfflineManagementPage')
)
const TrafficRampManagementPage = lazy(
  () => import('../pages/TrafficRampManagementPage')
)
const LayeredExperimentsManagementPage = lazy(
  () => import('../pages/LayeredExperimentsManagementPage')
)
const PowerAnalysisPage = lazy(() => import('../pages/PowerAnalysisPage'))
const DescriptiveStatisticsPage = lazy(
  () => import('../pages/DescriptiveStatisticsPage')
)
const MultipleTestingCorrectionPage = lazy(
  () => import('../pages/MultipleTestingCorrectionPage')
)
const ExperimentDashboardPage = lazy(
  () => import('../pages/experiments/ExperimentDashboardPage')
)
const StatisticalAnalysisPage = lazy(
  () => import('../pages/experiments/StatisticalAnalysisPage')
)
const TrafficAllocationPage = lazy(
  () => import('../pages/experiments/TrafficAllocationPage')
)
const EventTrackingPage = lazy(
  () => import('../pages/experiments/EventTrackingPage')
)
const ReleaseStrategyPage = lazy(
  () => import('../pages/experiments/ReleaseStrategyPage')
)
const MonitoringAlertsPage = lazy(
  () => import('../pages/experiments/MonitoringAlertsPage')
)
const AdvancedAlgorithmsPage = lazy(
  () => import('../pages/experiments/AdvancedAlgorithmsPage')
)

// 行为分析系统
const BehaviorAnalyticsPage = lazy(
  () => import('../pages/BehaviorAnalyticsPage')
)
const BehaviorAnalyticsPageEnhanced = lazy(
  () => import('../pages/BehaviorAnalyticsPageEnhanced')
)
const EventDataManagePage = lazy(
  () => import('../pages/behavior-analytics/EventDataManagePage')
)
const SessionManagePage = lazy(
  () => import('../pages/behavior-analytics/SessionManagePage')
)
const ReportCenterPage = lazy(
  () => import('../pages/behavior-analytics/ReportCenterPage')
)
const RealTimeMonitorPage = lazy(
  () => import('../pages/behavior-analytics/RealTimeMonitorPage')
)
const DataExportPage = lazy(
  () => import('../pages/behavior-analytics/DataExportPage')
)
const SystemConfigPage = lazy(
  () => import('../pages/behavior-analytics/SystemConfigPage')
)

// 强化学习系统监控页面
const RLSystemDashboardPage = lazy(
  () => import('../pages/RLSystemDashboardPage')
)
const RLPerformanceMonitorPage = lazy(
  () => import('../pages/RLPerformanceMonitorPage')
)
const RLIntegrationTestPage = lazy(
  () => import('../pages/RLIntegrationTestPage')
)
const RLAlertConfigPage = lazy(() => import('../pages/RLAlertConfigPage'))
const RLMetricsAnalysisPage = lazy(
  () => import('../pages/RLMetricsAnalysisPage')
)
const RLSystemHealthPage = lazy(() => import('../pages/RLSystemHealthPage'))

// 模型评估和基准测试系统 (Story 9.4)
const ModelEvaluationOverviewPage = lazy(
  () => import('../pages/ModelEvaluationOverviewPage')
)
const ModelEvaluationManagementPage = lazy(
  () => import('../pages/ModelEvaluationManagementPage')
)
const MemoryManagementMonitorPage = lazy(
  () => import('../pages/MemoryManagementMonitorPage')
)
const EvaluationEngineManagementPage = lazy(
  () => import('../pages/EvaluationEngineManagementPage')
)
const EvaluationTasksMonitorPage = lazy(
  () => import('../pages/EvaluationTasksMonitorPage')
)
const EvaluationReportsCenterPage = lazy(
  () => import('../pages/EvaluationReportsCenterPage')
)
const EvaluationApiManagementPage = lazy(
  () => import('../pages/EvaluationApiManagementPage')
)
const ModelComparisonDashboardPage = lazy(
  () => import('../pages/ModelComparisonDashboardPage')
)
const BenchmarkSuiteManagementPage = lazy(
  () => import('../pages/BenchmarkSuiteManagementPage')
)
const BenchmarkGlueManagementPage = lazy(
  () => import('../pages/BenchmarkGlueManagementPage')
)
const BenchmarkSupergluePage = lazy(
  () => import('../pages/BenchmarkSupergluePage')
)
const BenchmarkMmluPage = lazy(() => import('../pages/BenchmarkMmluPage'))
const EvaluationMetricsConfigPage = lazy(
  () => import('../pages/EvaluationMetricsConfigPage')
)
// 模型压缩和量化工具 (Story 9.2)
const ModelCompressionOverviewPage = lazy(
  () => import('../pages/ModelCompressionOverviewPage')
)
const QuantizationManagerPage = lazy(
  () => import('../pages/QuantizationManagerPage')
)
const KnowledgeDistillationPage = lazy(
  () => import('../pages/KnowledgeDistillationPage')
)
const ModelPruningPage = lazy(() => import('../pages/ModelPruningPage'))
const CompressionPipelinePage = lazy(
  () => import('../pages/CompressionPipelinePage')
)
const ModelCompressionEvaluationPage = lazy(
  () => import('../pages/ModelCompressionEvaluationPage')
)
const ModelPerformanceBenchmarkPage = lazy(
  () => import('../pages/ModelPerformanceBenchmarkPage')
)
const CompressionStrategyRecommendationPage = lazy(
  () => import('../pages/CompressionStrategyRecommendationPage')
)

// 模型服务部署平台 (Story 9.6)
const ModelRegistryPage = lazy(() => import('../pages/ModelRegistryPage'))
const ModelInferencePage = lazy(() => import('../pages/ModelInferencePage'))
const ModelDeploymentPage = lazy(() => import('../pages/ModelDeploymentPage'))
const ModelMonitoringPage = lazy(() => import('../pages/ModelMonitoringPage'))
const ModelServiceManagementPage = lazy(
  () => import('../pages/ModelServiceManagementPage')
)
const OnlineLearningPage = lazy(() => import('../pages/OnlineLearningPage'))

// 自动化超参数优化系统 (Story 9.3)
const HyperparameterOptimizationPage = lazy(
  () => import('../pages/HyperparameterOptimizationPage')
)
const HyperparameterOptimizationPageEnhanced = lazy(
  () => import('../pages/HyperparameterOptimizationPageEnhanced')
)
const HyperparameterExperimentsPage = lazy(
  () => import('../pages/HyperparameterExperimentsPage')
)
const HyperparameterAlgorithmsPage = lazy(
  () => import('../pages/HyperparameterAlgorithmsPage')
)
const HyperparameterVisualizationsPage = lazy(
  () => import('../pages/HyperparameterVisualizationsPage')
)
const HyperparameterMonitoringPage = lazy(
  () => import('../pages/HyperparameterMonitoringPage')
)
const HyperparameterResourcesPage = lazy(
  () => import('../pages/HyperparameterResourcesPage')
)
const HyperparameterSchedulerPage = lazy(
  () => import('../pages/HyperparameterSchedulerPage')
)
const HyperparameterReportsPage = lazy(
  () => import('../pages/HyperparameterReportsPage')
)

// 训练数据管理系统 (Story 9.5)
const TrainingDataManagementPage = lazy(
  () => import('../pages/TrainingDataManagementPage')
)
const TrainingDataManagementPageEnhanced = lazy(
  () => import('../pages/TrainingDataManagementPageEnhanced')
)
const DataSourceManagementPage = lazy(
  () => import('../pages/DataSourceManagementPage')
)
const DataCollectionPage = lazy(() => import('../pages/DataCollectionPage'))
const DataPreprocessingPage = lazy(
  () => import('../pages/DataPreprocessingPage')
)
const DataAnnotationManagementPage = lazy(
  () => import('../pages/DataAnnotationManagementPage')
)
const AnnotationTasksPage = lazy(() => import('../pages/AnnotationTasksPage'))
const AnnotationQualityControlPage = lazy(
  () => import('../pages/AnnotationQualityControlPage')
)
const DataVersionManagementPage = lazy(
  () => import('../pages/DataVersionManagementPage')
)

const MissingRouteFallback = () => {
  const location = useLocation()
  const resolvedKey = resolveMenuKey(location.pathname)
  if (MENU_KEY_SET.has(resolvedKey)) {
    const item = MENU_INDEX.itemByKey.get(resolvedKey)
    const label = item ? getMenuLabelText(item.label) : resolvedKey
    return (
      <FeatureComingSoon
        title={label || '功能正在建设'}
        description="该功能已在导航中规划，正在建设中。"
      />
    )
  }
  return <NotFoundPage />
}


const AppRoutes = () => {
  const defaultPath = readStoredLastRoute(MENU_KEY_SET) ?? '/workspace'
  return (
    <Routes>
    {/* 🤖 智能体系统 */}
    <Route path="/" element={<Navigate to={defaultPath} replace />} />
    <Route path="/workspace" element={<WorkspacePage />} />
    <Route path="/chat" element={<ChatPage />} />
    <Route path="/history" element={<ConversationHistoryPage />} />
    <Route path="/multi-agent" element={<MultiAgentPage />} />
    <Route
      path="/tensorflow-qlearning"
      element={<TensorFlowQLearningPage />}
    />
    <Route
      path="/testing"
      element={<Navigate to="/testing-management" replace />}
    />
    <Route
      path="/testing-management"
      element={<TestingManagementPage />}
    />
    <Route
      path="/hypothesis-testing"
      element={<HypothesisTestingPage />}
    />
    <Route path="/supervisor" element={<SupervisorPage />} />
    <Route path="/async-agents" element={<AsyncAgentPage />} />
    <Route path="/agent-interface" element={<AgentInterfacePage />} />
    <Route
      path="/agent-cluster-management"
      element={<AgentClusterManagementPage />}
    />
    <Route
      path="/agent-cluster-management-enhanced"
      element={<AgentClusterManagementPageEnhanced />}
    />

    {/* 增强版页面 */}
    <Route
      path="/multi-agent-enhanced"
      element={<MultiAgentEnhancedPage />}
    />
    <Route path="/rag-enhanced" element={<RAGEnhancedPage />} />
    <Route
      path="/experiments-platform"
      element={<ExperimentsPlatformPage />}
    />
    <Route
      path="/workflow-management"
      element={<WorkflowManagementPage />}
    />

    {/* 🔍 智能检索引擎 */}
    <Route path="/rag" element={<RagPage />} />
    <Route path="/agentic-rag" element={<AgenticRagPage />} />
    <Route path="/graphrag" element={<GraphRAGPage />} />
    <Route
      path="/graphrag-enhanced"
      element={<GraphRAGPageEnhanced />}
    />
    <Route
      path="/hybrid-search"
      element={<HybridSearchAdvancedPage />}
    />

    {/* 🧠 推理引擎 */}
    <Route path="/reasoning" element={<ReasoningPage />} />
    <Route
      path="/multi-step-reasoning"
      element={<MultiStepReasoningPage />}
    />
    <Route path="/explainable-ai" element={<ExplainableAiPage />} />
    <Route
      path="/targeting-rules"
      element={<TargetingRulesManagementPage />}
    />

    {/* 🗺️ 知识图谱引擎 */}
    <Route
      path="/knowledge-extraction-overview"
      element={<KnowledgeExtractionOverviewPage />}
    />
    <Route
      path="/entity-recognition"
      element={<EntityRecognitionPage />}
    />
    <Route
      path="/relation-extraction"
      element={<RelationExtractionPage />}
    />
    <Route path="/entity-linking" element={<EntityLinkingPage />} />
    <Route
      path="/multilingual-processing"
      element={<MultilingualProcessingPage />}
    />
    <Route
      path="/knowledge-graph-visualization"
      element={<KnowledgeGraphVisualizationPage />}
    />
    <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
    <Route
      path="/knowledge-graph-query"
      element={<GraphQueryEnginePage />}
    />
    <Route
      path="/knowledge-graph-analytics"
      element={<GraphAnalyticsPage />}
    />
    <Route
      path="/knowledge-batch-jobs"
      element={<KnowledgeBatchJobsPage />}
    />
    <Route
      path="/knowledge-batch-monitor"
      element={<KnowledgeBatchMonitorPage />}
    />
    <Route
      path="/knowledge-performance-optimization"
      element={<KnowledgePerformanceOptimizationPage />}
    />
    <Route
      path="/knowledge-cache-management"
      element={<KnowledgeCacheManagementPage />}
    />
    <Route
      path="/knowledge-validation"
      element={<KnowledgeValidationPage />}
    />
    <Route
      path="/knowledge-confidence-analysis"
      element={<KnowledgeConfidenceAnalysisPage />}
    />
    <Route
      path="/knowledge-error-analysis"
      element={<KnowledgeErrorAnalysisPage />}
    />
    <Route
      path="/knowledge-model-comparison"
      element={<KnowledgeModelComparisonPage />}
    />
    <Route
      path="/acl-protocol-management"
      element={<ACLProtocolManagementPage />}
    />

    {/* 🗺️ 动态知识图谱存储系统 (Story 8.2) */}
    <Route
      path="/kg-entity-management"
      element={<KnowledgeGraphEntityManagement />}
    />
    <Route
      path="/kg-relation-management"
      element={<KnowledgeGraphRelationManagement />}
    />
    <Route
      path="/kg-graph-query"
      element={<KnowledgeGraphQueryEngine />}
    />
    <Route
      path="/kg-incremental-update"
      element={<KnowledgeGraphIncrementalUpdate />}
    />
    <Route
      path="/kg-quality-assessment"
      element={<KnowledgeGraphQualityAssessment />}
    />
    <Route
      path="/kg-performance-monitor"
      element={<KnowledgeGraphPerformanceMonitor />}
    />
    <Route
      path="/kg-schema-management"
      element={<KnowledgeGraphSchemaManagement />}
    />
    <Route
      path="/kg-data-migration"
      element={<KnowledgeGraphDataMigration />}
    />

    {/* 📊 知识管理API接口 (Story 8.6) */}
    {/* SPARQL查询引擎 */}
    <Route
      path="/sparql-query-interface"
      element={<SparqlQueryInterface />}
    />
    <Route
      path="/sparql-optimization"
      element={<SparqlOptimization />}
    />
    <Route
      path="/sparql-performance"
      element={<SparqlPerformance />}
    />
    <Route path="/sparql-cache" element={<SparqlCache />} />

    {/* 知识管理REST API */}
    <Route path="/entity-api" element={<EntityApiPage />} />
    <Route path="/relation-api" element={<RelationApiPage />} />
    <Route
      path="/batch-operations"
      element={<BatchOperationsPage />}
    />
    <Route
      path="/graph-validation"
      element={<GraphValidationPage />}
    />
    <Route
      path="/basic-rag-management"
      element={<BasicRagManagementPage />}
    />
    <Route
      path="/supervisor-api-management"
      element={<SupervisorApiManagementPage />}
    />
    <Route
      path="/platform-api-management"
      element={<PlatformApiManagementPage />}
    />

    {/* 数据导入导出 */}
    <Route
      path="/rdf-import-export"
      element={<RdfImportExportPage />}
    />
    <Route
      path="/csv-excel-import"
      element={<CsvExcelImportPage />}
    />
    <Route
      path="/batch-import-jobs"
      element={<BatchImportJobsPage />}
    />
    <Route path="/export-formats" element={<ExportFormatsPage />} />

    {/* 版本控制系统 */}
    <Route path="/graph-snapshots" element={<GraphSnapshotsPage />} />
    <Route
      path="/version-comparison"
      element={<VersionComparisonPage />}
    />
    <Route
      path="/rollback-operations"
      element={<RollbackOperationsPage />}
    />
    <Route path="/change-tracking" element={<ChangeTrackingPage />} />

    {/* 认证与安全 */}
    <Route path="/jwt-auth" element={<JwtAuthPage />} />
    <Route
      path="/api-key-management"
      element={<ApiKeyManagementPage />}
    />
    <Route
      path="/role-permissions"
      element={<RolePermissionsPage />}
    />
    <Route path="/security-audit" element={<SecurityAuditPage />} />

    {/* 监控与日志 */}
    <Route
      path="/performance-metrics"
      element={<PerformanceMetricsPage />}
    />
    <Route path="/system-health" element={<SystemHealthPage />} />
    <Route
      path="/alert-management"
      element={<AlertManagementPage />}
    />
    <Route path="/audit-logs" element={<AuditLogsPage />} />

    {/* 🧠 知识图推理引擎 (Story 8.3) */}
    {/* 混合推理引擎 */}
    <Route
      path="/kg-reasoning-dashboard"
      element={<KGReasoningDashboardPage />}
    />
    <Route
      path="/kg-reasoning-query"
      element={<KGReasoningQueryPage />}
    />
    <Route
      path="/kg-reasoning-batch"
      element={<KGReasoningBatchPage />}
    />
    <Route
      path="/kg-reasoning-performance"
      element={<KGReasoningOptimizationPage />}
    />
    <Route
      path="/kg-reasoning-strategy"
      element={<KGReasoningConfigPage />}
    />
    <Route
      path="/kg-reasoning-explanation"
      element={<KGReasoningAnalysisPage />}
    />

    {/* 规则推理引擎 */}
    <Route
      path="/kg-rule-management"
      element={<KGRuleManagementPage />}
    />
    <Route
      path="/kg-rule-execution"
      element={<KGRuleExecutionPage />}
    />
    <Route
      path="/kg-rule-validation"
      element={<KGRuleValidationPage />}
    />
    <Route
      path="/kg-rule-conflict"
      element={<KGRuleConflictPage />}
    />

    {/* 嵌入推理引擎 */}
    <Route
      path="/kg-embedding-models"
      element={<KGEmbeddingModelsPage />}
    />
    <Route
      path="/kg-embedding-training"
      element={<KGEmbeddingTrainingPage />}
    />
    <Route
      path="/kg-embedding-similarity"
      element={<KGEmbeddingSimilarityPage />}
    />
    <Route
      path="/kg-embedding-index"
      element={<KGEmbeddingIndexPage />}
    />

    {/* 路径推理引擎 */}
    <Route
      path="/kg-path-discovery"
      element={<KGPathDiscoveryPage />}
    />
    <Route
      path="/kg-path-analysis"
      element={<KGPathAnalysisPage />}
    />
    <Route
      path="/kg-path-optimization"
      element={<KGPathOptimizationPage />}
    />
    <Route
      path="/kg-path-confidence"
      element={<KGPathConfidencePage />}
    />

    {/* 不确定性推理 */}
    <Route
      path="/kg-uncertainty-analysis"
      element={<KGUncertaintyAnalysisPage />}
    />
    <Route
      path="/kg-bayesian-network"
      element={<KGBayesianNetworkPage />}
    />
    <Route
      path="/kg-probability-calculation"
      element={<KGProbabilityCalculationPage />}
    />
    <Route
      path="/kg-confidence-interval"
      element={<KGConfidenceIntervalPage />}
    />

    {/* 🎯 推荐算法引擎 */}
    <Route
      path="/bandit-recommendation"
      element={<BanditRecommendationPage />}
    />

    {/* 🚀 个性化引擎 */}
    <Route
      path="/personalization-engine"
      element={<PersonalizationEnginePage />}
    />
    <Route
      path="/personalization-monitor"
      element={<PersonalizationMonitorPage />}
    />
    <Route
      path="/personalization-features"
      element={<PersonalizationFeaturePage />}
    />
    <Route
      path="/personalization-learning"
      element={<PersonalizationLearningPage />}
    />
    <Route
      path="/personalization-api"
      element={<PersonalizationApiPage />}
    />
    <Route
      path="/personalization-alerts"
      element={<PersonalizationAlertsPage />}
    />
    <Route
      path="/personalization-production"
      element={<PersonalizationProductionPage />}
    />
    <Route
      path="/personalization-websocket"
      element={<PersonalizationWebSocketPage />}
    />
    <Route
      path="/websocket-management"
      element={<WebSocketManagementPage />}
    />

    {/* 😊 高级情感智能系统 */}
    <Route
      path="/emotion-recognition-overview"
      element={<EmotionRecognitionOverviewPage />}
    />
    <Route
      path="/text-emotion-analysis"
      element={<TextEmotionAnalysisPage />}
    />
    <Route
      path="/audio-emotion-recognition"
      element={<AudioEmotionRecognitionPage />}
    />
    <Route
      path="/visual-emotion-analysis"
      element={<VisualEmotionAnalysisPage />}
    />
    <Route
      path="/multimodal-emotion-fusion"
      element={<MultiModalEmotionFusionPage />}
    />
    <Route
      path="/emotion-modeling"
      element={<EmotionModelingPage />}
    />

    {/* 社交情感理解系统 (Story 11.6) */}
    <Route
      path="/group-emotion-analysis"
      element={<GroupEmotionAnalysisPage />}
    />
    <Route
      path="/relationship-dynamics"
      element={<RelationshipDynamicsPage />}
    />
    <Route
      path="/social-context-adaptation"
      element={<SocialContextAdaptationPage />}
    />
    <Route
      path="/social-emotional-understanding"
      element={<SocialEmotionalUnderstandingPage />}
    />
    <Route
      path="/cultural-adaptation"
      element={<CulturalAdaptationPage />}
    />
    <Route
      path="/social-intelligence-decision"
      element={<SocialIntelligenceDecisionPage />}
    />

    {/* 情感记忆管理系统 (Story 11.4) */}
    <Route
      path="/emotional-memory-management"
      element={<EmotionalMemoryManagementPage />}
    />
    <Route
      path="/emotional-event-analysis"
      element={<EmotionalEventAnalysisPage />}
    />
    <Route
      path="/emotional-preference-learning"
      element={<EmotionalMemoryPage />}
    />
    <Route
      path="/emotional-trigger-patterns"
      element={<EmotionalMemoryPage />}
    />
    <Route
      path="/emotional-memory-retrieval"
      element={<EmotionalMemoryPage />}
    />

    {/* 情感智能决策引擎 (Story 11.5) */}
    <Route
      path="/emotional-intelligence-decision-engine"
      element={<EmotionalIntelligenceDecisionEnginePage />}
    />
    <Route
      path="/emotional-risk-assessment-dashboard"
      element={<EmotionalRiskAssessmentDashboardPage />}
    />
    <Route
      path="/crisis-detection-support"
      element={<CrisisDetectionSupportPage />}
    />
    <Route
      path="/intervention-strategy-management"
      element={<InterventionStrategyManagementPage />}
    />
    <Route
      path="/emotional-health-monitoring-dashboard"
      element={<EmotionalHealthMonitoringDashboardPage />}
    />
    <Route
      path="/decision-history-analysis"
      element={<DecisionHistoryAnalysisPage />}
    />
    <Route
      path="/empathy-response-generator"
      element={<EmpathyResponseGeneratorPage />}
    />

    {/* 🧠 强化学习系统 */}
    <Route path="/qlearning" element={<QLearningPage />} />
    <Route
      path="/qlearning-training"
      element={<QLearningTrainingPage />}
    />
    <Route
      path="/qlearning-strategy"
      element={<QLearningStrategyPage />}
    />
    <Route
      path="/qlearning-recommendation"
      element={<QLearningRecommendationPage />}
    />
    <Route
      path="/qlearning-performance"
      element={<QLearningPerformancePage />}
    />
    <Route
      path="/qlearning-performance-optimization"
      element={<QLearningPerformanceOptimizationPage />}
    />

    {/* 🤖 TensorFlow模型管理 */}
    <Route
      path="/tensorflow"
      element={<TensorFlowManagementPage />}
    />
    <Route
      path="/qlearning/tabular"
      element={<TabularQLearningPage />}
    />
    <Route path="/qlearning/dqn" element={<DQNPage />} />
    <Route path="/qlearning/variants" element={<DQNVariantsPage />} />
    <Route
      path="/exploration-strategies"
      element={<ExplorationStrategiesPage />}
    />
    <Route path="/ucb-strategies" element={<UCBStrategiesPage />} />
    <Route
      path="/thompson-sampling"
      element={<ThompsonSamplingPage />}
    />
    <Route
      path="/adaptive-exploration"
      element={<ExplorationStrategiesPage />}
    />
    <Route path="/basic-rewards" element={<BasicRewardsPage />} />
    <Route
      path="/composite-rewards"
      element={<CompositeRewardsPage />}
    />
    <Route
      path="/adaptive-rewards"
      element={<AdaptiveRewardsPage />}
    />
    <Route path="/reward-shaping" element={<RewardShapingPage />} />
    <Route path="/state-space" element={<StateSpacePage />} />
    <Route path="/action-space" element={<ActionSpacePage />} />
    <Route
      path="/environment-simulator"
      element={<EnvironmentSimulatorPage />}
    />
    <Route path="/grid-world" element={<GridWorldPage />} />
    <Route
      path="/training-manager"
      element={<TrainingManagerPage />}
    />
    <Route
      path="/learning-rate-scheduler"
      element={<LearningRateSchedulerPage />}
    />
    <Route path="/early-stopping" element={<EarlyStoppingPage />} />
    <Route
      path="/performance-tracker"
      element={<PerformanceTrackerPage />}
    />

    {/* ❤️ 用户反馈学习系统 */}
    <Route path="/feedback-system" element={<FeedbackSystemPage />} />
    <Route
      path="/feedback-analytics"
      element={<FeedbackAnalyticsPage />}
    />
    <Route
      path="/user-feedback-profiles"
      element={<UserFeedbackProfilesPage />}
    />
    <Route
      path="/item-feedback-analysis"
      element={<ItemFeedbackAnalysisPage />}
    />
    <Route
      path="/feedback-quality-monitor"
      element={<FeedbackQualityMonitorPage />}
    />

    {/* 📈 智能行为分析系统 */}
    <Route
      path="/behavior-analytics"
      element={<BehaviorAnalyticsPage />}
    />
    <Route
      path="/behavior-analytics-enhanced"
      element={<BehaviorAnalyticsPageEnhanced />}
    />
    <Route
      path="/behavior-analytics/events"
      element={<EventDataManagePage />}
    />
    <Route
      path="/behavior-analytics/sessions"
      element={<SessionManagePage />}
    />
    <Route
      path="/behavior-analytics/reports"
      element={<ReportCenterPage />}
    />
    <Route
      path="/behavior-analytics/realtime"
      element={<RealTimeMonitorPage />}
    />
    <Route
      path="/behavior-analytics/export"
      element={<DataExportPage />}
    />
    <Route
      path="/behavior-analytics/config"
      element={<SystemConfigPage />}
    />

    {/* 🧠 记忆管理系统 */}
    <Route
      path="/memory-hierarchy"
      element={<MemoryHierarchyPage />}
    />
    <Route path="/memory-recall" element={<MemoryRecallTestPage />} />
    <Route
      path="/memory-analytics"
      element={<MemoryAnalyticsDashboard />}
    />

    {/* 🌐 多模态处理 */}
    <Route path="/multimodal" element={<MultimodalPageComplete />} />
    <Route
      path="/file-management"
      element={<FileManagementPageComplete />}
    />

    {/* 🔧 平台集成优化 */}
    <Route
      path="/platform-integration-overview"
      element={<PlatformIntegrationOverviewPage />}
    />
    <Route
      path="/component-management"
      element={<ComponentManagementPage />}
    />
    <Route
      path="/workflow-orchestration"
      element={<WorkflowOrchestrationPage />}
    />
    <Route
      path="/performance-optimization"
      element={<PerformanceOptimizationPage />}
    />
    <Route
      path="/system-monitoring"
      element={<SystemMonitoringPage />}
    />
    <Route
      path="/documentation-management"
      element={<DocumentationManagementPage />}
    />
    <Route
      path="/realtime-communication"
      element={<RealtimeCommunicationPage />}
    />

    {/* 🛡️ 故障容错和恢复系统 (Story 10.5) */}
    <Route
      path="/fault-tolerance-overview"
      element={<FaultToleranceSystemPage />}
    />
    <Route path="/fault-detection" element={<FaultDetectionPage />} />
    <Route
      path="/recovery-management"
      element={<RecoveryManagementPage />}
    />
    <Route
      path="/backup-management"
      element={<BackupManagementPage />}
    />
    <Route
      path="/consistency-management"
      element={<ConsistencyManagementPage />}
    />
    <Route path="/fault-testing" element={<FaultTestingPage />} />

    {/* ⚡ 工作流引擎 */}
    <Route path="/workflow" element={<WorkflowPage />} />
    <Route path="/workflows" element={<WorkflowPage />} />
    <Route
      path="/langgraph-features"
      element={<LangGraphFeaturesPage />}
    />

    {/* 🔗 分布式任务协调引擎 */}
    <Route
      path="/distributed-task-coordination"
      element={<DistributedTaskCoordinationPage />}
    />
    <Route path="/task-decomposer" element={<TaskDecomposerPage />} />
    <Route
      path="/intelligent-assigner"
      element={<IntelligentAssignerPage />}
    />
    <Route path="/raft-consensus" element={<RaftConsensusPage />} />
    <Route
      path="/distributed-state-manager"
      element={<DistributedStateManagerPage />}
    />
    <Route
      path="/conflict-resolver"
      element={<ConflictResolverPage />}
    />
    <Route
      path="/distributed-task-monitor"
      element={<DistributedTaskMonitorPage />}
    />
    <Route
      path="/distributed-task-system-status"
      element={<DistributedTaskSystemStatusPage />}
    />
    <Route
      path="/distributed-task-management-enhanced"
      element={<DistributedTaskManagementPageEnhanced />}
    />
    <Route
      path="/dag-orchestrator"
      element={<DagOrchestratorPage />}
    />
    <Route path="/flow-control" element={<FlowControlPage />} />
    <Route
      path="/distributed-message-overview"
      element={<DistributedMessageOverviewPage />}
    />

    {/* 🏭 处理引擎 */}
    <Route
      path="/streaming-monitor"
      element={<StreamingMonitorPage />}
    />
    <Route path="/batch" element={<BatchJobsPageFixed />} />
    <Route
      path="/batch-processing"
      element={<BatchProcessingPage />}
    />
    <Route
      path="/intelligent-scheduling"
      element={<IntelligentSchedulingPage />}
    />
    <Route
      path="/unified-engine"
      element={<UnifiedEnginePageComplete />}
    />

    {/* 🛡️ 安全与合规 */}
    <Route path="/ai-trism" element={<AiTrismPage />} />
    <Route
      path="/security-management"
      element={<SecurityManagementPage />}
    />
    <Route
      path="/security-management-enhanced"
      element={<SecurityManagementEnhancedPage />}
    />
    <Route
      path="/distributed-security-monitor"
      element={<DistributedSecurityMonitorPage />}
    />
    <Route path="/auth-management" element={<AuthManagementPage />} />

    {/* 📊 事件与监控 */}
    <Route path="/events" element={<EventDashboardPage />} />
    <Route path="/health" element={<HealthMonitorPage />} />
    <Route path="/performance" element={<PerformanceMonitorPage />} />
    <Route path="/monitor" element={<UnifiedMonitorPage />} />
    <Route
      path="/monitoring-dashboard"
      element={<MonitoringDashboardPage />}
    />

    {/* 🗄️ 数据存储 */}
    <Route path="/pgvector" element={<PgVectorPage />} />
    <Route path="/vector-advanced" element={<VectorAdvancedPage />} />
    <Route path="/cache" element={<CacheMonitorPage />} />
    <Route
      path="/assignment-cache"
      element={<AssignmentCacheManagementPage />}
    />

    {/* 🔧 协议与工具 */}
    <Route path="/mcp-tools" element={<MCPToolsPage />} />

    {/* 🏢 企业架构 */}
    <Route
      path="/enterprise"
      element={<EnterpriseArchitecturePage />}
    />
    <Route
      path="/enterprise-architecture"
      element={<EnterpriseArchitecturePage />}
    />
    <Route
      path="/enterprise-config"
      element={<EnterpriseConfigPage />}
    />
    <Route path="/debug" element={<ArchitectureDebugPage />} />

    {/* 🔄 离线能力与同步机制 */}
    <Route path="/offline" element={<OfflineCapabilityPage />} />
    <Route path="/sync" element={<SyncManagementPage />} />
    <Route path="/conflicts" element={<ConflictResolutionPage />} />
    <Route
      path="/vector-clock"
      element={<VectorClockVisualizationPage />}
    />
    <Route
      path="/network-monitor"
      element={<NetworkMonitorDetailPage />}
    />
    <Route path="/sync-engine" element={<SyncEngineInternalPage />} />
    <Route path="/model-cache" element={<ModelCacheMonitorPage />} />

    {/* 🔬 开发测试 */}
    <Route
      path="/structured-errors"
      element={<StructuredErrorPage />}
    />
    <Route path="/test-coverage" element={<TestCoveragePage />} />
    <Route path="/test" element={<IntegrationTestPage />} />
    <Route path="/test-suite" element={<TestingSuitePage />} />
    <Route
      path="/document-processing"
      element={<DocumentProcessingPage />}
    />

    {/* 缺失页面补充 */}
    <Route
      path="/conflict-resolution-learning"
      element={<ConflictResolutionLearningPage />}
    />
    <Route
      path="/sync-engine-learning"
      element={<SyncEngineLearningPage />}
    />
    <Route
      path="/health-comprehensive"
      element={<HealthComprehensivePage />}
    />
    {/* 已移除不存在的multimodal-simple路由 */}
    {/* 已移除不存在的vector-advanced-simple路由 */}
    {/* 已移除不存在的vector-advanced-test路由 */}
    <Route
      path="/vector-clock-algorithm"
      element={<VectorClockAlgorithmPage />}
    />
    <Route
      path="/unified-engine-complete"
      element={<UnifiedEnginePageComplete />}
    />
    <Route path="/batch-jobs" element={<BatchJobsPageFixed />} />
    <Route
      path="/document-processing-simple"
      element={<DocumentProcessingPage />}
    />
    <Route
      path="/document-processing-advanced"
      element={<DocumentProcessingAdvancedPage />}
    />
    <Route path="/security" element={<SecurityPage />} />
    <Route
      path="/multimodal-basic"
      element={<MultimodalPageComplete />}
    />
    <Route
      path="/multimodal-complete"
      element={<MultimodalPageComplete />}
    />
    <Route
      path="/file-management-advanced"
      element={<FileManagementAdvancedPage />}
    />
    <Route
      path="/distributed-events"
      element={<DistributedEventsPage />}
    />
    <Route path="/langgraph-065" element={<LangGraph065Page />} />
    <Route path="/multimodal-rag" element={<MultimodalRagPage />} />
    <Route
      path="/multimodal-rag-management"
      element={<MultimodalRagManagementPage />}
    />
    <Route
      path="/document-management-complete"
      element={<DocumentManagementPageComplete />}
    />
    <Route
      path="/realtime-metrics-management"
      element={<RealtimeMetricsManagementPage />}
    />

    {/* 🧪 A/B测试实验平台 */}
    <Route path="/experiments" element={<ExperimentListPage />} />
    <Route
      path="/experiments/dashboard"
      element={<ExperimentDashboardPage />}
    />
    <Route
      path="/experiments/enhanced-analysis"
      element={<EnhancedExperimentAnalysisPage />}
    />
    <Route
      path="/experiments/statistical-analysis"
      element={<StatisticalAnalysisPage />}
    />
    <Route
      path="/experiments/traffic-allocation"
      element={<TrafficAllocationPage />}
    />
    <Route
      path="/experiments/event-tracking"
      element={<EventTrackingPage />}
    />
    <Route
      path="/experiments/traffic-ramp"
      element={<TrafficRampManagementPage />}
    />
    <Route
      path="/experiments/power-analysis"
      element={<PowerAnalysisPage />}
    />
    <Route
      path="/descriptive-statistics"
      element={<DescriptiveStatisticsPage />}
    />
    <Route
      path="/experiments/multiple-testing"
      element={<MultipleTestingCorrectionPage />}
    />
    <Route
      path="/experiments/layered-experiments"
      element={<LayeredExperimentsManagementPage />}
    />
    <Route
      path="/experiments/release-strategy"
      element={<ReleaseStrategyPage />}
    />
    <Route
      path="/experiments/monitoring-alerts"
      element={<MonitoringAlertsPage />}
    />
    <Route
      path="/experiments/advanced-algorithms"
      element={<AdvancedAlgorithmsPage />}
    />

    {/* 服务发现与离线管理 */}
    <Route
      path="/service-discovery-management"
      element={<ServiceDiscoveryManagementPage />}
    />
    <Route
      path="/offline-management"
      element={<OfflineManagementPage />}
    />

    {/* 📊 强化学习系统监控 */}
    <Route
      path="/rl-system-dashboard"
      element={<RLSystemDashboardPage />}
    />
    <Route
      path="/rl-performance-monitor"
      element={<RLPerformanceMonitorPage />}
    />
    <Route
      path="/rl-integration-test"
      element={<RLIntegrationTestPage />}
    />
    <Route path="/rl-alert-config" element={<RLAlertConfigPage />} />
    <Route
      path="/rl-metrics-analysis"
      element={<RLMetricsAnalysisPage />}
    />
    <Route
      path="/rl-system-health"
      element={<RLSystemHealthPage />}
    />

    {/* ⚡ LoRA/QLoRA微调框架 */}
    <Route
      path="/fine-tuning-jobs"
      element={<FineTuningJobsPage />}
    />
    <Route
      path="/fine-tuning-config"
      element={<FineTuningConfigPage />}
    />
    <Route
      path="/fine-tuning-monitor"
      element={<FineTuningMonitorPage />}
    />
    <Route
      path="/fine-tuning-models"
      element={<FineTuningModelsPage />}
    />
    <Route
      path="/fine-tuning-datasets"
      element={<FineTuningDatasetsPage />}
    />
    <Route path="/fine-tuning" element={<FineTuningPage />} />
    <Route
      path="/fine-tuning-enhanced"
      element={<FineTuningPageEnhanced />}
    />
    <Route
      path="/lora-training-overview"
      element={<LoRATrainingPage />}
    />
    <Route
      path="/lora-config-templates"
      element={<FineTuningConfigPage />}
    />
    <Route
      path="/lora-model-adapters"
      element={<ModelAdaptersPage />}
    />
    <Route
      path="/lora-performance-monitor"
      element={<FineTuningMonitorPage />}
    />
    <Route
      path="/qlora-training-overview"
      element={<QLoRATrainingPage />}
    />
    <Route
      path="/qlora-quantization-config"
      element={<FineTuningConfigPage />}
    />
    <Route
      path="/qlora-memory-optimization"
      element={<FineTuningMonitorPage />}
    />
    <Route
      path="/qlora-inference-optimization"
      element={<FineTuningMonitorPage />}
    />
    <Route
      path="/distributed-training-overview"
      element={<DistributedTrainingPage />}
    />
    <Route
      path="/auto-scaling-management"
      element={<AutoScalingManagementPage />}
    />
    <Route
      path="/risk-assessment-dashboard"
      element={<RiskAssessmentDashboard />}
    />
    <Route
      path="/statistical-analysis-dashboard"
      element={<StatisticalAnalysisDashboard />}
    />
    <Route
      path="/deepspeed-configuration"
      element={<FineTuningConfigPage />}
    />
    <Route
      path="/multi-gpu-monitoring"
      element={<TrainingMonitorDashboard />}
    />
    <Route
      path="/training-synchronization"
      element={<DistributedTrainingPage />}
    />
    <Route
      path="/training-dashboard"
      element={<TrainingMonitorDashboard />}
    />
    <Route
      path="/training-metrics"
      element={<FineTuningMonitorPage />}
    />
    <Route
      path="/training-anomaly-detection"
      element={<TrainingMonitorDashboard />}
    />
    <Route
      path="/anomaly-detection"
      element={<AnomalyDetectionPage />}
    />
    <Route
      path="/auto-scaling"
      element={<AutoScalingManagementPage />}
    />
    <Route
      path="/training-reports"
      element={<TrainingMonitorDashboard />}
    />
    <Route
      path="/supported-models"
      element={<FineTuningModelsPage />}
    />
    <Route
      path="/model-checkpoints"
      element={<FineTuningCheckpointsPage />}
    />
    <Route
      path="/model-performance-comparison"
      element={<ModelPerformanceComparison />}
    />
    {/* 📊 训练数据管理系统 (Story 9.5) */}
    <Route
      path="/training-data-management"
      element={<TrainingDataManagementPage />}
    />
    <Route
      path="/training-data-enhanced"
      element={<TrainingDataManagementPageEnhanced />}
    />
    <Route
      path="/data-sources"
      element={<DataSourceManagementPage />}
    />
    <Route path="/data-collection" element={<DataCollectionPage />} />
    <Route
      path="/data-preprocessing"
      element={<DataPreprocessingPage />}
    />
    <Route
      path="/data-annotation"
      element={<DataAnnotationManagementPage />}
    />
    <Route
      path="/annotation-tasks"
      element={<AnnotationTasksPage />}
    />
    <Route
      path="/annotation-quality"
      element={<AnnotationQualityControlPage />}
    />
    <Route
      path="/data-versioning"
      element={<DataVersionManagementPage />}
    />
    <Route
      path="/data-version-comparison"
      element={<TrainingDataManagementPage />}
    />
    <Route
      path="/data-export"
      element={<TrainingDataManagementPage />}
    />
    <Route
      path="/data-statistics"
      element={<TrainingDataManagementPage />}
    />
    <Route
      path="/quality-metrics"
      element={<TrainingDataManagementPage />}
    />

    {/* 📊 模型评估和基准测试系统 (Story 9.4) */}
    <Route
      path="/model-evaluation-overview"
      element={<ModelEvaluationOverviewPage />}
    />
    <Route
      path="/model-evaluation-management"
      element={<ModelEvaluationManagementPage />}
    />
    <Route
      path="/memory-management-monitor"
      element={<MemoryManagementMonitorPage />}
    />
    <Route
      path="/model-performance-benchmark"
      element={<ModelPerformanceBenchmarkPage />}
    />
    <Route
      path="/evaluation-engine-management"
      element={<EvaluationEngineManagementPage />}
    />
    <Route
      path="/benchmark-suite-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-tasks-monitor"
      element={<EvaluationTasksMonitorPage />}
    />
    <Route
      path="/evaluation-reports-center"
      element={<EvaluationReportsCenterPage />}
    />
    <Route
      path="/evaluation-api-management"
      element={<EvaluationApiManagementPage />}
    />
    <Route
      path="/model-comparison-dashboard"
      element={<ModelComparisonDashboardPage />}
    />
    <Route
      path="/benchmark-glue-management"
      element={<BenchmarkGlueManagementPage />}
    />
    <Route
      path="/benchmark-superglue-management"
      element={<BenchmarkSupergluePage />}
    />
    <Route
      path="/benchmark-mmlu-management"
      element={<BenchmarkMmluPage />}
    />
    <Route
      path="/benchmark-humaneval-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/benchmark-hellaswag-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/benchmark-custom-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-metrics-config"
      element={<EvaluationMetricsConfigPage />}
    />
    <Route
      path="/evaluation-performance-monitor"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-batch-processing"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-regression-detection"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-quality-assurance"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-automation-pipeline"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-alerts-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-data-management"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-resource-monitor"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-job-scheduler"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-results-analysis"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-export-import"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-version-control"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-compliance-audit"
      element={<BenchmarkSuiteManagementPage />}
    />
    <Route
      path="/evaluation-security-management"
      element={<BenchmarkSuiteManagementPage />}
    />

    {/* 🚀 模型压缩和量化工具 (Story 9.2) */}
    <Route
      path="/model-compression-overview"
      element={<ModelCompressionOverviewPage />}
    />
    <Route
      path="/quantization-manager"
      element={<QuantizationManagerPage />}
    />
    <Route
      path="/knowledge-distillation"
      element={<KnowledgeDistillationPage />}
    />
    <Route path="/model-pruning" element={<ModelPruningPage />} />
    <Route
      path="/compression-pipeline"
      element={<CompressionPipelinePage />}
    />
    <Route
      path="/compression-evaluation"
      element={<ModelCompressionEvaluationPage />}
    />
    <Route
      path="/performance-benchmark"
      element={<ModelPerformanceBenchmarkPage />}
    />
    <Route
      path="/hardware-benchmark"
      element={<ModelPerformanceBenchmarkPage />}
    />
    <Route
      path="/strategy-recommendation"
      element={<CompressionStrategyRecommendationPage />}
    />

    {/* 🚀 模型服务部署平台 (Story 9.6) */}
    <Route path="/model-registry" element={<ModelRegistryPage />} />
    <Route path="/model-inference" element={<ModelInferencePage />} />
    <Route
      path="/model-deployment"
      element={<ModelDeploymentPage />}
    />
    <Route
      path="/model-monitoring"
      element={<ModelMonitoringPage />}
    />
    <Route
      path="/model-service-management"
      element={<ModelServiceManagementPage />}
    />
    <Route path="/online-learning" element={<OnlineLearningPage />} />

    {/* 🚀 自动化超参数优化系统 (Story 9.3) */}
    <Route
      path="/hyperparameter-optimization"
      element={<HyperparameterOptimizationPage />}
    />
    <Route
      path="/hyperparameter-optimization-enhanced"
      element={<HyperparameterOptimizationPageEnhanced />}
    />
    <Route
      path="/hyperparameter-experiments"
      element={<HyperparameterExperimentsPage />}
    />
    <Route
      path="/hyperparameter-algorithms"
      element={<HyperparameterAlgorithmsPage />}
    />
    <Route
      path="/hyperparameter-visualizations"
      element={<HyperparameterVisualizationsPage />}
    />
    <Route
      path="/hyperparameter-monitoring"
      element={<HyperparameterMonitoringPage />}
    />
    <Route
      path="/hyperparameter-resources"
      element={<HyperparameterResourcesPage />}
    />
    <Route
      path="/hyperparameter-scheduler"
      element={<HyperparameterSchedulerPage />}
    />
    <Route
      path="/hyperparameter-reports"
      element={<HyperparameterReportsPage />}
    />

    {/* 🌐 智能代理服务发现系统 (Story 10.1) */}
    <Route
      path="/service-discovery-overview"
      element={<ServiceDiscoveryOverviewPage />}
    />
    <Route
      path="/agent-registry"
      element={<AgentRegistryManagementPage />}
    />
    <Route
      path="/service-routing"
      element={<ServiceRoutingManagementPage />}
    />
    <Route
      path="/load-balancer-config"
      element={<LoadBalancerConfigPage />}
    />
    <Route
      path="/service-health-monitor"
      element={<ServiceHealthMonitorPage />}
    />
    <Route
      path="/service-cluster-management"
      element={<ServiceClusterManagementPage />}
    />
    <Route
      path="/service-performance-dashboard"
      element={<ServicePerformanceDashboardPage />}
    />
    <Route
      path="/service-config-management"
      element={<ServiceConfigManagementPage />}
    />
    <Route path="*" element={<MissingRouteFallback />} />
    </Routes>
  )
}

export default AppRoutes
