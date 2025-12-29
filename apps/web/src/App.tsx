import React, { useState, Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Layout, Menu, Button, Typography, Space, Avatar, Spin } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  MessageOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  RobotOutlined,
  TeamOutlined,
  ControlOutlined,
  SearchOutlined,
  NodeIndexOutlined,
  ThunderboltOutlined,
  BellOutlined,
  SafetyOutlined,
  DashboardOutlined,
  BugOutlined,
  ApiOutlined,
  MonitorOutlined,
  WifiOutlined,
  CloudServerOutlined,
  SettingOutlined,
  ExceptionOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
  FileImageOutlined,
  BulbOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  FileTextOutlined,
  UserOutlined,
  HeartOutlined,
  TrophyOutlined,
  EyeOutlined,
  ShareAltOutlined,
  FundViewOutlined,
  RocketOutlined,
  UnorderedListOutlined,
  BarChartOutlined,
  ExportOutlined,
  HistoryOutlined,
  CameraOutlined,
  DiffOutlined,
  RollbackOutlined,
  SecurityScanOutlined,
  KeyOutlined,
  AuditOutlined,
  FileExcelOutlined,
  GoldOutlined,
  FundProjectionScreenOutlined,
  DeploymentUnitOutlined,
  CodeOutlined,
  SyncOutlined,
  GlobalOutlined,
  ClusterOutlined,
  CompressOutlined,
  ScissorOutlined,
  CloudUploadOutlined,
  InboxOutlined,
  EditOutlined,
  TagsOutlined,
  BranchesOutlined,
  CrownOutlined,
  AudioOutlined,
  CompassOutlined,
  WarningOutlined,
  FunctionOutlined,
} from '@ant-design/icons'

// 懒加载所有页面组件
const ChatPage = lazy(() => import('./pages/ChatPage'))
const ConversationHistoryPage = lazy(() => import('./pages/ConversationHistoryPage'))
const MultiAgentPage = lazy(() => import('./pages/MultiAgentPage'))
const TensorFlowQLearningPage = lazy(() => import('./pages/TensorFlowQLearningManagementPage'))
const TestingManagementPage = lazy(() => import('./pages/TestingManagementPage'))
const HypothesisTestingPage = lazy(() => import('./pages/HypothesisTestingPage'))
const EnhancedExperimentAnalysisPage = lazy(() => import('./pages/experiments/EnhancedExperimentAnalysisPage'))
const SupervisorPage = lazy(() => import('./pages/SupervisorPage'))
const RagPage = lazy(() => import('./pages/RagPage'))
const WorkflowPage = lazy(() => import('./pages/WorkflowPage'))
const AsyncAgentPage = lazy(() => import('./pages/AsyncAgentPage'))
const AgenticRagPage = lazy(() => import('./pages/AgenticRagPage'))
const GraphRAGPage = lazy(() => import('./pages/GraphRAGPage'))
const GraphRAGPageEnhanced = lazy(() => import('./pages/GraphRAGPageEnhanced'))
const MultimodalPageComplete = lazy(() => import('./pages/MultimodalPageComplete'))
const FlowControlPage = lazy(() => import('./pages/FlowControlPage'))
const DistributedMessageOverviewPage = lazy(() => import('./pages/DistributedMessageOverviewPage'))
const MCPToolsPage = lazy(() => import('./pages/MCPToolsPage'))
const PgVectorPage = lazy(() => import('./pages/PgVectorPage'))
const CacheMonitorPage = lazy(() => import('./pages/CacheMonitorPage'))
const BatchJobsPageFixed = lazy(() => import('./pages/BatchJobsPageFixed'))
const BatchProcessingPage = lazy(() => import('./pages/BatchProcessingPage'))
const IntelligentSchedulingPage = lazy(() => import('./pages/IntelligentSchedulingPage'))
const HealthMonitorPage = lazy(() => import('./pages/HealthMonitorPage'))
const PerformanceMonitorPage = lazy(() => import('./pages/PerformanceMonitorPage'))
const StreamingMonitorPage = lazy(() => import('./pages/StreamingMonitorPage'))
const MonitoringDashboardPage = lazy(() => import('./pages/MonitoringDashboardPage'))
const EnterpriseArchitecturePage = lazy(() => import('./pages/EnterpriseArchitecturePageSimple'))
const EnterpriseConfigPage = lazy(() => import('./pages/EnterpriseConfigPage'))
const ArchitectureDebugPage = lazy(() => import('./pages/ArchitectureDebugPage'))
const StructuredErrorPage = lazy(() => import('./pages/StructuredErrorPage'))
const TestCoveragePage = lazy(() => import('./pages/TestCoveragePage'))
const IntegrationTestPage = lazy(() => import('./pages/IntegrationTestPage'))
const TestingSuitePage = lazy(() => import('./pages/TestingSuitePage'))
const DocumentProcessingAdvancedPage = lazy(() => import('./pages/DocumentProcessingAdvancedPage'))
const QLearningPerformanceOptimizationPage = lazy(() => import('./pages/QLearningPerformanceOptimizationPage'))
const AuthManagementPage = lazy(() => import('./pages/AuthManagementPage'))
const LangGraphFeaturesPage = lazy(() => import('./pages/LangGraphFeaturesPage'))
const AgentInterfacePage = lazy(() => import('./pages/AgentInterfacePage'))

// 增强版页面
const MultiAgentEnhancedPage = lazy(() => import('./pages/MultiAgentEnhancedPage'))
const RAGEnhancedPage = lazy(() => import('./pages/RAGEnhancedPage'))
const ExperimentsPlatformPage = lazy(() => import('./pages/ExperimentsPlatformPage'))
const WorkflowManagementPage = lazy(() => import('./pages/WorkflowManagementPage'))
const AgentClusterManagementPage = lazy(() => import('./pages/AgentClusterManagementPage'))
const AgentClusterManagementPageEnhanced = lazy(() => import('./pages/AgentClusterManagementPageEnhanced'))
const MemoryHierarchyPage = lazy(() => import('./pages/MemoryHierarchyPage'))
const MemoryRecallTestPage = lazy(() => import('./pages/MemoryRecallTestPage'))
const MemoryAnalyticsDashboard = lazy(() => import('./pages/MemoryAnalyticsDashboard'))
const ReasoningPage = lazy(() => import('./pages/ReasoningPage'))
const MultiStepReasoningPage = lazy(() => import('./pages/MultiStepReasoningPage'))
const ExplainableAiPage = lazy(() => import('./pages/ExplainableAiPage'))
const TargetingRulesManagementPage = lazy(() => import('./pages/TargetingRulesManagementPage'))
const HybridSearchAdvancedPage = lazy(() => import('./pages/HybridSearchAdvancedPage'))
const FileManagementPageComplete = lazy(() => import('./pages/FileManagementPageComplete'))
const AnomalyDetectionPage = lazy(() => import('./pages/AnomalyDetectionPage'))
const AutoScalingManagementPage = lazy(() => import('./pages/AutoScalingManagementPage'))
const BatchOperationsPage = lazy(() => import('./pages/BatchOperationsPage'))
const AssignmentCacheManagementPage = lazy(() => import('./pages/AssignmentCacheManagementPage'))

// 智能代理服务发现系统页面 (Story 10.1)
const ServiceDiscoveryOverviewPage = lazy(() => import('./pages/ServiceDiscoveryOverviewPage'))
const AgentRegistryManagementPage = lazy(() => import('./pages/AgentRegistryManagementPage'))
const ServiceRoutingManagementPage = lazy(() => import('./pages/ServiceRoutingManagementPage'))
const LoadBalancerConfigPage = lazy(() => import('./pages/LoadBalancerConfigPage'))
const ServiceHealthMonitorPage = lazy(() => import('./pages/ServiceHealthMonitorPage'))
const ServiceClusterManagementPage = lazy(() => import('./pages/ServiceClusterManagementPage'))
const ServicePerformanceDashboardPage = lazy(() => import('./pages/ServicePerformanceDashboardPage'))
const ServiceConfigManagementPage = lazy(() => import('./pages/ServiceConfigManagementPage'))

// 平台集成优化系统页面
const PlatformIntegrationOverviewPage = lazy(() => import('./pages/PlatformIntegrationOverviewPage'))
const ComponentManagementPage = lazy(() => import('./pages/ComponentManagementPage'))
const WorkflowOrchestrationPage = lazy(() => import('./pages/WorkflowOrchestrationPage'))
const PerformanceOptimizationPage = lazy(() => import('./pages/PerformanceOptimizationPage'))
// const PerformanceOptimizationPageSimple = lazy(() => import('./pages/PerformanceOptimizationPageSimple'))
// const PerformanceOptimizationPageFixed = lazy(() => import('./pages/PerformanceOptimizationPageFixed'))
// const PerformanceOptimizationPageMinimal = lazy(() => import('./pages/PerformanceOptimizationPageMinimal'))
const SystemMonitoringPage = lazy(() => import('./pages/SystemMonitoringPage'))
const DocumentationManagementPage = lazy(() => import('./pages/DocumentationManagementPage'))

// 故障容错和恢复系统页面 (Story 10.5)
const FaultToleranceSystemPage = lazy(() => import('./pages/FaultToleranceSystemPage'))
const RealtimeCommunicationPage = lazy(() => import('./pages/RealtimeCommunicationPage'))
const FaultDetectionPage = lazy(() => import('./pages/FaultDetectionPage'))
const RecoveryManagementPage = lazy(() => import('./pages/RecoveryManagementPage'))
const BackupManagementPage = lazy(() => import('./pages/BackupManagementPage'))
const ConsistencyManagementPage = lazy(() => import('./pages/ConsistencyManagementPage'))
const FaultTestingPage = lazy(() => import('./pages/FaultTestingPage'))

// LoRA/QLoRA微调框架页面
const FineTuningJobsPage = lazy(() => import('./pages/FineTuningJobsPage'))
const FineTuningConfigPage = lazy(() => import('./pages/FineTuningConfigPage'))
const FineTuningMonitorPage = lazy(() => import('./pages/FineTuningMonitorPage'))
const FineTuningModelsPage = lazy(() => import('./pages/FineTuningModelsPage'))
const FineTuningDatasetsPage = lazy(() => import('./pages/FineTuningDatasetsPage'))
const FineTuningCheckpointsPage = lazy(() => import('./pages/FineTuningCheckpointsPage'))
const FineTuningPage = lazy(() => import('./pages/FineTuningPage'))
const FineTuningPageEnhanced = lazy(() => import('./pages/FineTuningPageEnhanced'))
const LoRATrainingPage = lazy(() => import('./pages/LoRATrainingPage'))
const QLoRATrainingPage = lazy(() => import('./pages/QLoRATrainingPage'))
const DistributedTrainingPage = lazy(() => import('./pages/DistributedTrainingPage'))
const RiskAssessmentDashboard = lazy(() => import('./pages/RiskAssessmentDashboard'))
const StatisticalAnalysisDashboard = lazy(() => import('./pages/StatisticalAnalysisDashboard'))
const ModelAdaptersPage = lazy(() => import('./pages/ModelAdaptersPage'))
const TrainingMonitorDashboard = lazy(() => import('./pages/TrainingMonitorDashboard'))
const ModelPerformanceComparison = lazy(() => import('./pages/ModelPerformanceComparison'))

// 分布式任务协调引擎页面 (Story 10.3)
const DistributedTaskCoordinationPage = lazy(() => import('./pages/DistributedTaskCoordinationPage'))
const TaskDecomposerPage = lazy(() => import('./pages/TaskDecomposerPage'))
const IntelligentAssignerPage = lazy(() => import('./pages/IntelligentAssignerPage'))
const RaftConsensusPage = lazy(() => import('./pages/RaftConsensusPage'))
const DistributedStateManagerPage = lazy(() => import('./pages/DistributedStateManagerPage'))
const ConflictResolverPage = lazy(() => import('./pages/ConflictResolverPage'))
const DistributedTaskMonitorPage = lazy(() => import('./pages/DistributedTaskMonitorPage'))
const DistributedTaskSystemStatusPage = lazy(() => import('./pages/DistributedTaskSystemStatusPage'))
const DistributedTaskManagementPageEnhanced = lazy(() => import('./pages/DistributedTaskManagementPageEnhanced'))

// 知识图谱引擎
const KnowledgeExtractionOverviewPage = lazy(() => import('./pages/KnowledgeExtractionOverviewPage'))
const EntityRecognitionPage = lazy(() => import('./pages/EntityRecognitionPage'))
const RelationExtractionPage = lazy(() => import('./pages/RelationExtractionPage'))
const EntityLinkingPage = lazy(() => import('./pages/EntityLinkingPage'))
const MultilingualProcessingPage = lazy(() => import('./pages/MultilingualProcessingPage'))
const KnowledgeGraphVisualizationPage = lazy(() => import('./pages/KnowledgeGraphVisualizationPage'))
const KnowledgeGraphPage = lazy(() => import('./pages/KnowledgeGraphManagementPage'))
const GraphQueryEnginePage = lazy(() => import('./pages/GraphQueryEnginePage'))
const GraphAnalyticsPage = lazy(() => import('./pages/GraphAnalyticsPage'))
const KnowledgeBatchJobsPage = lazy(() => import('./pages/KnowledgeBatchJobsPage'))
const KnowledgeBatchMonitorPage = lazy(() => import('./pages/KnowledgeBatchMonitorPage'))
const KnowledgePerformanceOptimizationPage = lazy(() => import('./pages/KnowledgePerformanceOptimizationPage'))
const KnowledgeCacheManagementPage = lazy(() => import('./pages/KnowledgeCacheManagementPage'))
const KnowledgeValidationPage = lazy(() => import('./pages/KnowledgeValidationPage'))
const KnowledgeConfidenceAnalysisPage = lazy(() => import('./pages/KnowledgeConfidenceAnalysisPage'))
const KnowledgeErrorAnalysisPage = lazy(() => import('./pages/KnowledgeErrorAnalysisPage'))
const KnowledgeModelComparisonPage = lazy(() => import('./pages/KnowledgeModelComparisonPage'))
const ACLProtocolManagementPage = lazy(() => import('./pages/ACLProtocolManagementPage'))

// 动态知识图谱存储系统 (Story 8.2)
const KnowledgeGraphEntityManagement = lazy(() => import('./pages/KnowledgeGraphEntityManagement'))
const KnowledgeGraphRelationManagement = lazy(() => import('./pages/KnowledgeGraphRelationManagement'))
const KnowledgeGraphQueryEngine = lazy(() => import('./pages/KnowledgeGraphQueryEngine'))
const KnowledgeGraphIncrementalUpdate = lazy(() => import('./pages/KnowledgeGraphIncrementalUpdate'))
const KnowledgeGraphQualityAssessment = lazy(() => import('./pages/KnowledgeGraphQualityAssessment'))
const KnowledgeGraphPerformanceMonitor = lazy(() => import('./pages/KnowledgeGraphPerformanceMonitorWorkingMinimal'))
const KnowledgeGraphSchemaManagement = lazy(() => import('./pages/KnowledgeGraphSchemaManagement'))
const KnowledgeGraphDataMigration = lazy(() => import('./pages/KnowledgeGraphDataMigration'))

// 知识管理API接口 (Story 8.6)
// SPARQL查询引擎
const SparqlQueryInterface = lazy(() => import('./pages/SparqlQueryInterface'))
const SparqlOptimization = lazy(() => import('./pages/SparqlOptimization'))
const SparqlPerformance = lazy(() => import('./pages/SparqlPerformance'))
const SparqlCache = lazy(() => import('./pages/SparqlCache'))

// 知识管理REST API
const EntityApiPage = lazy(() => import('./pages/EntityApiPage'))
const RelationApiPage = lazy(() => import('./pages/RelationApiPage'))
const GraphValidationPage = lazy(() => import('./pages/GraphValidationPage'))
const BasicRagManagementPage = lazy(() => import('./pages/BasicRagManagementPage'))
const SupervisorApiManagementPage = lazy(() => import('./pages/SupervisorApiManagementPage'))
const PlatformApiManagementPage = lazy(() => import('./pages/PlatformApiManagementPage'))

// 数据导入导出
const RdfImportExportPage = lazy(() => import('./pages/RdfImportExportPage'))
const CsvExcelImportPage = lazy(() => import('./pages/CsvExcelImportPage'))
const BatchImportJobsPage = lazy(() => import('./pages/BatchImportJobsPage'))
const ExportFormatsPage = lazy(() => import('./pages/ExportFormatsPage'))

// 版本控制系统
const GraphSnapshotsPage = lazy(() => import('./pages/GraphSnapshotsPage'))
const VersionComparisonPage = lazy(() => import('./pages/VersionComparisonPage'))
const RollbackOperationsPage = lazy(() => import('./pages/RollbackOperationsPage'))
const ChangeTrackingPage = lazy(() => import('./pages/ChangeTrackingPage'))

// 认证与安全
const JwtAuthPage = lazy(() => import('./pages/JwtAuthPage'))
const ApiKeyManagementPage = lazy(() => import('./pages/ApiKeyManagementPage'))
const RolePermissionsPage = lazy(() => import('./pages/RolePermissionsPage'))
const SecurityAuditPage = lazy(() => import('./pages/SecurityAuditPage'))
const SecurityPage = lazy(() => import('./pages/SecurityPage'))
const SecurityManagementPage = lazy(() => import('./pages/SecurityManagementPage'))
const SecurityManagementEnhancedPage = lazy(() => import('./pages/SecurityManagementEnhancedPage'))
const DistributedSecurityMonitorPage = lazy(() => import('./pages/DistributedSecurityMonitorPage'))

// 监控与日志
const PerformanceMetricsPage = lazy(() => import('./pages/PerformanceMetricsPage'))
const SystemHealthPage = lazy(() => import('./pages/SystemHealthPage'))
const AlertManagementPage = lazy(() => import('./pages/AlertManagementPage'))
const AuditLogsPage = lazy(() => import('./pages/AuditLogsPage'))

// 知识图推理引擎 (Story 8.3)
// 混合推理引擎
const KGReasoningDashboardPage = lazy(() => import('./pages/KGReasoningDashboardPage'))
const KGReasoningQueryPage = lazy(() => import('./pages/KGReasoningQueryPage'))
const KGReasoningBatchPage = lazy(() => import('./pages/KGReasoningBatchPage'))
const KGReasoningOptimizationPage = lazy(() => import('./pages/KGReasoningOptimizationPage'))
const KGReasoningConfigPage = lazy(() => import('./pages/KGReasoningConfigPage'))
const KGReasoningAnalysisPage = lazy(() => import('./pages/KGReasoningAnalysisPage'))

// 规则推理引擎
const KGRuleManagementPage = lazy(() => import('./pages/KGRuleManagementPage'))
const KGRuleExecutionPage = lazy(() => import('./pages/KGRuleExecutionPage'))
const KGRuleValidationPage = lazy(() => import('./pages/KGRuleValidationPage'))
const KGRuleConflictPage = lazy(() => import('./pages/KGRuleConflictPage'))

// 嵌入推理引擎
const KGEmbeddingModelsPage = lazy(() => import('./pages/KGEmbeddingModelsPage'))
const KGEmbeddingTrainingPage = lazy(() => import('./pages/KGEmbeddingTrainingPage'))
const KGEmbeddingSimilarityPage = lazy(() => import('./pages/KGEmbeddingSimilarityPage'))
const KGEmbeddingIndexPage = lazy(() => import('./pages/KGEmbeddingIndexPage'))

// 路径推理引擎
const KGPathDiscoveryPage = lazy(() => import('./pages/KGPathDiscoveryPage'))
const KGPathAnalysisPage = lazy(() => import('./pages/KGPathAnalysisPage'))
const KGPathOptimizationPage = lazy(() => import('./pages/KGPathOptimizationPage'))
const KGPathConfidencePage = lazy(() => import('./pages/KGPathConfidencePage'))

// 不确定性推理
const KGUncertaintyAnalysisPage = lazy(() => import('./pages/KGUncertaintyAnalysisPage'))
const KGBayesianNetworkPage = lazy(() => import('./pages/KGBayesianNetworkPage'))
const KGProbabilityCalculationPage = lazy(() => import('./pages/KGProbabilityCalculationPage'))
const KGConfidenceIntervalPage = lazy(() => import('./pages/KGConfidenceIntervalPage'))

// 用户反馈学习系统
const FeedbackSystemPage = lazy(() => import('./pages/FeedbackSystemPage'))
const FeedbackAnalyticsPage = lazy(() => import('./pages/FeedbackAnalyticsPage'))
const UserFeedbackProfilesPage = lazy(() => import('./pages/UserFeedbackProfilesPage'))
const ItemFeedbackAnalysisPage = lazy(() => import('./pages/ItemFeedbackAnalysisPage'))
const FeedbackQualityMonitorPage = lazy(() => import('./pages/FeedbackQualityMonitorPage'))

// Story 4.11 离线能力与同步机制
const OfflineCapabilityPage = lazy(() => import('./pages/OfflineCapabilityPage'))
const SyncManagementPage = lazy(() => import('./pages/SyncManagementPage'))
const ConflictResolutionPage = lazy(() => import('./pages/ConflictResolutionPage'))
const VectorClockVisualizationPage = lazy(() => import('./pages/VectorClockVisualizationPage'))
const NetworkMonitorDetailPage = lazy(() => import('./pages/NetworkMonitorDetailPage'))
const SyncEngineInternalPage = lazy(() => import('./pages/SyncEngineInternalPage'))
const ModelCacheMonitorPage = lazy(() => import('./pages/ModelCacheMonitorPage'))
const DagOrchestratorPage = lazy(() => import('./pages/DagOrchestratorPage'))
// const UnifiedEnginePage = lazy(() => import('./pages/UnifiedEnginePage'))
const UnifiedMonitorPage = lazy(() => import('./pages/UnifiedMonitorPage'))
const AiTrismPage = lazy(() => import('./pages/AiTrismPage'))
const EventDashboardPage = lazy(() => import('./pages/EventDashboardPage'))
const BanditRecommendationPage = lazy(() => import('./pages/BanditRecommendationPage'))
const QLearningPage = lazy(() => import('./pages/QLearningPage'))
const QLearningTrainingPage = lazy(() => import('./pages/QLearningTrainingPage'))
const QLearningStrategyPage = lazy(() => import('./pages/QLearningStrategyPage'))
const QLearningRecommendationPage = lazy(() => import('./pages/QLearningRecommendationPage'))
const QLearningPerformancePage = lazy(() => import('./pages/QLearningPerformancePage'))
const TabularQLearningPage = lazy(() => import('./pages/qlearning/TabularQLearningPage'))
const DQNPage = lazy(() => import('./pages/qlearning/DQNPage'))
const DQNVariantsPage = lazy(() => import('./pages/qlearning/DQNVariantsPage'))
const ExplorationStrategiesPage = lazy(() => import('./pages/qlearning/ExplorationStrategiesPage'))
const UCBStrategiesPage = lazy(() => import('./pages/qlearning/UCBStrategiesPage'))
const ThompsonSamplingPage = lazy(() => import('./pages/qlearning/ThompsonSamplingPage'))
const BasicRewardsPage = lazy(() => import('./pages/qlearning/BasicRewardsPage'))
const CompositeRewardsPage = lazy(() => import('./pages/qlearning/CompositeRewardsPage'))
const AdaptiveRewardsPage = lazy(() => import('./pages/qlearning/AdaptiveRewardsPage'))
const RewardShapingPage = lazy(() => import('./pages/qlearning/RewardShapingPage'))
const StateSpacePage = lazy(() => import('./pages/qlearning/StateSpacePage'))
const ActionSpacePage = lazy(() => import('./pages/qlearning/ActionSpacePage'))
const GridWorldPage = lazy(() => import('./pages/qlearning/GridWorldPage'))
const EnvironmentSimulatorPage = lazy(() => import('./pages/qlearning/EnvironmentSimulatorPage'))
const TrainingManagerPage = lazy(() => import('./pages/qlearning/TrainingManagerPage'))
const LearningRateSchedulerPage = lazy(() => import('./pages/qlearning/LearningRateSchedulerPage'))
const EarlyStoppingPage = lazy(() => import('./pages/qlearning/EarlyStoppingPage'))
const PerformanceTrackerPage = lazy(() => import('./pages/qlearning/PerformanceTrackerPage'))

// TensorFlow模型管理页面
const TensorFlowManagementPage = lazy(() => import('./pages/TensorFlowManagementPage'))

// 新增缺失页面（去重后）
const ConflictResolutionLearningPage = lazy(() => import('./pages/ConflictResolutionLearningPage'))
const SyncEngineLearningPage = lazy(() => import('./pages/SyncEngineLearningPage'))
const HealthComprehensivePage = lazy(() => import('./pages/HealthComprehensivePage'))
// 已移除不存在的MultimodalPageSimple
const VectorAdvancedPage = lazy(() => import('./pages/VectorAdvancedPage'))
// 已移除不存在的VectorAdvancedPageSimple
// 已移除不存在的VectorAdvancedTestPage
const VectorClockAlgorithmPage = lazy(() => import('./pages/VectorClockAlgorithmPage'))
const UnifiedEnginePageComplete = lazy(() => import('./pages/UnifiedEnginePageComplete'))
// 已移除不存在的BatchJobsPage，使用BatchJobsPageFixed
const DocumentProcessingPage = lazy(() => import('./pages/DocumentProcessingPage'))
// 已移除不存在的MultimodalPage，使用MultimodalPageComplete
const FileManagementAdvancedPage = lazy(() => import('./pages/FileManagementAdvancedPage'))
const DistributedEventsPage = lazy(() => import('./pages/DistributedEventsPage'))
const LangGraph065Page = lazy(() => import('./pages/LangGraph065Page'))
const MultimodalRagPage = lazy(() => import('./pages/MultimodalRagPage'))
const MultimodalRagManagementPage = lazy(() => import('./pages/MultimodalRagManagementPage'))
const DocumentManagementPageComplete = lazy(() => import('./pages/DocumentManagementPageComplete'))
const RealtimeMetricsManagementPage = lazy(() => import('./pages/RealtimeMetricsManagementPage'))

// 个性化引擎页面
const PersonalizationEnginePage = lazy(() => import('./pages/PersonalizationEnginePage'))
const PersonalizationMonitorPage = lazy(() => import('./pages/PersonalizationMonitorPage'))
const PersonalizationFeaturePage = lazy(() => import('./pages/PersonalizationFeaturePage'))
const PersonalizationLearningPage = lazy(() => import('./pages/PersonalizationLearningPage'))
const PersonalizationApiPage = lazy(() => import('./pages/PersonalizationApiPage'))
const PersonalizationAlertsPage = lazy(() => import('./pages/PersonalizationAlertsPage'))
const PersonalizationProductionPage = lazy(() => import('./pages/PersonalizationProductionPage'))
const PersonalizationWebSocketPage = lazy(() => import('./pages/PersonalizationWebSocketPage'))
const WebSocketManagementPage = lazy(() => import('./pages/WebSocketManagementPage'))

// 高级情感智能系统页面 (Story 11.1 & 11.2)
const EmotionRecognitionOverviewPage = lazy(() => import('./pages/EmotionRecognitionOverviewPage'))
const TextEmotionAnalysisPage = lazy(() => import('./pages/TextEmotionAnalysisPage'))
const AudioEmotionRecognitionPage = lazy(() => import('./pages/AudioEmotionRecognitionPage'))
const VisualEmotionAnalysisPage = lazy(() => import('./pages/VisualEmotionAnalysisPage'))
const MultiModalEmotionFusionPage = lazy(() => import('./pages/MultiModalEmotionFusionPage'))
const EmotionModelingPage = lazy(() => import('./pages/EmotionModelingPage'))

// 情感记忆管理系统 (Story 11.4)
const EmotionalMemoryPage = lazy(() => import('./pages/EmotionalMemoryPage'))
const EmotionalMemoryManagementPage = lazy(() => import('./pages/EmotionalMemoryManagementPage'))
const EmotionalEventAnalysisPage = lazy(() => import('./pages/EmotionalEventAnalysisPage'))

// 情感智能决策引擎 (Story 11.5)
const EmotionalIntelligenceDecisionEnginePage = lazy(() => import('./pages/EmotionalIntelligenceDecisionEnginePage'))
const EmotionalRiskAssessmentDashboardPage = lazy(() => import('./pages/EmotionalRiskAssessmentDashboardPage'))
const CrisisDetectionSupportPage = lazy(() => import('./pages/CrisisDetectionSupportPage'))
const InterventionStrategyManagementPage = lazy(() => import('./pages/InterventionStrategyManagementPage'))
const EmotionalHealthMonitoringDashboardPage = lazy(() => import('./pages/EmotionalHealthMonitoringDashboardPage'))
const DecisionHistoryAnalysisPage = lazy(() => import('./pages/DecisionHistoryAnalysisPage'))
const EmpathyResponseGeneratorPage = lazy(() => import('./pages/EmpathyResponseGeneratorPage'))

// 社交情感理解系统 (Story 11.6)
const GroupEmotionAnalysisPage = lazy(() => import('./pages/GroupEmotionAnalysisPage'))
const RelationshipDynamicsPage = lazy(() => import('./pages/RelationshipDynamicsPage'))
const SocialContextAdaptationPage = lazy(() => import('./pages/SocialContextAdaptationPage'))
const SocialEmotionalUnderstandingPage = lazy(() => import('./pages/SocialEmotionalUnderstandingPage'))
const CulturalAdaptationPage = lazy(() => import('./pages/CulturalAdaptationPage'))
const SocialIntelligenceDecisionPage = lazy(() => import('./pages/SocialIntelligenceDecisionPage'))

// A/B测试实验平台页面
const ExperimentListPage = lazy(() => import('./pages/experiments/ExperimentListPage'))

// 新增未使用API模块页面
const ServiceDiscoveryManagementPage = lazy(() => import('./pages/ServiceDiscoveryManagementPage'))
const OfflineManagementPage = lazy(() => import('./pages/OfflineManagementPage'))
const TrafficRampManagementPage = lazy(() => import('./pages/TrafficRampManagementPage'))
const LayeredExperimentsManagementPage = lazy(() => import('./pages/LayeredExperimentsManagementPage'))
const PowerAnalysisPage = lazy(() => import('./pages/PowerAnalysisPage'))
const DescriptiveStatisticsPage = lazy(() => import('./pages/DescriptiveStatisticsPage'))
const MultipleTestingCorrectionPage = lazy(() => import('./pages/MultipleTestingCorrectionPage'))
const ExperimentDashboardPage = lazy(() => import('./pages/experiments/ExperimentDashboardPage'))
const StatisticalAnalysisPage = lazy(() => import('./pages/experiments/StatisticalAnalysisPage'))
const TrafficAllocationPage = lazy(() => import('./pages/experiments/TrafficAllocationPage'))
const EventTrackingPage = lazy(() => import('./pages/experiments/EventTrackingPage'))
const ReleaseStrategyPage = lazy(() => import('./pages/experiments/ReleaseStrategyPage'))
const MonitoringAlertsPage = lazy(() => import('./pages/experiments/MonitoringAlertsPage'))
const AdvancedAlgorithmsPage = lazy(() => import('./pages/experiments/AdvancedAlgorithmsPage'))

// 行为分析系统
const BehaviorAnalyticsPage = lazy(() => import('./pages/BehaviorAnalyticsPage'))
const BehaviorAnalyticsPageEnhanced = lazy(() => import('./pages/BehaviorAnalyticsPageEnhanced'))
const EventDataManagePage = lazy(() => import('./pages/behavior-analytics/EventDataManagePage'))
const SessionManagePage = lazy(() => import('./pages/behavior-analytics/SessionManagePage'))
const ReportCenterPage = lazy(() => import('./pages/behavior-analytics/ReportCenterPage'))
const RealTimeMonitorPage = lazy(() => import('./pages/behavior-analytics/RealTimeMonitorPage'))
const DataExportPage = lazy(() => import('./pages/behavior-analytics/DataExportPage'))
const SystemConfigPage = lazy(() => import('./pages/behavior-analytics/SystemConfigPage'))

// 强化学习系统监控页面
const RLSystemDashboardPage = lazy(() => import('./pages/RLSystemDashboardPage'))
const RLPerformanceMonitorPage = lazy(() => import('./pages/RLPerformanceMonitorPage'))
const RLIntegrationTestPage = lazy(() => import('./pages/RLIntegrationTestPage'))
const RLAlertConfigPage = lazy(() => import('./pages/RLAlertConfigPage'))
const RLMetricsAnalysisPage = lazy(() => import('./pages/RLMetricsAnalysisPage'))
const RLSystemHealthPage = lazy(() => import('./pages/RLSystemHealthPage'))

// 模型评估和基准测试系统 (Story 9.4)
const ModelEvaluationOverviewPage = lazy(() => import('./pages/ModelEvaluationOverviewPage'))
const ModelEvaluationManagementPage = lazy(() => import('./pages/ModelEvaluationManagementPage'))
const MemoryManagementMonitorPage = lazy(() => import('./pages/MemoryManagementMonitorPage'))
const EvaluationEngineManagementPage = lazy(() => import('./pages/EvaluationEngineManagementPage'))
const EvaluationTasksMonitorPage = lazy(() => import('./pages/EvaluationTasksMonitorPage'))
const EvaluationReportsCenterPage = lazy(() => import('./pages/EvaluationReportsCenterPage'))
const EvaluationApiManagementPage = lazy(() => import('./pages/EvaluationApiManagementPage'))
const ModelComparisonDashboardPage = lazy(() => import('./pages/ModelComparisonDashboardPage'))
const BenchmarkSuiteManagementPage = lazy(() => import('./pages/BenchmarkSuiteManagementPage'))
const BenchmarkGlueManagementPage = lazy(() => import('./pages/BenchmarkGlueManagementPage'))
const BenchmarkSupergluePage = lazy(() => import('./pages/BenchmarkSupergluePage'))
const BenchmarkMmluPage = lazy(() => import('./pages/BenchmarkMmluPage'))
const EvaluationMetricsConfigPage = lazy(() => import('./pages/EvaluationMetricsConfigPage'))
// 模型压缩和量化工具 (Story 9.2)
const ModelCompressionOverviewPage = lazy(() => import('./pages/ModelCompressionOverviewPage'))
const QuantizationManagerPage = lazy(() => import('./pages/QuantizationManagerPage'))
const KnowledgeDistillationPage = lazy(() => import('./pages/KnowledgeDistillationPage'))
const ModelPruningPage = lazy(() => import('./pages/ModelPruningPage'))
const CompressionPipelinePage = lazy(() => import('./pages/CompressionPipelinePage'))
const ModelCompressionEvaluationPage = lazy(() => import('./pages/ModelCompressionEvaluationPage'))
const ModelPerformanceBenchmarkPage = lazy(() => import('./pages/ModelPerformanceBenchmarkPage'))
const CompressionStrategyRecommendationPage = lazy(() => import('./pages/CompressionStrategyRecommendationPage'))

// 模型服务部署平台 (Story 9.6)
const ModelRegistryPage = lazy(() => import('./pages/ModelRegistryPage'))
const ModelInferencePage = lazy(() => import('./pages/ModelInferencePage'))
const ModelDeploymentPage = lazy(() => import('./pages/ModelDeploymentPage'))
const ModelMonitoringPage = lazy(() => import('./pages/ModelMonitoringPage'))
const ModelServiceManagementPage = lazy(() => import('./pages/ModelServiceManagementPage'))
const OnlineLearningPage = lazy(() => import('./pages/OnlineLearningPage'))

// 自动化超参数优化系统 (Story 9.3)
const HyperparameterOptimizationPage = lazy(() => import('./pages/HyperparameterOptimizationPage'))
const HyperparameterOptimizationPageEnhanced = lazy(() => import('./pages/HyperparameterOptimizationPageEnhanced'))
const HyperparameterExperimentsPage = lazy(() => import('./pages/HyperparameterExperimentsPage'))
const HyperparameterAlgorithmsPage = lazy(() => import('./pages/HyperparameterAlgorithmsPage'))
const HyperparameterVisualizationsPage = lazy(() => import('./pages/HyperparameterVisualizationsPage'))
const HyperparameterMonitoringPage = lazy(() => import('./pages/HyperparameterMonitoringPage'))
const HyperparameterResourcesPage = lazy(() => import('./pages/HyperparameterResourcesPage'))
const HyperparameterSchedulerPage = lazy(() => import('./pages/HyperparameterSchedulerPage'))
const HyperparameterReportsPage = lazy(() => import('./pages/HyperparameterReportsPage'))

// 训练数据管理系统 (Story 9.5)
const TrainingDataManagementPage = lazy(() => import('./pages/TrainingDataManagementPage'))
const TrainingDataManagementPageEnhanced = lazy(() => import('./pages/TrainingDataManagementPageEnhanced'))
const DataSourceManagementPage = lazy(() => import('./pages/DataSourceManagementPage'))
const DataCollectionPage = lazy(() => import('./pages/DataCollectionPage'))
const DataPreprocessingPage = lazy(() => import('./pages/DataPreprocessingPage'))
const DataAnnotationManagementPage = lazy(() => import('./pages/DataAnnotationManagementPage'))
const AnnotationTasksPage = lazy(() => import('./pages/AnnotationTasksPage'))
const AnnotationQualityControlPage = lazy(() => import('./pages/AnnotationQualityControlPage'))
const DataVersionManagementPage = lazy(() => import('./pages/DataVersionManagementPage'))

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

const App: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false)
  const [siderBroken, setSiderBroken] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const hideSider = location.pathname === '/multi-step-reasoning'

  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/history') return 'chat-history'
    if (path === '/multi-agent') return 'multi-agent'
    if (path === '/supervisor') return 'supervisor'
    if (path === '/async-agents') return 'async-agents'
    if (path === '/agent-interface') return 'agent-interface'
    if (path === '/agent-cluster-management') return 'agent-cluster-management'
    if (path === '/agent-cluster-management-enhanced') return 'agent-cluster-management-enhanced'
    
    // 智能代理服务发现系统 (Story 10.1)
    if (path === '/service-discovery-overview') return 'service-discovery-overview'
    if (path === '/agent-registry') return 'agent-registry-management'
    if (path === '/service-routing') return 'service-routing-management'
    if (path === '/load-balancer-config') return 'load-balancer-config'
    if (path === '/service-health-monitor') return 'service-health-monitor'
    if (path === '/service-cluster-management') return 'service-cluster-management'
    if (path === '/service-performance-dashboard') return 'service-performance-dashboard'
    if (path === '/service-config-management') return 'service-config-management'
    
    if (path === '/rag') return 'rag'
    if (path === '/agentic-rag') return 'agentic-rag'
    if (path === '/graphrag') return 'graphrag'
    if (path === '/hybrid-search') return 'hybrid-search'
    if (path === '/multimodal') return 'multimodal'
    if (path === '/file-management') return 'file-management'
    if (path === '/workflows') return 'workflows'
    
    // 分布式任务协调引擎路径映射
    if (path === '/distributed-task-coordination') return 'distributed-task-coordination'
    if (path === '/task-decomposer') return 'task-decomposer'
    if (path === '/intelligent-assigner') return 'intelligent-assigner'
    if (path === '/raft-consensus') return 'raft-consensus'
    if (path === '/distributed-state-manager') return 'distributed-state-manager'
    if (path === '/conflict-resolver') return 'conflict-resolver'
    if (path === '/distributed-task-monitor') return 'distributed-task-monitor'
    if (path === '/distributed-task-system-status') return 'distributed-task-system-status'
    if (path === '/distributed-task-management-enhanced') return 'distributed-task-management-enhanced'
    
    if (path === '/dag-orchestrator') return 'dag-orchestrator'
    if (path === '/flow-control') return 'flow-control'
    if (path === '/streaming') return 'streaming'
    if (path === '/batch') return 'batch-jobs'
    if (path === '/unified-engine') return 'unified-engine'
    if (path === '/ai-trism') return 'ai-trism'
    if (path === '/distributed-events') return 'distributed-events'
    if (path === '/health') return 'health'
    if (path === '/performance') return 'performance'
    if (path === '/events') return 'events'
    if (path === '/monitor') return 'monitor'
    if (path === '/monitoring-dashboard') return 'monitoring-dashboard'
    if (path === '/pgvector') return 'pgvector'
    if (path === '/cache') return 'cache-monitor'
    if (path === '/vector-advanced') return 'vector-advanced'
    if (path === '/mcp-tools') return 'mcp-tools'
    if (path === '/enterprise') return 'enterprise'
    if (path === '/enterprise-config') return 'enterprise-config'
    if (path === '/security-management') return 'security-management'
    if (path === '/debug') return 'debug'
    if (path === '/structured-errors') return 'structured-errors'
    if (path === '/test-coverage') return 'test-coverage'
    if (path === '/test') return 'test-integration'
    if (path === '/test-suite') return 'test-suite'
    if (path === '/document-processing') return 'document-processing'
    if (path === '/qlearning-performance-optimization') return 'qlearning-performance-optimization'
    if (path === '/auth-management') return 'auth-management'
    if (path === '/security') return 'security'
    if (path === '/security-management') return 'security-management'
    if (path === '/security-audit') return 'security-audit'
    if (path === '/distributed-security-monitor') return 'distributed-security-monitor'
    if (path === '/langgraph-features') return 'langgraph-features'
    if (path === '/memory-hierarchy') return 'memory-hierarchy'
    if (path === '/memory-recall') return 'memory-recall'
    if (path === '/memory-analytics') return 'memory-analytics'
    if (path === '/reasoning') return 'reasoning'
    if (path === '/multi-step-reasoning') return 'multi-step-reasoning'
    if (path === '/explainable-ai') return 'explainable-ai'
    if (path === '/targeting-rules') return 'targeting-rules'
    if (path === '/offline') return 'offline'
    if (path === '/sync') return 'sync-management'
    if (path === '/conflicts') return 'conflict-resolution'
    if (path === '/vector-clock') return 'vector-clock-viz'
    if (path === '/network-monitor') return 'network-monitor-detail'
    if (path === '/sync-engine') return 'sync-engine-internal'
    if (path === '/model-cache') return 'model-cache-monitor'
    if (path === '/assignment-cache') return 'assignment-cache'
    if (path === '/bandit-recommendation') return 'bandit-recommendation'
    if (path === '/qlearning') return 'qlearning-dashboard'
    if (path === '/qlearning-training') return 'qlearning-training'
    if (path === '/qlearning-strategy') return 'qlearning-strategy'
    if (path === '/qlearning-recommendation') return 'qlearning-recommendation'
    if (path === '/qlearning-performance') return 'qlearning-performance'
    if (path === '/qlearning/tabular') return 'qlearning-tabular'
    if (path === '/qlearning/dqn') return 'qlearning-dqn'
    if (path === '/qlearning/variants') return 'qlearning-variants'
    if (path === '/exploration-strategies') return 'exploration-strategies'
    if (path === '/ucb-strategies') return 'ucb-strategies'
    if (path === '/thompson-sampling') return 'thompson-sampling'
    if (path === '/adaptive-exploration') return 'adaptive-exploration'
    if (path === '/basic-rewards') return 'basic-rewards'
    if (path === '/composite-rewards') return 'composite-rewards'
    if (path === '/adaptive-rewards') return 'adaptive-rewards'
    if (path === '/reward-shaping') return 'reward-shaping'
    if (path === '/state-space') return 'state-space'
    if (path === '/action-space') return 'action-space'
    if (path === '/environment-simulator') return 'environment-simulator'
    if (path === '/grid-world') return 'grid-world'
    if (path === '/training-manager') return 'training-manager'
    if (path === '/learning-rate-scheduler') return 'learning-rate-scheduler'
    if (path === '/early-stopping') return 'early-stopping'
    if (path === '/performance-tracker') return 'performance-tracker'
    if (path === '/feedback-system') return 'feedback-system'
    if (path === '/feedback-analytics') return 'feedback-analytics'
    if (path === '/user-feedback-profiles') return 'user-feedback-profiles'
    if (path === '/item-feedback-analysis') return 'item-feedback-analysis'
    if (path === '/feedback-quality-monitor') return 'feedback-quality-monitor'
    
    // 行为分析系统
    if (path === '/behavior-analytics') return 'behavior-analytics'
    if (path === '/behavior-analytics/events') return 'behavior-analytics-events'
    if (path === '/behavior-analytics/sessions') return 'behavior-analytics-sessions'
    if (path === '/behavior-analytics/reports') return 'behavior-analytics-reports'
    if (path === '/behavior-analytics/realtime') return 'behavior-analytics-realtime'
    if (path === '/behavior-analytics/export') return 'behavior-analytics-export'
    if (path === '/behavior-analytics/config') return 'behavior-analytics-config'
    
    // 缺失页面路径映射
    if (path === '/conflict-resolution-learning') return 'conflict-resolution-learning'
    if (path === '/sync-engine-learning') return 'sync-engine-learning'
    if (path === '/health-comprehensive') return 'health-comprehensive'
    // 已移除multimodal-simple路径映射
    // 已移除vector-advanced-simple路径映射
    // 已移除vector-advanced-test路径映射
    if (path === '/vector-clock-algorithm') return 'vector-clock-algorithm'
    if (path === '/unified-engine-complete') return 'unified-engine-complete'
    if (path === '/batch-jobs') return 'batch-jobs-basic'
    if (path === '/document-processing-simple') return 'document-processing-simple'
    if (path === '/security') return 'security-basic'
    if (path === '/multimodal-basic') return 'multimodal-basic'
    if (path === '/file-management-advanced') return 'file-management-advanced'
    if (path === '/distributed-events') return 'distributed-events-system'
    if (path === '/langgraph-065') return 'langgraph-065'
    if (path === '/multimodal-rag') return 'multimodal-rag-system'
    if (path === '/multimodal-rag-management') return 'multimodal-rag-management'
    if (path === '/document-management-complete') return 'document-management-complete'
    if (path === '/realtime-metrics-management') return 'realtime-metrics-management'
    
    // A/B测试实验平台路径映射
    if (path === '/experiments') return 'experiment-list'
    if (path === '/experiments/dashboard') return 'experiment-dashboard'
    if (path === '/experiments/enhanced-analysis') return 'enhanced-experiment-analysis'
    if (path === '/experiments/statistical-analysis') return 'statistical-analysis'
    if (path === '/experiments/traffic-allocation') return 'traffic-allocation'
    if (path === '/experiments/event-tracking') return 'event-tracking'
    if (path === '/experiments/traffic-ramp') return 'traffic-ramp-management'
    if (path === '/experiments/power-analysis') return 'power-analysis'
    if (path === '/experiments/multiple-testing') return 'multiple-testing-correction'
    if (path === '/experiments/layered-experiments') return 'layered-experiments-management'

    // 服务发现与离线管理路径映射
    if (path === '/service-discovery-management') return 'service-discovery-management'
    if (path === '/offline-management') return 'offline-management'
    if (path === '/experiments/release-strategy') return 'release-strategy'
    if (path === '/experiments/monitoring-alerts') return 'monitoring-alerts'
    if (path === '/experiments/advanced-algorithms') return 'advanced-algorithms'
    
    // 强化学习系统监控路径映射
    if (path === '/rl-system-dashboard') return 'rl-system-dashboard'
    if (path === '/rl-performance-monitor') return 'rl-performance-monitor'
    if (path === '/rl-integration-test') return 'rl-integration-test'
    if (path === '/rl-alert-config') return 'rl-alert-config'
    if (path === '/rl-metrics-analysis') return 'rl-metrics-analysis'
    if (path === '/rl-system-health') return 'rl-system-health'
    
    // 个性化引擎路径映射
    if (path === '/personalization-engine') return 'personalization-engine'
    if (path === '/personalization-monitor') return 'personalization-monitor'
    if (path === '/personalization-features') return 'personalization-features'
    if (path === '/personalization-learning') return 'personalization-learning'
    if (path === '/personalization-api') return 'personalization-api'
    if (path === '/personalization-alerts') return 'personalization-alerts'
    if (path === '/personalization-production') return 'personalization-production'
    if (path === '/personalization-websocket') return 'personalization-websocket'
    if (path === '/websocket-management') return 'websocket-management'
    
    // 高级情感智能系统路径映射
    if (path === '/emotion-recognition-overview') return 'emotion-recognition-overview'
    if (path === '/text-emotion-analysis') return 'text-emotion-analysis'
    if (path === '/audio-emotion-recognition') return 'audio-emotion-recognition'
    if (path === '/visual-emotion-analysis') return 'visual-emotion-analysis'
    if (path === '/multimodal-emotion-fusion') return 'multimodal-emotion-fusion'
    if (path === '/emotion-modeling') return 'emotion-modeling'
    if (path === '/empathy-response-generator') return 'empathy-response-generator'
    
    // 社交情感理解系统 (Story 11.6) 路径映射
    if (path === '/group-emotion-analysis') return 'group-emotion-analysis'
    if (path === '/relationship-dynamics') return 'relationship-dynamics'
    if (path === '/social-context-adaptation') return 'social-context-adaptation'
    if (path === '/cultural-adaptation') return 'cultural-adaptation'
    if (path === '/social-intelligence-decision') return 'social-intelligence-decision'
    
    // 动态知识图谱存储系统 (Story 8.2) 路径映射
    if (path === '/kg-entity-management') return 'kg-entity-management'
    if (path === '/kg-relation-management') return 'kg-relation-management'
    if (path === '/kg-graph-query') return 'kg-graph-query'
    if (path === '/kg-incremental-update') return 'kg-incremental-update'
    if (path === '/kg-quality-assessment') return 'kg-quality-assessment'
    if (path === '/kg-performance-monitor') return 'kg-performance-monitor'
    if (path === '/kg-schema-management') return 'kg-schema-management'
    if (path === '/kg-data-migration') return 'kg-data-migration'
    
    // 知识图推理引擎 (Story 8.3) 路径映射
    // 混合推理引擎
    if (path === '/kg-reasoning-dashboard') return 'kg-reasoning-dashboard'
    if (path === '/kg-reasoning-query') return 'kg-reasoning-query'
    if (path === '/kg-reasoning-batch') return 'kg-reasoning-batch'
    if (path === '/kg-reasoning-performance') return 'kg-reasoning-performance'
    if (path === '/kg-reasoning-strategy') return 'kg-reasoning-strategy'
    if (path === '/kg-reasoning-explanation') return 'kg-reasoning-explanation'
    
    // 规则推理引擎
    if (path === '/kg-rule-management') return 'kg-rule-management'
    if (path === '/kg-rule-execution') return 'kg-rule-execution'
    if (path === '/kg-rule-validation') return 'kg-rule-validation'
    if (path === '/kg-rule-conflict') return 'kg-rule-conflict'
    
    // 嵌入推理引擎
    if (path === '/kg-embedding-models') return 'kg-embedding-models'
    if (path === '/kg-embedding-training') return 'kg-embedding-training'
    if (path === '/kg-embedding-similarity') return 'kg-embedding-similarity'
    if (path === '/kg-embedding-index') return 'kg-embedding-index'
    
    // 路径推理引擎
    if (path === '/kg-path-discovery') return 'kg-path-discovery'
    if (path === '/kg-path-analysis') return 'kg-path-analysis'
    if (path === '/kg-path-optimization') return 'kg-path-optimization'
    if (path === '/kg-path-confidence') return 'kg-path-confidence'
    
    // 不确定性推理
    if (path === '/kg-uncertainty-analysis') return 'kg-uncertainty-analysis'
    if (path === '/kg-bayesian-network') return 'kg-bayesian-network'
    if (path === '/kg-probability-calculation') return 'kg-probability-calculation'
    if (path === '/kg-confidence-interval') return 'kg-confidence-interval'
    
    // LoRA/QLoRA微调框架路径处理
    if (path === '/fine-tuning-jobs') return 'fine-tuning-jobs'
    if (path === '/fine-tuning') return 'fine-tuning-management'
    if (path === '/fine-tuning-enhanced') return 'fine-tuning-enhanced'
    if (path === '/lora-training-overview') return 'lora-training-overview'
    if (path === '/lora-config-templates') return 'lora-config-templates'
    if (path === '/lora-model-adapters') return 'lora-model-adapters'
    if (path === '/lora-performance-monitor') return 'lora-performance-monitor'
    if (path === '/qlora-training-overview') return 'qlora-training-overview'
    if (path === '/qlora-quantization-config') return 'qlora-quantization-config'
    if (path === '/qlora-memory-optimization') return 'qlora-memory-optimization'
    if (path === '/qlora-inference-optimization') return 'qlora-inference-optimization'
    if (path === '/distributed-training-overview') return 'distributed-training-overview'
    if (path === '/auto-scaling-management') return 'auto-scaling-management'
    if (path === '/risk-assessment-dashboard') return 'risk-assessment-dashboard'
    if (path === '/statistical-analysis-dashboard') return 'statistical-analysis-dashboard'
    if (path === '/deepspeed-configuration') return 'deepspeed-configuration'
    if (path === '/multi-gpu-monitoring') return 'multi-gpu-monitoring'
    if (path === '/training-synchronization') return 'training-synchronization'
    if (path === '/training-dashboard') return 'training-dashboard'
    if (path === '/training-metrics') return 'training-metrics'
    if (path === '/training-anomaly-detection') return 'training-anomaly-detection'
    if (path === '/anomaly-detection') return 'anomaly-detection'
    if (path === '/auto-scaling') return 'auto-scaling'
    if (path === '/batch-operations') return 'batch-operations'  // Could also be 'batch-jobs-management' depending on context
    if (path === '/training-reports') return 'training-reports'
    if (path === '/supported-models') return 'supported-models'
    if (path === '/model-checkpoints') return 'model-checkpoints'
    if (path === '/model-performance-comparison') return 'model-performance-comparison'
    if (path === '/model-deployment') return 'model-deployment'
    // 训练数据管理系统路径匹配
    if (path === '/training-data-management') return 'training-data-overview'
    if (path === '/training-data-enhanced') return 'training-data-enhanced'
    if (path === '/data-sources') return 'data-sources'
    if (path === '/data-collection') return 'data-collection'
    if (path === '/data-preprocessing') return 'data-preprocessing'
    if (path === '/data-annotation') return 'data-annotation'
    if (path === '/annotation-tasks') return 'annotation-tasks'
    if (path === '/annotation-quality') return 'annotation-quality'
    if (path === '/data-versioning') return 'data-versioning'
    if (path === '/data-version-comparison') return 'data-version-comparison'
    if (path === '/data-export') return 'data-export'
    if (path === '/data-statistics') return 'data-statistics'
    if (path === '/quality-metrics') return 'quality-metrics'
    
    // 模型压缩和量化工具路径匹配
    if (path === '/model-compression-overview') return 'model-compression-overview'
    if (path === '/quantization-manager') return 'quantization-manager'
    if (path === '/knowledge-distillation') return 'knowledge-distillation'
    if (path === '/model-pruning') return 'model-pruning'
    if (path === '/compression-pipeline') return 'compression-pipeline'
    if (path === '/compression-evaluation') return 'compression-evaluation'
    if (path === '/performance-benchmark') return 'performance-benchmark'
    if (path === '/strategy-recommendation') return 'strategy-recommendation'
    
    // 自动化超参数优化系统路径匹配
    if (path === '/hyperparameter-optimization') return 'hyperparameter-optimization-dashboard'
    if (path === '/hyperparameter-optimization-enhanced') return 'hyperparameter-optimization-enhanced'
    if (path === '/hyperparameter-experiments') return 'hyperparameter-experiments'
    if (path === '/hyperparameter-algorithms') return 'hyperparameter-algorithms'
    if (path === '/hyperparameter-visualizations') return 'hyperparameter-visualizations'
    if (path === '/hyperparameter-monitoring') return 'hyperparameter-monitoring'
    if (path === '/hyperparameter-resources') return 'hyperparameter-resources'
    if (path === '/hyperparameter-scheduler') return 'hyperparameter-scheduler'
    if (path === '/hyperparameter-reports') return 'hyperparameter-reports'
    
    // 新增缺失页面路径匹配
    if (path === '/testing-management') return 'testing-management'
    if (path === '/hypothesis-testing') return 'hypothesis-testing'
    if (path === '/emotional-memory-management') return 'emotional-memory-management'
    
    return 'chat'
  }

  const menuItems = [
    // 🤖 智能体系统
    {
      key: 'ai-agents-group',
      label: '🤖 智能体系统',
      type: 'group' as const,
    },
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '单代理对话 (React Agent)',
    },
    {
      key: 'chat-history',
      icon: <HistoryOutlined />,
      label: '历史记录',
    },
    {
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多代理协作 (AutoGen v0.4)',
    },
    {
      key: 'tensorflow-qlearning',
      icon: <RobotOutlined />,
      label: 'TensorFlow Q学习管理',
    },
    {
      key: 'testing-management',
      icon: <BugOutlined />,
      label: '测试管理系统',
    },
    {
      key: 'hypothesis-testing',
      icon: <FunctionOutlined />,
      label: '假设检验统计',
    },
    {
      key: 'supervisor',
      icon: <ControlOutlined />,
      label: '监督者编排 (Supervisor)',
    },
    {
      key: 'async-agents',
      icon: <ThunderboltOutlined />,
      label: '异步事件驱动 (Event-Driven)',
    },
    {
      key: 'agent-interface',
      icon: <ApiOutlined />,
      label: '代理接口管理 (Interface)',
    },
    {
      key: 'agent-cluster-management',
      icon: <ClusterOutlined />,
      label: '智能体集群管理平台',
    },
    {
      key: 'agent-cluster-management-enhanced',
      icon: <ThunderboltOutlined />,
      label: '智能集群管理平台(增强)',
    },

    // 🚀 增强版页面
    {
      key: 'enhanced-pages-group',
      label: '🚀 增强版功能展示',
      type: 'group' as const,
    },
    {
      key: 'multi-agent-enhanced',
      icon: <TeamOutlined />,
      label: '多智能体协作系统(增强版)',
    },
    {
      key: 'rag-enhanced',
      icon: <FileTextOutlined />,
      label: 'RAG检索增强生成(增强版)',
    },
    {
      key: 'experiments-platform',
      icon: <ExperimentOutlined />,
      label: 'A/B测试实验平台',
    },
    {
      key: 'workflow-management',
      icon: <BranchesOutlined />,
      label: '工作流管理系统',
    },

    // 🌐 智能代理服务发现系统 (Story 10.1)
    {
      key: 'service-discovery-group',
      label: '🌐 智能代理服务发现系统',
      type: 'group' as const,
    },
    {
      key: 'intelligent-agent-service-discovery-system',
      icon: <GlobalOutlined />,
      label: '🌐 智能代理服务发现系统',
      children: [
        {
          key: 'service-discovery-overview',
          icon: <GlobalOutlined />,
          label: '服务发现总览',
        },
        {
          key: 'agent-registry-management', 
          icon: <DatabaseOutlined />,
          label: 'Agent注册管理',
        },
        {
          key: 'service-routing-management',
          icon: <ShareAltOutlined />,
          label: '服务路由管理',
        },
        {
          key: 'load-balancer-config',
          icon: <ClusterOutlined />,
          label: '负载均衡配置',
        },
        {
          key: 'service-health-monitor',
          icon: <HeartOutlined />,
          label: '服务健康监控',
        },
        {
          key: 'service-cluster-management',
          icon: <CloudServerOutlined />,
          label: '服务集群管理',
        },
        {
          key: 'service-performance-dashboard',
          icon: <DashboardOutlined />,
          label: '服务性能仪表板',
        },
        {
          key: 'service-config-management',
          icon: <SettingOutlined />,
          label: '服务配置管理',
        },
        {
          key: 'service-discovery-management',
          icon: <ApiOutlined />,
          label: '服务发现管理中心',
        },
        {
          key: 'offline-management',
          icon: <SyncOutlined />,
          label: '离线管理',
        },
      ],
    },

    // 🔍 智能检索引擎
    {
      key: 'retrieval-group',
      label: '🔍 智能检索引擎',
      type: 'group' as const,
    },
    {
      key: 'rag',
      icon: <SearchOutlined />,
      label: '基础RAG检索 (Vector Search)',
    },
    {
      key: 'agentic-rag',
      icon: <RobotOutlined />,
      label: 'Agentic RAG (智能检索)',
    },
    {
      key: 'graphrag',
      icon: <NodeIndexOutlined />,
      label: 'GraphRAG (图谱增强检索)',
    },
    {
      key: 'hybrid-search',
      icon: <DatabaseOutlined />,
      label: '混合检索 (pgvector + Qdrant)',
    },

    // 🧠 强化学习系统
    {
      key: 'reinforcement-learning-group',
      label: '🧠 强化学习系统',
      type: 'group' as const,
    },
    {
      key: 'qlearning',
      icon: <ThunderboltOutlined />,
      label: 'Q-Learning算法家族',
      children: [
        {
          key: 'qlearning-dashboard',
          icon: <DashboardOutlined />,
          label: '算法总览',
        },
        {
          key: 'qlearning-training',
          icon: <PlayCircleOutlined />,
          label: '训练监控',
        },
        {
          key: 'qlearning-strategy',
          icon: <BulbOutlined />,
          label: '策略推理',
        },
        {
          key: 'qlearning-recommendation',
          icon: <ExperimentOutlined />,
          label: '混合推荐',
        },
        {
          key: 'qlearning-performance',
          icon: <MonitorOutlined />,
          label: '性能分析',
        },
        {
          key: 'qlearning-performance-optimization',
          icon: <ThunderboltOutlined />,
          label: 'GPU性能优化中心',
        },
        {
          key: 'qlearning-tabular',
          icon: <DatabaseOutlined />,
          label: '表格Q-Learning',
        },
        {
          key: 'qlearning-dqn',
          icon: <RobotOutlined />,
          label: 'Deep Q-Network (DQN)',
        },
        {
          key: 'qlearning-variants',
          icon: <ExperimentOutlined />,
          label: 'DQN变体 (Double/Dueling)',
        },
      ],
    },
    {
      key: 'rl-strategies',
      icon: <SettingOutlined />,
      label: '探索策略系统',
      children: [
        {
          key: 'exploration-strategies',
          icon: <SearchOutlined />,
          label: 'Epsilon-Greedy系列',
        },
        {
          key: 'ucb-strategies',
          icon: <LineChartOutlined />,
          label: 'Upper Confidence Bound',
        },
        {
          key: 'thompson-sampling',
          icon: <ExperimentOutlined />,
          label: 'Thompson Sampling',
        },
        {
          key: 'adaptive-exploration',
          icon: <ControlOutlined />,
          label: '自适应探索策略',
        },
      ],
    },
    {
      key: 'rl-rewards',
      icon: <CheckCircleOutlined />,
      label: '奖励函数系统',
      children: [
        {
          key: 'basic-rewards',
          icon: <ThunderboltOutlined />,
          label: '基础奖励函数',
        },
        {
          key: 'composite-rewards',
          icon: <NodeIndexOutlined />,
          label: '复合奖励系统',
        },
        {
          key: 'adaptive-rewards',
          icon: <ControlOutlined />,
          label: '自适应奖励调整',
        },
        {
          key: 'reward-shaping',
          icon: <BulbOutlined />,
          label: '奖励塑形技术',
        },
      ],
    },
    {
      key: 'rl-environment',
      icon: <CloudServerOutlined />,
      label: '环境建模系统',
      children: [
        {
          key: 'state-space',
          icon: <DatabaseOutlined />,
          label: '状态空间设计',
        },
        {
          key: 'action-space',
          icon: <ApiOutlined />,
          label: '动作空间定义',
        },
        {
          key: 'environment-simulator',
          icon: <MonitorOutlined />,
          label: '环境模拟器',
        },
        {
          key: 'grid-world',
          icon: <DashboardOutlined />,
          label: 'GridWorld环境',
        },
      ],
    },
    {
      key: 'rl-training',
      icon: <PlayCircleOutlined />,
      label: '训练管理系统',
      children: [
        {
          key: 'training-manager',
          icon: <ControlOutlined />,
          label: '训练调度管理',
        },
        {
          key: 'learning-rate-scheduler',
          icon: <LineChartOutlined />,
          label: '学习率调度器',
        },
        {
          key: 'early-stopping',
          icon: <CheckCircleOutlined />,
          label: '早停机制',
        },
        {
          key: 'performance-tracker',
          icon: <MonitorOutlined />,
          label: '性能追踪器',
        },
      ],
    },

    // ❤️ 用户反馈学习系统
    {
      key: 'feedback-group',
      label: '❤️ 用户反馈学习系统',
      type: 'group' as const,
    },
    {
      key: 'feedback-system',
      icon: <HeartOutlined />,
      label: '反馈系统总览',
    },
    {
      key: 'feedback-analytics',
      icon: <LineChartOutlined />,
      label: '反馈数据分析',
    },
    {
      key: 'user-feedback-profiles',
      icon: <UserOutlined />,
      label: '用户反馈档案',
    },
    {
      key: 'item-feedback-analysis',
      icon: <TrophyOutlined />,
      label: '推荐项分析',
    },
    {
      key: 'feedback-quality-monitor',
      icon: <EyeOutlined />,
      label: '反馈质量监控',
    },

    // 📈 智能行为分析系统
    {
      key: 'behavior-analytics-group',
      label: '📈 智能行为分析系统',
      type: 'group' as const,
    },
    {
      key: 'behavior-analytics',
      icon: <BarChartOutlined />,
      label: '行为分析总览',
    },
    {
      key: 'behavior-analytics-events',
      icon: <DatabaseOutlined />,
      label: '事件数据管理',
    },
    {
      key: 'behavior-analytics-sessions',
      icon: <UserOutlined />,
      label: '会话管理中心',
    },
    {
      key: 'behavior-analytics-reports',
      icon: <FileTextOutlined />,
      label: '报告生成中心',
    },
    {
      key: 'behavior-analytics-realtime',
      icon: <MonitorOutlined />,
      label: '实时监控面板',
    },
    {
      key: 'behavior-analytics-export',
      icon: <ShareAltOutlined />,
      label: '数据导出工具',
    },
    {
      key: 'behavior-analytics-config',
      icon: <SettingOutlined />,
      label: '系统配置管理',
    },

    // 📊 强化学习系统监控
    {
      key: 'rl-monitoring-group',
      label: '📊 强化学习系统监控',
      type: 'group' as const,
    },
    {
      key: 'rl-system-dashboard',
      icon: <DashboardOutlined />,
      label: 'RL系统仪表板',
    },
    {
      key: 'rl-performance-monitor',
      icon: <MonitorOutlined />,
      label: 'RL性能监控',
    },
    {
      key: 'rl-integration-test',
      icon: <ExperimentOutlined />,
      label: 'RL集成测试',
    },
    {
      key: 'rl-alert-config',
      icon: <BellOutlined />,
      label: 'RL告警配置',
    },
    {
      key: 'rl-metrics-analysis',
      icon: <BarChartOutlined />,
      label: 'RL指标分析',
    },
    {
      key: 'rl-system-health',
      icon: <HeartOutlined />,
      label: 'RL系统健康监控',
    },

    // 🧠 推理引擎
    {
      key: 'reasoning-group',
      label: '🧠 推理引擎',
      type: 'group' as const,
    },
    {
      key: 'reasoning',
      icon: <BulbOutlined />,
      label: '链式推理 (CoT Reasoning)',
    },
    {
      key: 'multi-step-reasoning',
      icon: <NodeIndexOutlined />,
      label: '多步推理工作流 (DAG)',
    },
    {
      key: 'explainable-ai',
      icon: <BulbOutlined />,
      label: '可解释AI决策 (XAI)',
    },
    {
      key: 'targeting-rules',
      icon: <TagsOutlined />,
      label: '定向规则管理',
    },

    // 🧠 知识图推理引擎 (Story 8.3)
    {
      key: 'kg-reasoning-group',
      label: '🧠 知识图推理引擎 (Story 8.3)',
      type: 'group' as const,
    },
    {
      key: 'kg-reasoning-engine',
      icon: <ThunderboltOutlined />,
      label: '混合推理引擎',
      children: [
        {
          key: 'kg-reasoning-dashboard',
          icon: <DashboardOutlined />,
          label: '推理引擎总览',
        },
        {
          key: 'kg-reasoning-query',
          icon: <SearchOutlined />,
          label: '推理查询中心',
        },
        {
          key: 'kg-reasoning-batch',
          icon: <CloudServerOutlined />,
          label: '批量推理处理',
        },
        {
          key: 'kg-reasoning-performance',
          icon: <MonitorOutlined />,
          label: '推理性能监控',
        },
        {
          key: 'kg-reasoning-strategy',
          icon: <SettingOutlined />,
          label: '推理策略配置',
        },
        {
          key: 'kg-reasoning-explanation',
          icon: <BulbOutlined />,
          label: '推理结果解释',
        },
      ],
    },
    {
      key: 'kg-rule-engine',
      icon: <RobotOutlined />,
      label: '规则推理引擎',
      children: [
        {
          key: 'kg-rule-management',
          icon: <DatabaseOutlined />,
          label: '规则库管理',
        },
        {
          key: 'kg-rule-execution',
          icon: <PlayCircleOutlined />,
          label: '规则执行监控',
        },
        {
          key: 'kg-rule-validation',
          icon: <CheckCircleOutlined />,
          label: '规则验证测试',
        },
        {
          key: 'kg-rule-conflict',
          icon: <ExceptionOutlined />,
          label: '规则冲突检测',
        },
      ],
    },
    {
      key: 'kg-embedding-engine',
      icon: <NodeIndexOutlined />,
      label: '嵌入推理引擎',
      children: [
        {
          key: 'kg-embedding-models',
          icon: <RobotOutlined />,
          label: '嵌入模型管理',
        },
        {
          key: 'kg-embedding-training',
          icon: <PlayCircleOutlined />,
          label: '模型训练监控',
        },
        {
          key: 'kg-embedding-similarity',
          icon: <ShareAltOutlined />,
          label: '相似度计算',
        },
        {
          key: 'kg-embedding-index',
          icon: <DatabaseOutlined />,
          label: '向量索引管理',
        },
      ],
    },
    {
      key: 'kg-path-reasoning',
      icon: <ShareAltOutlined />,
      label: '路径推理引擎',
      children: [
        {
          key: 'kg-path-discovery',
          icon: <SearchOutlined />,
          label: '路径发现中心',
        },
        {
          key: 'kg-path-analysis',
          icon: <LineChartOutlined />,
          label: '路径分析可视化',
        },
        {
          key: 'kg-path-optimization',
          icon: <ThunderboltOutlined />,
          label: '路径优化算法',
        },
        {
          key: 'kg-path-confidence',
          icon: <TrophyOutlined />,
          label: '置信度计算',
        },
      ],
    },
    {
      key: 'kg-uncertainty-reasoning',
      icon: <ExperimentOutlined />,
      label: '不确定性推理',
      children: [
        {
          key: 'kg-uncertainty-analysis',
          icon: <LineChartOutlined />,
          label: '不确定性分析',
        },
        {
          key: 'kg-bayesian-network',
          icon: <NodeIndexOutlined />,
          label: '贝叶斯网络',
        },
        {
          key: 'kg-probability-calculation',
          icon: <ExperimentOutlined />,
          label: '概率推理计算',
        },
        {
          key: 'kg-confidence-interval',
          icon: <BarChartOutlined />,
          label: '置信区间估计',
        },
      ],
    },

    // 🎯 推荐算法引擎
    {
      key: 'recommendation-group',
      label: '🎯 推荐算法引擎',
      type: 'group' as const,
    },
    {
      key: 'bandit-recommendation',
      icon: <ThunderboltOutlined />,
      label: '多臂老虎机推荐 (MAB)',
    },

    // 🗺️ 动态知识图谱存储 (Story 8.2)
    {
      key: 'dynamic-knowledge-graph-group',
      label: '🗺️ 动态知识图谱存储 (Story 8.2)',
      type: 'group' as const,
    },
    {
      key: 'dynamic-knowledge-graph',
      icon: <DatabaseOutlined />,
      label: '动态知识图谱存储 (Story 8.2)',
      children: [
        {
          key: 'kg-entity-management',
          icon: <NodeIndexOutlined />,
          label: '实体管理',
        },
        {
          key: 'kg-relation-management',
          icon: <ShareAltOutlined />,
          label: '关系管理',
        },
        {
          key: 'kg-graph-query',
          icon: <SearchOutlined />,
          label: '图查询引擎',
        },
        {
          key: 'kg-incremental-update',
          icon: <ThunderboltOutlined />,
          label: '增量更新监控',
        },
        {
          key: 'kg-quality-assessment',
          icon: <CheckCircleOutlined />,
          label: '质量评估仪表板',
        },
        {
          key: 'kg-performance-monitor',
          icon: <MonitorOutlined />,
          label: '性能监控',
        },
        {
          key: 'kg-schema-management',
          icon: <SettingOutlined />,
          label: '图模式管理',
        },
        {
          key: 'kg-data-migration',
          icon: <CloudServerOutlined />,
          label: '数据迁移工具',
        },
      ],
    },

    // 📊 知识管理API接口 (Story 8.6)
    {
      key: 'knowledge-management-api-group',
      label: '📊 知识管理API接口 (Story 8.6)',
      type: 'group' as const,
    },
    {
      key: 'sparql-engine',
      icon: <SearchOutlined />,
      label: 'SPARQL查询引擎',
      children: [
        {
          key: 'sparql-query-interface',
          icon: <SearchOutlined />,
          label: 'SPARQL查询界面',
        },
        {
          key: 'sparql-optimization',
          icon: <ThunderboltOutlined />,
          label: '查询优化器',
        },
        {
          key: 'sparql-performance',
          icon: <MonitorOutlined />,
          label: '性能监控',
        },
        {
          key: 'sparql-cache',
          icon: <DatabaseOutlined />,
          label: '查询缓存管理',
        },
      ],
    },
    {
      key: 'knowledge-api',
      icon: <ApiOutlined />,
      label: '知识管理REST API',
      children: [
        {
          key: 'entity-api',
          icon: <NodeIndexOutlined />,
          label: '实体CRUD API',
        },
        {
          key: 'relation-api',
          icon: <ShareAltOutlined />,
          label: '关系CRUD API',
        },
        {
          key: 'batch-operations',
          icon: <CloudServerOutlined />,
          label: '批量操作API',
        },
        {
          key: 'graph-validation',
          icon: <CheckCircleOutlined />,
          label: '图验证API',
        },
        {
          key: 'basic-rag-management',
          icon: <DatabaseOutlined />,
          label: '基础RAG管理',
        },
        {
          key: 'supervisor-api-management',
          icon: <ControlOutlined />,
          label: '监督者API管理',
        },
        {
          key: 'platform-api-management',
          icon: <CloudServerOutlined />,
          label: '平台API管理',
        },
      ],
    },
    {
      key: 'data-import-export',
      icon: <ExportOutlined />,
      label: '数据导入导出',
      children: [
        {
          key: 'rdf-import-export',
          icon: <FileTextOutlined />,
          label: 'RDF数据处理',
        },
        {
          key: 'csv-excel-import',
          icon: <FileExcelOutlined />,
          label: 'CSV/Excel导入',
        },
        {
          key: 'batch-import-jobs',
          icon: <CloudServerOutlined />,
          label: '批量导入任务',
        },
        {
          key: 'export-formats',
          icon: <ExportOutlined />,
          label: '多格式导出',
        },
      ],
    },
    {
      key: 'version-control',
      icon: <HistoryOutlined />,
      label: '版本控制系统',
      children: [
        {
          key: 'graph-snapshots',
          icon: <CameraOutlined />,
          label: '图快照管理',
        },
        {
          key: 'version-comparison',
          icon: <DiffOutlined />,
          label: '版本比较',
        },
        {
          key: 'rollback-operations',
          icon: <RollbackOutlined />,
          label: '回滚操作',
        },
        {
          key: 'change-tracking',
          icon: <EyeOutlined />,
          label: '变更追踪',
        },
      ],
    },
    {
      key: 'kg-auth-security',
      icon: <SecurityScanOutlined />,
      label: '认证与安全',
      children: [
        {
          key: 'jwt-auth',
          icon: <UserOutlined />,
          label: 'JWT身份认证',
        },
        {
          key: 'api-key-management',
          icon: <KeyOutlined />,
          label: 'API密钥管理',
        },
        {
          key: 'role-permissions',
          icon: <TeamOutlined />,
          label: '角色权限管理',
        },
        {
          key: 'security-audit',
          icon: <AuditOutlined />,
          label: '安全审计日志',
        },
      ],
    },
    {
      key: 'kg-monitoring',
      icon: <MonitorOutlined />,
      label: '监控与日志',
      children: [
        {
          key: 'performance-metrics',
          icon: <BarChartOutlined />,
          label: '性能指标监控',
        },
        {
          key: 'system-health',
          icon: <HeartOutlined />,
          label: '系统健康检查',
        },
        {
          key: 'alert-management',
          icon: <BellOutlined />,
          label: '告警管理',
        },
        {
          key: 'audit-logs',
          icon: <FileTextOutlined />,
          label: '审计日志查看',
        },
      ],
    },

    // 🧠 记忆管理系统
    {
      key: 'memory-group',
      label: '🧠 记忆管理系统',
      type: 'group' as const,
    },
    {
      key: 'memory-hierarchy',
      icon: <DatabaseOutlined />,
      label: '记忆层级架构 (Memory Hierarchy)',
    },
    {
      key: 'memory-recall',
      icon: <SearchOutlined />,
      label: '记忆召回测试 (Memory Recall)',
    },
    {
      key: 'memory-analytics',
      icon: <DashboardOutlined />,
      label: '记忆分析仪表板 (Memory Analytics)',
    },
    {
      key: 'memory-management-monitor',
      icon: <MonitorOutlined />,
      label: '记忆管理监控 (Memory Management Monitor)',
    },

    // 🌐 多模态处理
    {
      key: 'multimodal-group',
      label: '🌐 多模态处理',
      type: 'group' as const,
    },
    {
      key: 'content-understanding',
      icon: <FileImageOutlined />,
      label: '内容理解',
      children: [
        {
          key: 'multimodal-complete',
          icon: <FileImageOutlined />,
          label: '多模态完整版',
        },
        // 已移除不存在的多模态简化版
        {
          key: 'multimodal-basic',
          icon: <FileImageOutlined />,
          label: '多模态基础版',
        },
        {
          key: 'multimodal-rag-system',
          icon: <SearchOutlined />,
          label: '多模态RAG系统',
        },
        {
          key: 'multimodal-rag-management',
          icon: <ThunderboltOutlined />,
          label: '多模态RAG管理',
        },
        {
          key: 'document-management-complete',
          icon: <FileTextOutlined />,
          label: '智能文档管理',
        },
        {
          key: 'realtime-metrics-management',
          icon: <LineChartOutlined />,
          label: '实时指标监控',
        },
      ],
    },
    {
      key: 'file-management-system',
      icon: <DatabaseOutlined />,
      label: '文件管理系统',
      children: [
        {
          key: 'file-management-standard',
          icon: <DatabaseOutlined />,
          label: '标准文件管理',
        },
        {
          key: 'file-management-advanced',
          icon: <DatabaseOutlined />,
          label: '高级文件管理',
        },
      ],
    },
    {
      key: 'document-processing-center',
      icon: <FileTextOutlined />,
      label: '文档处理中心',
      children: [
        {
          key: 'document-processing-advanced',
          icon: <FileTextOutlined />,
          label: '高级文档处理',
        },
        {
          key: 'document-processing-simple',
          icon: <FileTextOutlined />,
          label: '简化文档处理',
        },
      ],
    },

    // 🔧 平台集成优化
    {
      key: 'platform-integration-group',
      label: '🔧 平台集成优化',
      type: 'group' as const,
    },
    {
      key: 'platform-integration-overview',
      icon: <SettingOutlined />,
      label: '平台集成总览',
    },
    {
      key: 'component-management',
      icon: <ApiOutlined />,
      label: '组件管理',
    },
    {
      key: 'workflow-orchestration',
      icon: <RocketOutlined />,
      label: '工作流编排',
    },
    {
      key: 'performance-optimization',
      icon: <ThunderboltOutlined />,
      label: '性能优化',
    },
    {
      key: 'system-monitoring',
      icon: <MonitorOutlined />,
      label: '系统监控',
    },
    {
      key: 'documentation-management',
      icon: <FileTextOutlined />,
      label: '文档管理',
    },
    {
      key: 'realtime-communication',
      icon: <WifiOutlined />,
      label: '实时通信系统',
    },

    // ⚡ 工作流引擎
    {
      key: 'workflow-group',
      label: '⚡ 工作流引擎',
      type: 'group' as const,
    },
    {
      key: 'workflows',
      icon: <NodeIndexOutlined />,
      label: 'LangGraph工作流',
      children: [
        {
          key: 'workflows-visualization',
          icon: <NodeIndexOutlined />,
          label: '工作流可视化',
        },
        {
          key: 'langgraph-features',
          icon: <ApiOutlined />,
          label: 'LangGraph新特性',
        },
        {
          key: 'langgraph-065',
          icon: <ApiOutlined />,
          label: 'LangGraph 0.6.5',
        },
      ],
    },
    
    // 分布式任务协调引擎分组
    {
      key: 'distributed-task-coordination-group',
      label: '🔗 分布式任务协调引擎',
      type: 'group' as const,
    },
    {
      key: 'distributed-task-coordination',
      icon: <ClusterOutlined />,
      label: '任务协调引擎',
    },
    {
      key: 'task-decomposer',
      icon: <BranchesOutlined />,
      label: '任务分解器',
    },
    {
      key: 'intelligent-assigner',
      icon: <TeamOutlined />,
      label: '智能分配器',
    },
    {
      key: 'raft-consensus',
      icon: <CrownOutlined />,
      label: 'Raft共识引擎',
    },
    {
      key: 'distributed-state-manager',
      icon: <DatabaseOutlined />,
      label: '分布式状态管理',
    },
    {
      key: 'conflict-resolver',
      icon: <ExceptionOutlined />,
      label: '冲突解决器',
    },
    {
      key: 'distributed-task-monitor',
      icon: <MonitorOutlined />,
      label: '任务监控',
    },
    {
      key: 'distributed-task-system-status',
      icon: <DashboardOutlined />,
      label: '系统状态',
    },
    {
      key: 'distributed-task-management-enhanced',
      icon: <SettingOutlined />,
      label: '任务管理增强',
    },

    {
      key: 'dag-orchestrator',
      icon: <ControlOutlined />,
      label: 'DAG编排器',
    },
    {
      key: 'flow-control',
      icon: <ThunderboltOutlined />,
      label: '流控背压监控',
    },

    // 🏭 系统处理引擎
    {
      key: 'processing-group',
      label: '🏭 系统处理引擎',
      type: 'group' as const,
    },
    {
      key: 'streaming',
      icon: <ThunderboltOutlined />,
      label: '流式处理',
    },
    {
      key: 'batch-processing',
      icon: <CloudServerOutlined />,
      label: '批处理系统',
      children: [
        {
          key: 'batch-jobs-management',
          icon: <CloudServerOutlined />,
          label: '批处理作业管理',
        },
        {
          key: 'batch-jobs-basic',
          icon: <CloudServerOutlined />,
          label: '基础批处理',
        },
        {
          key: 'intelligent-scheduling',
          icon: <ThunderboltOutlined />,
          label: '智能调度监控',
        },
      ],
    },
    {
      key: 'unified-engines',
      icon: <SettingOutlined />,
      label: '统一处理引擎',
      children: [
        {
          key: 'unified-engine',
          icon: <SettingOutlined />,
          label: '统一引擎',
        },
        {
          key: 'unified-engine-complete',
          icon: <SettingOutlined />,
          label: '完整统一引擎',
        },
      ],
    },
    {
      key: 'offline-sync',
      icon: <CloudServerOutlined />,
      label: '🔄 离线能力与同步',
      children: [
        {
          key: 'offline',
          icon: <CloudServerOutlined />,
          label: '离线能力监控',
        },
        {
          key: 'sync-management',
          icon: <ThunderboltOutlined />,
          label: '数据同步管理',
        },
        {
          key: 'conflict-resolution',
          icon: <ExceptionOutlined />,
          label: '冲突解决中心',
        },
        {
          key: 'conflict-resolution-learning',
          icon: <ExceptionOutlined />,
          label: '冲突解决学习',
        },
        {
          key: 'vector-clock-viz',
          icon: <NodeIndexOutlined />,
          label: '向量时钟可视化',
        },
        {
          key: 'vector-clock-algorithm',
          icon: <NodeIndexOutlined />,
          label: '向量时钟算法',
        },
        {
          key: 'sync-engine-internal',
          icon: <SettingOutlined />,
          label: '同步引擎内部机制',
        },
        {
          key: 'sync-engine-learning',
          icon: <ThunderboltOutlined />,
          label: '同步引擎学习',
        },
      ],
    },

    // 📊 系统监控运维
    {
      key: 'monitoring-group',
      label: '📊 系统监控运维',
      type: 'group' as const,
    },
    {
      key: 'distributed-events',
      icon: <BellOutlined />,
      label: '分布式事件',
      children: [
        {
          key: 'distributed-events-bus',
          icon: <BellOutlined />,
          label: '事件总线',
        },
        {
          key: 'distributed-events-system',
          icon: <BellOutlined />,
          label: '分布式事件系统',
        },
      ],
    },
    {
      key: 'system-monitoring-advanced',
      icon: <DashboardOutlined />,
      label: '系统监控',
      children: [
        {
          key: 'health',
          icon: <DashboardOutlined />,
          label: '健康监控',
        },
        {
          key: 'health-comprehensive',
          icon: <DashboardOutlined />,
          label: '综合健康监控',
        },
        {
          key: 'performance',
          icon: <AlertOutlined />,
          label: '性能分析',
        },
        {
          key: 'monitoring-dashboard',
          icon: <MonitorOutlined />,
          label: '监控仪表板',
        },
        {
          key: 'websocket-management',
          icon: <WifiOutlined />,
          label: 'WebSocket管理',
        },
        {
          key: 'cache-monitor',
          icon: <ThunderboltOutlined />,
          label: '缓存监控',
        },
        {
          key: 'model-cache-monitor',
          icon: <DatabaseOutlined />,
          label: '本地模型缓存监控',
        },
        {
          key: 'assignment-cache',
          icon: <UserOutlined />,
          label: '用户分配缓存',
        },
        {
          key: 'network-monitor-detail',
          icon: <MonitorOutlined />,
          label: '网络监控详情',
        },
      ],
    },

    // 🛡️ 安全管理
    {
      key: 'security-group',
      label: '🛡️ 安全管理',
      type: 'group' as const,
    },
    {
      key: 'security-systems',
      icon: <SafetyOutlined />,
      label: '安全管理系统',
      children: [
        {
          key: 'ai-trism',
          icon: <SafetyOutlined />,
          label: 'AI TRiSM框架',
        },
        {
          key: 'security-management',
          icon: <SafetyOutlined />,
          label: '安全策略管理',
        },
        {
          key: 'security',
          icon: <SafetyOutlined />,
          label: '安全管理中心',
        },
        {
          key: 'risk-assessment-dashboard',
          icon: <WarningOutlined />,
          label: '风险评估与回滚',
        },
        {
          key: 'statistical-analysis-dashboard',
          icon: <BarChartOutlined />,
          label: '统计分析仪表板',
        },
        {
          key: 'security-audit-system',
          icon: <AuditOutlined />,
          label: '安全审计系统',
        },
        {
          key: 'distributed-security-monitor',
          icon: <SecurityScanOutlined />,
          label: '分布式安全监控',
        },
        {
          key: 'security-basic',
          icon: <SafetyOutlined />,
          label: '基础安全系统',
        },
        {
          key: 'auth-management',
          icon: <UserOutlined />,
          label: '认证权限管理',
        },
      ],
    },

    // 🗄️ 数据存储
    {
      key: 'storage-group',
      label: '🗄️ 数据存储',
      type: 'group' as const,
    },
    {
      key: 'pgvector',
      icon: <DatabaseOutlined />,
      label: 'pgvector量化',
      children: [
        {
          key: 'pgvector-quantization',
          icon: <DatabaseOutlined />,
          label: 'pgvector量化',
        },
        // 已移除不存在的向量索引简化版
        // 已移除不存在的向量索引测试版
      ],
    },

    // 🔧 协议与工具
    {
      key: 'tools-group',
      label: '🔧 协议与工具',
      type: 'group' as const,
    },
    {
      key: 'mcp-tools',
      icon: <ApiOutlined />,
      label: 'MCP 1.0协议工具 (Protocol)',
    },

    // 🏢 企业架构
    {
      key: 'enterprise-group',
      label: '🏢 企业架构',
      type: 'group' as const,
    },
    {
      key: 'enterprise',
      icon: <CloudServerOutlined />,
      label: '架构管理总览 (Overview)',
    },
    {
      key: 'enterprise-config',
      icon: <SettingOutlined />,
      label: '企业配置中心 (Config Center)',
    },
    {
      key: 'debug',
      icon: <BugOutlined />,
      label: '架构调试工具 (Debug Tools)',
    },

    // 🔬 开发测试
    {
      key: 'dev-test-group',
      label: '🔬 开发测试',
      type: 'group' as const,
    },
    {
      key: 'structured-errors',
      icon: <ExceptionOutlined />,
      label: '结构化错误处理 (Error Handling)',
    },
    {
      key: 'test-coverage',
      icon: <CheckCircleOutlined />,
      label: '测试覆盖率分析 (Coverage)',
    },
    {
      key: 'test-integration',
      icon: <DatabaseOutlined />,
      label: '集成测试管理 (Integration Test)',
    },
    {
      key: 'test-suite',
      icon: <ExperimentOutlined />,
      label: '测试套件中心 (Test Suite)',
    },
    
    
    // 🧪 A/B测试实验平台
    {
      key: 'ab-testing-group',
      label: '🧪 A/B测试实验平台',
      type: 'group' as const,
    },
    // 实验管理
    {
      key: 'experiment-management',
      icon: <ExperimentOutlined />,
      label: '实验管理',
      children: [
        {
          key: 'experiment-list',
          icon: <UnorderedListOutlined />,
          label: '实验列表管理',
        },
        {
          key: 'experiment-dashboard',
          icon: <DashboardOutlined />,
          label: '实验仪表板',
        },
        {
          key: 'enhanced-experiment-analysis',
          icon: <BarChartOutlined />,
          label: '增强实验分析',
        },
      ],
    },
    // 流量管理
    {
      key: 'traffic-management',
      icon: <ShareAltOutlined />,
      label: '流量管理',
      children: [
        {
          key: 'traffic-allocation',
          icon: <ShareAltOutlined />,
          label: '流量分配管理',
        },
        {
          key: 'traffic-ramp-management',
          icon: <RocketOutlined />,
          label: '流量爬坡管理',
        },
      ],
    },
    // 数据分析
    {
      key: 'data-analysis',
      icon: <BarChartOutlined />,
      label: '数据分析',
      children: [
        {
          key: 'statistical-analysis',
          icon: <BarChartOutlined />,
          label: '统计分析',
        },
        {
          key: 'power-analysis',
          icon: <FunctionOutlined />,
          label: '统计功效分析',
        },
        {
          key: 'multiple-testing-correction',
          icon: <ScissorOutlined />,
          label: '多重检验校正',
        },
        {
          key: 'layered-experiments-management',
          icon: <BranchesOutlined />,
          label: '分层实验管理',
        },
      ],
    },
    // 事件跟踪
    {
      key: 'event-tracking-group',
      icon: <FundViewOutlined />,
      label: '事件跟踪',
      children: [
        {
          key: 'event-tracking',
          icon: <FundViewOutlined />,
          label: '事件跟踪管理',
        },
      ],
    },
    // 发布策略
    {
      key: 'release-strategy-group',
      icon: <RocketOutlined />,
      label: '发布策略',
      children: [
        {
          key: 'release-strategy',
          icon: <RocketOutlined />,
          label: '发布策略管理',
        },
      ],
    },
    // 监控告警
    {
      key: 'monitoring-alerts-group',
      icon: <MonitorOutlined />,
      label: '监控告警',
      children: [
        {
          key: 'monitoring-alerts',
          icon: <MonitorOutlined />,
          label: '监控告警系统',
        },
        {
          key: 'anomaly-detection',
          icon: <AlertOutlined />,
          label: '异常检测系统',
        },
        {
          key: 'auto-scaling',
          icon: <ThunderboltOutlined />,
          label: '自动扩量管理',
        },
      ],
    },
    // 高级算法
    {
      key: 'advanced-algorithms-group',
      icon: <ThunderboltOutlined />,
      label: '高级算法',
      children: [
        {
          key: 'advanced-algorithms',
          icon: <ThunderboltOutlined />,
          label: '高级算法引擎',
        },
      ],
    },

    // ⚡ LoRA/QLoRA微调框架
    {
      key: 'fine-tuning-group',
      label: '⚡ LoRA/QLoRA微调框架',
      type: 'group' as const,
    },
    {
      key: 'fine-tuning-jobs',
      icon: <UnorderedListOutlined />,
      label: '微调任务管理',
    },
    {
      key: 'fine-tuning-management',
      icon: <ExperimentOutlined />,
      label: '模型微调中心',
    },
    {
      key: 'fine-tuning-enhanced',
      icon: <RocketOutlined />,
      label: '高级微调管理中心',
    },
    {
      key: 'lora-training',
      icon: <GoldOutlined />,
      label: 'LoRA参数高效微调',
      children: [
        {
          key: 'lora-training-overview',
          icon: <DashboardOutlined />,
          label: 'LoRA训练总览',
        },
        {
          key: 'lora-config-templates',
          icon: <SettingOutlined />,
          label: 'LoRA配置模板',
        },
        {
          key: 'lora-model-adapters',
          icon: <DeploymentUnitOutlined />,
          label: '模型架构适配器',
        },
        {
          key: 'lora-performance-monitor',
          icon: <MonitorOutlined />,
          label: 'LoRA性能监控',
        },
      ],
    },
    {
      key: 'qlora-training',
      icon: <ThunderboltOutlined />,
      label: 'QLoRA量化微调',
      children: [
        {
          key: 'qlora-training-overview',
          icon: <ThunderboltOutlined />,
          label: 'QLoRA训练总览',
        },
        {
          key: 'qlora-quantization-config',
          icon: <CodeOutlined />,
          label: '量化配置管理',
        },
        {
          key: 'qlora-memory-optimization',
          icon: <DatabaseOutlined />,
          label: '内存优化监控',
        },
        {
          key: 'qlora-inference-optimization',
          icon: <RocketOutlined />,
          label: '推理优化加速',
        },
      ],
    },
    {
      key: 'distributed-training',
      icon: <ClusterOutlined />,
      label: '分布式训练管理',
      children: [
        {
          key: 'distributed-training-overview',
          icon: <GlobalOutlined />,
          label: '分布式训练总览',
        },
        {
          key: 'auto-scaling-management',
          icon: <ThunderboltOutlined />,
          label: '自动扩缩容管理',
        },
        {
          key: 'deepspeed-configuration',
          icon: <SettingOutlined />,
          label: 'DeepSpeed配置',
        },
        {
          key: 'multi-gpu-monitoring',
          icon: <MonitorOutlined />,
          label: '多GPU监控',
        },
        {
          key: 'training-synchronization',
          icon: <SyncOutlined />,
          label: '训练同步管理',
        },
      ],
    },
    {
      key: 'training-monitoring',
      icon: <FundProjectionScreenOutlined />,
      label: '训练监控可视化',
      children: [
        {
          key: 'training-dashboard',
          icon: <DashboardOutlined />,
          label: '训练仪表板',
        },
        {
          key: 'training-metrics',
          icon: <LineChartOutlined />,
          label: '训练指标分析',
        },
        {
          key: 'training-anomaly-detection',
          icon: <AlertOutlined />,
          label: '异常检测告警',
        },
        {
          key: 'training-reports',
          icon: <FileTextOutlined />,
          label: '训练报告生成',
        },
      ],
    },
    {
      key: 'model-management',
      icon: <DeploymentUnitOutlined />,
      label: '模型管理中心',
      children: [
        {
          key: 'supported-models',
          icon: <RobotOutlined />,
          label: '支持的模型列表',
        },
        {
          key: 'model-checkpoints',
          icon: <DatabaseOutlined />,
          label: '模型检查点管理',
        },
        {
          key: 'model-performance-comparison',
          icon: <BarChartOutlined />,
          label: '模型性能对比',
        },
        {
          key: 'model-deployment',
          icon: <CloudServerOutlined />,
          label: '模型部署管理',
        },
        {
          key: 'model-service-management',
          icon: <ApiOutlined />,
          label: '模型服务管理',
        },
      ],
    },
    {
      key: 'training-data-management',
      icon: <DatabaseOutlined />,
      label: '训练数据管理系统',
      children: [
        {
          key: 'training-data-overview',
          icon: <DashboardOutlined />,
          label: '数据管理总览',
        },
        {
          key: 'training-data-enhanced',
          icon: <RocketOutlined />,
          label: '增强训练数据管理',
        },
        {
          key: 'data-sources',
          icon: <CloudUploadOutlined />,
          label: '数据源管理',
        },
        {
          key: 'data-collection',
          icon: <InboxOutlined />,
          label: '数据收集',
        },
        {
          key: 'data-preprocessing',
          icon: <SettingOutlined />,
          label: '数据预处理',
        },
        {
          key: 'data-annotation',
          icon: <EditOutlined />,
          label: '数据标注管理',
        },
        {
          key: 'annotation-tasks',
          icon: <TagsOutlined />,
          label: '标注任务',
        },
        {
          key: 'annotation-quality',
          icon: <CheckCircleOutlined />,
          label: '标注质量控制',
        },
        {
          key: 'data-versioning',
          icon: <BranchesOutlined />,
          label: '数据版本管理',
        },
        {
          key: 'data-version-comparison',
          icon: <DiffOutlined />,
          label: '版本对比分析',
        },
        {
          key: 'data-export',
          icon: <ShareAltOutlined />,
          label: '数据导出工具',
        },
        {
          key: 'data-statistics',
          icon: <BarChartOutlined />,
          label: '数据统计分析',
        },
        {
          key: 'quality-metrics',
          icon: <MonitorOutlined />,
          label: '质量指标监控',
        },
      ],
    },

    // 🚀 模型优化系统
    {
      key: 'model-optimization-group',
      label: '🚀 模型优化系统',
      type: 'group' as const,
    },
    {
      key: 'model-compression',
      icon: <CompressOutlined />,
      label: '模型压缩和量化',
      children: [
        {
          key: 'model-compression-overview',
          icon: <DatabaseOutlined />,
          label: '压缩概览',
        },
        {
          key: 'quantization-manager',
          icon: <SettingOutlined />,
          label: '量化管理器',
        },
        {
          key: 'knowledge-distillation',
          icon: <ShareAltOutlined />,
          label: '知识蒸馏',
        },
        {
          key: 'model-pruning',
          icon: <ScissorOutlined />,
          label: '模型剪枝',
        },
        {
          key: 'compression-pipeline',
          icon: <NodeIndexOutlined />,
          label: '压缩流水线',
        },
        {
          key: 'compression-evaluation',
          icon: <BarChartOutlined />,
          label: '压缩评估',
        },
        {
          key: 'performance-benchmark',
          icon: <ThunderboltOutlined />,
          label: '性能基准测试',
        },
        {
          key: 'strategy-recommendation',
          icon: <BulbOutlined />,
          label: '策略推荐',
        },
      ],
    },
    {
      key: 'hyperparameter-optimization',
      icon: <ExperimentOutlined />,
      label: '超参数优化系统',
      children: [
        {
          key: 'hyperparameter-optimization-dashboard',
          icon: <DashboardOutlined />,
          label: '实验管理中心',
        },
        {
          key: 'hyperparameter-optimization-enhanced',
          icon: <ThunderboltOutlined />,
          label: '增强管理中心',
        },
        {
          key: 'hyperparameter-experiments',
          icon: <ExperimentOutlined />,
          label: '实验列表',
        },
        {
          key: 'hyperparameter-algorithms',
          icon: <SettingOutlined />,
          label: '算法配置',
        },
        {
          key: 'hyperparameter-visualizations',
          icon: <BarChartOutlined />,
          label: '可视化分析',
        },
        {
          key: 'hyperparameter-monitoring',
          icon: <MonitorOutlined />,
          label: '性能监控',
        },
        {
          key: 'hyperparameter-resources',
          icon: <CloudServerOutlined />,
          label: '资源管理',
        },
        {
          key: 'hyperparameter-scheduler',
          icon: <ClusterOutlined />,
          label: '试验调度器',
        },
        {
          key: 'hyperparameter-reports',
          icon: <FileTextOutlined />,
          label: '分析报告',
        },
      ],
    },

    // 🚀 个性化引擎
    {
      key: 'personalization-group',
      label: '🚀 个性化引擎',
      type: 'group' as const,
    },
    {
      key: 'personalization-system',
      icon: <UserOutlined />,
      label: '个性化系统',
      children: [
        {
          key: 'personalization-engine',
          icon: <UserOutlined />,
          label: '个性化引擎',
        },
        {
          key: 'personalization-monitor',
          icon: <MonitorOutlined />,
          label: '个性化监控',
        },
        {
          key: 'personalization-features',
          icon: <SettingOutlined />,
          label: '特征工程',
        },
        {
          key: 'personalization-learning',
          icon: <BulbOutlined />,
          label: '学习算法',
        },
        {
          key: 'personalization-api',
          icon: <ApiOutlined />,
          label: 'API管理',
        },
        {
          key: 'personalization-alerts',
          icon: <AlertOutlined />,
          label: '告警系统',
        },
        {
          key: 'personalization-production',
          icon: <CloudServerOutlined />,
          label: '生产部署',
        },
        {
          key: 'personalization-websocket',
          icon: <ShareAltOutlined />,
          label: 'WebSocket实时',
        },
      ],
    },

    // 😊 高级情感智能系统
    {
      key: 'emotional-intelligence-group',
      label: '😊 高级情感智能系统',
      type: 'group' as const,
    },
    {
      key: 'multimodal-emotion-recognition',
      icon: <HeartOutlined />,
      label: '多模态情感识别引擎',
      children: [
        {
          key: 'emotion-recognition-overview',
          icon: <EyeOutlined />,
          label: '情感识别总览',
        },
        {
          key: 'text-emotion-analysis',
          icon: <FileTextOutlined />,
          label: '文本情感分析',
        },
        {
          key: 'audio-emotion-recognition',
          icon: <AudioOutlined />,
          label: '音频情感识别',
        },
        {
          key: 'visual-emotion-analysis',
          icon: <CameraOutlined />,
          label: '视觉情感分析',
        },
        {
          key: 'multimodal-emotion-fusion',
          icon: <ShareAltOutlined />,
          label: '多模态情感融合',
        },
      ],
    },
    {
      key: 'emotion-state-modeling',
      icon: <BulbOutlined />,
      label: '情感状态建模系统',
      children: [
        {
          key: 'emotion-modeling',
          icon: <NodeIndexOutlined />,
          label: '情感建模总览',
        },
        {
          key: 'empathy-response-generator',
          icon: <HeartOutlined />,
          label: '共情响应生成器',
        },
      ],
    },
    {
      key: 'social-emotional-understanding',
      icon: <TeamOutlined />,
      label: '社交情感理解系统',
      children: [
        {
          key: 'group-emotion-analysis',
          icon: <UserOutlined />,
          label: '群体情感分析',
        },
        {
          key: 'relationship-dynamics',
          icon: <HeartOutlined />,
          label: '关系动态分析',
        },
        {
          key: 'social-context-adaptation',
          icon: <GlobalOutlined />,
          label: '社交情境适应',
        },
        {
          key: 'social-emotional-understanding-system',
          icon: <TeamOutlined />,
          label: '社交情感理解管理',
        },
        {
          key: 'cultural-adaptation',
          icon: <CompassOutlined />,
          label: '文化背景适应',
        },
        {
          key: 'social-intelligence-decision',
          icon: <BulbOutlined />,
          label: '社交智能决策',
        },
      ],
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {!hideSider && (
        <Sider
          data-testid="sidebar"
          trigger={null}
          collapsible
          collapsed={collapsed}
          breakpoint="lg"
          onBreakpoint={(broken) => {
            setSiderBroken(broken)
            setCollapsed(broken)
          }}
          width={280}
          collapsedWidth={siderBroken ? 0 : 80}
          style={{ 
            background: '#fff', 
            borderRight: siderBroken && collapsed ? 'none' : '1px solid #e8e8e8',
            position: 'sticky',
            top: 0,
            height: '100vh',
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          <div style={{ 
            padding: '16px', 
            borderBottom: '1px solid #e8e8e8',
            textAlign: collapsed ? 'center' : 'left',
            height: 'auto',
            minHeight: 'auto'
          }}>
            <Space align="center" style={{ justifyContent: collapsed ? 'center' : 'flex-start' }}>
              <Avatar 
                size={collapsed ? 32 : 40} 
                icon={<RobotOutlined />} 
                style={{ backgroundColor: '#1890ff' }}
              />
              {!collapsed && (
                <div>
                  <Title level={5} style={{ margin: 0 }}>AI Agent</Title>
                  <Text type="secondary" style={{ fontSize: '12px' }}>完整技术架构映射</Text>
                </div>
              )}
            </Space>
          </div>
          
          {/* 为E2E测试提供的反馈导航链接 */}
          <div 
            data-testid="nav-feedback" 
            onClick={() => navigate('/feedback-system')}
            style={{ 
              position: 'fixed', 
              top: '10px', 
              left: '10px', 
              width: '10px', 
              height: '10px', 
              opacity: 0.01,
              pointerEvents: 'auto',
              zIndex: 9999,
              backgroundColor: 'transparent'
            }}
          />
          
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          style={{ 
            border: 'none',
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            background: '#fff'
          }}
          onClick={({ key }) => {
            switch (key) {
              case 'chat': navigate('/chat'); break;
              case 'chat-history': navigate('/history'); break;
              case 'multi-agent': navigate('/multi-agent'); break;
              case 'supervisor': navigate('/supervisor'); break;
              case 'async-agents': navigate('/async-agents'); break;
              case 'agent-interface': navigate('/agent-interface'); break;
              case 'agent-cluster-management': navigate('/agent-cluster-management'); break;
              case 'agent-cluster-management-enhanced': navigate('/agent-cluster-management-enhanced'); break;

              // 增强版页面导航
              case 'multi-agent-enhanced': navigate('/multi-agent-enhanced'); break;
              case 'rag-enhanced': navigate('/rag-enhanced'); break;
              case 'experiments-platform': navigate('/experiments-platform'); break;
              case 'workflow-management': navigate('/workflow-management'); break;

              // 智能代理服务发现系统导航
              case 'service-discovery-overview': navigate('/service-discovery-overview'); break;
              case 'agent-registry-management': navigate('/agent-registry'); break;
              case 'service-routing-management': navigate('/service-routing'); break;
              case 'load-balancer-config': navigate('/load-balancer-config'); break;
              case 'service-health-monitor': navigate('/service-health-monitor'); break;
              case 'service-cluster-management': navigate('/service-cluster-management'); break;
              case 'service-performance-dashboard': navigate('/service-performance-dashboard'); break;
              case 'service-config-management': navigate('/service-config-management'); break;
              
              case 'rag': navigate('/rag'); break;
              case 'agentic-rag': navigate('/agentic-rag'); break;
              case 'graphrag': navigate('/graphrag'); break;
              case 'hybrid-search': navigate('/hybrid-search'); break;
              case 'multimodal': navigate('/multimodal'); break;
              case 'content-understanding': navigate('/content-understanding'); break;
              case 'multimodal-complete': navigate('/multimodal-complete'); break;
              case 'file-management-system': navigate('/file-management'); break;
              case 'file-management-standard': navigate('/file-management-standard'); break;
              case 'document-processing-center': navigate('/document-processing'); break;
              case 'document-processing-advanced': navigate('/document-processing-advanced'); break;
              case 'file-management': navigate('/file-management'); break;
              case 'platform-integration-overview': navigate('/platform-integration-overview'); break;
              case 'component-management': navigate('/component-management'); break;
              case 'workflow-orchestration': navigate('/workflow-orchestration'); break;
              case 'performance-optimization': navigate('/performance-optimization'); break;
              case 'system-monitoring': navigate('/system-monitoring'); break;
              case 'documentation-management': navigate('/documentation-management'); break;
              case 'workflows': navigate('/workflows'); break;
              case 'workflows-visualization': navigate('/workflow'); break;
              case 'langgraph-features': navigate('/langgraph-features'); break;
              
              // 分布式任务协调引擎导航
              case 'distributed-task-coordination': navigate('/distributed-task-coordination'); break;
              case 'task-decomposer': navigate('/task-decomposer'); break;
              case 'intelligent-assigner': navigate('/intelligent-assigner'); break;
              case 'raft-consensus': navigate('/raft-consensus'); break;
              case 'distributed-state-manager': navigate('/distributed-state-manager'); break;
              case 'conflict-resolver': navigate('/conflict-resolver'); break;
              case 'distributed-task-monitor': navigate('/distributed-task-monitor'); break;
              case 'distributed-task-system-status': navigate('/distributed-task-system-status'); break;
              case 'distributed-task-management-enhanced': navigate('/distributed-task-management-enhanced'); break;
              
              case 'dag-orchestrator': navigate('/dag-orchestrator'); break;
              case 'flow-control': navigate('/flow-control'); break;
              case 'streaming': navigate('/streaming-monitor'); break;
              case 'batch-jobs': navigate('/batch'); break;
              case 'unified-engine': navigate('/unified-engine'); break;
              case 'ai-trism': navigate('/ai-trism'); break;
              case 'security-management': navigate('/security-management'); break;
              case 'security': navigate('/security'); break;
              case 'security-audit': navigate('/security-audit'); break;
              case 'security-audit-system': navigate('/security-audit'); break;
              case 'distributed-security-monitor': navigate('/distributed-security-monitor'); break;
              case 'auth-management': navigate('/auth-management'); break;
              case 'events': navigate('/events'); break;
              case 'health': navigate('/health'); break;
              case 'performance': navigate('/performance'); break;
              case 'monitor': navigate('/monitor'); break;
              case 'monitoring-dashboard': navigate('/monitoring-dashboard'); break;
              case 'websocket-management': navigate('/websocket-management'); break;
              case 'pgvector': navigate('/pgvector'); break;
              case 'pgvector-quantization': navigate('/pgvector'); break;
              case 'cache-monitor': navigate('/cache'); break;
              case 'mcp-tools': navigate('/mcp-tools'); break;
              case 'enterprise': navigate('/enterprise'); break;
              case 'enterprise-config': navigate('/enterprise-config'); break;
              case 'debug': navigate('/debug'); break;
              case 'structured-errors': navigate('/structured-errors'); break;
              case 'test-coverage': navigate('/test-coverage'); break;
              case 'test-integration': navigate('/test'); break;
              case 'test-suite': navigate('/test-suite'); break;
              case 'document-processing': navigate('/document-processing'); break;
              case 'reasoning': navigate('/reasoning'); break;
              case 'multi-step-reasoning': navigate('/multi-step-reasoning'); break;
              case 'explainable-ai': navigate('/explainable-ai'); break;
              case 'targeting-rules': navigate('/targeting-rules'); break;
              case 'memory-hierarchy': navigate('/memory-hierarchy'); break;
              case 'memory-recall': navigate('/memory-recall'); break;
              case 'memory-analytics': navigate('/memory-analytics'); break;
              case 'offline': navigate('/offline'); break;
              case 'sync-management': navigate('/sync'); break;
              case 'conflict-resolution': navigate('/conflicts'); break;
              case 'vector-clock-viz': navigate('/vector-clock'); break;
              case 'network-monitor-detail': navigate('/network-monitor'); break;
              case 'sync-engine-internal': navigate('/sync-engine'); break;
              case 'model-cache-monitor': navigate('/model-cache'); break;
              case 'assignment-cache': navigate('/assignment-cache'); break;
              case 'bandit-recommendation': navigate('/bandit-recommendation'); break;
              // Q-Learning算法家族
              case 'qlearning': navigate('/qlearning'); break;
              case 'qlearning-dashboard': navigate('/qlearning'); break;
              case 'qlearning-training': navigate('/qlearning-training'); break;
              case 'qlearning-strategy': navigate('/qlearning-strategy'); break;
              case 'qlearning-recommendation': navigate('/qlearning-recommendation'); break;
              case 'qlearning-performance': navigate('/qlearning-performance'); break;
              case 'qlearning-performance-optimization': navigate('/qlearning-performance-optimization'); break;
              case 'qlearning-tabular': navigate('/qlearning/tabular'); break;
              case 'qlearning-dqn': navigate('/qlearning/dqn'); break;
              case 'qlearning-variants': navigate('/qlearning/variants'); break;
              
              // TensorFlow Q-Learning管理
              case 'tensorflow-qlearning': navigate('/tensorflow-qlearning'); break;
              
              // 测试管理系统
              case 'testing-management': navigate('/testing-management'); break;
              
              // 假设检验统计
              case 'hypothesis-testing': navigate('/hypothesis-testing'); break;
              
              // 探索策略系统
              case 'exploration-strategies': navigate('/exploration-strategies'); break;
              case 'ucb-strategies': navigate('/ucb-strategies'); break;
              case 'thompson-sampling': navigate('/thompson-sampling'); break;
              case 'adaptive-exploration': navigate('/adaptive-exploration'); break;
              
              // 奖励函数系统
              case 'basic-rewards': navigate('/basic-rewards'); break;
              case 'composite-rewards': navigate('/composite-rewards'); break;
              case 'adaptive-rewards': navigate('/adaptive-rewards'); break;
              case 'reward-shaping': navigate('/reward-shaping'); break;
              
              // 环境建模系统
              case 'state-space': navigate('/state-space'); break;
              case 'action-space': navigate('/action-space'); break;
              case 'environment-simulator': navigate('/environment-simulator'); break;
              case 'grid-world': navigate('/grid-world'); break;
              
              // 训练管理系统
              case 'training-manager': navigate('/training-manager'); break;
              case 'learning-rate-scheduler': navigate('/learning-rate-scheduler'); break;
              case 'early-stopping': navigate('/early-stopping'); break;
              case 'performance-tracker': navigate('/performance-tracker'); break;
              
              // 用户反馈学习系统
              case 'feedback-system': navigate('/feedback-system'); break;
              case 'feedback-analytics': navigate('/feedback-analytics'); break;
              case 'user-feedback-profiles': navigate('/user-feedback-profiles'); break;
              case 'item-feedback-analysis': navigate('/item-feedback-analysis'); break;
              case 'feedback-quality-monitor': navigate('/feedback-quality-monitor'); break;
              
              // 智能行为分析系统
              case 'behavior-analytics': navigate('/behavior-analytics'); break;
              case 'behavior-analytics-events': navigate('/behavior-analytics/events'); break;
              case 'behavior-analytics-sessions': navigate('/behavior-analytics/sessions'); break;
              case 'behavior-analytics-reports': navigate('/behavior-analytics/reports'); break;
              case 'behavior-analytics-realtime': navigate('/behavior-analytics/realtime'); break;
              case 'behavior-analytics-export': navigate('/behavior-analytics/export'); break;
              case 'behavior-analytics-config': navigate('/behavior-analytics/config'); break;
              
              // 整合页面导航
              case 'conflict-resolution-learning': navigate('/conflict-resolution-learning'); break;
              case 'sync-engine-learning': navigate('/sync-engine-learning'); break;
              case 'health-comprehensive': navigate('/health-comprehensive'); break;
              // 已移除multimodal-simple导航
              // 已移除vector-advanced-simple导航
              // 已移除vector-advanced-test导航
              case 'vector-clock-algorithm': navigate('/vector-clock-algorithm'); break;
              case 'unified-engine-complete': navigate('/unified-engine-complete'); break;
              case 'batch-jobs-basic': navigate('/batch-jobs'); break;
              case 'document-processing-simple': navigate('/document-processing-simple'); break;
              case 'security-basic': navigate('/security'); break;
              case 'multimodal-basic': navigate('/multimodal-basic'); break;
              case 'file-management-advanced': navigate('/file-management-advanced'); break;
              case 'distributed-events-bus': navigate('/distributed-events'); break;
              case 'distributed-events-system': navigate('/distributed-events'); break;
              case 'langgraph-065': navigate('/langgraph-065'); break;
              case 'multimodal-rag-system': navigate('/multimodal-rag'); break;
              case 'multimodal-rag-management': navigate('/multimodal-rag-management'); break;
              case 'document-management-complete': navigate('/document-management-complete'); break;
              case 'realtime-metrics-management': navigate('/realtime-metrics-management'); break;
              
              // A/B测试实验平台导航
              case 'experiment-list': navigate('/experiments'); break;
              case 'experiment-dashboard': navigate('/experiments/dashboard'); break;
              case 'enhanced-experiment-analysis': navigate('/experiments/enhanced-analysis'); break;
              case 'statistical-analysis': navigate('/experiments/statistical-analysis'); break;
              case 'traffic-allocation': navigate('/experiments/traffic-allocation'); break;
              case 'event-tracking': navigate('/experiments/event-tracking'); break;
              case 'traffic-ramp-management': navigate('/experiments/traffic-ramp'); break;
              case 'power-analysis': navigate('/experiments/power-analysis'); break;
              case 'multiple-testing-correction': navigate('/experiments/multiple-testing'); break;
              case 'layered-experiments-management': navigate('/experiments/layered-experiments'); break;
              case 'release-strategy': navigate('/experiments/release-strategy'); break;
              case 'monitoring-alerts': navigate('/experiments/monitoring-alerts'); break;
              case 'advanced-algorithms': navigate('/experiments/advanced-algorithms'); break;

              // 服务发现与离线管理导航
              case 'service-discovery-management': navigate('/service-discovery-management'); break;
              case 'offline-management': navigate('/offline-management'); break;
              
              // 强化学习系统监控导航
              case 'rl-system-dashboard': navigate('/rl-system-dashboard'); break;
              case 'rl-performance-monitor': navigate('/rl-performance-monitor'); break;
              case 'rl-integration-test': navigate('/rl-integration-test'); break;
              case 'rl-alert-config': navigate('/rl-alert-config'); break;
              case 'rl-metrics-analysis': navigate('/rl-metrics-analysis'); break;
              case 'rl-system-health': navigate('/rl-system-health'); break;
              
              // 动态知识图谱存储系统 (Story 8.2) 导航
              case 'kg-entity-management': navigate('/kg-entity-management'); break;
              case 'kg-relation-management': navigate('/kg-relation-management'); break;
              case 'kg-graph-query': navigate('/kg-graph-query'); break;
              case 'kg-incremental-update': navigate('/kg-incremental-update'); break;
              case 'kg-quality-assessment': navigate('/kg-quality-assessment'); break;
              case 'kg-performance-monitor': navigate('/kg-performance-monitor'); break;
              case 'kg-schema-management': navigate('/kg-schema-management'); break;
              case 'kg-data-migration': navigate('/kg-data-migration'); break;
              
              // 知识管理API接口 (Story 8.6) 导航
              // SPARQL查询引擎
              case 'sparql-query-interface': navigate('/sparql-query-interface'); break;
              case 'sparql-optimization': navigate('/sparql-optimization'); break;
              case 'sparql-performance': navigate('/sparql-performance'); break;
              case 'sparql-cache': navigate('/sparql-cache'); break;
              
              // 知识管理REST API
              case 'entity-api': navigate('/entity-api'); break;
              case 'relation-api': navigate('/relation-api'); break;
              case 'batch-operations': navigate('/batch-operations'); break;
              case 'batch-jobs-management': navigate('/batch-operations'); break;
              case 'graph-validation': navigate('/graph-validation'); break;
              case 'basic-rag-management': navigate('/basic-rag-management'); break;
              case 'supervisor-api-management': navigate('/supervisor-api-management'); break;
              case 'platform-api-management': navigate('/platform-api-management'); break;
              
              // 数据导入导出
              case 'rdf-import-export': navigate('/rdf-import-export'); break;
              case 'csv-excel-import': navigate('/csv-excel-import'); break;
              case 'batch-import-jobs': navigate('/batch-import-jobs'); break;
              case 'export-formats': navigate('/export-formats'); break;
              
              // 版本控制系统
              case 'graph-snapshots': navigate('/graph-snapshots'); break;
              case 'version-comparison': navigate('/version-comparison'); break;
              case 'rollback-operations': navigate('/rollback-operations'); break;
              case 'change-tracking': navigate('/change-tracking'); break;
              
              // 认证与安全
              case 'jwt-auth': navigate('/jwt-auth'); break;
              case 'api-key-management': navigate('/api-key-management'); break;
              case 'role-permissions': navigate('/role-permissions'); break;
              
              // 监控与日志
              case 'performance-metrics': navigate('/performance-metrics'); break;
              case 'system-health': navigate('/system-health'); break;
              case 'alert-management': navigate('/alert-management'); break;
              case 'audit-logs': navigate('/audit-logs'); break;
              
              // 知识图推理引擎 (Story 8.3) 导航
              // 混合推理引擎
              case 'kg-reasoning-dashboard': navigate('/kg-reasoning-dashboard'); break;
              case 'kg-reasoning-query': navigate('/kg-reasoning-query'); break;
              case 'kg-reasoning-batch': navigate('/kg-reasoning-batch'); break;
              case 'kg-reasoning-performance': navigate('/kg-reasoning-performance'); break;
              case 'kg-reasoning-strategy': navigate('/kg-reasoning-strategy'); break;
              case 'kg-reasoning-explanation': navigate('/kg-reasoning-explanation'); break;
              
              // 规则推理引擎
              case 'kg-rule-management': navigate('/kg-rule-management'); break;
              case 'kg-rule-execution': navigate('/kg-rule-execution'); break;
              case 'kg-rule-validation': navigate('/kg-rule-validation'); break;
              case 'kg-rule-conflict': navigate('/kg-rule-conflict'); break;
              
              // 嵌入推理引擎
              case 'kg-embedding-models': navigate('/kg-embedding-models'); break;
              case 'kg-embedding-training': navigate('/kg-embedding-training'); break;
              case 'kg-embedding-similarity': navigate('/kg-embedding-similarity'); break;
              case 'kg-embedding-index': navigate('/kg-embedding-index'); break;
              
              // 路径推理引擎
              case 'kg-path-discovery': navigate('/kg-path-discovery'); break;
              case 'kg-path-analysis': navigate('/kg-path-analysis'); break;
              case 'kg-path-optimization': navigate('/kg-path-optimization'); break;
              case 'kg-path-confidence': navigate('/kg-path-confidence'); break;
              
              // 不确定性推理
              case 'kg-uncertainty-analysis': navigate('/kg-uncertainty-analysis'); break;
              case 'kg-bayesian-network': navigate('/kg-bayesian-network'); break;
              case 'kg-probability-calculation': navigate('/kg-probability-calculation'); break;
              case 'kg-confidence-interval': navigate('/kg-confidence-interval'); break;
              
              // LoRA/QLoRA微调框架导航
              case 'fine-tuning-jobs': navigate('/fine-tuning-jobs'); break;
              case 'fine-tuning-management': navigate('/fine-tuning'); break;
              case 'fine-tuning-enhanced': navigate('/fine-tuning-enhanced'); break;
              case 'lora-training': navigate('/lora-training'); break;
              case 'lora-training-overview': navigate('/lora-training-overview'); break;
              case 'lora-config-templates': navigate('/lora-config-templates'); break;
              case 'lora-model-adapters': navigate('/lora-model-adapters'); break;
              case 'lora-performance-monitor': navigate('/lora-performance-monitor'); break;
              case 'qlora-training': navigate('/qlora-training'); break;
              case 'qlora-training-overview': navigate('/qlora-training-overview'); break;
              case 'qlora-quantization-config': navigate('/qlora-quantization-config'); break;
              case 'qlora-memory-optimization': navigate('/qlora-memory-optimization'); break;
              case 'qlora-inference-optimization': navigate('/qlora-inference-optimization'); break;
              case 'distributed-training': navigate('/distributed-training'); break;
              case 'distributed-training-overview': navigate('/distributed-training-overview'); break;
              case 'deepspeed-configuration': navigate('/deepspeed-configuration'); break;
              case 'multi-gpu-monitoring': navigate('/multi-gpu-monitoring'); break;
              case 'training-synchronization': navigate('/training-synchronization'); break;
              case 'training-monitoring': navigate('/training-monitoring'); break;
              case 'training-dashboard': navigate('/training-dashboard'); break;
              case 'training-metrics': navigate('/training-metrics'); break;
              case 'training-anomaly-detection': navigate('/training-anomaly-detection'); break;
              case 'anomaly-detection': navigate('/anomaly-detection'); break;
              case 'auto-scaling': navigate('/auto-scaling'); break;
              case 'training-reports': navigate('/training-reports'); break;
              case 'model-management': navigate('/model-management'); break;
              case 'supported-models': navigate('/supported-models'); break;
              case 'model-checkpoints': navigate('/model-checkpoints'); break;
              case 'model-performance-comparison': navigate('/model-performance-comparison'); break;
              case 'model-deployment': navigate('/model-deployment'); break;
              // 训练数据管理系统导航
              case 'training-data-management': navigate('/training-data-management'); break;
              case 'training-data-overview': navigate('/training-data-management'); break;
              case 'training-data-enhanced': navigate('/training-data-enhanced'); break;
              case 'data-sources': navigate('/data-sources'); break;
              case 'data-collection': navigate('/data-collection'); break;
              case 'data-preprocessing': navigate('/data-preprocessing'); break;
              case 'data-annotation': navigate('/data-annotation'); break;
              case 'annotation-tasks': navigate('/annotation-tasks'); break;
              case 'annotation-quality': navigate('/annotation-quality'); break;
              case 'data-versioning': navigate('/data-versioning'); break;
              case 'data-version-comparison': navigate('/data-version-comparison'); break;
              case 'data-export': navigate('/data-export'); break;
              case 'data-statistics': navigate('/data-statistics'); break;
              case 'quality-metrics': navigate('/quality-metrics'); break;
              
              // 高级情感智能系统导航
              case 'emotion-recognition-overview': navigate('/emotion-recognition-overview'); break;
              case 'text-emotion-analysis': navigate('/text-emotion-analysis'); break;
              case 'audio-emotion-recognition': navigate('/audio-emotion-recognition'); break;
              case 'visual-emotion-analysis': navigate('/visual-emotion-analysis'); break;
              case 'multimodal-emotion-fusion': navigate('/multimodal-emotion-fusion'); break;
              case 'emotion-modeling': navigate('/emotion-modeling'); break;
              case 'empathy-response-generator': navigate('/empathy-response-generator'); break;
              
              // 社交情感理解系统导航
              case 'group-emotion-analysis': navigate('/group-emotion-analysis'); break;
              case 'relationship-dynamics': navigate('/relationship-dynamics'); break;
              case 'social-context-adaptation': navigate('/social-context-adaptation'); break;
              case 'social-emotional-understanding-system': navigate('/social-emotional-understanding'); break;
              case 'cultural-adaptation': navigate('/cultural-adaptation'); break;
              case 'social-intelligence-decision': navigate('/social-intelligence-decision'); break;
              
              // 个性化系统导航
              case 'personalization-engine': navigate('/personalization-engine'); break;
              case 'personalization-monitor': navigate('/personalization-monitor'); break;
              case 'personalization-features': navigate('/personalization-features'); break;
              case 'personalization-learning': navigate('/personalization-learning'); break;
              case 'personalization-api': navigate('/personalization-api'); break;
              case 'personalization-websocket': navigate('/personalization-websocket'); break;
              case 'personalization-production': navigate('/personalization-production'); break;
              case 'personalization-alerts': navigate('/personalization-alerts'); break;
              
              // 模型优化系统导航
              // 模型压缩和量化
              case 'model-compression': navigate('/compression-pipeline'); break;
              case 'model-compression-overview': navigate('/model-compression-overview'); break;
              case 'quantization-manager': navigate('/quantization-manager'); break;
              case 'knowledge-distillation': navigate('/knowledge-distillation'); break;
              case 'model-pruning': navigate('/model-pruning'); break;
              case 'compression-pipeline': navigate('/compression-pipeline'); break;
              case 'compression-evaluation': navigate('/compression-evaluation'); break;
              case 'performance-benchmark': navigate('/performance-benchmark'); break;
              case 'strategy-recommendation': navigate('/strategy-recommendation'); break;
              
              // 超参数优化系统
              case 'hyperparameter-optimization': navigate('/hyperparameter-optimization'); break;
              case 'hyperparameter-optimization-dashboard': navigate('/hyperparameter-optimization'); break;
              case 'hyperparameter-optimization-enhanced': navigate('/hyperparameter-optimization-enhanced'); break;
              case 'hyperparameter-experiments': navigate('/hyperparameter-experiments'); break;
              case 'hyperparameter-algorithms': navigate('/hyperparameter-algorithms'); break;
              case 'hyperparameter-visualizations': navigate('/hyperparameter-visualizations'); break;
              case 'hyperparameter-monitoring': navigate('/hyperparameter-monitoring'); break;
              case 'hyperparameter-resources': navigate('/hyperparameter-resources'); break;
              case 'hyperparameter-scheduler': navigate('/hyperparameter-scheduler'); break;
              case 'hyperparameter-reports': navigate('/hyperparameter-reports'); break;
            }
          }}
        />
      </Sider>
      )}

      <Layout>
        <Header style={{ 
          background: '#fff', 
          borderBottom: '1px solid #e8e8e8',
          padding: '0 20px',
          height: '60px',
          lineHeight: '60px'
        }}>
          <Space align="center">
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: '16px' }}
            />
            {siderBroken && (
              <Title level={5} style={{ margin: 0 }}>
                AI Agent
              </Title>
            )}
          </Space>
        </Header>

        <Content style={{ 
          background: '#f0f2f5',
          overflow: 'auto'
        }}>
          <Suspense fallback={
            <div style={{
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              height: 'calc(100vh - 120px)',
              flexDirection: 'column',
              gap: '16px'
            }}>
              <Spin size="large" />
              <div style={{ color: '#666', fontSize: '16px' }}>加载中...</div>
            </div>
          }>
          <Routes>
            {/* 🤖 智能体系统 */}
            <Route path="/" element={<ChatPage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/history" element={<ConversationHistoryPage />} />
            <Route path="/multi-agent" element={<MultiAgentPage />} />
            <Route path="/tensorflow-qlearning" element={<TensorFlowQLearningPage />} />
            <Route path="/testing" element={<Navigate to="/testing-management" replace />} />
            <Route path="/testing-management" element={<TestingManagementPage />} />
            <Route path="/hypothesis-testing" element={<HypothesisTestingPage />} />
            <Route path="/supervisor" element={<SupervisorPage />} />
            <Route path="/async-agents" element={<AsyncAgentPage />} />
            <Route path="/agent-interface" element={<AgentInterfacePage />} />
            <Route path="/agent-cluster-management" element={<AgentClusterManagementPage />} />
            <Route path="/agent-cluster-management-enhanced" element={<AgentClusterManagementPageEnhanced />} />

            {/* 增强版页面 */}
            <Route path="/multi-agent-enhanced" element={<MultiAgentEnhancedPage />} />
            <Route path="/rag-enhanced" element={<RAGEnhancedPage />} />
            <Route path="/experiments-platform" element={<ExperimentsPlatformPage />} />
            <Route path="/workflow-management" element={<WorkflowManagementPage />} />

            {/* 🔍 智能检索引擎 */}
            <Route path="/rag" element={<RagPage />} />
            <Route path="/agentic-rag" element={<AgenticRagPage />} />
            <Route path="/graphrag" element={<GraphRAGPage />} />
            <Route path="/graphrag-enhanced" element={<GraphRAGPageEnhanced />} />
            <Route path="/hybrid-search" element={<HybridSearchAdvancedPage />} />
            
            {/* 🧠 推理引擎 */}
            <Route path="/reasoning" element={<ReasoningPage />} />
            <Route path="/multi-step-reasoning" element={<MultiStepReasoningPage />} />
            <Route path="/explainable-ai" element={<ExplainableAiPage />} />
            <Route path="/targeting-rules" element={<TargetingRulesManagementPage />} />
            
            {/* 🗺️ 知识图谱引擎 */}
            <Route path="/knowledge-extraction-overview" element={<KnowledgeExtractionOverviewPage />} />
            <Route path="/entity-recognition" element={<EntityRecognitionPage />} />
            <Route path="/relation-extraction" element={<RelationExtractionPage />} />
            <Route path="/entity-linking" element={<EntityLinkingPage />} />
            <Route path="/multilingual-processing" element={<MultilingualProcessingPage />} />
            <Route path="/knowledge-graph-visualization" element={<KnowledgeGraphVisualizationPage />} />
            <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
            <Route path="/knowledge-graph-query" element={<GraphQueryEnginePage />} />
            <Route path="/knowledge-graph-analytics" element={<GraphAnalyticsPage />} />
            <Route path="/knowledge-batch-jobs" element={<KnowledgeBatchJobsPage />} />
            <Route path="/knowledge-batch-monitor" element={<KnowledgeBatchMonitorPage />} />
            <Route path="/knowledge-performance-optimization" element={<KnowledgePerformanceOptimizationPage />} />
            <Route path="/knowledge-cache-management" element={<KnowledgeCacheManagementPage />} />
            <Route path="/knowledge-validation" element={<KnowledgeValidationPage />} />
            <Route path="/knowledge-confidence-analysis" element={<KnowledgeConfidenceAnalysisPage />} />
            <Route path="/knowledge-error-analysis" element={<KnowledgeErrorAnalysisPage />} />
            <Route path="/knowledge-model-comparison" element={<KnowledgeModelComparisonPage />} />
            <Route path="/acl-protocol-management" element={<ACLProtocolManagementPage />} />
            
            {/* 🗺️ 动态知识图谱存储系统 (Story 8.2) */}
            <Route path="/kg-entity-management" element={<KnowledgeGraphEntityManagement />} />
            <Route path="/kg-relation-management" element={<KnowledgeGraphRelationManagement />} />
            <Route path="/kg-graph-query" element={<KnowledgeGraphQueryEngine />} />
            <Route path="/kg-incremental-update" element={<KnowledgeGraphIncrementalUpdate />} />
            <Route path="/kg-quality-assessment" element={<KnowledgeGraphQualityAssessment />} />
            <Route path="/kg-performance-monitor" element={<KnowledgeGraphPerformanceMonitor />} />
            <Route path="/kg-schema-management" element={<KnowledgeGraphSchemaManagement />} />
            <Route path="/kg-data-migration" element={<KnowledgeGraphDataMigration />} />
            
            {/* 📊 知识管理API接口 (Story 8.6) */}
            {/* SPARQL查询引擎 */}
            <Route path="/sparql-query-interface" element={<SparqlQueryInterface />} />
            <Route path="/sparql-optimization" element={<SparqlOptimization />} />
            <Route path="/sparql-performance" element={<SparqlPerformance />} />
            <Route path="/sparql-cache" element={<SparqlCache />} />
            
            {/* 知识管理REST API */}
            <Route path="/entity-api" element={<EntityApiPage />} />
            <Route path="/relation-api" element={<RelationApiPage />} />
            <Route path="/batch-operations" element={<BatchOperationsPage />} />
            <Route path="/graph-validation" element={<GraphValidationPage />} />
            <Route path="/basic-rag-management" element={<BasicRagManagementPage />} />
            <Route path="/supervisor-api-management" element={<SupervisorApiManagementPage />} />
            <Route path="/platform-api-management" element={<PlatformApiManagementPage />} />
            
            {/* 数据导入导出 */}
            <Route path="/rdf-import-export" element={<RdfImportExportPage />} />
            <Route path="/csv-excel-import" element={<CsvExcelImportPage />} />
            <Route path="/batch-import-jobs" element={<BatchImportJobsPage />} />
            <Route path="/export-formats" element={<ExportFormatsPage />} />
            
            {/* 版本控制系统 */}
            <Route path="/graph-snapshots" element={<GraphSnapshotsPage />} />
            <Route path="/version-comparison" element={<VersionComparisonPage />} />
            <Route path="/rollback-operations" element={<RollbackOperationsPage />} />
            <Route path="/change-tracking" element={<ChangeTrackingPage />} />
            
            {/* 认证与安全 */}
            <Route path="/jwt-auth" element={<JwtAuthPage />} />
            <Route path="/api-key-management" element={<ApiKeyManagementPage />} />
            <Route path="/role-permissions" element={<RolePermissionsPage />} />
            <Route path="/security-audit" element={<SecurityAuditPage />} />
            
            {/* 监控与日志 */}
            <Route path="/performance-metrics" element={<PerformanceMetricsPage />} />
            <Route path="/system-health" element={<SystemHealthPage />} />
            <Route path="/alert-management" element={<AlertManagementPage />} />
            <Route path="/audit-logs" element={<AuditLogsPage />} />
            
            {/* 🧠 知识图推理引擎 (Story 8.3) */}
            {/* 混合推理引擎 */}
            <Route path="/kg-reasoning-dashboard" element={<KGReasoningDashboardPage />} />
            <Route path="/kg-reasoning-query" element={<KGReasoningQueryPage />} />
            <Route path="/kg-reasoning-batch" element={<KGReasoningBatchPage />} />
            <Route path="/kg-reasoning-performance" element={<KGReasoningOptimizationPage />} />
            <Route path="/kg-reasoning-strategy" element={<KGReasoningConfigPage />} />
            <Route path="/kg-reasoning-explanation" element={<KGReasoningAnalysisPage />} />
            
            {/* 规则推理引擎 */}
            <Route path="/kg-rule-management" element={<KGRuleManagementPage />} />
            <Route path="/kg-rule-execution" element={<KGRuleExecutionPage />} />
            <Route path="/kg-rule-validation" element={<KGRuleValidationPage />} />
            <Route path="/kg-rule-conflict" element={<KGRuleConflictPage />} />
            
            {/* 嵌入推理引擎 */}
            <Route path="/kg-embedding-models" element={<KGEmbeddingModelsPage />} />
            <Route path="/kg-embedding-training" element={<KGEmbeddingTrainingPage />} />
            <Route path="/kg-embedding-similarity" element={<KGEmbeddingSimilarityPage />} />
            <Route path="/kg-embedding-index" element={<KGEmbeddingIndexPage />} />
            
            {/* 路径推理引擎 */}
            <Route path="/kg-path-discovery" element={<KGPathDiscoveryPage />} />
            <Route path="/kg-path-analysis" element={<KGPathAnalysisPage />} />
            <Route path="/kg-path-optimization" element={<KGPathOptimizationPage />} />
            <Route path="/kg-path-confidence" element={<KGPathConfidencePage />} />
            
            {/* 不确定性推理 */}
            <Route path="/kg-uncertainty-analysis" element={<KGUncertaintyAnalysisPage />} />
            <Route path="/kg-bayesian-network" element={<KGBayesianNetworkPage />} />
            <Route path="/kg-probability-calculation" element={<KGProbabilityCalculationPage />} />
            <Route path="/kg-confidence-interval" element={<KGConfidenceIntervalPage />} />
            
            {/* 🎯 推荐算法引擎 */}
            <Route path="/bandit-recommendation" element={<BanditRecommendationPage />} />
            
            {/* 🚀 个性化引擎 */}
            <Route path="/personalization-engine" element={<PersonalizationEnginePage />} />
            <Route path="/personalization-monitor" element={<PersonalizationMonitorPage />} />
            <Route path="/personalization-features" element={<PersonalizationFeaturePage />} />
            <Route path="/personalization-learning" element={<PersonalizationLearningPage />} />
            <Route path="/personalization-api" element={<PersonalizationApiPage />} />
            <Route path="/personalization-alerts" element={<PersonalizationAlertsPage />} />
            <Route path="/personalization-production" element={<PersonalizationProductionPage />} />
            <Route path="/personalization-websocket" element={<PersonalizationWebSocketPage />} />
            <Route path="/websocket-management" element={<WebSocketManagementPage />} />
            
            {/* 😊 高级情感智能系统 */}
            <Route path="/emotion-recognition-overview" element={<EmotionRecognitionOverviewPage />} />
            <Route path="/text-emotion-analysis" element={<TextEmotionAnalysisPage />} />
            <Route path="/audio-emotion-recognition" element={<AudioEmotionRecognitionPage />} />
            <Route path="/visual-emotion-analysis" element={<VisualEmotionAnalysisPage />} />
            <Route path="/multimodal-emotion-fusion" element={<MultiModalEmotionFusionPage />} />
            <Route path="/emotion-modeling" element={<EmotionModelingPage />} />
            
            {/* 社交情感理解系统 (Story 11.6) */}
            <Route path="/group-emotion-analysis" element={<GroupEmotionAnalysisPage />} />
            <Route path="/relationship-dynamics" element={<RelationshipDynamicsPage />} />
            <Route path="/social-context-adaptation" element={<SocialContextAdaptationPage />} />
            <Route path="/social-emotional-understanding" element={<SocialEmotionalUnderstandingPage />} />
            <Route path="/cultural-adaptation" element={<CulturalAdaptationPage />} />
            <Route path="/social-intelligence-decision" element={<SocialIntelligenceDecisionPage />} />
            
            {/* 情感记忆管理系统 (Story 11.4) */}
            <Route path="/emotional-memory-management" element={<EmotionalMemoryManagementPage />} />
            <Route path="/emotional-event-analysis" element={<EmotionalEventAnalysisPage />} />
            <Route path="/emotional-preference-learning" element={<EmotionalMemoryPage />} />
            <Route path="/emotional-trigger-patterns" element={<EmotionalMemoryPage />} />
            <Route path="/emotional-memory-retrieval" element={<EmotionalMemoryPage />} />
            
            {/* 情感智能决策引擎 (Story 11.5) */}
            <Route path="/emotional-intelligence-decision-engine" element={<EmotionalIntelligenceDecisionEnginePage />} />
            <Route path="/emotional-risk-assessment-dashboard" element={<EmotionalRiskAssessmentDashboardPage />} />
            <Route path="/crisis-detection-support" element={<CrisisDetectionSupportPage />} />
            <Route path="/intervention-strategy-management" element={<InterventionStrategyManagementPage />} />
            <Route path="/emotional-health-monitoring-dashboard" element={<EmotionalHealthMonitoringDashboardPage />} />
            <Route path="/decision-history-analysis" element={<DecisionHistoryAnalysisPage />} />
            <Route path="/empathy-response-generator" element={<EmpathyResponseGeneratorPage />} />
            
            {/* 🧠 强化学习系统 */}
            <Route path="/qlearning" element={<QLearningPage />} />
            <Route path="/qlearning-training" element={<QLearningTrainingPage />} />
            <Route path="/qlearning-strategy" element={<QLearningStrategyPage />} />
            <Route path="/qlearning-recommendation" element={<QLearningRecommendationPage />} />
            <Route path="/qlearning-performance" element={<QLearningPerformancePage />} />
            <Route path="/qlearning-performance-optimization" element={<QLearningPerformanceOptimizationPage />} />
            
            {/* 🤖 TensorFlow模型管理 */}
            <Route path="/tensorflow" element={<TensorFlowManagementPage />} />
            <Route path="/qlearning/tabular" element={<TabularQLearningPage />} />
            <Route path="/qlearning/dqn" element={<DQNPage />} />
            <Route path="/qlearning/variants" element={<DQNVariantsPage />} />
            <Route path="/exploration-strategies" element={<ExplorationStrategiesPage />} />
            <Route path="/ucb-strategies" element={<UCBStrategiesPage />} />
            <Route path="/thompson-sampling" element={<ThompsonSamplingPage />} />
            <Route path="/adaptive-exploration" element={<ExplorationStrategiesPage />} />
            <Route path="/basic-rewards" element={<BasicRewardsPage />} />
            <Route path="/composite-rewards" element={<CompositeRewardsPage />} />
            <Route path="/adaptive-rewards" element={<AdaptiveRewardsPage />} />
            <Route path="/reward-shaping" element={<RewardShapingPage />} />
            <Route path="/state-space" element={<StateSpacePage />} />
            <Route path="/action-space" element={<ActionSpacePage />} />
            <Route path="/environment-simulator" element={<EnvironmentSimulatorPage />} />
            <Route path="/grid-world" element={<GridWorldPage />} />
            <Route path="/training-manager" element={<TrainingManagerPage />} />
            <Route path="/learning-rate-scheduler" element={<LearningRateSchedulerPage />} />
            <Route path="/early-stopping" element={<EarlyStoppingPage />} />
            <Route path="/performance-tracker" element={<PerformanceTrackerPage />} />
            
            {/* ❤️ 用户反馈学习系统 */}
            <Route path="/feedback-system" element={<FeedbackSystemPage />} />
            <Route path="/feedback-analytics" element={<FeedbackAnalyticsPage />} />
            <Route path="/user-feedback-profiles" element={<UserFeedbackProfilesPage />} />
            <Route path="/item-feedback-analysis" element={<ItemFeedbackAnalysisPage />} />
            <Route path="/feedback-quality-monitor" element={<FeedbackQualityMonitorPage />} />
            
            {/* 📈 智能行为分析系统 */}
            <Route path="/behavior-analytics" element={<BehaviorAnalyticsPage />} />
            <Route path="/behavior-analytics-enhanced" element={<BehaviorAnalyticsPageEnhanced />} />
            <Route path="/behavior-analytics/events" element={<EventDataManagePage />} />
            <Route path="/behavior-analytics/sessions" element={<SessionManagePage />} />
            <Route path="/behavior-analytics/reports" element={<ReportCenterPage />} />
            <Route path="/behavior-analytics/realtime" element={<RealTimeMonitorPage />} />
            <Route path="/behavior-analytics/export" element={<DataExportPage />} />
            <Route path="/behavior-analytics/config" element={<SystemConfigPage />} />
            
            {/* 🧠 记忆管理系统 */}
            <Route path="/memory-hierarchy" element={<MemoryHierarchyPage />} />
            <Route path="/memory-recall" element={<MemoryRecallTestPage />} />
            <Route path="/memory-analytics" element={<MemoryAnalyticsDashboard />} />
            
            {/* 🌐 多模态处理 */}
            <Route path="/multimodal" element={<MultimodalPageComplete />} />
            <Route path="/file-management" element={<FileManagementPageComplete />} />
            
            {/* 🔧 平台集成优化 */}
            <Route path="/platform-integration-overview" element={<PlatformIntegrationOverviewPage />} />
            <Route path="/component-management" element={<ComponentManagementPage />} />
            <Route path="/workflow-orchestration" element={<WorkflowOrchestrationPage />} />
            <Route path="/performance-optimization" element={<PerformanceOptimizationPage />} />
            <Route path="/system-monitoring" element={<SystemMonitoringPage />} />
            <Route path="/documentation-management" element={<DocumentationManagementPage />} />
            <Route path="/realtime-communication" element={<RealtimeCommunicationPage />} />
            
            {/* 🛡️ 故障容错和恢复系统 (Story 10.5) */}
            <Route path="/fault-tolerance-overview" element={<FaultToleranceSystemPage />} />
            <Route path="/fault-detection" element={<FaultDetectionPage />} />
            <Route path="/recovery-management" element={<RecoveryManagementPage />} />
            <Route path="/backup-management" element={<BackupManagementPage />} />
            <Route path="/consistency-management" element={<ConsistencyManagementPage />} />
            <Route path="/fault-testing" element={<FaultTestingPage />} />
            
            {/* ⚡ 工作流引擎 */}
            <Route path="/workflow" element={<WorkflowPage />} />
            <Route path="/workflows" element={<WorkflowPage />} />
            <Route path="/langgraph-features" element={<LangGraphFeaturesPage />} />
            
            {/* 🔗 分布式任务协调引擎 */}
            <Route path="/distributed-task-coordination" element={<DistributedTaskCoordinationPage />} />
            <Route path="/task-decomposer" element={<TaskDecomposerPage />} />
            <Route path="/intelligent-assigner" element={<IntelligentAssignerPage />} />
            <Route path="/raft-consensus" element={<RaftConsensusPage />} />
            <Route path="/distributed-state-manager" element={<DistributedStateManagerPage />} />
            <Route path="/conflict-resolver" element={<ConflictResolverPage />} />
            <Route path="/distributed-task-monitor" element={<DistributedTaskMonitorPage />} />
            <Route path="/distributed-task-system-status" element={<DistributedTaskSystemStatusPage />} />
            <Route path="/distributed-task-management-enhanced" element={<DistributedTaskManagementPageEnhanced />} />
            <Route path="/dag-orchestrator" element={<DagOrchestratorPage />} />
            <Route path="/flow-control" element={<FlowControlPage />} />
            <Route path="/distributed-message-overview" element={<DistributedMessageOverviewPage />} />
            
            {/* 🏭 处理引擎 */}
            <Route path="/streaming-monitor" element={<StreamingMonitorPage />} />
            <Route path="/batch" element={<BatchJobsPageFixed />} />
            <Route path="/batch-processing" element={<BatchProcessingPage />} />
            <Route path="/intelligent-scheduling" element={<IntelligentSchedulingPage />} />
            <Route path="/unified-engine" element={<UnifiedEnginePageComplete />} />
            
            {/* 🛡️ 安全与合规 */}
            <Route path="/ai-trism" element={<AiTrismPage />} />
            <Route path="/security-management" element={<SecurityManagementPage />} />
            <Route path="/security-management-enhanced" element={<SecurityManagementEnhancedPage />} />
            <Route path="/security-audit" element={<SecurityAuditPage />} />
            <Route path="/distributed-security-monitor" element={<DistributedSecurityMonitorPage />} />
            <Route path="/auth-management" element={<AuthManagementPage />} />
            
            {/* 📊 事件与监控 */}
            <Route path="/events" element={<EventDashboardPage />} />
            <Route path="/health" element={<HealthMonitorPage />} />
            <Route path="/performance" element={<PerformanceMonitorPage />} />
            <Route path="/monitor" element={<UnifiedMonitorPage />} />
            <Route path="/monitoring-dashboard" element={<MonitoringDashboardPage />} />
            
            {/* 🗄️ 数据存储 */}
            <Route path="/pgvector" element={<PgVectorPage />} />
            <Route path="/vector-advanced" element={<VectorAdvancedPage />} />
            <Route path="/cache" element={<CacheMonitorPage />} />
            <Route path="/assignment-cache" element={<AssignmentCacheManagementPage />} />
            
            {/* 🔧 协议与工具 */}
            <Route path="/mcp-tools" element={<MCPToolsPage />} />
            
            {/* 🏢 企业架构 */}
            <Route path="/enterprise" element={<EnterpriseArchitecturePage />} />
            <Route path="/enterprise-architecture" element={<EnterpriseArchitecturePage />} />
            <Route path="/enterprise-config" element={<EnterpriseConfigPage />} />
            <Route path="/debug" element={<ArchitectureDebugPage />} />
            
            {/* 🔄 离线能力与同步机制 */}
            <Route path="/offline" element={<OfflineCapabilityPage />} />
            <Route path="/sync" element={<SyncManagementPage />} />
            <Route path="/conflicts" element={<ConflictResolutionPage />} />
            <Route path="/vector-clock" element={<VectorClockVisualizationPage />} />
            <Route path="/network-monitor" element={<NetworkMonitorDetailPage />} />
            <Route path="/sync-engine" element={<SyncEngineInternalPage />} />
            <Route path="/model-cache" element={<ModelCacheMonitorPage />} />
            
            {/* 🔬 开发测试 */}
            <Route path="/structured-errors" element={<StructuredErrorPage />} />
            <Route path="/test-coverage" element={<TestCoveragePage />} />
            <Route path="/test" element={<IntegrationTestPage />} />
            <Route path="/test-suite" element={<TestingSuitePage />} />
            <Route path="/document-processing" element={<DocumentProcessingPage />} />
            
            {/* 缺失页面补充 */}
            <Route path="/conflict-resolution-learning" element={<ConflictResolutionLearningPage />} />
            <Route path="/sync-engine-learning" element={<SyncEngineLearningPage />} />
            <Route path="/health-comprehensive" element={<HealthComprehensivePage />} />
            {/* 已移除不存在的multimodal-simple路由 */}
            {/* 已移除不存在的vector-advanced-simple路由 */}
            {/* 已移除不存在的vector-advanced-test路由 */}
            <Route path="/vector-clock-algorithm" element={<VectorClockAlgorithmPage />} />
            <Route path="/unified-engine-complete" element={<UnifiedEnginePageComplete />} />
            <Route path="/batch-jobs" element={<BatchJobsPageFixed />} />
            <Route path="/document-processing-simple" element={<DocumentProcessingPage />} />
            <Route path="/document-processing-advanced" element={<DocumentProcessingAdvancedPage />} />
            <Route path="/security" element={<SecurityPage />} />
            <Route path="/multimodal-basic" element={<MultimodalPageComplete />} />
            <Route path="/multimodal-complete" element={<MultimodalPageComplete />} />
            <Route path="/file-management-advanced" element={<FileManagementAdvancedPage />} />
            <Route path="/distributed-events" element={<DistributedEventsPage />} />
            <Route path="/langgraph-065" element={<LangGraph065Page />} />
            <Route path="/multimodal-rag" element={<MultimodalRagPage />} />
            <Route path="/multimodal-rag-management" element={<MultimodalRagManagementPage />} />
            <Route path="/document-management-complete" element={<DocumentManagementPageComplete />} />
            <Route path="/realtime-metrics-management" element={<RealtimeMetricsManagementPage />} />
            
            {/* 🧪 A/B测试实验平台 */}
            <Route path="/experiments" element={<ExperimentListPage />} />
            <Route path="/experiments/dashboard" element={<ExperimentDashboardPage />} />
            <Route path="/experiments/enhanced-analysis" element={<EnhancedExperimentAnalysisPage />} />
            <Route path="/experiments/statistical-analysis" element={<StatisticalAnalysisPage />} />
            <Route path="/experiments/traffic-allocation" element={<TrafficAllocationPage />} />
            <Route path="/experiments/event-tracking" element={<EventTrackingPage />} />
            <Route path="/experiments/traffic-ramp" element={<TrafficRampManagementPage />} />
            <Route path="/experiments/power-analysis" element={<PowerAnalysisPage />} />
            <Route path="/descriptive-statistics" element={<DescriptiveStatisticsPage />} />
            <Route path="/experiments/multiple-testing" element={<MultipleTestingCorrectionPage />} />
            <Route path="/experiments/layered-experiments" element={<LayeredExperimentsManagementPage />} />
            <Route path="/experiments/release-strategy" element={<ReleaseStrategyPage />} />
            <Route path="/experiments/monitoring-alerts" element={<MonitoringAlertsPage />} />
            <Route path="/experiments/advanced-algorithms" element={<AdvancedAlgorithmsPage />} />
            
            {/* 服务发现与离线管理 */}
            <Route path="/service-discovery-management" element={<ServiceDiscoveryManagementPage />} />
            <Route path="/offline-management" element={<OfflineManagementPage />} />
            
            {/* 📊 强化学习系统监控 */}
            <Route path="/rl-system-dashboard" element={<RLSystemDashboardPage />} />
            <Route path="/rl-performance-monitor" element={<RLPerformanceMonitorPage />} />
            <Route path="/rl-integration-test" element={<RLIntegrationTestPage />} />
            <Route path="/rl-alert-config" element={<RLAlertConfigPage />} />
            <Route path="/rl-metrics-analysis" element={<RLMetricsAnalysisPage />} />
            <Route path="/rl-system-health" element={<RLSystemHealthPage />} />
            
            {/* ⚡ LoRA/QLoRA微调框架 */}
            <Route path="/fine-tuning-jobs" element={<FineTuningJobsPage />} />
            <Route path="/fine-tuning-config" element={<FineTuningConfigPage />} />
            <Route path="/fine-tuning-monitor" element={<FineTuningMonitorPage />} />
            <Route path="/fine-tuning-models" element={<FineTuningModelsPage />} />
            <Route path="/fine-tuning-datasets" element={<FineTuningDatasetsPage />} />
            <Route path="/fine-tuning" element={<FineTuningPage />} />
            <Route path="/fine-tuning-enhanced" element={<FineTuningPageEnhanced />} />
            <Route path="/lora-training-overview" element={<LoRATrainingPage />} />
            <Route path="/lora-config-templates" element={<FineTuningConfigPage />} />
            <Route path="/lora-model-adapters" element={<ModelAdaptersPage />} />
            <Route path="/lora-performance-monitor" element={<FineTuningMonitorPage />} />
            <Route path="/qlora-training-overview" element={<QLoRATrainingPage />} />
            <Route path="/qlora-quantization-config" element={<FineTuningConfigPage />} />
            <Route path="/qlora-memory-optimization" element={<FineTuningMonitorPage />} />
            <Route path="/qlora-inference-optimization" element={<FineTuningMonitorPage />} />
            <Route path="/distributed-training-overview" element={<DistributedTrainingPage />} />
            <Route path="/auto-scaling-management" element={<AutoScalingManagementPage />} />
            <Route path="/risk-assessment-dashboard" element={<RiskAssessmentDashboard />} />
            <Route path="/statistical-analysis-dashboard" element={<StatisticalAnalysisDashboard />} />
            <Route path="/deepspeed-configuration" element={<FineTuningConfigPage />} />
            <Route path="/multi-gpu-monitoring" element={<TrainingMonitorDashboard />} />
            <Route path="/training-synchronization" element={<DistributedTrainingPage />} />
            <Route path="/training-dashboard" element={<TrainingMonitorDashboard />} />
            <Route path="/training-metrics" element={<FineTuningMonitorPage />} />
            <Route path="/training-anomaly-detection" element={<TrainingMonitorDashboard />} />
            <Route path="/anomaly-detection" element={<AnomalyDetectionPage />} />
            <Route path="/auto-scaling" element={<AutoScalingManagementPage />} />
            <Route path="/batch-operations" element={<BatchOperationsPage />} />
            <Route path="/training-reports" element={<TrainingMonitorDashboard />} />
            <Route path="/supported-models" element={<FineTuningModelsPage />} />
            <Route path="/model-checkpoints" element={<FineTuningCheckpointsPage />} />
            <Route path="/model-performance-comparison" element={<ModelPerformanceComparison />} />
            <Route path="/model-deployment" element={<FineTuningModelsPage />} />
            {/* 📊 训练数据管理系统 (Story 9.5) */}
            <Route path="/training-data-management" element={<TrainingDataManagementPage />} />
            <Route path="/training-data-enhanced" element={<TrainingDataManagementPageEnhanced />} />
            <Route path="/data-sources" element={<DataSourceManagementPage />} />
            <Route path="/data-collection" element={<DataCollectionPage />} />
            <Route path="/data-preprocessing" element={<DataPreprocessingPage />} />
            <Route path="/data-annotation" element={<DataAnnotationManagementPage />} />
            <Route path="/annotation-tasks" element={<AnnotationTasksPage />} />
            <Route path="/annotation-quality" element={<AnnotationQualityControlPage />} />
            <Route path="/data-versioning" element={<DataVersionManagementPage />} />
            <Route path="/data-version-comparison" element={<TrainingDataManagementPage />} />
            <Route path="/data-export" element={<TrainingDataManagementPage />} />
            <Route path="/data-statistics" element={<TrainingDataManagementPage />} />
            <Route path="/quality-metrics" element={<TrainingDataManagementPage />} />
            
            {/* 📊 模型评估和基准测试系统 (Story 9.4) */}
            <Route path="/model-evaluation-overview" element={<ModelEvaluationOverviewPage />} />
            <Route path="/model-evaluation-management" element={<ModelEvaluationManagementPage />} />
            <Route path="/memory-management-monitor" element={<MemoryManagementMonitorPage />} />
            <Route path="/model-performance-benchmark" element={<ModelPerformanceBenchmarkPage />} />
            <Route path="/evaluation-engine-management" element={<EvaluationEngineManagementPage />} />
            <Route path="/benchmark-suite-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-tasks-monitor" element={<EvaluationTasksMonitorPage />} />
            <Route path="/evaluation-reports-center" element={<EvaluationReportsCenterPage />} />
            <Route path="/evaluation-api-management" element={<EvaluationApiManagementPage />} />
            <Route path="/model-comparison-dashboard" element={<ModelComparisonDashboardPage />} />
            <Route path="/benchmark-glue-management" element={<BenchmarkGlueManagementPage />} />
            <Route path="/benchmark-superglue-management" element={<BenchmarkSupergluePage />} />
            <Route path="/benchmark-mmlu-management" element={<BenchmarkMmluPage />} />
            <Route path="/benchmark-humaneval-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/benchmark-hellaswag-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/benchmark-custom-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-metrics-config" element={<EvaluationMetricsConfigPage />} />
            <Route path="/evaluation-performance-monitor" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-batch-processing" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-regression-detection" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-quality-assurance" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-automation-pipeline" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-alerts-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-data-management" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-resource-monitor" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-job-scheduler" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-results-analysis" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-export-import" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-version-control" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-compliance-audit" element={<BenchmarkSuiteManagementPage />} />
            <Route path="/evaluation-security-management" element={<BenchmarkSuiteManagementPage />} />
            
            {/* 🚀 模型压缩和量化工具 (Story 9.2) */}
            <Route path="/model-compression-overview" element={<ModelCompressionOverviewPage />} />
            <Route path="/quantization-manager" element={<QuantizationManagerPage />} />
            <Route path="/knowledge-distillation" element={<KnowledgeDistillationPage />} />
            <Route path="/model-pruning" element={<ModelPruningPage />} />
            <Route path="/compression-pipeline" element={<CompressionPipelinePage />} />
            <Route path="/compression-evaluation" element={<ModelCompressionEvaluationPage />} />
            <Route path="/performance-benchmark" element={<ModelPerformanceBenchmarkPage />} />
            <Route path="/hardware-benchmark" element={<ModelPerformanceBenchmarkPage />} />
            <Route path="/strategy-recommendation" element={<CompressionStrategyRecommendationPage />} />
            
            {/* 🚀 模型服务部署平台 (Story 9.6) */}
            <Route path="/model-registry" element={<ModelRegistryPage />} />
            <Route path="/model-inference" element={<ModelInferencePage />} />
            <Route path="/model-deployment" element={<ModelDeploymentPage />} />
            <Route path="/model-monitoring" element={<ModelMonitoringPage />} />
            <Route path="/model-service-management" element={<ModelServiceManagementPage />} />
            <Route path="/online-learning" element={<OnlineLearningPage />} />
            
            {/* 🚀 自动化超参数优化系统 (Story 9.3) */}
            <Route path="/hyperparameter-optimization" element={<HyperparameterOptimizationPage />} />
            <Route path="/hyperparameter-optimization-enhanced" element={<HyperparameterOptimizationPageEnhanced />} />
            <Route path="/hyperparameter-experiments" element={<HyperparameterExperimentsPage />} />
            <Route path="/hyperparameter-algorithms" element={<HyperparameterAlgorithmsPage />} />
            <Route path="/hyperparameter-visualizations" element={<HyperparameterVisualizationsPage />} />
            <Route path="/hyperparameter-monitoring" element={<HyperparameterMonitoringPage />} />
            <Route path="/hyperparameter-resources" element={<HyperparameterResourcesPage />} />
            <Route path="/hyperparameter-scheduler" element={<HyperparameterSchedulerPage />} />
            <Route path="/hyperparameter-reports" element={<HyperparameterReportsPage />} />
            
            {/* 🌐 智能代理服务发现系统 (Story 10.1) */}
            <Route path="/service-discovery-overview" element={<ServiceDiscoveryOverviewPage />} />
            <Route path="/agent-registry" element={<AgentRegistryManagementPage />} />
            <Route path="/service-routing" element={<ServiceRoutingManagementPage />} />
            <Route path="/load-balancer-config" element={<LoadBalancerConfigPage />} />
	            <Route path="/service-health-monitor" element={<ServiceHealthMonitorPage />} />
	            <Route path="/service-cluster-management" element={<ServiceClusterManagementPage />} />
	            <Route path="/service-performance-dashboard" element={<ServicePerformanceDashboardPage />} />
	            <Route path="/service-config-management" element={<ServiceConfigManagementPage />} />
	            <Route path="*" element={<Navigate to="/chat" replace />} />
	          </Routes>
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}

export default App
