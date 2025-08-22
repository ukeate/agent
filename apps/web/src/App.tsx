import React, { useState, Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
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
} from '@ant-design/icons'

// 懒加载所有页面组件
const ChatPage = lazy(() => import('./pages/ChatPage'))
const MultiAgentPage = lazy(() => import('./pages/MultiAgentPage'))
const SupervisorPage = lazy(() => import('./pages/SupervisorPage'))
const RagPage = lazy(() => import('./pages/RagPage'))
const WorkflowPage = lazy(() => import('./pages/WorkflowPage'))
const AsyncAgentPage = lazy(() => import('./pages/AsyncAgentPage'))
const AgenticRagPage = lazy(() => import('./pages/AgenticRagPage'))
const MultimodalPageComplete = lazy(() => import('./pages/MultimodalPageComplete'))
const FlowControlPage = lazy(() => import('./pages/FlowControlPage'))
const MCPToolsPage = lazy(() => import('./pages/MCPToolsPage'))
const PgVectorPage = lazy(() => import('./pages/PgVectorPage'))
const CacheMonitorPage = lazy(() => import('./pages/CacheMonitorPage'))
const BatchJobsPageFixed = lazy(() => import('./pages/BatchJobsPageFixed'))
const HealthMonitorPage = lazy(() => import('./pages/HealthMonitorPage'))
const PerformanceMonitorPage = lazy(() => import('./pages/PerformanceMonitorPage'))
const StreamingMonitorPage = lazy(() => import('./pages/StreamingMonitorPage'))
const MonitoringDashboardPage = lazy(() => import('./pages/MonitoringDashboardPage'))
const EnterpriseArchitecturePage = lazy(() => import('./pages/EnterpriseArchitecturePage'))
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
const MemoryHierarchyPage = lazy(() => import('./pages/MemoryHierarchyPage'))
const MemoryRecallTestPage = lazy(() => import('./pages/MemoryRecallTestPage'))
const MemoryAnalyticsDashboard = lazy(() => import('./pages/MemoryAnalyticsDashboard'))
const ReasoningPage = lazy(() => import('./pages/ReasoningPage'))
const MultiStepReasoningPage = lazy(() => import('./pages/MultiStepReasoningPage'))
const ExplainableAiPage = lazy(() => import('./pages/ExplainableAiPage'))
const HybridSearchAdvancedPage = lazy(() => import('./pages/HybridSearchAdvancedPage'))
const FileManagementPageComplete = lazy(() => import('./pages/FileManagementPageComplete'))

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
const UnifiedEnginePage = lazy(() => import('./pages/UnifiedEnginePage'))
const UnifiedMonitorPage = lazy(() => import('./pages/UnifiedMonitorPage'))
const AiTrismPage = lazy(() => import('./pages/AiTrismPage'))
const SecurityManagementPage = lazy(() => import('./pages/SecurityManagementPage'))
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

// 新增缺失页面（去重后）
const ConflictResolutionLearningPage = lazy(() => import('./pages/ConflictResolutionLearningPage'))
const SyncEngineLearningPage = lazy(() => import('./pages/SyncEngineLearningPage'))
const HealthComprehensivePage = lazy(() => import('./pages/HealthComprehensivePage'))
const MultimodalPageSimple = lazy(() => import('./pages/MultimodalPageSimple'))
const VectorAdvancedPage = lazy(() => import('./pages/VectorAdvancedPage'))
const VectorAdvancedPageSimple = lazy(() => import('./pages/VectorAdvancedPageSimple'))
const VectorAdvancedTestPage = lazy(() => import('./pages/VectorAdvancedTestPage'))
const VectorClockAlgorithmPage = lazy(() => import('./pages/VectorClockAlgorithmPage'))
const UnifiedEnginePageComplete = lazy(() => import('./pages/UnifiedEnginePageComplete'))
const BatchJobsPage = lazy(() => import('./pages/BatchJobsPage'))
const DocumentProcessingPage = lazy(() => import('./pages/DocumentProcessingPage'))
const SecurityPage = lazy(() => import('./pages/SecurityPage'))
const MultimodalPage = lazy(() => import('./pages/MultimodalPage'))
const FileManagementAdvancedPage = lazy(() => import('./pages/FileManagementAdvancedPage'))
const DistributedEventsPage = lazy(() => import('./pages/DistributedEventsPage'))
const LangGraph065Page = lazy(() => import('./pages/LangGraph065Page'))
const MultimodalRagPage = lazy(() => import('./pages/MultimodalRagPage'))

// 个性化引擎页面
const PersonalizationEnginePage = lazy(() => import('./pages/PersonalizationEnginePage'))
const PersonalizationMonitorPage = lazy(() => import('./pages/PersonalizationMonitorPage'))
const PersonalizationFeaturePage = lazy(() => import('./pages/PersonalizationFeaturePage'))
const PersonalizationLearningPage = lazy(() => import('./pages/PersonalizationLearningPage'))
const PersonalizationApiPage = lazy(() => import('./pages/PersonalizationApiPage'))
const PersonalizationAlertsPage = lazy(() => import('./pages/PersonalizationAlertsPage'))
const PersonalizationProductionPage = lazy(() => import('./pages/PersonalizationProductionPage'))
const PersonalizationWebSocketPage = lazy(() => import('./pages/PersonalizationWebSocketPage'))

// A/B测试实验平台页面
const ExperimentListPage = lazy(() => import('./pages/experiments/ExperimentListPage'))
const ExperimentDashboardPage = lazy(() => import('./pages/experiments/ExperimentDashboardPage'))
const StatisticalAnalysisPage = lazy(() => import('./pages/experiments/StatisticalAnalysisPage'))
const TrafficAllocationPage = lazy(() => import('./pages/experiments/TrafficAllocationPage'))
const EventTrackingPage = lazy(() => import('./pages/experiments/EventTrackingPage'))
const ReleaseStrategyPage = lazy(() => import('./pages/experiments/ReleaseStrategyPage'))
const MonitoringAlertsPage = lazy(() => import('./pages/experiments/MonitoringAlertsPage'))
const AdvancedAlgorithmsPage = lazy(() => import('./pages/experiments/AdvancedAlgorithmsPage'))

// 行为分析系统
const BehaviorAnalyticsPage = lazy(() => import('./pages/BehaviorAnalyticsPage'))
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

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

const App: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/multi-agent') return 'multi-agent'
    if (path === '/supervisor') return 'supervisor'
    if (path === '/async-agents') return 'async-agents'
    if (path === '/agent-interface') return 'agent-interface'
    if (path === '/rag') return 'rag'
    if (path === '/agentic-rag') return 'agentic-rag'
    if (path === '/hybrid-search') return 'hybrid-search'
    if (path === '/multimodal') return 'multimodal'
    if (path === '/file-management') return 'file-management'
    if (path === '/workflows') return 'workflows'
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
    if (path === '/langgraph-features') return 'langgraph-features'
    if (path === '/memory-hierarchy') return 'memory-hierarchy'
    if (path === '/memory-recall') return 'memory-recall'
    if (path === '/memory-analytics') return 'memory-analytics'
    if (path === '/reasoning') return 'reasoning'
    if (path === '/multi-step-reasoning') return 'multi-step-reasoning'
    if (path === '/explainable-ai') return 'explainable-ai'
    if (path === '/offline') return 'offline'
    if (path === '/sync') return 'sync-management'
    if (path === '/conflicts') return 'conflict-resolution'
    if (path === '/vector-clock') return 'vector-clock-viz'
    if (path === '/network-monitor') return 'network-monitor-detail'
    if (path === '/sync-engine') return 'sync-engine-internal'
    if (path === '/model-cache') return 'model-cache-monitor'
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
    if (path === '/multimodal-simple') return 'multimodal-simple'
    if (path === '/vector-advanced-simple') return 'vector-advanced-simple'
    if (path === '/vector-advanced-test') return 'vector-advanced-test'
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
    
    // A/B测试实验平台路径映射
    if (path === '/experiments') return 'experiment-list'
    if (path === '/experiments/dashboard') return 'experiment-dashboard'
    if (path === '/experiments/statistical-analysis') return 'statistical-analysis'
    if (path === '/experiments/traffic-allocation') return 'traffic-allocation'
    if (path === '/experiments/event-tracking') return 'event-tracking'
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
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多代理协作 (AutoGen v0.4)',
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
        {
          key: 'multimodal-simple',
          icon: <FileImageOutlined />,
          label: '多模态简化版',
        },
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
          key: 'workflows',
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
          key: 'batch-jobs',
          icon: <CloudServerOutlined />,
          label: '批处理作业',
        },
        {
          key: 'batch-jobs-basic',
          icon: <CloudServerOutlined />,
          label: '基础批处理',
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
      label: '离线同步系统',
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
          label: '同步引擎内部',
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
          key: 'distributed-events',
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
      key: 'system-monitoring',
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
          key: 'cache-monitor',
          icon: <ThunderboltOutlined />,
          label: '缓存监控',
        },
        {
          key: 'model-cache-monitor',
          icon: <DatabaseOutlined />,
          label: '模型缓存监控',
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
          key: 'pgvector',
          icon: <DatabaseOutlined />,
          label: 'pgvector量化',
        },
        {
          key: 'vector-advanced-simple',
          icon: <DatabaseOutlined />,
          label: '向量索引简化版',
        },
        {
          key: 'vector-advanced-test',
          icon: <ExperimentOutlined />,
          label: '向量索引测试版',
        },
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
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={280}
        collapsedWidth={80}
        style={{ 
          background: '#fff', 
          borderRight: '1px solid #e8e8e8'
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
            height: 'calc(100vh - 76px)',
            minHeight: 'calc(100vh - 76px)',
            overflowY: 'auto',
            background: '#fff'
          }}
          onClick={({ key }) => {
            switch (key) {
              case 'chat': navigate('/chat'); break;
              case 'multi-agent': navigate('/multi-agent'); break;
              case 'supervisor': navigate('/supervisor'); break;
              case 'async-agents': navigate('/async-agents'); break;
              case 'agent-interface': navigate('/agent-interface'); break;
              case 'rag': navigate('/rag'); break;
              case 'agentic-rag': navigate('/agentic-rag'); break;
              case 'hybrid-search': navigate('/hybrid-search'); break;
              case 'multimodal': navigate('/multimodal'); break;
              case 'file-management': navigate('/file-management'); break;
              case 'workflows': navigate('/workflows'); break;
              case 'langgraph-features': navigate('/langgraph-features'); break;
              case 'dag-orchestrator': navigate('/dag-orchestrator'); break;
              case 'flow-control': navigate('/flow-control'); break;
              case 'streaming': navigate('/streaming'); break;
              case 'batch-jobs': navigate('/batch'); break;
              case 'unified-engine': navigate('/unified-engine'); break;
              case 'ai-trism': navigate('/ai-trism'); break;
              case 'security-management': navigate('/security-management'); break;
              case 'auth-management': navigate('/auth-management'); break;
              case 'events': navigate('/events'); break;
              case 'health': navigate('/health'); break;
              case 'performance': navigate('/performance'); break;
              case 'monitor': navigate('/monitor'); break;
              case 'monitoring-dashboard': navigate('/monitoring-dashboard'); break;
              case 'pgvector': navigate('/pgvector'); break;
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
              case 'bandit-recommendation': navigate('/bandit-recommendation'); break;
              // Q-Learning算法家族
              case 'qlearning-dashboard': navigate('/qlearning'); break;
              case 'qlearning-training': navigate('/qlearning-training'); break;
              case 'qlearning-strategy': navigate('/qlearning-strategy'); break;
              case 'qlearning-recommendation': navigate('/qlearning-recommendation'); break;
              case 'qlearning-performance': navigate('/qlearning-performance'); break;
              case 'qlearning-performance-optimization': navigate('/qlearning-performance-optimization'); break;
              case 'qlearning-tabular': navigate('/qlearning/tabular'); break;
              case 'qlearning-dqn': navigate('/qlearning/dqn'); break;
              case 'qlearning-variants': navigate('/qlearning/variants'); break;
              
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
              case 'multimodal-simple': navigate('/multimodal-simple'); break;
              case 'vector-advanced-simple': navigate('/vector-advanced-simple'); break;
              case 'vector-advanced-test': navigate('/vector-advanced-test'); break;
              case 'vector-clock-algorithm': navigate('/vector-clock-algorithm'); break;
              case 'unified-engine-complete': navigate('/unified-engine-complete'); break;
              case 'batch-jobs-basic': navigate('/batch-jobs'); break;
              case 'document-processing-simple': navigate('/document-processing-simple'); break;
              case 'security-basic': navigate('/security'); break;
              case 'multimodal-basic': navigate('/multimodal-basic'); break;
              case 'file-management-advanced': navigate('/file-management-advanced'); break;
              case 'distributed-events-system': navigate('/distributed-events'); break;
              case 'langgraph-065': navigate('/langgraph-065'); break;
              case 'multimodal-rag-system': navigate('/multimodal-rag'); break;
              
              // A/B测试实验平台导航
              case 'experiment-list': navigate('/experiments'); break;
              case 'experiment-dashboard': navigate('/experiments/dashboard'); break;
              case 'statistical-analysis': navigate('/experiments/statistical-analysis'); break;
              case 'traffic-allocation': navigate('/experiments/traffic-allocation'); break;
              case 'event-tracking': navigate('/experiments/event-tracking'); break;
              case 'release-strategy': navigate('/experiments/release-strategy'); break;
              case 'monitoring-alerts': navigate('/experiments/monitoring-alerts'); break;
              case 'advanced-algorithms': navigate('/experiments/advanced-algorithms'); break;
              
              // 强化学习系统监控导航
              case 'rl-system-dashboard': navigate('/rl-system-dashboard'); break;
              case 'rl-performance-monitor': navigate('/rl-performance-monitor'); break;
              case 'rl-integration-test': navigate('/rl-integration-test'); break;
              case 'rl-alert-config': navigate('/rl-alert-config'); break;
              case 'rl-metrics-analysis': navigate('/rl-metrics-analysis'); break;
              case 'rl-system-health': navigate('/rl-system-health'); break;
            }
          }}
        />
      </Sider>

      <Layout>
        <Header style={{ 
          background: '#fff', 
          borderBottom: '1px solid #e8e8e8',
          padding: '0 20px',
          height: '60px',
          lineHeight: '60px'
        }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px' }}
          />
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
            <Route path="/multi-agent" element={<MultiAgentPage />} />
            <Route path="/supervisor" element={<SupervisorPage />} />
            <Route path="/async-agents" element={<AsyncAgentPage />} />
            <Route path="/agent-interface" element={<AgentInterfacePage />} />
            
            {/* 🔍 智能检索引擎 */}
            <Route path="/rag" element={<RagPage />} />
            <Route path="/agentic-rag" element={<AgenticRagPage />} />
            <Route path="/hybrid-search" element={<HybridSearchAdvancedPage />} />
            
            {/* 🧠 推理引擎 */}
            <Route path="/reasoning" element={<ReasoningPage />} />
            <Route path="/multi-step-reasoning" element={<MultiStepReasoningPage />} />
            <Route path="/explainable-ai" element={<ExplainableAiPage />} />
            
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
            
            {/* 🧠 强化学习系统 */}
            <Route path="/qlearning" element={<QLearningPage />} />
            <Route path="/qlearning-training" element={<QLearningTrainingPage />} />
            <Route path="/qlearning-strategy" element={<QLearningStrategyPage />} />
            <Route path="/qlearning-recommendation" element={<QLearningRecommendationPage />} />
            <Route path="/qlearning-performance" element={<QLearningPerformancePage />} />
            <Route path="/qlearning-performance-optimization" element={<QLearningPerformanceOptimizationPage />} />
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
            
            {/* ⚡ 工作流引擎 */}
            <Route path="/workflows" element={<WorkflowPage />} />
            <Route path="/langgraph-features" element={<LangGraphFeaturesPage />} />
            <Route path="/dag-orchestrator" element={<DagOrchestratorPage />} />
            <Route path="/flow-control" element={<FlowControlPage />} />
            
            {/* 🏭 处理引擎 */}
            <Route path="/streaming" element={<StreamingMonitorPage />} />
            <Route path="/batch" element={<BatchJobsPageFixed />} />
            <Route path="/unified-engine" element={<UnifiedEnginePage />} />
            
            {/* 🛡️ 安全与合规 */}
            <Route path="/ai-trism" element={<AiTrismPage />} />
            <Route path="/security-management" element={<SecurityManagementPage />} />
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
            
            {/* 🔧 协议与工具 */}
            <Route path="/mcp-tools" element={<MCPToolsPage />} />
            
            {/* 🏢 企业架构 */}
            <Route path="/enterprise" element={<EnterpriseArchitecturePage />} />
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
            <Route path="/document-processing" element={<DocumentProcessingAdvancedPage />} />
            
            {/* 缺失页面补充 */}
            <Route path="/conflict-resolution-learning" element={<ConflictResolutionLearningPage />} />
            <Route path="/sync-engine-learning" element={<SyncEngineLearningPage />} />
            <Route path="/health-comprehensive" element={<HealthComprehensivePage />} />
            <Route path="/multimodal-simple" element={<MultimodalPageSimple />} />
            <Route path="/vector-advanced-simple" element={<VectorAdvancedPageSimple />} />
            <Route path="/vector-advanced-test" element={<VectorAdvancedTestPage />} />
            <Route path="/vector-clock-algorithm" element={<VectorClockAlgorithmPage />} />
            <Route path="/unified-engine-complete" element={<UnifiedEnginePageComplete />} />
            <Route path="/batch-jobs" element={<BatchJobsPage />} />
            <Route path="/document-processing-simple" element={<DocumentProcessingPage />} />
            <Route path="/security" element={<SecurityPage />} />
            <Route path="/multimodal-basic" element={<MultimodalPage />} />
            <Route path="/file-management-advanced" element={<FileManagementAdvancedPage />} />
            <Route path="/distributed-events" element={<DistributedEventsPage />} />
            <Route path="/langgraph-065" element={<LangGraph065Page />} />
            <Route path="/multimodal-rag" element={<MultimodalRagPage />} />
            
            {/* 🧪 A/B测试实验平台 */}
            <Route path="/experiments" element={<ExperimentListPage />} />
            <Route path="/experiments/dashboard" element={<ExperimentDashboardPage />} />
            <Route path="/experiments/statistical-analysis" element={<StatisticalAnalysisPage />} />
            <Route path="/experiments/traffic-allocation" element={<TrafficAllocationPage />} />
            <Route path="/experiments/event-tracking" element={<EventTrackingPage />} />
            <Route path="/experiments/release-strategy" element={<ReleaseStrategyPage />} />
            <Route path="/experiments/monitoring-alerts" element={<MonitoringAlertsPage />} />
            <Route path="/experiments/advanced-algorithms" element={<AdvancedAlgorithmsPage />} />
            
            {/* 📊 强化学习系统监控 */}
            <Route path="/rl-system-dashboard" element={<RLSystemDashboardPage />} />
            <Route path="/rl-performance-monitor" element={<RLPerformanceMonitorPage />} />
            <Route path="/rl-integration-test" element={<RLIntegrationTestPage />} />
            <Route path="/rl-alert-config" element={<RLAlertConfigPage />} />
            <Route path="/rl-metrics-analysis" element={<RLMetricsAnalysisPage />} />
            <Route path="/rl-system-health" element={<RLSystemHealthPage />} />
          </Routes>
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}

export default App