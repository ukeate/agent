import React, { useState } from 'react'
import { Layout, Menu, Button, Typography, Space, Avatar } from 'antd'
import type { MenuProps } from 'antd'
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
  HeartOutlined,
  UserOutlined,
  TrophyOutlined,
  EyeOutlined,
  CommentOutlined,
  StarOutlined,
  LikeOutlined,
  ClusterOutlined,
  FundOutlined,
  BarChartOutlined,
  PieChartOutlined,
  DeploymentUnitOutlined,
  SyncOutlined,
  LockOutlined,
  HistoryOutlined,
  RadarChartOutlined,
  PartitionOutlined,
  GlobalOutlined,
  FileTextOutlined,
  FolderOutlined,
  CloudOutlined,
  BranchesOutlined,
  CodeOutlined,
  ConsoleSqlOutlined,
  SaveOutlined,
  SlidersOutlined,
  AppstoreOutlined,
  CalculatorOutlined,
  ImportOutlined,
  ExportOutlined,
  PictureOutlined,
  SecurityScanOutlined,
  PlayCircleOutlined,
  ScheduleOutlined,
  ShareAltOutlined,
  ShareAltOutlined as NetworkOutlined,
  TabletOutlined,
  FolderOpenOutlined,
  GoldOutlined,
  PercentageOutlined,
  AimOutlined,
  FlagOutlined,
  RocketOutlined,
  FunctionOutlined,
  StockOutlined,
  CarryOutOutlined,
  ProjectOutlined,
  BookOutlined,
  AuditOutlined,
  FieldTimeOutlined,
  TransactionOutlined,
  BuildOutlined,
  SafetyCertificateOutlined,
  UnorderedListOutlined,
  UndoOutlined,
  TranslationOutlined,
  SmileOutlined,
  AudioOutlined,
  CameraOutlined,
  WifiOutlined,
  CompressOutlined,
  ScissorOutlined,
  SolutionOutlined,
  TestTubeOutlined,
  AreaChartOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  FileSearchOutlined,
  DownloadOutlined,
  SwapOutlined,
  ReloadOutlined
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

interface MainLayoutProps {
  children: React.ReactNode
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()

  // 根据当前路径确定选中的菜单项
  const getSelectedKey = () => {
    const path = location.pathname;
    
    // 智能体系统
    if (path === '/multi-agent') return 'multi-agent'
    if (path === '/supervisor') return 'supervisor'
    if (path === '/async-agents') return 'async-agents'
    
    // 智能代理服务发现系统 (Story 10.1)
    if (path === '/service-discovery-overview') return 'service-discovery-overview'
    if (path === '/agent-registry') return 'agent-registry-management'
    if (path === '/service-routing') return 'service-routing-management'
    if (path === '/load-balancer-config') return 'load-balancer-config'
    if (path === '/service-health-monitor') return 'service-health-monitor'
    if (path === '/service-cluster-management') return 'service-cluster-management'
    if (path === '/service-performance-dashboard') return 'service-performance-dashboard'
    if (path === '/service-config-management') return 'service-config-management'
    
    // 分布式消息通信框架 (Story 10.2)
    if (path === '/distributed-message-overview') return 'distributed-message-overview'
    if (path === '/nats-cluster-management') return 'nats-cluster-management'
    if (path === '/basic-message-communication') return 'basic-message-communication'
    if (path === '/acl-protocol-management') return 'acl-protocol-management'
    if (path === '/request-response-mechanism') return 'request-response-mechanism'
    if (path === '/message-reliability-management') return 'message-reliability-management'
    if (path === '/advanced-communication-patterns') return 'advanced-communication-patterns'
    if (path === '/monitoring-performance-optimization') return 'monitoring-performance-optimization'
    
    // 故障容错和恢复系统 (Story 10.5)
    if (path === '/fault-tolerance-overview') return 'fault-tolerance-overview'
    if (path === '/fault-detection') return 'fault-detection'
    if (path === '/recovery-management') return 'recovery-management'
    if (path === '/backup-management') return 'backup-management'
    if (path === '/consistency-management') return 'consistency-management'
    if (path === '/system-monitoring') return 'system-monitoring'
    if (path === '/fault-testing') return 'fault-testing'
    
    // 智能检索引擎  
    if (path === '/rag') return 'rag'
    if (path === '/agentic-rag') return 'agentic-rag'
    if (path === '/hybrid-search-advanced') return 'hybrid-search-advanced'
    if (path === '/multimodal-rag') return 'multimodal-rag'
    
    // 动态知识图谱存储系统 (Story 8.2)
    if (path === '/kg-entity-management') return 'kg-entity-management'
    if (path === '/kg-relation-management') return 'kg-relation-management'
    if (path === '/kg-graph-query') return 'kg-graph-query'
    if (path === '/kg-incremental-update') return 'kg-incremental-update'
    if (path === '/kg-quality-assessment') return 'kg-quality-assessment'
    if (path === '/kg-performance-monitor') return 'kg-performance-monitor'
    if (path === '/kg-schema-management') return 'kg-schema-management'
    if (path === '/kg-data-migration') return 'kg-data-migration'
    
    // 推理引擎
    if (path === '/reasoning') return 'reasoning'
    if (path === '/multi-step-reasoning') return 'multi-step-reasoning'
    if (path === '/explainable-ai') return 'explainable-ai'
    
    // 推荐算法引擎
    if (path === '/bandit-recommendation') return 'bandit-recommendation'
    
    // 个性化引擎
    if (path === '/personalization-engine') return 'personalization-engine'
    if (path === '/personalization-monitor') return 'personalization-monitor'
    if (path === '/personalization-features') return 'personalization-features'
    if (path === '/personalization-learning') return 'personalization-learning'
    if (path === '/personalization-api') return 'personalization-api'
    if (path === '/personalization-alerts') return 'personalization-alerts'
    if (path === '/personalization-production') return 'personalization-production'
    if (path === '/personalization-websocket') return 'personalization-websocket'
    
    // 高级情感智能系统
    if (path === '/emotion-recognition-overview') return 'emotion-recognition-overview'
    if (path === '/text-emotion-analysis') return 'text-emotion-analysis'
    if (path === '/audio-emotion-recognition') return 'audio-emotion-recognition'
    if (path === '/visual-emotion-analysis') return 'visual-emotion-analysis'
    if (path === '/multimodal-emotion-fusion') return 'multimodal-emotion-fusion'
    if (path === '/emotion-modeling') return 'emotion-modeling'
    
    // 情感记忆管理系统 (Story 11.4)
    if (path === '/emotional-memory-management') return 'emotional-memory-management'
    if (path === '/emotional-event-analysis') return 'emotional-event-analysis'
    if (path === '/emotional-preference-learning') return 'emotional-preference-learning'
    if (path === '/emotional-trigger-patterns') return 'emotional-trigger-patterns'
    if (path === '/emotional-memory-retrieval') return 'emotional-memory-retrieval'
    
    // 社交情感理解系统 (Story 11.6)
    if (path === '/social-emotion-system') return 'social-emotion-system'
    if (path === '/emotion-flow-analysis') return 'emotion-flow-analysis'
    if (path === '/social-network-emotion-map') return 'social-network-emotion-map'
    if (path === '/cultural-context-analysis') return 'cultural-context-analysis'
    if (path === '/social-intelligence-decision') return 'social-intelligence-decision'
    if (path === '/privacy-ethics') return 'privacy-ethics'
    
    // 强化学习系统 - 主要功能
    if (path === '/qlearning') return 'qlearning'
    if (path === '/qlearning-training') return 'qlearning-training'
    if (path === '/qlearning-strategy') return 'qlearning-strategy'
    if (path === '/qlearning-recommendation') return 'qlearning-recommendation'
    if (path === '/qlearning-performance') return 'qlearning-performance'
    if (path === '/qlearning-performance-optimization') return 'qlearning-performance-optimization'
    
    // Q-Learning算法详细页面
    if (path === '/qlearning/tabular') return 'tabular-qlearning'
    if (path === '/qlearning/dqn') return 'dqn'
    if (path === '/qlearning/variants') return 'dqn-variants'
    if (path === '/qlearning/exploration-strategies') return 'exploration-strategies'
    if (path === '/qlearning/ucb-strategies') return 'ucb-strategies'
    if (path === '/qlearning/thompson-sampling') return 'thompson-sampling'
    if (path === '/qlearning/basic-rewards') return 'basic-rewards'
    if (path === '/qlearning/composite-rewards') return 'composite-rewards'
    if (path === '/qlearning/adaptive-rewards') return 'adaptive-rewards'
    if (path === '/qlearning/reward-shaping') return 'reward-shaping'
    if (path === '/qlearning/state-space') return 'state-space'
    if (path === '/qlearning/action-space') return 'action-space'
    if (path === '/qlearning/environment-simulator') return 'environment-simulator'
    if (path === '/qlearning/grid-world') return 'grid-world'
    if (path === '/qlearning/training-manager') return 'training-manager'
    if (path === '/qlearning/learning-rate-scheduler') return 'learning-rate-scheduler'
    if (path === '/qlearning/early-stopping') return 'early-stopping'
    if (path === '/qlearning/performance-tracker') return 'performance-tracker'
    
    // A/B测试实验平台
    if (path === '/experiments') return 'experiments'
    if (path === '/experiments/dashboard') return 'experiment-dashboard'
    if (path === '/experiments/statistical-analysis') return 'statistical-analysis'
    if (path === '/experiments/traffic-allocation') return 'traffic-allocation'
    if (path === '/experiments/event-tracking') return 'event-tracking'
    if (path === '/experiments/release-strategy') return 'release-strategy'
    if (path === '/experiments/monitoring-alerts') return 'monitoring-alerts'
    if (path === '/experiments/advanced-algorithms') return 'advanced-algorithms'
    if (path === '/experiment-list') return 'experiment-list'
    if (path === '/experiment-report') return 'experiment-report'
    if (path === '/experiment-config') return 'experiment-config'
    if (path === '/user-segmentation') return 'user-segmentation'
    if (path === '/conversion-funnel') return 'conversion-funnel'
    if (path === '/cohort-analysis') return 'cohort-analysis'
    if (path === '/release-strategies') return 'release-strategies'
    if (path === '/canary-deployment') return 'canary-deployment'
    if (path === '/blue-green-deployment') return 'blue-green-deployment'
    if (path === '/feature-flags') return 'feature-flags'
    if (path === '/risk-assessment') return 'risk-assessment'
    if (path === '/auto-rollback') return 'auto-rollback'
    if (path === '/experiment-monitoring') return 'experiment-monitoring'
    if (path === '/alert-management') return 'alert-management'
    if (path === '/data-quality') return 'data-quality'
    if (path === '/sample-size-calculator') return 'sample-size-calculator'
    if (path === '/power-analysis') return 'power-analysis'
    if (path === '/confidence-intervals') return 'confidence-intervals'
    if (path === '/hypothesis-testing') return 'hypothesis-testing'
    if (path === '/bayesian-analysis') return 'bayesian-analysis'
    if (path === '/sequential-testing') return 'sequential-testing'
    if (path === '/multi-armed-bandit') return 'multi-armed-bandit'
    
    // 用户行为分析系统
    if (path === '/behavior-analytics') return 'behavior-analytics'
    if (path === '/behavior-analytics/realtime') return 'realtime-monitor'
    if (path === '/behavior-analytics/sessions') return 'session-manage'
    if (path === '/behavior-analytics/events') return 'event-data-manage'
    if (path === '/behavior-analytics/reports') return 'report-center'
    if (path === '/behavior-analytics/export') return 'data-export'
    if (path === '/behavior-analytics/config') return 'system-config'
    
    // 用户反馈学习系统
    if (path === '/feedback-system') return 'feedback-system'
    if (path === '/feedback-analytics') return 'feedback-analytics'
    if (path === '/user-feedback-profiles') return 'user-feedback-profiles'
    if (path === '/item-feedback-analysis') return 'item-feedback-analysis'
    if (path === '/feedback-quality-monitor') return 'feedback-quality-monitor'
    
    // 模型服务部署平台 (Story 9.6)
    if (path === '/model-registry') return 'model-registry'
    if (path === '/model-inference') return 'model-inference'
    if (path === '/model-deployment') return 'model-deployment'
    if (path === '/model-monitoring') return 'model-monitoring'
    if (path === '/online-learning') return 'online-learning'
    
    // 模型评估和基准测试系统 (Story 9.4)
    if (path === '/model-evaluation-overview') return 'model-evaluation-overview'
    if (path === '/model-performance-benchmark') return 'model-performance-benchmark'
    if (path === '/evaluation-engine-management') return 'evaluation-engine-management'
    if (path === '/benchmark-suite-management') return 'benchmark-suite-management'
    if (path === '/evaluation-tasks-monitor') return 'evaluation-tasks-monitor'
    if (path === '/evaluation-reports-center') return 'evaluation-reports-center'
    if (path === '/evaluation-api-management') return 'evaluation-api-management'
    if (path === '/model-comparison-dashboard') return 'model-comparison-dashboard'
    if (path === '/benchmark-glue-management') return 'benchmark-glue-management'
    if (path === '/benchmark-superglue-management') return 'benchmark-superglue-management'
    if (path === '/benchmark-mmlu-management') return 'benchmark-mmlu-management'
    if (path === '/benchmark-humaneval-management') return 'benchmark-humaneval-management'
    if (path === '/benchmark-hellaswag-management') return 'benchmark-hellaswag-management'
    if (path === '/benchmark-custom-management') return 'benchmark-custom-management'
    if (path === '/evaluation-metrics-config') return 'evaluation-metrics-config'
    if (path === '/evaluation-performance-monitor') return 'evaluation-performance-monitor'
    if (path === '/evaluation-batch-processing') return 'evaluation-batch-processing'
    if (path === '/evaluation-regression-detection') return 'evaluation-regression-detection'
    if (path === '/evaluation-quality-assurance') return 'evaluation-quality-assurance'
    if (path === '/evaluation-automation-pipeline') return 'evaluation-automation-pipeline'
    if (path === '/evaluation-alerts-management') return 'evaluation-alerts-management'
    if (path === '/evaluation-data-management') return 'evaluation-data-management'
    if (path === '/evaluation-resource-monitor') return 'evaluation-resource-monitor'
    if (path === '/evaluation-job-scheduler') return 'evaluation-job-scheduler'
    if (path === '/evaluation-results-analysis') return 'evaluation-results-analysis'
    if (path === '/evaluation-export-import') return 'evaluation-export-import'
    if (path === '/evaluation-version-control') return 'evaluation-version-control'
    if (path === '/evaluation-compliance-audit') return 'evaluation-compliance-audit'
    if (path === '/evaluation-security-management') return 'evaluation-security-management'
    
    // 模型压缩和量化工具
    if (path === '/model-compression-overview') return 'model-compression-overview'
    if (path === '/quantization-manager') return 'quantization-manager'
    if (path === '/quantization-ptq') return 'quantization-ptq'
    if (path === '/quantization-qat') return 'quantization-qat'
    if (path === '/quantization-advanced') return 'quantization-advanced'
    if (path === '/quantization-config') return 'quantization-config'
    if (path === '/knowledge-distillation') return 'knowledge-distillation'
    if (path === '/distillation-trainer') return 'distillation-trainer'
    if (path === '/distillation-strategies') return 'distillation-strategies'
    if (path === '/distillation-monitor') return 'distillation-monitor'
    if (path === '/model-pruning') return 'model-pruning'
    if (path === '/pruning-structured') return 'pruning-structured'
    if (path === '/pruning-unstructured') return 'pruning-unstructured'
    if (path === '/pruning-strategies') return 'pruning-strategies'
    if (path === '/compression-pipeline') return 'compression-pipeline'
    if (path === '/compression-jobs') return 'compression-jobs'
    if (path === '/compression-monitor') return 'compression-monitor'
    if (path === '/compression-scheduler') return 'compression-scheduler'
    if (path === '/compression-evaluator') return 'compression-evaluator'
    if (path === '/model-comparison') return 'model-comparison'
    if (path === '/performance-analysis') return 'performance-analysis'
    if (path === '/compression-reports') return 'compression-reports'
    if (path === '/hardware-benchmark') return 'hardware-benchmark'
    if (path === '/inference-optimization') return 'inference-optimization'
    if (path === '/deployment-optimization') return 'deployment-optimization'
    if (path === '/strategy-recommendation') return 'strategy-recommendation'
    if (path === '/compression-templates') return 'compression-templates'
    if (path === '/model-registry-compression') return 'model-registry-compression'
    
    // 记忆管理系统
    if (path === '/memory-hierarchy') return 'memory-hierarchy'
    if (path === '/memory-recall') return 'memory-recall'
    if (path === '/memory-analytics') return 'memory-analytics'
    
    // 多模态处理
    if (path === '/multimodal') return 'multimodal'
    if (path === '/multimodal-complete') return 'multimodal-complete'
    if (path === '/multimodal-simple') return 'multimodal-simple'
    if (path === '/file-management-advanced') return 'file-management-advanced'
    if (path === '/file-management-complete') return 'file-management-complete'
    if (path === '/document-processing') return 'document-processing'
    if (path === '/document-processing-advanced') return 'document-processing-advanced'
    
    // 工作流引擎
    if (path === '/workflows') return 'workflows'
    if (path === '/langgraph-features') return 'langgraph-features'
    if (path === '/langgraph-065') return 'langgraph-065'
    if (path === '/dag-orchestrator') return 'dag-orchestrator'
    if (path === '/flow-control') return 'flow-control'
    
    // 离线能力与同步机制
    if (path === '/offline') return 'offline'
    if (path === '/sync-management') return 'sync-management'
    if (path === '/conflict-resolution') return 'conflict-resolution'
    if (path === '/conflict-resolution-learning') return 'conflict-resolution-learning'
    if (path === '/vector-clock-algorithm') return 'vector-clock-algorithm'
    if (path === '/vector-clock-visualization') return 'vector-clock-visualization'
    if (path === '/network-monitor-detail') return 'network-monitor-detail'
    if (path === '/sync-engine-internal') return 'sync-engine-internal'
    if (path === '/sync-engine-learning') return 'sync-engine-learning'
    if (path === '/model-cache-monitor') return 'model-cache-monitor'
    
    // 处理引擎
    if (path === '/streaming-monitor') return 'streaming-monitor'
    if (path === '/batch-jobs') return 'batch-jobs'
    if (path === '/batch-jobs-fixed') return 'batch-jobs-fixed'
    if (path === '/unified-engine') return 'unified-engine'
    if (path === '/unified-engine-complete') return 'unified-engine-complete'
    if (path === '/unified-monitor') return 'unified-monitor'
    
    // 安全与合规
    if (path === '/ai-trism') return 'ai-trism'
    if (path === '/security-management') return 'security-management'
    if (path === '/security') return 'security'
    if (path === '/auth-management') return 'auth-management'
    
    // 事件与监控
    if (path === '/event-dashboard') return 'event-dashboard'
    if (path === '/distributed-events') return 'distributed-events'
    if (path === '/health-monitor') return 'health-monitor'
    if (path === '/health-comprehensive') return 'health-comprehensive'
    if (path === '/performance-monitor') return 'performance-monitor'
    if (path === '/monitoring-dashboard') return 'monitoring-dashboard'
    if (path === '/cache-monitor') return 'cache-monitor'
    
    // 数据存储
    if (path === '/pgvector') return 'pgvector'
    if (path === '/vector-advanced') return 'vector-advanced'
    if (path === '/vector-advanced-simple') return 'vector-advanced-simple'
    if (path === '/vector-advanced-test') return 'vector-advanced-test'
    
    // 协议与工具
    if (path === '/mcp-tools') return 'mcp-tools'
    
    // 企业架构
    if (path === '/enterprise-architecture') return 'enterprise-architecture'
    if (path === '/enterprise-config') return 'enterprise-config'
    if (path === '/architecture-debug') return 'architecture-debug'
    
    // 开发测试
    if (path === '/structured-error') return 'structured-error'
    if (path === '/test-coverage') return 'test-coverage'
    if (path === '/integration-test') return 'integration-test'
    if (path === '/testing-suite') return 'testing-suite'
    
    if (path === '/chat' || path === '/') return 'chat'
    return 'chat'
  }

  const menuItems: MenuProps['items'] = [
    // 🤖 智能体系统
    {
      key: 'agents-group',
      label: '🤖 智能体系统',
      type: 'group',
    },
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '单代理对话',
    },
    {
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多代理协作 (AutoGen)',
      children: [
        {
          key: 'multi-agent',
          icon: <TeamOutlined />,
          label: '多代理协作',
        },
        {
          key: 'async-agents',
          icon: <ThunderboltOutlined />,
          label: '异步智能体 (v0.4)',
        },
        {
          key: 'flow-control',
          icon: <ThunderboltOutlined />,
          label: '流控背压监控',
        },
        {
          key: 'distributed-events-multi-agent',
          icon: <ShareAltOutlined />,
          label: '分布式事件',
        },
      ],
    },
    {
      key: 'supervisor',
      icon: <ControlOutlined />,
      label: '监督者模式 (Supervisor)',
    },
    // 🌐 智能代理服务发现系统 (Story 10.1)
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
      ],
    },

    // 📡 分布式消息通信框架 (Story 10.2)
    {
      key: 'distributed-message-communication-framework',
      icon: <NetworkOutlined />,
      label: '📡 分布式消息通信框架',
      children: [
        {
          key: 'distributed-message-overview',
          icon: <NetworkOutlined />,
          label: '消息通信总览',
        },
        {
          key: 'nats-cluster-management',
          icon: <ClusterOutlined />,
          label: 'NATS集群管理',
        },
        {
          key: 'basic-message-communication',
          icon: <MessageOutlined />,
          label: '基础消息通信',
        },
        {
          key: 'acl-protocol-management',
          icon: <ApiOutlined />,
          label: 'ACL协议管理',
        },
        {
          key: 'request-response-mechanism',
          icon: <SwapOutlined />,
          label: '请求响应机制',
        },
        {
          key: 'message-reliability-management',
          icon: <SafetyCertificateOutlined />,
          label: '消息可靠性管理',
        },
        {
          key: 'advanced-communication-patterns',
          icon: <ShareAltOutlined />,
          label: '高级通信模式',
        },
        {
          key: 'monitoring-performance-optimization',
          icon: <MonitorOutlined />,
          label: '监控和性能优化',
        },
      ],
    },

    // 🛡️ 故障容错和恢复系统 (Story 10.5)
    {
      key: 'fault-tolerance-system',
      icon: <SafetyCertificateOutlined />,
      label: '🛡️ 故障容错和恢复系统',
      children: [
        {
          key: 'fault-tolerance-overview',
          icon: <DashboardOutlined />,
          label: '故障容错系统总览',
        },
        {
          key: 'fault-detection',
          icon: <MonitorOutlined />,
          label: '故障检测监控',
        },
        {
          key: 'recovery-management',
          icon: <ReloadOutlined />,
          label: '恢复管理中心',
        },
        {
          key: 'backup-management',
          icon: <DatabaseOutlined />,
          label: '备份管理系统',
        },
        {
          key: 'consistency-management',
          icon: <SyncOutlined />,
          label: '一致性管理',
        },
        {
          key: 'system-monitoring',
          icon: <MonitorOutlined />,
          label: '系统监控平台',
        },
        {
          key: 'fault-testing',
          icon: <ExperimentOutlined />,
          label: '故障测试平台',
        },
      ],
    },

    // 🔍 智能检索引擎
    {
      key: 'rag-group',
      label: '🔍 智能检索引擎',
      type: 'group',
    },

    {
      key: 'rag',
      icon: <SearchOutlined />,
      label: 'RAG检索系统',
      children: [
        {
          key: 'rag',
          icon: <SearchOutlined />,
          label: '基础RAG检索',
        },
        {
          key: 'agentic-rag',
          icon: <RobotOutlined />,
          label: 'Agentic RAG智能检索',
        },
        {
          key: 'hybrid-search-advanced',
          icon: <ClusterOutlined />,
          label: '混合搜索引擎',
        },
        {
          key: 'multimodal-rag',
          icon: <FileImageOutlined />,
          label: '多模态RAG系统',
        },
      ],
    },

    // 🧠 推理引擎
    {
      key: 'reasoning-group',
      label: '🧠 推理引擎',
      type: 'group',
    },
    {
      key: 'reasoning',
      icon: <BulbOutlined />,
      label: '智能推理系统',
      children: [
        {
          key: 'reasoning',
          icon: <BulbOutlined />,
          label: '链式推理 (CoT)',
        },
        {
          key: 'multi-step-reasoning',
          icon: <BranchesOutlined />,
          label: '多步推理工作流',
        },
        {
          key: 'explainable-ai',
          icon: <EyeOutlined />,
          label: '可解释AI决策',
        },
      ],
    },

    // 🗺️ 知识图谱引擎
    {
      key: 'knowledge-graph-group',
      label: '🗺️ 知识图谱引擎',
      type: 'group',
    },
    {
      key: 'knowledge-extraction',
      icon: <NodeIndexOutlined />,
      label: '知识抽取系统 (Story 8.1)',
      children: [
        {
          key: 'knowledge-extraction-overview',
          icon: <DashboardOutlined />,
          label: '知识抽取总览',
        },
        {
          key: 'entity-recognition',
          icon: <BranchesOutlined />,
          label: '实体识别管理',
        },
        {
          key: 'relation-extraction',
          icon: <ShareAltOutlined />,
          label: '关系抽取管理',
        },
        {
          key: 'entity-linking',
          icon: <GlobalOutlined />,
          label: '实体链接管理',
        },
        {
          key: 'multilingual-processing',
          icon: <TranslationOutlined />,
          label: '多语言处理',
        },
      ],
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
          icon: <SyncOutlined />,
          label: '增量更新系统',
        },
        {
          key: 'kg-quality-assessment',
          icon: <SafetyCertificateOutlined />,
          label: '质量评估管理',
        },
        {
          key: 'kg-performance-monitor',
          icon: <ThunderboltOutlined />,
          label: '性能监控优化',
        },
        {
          key: 'kg-schema-management',
          icon: <SettingOutlined />,
          label: '图模式管理',
        },
        {
          key: 'kg-data-migration',
          icon: <ExportOutlined />,
          label: '数据迁移工具',
        },
      ],
    },
    {
      key: 'knowledge-graph-management',
      icon: <ClusterOutlined />,
      label: '知识图谱管理',
      children: [
        {
          key: 'knowledge-graph-visualization',
          icon: <NodeIndexOutlined />,
          label: '图谱可视化',
        },
        {
          key: 'knowledge-graph-query',
          icon: <SearchOutlined />,
          label: '图谱查询引擎',
        },
        {
          key: 'knowledge-graph-analytics',
          icon: <BarChartOutlined />,
          label: '图谱分析统计',
        },
        {
          key: 'knowledge-graph-export',
          icon: <ExportOutlined />,
          label: '图谱数据导出',
        },
      ],
    },
    {
      key: 'knowledge-batch-processing',
      icon: <CloudServerOutlined />,
      label: '批量处理引擎',
      children: [
        {
          key: 'knowledge-batch-jobs',
          icon: <DatabaseOutlined />,
          label: '批处理作业管理',
        },
        {
          key: 'knowledge-batch-monitor',
          icon: <MonitorOutlined />,
          label: '批处理监控',
        },
        {
          key: 'knowledge-performance-optimization',
          icon: <ThunderboltOutlined />,
          label: '性能优化中心',
        },
        {
          key: 'knowledge-cache-management',
          icon: <SaveOutlined />,
          label: '缓存管理',
        },
      ],
    },
    {
      key: 'knowledge-quality-management',
      icon: <SafetyCertificateOutlined />,
      label: '知识质量管理',
      children: [
        {
          key: 'knowledge-validation',
          icon: <CheckCircleOutlined />,
          label: '知识验证评估',
        },
        {
          key: 'knowledge-confidence-analysis',
          icon: <LineChartOutlined />,
          label: '置信度分析',
        },
        {
          key: 'knowledge-error-analysis',
          icon: <ExceptionOutlined />,
          label: '错误分析报告',
        },
        {
          key: 'knowledge-model-comparison',
          icon: <CompareOutlined />,
          label: '模型对比评测',
        },
      ],
    },

    // 🎯 推荐与学习系统  
    {
      key: 'recommendation-group',
      label: '🎯 推荐与学习系统',
      type: 'group',
    },
    {
      key: 'bandit-recommendation',
      icon: <FundOutlined />,
      label: '多臂老虎机推荐引擎',
    },
    {
      key: 'personalization-engine',
      icon: <RocketOutlined />,
      label: '实时个性化引擎',
      children: [
        {
          key: 'personalization-engine',
          icon: <RocketOutlined />,
          label: '引擎控制台',
        },
        {
          key: 'personalization-monitor',
          icon: <DashboardOutlined />,
          label: '性能监控',
        },
        {
          key: 'personalization-features',
          icon: <FunctionOutlined />,
          label: '特征工程',
        },
        {
          key: 'personalization-learning',
          icon: <BranchesOutlined />,
          label: '在线学习',
        },
        {
          key: 'personalization-api',
          icon: <ApiOutlined />,
          label: 'API管理',
        },
        {
          key: 'personalization-alerts',
          icon: <BellOutlined />,
          label: '告警管理',
        },
        {
          key: 'personalization-production',
          icon: <MonitorOutlined />,
          label: '生产监控',
        },
        {
          key: 'personalization-websocket',
          icon: <WifiOutlined />,
          label: 'WebSocket实时推荐',
        },
      ],
    },
    {
      key: 'emotion-intelligence',
      icon: <SmileOutlined />,
      label: '高级情感智能系统',
      children: [
        {
          key: 'emotion-recognition-overview',
          icon: <HeartOutlined />,
          label: '情感识别引擎总览',
        },
        {
          key: 'text-emotion-analysis',
          icon: <FileTextOutlined />,
          label: '文本情感分析',
        },
        {
          key: 'audio-emotion-recognition',
          icon: <AudioOutlined />,
          label: '语音情感识别',
        },
        {
          key: 'visual-emotion-analysis',
          icon: <CameraOutlined />,
          label: '视觉情感分析',
        },
        {
          key: 'multimodal-emotion-fusion',
          icon: <MergeCellsOutlined />,
          label: '多模态情感融合',
        },
        {
          key: 'emotion-modeling',
          icon: <RadarChartOutlined />,
          label: '情感状态建模系统',
        },
        {
          type: 'divider',
        },
        {
          key: 'emotional-memory-management',
          icon: <BulbOutlined />,
          label: '📊 情感记忆管理系统',
        },
        {
          key: 'emotional-event-analysis',
          icon: <LineChartOutlined />,
          label: '事件分析引擎',
        },
        {
          key: 'emotional-preference-learning',
          icon: <UserOutlined />,
          label: '个人偏好学习',
        },
        {
          key: 'emotional-trigger-patterns',
          icon: <ExperimentOutlined />,
          label: '触发模式识别',
        },
        {
          key: 'emotional-memory-retrieval',
          icon: <SearchOutlined />,
          label: '记忆检索系统',
        },
        {
          type: 'divider',
        },
        {
          key: 'emotional-intelligence-decision-engine',
          icon: <BulbOutlined />,
          label: '情感智能决策引擎',
        },
        {
          key: 'emotional-risk-assessment-dashboard',
          icon: <ExclamationCircleOutlined />,
          label: '风险评估仪表盘',
        },
        {
          key: 'crisis-detection-support',
          icon: <AlertOutlined />,
          label: '危机检测和支持',
        },
        {
          key: 'intervention-strategy-management',
          icon: <ToolOutlined />,
          label: '干预策略管理',
        },
        {
          key: 'emotional-health-monitoring-dashboard',
          icon: <HeartOutlined />,
          label: '健康监测仪表盘',
        },
        {
          key: 'decision-history-analysis',
          icon: <HistoryOutlined />,
          label: '决策历史分析',
        },
        {
          type: 'divider',
        },
        {
          key: 'social-emotion-system',
          icon: <DashboardOutlined />,
          label: '🌟 社交情感理解系统',
        },
        {
          key: 'emotion-flow-analysis',
          icon: <LineChartOutlined />,
          label: '情感流分析',
        },
        {
          key: 'social-network-emotion-map',
          icon: <TeamOutlined />,
          label: '社交网络情感地图',
        },
        {
          key: 'cultural-context-analysis',
          icon: <GlobalOutlined />,
          label: '文化背景分析',
        },
        {
          key: 'social-intelligence-decision',
          icon: <BulbOutlined />,
          label: '社交智能决策',
        },
        {
          key: 'privacy-ethics',
          icon: <ShieldOutlined />,
          label: '隐私保护与伦理',
        },
      ],
    },
    {
      key: 'qlearning-system',
      icon: <ExperimentOutlined />,
      label: 'Q-Learning强化学习',
      children: [
        {
          key: 'qlearning',
          icon: <ExperimentOutlined />,
          label: '智能体管理',
        },
        {
          key: 'qlearning-training',
          icon: <LineChartOutlined />,
          label: '训练监控',
        },
        {
          key: 'qlearning-strategy',
          icon: <BulbOutlined />,
          label: '策略推理',
        },
        {
          key: 'qlearning-recommendation',
          icon: <TrophyOutlined />,
          label: '混合推荐',
        },
        {
          key: 'qlearning-performance',
          icon: <MonitorOutlined />,
          label: '性能分析',
        },
        {
          key: 'qlearning-performance-optimization',
          icon: <RiseOutlined />,
          label: '性能优化',
        },
      ],
    },
    {
      key: 'qlearning-algorithms',
      icon: <CodeOutlined />,
      label: 'Q-Learning算法库',
      children: [
        {
          key: 'tabular-qlearning',
          icon: <ConsoleSqlOutlined />,
          label: '表格Q-Learning',
        },
        {
          key: 'dqn',
          icon: <DeploymentUnitOutlined />,
          label: 'DQN深度Q网络',
        },
        {
          key: 'dqn-variants',
          icon: <NetworkOutlined />,
          label: 'DQN变种算法',
        },
        {
          key: 'exploration-strategies',
          icon: <RadarChartOutlined />,
          label: '自适应探索策略',
        },
        {
          key: 'ucb-strategies',
          icon: <BarChartOutlined />,
          label: 'UCB策略算法',
        },
        {
          key: 'thompson-sampling',
          icon: <PieChartOutlined />,
          label: '汤普森采样',
        },
      ],
    },
    {
      key: 'rl-components',
      icon: <SettingOutlined />,
      label: '强化学习组件',
      children: [
        {
          key: 'basic-rewards',
          icon: <StarOutlined />,
          label: '基础奖励机制',
        },
        {
          key: 'composite-rewards',
          icon: <ClusterOutlined />,
          label: '复合奖励机制',
        },
        {
          key: 'adaptive-rewards',
          icon: <SlidersOutlined />,
          label: '自适应奖励',
        },
        {
          key: 'reward-shaping',
          icon: <InteractionOutlined />,
          label: '奖励塑形',
        },
        {
          key: 'state-space',
          icon: <PartitionOutlined />,
          label: '状态空间设计',
        },
        {
          key: 'action-space',
          icon: <BranchesOutlined />,
          label: '动作空间定义',
        },
        {
          key: 'environment-simulator',
          icon: <GlobalOutlined />,
          label: '环境模拟器',
        },
        {
          key: 'grid-world',
          icon: <DatabaseOutlined />,
          label: '网格世界',
        },
        {
          key: 'training-manager',
          icon: <SaveOutlined />,
          label: '训练管理器',
        },
        {
          key: 'learning-rate-scheduler',
          icon: <RiseOutlined />,
          label: '学习率调度器',
        },
        {
          key: 'early-stopping',
          icon: <AlertOutlined />,
          label: '早停机制',
        },
        {
          key: 'performance-tracker',
          icon: <LineChartOutlined />,
          label: '性能跟踪器',
        },
      ],
    },
    {
      key: 'feedback-system',
      icon: <HeartOutlined />,
      label: '用户反馈学习系统',
      children: [
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
      ],
    },

    // 📊 模型评估和基准测试系统 (Story 9.4)
    {
      key: 'model-evaluation-group',
      label: '📊 模型评估和基准测试系统',
      type: 'group',
    },
    {
      key: 'model-evaluation-overview',
      icon: <ModelOutlined />,
      label: '模型评估总览',
    },
    {
      key: 'model-performance-benchmark',
      icon: <TrophyOutlined />,
      label: '性能基准测试',
    },
    {
      key: 'evaluation-engine-system',
      icon: <ExperimentOutlined />,
      label: '评估引擎管理',
      children: [
        {
          key: 'evaluation-engine-management',
          icon: <ControlOutlined />,
          label: '引擎控制中心',
        },
        {
          key: 'evaluation-tasks-monitor',
          icon: <MonitorOutlined />,
          label: '任务监控面板',
        },
        {
          key: 'evaluation-batch-processing',
          icon: <DatabaseOutlined />,
          label: '批量处理管理',
        },
        {
          key: 'evaluation-job-scheduler',
          icon: <ScheduleOutlined />,
          label: '任务调度器',
        },
      ],
    },
    {
      key: 'benchmark-management-system',
      icon: <BenchmarkOutlined />,
      label: '基准测试管理',
      children: [
        {
          key: 'benchmark-suite-management',
          icon: <AppstoreOutlined />,
          label: '测试套件管理',
        },
        {
          key: 'benchmark-glue-management',
          icon: <FileTextOutlined />,
          label: 'GLUE基准管理',
        },
        {
          key: 'benchmark-superglue-management',
          icon: <RocketOutlined />,
          label: 'SuperGLUE基准管理',
        },
        {
          key: 'benchmark-mmlu-management',
          icon: <BookOutlined />,
          label: 'MMLU基准管理',
        },
        {
          key: 'benchmark-humaneval-management',
          icon: <CodeOutlined />,
          label: 'HumanEval基准管理',
        },
        {
          key: 'benchmark-hellaswag-management',
          icon: <BulbOutlined />,
          label: 'HellaSwag基准管理',
        },
        {
          key: 'benchmark-custom-management',
          icon: <SolutionOutlined />,
          label: '自定义基准管理',
        },
      ],
    },
    {
      key: 'evaluation-analysis-system',
      icon: <AreaChartOutlined />,
      label: '评估分析系统',
      children: [
        {
          key: 'model-comparison-dashboard',
          icon: <CompareOutlined />,
          label: '模型对比分析',
        },
        {
          key: 'evaluation-results-analysis',
          icon: <DotChartOutlined />,
          label: '结果深度分析',
        },
        {
          key: 'evaluation-regression-detection',
          icon: <AlertOutlined />,
          label: '回归检测系统',
        },
        {
          key: 'evaluation-quality-assurance',
          icon: <SafetyCertificateOutlined />,
          label: '质量保证管控',
        },
      ],
    },
    {
      key: 'evaluation-reports-system',
      icon: <FileTextOutlined />,
      label: '报告生成系统',
      children: [
        {
          key: 'evaluation-reports-center',
          icon: <FolderOutlined />,
          label: '报告生成中心',
        },
        {
          key: 'evaluation-export-import',
          icon: <ExportOutlined />,
          label: '数据导入导出',
        },
        {
          key: 'evaluation-version-control',
          icon: <BranchesOutlined />,
          label: '版本控制管理',
        },
      ],
    },
    {
      key: 'evaluation-monitoring-system',
      icon: <RadarChartOutlined />,
      label: '监控与运维',
      children: [
        {
          key: 'evaluation-performance-monitor',
          icon: <LineChartOutlined />,
          label: '性能监控面板',
        },
        {
          key: 'evaluation-resource-monitor',
          icon: <CloudServerOutlined />,
          label: '资源使用监控',
        },
        {
          key: 'evaluation-alerts-management',
          icon: <BellOutlined />,
          label: '告警管理系统',
        },
        {
          key: 'evaluation-automation-pipeline',
          icon: <DeploymentUnitOutlined />,
          label: '自动化流水线',
        },
      ],
    },
    {
      key: 'evaluation-configuration-system',
      icon: <SettingOutlined />,
      label: '配置与管理',
      children: [
        {
          key: 'evaluation-metrics-config',
          icon: <SlidersOutlined />,
          label: '评估指标配置',
        },
        {
          key: 'evaluation-data-management',
          icon: <DatabaseOutlined />,
          label: '数据集管理',
        },
        {
          key: 'evaluation-api-management',
          icon: <ApiOutlined />,
          label: 'API接口管理',
        },
        {
          key: 'evaluation-security-management',
          icon: <ShieldOutlined />,
          label: '安全权限管理',
        },
        {
          key: 'evaluation-compliance-audit',
          icon: <AuditOutlined />,
          label: '合规审计管理',
        },
      ],
    },

    // 🚀 模型服务部署平台 (Story 9.6)
    {
      key: 'model-service-group',
      label: '🚀 模型服务部署平台',
      type: 'group',
    },
    {
      key: 'model-service-system',
      icon: <RocketOutlined />,
      label: '模型服务部署平台',
      children: [
        {
          key: 'model-registry',
          icon: <DatabaseOutlined />,
          label: '模型注册中心',
        },
        {
          key: 'model-inference',
          icon: <RocketOutlined />,
          label: '模型推理服务',
        },
        {
          key: 'model-deployment',
          icon: <CloudServerOutlined />,
          label: '部署管理',
        },
        {
          key: 'model-monitoring',
          icon: <RadarChartOutlined />,
          label: '监控与告警',
        },
        {
          key: 'online-learning',
          icon: <ExperimentOutlined />,
          label: '在线学习与A/B测试',
        },
      ],
    },

    // 🚀 模型压缩和量化工具 (Story 9.2)
    {
      key: 'model-compression-group',
      label: '🚀 模型压缩和量化工具',
      type: 'group',
    },
    {
      key: 'model-compression-overview',
      icon: <CompressOutlined />,
      label: '模型压缩总览',
    },
    {
      key: 'quantization-system',
      icon: <CompressOutlined />,
      label: '量化压缩引擎',
      children: [
        {
          key: 'quantization-manager',
          icon: <ControlOutlined />,
          label: '量化管理中心',
        },
        {
          key: 'quantization-ptq',
          icon: <ThunderboltOutlined />,
          label: '后训练量化 (PTQ)',
        },
        {
          key: 'quantization-qat',
          icon: <ExperimentOutlined />,
          label: '量化感知训练 (QAT)',
        },
        {
          key: 'quantization-advanced',
          icon: <RocketOutlined />,
          label: '高级量化算法',
        },
        {
          key: 'quantization-config',
          icon: <SettingOutlined />,
          label: '量化配置管理',
        },
      ],
    },
    {
      key: 'distillation-system',
      icon: <ExperimentFilled />,
      label: '知识蒸馏引擎',
      children: [
        {
          key: 'knowledge-distillation',
          icon: <BulbOutlined />,
          label: '知识蒸馏管理',
        },
        {
          key: 'distillation-trainer',
          icon: <BuildOutlined />,
          label: '蒸馏训练器',
        },
        {
          key: 'distillation-strategies',
          icon: <SolutionOutlined />,
          label: '蒸馏策略配置',
        },
        {
          key: 'distillation-monitor',
          icon: <MonitorOutlined />,
          label: '蒸馏监控面板',
        },
      ],
    },
    {
      key: 'pruning-system',
      icon: <ScissorOutlined />,
      label: '模型剪枝引擎',
      children: [
        {
          key: 'model-pruning',
          icon: <ScissorOutlined />,
          label: '剪枝管理中心',
        },
        {
          key: 'pruning-structured',
          icon: <PartitionOutlined />,
          label: '结构化剪枝',
        },
        {
          key: 'pruning-unstructured',
          icon: <ClusterOutlined />,
          label: '非结构化剪枝',
        },
        {
          key: 'pruning-strategies',
          icon: <AimOutlined />,
          label: '剪枝策略配置',
        },
      ],
    },
    {
      key: 'compression-pipeline-system',
      icon: <DeploymentUnitOutlined />,
      label: '压缩流水线管理',
      children: [
        {
          key: 'compression-pipeline',
          icon: <BranchesOutlined />,
          label: '流水线总览',
        },
        {
          key: 'compression-jobs',
          icon: <ProjectOutlined />,
          label: '压缩任务管理',
        },
        {
          key: 'compression-monitor',
          icon: <MonitorOutlined />,
          label: '任务监控面板',
        },
        {
          key: 'compression-scheduler',
          icon: <ScheduleOutlined />,
          label: '任务调度器',
        },
      ],
    },
    {
      key: 'compression-evaluation',
      icon: <BarChartOutlined />,
      label: '压缩评估与对比',
      children: [
        {
          key: 'compression-evaluator',
          icon: <DotChartOutlined />,
          label: '压缩评估器',
        },
        {
          key: 'model-comparison',
          icon: <CompareOutlined />,
          label: '模型对比分析',
        },
        {
          key: 'performance-analysis',
          icon: <LineChartOutlined />,
          label: '性能分析报告',
        },
        {
          key: 'compression-reports',
          icon: <FileTextOutlined />,
          label: '压缩效果报告',
        },
      ],
    },
    {
      key: 'hardware-optimization',
      icon: <ThunderboltOutlined />,
      label: '硬件性能优化',
      children: [
        {
          key: 'hardware-benchmark',
          icon: <RiseOutlined />,
          label: '硬件基准测试',
        },
        {
          key: 'inference-optimization',
          icon: <RocketOutlined />,
          label: '推理引擎优化',
        },
        {
          key: 'deployment-optimization',
          icon: <CloudServerOutlined />,
          label: '部署优化配置',
        },
      ],
    },
    {
      key: 'strategy-management',
      icon: <BulbOutlined />,
      label: '智能策略推荐',
      children: [
        {
          key: 'strategy-recommendation',
          icon: <SolutionOutlined />,
          label: '策略推荐引擎',
        },
        {
          key: 'compression-templates',
          icon: <BookOutlined />,
          label: '压缩模板库',
        },
        {
          key: 'model-registry-compression',
          icon: <DatabaseOutlined />,
          label: '模型注册中心',
        },
      ],
    },

    // 📈 用户行为分析系统
    {
      key: 'behavior-analytics-group',
      label: '📈 用户行为分析系统',
      type: 'group',
    },
    {
      key: 'behavior-analytics',
      icon: <BarChartOutlined />,
      label: '智能行为分析仪表板',
      children: [
        {
          key: 'behavior-analytics',
          icon: <BarChartOutlined />,
          label: '行为分析总览',
        },
        {
          key: 'realtime-monitor',
          icon: <MonitorOutlined />,
          label: '实时监控',
        },
        {
          key: 'session-manage',
          icon: <UserOutlined />,
          label: '会话管理',
        },
        {
          key: 'event-data-manage',
          icon: <DatabaseOutlined />,
          label: '事件数据管理',
        },
        {
          key: 'report-center',
          icon: <FileTextOutlined />,
          label: '报告中心',
        },
        {
          key: 'data-export',
          icon: <ExportOutlined />,
          label: '数据导出',
        },
        {
          key: 'system-config',
          icon: <SettingOutlined />,
          label: '系统配置',
        },
      ],
    },

    // 🧪 A/B测试实验平台
    {
      key: 'ab-testing-group',
      label: '🧪 A/B测试实验平台',
      type: 'group',
    },
    {
      key: 'experiments',
      icon: <ExperimentOutlined />,
      label: '实验管理中心',
      children: [
        {
          key: 'experiments',
          icon: <ProjectOutlined />,
          label: '实验总览',
        },
        {
          key: 'experiment-dashboard',
          icon: <DashboardOutlined />,
          label: '实验仪表板',
        },
        {
          key: 'statistical-analysis',
          icon: <DotChartOutlined />,
          label: '统计分析',
        },
        {
          key: 'traffic-allocation',
          icon: <PercentageOutlined />,
          label: '流量分配',
        },
        {
          key: 'event-tracking',
          icon: <FieldTimeOutlined />,
          label: '事件收集',
        },
        {
          key: 'release-strategy',
          icon: <BuildOutlined />,
          label: '发布策略',
        },
        {
          key: 'monitoring-alerts',
          icon: <AlertOutlined />,
          label: '监控告警',
        },
        {
          key: 'advanced-algorithms',
          icon: <ThunderboltOutlined />,
          label: '高级算法',
        },
      ],
    },
    {
      key: 'ab-traffic-management',
      icon: <SplitCellsOutlined />,
      label: '流量与分群',
      children: [
        {
          key: 'traffic-allocation',
          icon: <PercentageOutlined />,
          label: '流量分配',
        },
        {
          key: 'user-segmentation',
          icon: <UserOutlined />,
          label: '用户分群',
        },
        {
          key: 'feature-flags',
          icon: <FlagOutlined />,
          label: '功能开关',
        },
      ],
    },
    {
      key: 'ab-statistical-analysis',
      icon: <BarChartOutlined />,
      label: '统计分析工具',
      children: [
        {
          key: 'statistical-analysis',
          icon: <DotChartOutlined />,
          label: '统计分析',
        },
        {
          key: 'hypothesis-testing',
          icon: <AimOutlined />,
          label: '假设检验',
        },
        {
          key: 'confidence-intervals',
          icon: <StockOutlined />,
          label: '置信区间',
        },
        {
          key: 'power-analysis',
          icon: <RiseOutlined />,
          label: '功效分析',
        },
        {
          key: 'sample-size-calculator',
          icon: <CalculatorOutlined />,
          label: '样本量计算',
        },
        {
          key: 'bayesian-analysis',
          icon: <FundOutlined />,
          label: '贝叶斯分析',
        },
        {
          key: 'sequential-testing',
          icon: <CarryOutOutlined />,
          label: '序列检验',
        },
        {
          key: 'multi-armed-bandit',
          icon: <ThunderboltOutlined />,
          label: '多臂老虎机算法',
        },
      ],
    },
    {
      key: 'ab-data-pipeline',
      icon: <TransactionOutlined />,
      label: '数据管道',
      children: [
        {
          key: 'event-tracking',
          icon: <FieldTimeOutlined />,
          label: '事件收集',
        },
        {
          key: 'conversion-funnel',
          icon: <FunnelPlotOutlined />,
          label: '转化漏斗',
        },
        {
          key: 'cohort-analysis',
          icon: <DataViewOutlined />,
          label: '队列分析',
        },
        {
          key: 'data-quality',
          icon: <SafetyCertificateOutlined />,
          label: '数据质量',
        },
      ],
    },
    {
      key: 'ab-deployment',
      icon: <RocketOutlined />,
      label: '发布与部署',
      children: [
        {
          key: 'release-strategies',
          icon: <BuildOutlined />,
          label: '发布策略',
        },
        {
          key: 'canary-deployment',
          icon: <BranchesOutlined />,
          label: '金丝雀发布',
        },
        {
          key: 'blue-green-deployment',
          icon: <SyncOutlined />,
          label: '蓝绿部署',
        },
        {
          key: 'risk-assessment',
          icon: <ShieldOutlined />,
          label: '风险评估',
        },
        {
          key: 'auto-rollback',
          icon: <UndoOutlined />,
          label: '自动回滚',
        },
        {
          key: 'experiment-monitoring',
          icon: <EyeOutlined />,
          label: '实验监控',
        },
        {
          key: 'alert-management',
          icon: <AlertOutlined />,
          label: '告警管理',
        },
      ],
    },

    // 🧠 智能存储与记忆
    {
      key: 'storage-memory-group',
      label: '🧠 智能存储与记忆',
      type: 'group',
    },
    {
      key: 'memory-hierarchy',
      icon: <CloudOutlined />,
      label: '记忆管理系统',
      children: [
        {
          key: 'memory-hierarchy',
          icon: <CloudOutlined />,
          label: '智能记忆层次',
        },
        {
          key: 'memory-recall',
          icon: <HistoryOutlined />,
          label: '记忆召回测试',
        },
        {
          key: 'memory-analytics',
          icon: <BarChartOutlined />,
          label: '记忆分析仪表板',
        },
      ],
    },
    {
      key: 'pgvector',
      icon: <DatabaseOutlined />,
      label: '向量数据库',
      children: [
        {
          key: 'pgvector-quantization',
          icon: <DatabaseOutlined />,
          label: 'pgvector量化 (v0.8)',
        },
        {
          key: 'vector-advanced',
          icon: <ClusterOutlined />,
          label: '高级向量索引',
        },
        {
          key: 'vector-advanced-simple',
          icon: <CalculatorOutlined />,
          label: '向量索引简化版',
        },
        {
          key: 'vector-advanced-test',
          icon: <ExperimentOutlined />,
          label: '向量索引测试版',
        },
      ],
    },

    // 🌐 多模态处理引擎
    {
      key: 'multimodal-group',
      label: '🌐 多模态处理引擎',
      type: 'group',
    },
    {
      key: 'multimodal',
      icon: <FileImageOutlined />,
      label: 'Claude-4多模态处理',
      children: [
        {
          key: 'multimodal',
          icon: <FileImageOutlined />,
          label: 'Claude-4多模态处理',
        },
        {
          key: 'multimodal-complete',
          icon: <TabletOutlined />,
          label: '多模态处理完整版',
        },
        {
          key: 'multimodal-simple',
          icon: <PictureOutlined />,
          label: '多模态处理简化版',
        },
      ],
    },
    {
      key: 'document-processing',
      icon: <FileTextOutlined />,
      label: '文档处理系统',
      children: [
        {
          key: 'document-processing',
          icon: <FileTextOutlined />,
          label: '智能文档处理',
        },
        {
          key: 'document-processing-advanced',
          icon: <ImportOutlined />,
          label: '高级文档处理',
        },
        {
          key: 'file-management-advanced',
          icon: <FolderOutlined />,
          label: '高级文件管理',
        },
        {
          key: 'file-management-complete',
          icon: <FolderOpenOutlined />,
          label: '完整文件管理系统',
        },
      ],
    },

    // ⚡ 工作流与编排
    {
      key: 'workflow-group',
      label: '⚡ 工作流与编排',
      type: 'group',
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
          icon: <DeploymentUnitOutlined />,
          label: 'LangGraph新特性',
        },
        {
          key: 'langgraph-065',
          icon: <AppstoreOutlined />,
          label: 'LangGraph 0.6.5',
        },
      ],
    },
    {
      key: 'dag-orchestrator',
      icon: <BranchesOutlined />,
      label: 'DAG编排器',
    },

    // 🏭 系统处理引擎
    {
      key: 'processing-group',
      label: '🏭 系统处理引擎',
      type: 'group',
    },
    {
      key: 'streaming-batch',
      icon: <ThunderboltOutlined />,
      label: '流式与批处理',
      children: [
        {
          key: 'streaming-monitor',
          icon: <ThunderboltOutlined />,
          label: '流式监控',
        },
        {
          key: 'batch-jobs',
          icon: <DatabaseOutlined />,
          label: '批处理作业',
        },
        {
          key: 'batch-jobs-fixed',
          icon: <CheckCircleOutlined />,
          label: '批处理作业修复版',
        },
      ],
    },
    {
      key: 'unified-engine',
      icon: <DeploymentUnitOutlined />,
      label: '统一处理引擎',
      children: [
        {
          key: 'unified-engine',
          icon: <DeploymentUnitOutlined />,
          label: '统一引擎',
        },
        {
          key: 'unified-engine-complete',
          icon: <CloudServerOutlined />,
          label: '统一引擎完整版',
        },
        {
          key: 'unified-monitor',
          icon: <MonitorOutlined />,
          label: '统一监控',
        },
      ],
    },
    {
      key: 'offline-sync',
      icon: <CloudSyncOutlined />,
      label: '离线同步系统',
      children: [
        {
          key: 'offline',
          icon: <CloudSyncOutlined />,
          label: '离线能力',
        },
        {
          key: 'sync-management',
          icon: <SyncOutlined />,
          label: '同步管理',
        },
        {
          key: 'conflict-resolution',
          icon: <ExceptionOutlined />,
          label: '冲突解决',
        },
        {
          key: 'conflict-resolution-learning',
          icon: <ShareAltOutlined />,
          label: '冲突解决学习',
        },
        {
          key: 'vector-clock-algorithm',
          icon: <ClusterOutlined />,
          label: '向量时钟算法',
        },
        {
          key: 'vector-clock-visualization',
          icon: <RadarChartOutlined />,
          label: '向量时钟可视化',
        },
        {
          key: 'sync-engine-internal',
          icon: <FileSyncOutlined />,
          label: '同步引擎内部',
        },
        {
          key: 'sync-engine-learning',
          icon: <BulbOutlined />,
          label: '同步引擎学习',
        },
      ],
    },

    // 📊 系统监控运维
    {
      key: 'monitoring-group',
      label: '📊 系统监控运维',
      type: 'group',
    },
    {
      key: 'rl-system-monitoring',
      icon: <RadarChartOutlined />,
      label: '强化学习系统监控',
      children: [
        {
          key: 'rl-system-dashboard',
          icon: <DashboardOutlined />,
          label: 'RL系统仪表板',
        },
        {
          key: 'rl-performance-monitor',
          icon: <LineChartOutlined />,
          label: 'RL性能监控',
        },
        {
          key: 'rl-integration-test',
          icon: <ExperimentOutlined />,
          label: 'RL集成测试',
        },
        {
          key: 'rl-alerts-config',
          icon: <AlertOutlined />,
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
          label: 'RL系统健康',
        },
      ],
    },
    {
      key: 'system-monitoring',
      icon: <DashboardOutlined />,
      label: '系统监控',
      children: [
        {
          key: 'health-monitor',
          icon: <DashboardOutlined />,
          label: '健康监控',
        },
        {
          key: 'health-comprehensive',
          icon: <HeartOutlined />,
          label: '综合健康监控',
        },
        {
          key: 'performance-monitor',
          icon: <LineChartOutlined />,
          label: '性能监控',
        },
        {
          key: 'monitoring-dashboard',
          icon: <AlertOutlined />,
          label: '监控仪表板',
        },
        {
          key: 'cache-monitor',
          icon: <SaveOutlined />,
          label: '缓存监控',
        },
        {
          key: 'model-cache-monitor',
          icon: <SaveOutlined />,
          label: '模型缓存监控',
        },
        {
          key: 'network-monitor-detail',
          icon: <NetworkOutlined />,
          label: '网络监控详情',
        },
      ],
    },
    {
      key: 'event-dashboard',
      icon: <BellOutlined />,
      label: '事件仪表板',
    },

    // 🛡️ 安全管理
    {
      key: 'security-group',
      label: '🛡️ 安全管理',
      type: 'group',
    },
    {
      key: 'security-system',
      icon: <LockOutlined />,
      label: '安全管理系统',
      children: [
        {
          key: 'ai-trism',
          icon: <LockOutlined />,
          label: 'AI TRiSM框架',
        },
        {
          key: 'security-management',
          icon: <SafetyOutlined />,
          label: '安全管理',
        },
        {
          key: 'security',
          icon: <SecurityScanOutlined />,
          label: '安全页面',
        },
        {
          key: 'auth-management',
          icon: <UserOutlined />,
          label: '认证管理',
        },
      ],
    },

    // 🏢 企业级架构
    {
      key: 'enterprise-group',
      label: '🏢 企业级架构',
      type: 'group',
    },
    {
      key: 'enterprise-architecture',
      icon: <CloudServerOutlined />,
      label: '企业架构管理',
      children: [
        {
          key: 'enterprise-architecture',
          icon: <CloudServerOutlined />,
          label: '企业架构管理',
        },
        {
          key: 'enterprise-config',
          icon: <SettingOutlined />,
          label: '企业配置管理',
        },
        {
          key: 'architecture-debug',
          icon: <BugOutlined />,
          label: '架构调试',
        },
      ],
    },
    {
      key: 'mcp-tools',
      icon: <ApiOutlined />,
      label: 'MCP协议工具 (v1.0)',
    },

    // 🎯 LoRA/QLoRA细粒度调优框架 (Story 9.1)
    {
      key: 'fine-tuning-group',
      label: '🎯 LoRA/QLoRA细粒度调优框架',
      type: 'group',
    },
    {
      key: 'fine-tuning-system',
      icon: <ExperimentOutlined />,
      label: '细粒度调优系统',
      children: [
        {
          key: 'fine-tuning-jobs',
          icon: <ProjectOutlined />,
          label: '调优任务管理',
        },
        {
          key: 'fine-tuning-config',
          icon: <SettingOutlined />,
          label: '调优配置中心',
        },
        {
          key: 'fine-tuning-monitor',
          icon: <MonitorOutlined />,
          label: '调优监控面板',
        },
        {
          key: 'fine-tuning-models',
          icon: <DatabaseOutlined />,
          label: '模型库管理',
        },
        {
          key: 'fine-tuning-datasets',
          icon: <FolderOutlined />,
          label: '数据集管理',
        },
        {
          key: 'fine-tuning-checkpoints',
          icon: <SaveOutlined />,
          label: '检查点管理',
        },
        {
          key: 'lora-training',
          icon: <RocketOutlined />,
          label: 'LoRA训练引擎',
        },
        {
          key: 'qlora-training',
          icon: <ThunderboltOutlined />,
          label: 'QLoRA训练引擎',
        },
        {
          key: 'distributed-training',
          icon: <ClusterOutlined />,
          label: '分布式训练',
        },
        {
          key: 'model-adapters',
          icon: <ApiOutlined />,
          label: '模型适配器',
        },
        {
          key: 'training-monitor-dashboard',
          icon: <DashboardOutlined />,
          label: '训练监控大屏',
        },
        {
          key: 'model-performance-comparison',
          icon: <BarChartOutlined />,
          label: '性能对比分析',
        },
      ],
    },

    // 🔄 自动超参数优化系统 (Story 9.3)
    {
      key: 'hyperparameter-group',
      label: '🔄 自动超参数优化系统',
      type: 'group',
    },
    {
      key: 'hyperparameter-optimization',
      icon: <SlidersOutlined />,
      label: '超参数优化平台',
      children: [
        {
          key: 'hyperparameter-optimization',
          icon: <SlidersOutlined />,
          label: '优化控制中心',
        },
        {
          key: 'hyperparameter-experiments',
          icon: <ExperimentOutlined />,
          label: '实验管理',
        },
        {
          key: 'hyperparameter-algorithms',
          icon: <FunctionOutlined />,
          label: '优化算法库',
        },
        {
          key: 'hyperparameter-monitoring',
          icon: <MonitorOutlined />,
          label: '监控面板',
        },
        {
          key: 'hyperparameter-reports',
          icon: <FileTextOutlined />,
          label: '报告中心',
        },
        {
          key: 'hyperparameter-resources',
          icon: <CloudServerOutlined />,
          label: '资源管理',
        },
        {
          key: 'hyperparameter-scheduler',
          icon: <ScheduleOutlined />,
          label: '任务调度器',
        },
        {
          key: 'hyperparameter-visualizations',
          icon: <AreaChartOutlined />,
          label: '可视化分析',
        },
      ],
    },

    // 📚 训练数据管理系统 (Story 9.5)
    {
      key: 'training-data-group',
      label: '📚 训练数据管理系统',
      type: 'group',
    },
    {
      key: 'training-data-management',
      icon: <DatabaseOutlined />,
      label: '训练数据管理',
      children: [
        {
          key: 'training-data-management',
          icon: <DatabaseOutlined />,
          label: '数据管理中心',
        },
        {
          key: 'data-collection',
          icon: <ImportOutlined />,
          label: '数据采集',
        },
        {
          key: 'data-preprocessing',
          icon: <BuildOutlined />,
          label: '数据预处理',
        },
        {
          key: 'data-annotation-management',
          icon: <EditOutlined />,
          label: '数据标注管理',
        },
        {
          key: 'annotation-tasks',
          icon: <ProjectOutlined />,
          label: '标注任务',
        },
        {
          key: 'annotation-quality-control',
          icon: <SafetyCertificateOutlined />,
          label: '标注质量控制',
        },
        {
          key: 'data-version-management',
          icon: <BranchesOutlined />,
          label: '数据版本管理',
        },
        {
          key: 'data-source-management',
          icon: <GlobalOutlined />,
          label: '数据源管理',
        },
      ],
    },

    // 🔬 开发与测试
    {
      key: 'dev-test-group',
      label: '🔬 开发与测试',
      type: 'group',
    },
    {
      key: 'testing-system',
      icon: <CodeOutlined />,
      label: '测试系统',
      children: [
        {
          key: 'test-coverage',
          icon: <CheckCircleOutlined />,
          label: '测试覆盖率',
        },
        {
          key: 'integration-test',
          icon: <PlayCircleOutlined />,
          label: '集成测试',
        },
        {
          key: 'testing-suite',
          icon: <CodeOutlined />,
          label: '测试套件',
        },
        {
          key: 'structured-error',
          icon: <ExceptionOutlined />,
          label: '结构化错误',
        },
      ],
    },

  ]

  const handleNavigation = ({ key }: { key: string }) => {
    switch (key) {
      // 智能体系统
      case 'chat': navigate('/chat'); break;
      case 'multi-agent': navigate('/multi-agent'); break;
      case 'supervisor': navigate('/supervisor'); break;
      case 'async-agents': navigate('/async-agents'); break;
      case 'service-discovery-test': navigate('/service-discovery-overview'); break;
      
      // 智能代理服务发现系统 (Story 10.1)
      case 'service-discovery-overview': navigate('/service-discovery-overview'); break;
      case 'agent-registry-management': navigate('/agent-registry'); break;
      case 'service-routing-management': navigate('/service-routing'); break;
      case 'load-balancer-config': navigate('/load-balancer-config'); break;
      case 'service-health-monitor': navigate('/service-health-monitor'); break;
      case 'service-cluster-management': navigate('/service-cluster-management'); break;
      case 'service-performance-dashboard': navigate('/service-performance-dashboard'); break;
      case 'service-config-management': navigate('/service-config-management'); break;
      case 'intelligent-agent-service-discovery-system': navigate('/service-discovery-overview'); break;
      
      // 分布式消息通信框架 (Story 10.2)
      case 'distributed-message-overview': navigate('/distributed-message-overview'); break;
      case 'nats-cluster-management': navigate('/nats-cluster-management'); break;
      case 'basic-message-communication': navigate('/basic-message-communication'); break;
      case 'acl-protocol-management': navigate('/acl-protocol-management'); break;
      case 'request-response-mechanism': navigate('/request-response-mechanism'); break;
      case 'message-reliability-management': navigate('/message-reliability-management'); break;
      case 'advanced-communication-patterns': navigate('/advanced-communication-patterns'); break;
      case 'monitoring-performance-optimization': navigate('/monitoring-performance-optimization'); break;
      
      // 故障容错和恢复系统 (Story 10.5)
      case 'fault-tolerance-overview': navigate('/fault-tolerance-overview'); break;
      case 'fault-detection': navigate('/fault-detection'); break;
      case 'recovery-management': navigate('/recovery-management'); break;
      case 'backup-management': navigate('/backup-management'); break;
      case 'consistency-management': navigate('/consistency-management'); break;
      case 'system-monitoring': navigate('/system-monitoring'); break;
      case 'fault-testing': navigate('/fault-testing'); break;
      
      // 智能检索引擎
      case 'rag': navigate('/rag'); break;
      case 'agentic-rag': navigate('/agentic-rag'); break;
      case 'hybrid-search-advanced': navigate('/hybrid-search-advanced'); break;
      case 'multimodal-rag': navigate('/multimodal-rag'); break;
      
      // 动态知识图谱存储系统 (Story 8.2)
      case 'kg-entity-management': navigate('/kg-entity-management'); break;
      case 'kg-relation-management': navigate('/kg-relation-management'); break;
      case 'kg-graph-query': navigate('/kg-graph-query'); break;
      case 'kg-incremental-update': navigate('/kg-incremental-update'); break;
      case 'kg-quality-assessment': navigate('/kg-quality-assessment'); break;
      case 'kg-performance-monitor': navigate('/kg-performance-monitor'); break;
      case 'kg-schema-management': navigate('/kg-schema-management'); break;
      case 'kg-data-migration': navigate('/kg-data-migration'); break;
      
      // 推理引擎
      case 'reasoning': navigate('/reasoning'); break;
      case 'multi-step-reasoning': navigate('/multi-step-reasoning'); break;
      case 'explainable-ai': navigate('/explainable-ai'); break;
      
      // 推荐算法引擎
      case 'bandit-recommendation': navigate('/bandit-recommendation'); break;
      
      // 个性化引擎
      case 'personalization-engine': navigate('/personalization-engine'); break;
      case 'personalization-monitor': navigate('/personalization-monitor'); break;
      case 'personalization-features': navigate('/personalization-features'); break;
      case 'personalization-learning': navigate('/personalization-learning'); break;
      case 'personalization-api': navigate('/personalization-api'); break;
      case 'personalization-alerts': navigate('/personalization-alerts'); break;
      case 'personalization-production': navigate('/personalization-production'); break;
      case 'personalization-websocket': navigate('/personalization-websocket'); break;
      
      // 高级情感智能系统
      case 'emotion-recognition-overview': navigate('/emotion-recognition-overview'); break;
      case 'text-emotion-analysis': navigate('/text-emotion-analysis'); break;
      case 'audio-emotion-recognition': navigate('/audio-emotion-recognition'); break;
      case 'visual-emotion-analysis': navigate('/visual-emotion-analysis'); break;
      case 'multimodal-emotion-fusion': navigate('/multimodal-emotion-fusion'); break;
      case 'emotion-modeling': navigate('/emotion-modeling'); break;
      
      // 情感记忆管理系统
      case 'emotional-memory-management': navigate('/emotional-memory-management'); break;
      case 'emotional-event-analysis': navigate('/emotional-event-analysis'); break;
      case 'emotional-preference-learning': navigate('/emotional-preference-learning'); break;
      case 'emotional-trigger-patterns': navigate('/emotional-trigger-patterns'); break;
      case 'emotional-memory-retrieval': navigate('/emotional-memory-retrieval'); break;
      
      // 社交情感理解系统 (Story 11.6)
      case 'social-emotion-system': navigate('/social-emotion-system'); break;
      case 'emotion-flow-analysis': navigate('/emotion-flow-analysis'); break;
      case 'social-network-emotion-map': navigate('/social-network-emotion-map'); break;
      case 'cultural-context-analysis': navigate('/cultural-context-analysis'); break;
      case 'social-intelligence-decision': navigate('/social-intelligence-decision'); break;
      case 'privacy-ethics': navigate('/privacy-ethics'); break;
      
      // 情感智能决策引擎 (Story 11.5)
      case 'emotional-intelligence-decision-engine': navigate('/emotional-intelligence-decision-engine'); break;
      case 'emotional-risk-assessment-dashboard': navigate('/emotional-risk-assessment-dashboard'); break;
      case 'crisis-detection-support': navigate('/crisis-detection-support'); break;
      case 'intervention-strategy-management': navigate('/intervention-strategy-management'); break;
      case 'emotional-health-monitoring-dashboard': navigate('/emotional-health-monitoring-dashboard'); break;
      case 'decision-history-analysis': navigate('/decision-history-analysis'); break;
      
      // 强化学习系统
      case 'qlearning': navigate('/qlearning'); break;
      case 'qlearning-training': navigate('/qlearning-training'); break;
      case 'qlearning-strategy': navigate('/qlearning-strategy'); break;
      case 'qlearning-recommendation': navigate('/qlearning-recommendation'); break;
      case 'qlearning-performance': navigate('/qlearning-performance'); break;
      case 'qlearning-performance-optimization': navigate('/qlearning-performance-optimization'); break;
      // Q-Learning 算法
      case 'tabular-qlearning': navigate('/qlearning/tabular'); break;
      case 'dqn': navigate('/qlearning/dqn'); break;
      case 'dqn-variants': navigate('/qlearning/variants'); break;
      // 探索策略
      case 'exploration-strategies': navigate('/qlearning/exploration-strategies'); break;
      case 'ucb-strategies': navigate('/qlearning/ucb-strategies'); break;
      case 'thompson-sampling': navigate('/qlearning/thompson-sampling'); break;
      // 奖励系统
      case 'basic-rewards': navigate('/qlearning/basic-rewards'); break;
      case 'composite-rewards': navigate('/qlearning/composite-rewards'); break;
      case 'adaptive-rewards': navigate('/qlearning/adaptive-rewards'); break;
      case 'reward-shaping': navigate('/qlearning/reward-shaping'); break;
      // 环境建模
      case 'state-space': navigate('/qlearning/state-space'); break;
      case 'action-space': navigate('/qlearning/action-space'); break;
      case 'environment-simulator': navigate('/qlearning/environment-simulator'); break;
      case 'grid-world': navigate('/qlearning/grid-world'); break;
      // 训练管理
      case 'training-manager': navigate('/qlearning/training-manager'); break;
      case 'learning-rate-scheduler': navigate('/qlearning/learning-rate-scheduler'); break;
      case 'early-stopping': navigate('/qlearning/early-stopping'); break;
      case 'performance-tracker': navigate('/qlearning/performance-tracker'); break;
      
      // A/B测试实验平台
      case 'experiments': navigate('/experiments'); break;
      case 'experiment-dashboard': navigate('/experiments/dashboard'); break;
      case 'statistical-analysis': navigate('/experiments/statistical-analysis'); break;
      case 'traffic-allocation': navigate('/experiments/traffic-allocation'); break;
      case 'event-tracking': navigate('/experiments/event-tracking'); break;
      case 'release-strategy': navigate('/experiments/release-strategy'); break;
      case 'monitoring-alerts': navigate('/experiments/monitoring-alerts'); break;
      case 'advanced-algorithms': navigate('/experiments/advanced-algorithms'); break;
      case 'experiment-list': navigate('/experiment-list'); break;
      case 'experiment-report': navigate('/experiment-report'); break;
      case 'experiment-config': navigate('/experiment-config'); break;
      case 'user-segmentation': navigate('/user-segmentation'); break;
      case 'conversion-funnel': navigate('/conversion-funnel'); break;
      case 'cohort-analysis': navigate('/cohort-analysis'); break;
      case 'release-strategies': navigate('/release-strategies'); break;
      case 'canary-deployment': navigate('/canary-deployment'); break;
      case 'blue-green-deployment': navigate('/blue-green-deployment'); break;
      case 'feature-flags': navigate('/feature-flags'); break;
      case 'risk-assessment': navigate('/risk-assessment'); break;
      case 'auto-rollback': navigate('/auto-rollback'); break;
      case 'experiment-monitoring': navigate('/experiment-monitoring'); break;
      case 'alert-management': navigate('/alert-management'); break;
      case 'data-quality': navigate('/data-quality'); break;
      case 'sample-size-calculator': navigate('/sample-size-calculator'); break;
      case 'power-analysis': navigate('/power-analysis'); break;
      case 'confidence-intervals': navigate('/confidence-intervals'); break;
      case 'hypothesis-testing': navigate('/hypothesis-testing'); break;
      case 'bayesian-analysis': navigate('/bayesian-analysis'); break;
      case 'sequential-testing': navigate('/sequential-testing'); break;
      case 'multi-armed-bandit': navigate('/multi-armed-bandit'); break;
      
      // 用户行为分析系统
      case 'behavior-analytics': navigate('/behavior-analytics'); break;
      case 'realtime-monitor': navigate('/behavior-analytics/realtime'); break;
      case 'session-manage': navigate('/behavior-analytics/sessions'); break;
      case 'event-data-manage': navigate('/behavior-analytics/events'); break;
      case 'report-center': navigate('/behavior-analytics/reports'); break;
      case 'data-export': navigate('/behavior-analytics/export'); break;
      case 'system-config': navigate('/behavior-analytics/config'); break;
      
      // 用户反馈学习系统
      case 'feedback-system': navigate('/feedback-system'); break;
      case 'feedback-analytics': navigate('/feedback-analytics'); break;
      case 'user-feedback-profiles': navigate('/user-feedback-profiles'); break;
      case 'item-feedback-analysis': navigate('/item-feedback-analysis'); break;
      case 'feedback-quality-monitor': navigate('/feedback-quality-monitor'); break;
      
      // 模型服务部署平台 (Story 9.6)
      case 'model-registry': navigate('/model-registry'); break;
      case 'model-inference': navigate('/model-inference'); break;
      case 'model-deployment': navigate('/model-deployment'); break;
      case 'model-monitoring': navigate('/model-monitoring'); break;
      case 'online-learning': navigate('/online-learning'); break;
      
      // 模型评估和基准测试系统 (Story 9.4)
      case 'model-evaluation-overview': navigate('/model-evaluation-overview'); break;
      case 'model-performance-benchmark': navigate('/model-performance-benchmark'); break;
      case 'evaluation-engine-management': navigate('/evaluation-engine-management'); break;
      case 'benchmark-suite-management': navigate('/benchmark-suite-management'); break;
      case 'evaluation-tasks-monitor': navigate('/evaluation-tasks-monitor'); break;
      case 'evaluation-reports-center': navigate('/evaluation-reports-center'); break;
      case 'evaluation-api-management': navigate('/evaluation-api-management'); break;
      case 'model-comparison-dashboard': navigate('/model-comparison-dashboard'); break;
      case 'benchmark-glue-management': navigate('/benchmark-glue-management'); break;
      case 'benchmark-superglue-management': navigate('/benchmark-superglue-management'); break;
      case 'benchmark-mmlu-management': navigate('/benchmark-mmlu-management'); break;
      case 'benchmark-humaneval-management': navigate('/benchmark-humaneval-management'); break;
      case 'benchmark-hellaswag-management': navigate('/benchmark-hellaswag-management'); break;
      case 'benchmark-custom-management': navigate('/benchmark-custom-management'); break;
      case 'evaluation-metrics-config': navigate('/evaluation-metrics-config'); break;
      case 'evaluation-performance-monitor': navigate('/evaluation-performance-monitor'); break;
      case 'evaluation-batch-processing': navigate('/evaluation-batch-processing'); break;
      case 'evaluation-regression-detection': navigate('/evaluation-regression-detection'); break;
      case 'evaluation-quality-assurance': navigate('/evaluation-quality-assurance'); break;
      case 'evaluation-automation-pipeline': navigate('/evaluation-automation-pipeline'); break;
      case 'evaluation-alerts-management': navigate('/evaluation-alerts-management'); break;
      case 'evaluation-data-management': navigate('/evaluation-data-management'); break;
      case 'evaluation-resource-monitor': navigate('/evaluation-resource-monitor'); break;
      case 'evaluation-job-scheduler': navigate('/evaluation-job-scheduler'); break;
      case 'evaluation-results-analysis': navigate('/evaluation-results-analysis'); break;
      case 'evaluation-export-import': navigate('/evaluation-export-import'); break;
      case 'evaluation-version-control': navigate('/evaluation-version-control'); break;
      case 'evaluation-compliance-audit': navigate('/evaluation-compliance-audit'); break;
      case 'evaluation-security-management': navigate('/evaluation-security-management'); break;
      
      // 模型压缩和量化工具
      case 'model-compression-overview': navigate('/model-compression-overview'); break;
      case 'quantization-manager': navigate('/quantization-manager'); break;
      case 'quantization-ptq': navigate('/quantization-ptq'); break;
      case 'quantization-qat': navigate('/quantization-qat'); break;
      case 'quantization-advanced': navigate('/quantization-advanced'); break;
      case 'quantization-config': navigate('/quantization-config'); break;
      case 'knowledge-distillation': navigate('/knowledge-distillation'); break;
      case 'distillation-trainer': navigate('/distillation-trainer'); break;
      case 'distillation-strategies': navigate('/distillation-strategies'); break;
      case 'distillation-monitor': navigate('/distillation-monitor'); break;
      case 'model-pruning': navigate('/model-pruning'); break;
      case 'pruning-structured': navigate('/pruning-structured'); break;
      case 'pruning-unstructured': navigate('/pruning-unstructured'); break;
      case 'pruning-strategies': navigate('/pruning-strategies'); break;
      case 'compression-pipeline': navigate('/compression-pipeline'); break;
      case 'compression-jobs': navigate('/compression-jobs'); break;
      case 'compression-monitor': navigate('/compression-monitor'); break;
      case 'compression-scheduler': navigate('/compression-scheduler'); break;
      case 'compression-evaluator': navigate('/compression-evaluator'); break;
      case 'model-comparison': navigate('/model-comparison'); break;
      case 'performance-analysis': navigate('/performance-analysis'); break;
      case 'compression-reports': navigate('/compression-reports'); break;
      case 'hardware-benchmark': navigate('/hardware-benchmark'); break;
      case 'inference-optimization': navigate('/inference-optimization'); break;
      case 'deployment-optimization': navigate('/deployment-optimization'); break;
      case 'strategy-recommendation': navigate('/strategy-recommendation'); break;
      case 'compression-templates': navigate('/compression-templates'); break;
      case 'model-registry-compression': navigate('/model-registry-compression'); break;
      
      // 记忆管理系统
      case 'memory-hierarchy': navigate('/memory-hierarchy'); break;
      case 'memory-recall': navigate('/memory-recall'); break;
      case 'memory-analytics': navigate('/memory-analytics'); break;
      
      // 多模态处理
      case 'multimodal': navigate('/multimodal'); break;
      case 'multimodal-complete': navigate('/multimodal-complete'); break;
      case 'multimodal-simple': navigate('/multimodal-simple'); break;
      case 'file-management-advanced': navigate('/file-management-advanced'); break;
      case 'file-management-complete': navigate('/file-management-complete'); break;
      case 'document-processing': navigate('/document-processing'); break;
      case 'document-processing-advanced': navigate('/document-processing-advanced'); break;
      
      // 工作流引擎
      case 'workflows': navigate('/workflows'); break;
      case 'langgraph-features': navigate('/langgraph-features'); break;
      case 'langgraph-065': navigate('/langgraph-065'); break;
      case 'dag-orchestrator': navigate('/dag-orchestrator'); break;
      case 'flow-control': navigate('/flow-control'); break;
      
      // 离线能力与同步机制
      case 'offline': navigate('/offline'); break;
      case 'sync-management': navigate('/sync-management'); break;
      case 'conflict-resolution': navigate('/conflict-resolution'); break;
      case 'conflict-resolution-learning': navigate('/conflict-resolution-learning'); break;
      case 'vector-clock-algorithm': navigate('/vector-clock-algorithm'); break;
      case 'vector-clock-visualization': navigate('/vector-clock-visualization'); break;
      case 'network-monitor-detail': navigate('/network-monitor-detail'); break;
      case 'sync-engine-internal': navigate('/sync-engine-internal'); break;
      case 'sync-engine-learning': navigate('/sync-engine-learning'); break;
      case 'model-cache-monitor': navigate('/model-cache-monitor'); break;
      
      // 处理引擎
      case 'streaming-monitor': navigate('/streaming-monitor'); break;
      case 'batch-jobs': navigate('/batch-jobs'); break;
      case 'batch-jobs-fixed': navigate('/batch-jobs-fixed'); break;
      case 'unified-engine': navigate('/unified-engine'); break;
      case 'unified-engine-complete': navigate('/unified-engine-complete'); break;
      case 'unified-monitor': navigate('/unified-monitor'); break;
      
      // 安全与合规
      case 'ai-trism': navigate('/ai-trism'); break;
      case 'security-management': navigate('/security-management'); break;
      case 'security': navigate('/security'); break;
      case 'auth-management': navigate('/auth-management'); break;
      
      // 事件与监控
      case 'event-dashboard': navigate('/event-dashboard'); break;
      case 'distributed-events': navigate('/distributed-events'); break;
      case 'health-monitor': navigate('/health-monitor'); break;
      case 'health-comprehensive': navigate('/health-comprehensive'); break;
      case 'performance-monitor': navigate('/performance-monitor'); break;
      case 'monitoring-dashboard': navigate('/monitoring-dashboard'); break;
      case 'cache-monitor': navigate('/cache-monitor'); break;
      
      // 数据存储
      case 'pgvector': navigate('/pgvector'); break;
      case 'pgvector-quantization': navigate('/pgvector'); break;
      case 'vector-advanced': navigate('/vector-advanced'); break;
      case 'vector-advanced-simple': navigate('/vector-advanced-simple'); break;
      case 'vector-advanced-test': navigate('/vector-advanced-test'); break;
      
      // 协议与工具
      case 'mcp-tools': navigate('/mcp-tools'); break;
      
      // 企业架构
      case 'enterprise-architecture': navigate('/enterprise-architecture'); break;
      case 'enterprise-config': navigate('/enterprise-config'); break;
      case 'architecture-debug': navigate('/architecture-debug'); break;
      
      // 开发测试
      case 'structured-error': navigate('/structured-error'); break;
      case 'test-coverage': navigate('/test-coverage'); break;
      case 'integration-test': navigate('/integration-test'); break;
      case 'testing-suite': navigate('/testing-suite'); break;
      
      // LoRA/QLoRA细粒度调优框架 (Story 9.1)
      case 'fine-tuning-jobs': navigate('/fine-tuning-jobs'); break;
      case 'fine-tuning-config': navigate('/fine-tuning-config'); break;
      case 'fine-tuning-monitor': navigate('/fine-tuning-monitor'); break;
      case 'fine-tuning-models': navigate('/fine-tuning-models'); break;
      case 'fine-tuning-datasets': navigate('/fine-tuning-datasets'); break;
      case 'fine-tuning-checkpoints': navigate('/fine-tuning-checkpoints'); break;
      case 'lora-training': navigate('/lora-training'); break;
      case 'qlora-training': navigate('/qlora-training'); break;
      case 'distributed-training': navigate('/distributed-training'); break;
      case 'model-adapters': navigate('/model-adapters'); break;
      case 'training-monitor-dashboard': navigate('/training-monitor-dashboard'); break;
      case 'model-performance-comparison': navigate('/model-performance-comparison'); break;
      
      // 自动超参数优化系统 (Story 9.3)
      case 'hyperparameter-optimization': navigate('/hyperparameter-optimization'); break;
      case 'hyperparameter-experiments': navigate('/hyperparameter-experiments'); break;
      case 'hyperparameter-algorithms': navigate('/hyperparameter-algorithms'); break;
      case 'hyperparameter-monitoring': navigate('/hyperparameter-monitoring'); break;
      case 'hyperparameter-reports': navigate('/hyperparameter-reports'); break;
      case 'hyperparameter-resources': navigate('/hyperparameter-resources'); break;
      case 'hyperparameter-scheduler': navigate('/hyperparameter-scheduler'); break;
      case 'hyperparameter-visualizations': navigate('/hyperparameter-visualizations'); break;
      
      // 训练数据管理系统 (Story 9.5)
      case 'training-data-management': navigate('/training-data-management'); break;
      case 'data-collection': navigate('/data-collection'); break;
      case 'data-preprocessing': navigate('/data-preprocessing'); break;
      case 'data-annotation-management': navigate('/data-annotation-management'); break;
      case 'annotation-tasks': navigate('/annotation-tasks'); break;
      case 'annotation-quality-control': navigate('/annotation-quality-control'); break;
      case 'data-version-management': navigate('/data-version-management'); break;
      case 'data-source-management': navigate('/data-source-management'); break;
    }
  }

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
          borderRight: '1px solid #f0f0f0',
          boxShadow: '2px 0 8px rgba(0,0,0,0.1)',
          display: 'flex',
          flexDirection: 'column',
          height: '100vh'
        }}
      >
        <div style={{ 
          padding: '16px', 
          borderBottom: '1px solid #f0f0f0',
          textAlign: collapsed ? 'center' : 'left'
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
                <Text type="secondary" style={{ fontSize: '12px' }}>多代理学习系统</Text>
              </div>
            )}
          </Space>
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          style={{ border: 'none', flex: 1, overflowY: 'auto' }}
          onClick={handleNavigation}
        />
      </Sider>

      <Layout>
        <Header style={{ 
          background: '#fff', 
          borderBottom: '1px solid #f0f0f0', 
          padding: '0 24px' 
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between' 
          }}>
            <Space align="center">
              <Button
                type="text"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed(!collapsed)}
                style={{ fontSize: '16px' }}
              />
            </Space>
          </div>
        </Header>

        <Content style={{ 
          background: '#f5f5f5', 
          display: 'flex', 
          flexDirection: 'column' 
        }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default MainLayout
export { MainLayout }
