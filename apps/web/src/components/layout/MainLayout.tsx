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
  CloudSyncOutlined,
  LockOutlined,
  FileSyncOutlined,
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
  RiseOutlined,
  RobotFilled,
  InteractionOutlined,
  NetworkOutlined,
  SaveOutlined,
  SlidersOutlined,
  AppstoreOutlined,
  CloudOutlined,
  CalculatorOutlined,
  ImportOutlined,
  ExportOutlined,
  PictureOutlined,
  SecurityScanOutlined,
  PlayCircleOutlined,
  ScheduleOutlined,
  ShareAltOutlined,
  TabletOutlined,
  FolderOpenOutlined,
  GoldOutlined,
  DotChartOutlined,
  PercentageOutlined,
  AimOutlined,
  FlagOutlined,
  RiseOutlined,
  SplitCellsOutlined,
  FunnelPlotOutlined,
  RocketOutlined,
  FunctionOutlined,
  DataViewOutlined,
  StockOutlined,
  CarryOutOutlined,
  ProjectOutlined,
  BookOutlined,
  AuditOutlined,
  FieldTimeOutlined,
  TransactionOutlined,
  BuildOutlined,
  RocketOutlined,
  SafetyCertificateOutlined,
  ShieldOutlined,
  UnorderedListOutlined,
  UndoOutlined
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
    if (path === '/agent-interface') return 'agent-interface'
    
    // 智能检索引擎  
    if (path === '/rag') return 'rag'
    if (path === '/agentic-rag') return 'agentic-rag'
    if (path === '/hybrid-search-advanced') return 'hybrid-search-advanced'
    if (path === '/multimodal-rag') return 'multimodal-rag'
    
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
          key: 'distributed-events',
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
    {
      key: 'agent-interface',
      icon: <ApiOutlined />,
      label: 'Agent接口管理',
    },

    // 🔍 智能检索引擎
    {
      key: 'retrieval-group',
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
          key: 'pgvector',
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
          key: 'workflows',
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
      ],
    },
    {
      key: 'structured-error',
      icon: <ExceptionOutlined />,
      label: '结构化错误处理',
    },
  ]

  const handleNavigation = ({ key }: { key: string }) => {
    switch (key) {
      // 智能体系统
      case 'chat': navigate('/chat'); break;
      case 'multi-agent': navigate('/multi-agent'); break;
      case 'supervisor': navigate('/supervisor'); break;
      case 'async-agents': navigate('/async-agents'); break;
      case 'agent-interface': navigate('/agent-interface'); break;
      
      // 智能检索引擎
      case 'rag': navigate('/rag'); break;
      case 'agentic-rag': navigate('/agentic-rag'); break;
      case 'hybrid-search-advanced': navigate('/hybrid-search-advanced'); break;
      case 'multimodal-rag': navigate('/multimodal-rag'); break;
      
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
          boxShadow: '2px 0 8px rgba(0,0,0,0.1)'
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
          style={{ border: 'none', height: 'calc(100vh - 88px)', overflowY: 'auto' }}
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