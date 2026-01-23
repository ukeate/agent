import type { MenuProps } from 'antd'
import { isValidElement } from 'react'
import {
  MessageOutlined,
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

const MENU_KEY_TO_PATH_OVERRIDES: Record<string, string> = {
  'advanced-algorithms': '/experiments/advanced-algorithms',
  'agent-registry-management': '/agent-registry',
  'batch-jobs-basic': '/batch-jobs',
  'batch-jobs-management': '/batch-operations',
  'behavior-analytics-config': '/behavior-analytics/config',
  'behavior-analytics-events': '/behavior-analytics/events',
  'behavior-analytics-export': '/behavior-analytics/export',
  'behavior-analytics-realtime': '/behavior-analytics/realtime',
  'behavior-analytics-reports': '/behavior-analytics/reports',
  'behavior-analytics-sessions': '/behavior-analytics/sessions',
  'cache-monitor': '/cache',
  'chat-history': '/history',
  'conflict-resolution': '/conflicts',
  'distributed-events-system': '/distributed-events',
  'enhanced-experiment-analysis': '/experiments/enhanced-analysis',
  'event-tracking': '/experiments/event-tracking',
  'experiment-dashboard': '/experiments/dashboard',
  'experiment-list': '/experiments',
  'file-management-standard': '/file-management',
  'fine-tuning-management': '/fine-tuning',
  'hyperparameter-optimization-dashboard': '/hyperparameter-optimization',
  'layered-experiments-management': '/experiments/layered-experiments',
  'model-cache-monitor': '/model-cache',
  'monitoring-alerts': '/experiments/monitoring-alerts',
  'multimodal-rag-system': '/multimodal-rag',
  'multiple-testing-correction': '/experiments/multiple-testing',
  'network-monitor-detail': '/network-monitor',
  'pgvector-quantization': '/pgvector',
  'power-analysis': '/experiments/power-analysis',
  'qlearning-dashboard': '/qlearning',
  'qlearning-dqn': '/qlearning/dqn',
  'qlearning-tabular': '/qlearning/tabular',
  'qlearning-variants': '/qlearning/variants',
  'release-strategy': '/experiments/release-strategy',
  'security-audit-system': '/security-audit',
  'service-routing-management': '/service-routing',
  'social-emotional-understanding-system': '/social-emotional-understanding',
  'statistical-analysis': '/experiments/statistical-analysis',
  streaming: '/streaming-monitor',
  'sync-engine-internal': '/sync-engine',
  'sync-management': '/sync',
  'test-integration': '/test',
  'traffic-allocation': '/experiments/traffic-allocation',
  'traffic-ramp-management': '/experiments/traffic-ramp',
  'training-data-overview': '/training-data-management',
  'vector-clock-viz': '/vector-clock',
  'workflows-visualization': '/workflow',
}

type MenuItem = NonNullable<MenuProps['items']>[number]

export const getMenuLabelText = (label: MenuItem['label']): string => {
  if (typeof label === 'string' || typeof label === 'number')
    return String(label)
  if (Array.isArray(label)) return label.map(getMenuLabelText).join('')
  if (isValidElement(label)) return getMenuLabelText(label.props?.children)
  return ''
}

const PATH_TO_MENU_KEY_OVERRIDES: Record<string, string> = {
  ...Object.fromEntries(
    Object.entries(MENU_KEY_TO_PATH_OVERRIDES).map(([key, path]) => [path, key])
  ),
  '/batch': 'batch-jobs-basic',
  '/document-processing': 'document-processing-simple',
  '/enterprise-architecture': 'enterprise',
  '/multimodal': 'multimodal-complete',
  '/workflows': 'workflows-visualization',
}


export const resolveMenuPath = (menuKey: string) => {
  return MENU_KEY_TO_PATH_OVERRIDES[menuKey] ?? `/${menuKey}`
}

export const resolveMenuKey = (path: string) => {
  const normalizedPath = path !== '/' ? path.replace(/\/+$/, '') : path
  return (
    PATH_TO_MENU_KEY_OVERRIDES[normalizedPath] ??
    normalizedPath.replace(/^\//, '')
  )
}


export const MENU_ITEMS: MenuProps['items'] = [
  // 🤖 智能体系统
  {
    key: 'ai-agents-group',
    label: '🤖 智能体系统',
    type: 'group' as const,
  },
  {
    key: 'workspace',
    icon: <DashboardOutlined />,
    label: '工作台概览',
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

  // 🛡️ 故障容错与恢复
  {
    key: 'fault-tolerance-group',
    label: '🛡️ 故障容错与恢复',
    type: 'group' as const,
  },
  {
    key: 'fault-tolerance-overview',
    icon: <SafetyOutlined />,
    label: '故障容错总览',
  },
  {
    key: 'fault-detection',
    icon: <AlertOutlined />,
    label: '故障检测',
  },
  {
    key: 'recovery-management',
    icon: <RollbackOutlined />,
    label: '恢复管理',
  },
  {
    key: 'backup-management',
    icon: <DatabaseOutlined />,
    label: '备份管理',
  },
  {
    key: 'consistency-management',
    icon: <CheckCircleOutlined />,
    label: '一致性管理',
  },
  {
    key: 'fault-testing',
    icon: <BugOutlined />,
    label: '故障演练',
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
