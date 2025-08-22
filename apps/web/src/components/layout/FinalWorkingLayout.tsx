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
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

interface FinalWorkingLayoutProps {
  children: React.ReactNode
}

const FinalWorkingLayout: React.FC<FinalWorkingLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const getSelectedKey = () => {
    const path = location.pathname
    if (path === '/multi-agent') return 'multi-agent'
    if (path === '/supervisor') return 'supervisor'
    if (path === '/rag') return 'rag'
    if (path === '/workflows') return 'workflows'
    if (path === '/async-agents') return 'async-agents'
    if (path === '/events') return 'events'
    if (path === '/security') return 'security'
    if (path === '/performance') return 'performance'
    if (path === '/debug') return 'debug'
    if (path === '/agentic-rag') return 'agentic-rag'
    if (path === '/mcp-tools') return 'mcp-tools'
    if (path === '/monitor') return 'monitor'
    if (path === '/enterprise') return 'enterprise'
    if (path === '/streaming') return 'streaming'
    if (path === '/enterprise-config') return 'enterprise-config'
    if (path === '/flow-control') return 'flow-control'
    if (path === '/structured-errors') return 'structured-errors'
    if (path === '/monitoring-dashboard') return 'monitoring-dashboard'
    if (path === '/test-coverage') return 'test-coverage'
    if (path === '/pgvector') return 'pgvector'
    if (path === '/vector-advanced') return 'vector-advanced'
    if (path === '/hybrid-search') return 'hybrid-search'
    if (path === '/multimodal') return 'multimodal'
    if (path === '/multimodal-rag') return 'multimodal-rag'
    if (path === '/cache') return 'cache-monitor'
    if (path === '/batch') return 'batch-jobs'
    if (path === '/health') return 'health'
    if (path === '/test') return 'test-integration'
    if (path === '/agent-interface') return 'agent-interface'
    if (path === '/memory-hierarchy') return 'memory-hierarchy'
    if (path === '/memory-recall') return 'memory-recall'
    if (path === '/memory-analytics') return 'memory-analytics'
    if (path === '/reasoning') return 'reasoning'
    if (path === '/multi-step-reasoning') return 'multi-step-reasoning'
    if (path === '/explainable-ai') return 'explainable-ai'
    if (path === '/offline') return 'offline'
    if (path === '/sync') return 'sync'
    if (path === '/conflicts') return 'conflicts'
    if (path === '/vector-clock') return 'vector-clock'
    if (path === '/network-monitor') return 'network-monitor'
    if (path === '/sync-engine') return 'sync-engine'
    if (path === '/model-cache') return 'model-cache'
    if (path === '/dag-orchestrator') return 'dag-orchestrator'
    return 'chat'
  }

  const menuItems: MenuProps['items'] = [
    // 🤖 核心AI智能体
    {
      key: 'ai-agents-group',
      label: '🤖 核心AI智能体',
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
    },
    {
      key: 'supervisor',
      icon: <ControlOutlined />,
      label: '监督者模式 (Supervisor)',
    },
    {
      key: 'async-agents',
      icon: <ThunderboltOutlined />,
      label: '异步智能体 (AutoGen v0.4)',
    },

    // 🔍 RAG检索系统
    {
      key: 'rag-group',
      label: '🔍 RAG检索系统',
      type: 'group',
    },
    {
      key: 'rag',
      icon: <SearchOutlined />,
      label: 'RAG检索 (基础RAG)',
    },
    {
      key: 'agentic-rag',
      icon: <RobotOutlined />,
      label: 'Agentic RAG (智能检索)',
    },

    // 🧠 记忆管理系统
    {
      key: 'memory-group',
      label: '🧠 记忆管理系统',
      type: 'group',
    },
    {
      key: 'memory-hierarchy',
      icon: <DatabaseOutlined />,
      label: '记忆层级架构',
    },
    {
      key: 'memory-recall',
      icon: <SearchOutlined />,
      label: '记忆召回测试',
    },
    {
      key: 'memory-analytics',
      icon: <DashboardOutlined />,
      label: '记忆分析仪表板',
    },
    {
      key: 'reasoning',
      icon: <BulbOutlined />,
      label: '链式推理 (CoT)',
    },
    {
      key: 'multi-step-reasoning',
      icon: <NodeIndexOutlined />,
      label: '多步推理工作流 (DAG)',
    },
    {
      key: 'dag-orchestrator',
      icon: <NodeIndexOutlined />,
      label: 'DAG编排器 (任务依赖)',
    },
    {
      key: 'explainable-ai',
      icon: <BulbOutlined />,
      label: '可解释AI (XAI)',
    },

    // 🌐 多模态处理
    {
      key: 'multimodal-group',
      label: '🌐 多模态处理',
      type: 'group',
    },
    {
      key: 'multimodal',
      icon: <FileImageOutlined />,
      label: 'GPT-4o多模态 (OpenAI API)',
    },
    {
      key: 'multimodal-rag',
      icon: <SearchOutlined />,
      label: '多模态RAG系统',
    },

    // ⚡ 工作流引擎
    {
      key: 'workflow-group',
      label: '⚡ 工作流引擎',
      type: 'group',
    },
    {
      key: 'workflows',
      icon: <NodeIndexOutlined />,
      label: '工作流可视化 (LangGraph)',
    },
    {
      key: 'flow-control',
      icon: <ThunderboltOutlined />,
      label: '流控背压监控 (AutoGen)',
    },

    // 🔧 协议与工具
    {
      key: 'tools-group',
      label: '🔧 协议与工具',
      type: 'group',
    },
    {
      key: 'mcp-tools',
      icon: <ApiOutlined />,
      label: 'MCP工具 (MCP 1.0协议)',
    },

    // 📊 数据库与向量
    {
      key: 'database-group',
      label: '📊 数据库与向量',
      type: 'group',
    },
    {
      key: 'pgvector',
      icon: <DatabaseOutlined />,
      label: 'pgvector量化 (pgvector 0.8)',
    },
    {
      key: 'vector-advanced',
      icon: <ThunderboltOutlined />,
      label: '高级向量索引 (向量检索)',
    },
    {
      key: 'hybrid-search',
      icon: <SearchOutlined />,
      label: '混合检索 (pgvector + Qdrant)',
    },
    {
      key: 'cache-monitor',
      icon: <DatabaseOutlined />,
      label: '缓存监控 (LangGraph缓存)',
    },

    // 🔄 离线能力与同步
    {
      key: 'offline-group',
      label: '🔄 离线能力与同步',
      type: 'group',
    },
    {
      key: 'offline',
      icon: <CloudServerOutlined />,
      label: '离线能力监控',
    },
    {
      key: 'sync',
      icon: <ThunderboltOutlined />,
      label: '数据同步管理',
    },
    {
      key: 'conflicts',
      icon: <ExceptionOutlined />,
      label: '冲突解决中心',
    },
    {
      key: 'vector-clock',
      icon: <NodeIndexOutlined />,
      label: '向量时钟可视化',
    },
    {
      key: 'network-monitor',
      icon: <MonitorOutlined />,
      label: '网络监控详情',
    },
    {
      key: 'sync-engine',
      icon: <SettingOutlined />,
      label: '同步引擎内部机制',
    },
    {
      key: 'model-cache',
      icon: <DatabaseOutlined />,
      label: '本地模型缓存监控',
    },

    // 🏭 批处理系统
    {
      key: 'batch-group',
      label: '🏭 批处理系统',
      type: 'group',
    },
    {
      key: 'batch-jobs',
      icon: <ThunderboltOutlined />,
      label: '批处理作业 (作业管理)',
    },

    // 📈 系统监控
    {
      key: 'monitoring-group',
      label: '📈 系统监控',
      type: 'group',
    },
    {
      key: 'health',
      icon: <DashboardOutlined />,
      label: '系统健康监控 (健康检查)',
    },
    {
      key: 'performance',
      icon: <DashboardOutlined />,
      label: '性能监控',
    },
    {
      key: 'events',
      icon: <BellOutlined />,
      label: '事件监控',
    },
    {
      key: 'streaming',
      icon: <ThunderboltOutlined />,
      label: '流式监控',
    },
    {
      key: 'monitor',
      icon: <MonitorOutlined />,
      label: '统一监控',
    },
    {
      key: 'monitoring-dashboard',
      icon: <AlertOutlined />,
      label: '监控仪表板',
    },

    // 🏢 企业架构
    {
      key: 'enterprise-group',
      label: '🏢 企业架构',
      type: 'group',
    },
    {
      key: 'enterprise',
      icon: <CloudServerOutlined />,
      label: '企业架构管理',
    },
    {
      key: 'enterprise-config',
      icon: <SettingOutlined />,
      label: '企业配置管理',
    },
    {
      key: 'security',
      icon: <SafetyOutlined />,
      label: '安全管理',
    },
    {
      key: 'debug',
      icon: <BugOutlined />,
      label: '架构调试',
    },

    // 🔬 开发测试
    {
      key: 'dev-test-group',
      label: '🔬 开发测试',
      type: 'group',
    },
    {
      key: 'structured-errors',
      icon: <ExceptionOutlined />,
      label: '结构化错误处理',
    },
    {
      key: 'test-coverage',
      icon: <CheckCircleOutlined />,
      label: '测试覆盖率',
    },
    {
      key: 'test-integration',
      icon: <CheckCircleOutlined />,
      label: '集成测试 (异步数据库/Redis)',
    },
    {
      key: 'agent-interface',
      icon: <ApiOutlined />,
      label: 'Agent接口管理',
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={260}
        collapsedWidth={80}
        style={{ 
          background: '#fff', 
          borderRight: '1px solid #e8e8e8'
        }}
      >
        <div style={{ 
          padding: '16px', 
          borderBottom: '1px solid #e8e8e8',
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
          style={{ 
            border: 'none',
            height: 'calc(100vh - 80px)',
            overflowY: 'auto'
          }}
          onClick={({ key }) => {
            switch (key) {
              case 'chat': navigate('/chat'); break;
              case 'multi-agent': navigate('/multi-agent'); break;
              case 'supervisor': navigate('/supervisor'); break;
              case 'async-agents': navigate('/async-agents'); break;
              case 'rag': navigate('/rag'); break;
              case 'agentic-rag': navigate('/agentic-rag'); break;
              case 'multimodal': navigate('/multimodal'); break;
              case 'multimodal-rag': navigate('/multimodal-rag'); break;
              case 'workflows': navigate('/workflows'); break;
              case 'flow-control': navigate('/flow-control'); break;
              case 'mcp-tools': navigate('/mcp-tools'); break;
              case 'pgvector': navigate('/pgvector'); break;
              case 'vector-advanced': navigate('/vector-advanced'); break;
              case 'hybrid-search': navigate('/hybrid-search'); break;
              case 'cache-monitor': navigate('/cache'); break;
              case 'batch-jobs': navigate('/batch'); break;
              case 'health': navigate('/health'); break;
              case 'performance': navigate('/performance'); break;
              case 'events': navigate('/events'); break;
              case 'streaming': navigate('/streaming'); break;
              case 'monitor': navigate('/monitor'); break;
              case 'monitoring-dashboard': navigate('/monitoring-dashboard'); break;
              case 'enterprise': navigate('/enterprise'); break;
              case 'enterprise-config': navigate('/enterprise-config'); break;
              case 'security': navigate('/security'); break;
              case 'debug': navigate('/debug'); break;
              case 'structured-errors': navigate('/structured-errors'); break;
              case 'test-coverage': navigate('/test-coverage'); break;
              case 'test-integration': navigate('/test'); break;
              case 'agent-interface': navigate('/agent-interface'); break;
              case 'memory-hierarchy': navigate('/memory-hierarchy'); break;
              case 'memory-recall': navigate('/memory-recall'); break;
              case 'memory-analytics': navigate('/memory-analytics'); break;
              case 'reasoning': navigate('/reasoning'); break;
              case 'multi-step-reasoning': navigate('/multi-step-reasoning'); break;
              case 'explainable-ai': navigate('/explainable-ai'); break;
              case 'offline': navigate('/offline'); break;
              case 'sync': navigate('/sync'); break;
              case 'conflicts': navigate('/conflicts'); break;
              case 'vector-clock': navigate('/vector-clock'); break;
              case 'network-monitor': navigate('/network-monitor'); break;
              case 'sync-engine': navigate('/sync-engine'); break;
              case 'model-cache': navigate('/model-cache'); break;
              case 'dag-orchestrator': navigate('/dag-orchestrator'); break;
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
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default FinalWorkingLayout
export { FinalWorkingLayout }