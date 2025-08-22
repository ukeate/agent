import React, { useState } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Typography } from 'antd'
import { 
  MessageOutlined, 
  RobotOutlined, 
  ControlOutlined,
  SearchOutlined,
  MonitorOutlined,
  TrophyOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PlayCircleOutlined 
} from '@ant-design/icons'

const { Sider, Content } = Layout
const { Title } = Typography

const WorkingCompleteApp: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)

  // RL监控相关页面菜单项
  const rlMonitoringItems = [
    {
      key: '/qlearning',
      icon: <TrophyOutlined />,
      label: 'Q-Learning系统'
    },
    {
      key: '/qlearning-performance',
      icon: <BarChartOutlined />,
      label: 'Q-Learning性能监控'
    },
    {
      key: '/qlearning-training',
      icon: <PlayCircleOutlined />,
      label: 'Q-Learning训练页面'
    },
    {
      key: '/qlearning-recommendation',
      icon: <LineChartOutlined />,
      label: 'Q-Learning推荐系统'
    },
    {
      key: '/qlearning-strategy',
      icon: <ControlOutlined />,
      label: 'Q-Learning策略页面'
    }
  ]

  // 核心功能菜单项
  const coreItems = [
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: '聊天对话'
    },
    {
      key: '/multi-agent',
      icon: <RobotOutlined />,
      label: '多智能体'
    },
    {
      key: '/supervisor',
      icon: <ControlOutlined />,
      label: '监督者'
    },
    {
      key: '/rag',
      icon: <SearchOutlined />,
      label: 'RAG检索'
    },
    {
      key: '/workflow',
      icon: <MonitorOutlined />,
      label: '工作流'
    }
  ]

  const menuItems = [
    {
      key: 'core',
      label: '核心功能',
      type: 'group',
      children: coreItems
    },
    {
      key: 'rl-monitoring',
      label: 'RL监控系统',
      type: 'group', 
      children: rlMonitoringItems
    }
  ]

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key)
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        width={250} 
        theme="light" 
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
      >
        <div style={{ padding: '16px', textAlign: 'center' }}>
          <Title level={4} style={{ margin: 0 }}>
            {collapsed ? 'AI' : 'AI Agent 系统'}
          </Title>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
          style={{ borderRight: 0 }}
        />
      </Sider>
      <Layout>
        <Content style={{ padding: '24px', background: '#f0f2f5' }}>
          <div style={{ background: '#fff', padding: '24px', borderRadius: '6px' }}>
            <Routes>
              <Route path="/" element={
                <div>
                  <Title level={2}>欢迎使用AI Agent系统</Title>
                  <p>这是一个完整的多智能体系统，包含完整的RL监控功能</p>
                </div>
              } />
              <Route path="/chat" element={<div><Title level={3}>聊天页面</Title><p>智能对话功能</p></div>} />
              <Route path="/multi-agent" element={<div><Title level={3}>多智能体页面</Title><p>多智能体协作系统</p></div>} />
              <Route path="/supervisor" element={<div><Title level={3}>监督者页面</Title><p>系统监督和管理</p></div>} />
              <Route path="/rag" element={<div><Title level={3}>RAG检索页面</Title><p>知识检索增强生成</p></div>} />
              <Route path="/workflow" element={<div><Title level={3}>工作流页面</Title><p>智能工作流管理</p></div>} />
              
              {/* RL监控页面 */}
              <Route path="/qlearning" element={
                <div>
                  <Title level={3}>Q-Learning系统</Title>
                  <p>强化学习Q-Learning算法系统</p>
                  <div style={{ marginTop: '20px' }}>
                    <h4>功能特性：</h4>
                    <ul>
                      <li>Q表管理和更新</li>
                      <li>策略评估和优化</li>
                      <li>学习率自适应调整</li>
                      <li>探索与利用平衡</li>
                    </ul>
                  </div>
                </div>
              } />
              <Route path="/qlearning-performance" element={
                <div>
                  <Title level={3}>Q-Learning性能监控</Title>
                  <p>实时监控Q-Learning系统性能指标</p>
                  <div style={{ marginTop: '20px' }}>
                    <h4>监控指标：</h4>
                    <ul>
                      <li>学习收敛率</li>
                      <li>奖励函数趋势</li>
                      <li>Q值分布统计</li>
                      <li>动作选择频率</li>
                    </ul>
                  </div>
                </div>
              } />
              <Route path="/qlearning-training" element={
                <div>
                  <Title level={3}>Q-Learning训练页面</Title>
                  <p>Q-Learning模型训练和参数调优</p>
                  <div style={{ marginTop: '20px' }}>
                    <h4>训练功能：</h4>
                    <ul>
                      <li>训练过程可视化</li>
                      <li>超参数调优</li>
                      <li>批量训练管理</li>
                      <li>模型版本控制</li>
                    </ul>
                  </div>
                </div>
              } />
              <Route path="/qlearning-recommendation" element={
                <div>
                  <Title level={3}>Q-Learning推荐系统</Title>
                  <p>基于Q-Learning的智能推荐系统</p>
                  <div style={{ marginTop: '20px' }}>
                    <h4>推荐功能：</h4>
                    <ul>
                      <li>个性化推荐</li>
                      <li>多臂老虎机算法</li>
                      <li>上下文感知推荐</li>
                      <li>推荐效果评估</li>
                    </ul>
                  </div>
                </div>
              } />
              <Route path="/qlearning-strategy" element={
                <div>
                  <Title level={3}>Q-Learning策略页面</Title>
                  <p>Q-Learning策略制定和优化</p>
                  <div style={{ marginTop: '20px' }}>
                    <h4>策略管理：</h4>
                    <ul>
                      <li>策略模板管理</li>
                      <li>策略效果对比</li>
                      <li>A/B测试框架</li>
                      <li>策略自动切换</li>
                    </ul>
                  </div>
                </div>
              } />
            </Routes>
          </div>
        </Content>
      </Layout>
    </Layout>
  )
}

export default WorkingCompleteApp