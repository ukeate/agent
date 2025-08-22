import React from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Typography } from 'antd'
import { MessageOutlined, RobotOutlined } from '@ant-design/icons'

const { Sider, Content } = Layout
const { Title } = Typography

const MinimalApp: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: '聊天对话'
    },
    {
      key: '/multi-agent',
      icon: <RobotOutlined />,
      label: '多智能体'
    }
  ]

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key)
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="light">
        <div style={{ padding: '16px', textAlign: 'center' }}>
          <Title level={4} style={{ margin: 0 }}>AI Agent</Title>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={handleMenuClick}
        />
      </Sider>
      <Layout>
        <Content style={{ padding: '24px' }}>
          <Routes>
            <Route path="/" element={<div>欢迎使用AI Agent系统</div>} />
            <Route path="/chat" element={<div>聊天页面</div>} />
            <Route path="/multi-agent" element={<div>多智能体页面</div>} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}

export default MinimalApp