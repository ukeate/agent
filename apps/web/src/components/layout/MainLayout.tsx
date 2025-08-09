import React, { useState } from 'react'
import { Layout, Menu, Button, Typography, Space, Avatar } from 'antd'
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
    if (location.pathname === '/multi-agent') return 'multi-agent'
    if (location.pathname === '/supervisor') return 'supervisor'
    if (location.pathname === '/rag') return 'rag'
    if (location.pathname === '/workflows') return 'workflows'
    if (location.pathname === '/chat' || location.pathname === '/') return 'chat'
    return 'chat'
  }

  const menuItems = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '单代理对话',
      onClick: () => navigate('/chat'),
    },
    {
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多代理协作',
      onClick: () => navigate('/multi-agent'),
    },
    {
      key: 'supervisor',
      icon: <ControlOutlined />,
      label: '监督者模式',
      onClick: () => navigate('/supervisor'),
    },
    {
      key: 'rag',
      icon: <SearchOutlined />,
      label: 'RAG检索',
      onClick: () => navigate('/rag'),
    },
    {
      key: 'workflows',
      icon: <NodeIndexOutlined />,
      label: '工作流可视化',
      onClick: () => navigate('/workflows'),
    },
  ]

  return (
    <Layout className="min-h-screen">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={240}
        collapsedWidth={80}
        className="bg-white border-r border-gray-200 shadow-sm"
      >
        <div className="p-4 border-b border-gray-200">
          <Space align="center" className={collapsed ? 'justify-center' : ''}>
            <Avatar 
              size={collapsed ? 32 : 40} 
              icon={<RobotOutlined />} 
              className="bg-primary-500"
            />
            {!collapsed && (
              <div>
                <Title level={5} className="!mb-0">AI Agent</Title>
                <Text type="secondary" className="text-xs">智能助手系统</Text>
              </div>
            )}
          </Space>
        </div>
        
        <Menu
          mode="inline"
          selectedKeys={[getSelectedKey()]}
          items={menuItems}
          className="border-none"
          onClick={({ key }) => {
            const item = menuItems.find(item => item.key === key)
            item?.onClick?.()
          }}
        />
      </Sider>

      <Layout>
        <Header className="bg-white border-b border-gray-200 !px-6 !py-3">
          <div className="flex items-center justify-between">
            <Space align="center">
              <Button
                type="text"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={() => setCollapsed(!collapsed)}
                className="text-lg"
              />
            </Space>
          </div>
        </Header>

        <Content className="bg-gray-50 flex flex-col">
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default MainLayout
export { MainLayout }