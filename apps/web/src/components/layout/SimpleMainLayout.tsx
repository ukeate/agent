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
  SearchOutlined,
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

interface SimpleMainLayoutProps {
  children: React.ReactNode
}

const SimpleMainLayout: React.FC<SimpleMainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()

  // 简化的菜单项
  const menuItems: MenuProps['items'] = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '单代理对话',
    },
    {
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多代理协作',
    },
    {
      key: 'rag',
      icon: <SearchOutlined />,
      label: 'RAG检索',
    },
  ]

  // 简化的选中逻辑
  const getSelectedKey = () => {
    if (location.pathname === '/multi-agent') return 'multi-agent'
    if (location.pathname === '/rag') return 'rag'
    return 'chat'
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={240}
        collapsedWidth={80}
        style={{ background: '#fff', borderRight: '1px solid #f0f0f0' }}
      >
        <div style={{ padding: '16px', borderBottom: '1px solid #f0f0f0' }}>
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
          style={{ border: 'none' }}
          onClick={({ key }) => {
            switch (key) {
              case 'chat': navigate('/chat'); break;
              case 'multi-agent': navigate('/multi-agent'); break;
              case 'rag': navigate('/rag'); break;
            }
          }}
        />
      </Sider>

      <Layout>
        <Header style={{ background: '#fff', borderBottom: '1px solid #f0f0f0', padding: '0 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
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

        <Content style={{ background: '#f5f5f5', display: 'flex', flexDirection: 'column' }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default SimpleMainLayout
export { SimpleMainLayout }