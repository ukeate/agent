import React, { useState } from 'react'
import { Layout, Menu, Button, Typography, Space, Avatar } from 'antd'
import type { MenuProps } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  MessageOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  RobotOutlined,
} from '@ant-design/icons'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

interface MinimalMainLayoutProps {
  children: React.ReactNode
}

const MinimalMainLayout: React.FC<MinimalMainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()

  // 最简化的菜单项
  const menuItems: MenuProps['items'] = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '单代理对话',
    },
    {
      key: 'cache',
      icon: <MessageOutlined />,
      label: '缓存监控',
    },
    {
      key: 'batch',
      icon: <MessageOutlined />,
      label: '批处理作业',
    },
    {
      key: 'health',
      icon: <MessageOutlined />,
      label: '系统健康',
    },
  ]

  const getSelectedKey = () => {
    if (location.pathname === '/cache') return 'cache'
    if (location.pathname === '/batch') return 'batch'
    if (location.pathname === '/health') return 'health'
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
        style={{ 
          background: '#fff', 
          borderRight: '1px solid #f0f0f0'
        }}
      >
        <div style={{ 
          padding: '16px', 
          borderBottom: '1px solid #f0f0f0'
        }}>
          <Space align="center">
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
              case 'cache': navigate('/cache'); break;
              case 'batch': navigate('/batch'); break;
              case 'health': navigate('/health'); break;
            }
          }}
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
            alignItems: 'center'
          }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: '16px' }}
            />
          </div>
        </Header>

        <Content style={{ 
          background: '#f5f5f5', 
          padding: '24px'
        }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default MinimalMainLayout
export { MinimalMainLayout }