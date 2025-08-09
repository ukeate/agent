import React, { useState } from 'react'
import { Layout, Menu, Button, Typography, Space, Avatar, Breadcrumb, Drawer } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  MessageOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  UserOutlined,
  RobotOutlined,
  HistoryOutlined,
  TeamOutlined,
  HomeOutlined,
  ControlOutlined,
} from '@ant-design/icons'
import ConversationHistory from '../conversation/ConversationHistory'
import MultiAgentHistory from '../multi-agent/MultiAgentHistory'
import { useConversationStore } from '../../stores/conversationStore'
import { useMultiAgentStore, ConversationSession } from '../../stores/multiAgentStore'
import { Conversation } from '../../types'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

interface MainLayoutProps {
  children: React.ReactNode
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  const [showChatHistory, setShowChatHistory] = useState(false)
  const [showMultiAgentHistory, setShowMultiAgentHistory] = useState(false)
  
  const navigate = useNavigate()
  const location = useLocation()
  const { setCurrentConversation } = useConversationStore()
  const { loadSessionHistory } = useMultiAgentStore()

  const handleSelectConversation = (conversation: Conversation) => {
    setCurrentConversation(conversation)
    navigate('/chat')
    setShowChatHistory(false)
  }

  const handleSelectMultiAgentSession = async (session: ConversationSession) => {
    try {
      await loadSessionHistory(session.session_id)
      navigate('/multi-agent')
      setShowMultiAgentHistory(false)
    } catch (error) {
      console.error('加载多智能体会话失败:', error)
    }
  }

  // 根据当前路径确定选中的菜单项
  const getSelectedKey = () => {
    if (location.pathname === '/multi-agent') return 'multi-agent'
    if (location.pathname === '/supervisor') return 'supervisor'
    if (location.pathname === '/chat' || location.pathname === '/') return 'chat'
    return 'chat'
  }

  // 根据当前路径生成面包屑项目
  const getBreadcrumbItems = () => {
    const items = [
      {
        title: (
          <span className="cursor-pointer flex items-center gap-1" onClick={() => navigate('/chat')}>
            <HomeOutlined />
            首页
          </span>
        ),
      },
    ]

    if (location.pathname === '/multi-agent') {
      items.push({
        title: <span>多智能体协作</span>,
      })
    } else if (location.pathname === '/supervisor') {
      items.push({
        title: <span>Supervisor 监控</span>,
      })
    } else if (location.pathname === '/chat' || location.pathname === '/') {
      items.push({
        title: <span>智能对话</span>,
      })
    }

    return items
  }

  const menuItems = [
    {
      key: 'chat',
      icon: <MessageOutlined />,
      label: '智能对话',
      onClick: () => navigate('/chat'),
    },
    {
      key: 'multi-agent',
      icon: <TeamOutlined />,
      label: '多智能体协作',
      onClick: () => navigate('/multi-agent'),
    },
    {
      key: 'supervisor',
      icon: <ControlOutlined />,
      label: 'Supervisor 监控',
      onClick: () => navigate('/supervisor'),
    },
    {
      key: 'chat-history',
      icon: <HistoryOutlined />,
      label: '对话历史',
      onClick: () => setShowChatHistory(!showChatHistory),
    },
    {
      key: 'multi-agent-history',
      icon: <HistoryOutlined />,
      label: '协作历史',
      onClick: () => setShowMultiAgentHistory(!showMultiAgentHistory),
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '设置',
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
              <Breadcrumb 
                items={getBreadcrumbItems()}
                className="text-sm"
              />
            </Space>

            <Space align="center">
              <Avatar icon={<UserOutlined />} size="small" />
              <Text className="text-sm">用户</Text>
            </Space>
          </div>
        </Header>

        <Content className="bg-gray-50 flex flex-col">
          {children}
        </Content>
      </Layout>

      {/* 对话历史抽屉 */}
      <Drawer
        title="对话历史"
        placement="right"
        open={showChatHistory}
        onClose={() => setShowChatHistory(false)}
        width={400}
        className="history-drawer"
      >
        <ConversationHistory
          visible={showChatHistory}
          onSelectConversation={(conversation) => {
            handleSelectConversation(conversation)
            setShowChatHistory(false)
          }}
        />
      </Drawer>

      {/* 多智能体历史抽屉 */}
      <Drawer
        title="多智能体协作历史"
        placement="right"
        open={showMultiAgentHistory}
        onClose={() => setShowMultiAgentHistory(false)}
        width={400}
        className="history-drawer"
      >
        <MultiAgentHistory
          visible={showMultiAgentHistory}
          onSelectSession={(session) => {
            handleSelectMultiAgentSession(session)
            setShowMultiAgentHistory(false)
          }}
        />
      </Drawer>
    </Layout>
  )
}

export default MainLayout
export { MainLayout }