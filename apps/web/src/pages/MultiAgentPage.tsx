import React, { useState } from 'react'
import { Typography, Tabs } from 'antd'
import { MultiAgentChatContainer } from '../components/multi-agent/MultiAgentChatContainer'
import MultiAgentHistory from '../components/multi-agent/MultiAgentHistory'
import ErrorBoundary from '../components/ui/ErrorBoundary'

const { Title } = Typography

const MultiAgentPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('chat')

  return (
    <div className="h-full flex flex-col p-6">
      <div className="mb-4">
        <Title level={1}>多智能体协作</Title>
        <p className="text-gray-600">多个AI专家协作讨论</p>
      </div>
      <ErrorBoundary>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          className="flex-1 min-h-0"
          items={[
            {
              key: 'chat',
              label: '实时协作',
              children: <MultiAgentChatContainer className="h-full" />,
            },
            {
              key: 'history',
              label: '历史记录',
              children: (
                <MultiAgentHistory
                  visible={activeTab === 'history'}
                  onSelectSession={() => setActiveTab('chat')}
                />
              ),
            },
          ]}
        />
      </ErrorBoundary>
    </div>
  )
}

export default MultiAgentPage
