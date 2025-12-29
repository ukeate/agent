import React from 'react'
import { Typography } from 'antd'
import { MultiAgentChatContainer } from '../components/multi-agent/MultiAgentChatContainer'
import ErrorBoundary from '../components/ui/ErrorBoundary'

const { Title } = Typography

const MultiAgentPage: React.FC = () => {
  return (
    <div className="h-full flex flex-col p-6">
      <Title level={1}>多智能体协作</Title>
      <p className="text-gray-600 mb-4">多个AI专家协作讨论</p>
      <ErrorBoundary>
        <MultiAgentChatContainer className="h-full" />
      </ErrorBoundary>
    </div>
  )
}

export default MultiAgentPage
