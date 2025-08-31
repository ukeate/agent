import React from 'react'
import { Typography } from 'antd'
import { MultiAgentChatContainer } from '../components/multi-agent/MultiAgentChatContainer'
import ErrorBoundary from '../components/ui/ErrorBoundary'

const { Title } = Typography

const MultiAgentPage: React.FC = () => {
  return (
    <div className="h-full flex flex-col p-6">
      <Title level={1}>多智能体协作</Title>
      <ErrorBoundary>
        <MultiAgentChatContainer className="h-full" />
      </ErrorBoundary>
    </div>
  )
}

export default MultiAgentPage