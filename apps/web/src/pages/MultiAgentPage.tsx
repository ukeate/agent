import React from 'react'
import { MultiAgentChatContainer } from '../components/multi-agent/MultiAgentChatContainer'
import { ErrorBoundary } from '../components/ui/ErrorBoundary'

const MultiAgentPage: React.FC = () => {
  return (
    <div className="h-full flex flex-col p-6">
      <ErrorBoundary>
        <MultiAgentChatContainer className="h-full" />
      </ErrorBoundary>
    </div>
  )
}

export default MultiAgentPage