import React from 'react'
import { MainLayout } from '../components/layout/MainLayout'
import { MultiAgentChatContainer } from '../components/multi-agent/MultiAgentChatContainer'
import { ErrorBoundary } from '../components/ui/ErrorBoundary'

export const MultiAgentPage: React.FC = () => {
  return (
    <MainLayout>
      <div className="h-full flex flex-col p-6">
        <ErrorBoundary>
          <MultiAgentChatContainer className="h-full" />
        </ErrorBoundary>
      </div>
    </MainLayout>
  )
}