import React from 'react'
import { MainLayout } from '../components/layout/MainLayout'
import ConversationContainer from '../components/conversation/ConversationContainer'
import { useChat } from '../hooks/useChat'

export const ChatPage: React.FC = () => {
  const { messages, loading, error, sendMessage, clearChat } = useChat()

  return (
    <MainLayout>
      <ConversationContainer
        messages={messages}
        loading={loading}
        error={error}
        onSendMessage={sendMessage}
        onClearHistory={clearChat}
      />
    </MainLayout>
  )
}