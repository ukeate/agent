import React from 'react'
import ConversationContainer from '../components/conversation/ConversationContainer'
import { useChat } from '../hooks/useChat'

const ChatPage: React.FC = () => {
  const { messages, loading, error, sendMessage, clearChat } = useChat()

  return (
    <ConversationContainer
      messages={messages}
      loading={loading}
      error={error}
      onSendMessage={sendMessage}
      onClearHistory={clearChat}
    />
  )
}

export default ChatPage