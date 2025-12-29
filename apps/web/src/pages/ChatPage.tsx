import React from 'react'
import ConversationContainer from '../components/conversation/ConversationContainer'
import { useChat } from '../hooks/useChat'

const ChatPage: React.FC = () => {
  const { messages, loading, error, sendMessage, clearChat } = useChat()

  return (
    <div className="flex-1 flex flex-col">
      <h1 className="m-4 mb-0 text-xl font-semibold">单代理对话</h1>
      <ConversationContainer
        messages={messages}
        loading={loading}
        error={error}
        onSendMessage={sendMessage}
        onClearHistory={clearChat}
      />
    </div>
  )
}

export default ChatPage
