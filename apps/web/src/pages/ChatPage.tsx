import React from 'react'
import { Button } from 'antd'
import { HistoryOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import ConversationContainer from '../components/conversation/ConversationContainer'
import AgentStatusBar from '../components/agent/AgentStatusBar'
import { useChat } from '../hooks/useChat'

const ChatPage: React.FC = () => {
  const navigate = useNavigate()
  const {
    currentConversation,
    messages,
    loading,
    error,
    sendMessage,
    stopStreaming,
    retryLastMessage,
    dismissError,
    clearChat,
    startNewConversation,
  } = useChat()

  return (
    <div className="flex-1 flex flex-col">
      <div className="m-4 mb-0 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold">单代理对话</h1>
          <AgentStatusBar />
        </div>
        <Button
          size="small"
          icon={<HistoryOutlined />}
          onClick={() => navigate('/history')}
        >
          历史记录
        </Button>
      </div>
      <ConversationContainer
        messages={messages}
        loading={loading}
        error={error}
        onSendMessage={sendMessage}
        onStop={stopStreaming}
        onClearHistory={clearChat}
        onStartNewConversation={startNewConversation}
        onRetrySend={retryLastMessage}
        onDismissError={dismissError}
        draftKey={currentConversation?.id}
        conversationId={currentConversation?.id}
      />
    </div>
  )
}

export default ChatPage
