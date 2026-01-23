import React from 'react'
import { Card } from 'antd'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import NetworkErrorAlert from '../ui/NetworkErrorAlert'
import { Message } from '../../types'

interface ConversationContainerProps {
  messages: Message[]
  loading: boolean
  error?: string | null
  onSendMessage: (message: string) => void
  onStop?: () => void
  onClearHistory?: () => void | Promise<void>
  onStartNewConversation?: () => void | Promise<void>
  onRetrySend?: () => void
  onDismissError?: () => void
  draftKey?: string
  conversationId?: string
}

const ConversationContainer: React.FC<ConversationContainerProps> = ({
  messages,
  loading,
  error,
  onSendMessage,
  onStop,
  onClearHistory,
  onStartNewConversation,
  onRetrySend,
  onDismissError,
  draftKey,
  conversationId,
}) => {
  return (
    <div className="flex-1 flex flex-col max-h-full">
      {error && (
        <div className="m-4 mb-0">
          <NetworkErrorAlert
            error={error}
            onRetry={onRetrySend}
            onDismiss={onDismissError}
          />
        </div>
      )}
      <Card
        className="flex-1 flex flex-col m-4 !p-0 overflow-hidden"
        styles={{
          body: {
            padding: 0,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        <MessageList
          messages={messages}
          loading={loading}
          onClearHistory={onClearHistory}
          onStartNewConversation={onStartNewConversation}
          onRetrySend={onRetrySend}
          conversationId={conversationId}
        />
        <MessageInput
          onSendMessage={onSendMessage}
          onStop={onStop}
          loading={loading}
          draftKey={draftKey}
        />
      </Card>
    </div>
  )
}

export default ConversationContainer
