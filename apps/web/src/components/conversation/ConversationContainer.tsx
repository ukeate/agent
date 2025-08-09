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
  onClearHistory?: () => void
}

const ConversationContainer: React.FC<ConversationContainerProps> = ({
  messages,
  loading,
  error,
  onSendMessage,
  onClearHistory,
}) => {
  return (
    <div className="flex-1 flex flex-col max-h-full">
      {error && (
        <div className="m-4 mb-0">
          <NetworkErrorAlert 
            error={error}
            onRetry={() => window.location.reload()}
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
            flexDirection: 'column' 
          }
        }}
      >
        <MessageList 
          messages={messages} 
          loading={loading}
          onClearHistory={onClearHistory}
        />
        <MessageInput 
          onSendMessage={onSendMessage} 
          loading={loading} 
        />
      </Card>
    </div>
  )
}

export default ConversationContainer