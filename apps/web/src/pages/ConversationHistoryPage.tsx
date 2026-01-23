import React from 'react'
import { message } from 'antd'
import { useNavigate } from 'react-router-dom'
import ConversationHistory from '../components/conversation/ConversationHistory'
import { useConversationStore } from '../stores/conversationStore'

const ConversationHistoryPage: React.FC = () => {
  const navigate = useNavigate()
  const { loadConversation } = useConversationStore()

  return (
    <ConversationHistory
      visible
      onSelectConversation={async conversation => {
        try {
          await loadConversation(conversation.id)
          navigate('/chat')
        } catch {
          message.error('加载对话失败')
        }
      }}
    />
  )
}

export default ConversationHistoryPage
