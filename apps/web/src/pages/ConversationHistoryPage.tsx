import React from 'react'
import { useNavigate } from 'react-router-dom'
import ConversationHistory from '../components/conversation/ConversationHistory'
import { useConversationStore } from '../stores/conversationStore'

const ConversationHistoryPage: React.FC = () => {
  const navigate = useNavigate()
  const { loadConversation } = useConversationStore()

  return (
    <ConversationHistory
      visible
      onSelectConversation={async (conversation) => {
        await loadConversation(conversation.id)
        navigate('/chat')
      }}
    />
  )
}

export default ConversationHistoryPage
