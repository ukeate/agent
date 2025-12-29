import React, { useEffect } from 'react'
import { List, Card, Typography, Space, Button, Empty, Tag } from 'antd'
import { MessageOutlined, DeleteOutlined, CalendarOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { Conversation } from '@/types'
import { useConversationStore } from '@/stores/conversationStore'

const { Text, Title } = Typography

interface ConversationHistoryProps {
  visible: boolean
  onSelectConversation: (conversation: Conversation) => void
}

const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  visible,
  onSelectConversation,
}) => {
  const { conversations, deleteConversation, currentConversation, refreshConversations } = useConversationStore()

  useEffect(() => {
    refreshConversations()
  }, [refreshConversations])

  const handleDeleteConversation = (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteConversation(conversationId)
  }

  if (!visible) return null

  return (
    <div className="p-4">
      <div className="mb-4">
        <Title level={4} className="!mb-2">
          对话历史
        </Title>
        <Text type="secondary">共 {conversations.length} 个对话</Text>
      </div>

      {conversations.length === 0 ? (
        <Empty
          description="暂无对话历史"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <List
          dataSource={conversations}
          renderItem={(conversation) => (
            <List.Item className="!px-0">
              <Card
                hoverable
                size="small"
                className={`w-full cursor-pointer ${
                  currentConversation?.id === conversation.id
                    ? 'border-primary-400 bg-primary-50'
                    : ''
                }`}
                onClick={() => onSelectConversation(conversation)}
                actions={[
                  <Button
                    key="delete"
                    type="text"
                    size="small"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={(e) => handleDeleteConversation(conversation.id, e)}
                  />,
                ]}
              >
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Text strong className="text-sm truncate">
                      {conversation.title}
                    </Text>
                    <Tag color="blue">
                      {(conversation.messageCount ?? conversation.messages.length) || 0} 条消息
                    </Tag>
                  </div>

                  {conversation.messages.length > 0 && (
                    <Text
                      type="secondary"
                      className="text-xs line-clamp-2"
                    >
                      {conversation.messages[0].content}
                    </Text>
                  )}

                  <div className="flex items-center justify-between">
                    <Space size="small">
                      <MessageOutlined className="text-gray-400" />
                      <Text type="secondary" className="text-xs">
                        {(conversation.userMessageCount ??
                          conversation.messages.filter((m) => m.role === 'user').length) || 0}{' '}
                        次提问
                      </Text>
                    </Space>
                    <Space size="small">
                      <CalendarOutlined className="text-gray-400" />
                      <Text type="secondary" className="text-xs">
                        {dayjs(conversation.updatedAt).format('MM-DD HH:mm')}
                      </Text>
                    </Space>
                  </div>
                </div>
              </Card>
            </List.Item>
          )}
        />
      )}
    </div>
  )
}

export default ConversationHistory
