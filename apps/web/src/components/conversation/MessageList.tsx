import React, { useEffect, useRef } from 'react'
import { Empty, Button, Space, Spin } from 'antd'
import { ClearOutlined } from '@ant-design/icons'
import MessageItem from './MessageItem'
import { Message } from '@/types'

interface MessageListProps {
  messages: Message[]
  loading: boolean
  onClearHistory?: () => void
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  loading,
  onClearHistory,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex flex-col">
        {/* 始终显示清空对话按钮 */}
        <div className="border-b border-gray-100 p-3">
          <Space>
            <Button
              type="text"
              size="small"
              icon={<ClearOutlined />}
              onClick={onClearHistory}
              className="text-gray-500"
            >
              新建对话
            </Button>
          </Space>
        </div>

        <div className="flex-1 flex items-center justify-center p-8">
          <Empty
            description="开始与AI智能体对话"
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <p className="text-gray-500 text-sm mt-4">
              你可以询问任何问题，智能体会使用各种工具来帮助你
            </p>
          </Empty>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* 始终显示清空对话按钮 */}
      <div className="border-b border-gray-100 p-3">
        <Space>
          <Button
            type="text"
            size="small"
            icon={<ClearOutlined />}
            onClick={onClearHistory}
            className="text-gray-500"
          >
            {messages.length > 0 ? '清空对话' : '新建对话'}
          </Button>
          {messages.length > 0 && (
            <span className="text-gray-400 text-xs">
              共 {messages.length} 条消息
            </span>
          )}
        </Space>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="p-4 space-y-4">
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
          
          {loading && (
            <div className="flex justify-center py-4">
              <Spin size="small" />
              <span className="ml-2 text-gray-500">智能体正在思考...</span>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>
    </div>
  )
}

export default MessageList