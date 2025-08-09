import React from 'react'
import { Avatar, Typography, Tag, Space } from 'antd'
import { UserOutlined, RobotOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { Message } from '../../types'
import ToolCallDisplay from '../agent/ToolCallDisplay'
import ReasoningSteps from '../agent/ReasoningSteps'
import MarkdownRenderer from '../ui/MarkdownRenderer'

const { Text, Paragraph } = Typography

interface MessageItemProps {
  message: Message
}

const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const isUser = message.role === 'user'
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} message-item mb-4`}>
      <div className={`flex max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start gap-3`}>
        <Avatar
          size={36}
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          className={`flex-shrink-0 shadow-sm ${
            isUser ? 'bg-gradient-to-r from-blue-500 to-blue-600' : 'bg-gradient-to-r from-emerald-500 to-emerald-600'
          }`}
        />
        
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm ${
            isUser 
              ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white' 
              : 'bg-white border border-gray-100'
          }`}
        >
          <div className="space-y-2">
            {/* 消息头部 */}
            <div className="flex items-center justify-between">
              <Space size="small">
                <Text strong className={`text-sm ${isUser ? 'text-white' : 'text-gray-700'}`}>
                  {isUser ? '用户' : 'AI助手'}
                </Text>
                {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
                  <Tag color="blue">
                    {message.toolCalls.length}个工具
                  </Tag>
                )}
              </Space>
              <Text className={`text-xs ${isUser ? 'text-blue-100' : 'text-gray-400'}`}>
                {dayjs(message.timestamp).format('HH:mm:ss')}
              </Text>
            </div>

            {/* 推理步骤 */}
            {!isUser && message.reasoningSteps && message.reasoningSteps.length > 0 && (
              <ReasoningSteps steps={message.reasoningSteps} />
            )}

            {/* 消息内容 */}
            <div className={`${isUser ? 'text-white' : 'text-gray-800'}`}>
              {isUser ? (
                <Paragraph className={`!mb-0 whitespace-pre-wrap ${isUser ? 'text-white' : ''}`}>
                  {message.content}
                </Paragraph>
              ) : (
                <MarkdownRenderer 
                  content={message.content} 
                  className="!mb-0"
                />
              )}
            </div>

            {/* 工具调用显示 */}
            {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
              <div className="mt-3">
                <ToolCallDisplay toolCalls={message.toolCalls} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MessageItem