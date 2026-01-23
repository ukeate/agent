import React from 'react'
import {
  Avatar,
  Typography,
  Tag,
  Space,
  Button,
  Tooltip,
  message as antdMessage,
} from 'antd'
import { UserOutlined, RobotOutlined, CopyOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { Message } from '../../types'
import ToolCallDisplay from '../agent/ToolCallDisplay'
import ReasoningSteps from '../agent/ReasoningSteps'
import MarkdownRenderer from '../ui/MarkdownRenderer'
import { copyToClipboard } from '../../utils/clipboard'
import { renderHighlightedText } from '../../utils/highlightText'
import { buildMessageExportText } from '@/utils/conversationExport'

const { Text, Paragraph } = Typography

interface MessageItemProps {
  message: Message
  highlightQuery?: string
}

const MessageItem: React.FC<MessageItemProps> = ({
  message,
  highlightQuery,
}) => {
  const isUser = message.role === 'user'
  const exportText = buildMessageExportText(message)
  const canCopy = exportText.trim().length > 0
  const timestamp = message.timestamp ? dayjs(message.timestamp) : null
  const timeText = timestamp ? timestamp.format('HH:mm:ss') : '--'
  const fullTimeText = timestamp
    ? timestamp.format('YYYY-MM-DD HH:mm:ss')
    : ''

  const handleCopy = async () => {
    if (!canCopy) return
    try {
      await copyToClipboard(exportText)
      antdMessage.success('已复制')
    } catch {
      antdMessage.error('复制失败')
    }
  }

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} message-item mb-4`}
    >
      <div
        className={`flex max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start gap-3`}
      >
        <Avatar
          size={36}
          icon={isUser ? <UserOutlined /> : <RobotOutlined />}
          className={`flex-shrink-0 shadow-sm ${
            isUser
              ? 'bg-gradient-to-r from-blue-500 to-blue-600'
              : 'bg-gradient-to-r from-emerald-500 to-emerald-600'
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
                <Text
                  strong
                  className={`text-sm ${isUser ? 'text-white' : 'text-gray-700'}`}
                >
                  {isUser ? '用户' : 'AI助手'}
                </Text>
                {!isUser &&
                  message.toolCalls &&
                  message.toolCalls.length > 0 && (
                    <Tag color="blue">{message.toolCalls.length}个工具</Tag>
                  )}
              </Space>
              <Space size="small">
                {fullTimeText ? (
                  <Tooltip title={fullTimeText}>
                    <Text
                      className={`text-xs ${isUser ? 'text-blue-100' : 'text-gray-400'}`}
                    >
                      {timeText}
                    </Text>
                  </Tooltip>
                ) : (
                  <Text
                    className={`text-xs ${isUser ? 'text-blue-100' : 'text-gray-400'}`}
                  >
                    {timeText}
                  </Text>
                )}
                <Button
                  type="text"
                  size="small"
                  icon={<CopyOutlined />}
                  onClick={handleCopy}
                  disabled={!canCopy}
                  className={`text-xs ${isUser ? 'text-blue-100' : 'text-gray-400'}`}
                >
                  复制
                </Button>
              </Space>
            </div>

            {/* 推理步骤 */}
            {!isUser &&
              message.reasoningSteps &&
              message.reasoningSteps.length > 0 && (
                <ReasoningSteps
                  steps={message.reasoningSteps}
                  highlightQuery={highlightQuery}
                />
              )}

            {/* 消息内容 */}
            <div className={`${isUser ? 'text-white' : 'text-gray-800'}`}>
              {isUser ? (
                <Paragraph
                  className={`!mb-0 whitespace-pre-wrap ${isUser ? 'text-white' : ''}`}
                >
                  {highlightQuery
                    ? renderHighlightedText(message.content, highlightQuery)
                    : message.content}
                </Paragraph>
              ) : (
                <MarkdownRenderer
                  content={message.content}
                  className="!mb-0"
                  highlightQuery={highlightQuery}
                />
              )}
            </div>

            {/* 工具调用显示 */}
            {!isUser && message.toolCalls && message.toolCalls.length > 0 && (
              <div className="mt-3">
                <ToolCallDisplay
                  toolCalls={message.toolCalls}
                  highlightQuery={highlightQuery}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MessageItem
