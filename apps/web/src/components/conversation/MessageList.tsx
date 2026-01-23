import React, { useDeferredValue, useEffect, useMemo, useState } from 'react'
import {
  Empty,
  Button,
  Space,
  Spin,
  Popconfirm,
  Input,
  message as antdMessage,
  Tooltip,
} from 'antd'
import {
  DeleteOutlined,
  PlusOutlined,
  DownOutlined,
  ReloadOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import MessageItem from './MessageItem'
import { Message } from '@/types'
import { useSmartAutoScroll } from '../../hooks/useSmartAutoScroll'
import {
  matchSearchTokens,
  normalizeSearchText,
  splitSearchTokens,
} from '@/utils/searchText'
import { copyToClipboard } from '@/utils/clipboard'
import {
  buildConversationExportText,
  safeStringify,
} from '@/utils/conversationExport'

const buildMessageSearchText = (message: Message) => {
  const parts = [message.content]
  if (message.reasoningSteps?.length) {
    parts.push(message.reasoningSteps.map(step => step.content).join(' '))
  }
  if (message.toolCalls?.length) {
    parts.push(
      message.toolCalls
        .map(toolCall =>
          [
            toolCall.name,
            safeStringify(toolCall.args),
            safeStringify(toolCall.result),
          ]
            .filter(Boolean)
            .join(' ')
        )
        .join(' ')
    )
  }
  return normalizeSearchText(parts.join(' '))
}


interface MessageListProps {
  messages: Message[]
  loading: boolean
  onClearHistory?: () => void | Promise<void>
  onStartNewConversation?: () => void | Promise<void>
  onRetrySend?: () => void
  conversationId?: string
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  loading,
  onClearHistory,
  onStartNewConversation,
  onRetrySend,
  conversationId,
}) => {
  const [keyword, setKeyword] = useState('')
  const deferredKeyword = useDeferredValue(keyword)
  useEffect(() => {
    setKeyword('')
  }, [conversationId])
  const searchTokens = useMemo(
    () => splitSearchTokens(deferredKeyword),
    [deferredKeyword]
  )
  const hasKeyword = searchTokens.length > 0
  const messageSearchIndex = useMemo(() => {
    const index = new Map<string, string>()
    messages.forEach(message => {
      index.set(message.id, buildMessageSearchText(message))
    })
    return index
  }, [messages])
  const filteredMessages = useMemo(() => {
    if (!hasKeyword) return messages
    return messages.filter(message =>
      matchSearchTokens(messageSearchIndex.get(message.id) || '', searchTokens)
    )
  }, [hasKeyword, messageSearchIndex, messages, searchTokens])
  const displayMessages = hasKeyword ? filteredMessages : messages
  const { containerRef, scrollToBottom, isAtBottom } = useSmartAutoScroll({
    messages: displayMessages,
    resetKey: conversationId,
    enabled: !hasKeyword,
  })
  const showScrollToBottom = !isAtBottom && displayMessages.length > 0
  const hasMessages = messages.length > 0
  const showDeleteAction = hasMessages && !!onClearHistory
  const showRetryAction = hasMessages && !!onRetrySend
  const deleteButton = (
    <Button
      type="text"
      size="small"
      icon={<DeleteOutlined />}
      onClick={showDeleteAction ? undefined : onClearHistory}
      disabled={!onClearHistory || loading}
      className="text-gray-500"
    >
      删除对话
    </Button>
  )
  const deleteAction = showDeleteAction ? (
    <Popconfirm
      title="确认删除当前对话？会话将从历史中移除"
      okText="删除"
      cancelText="取消"
      onConfirm={onClearHistory}
    >
      {deleteButton}
    </Popconfirm>
  ) : (
    deleteButton
  )

  const newConversationButton = (
    <Button
      type="text"
      size="small"
      icon={<PlusOutlined />}
      onClick={onStartNewConversation}
      disabled={!onStartNewConversation || loading}
      className="text-gray-500"
    >
      新建对话
    </Button>
  )
  const retryButton = showRetryAction ? (
    <Button
      type="text"
      size="small"
      icon={<ReloadOutlined />}
      onClick={onRetrySend}
      disabled={!onRetrySend || loading}
      className="text-gray-500"
    >
      重新生成
    </Button>
  ) : null
  const copyLabel = hasKeyword ? '复制当前列表' : '复制对话'
  const handleCopyConversation = async () => {
    if (displayMessages.length === 0) return
    const exportText = buildConversationExportText(displayMessages).trim()
    if (!exportText) {
      antdMessage.warning('暂无可复制内容')
      return
    }
    try {
      await copyToClipboard(exportText)
      antdMessage.success('对话已复制')
    } catch {
      antdMessage.error('复制失败')
    }
  }

  if (messages.length === 0 && !loading) {
    return (
      <div className="flex-1 flex flex-col">
        {/* 始终显示会话操作 */}
        <div className="border-b border-gray-100 p-3">
          <Space>
            {newConversationButton}
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
      {/* 始终显示会话操作 */}
      <div className="border-b border-gray-100 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Space>
            {newConversationButton}
            {retryButton}
            {showDeleteAction && deleteAction}
            {messages.length > 0 && (
              <span className="text-gray-400 text-xs">
                共 {messages.length} 条消息
              </span>
            )}
            {hasKeyword && (
              <span className="text-gray-400 text-xs">
                匹配 {displayMessages.length} 条
              </span>
            )}
          </Space>
          <Space size="small">
            <Input
              allowClear
              value={keyword}
              onChange={e => setKeyword(e.target.value)}
              placeholder="搜索当前对话"
              size="small"
              disabled={!hasMessages}
              className="w-48"
            />
            <Tooltip title={copyLabel}>
              <Button
                type="text"
                size="small"
                icon={<CopyOutlined />}
                onClick={handleCopyConversation}
                disabled={displayMessages.length === 0}
                className="text-gray-500"
              >
                {copyLabel}
              </Button>
            </Tooltip>
          </Space>
        </div>
      </div>

      <div
        className="flex-1 overflow-y-auto custom-scrollbar relative"
        ref={containerRef}
      >
        <div className="p-4 space-y-4">
          {displayMessages.length === 0 && !loading ? (
            <div className="py-8">
              <Empty
                description={hasKeyword ? '未找到匹配消息' : '暂无消息'}
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
              {hasKeyword && (
                <div className="mt-3 flex justify-center">
                  <Button size="small" onClick={() => setKeyword('')}>
                    清除筛选
                  </Button>
                </div>
              )}
            </div>
          ) : (
            displayMessages.map(message => (
              <MessageItem
                key={message.id}
                message={message}
                highlightQuery={hasKeyword ? deferredKeyword : undefined}
              />
            ))
          )}

          {loading && (
            <div className="flex justify-center py-4">
              <Spin size="small" />
              <span className="ml-2 text-gray-500">智能体正在思考...</span>
            </div>
          )}
        </div>
        {showScrollToBottom && (
          <div className="absolute bottom-4 left-0 right-0 flex justify-center pointer-events-none">
            <Button
              size="small"
              type="primary"
              icon={<DownOutlined />}
              onClick={scrollToBottom}
              className="pointer-events-auto shadow-sm"
            >
              回到底部
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default MessageList
