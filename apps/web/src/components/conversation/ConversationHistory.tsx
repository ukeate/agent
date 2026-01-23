import React, { useCallback, useEffect, useState } from 'react'
import {
  List,
  Card,
  Typography,
  Space,
  Button,
  Empty,
  Tag,
  Popconfirm,
  message,
  Input,
  Modal,
} from 'antd'
import {
  MessageOutlined,
  DeleteOutlined,
  CalendarOutlined,
  EditOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import { Conversation } from '@/types'
import { useConversationStore } from '@/stores/conversationStore'
import NetworkErrorAlert from '../ui/NetworkErrorAlert'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { renderHighlightedText } from '@/utils/highlightText'
import {
  buildSearchIndexText,
  matchSearchTokens,
  splitSearchTokens,
} from '@/utils/searchText'

const { Text, Title } = Typography

interface ConversationHistoryProps {
  visible: boolean
  onSelectConversation: (conversation: Conversation) => void
}

const ConversationHistory: React.FC<ConversationHistoryProps> = ({
  visible,
  onSelectConversation,
}) => {
  const {
    conversations,
    conversationTotal,
    deleteConversation,
    currentConversation,
    refreshConversations,
    historyLoading,
    historyError,
    setHistoryError,
    renameConversation,
  } = useConversationStore()
  const [keyword, setKeyword] = useState('')
  const [renameTarget, setRenameTarget] = useState<Conversation | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renameLoading, setRenameLoading] = useState(false)
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [loadMoreLoading, setLoadMoreLoading] = useState(false)
  const debouncedKeyword = useDebouncedValue(keyword, 300)
  const normalizedKeyword = keyword.trim()
  const activeQuery = debouncedKeyword.trim()
  const queryTokens = splitSearchTokens(activeQuery)
  const hasQueryTokens = queryTokens.length > 0
  const isSearching = normalizedKeyword !== activeQuery
  const showQueryHint = normalizedKeyword.length > 0 || activeQuery.length > 0
  const pageSize = 20
  const totalLabel =
    conversationTotal > conversations.length
      ? `已加载 ${conversations.length}/${conversationTotal} 个对话`
      : `共 ${conversationTotal} 个对话`

  const loadConversations = useCallback(async (query?: string) => {
    const result = await refreshConversations({
      limit: pageSize,
      offset: 0,
      query,
    })
    setOffset(result.items.length)
    setHasMore(result.hasMore)
  }, [pageSize, refreshConversations])

  useEffect(() => {
    if (!visible) return
    const query = activeQuery || undefined
    setOffset(0)
    setHasMore(true)
    setLoadMoreLoading(false)
    loadConversations(query)
  }, [activeQuery, loadConversations, visible])

  const handleDeleteConversation = async (conversationId: string) => {
    try {
      await deleteConversation(conversationId, { refresh: false })
      await loadConversations(activeQuery || undefined)
    } catch {
      message.error('删除对话失败')
    }
  }

  const handleStartRename = (
    conversation: Conversation,
    e: React.MouseEvent
  ) => {
    stopPropagation(e)
    setRenameTarget(conversation)
    setRenameValue(conversation.title || '')
  }

  const handleRenameCancel = () => {
    setRenameTarget(null)
    setRenameValue('')
  }

  const handleRenameConfirm = async () => {
    if (!renameTarget) return
    const normalized = renameValue.trim()
    if (!normalized) {
      message.error('标题不能为空')
      return
    }
    if (normalized === renameTarget.title) {
      handleRenameCancel()
      return
    }
    try {
      setRenameLoading(true)
      await renameConversation(renameTarget.id, normalized)
      handleRenameCancel()
    } catch {
      message.error('更新标题失败')
    } finally {
      setRenameLoading(false)
    }
  }

  const stopPropagation = (e: React.MouseEvent) => {
    e.stopPropagation()
  }

  const handleLoadMore = async () => {
    if (historyLoading || loadMoreLoading || !hasMore) return
    try {
      setLoadMoreLoading(true)
      const query = activeQuery || undefined
      const result = await refreshConversations({
        limit: pageSize,
        offset,
        append: true,
        query,
      })
      setOffset(prev => prev + result.items.length)
      setHasMore(result.hasMore)
    } finally {
      setLoadMoreLoading(false)
    }
  }

  if (!visible) return null

  return (
    <div className="p-4">
      <div className="mb-4">
          <Title level={4} className="!mb-2">
            对话历史
          </Title>
        {historyError && (
          <div className="mb-3">
            <NetworkErrorAlert
              error={historyError}
              onRetry={() => loadConversations(debouncedKeyword.trim() || undefined)}
              onDismiss={() => setHistoryError(null)}
            />
          </div>
        )}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Text type="secondary">{totalLabel}</Text>
          <Input
            allowClear
            value={keyword}
            onChange={e => setKeyword(e.target.value)}
            placeholder="搜索对话标题或最新消息"
            className="w-full sm:w-64"
          />
        </div>
        {showQueryHint && (
          <Text type="secondary" className="text-xs">
            {isSearching ? '搜索中...' : `匹配 ${conversationTotal} 个对话`}
          </Text>
        )}
      </div>

      {conversations.length === 0 && !historyLoading ? (
        <div className="space-y-3">
          <Empty
            description={activeQuery ? '未找到匹配的对话' : '暂无对话历史'}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
          {hasMore && (
            <div className="flex justify-center">
              <Button
                onClick={handleLoadMore}
                loading={loadMoreLoading}
                disabled={historyLoading}
              >
                加载更多
              </Button>
            </div>
          )}
        </div>
      ) : (
        <List
          loading={historyLoading}
          dataSource={conversations}
          rowKey={conversation => conversation.id}
          loadMore={
            hasMore ? (
              <div className="py-3 text-center">
                <Button
                  onClick={handleLoadMore}
                  loading={loadMoreLoading}
                  disabled={historyLoading}
                >
                  加载更多
                </Button>
              </div>
            ) : null
          }
          renderItem={conversation => {
            const lastMessage =
              conversation.lastMessage ||
              conversation.messages[conversation.messages.length - 1]
            const lastMessageLabel =
              lastMessage?.role === 'user' ? '用户' : '助手'
            const statusLabel =
              conversation.status === 'active'
                ? '进行中'
                : conversation.status === 'closed'
                  ? '已关闭'
                  : conversation.status
            const matchedInTitle = hasQueryTokens
              ? matchSearchTokens(
                  buildSearchIndexText(conversation.title || ''),
                  queryTokens
                )
              : false
            const matchedInLastMessage = hasQueryTokens
              ? matchSearchTokens(
                  buildSearchIndexText(lastMessage?.content || ''),
                  queryTokens
                )
              : false
            const isHistoryHit =
              hasQueryTokens && !matchedInTitle && !matchedInLastMessage
            return (
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
                      key="rename"
                      type="text"
                      size="small"
                      icon={<EditOutlined />}
                      onClick={e => handleStartRename(conversation, e)}
                    />,
                    <Popconfirm
                      key="delete"
                      title="确认删除该对话？"
                      okText="删除"
                      cancelText="取消"
                      onConfirm={() => handleDeleteConversation(conversation.id)}
                    >
                      <Button
                        type="text"
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={stopPropagation}
                      />
                    </Popconfirm>,
                  ]}
                >
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Text strong className="text-sm truncate">
                        {renderHighlightedText(
                          conversation.title || '',
                          activeQuery
                        )}
                      </Text>
                      <Space size="small">
                        {statusLabel && (
                          <Tag
                            color={
                              conversation.status === 'active'
                                ? 'green'
                                : 'default'
                            }
                          >
                            {statusLabel}
                          </Tag>
                        )}
                        {isHistoryHit && (
                          <Tag color="geekblue">历史命中</Tag>
                        )}
                        <Tag color="blue">
                          {(conversation.messageCount ??
                            conversation.messages.length) ||
                            0}{' '}
                          条消息
                        </Tag>
                      </Space>
                    </div>

                    {lastMessage && (
                      <Text type="secondary" className="text-xs line-clamp-2">
                        {lastMessageLabel}:{' '}
                        {renderHighlightedText(
                          lastMessage.content || '',
                          activeQuery
                        )}
                      </Text>
                    )}

                    <div className="flex items-center justify-between">
                      <Space size="small">
                        <MessageOutlined className="text-gray-400" />
                        <Text type="secondary" className="text-xs">
                          {(conversation.userMessageCount ??
                            conversation.messages.filter(
                              m => m.role === 'user'
                            ).length) ||
                            0}{' '}
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
            )
          }}
        />
      )}

      <Modal
        title="重命名对话"
        open={!!renameTarget}
        onOk={handleRenameConfirm}
        onCancel={handleRenameCancel}
        confirmLoading={renameLoading}
        okText="保存"
        cancelText="取消"
        destroyOnClose
      >
        <Input
          value={renameValue}
          onChange={e => setRenameValue(e.target.value)}
          placeholder="请输入对话标题"
          maxLength={120}
          onPressEnter={handleRenameConfirm}
        />
      </Modal>
    </div>
  )
}

export default ConversationHistory
