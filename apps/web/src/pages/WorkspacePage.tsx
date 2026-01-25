import React, {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from 'react'
import {
  Button,
  Card,
  Col,
  Empty,
  Input,
  List,
  message,
  Row,
  Space,
  Tag,
  Typography,
  Tooltip,
} from 'antd'
import { useNavigate } from 'react-router-dom'
import {
  HistoryOutlined,
  ReloadOutlined,
  SearchOutlined,
  CopyOutlined,
  StarFilled,
  StarOutlined,
} from '@ant-design/icons'
import { getMenuLabelText, resolveMenuKey } from '../routes/menuConfig'
import {
  MENU_INDEX,
  MENU_KEY_SET,
  type MenuItem,
} from '../routes/menuIndex'
import {
  buildNavigationResults,
  getMenuItemIcon,
  getMenuMetaText,
  getNavigationMetaText,
  resolveDirectNavigationMeta,
  resolveNavigationPath,
} from '../routes/menuSearch'
import { renderHighlightedText } from '../utils/highlightText'
import { normalizeSearchText, resolveDeferredQuery } from '../utils/searchText'
import { getHealthStatusInfo } from '../utils/healthStatus'
import { clampIndex, wrapIndex } from '../utils/number'
import { copyToClipboard } from '../utils/clipboard'
import {
  MENU_STORAGE_EVENTS,
  readStoredLastRoute,
} from '../routes/navigationStorage'
import { useConversationStore } from '../stores/conversationStore'
import type { Conversation } from '../types'
import NetworkErrorAlert from '../components/ui/NetworkErrorAlert'
import useHealthMonitor from '../hooks/useHealthMonitor'
import useMenuShortcuts from '../hooks/useMenuShortcuts'

const { Title, Text } = Typography

const CORE_MENU_KEYS = [
  'chat',
  'multi-agent',
  'rag',
  'workflows-visualization',
  'monitoring-dashboard',
  'experiments-platform',
]
const SEARCH_LIMIT = 8
const RECENT_CONVERSATIONS_LIMIT = 5

const WorkspacePage: React.FC = () => {
  const navigate = useNavigate()
  const [searchValue, setSearchValue] = useState('')
  const deferredSearchValue = useDeferredValue(searchValue)
  const [searchActiveIndex, setSearchActiveIndex] = useState(0)
  const { favoriteKeys, recentKeys, setFavoriteKeys, setRecentKeys } =
    useMenuShortcuts(MENU_KEY_SET)
  const [lastRoute, setLastRoute] = useState<string | null>(null)
  const {
    conversations,
    historyLoading,
    historyError,
    refreshConversations,
    loadConversation,
    setHistoryError,
  } = useConversationStore()
  const {
    status: healthStatus,
    timestamp: healthTimestamp,
    loading: healthLoading,
    error: healthError,
    refresh: refreshHealth,
  } = useHealthMonitor({ intervalMs: 30000 })

  const refreshLastRoute = useCallback(() => {
    setLastRoute(readStoredLastRoute(MENU_KEY_SET))
  }, [])

  const refreshRecentConversations = useCallback(async () => {
    await refreshConversations({
      limit: RECENT_CONVERSATIONS_LIMIT,
      offset: 0,
    })
  }, [refreshConversations])

  useEffect(() => {
    refreshLastRoute()
    if (typeof window === 'undefined') return
    const handleStorage = () => refreshLastRoute()
    window.addEventListener(MENU_STORAGE_EVENTS.lastRoute, handleStorage)
    window.addEventListener('storage', handleStorage)
    return () => {
      window.removeEventListener(MENU_STORAGE_EVENTS.lastRoute, handleStorage)
      window.removeEventListener('storage', handleStorage)
    }
  }, [refreshLastRoute])

  useEffect(() => {
    refreshRecentConversations()
  }, [refreshRecentConversations])

  useEffect(() => {
    setSearchActiveIndex(0)
  }, [searchValue])

  const searchQuery = resolveDeferredQuery(searchValue, deferredSearchValue)
  const searchQueryTrimmed = searchQuery.trim()
  const normalizedQuery = normalizeSearchText(searchQuery)
  const hasQuery = normalizedQuery.length > 0
  const {
    path: directPath,
    targetPath: directTargetPath,
    label: directLabel,
  } = resolveDirectNavigationMeta(searchValue, MENU_KEY_SET)
  const favoriteKeySet = useMemo(() => new Set(favoriteKeys), [favoriteKeys])

  const coreItems = useMemo(
    () =>
      CORE_MENU_KEYS.map(key => MENU_INDEX.itemByKey.get(key)).filter(
        Boolean
      ) as MenuItem[],
    []
  )

  const favoriteItems = useMemo(
    () =>
      favoriteKeys
        .map(key => MENU_INDEX.itemByKey.get(key))
        .filter(Boolean) as MenuItem[],
    [favoriteKeys]
  )

  const recentItems = useMemo(
    () =>
      recentKeys
        .map(key => MENU_INDEX.itemByKey.get(key))
        .filter(Boolean) as MenuItem[],
    [recentKeys]
  )

  const recentConversations = useMemo(
    () => conversations.slice(0, RECENT_CONVERSATIONS_LIMIT),
    [conversations]
  )

  const searchResults = useMemo(
    () =>
      hasQuery
        ? buildNavigationResults(
            MENU_INDEX.menuKeys,
            searchQuery,
            SEARCH_LIMIT,
            {
              favorites: favoriteKeySet,
              recents: recentKeys,
            }
          )
        : [],
    [favoriteKeySet, hasQuery, recentKeys, searchQuery]
  )

  useEffect(() => {
    setSearchActiveIndex(index => {
      const next = clampIndex(index, searchResults.length)
      return next === index ? index : next
    })
  }, [searchResults.length])

  const lastRouteKey = lastRoute ? resolveMenuKey(lastRoute) : ''
  const lastRouteItem = lastRouteKey
    ? MENU_INDEX.itemByKey.get(lastRouteKey) ?? null
    : null
  const lastRouteLabel = lastRouteItem
    ? getMenuLabelText(lastRouteItem.label)
    : lastRoute
  const showLastRoute = Boolean(lastRoute && lastRoute !== '/workspace')

  const healthInfo = getHealthStatusInfo(healthStatus, healthError)
  const healthUpdatedAt = healthTimestamp
    ? new Date(healthTimestamp).toLocaleString()
    : ''
  const healthTooltip = healthError
    ? `${healthError}${healthUpdatedAt ? ` · 最近更新 ${healthUpdatedAt}` : ''}`
    : healthUpdatedAt
      ? `最近更新 ${healthUpdatedAt}`
      : '暂无健康数据'

  const handleOpenConversation = useCallback(
    async (conversation: Conversation) => {
      try {
        await loadConversation(conversation.id)
        navigate('/chat')
      } catch {
        message.error('加载对话失败')
      }
    },
    [loadConversation, navigate]
  )
  const toggleFavorite = useCallback(
    (menuKey: string) => {
      if (!MENU_KEY_SET.has(menuKey)) return
      setFavoriteKeys(prev => {
        return prev.includes(menuKey)
          ? prev.filter(key => key !== menuKey)
          : [menuKey, ...prev]
      })
    },
    [setFavoriteKeys]
  )
  const clearFavorites = useCallback(() => {
    setFavoriteKeys([])
  }, [setFavoriteKeys])
  const clearRecents = useCallback(() => {
    setRecentKeys([])
  }, [setRecentKeys])
  const handleCopyPath = useCallback(async (path: string) => {
    if (!path) return
    try {
      await copyToClipboard(path)
      message.success('路径已复制')
    } catch {
      message.error('复制失败')
    }
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>
            工作台
          </Title>
          <Text type="secondary">
            快速进入核心功能，查看系统状态与常用入口
          </Text>
          {showLastRoute && lastRoute && (
            <div style={{ marginTop: 12 }}>
              <Button
                type="primary"
                icon={<HistoryOutlined />}
                onClick={() => navigate(lastRoute)}
              >
                继续上次访问：{lastRouteLabel}
              </Button>
            </div>
          )}
        </div>

        <Row gutter={[16, 16]}>
          <Col xs={24} lg={14}>
            <Card title="快速搜索">
              <Input
                allowClear
                placeholder="搜索功能或直接输入 /path"
                prefix={<SearchOutlined style={{ color: '#999' }} />}
                value={searchValue}
                onChange={e => setSearchValue(e.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Escape' && searchValue) {
                    event.preventDefault()
                    setSearchValue('')
                    return
                  }
                  if (searchResults.length === 0) return
                  if (event.key === 'ArrowDown') {
                    event.preventDefault()
                    setSearchActiveIndex(index =>
                      wrapIndex(index + 1, searchResults.length)
                    )
                  }
                  if (event.key === 'ArrowUp') {
                    event.preventDefault()
                    setSearchActiveIndex(index =>
                      wrapIndex(index - 1, searchResults.length)
                    )
                  }
                }}
                onPressEnter={() => {
                  if (directPath) {
                    setSearchValue('')
                    navigate(directTargetPath)
                    return
                  }
                  const target =
                    searchResults[searchActiveIndex] || searchResults[0]
                  if (!target) return
                  setSearchValue('')
                  navigate(resolveNavigationPath(String(target.key)))
                }}
              />
              <div style={{ marginTop: 12 }}>
                {directPath && (
                  <div style={{ marginBottom: 8 }}>
                    <Button
                      size="small"
                      type="dashed"
                      onClick={() => {
                        setSearchValue('')
                        navigate(directTargetPath)
                      }}
                    >
                      {directLabel}
                    </Button>
                  </div>
                )}
                {hasQuery ? (
                  <>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {searchResults.length > 0
                        ? `已显示 ${searchResults.length} 项，↑↓ 选择后回车跳转`
                        : '未找到匹配项'}
                    </Text>
                    {searchResults.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <Space size={[8, 8]} wrap>
                          {searchResults.map((item, index) => {
                            const active = index === searchActiveIndex
                            const menuKey = String(item.key)
                            const canFavorite = MENU_KEY_SET.has(menuKey)
                            const isFavorite =
                              canFavorite && favoriteKeySet.has(menuKey)
                            const copyPath = resolveNavigationPath(menuKey)
                            return (
                              <div
                                key={`workspace-search-${menuKey}`}
                                style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: 4,
                                }}
                              >
                                <Button
                                  size="small"
                                  type={active ? 'primary' : 'text'}
                                  icon={getMenuItemIcon(item)}
                                  title={getNavigationMetaText(menuKey)}
                                  onClick={() => {
                                    setSearchValue('')
                                    navigate(resolveNavigationPath(menuKey))
                                  }}
                                  onMouseEnter={() =>
                                    setSearchActiveIndex(index)
                                  }
                                  style={{ paddingInline: 4 }}
                                >
                                  {renderHighlightedText(
                                    getMenuLabelText(item.label),
                                    searchQuery
                                  )}
                                </Button>
                                {canFavorite && (
                                  <Tooltip
                                    title={isFavorite ? '取消收藏' : '收藏'}
                                  >
                                    <Button
                                      size="small"
                                      type="text"
                                      icon={
                                        isFavorite ? (
                                          <StarFilled
                                            style={{ color: '#fadb14' }}
                                          />
                                        ) : (
                                          <StarOutlined />
                                        )
                                      }
                                      onClick={event => {
                                        event.preventDefault()
                                        event.stopPropagation()
                                        toggleFavorite(menuKey)
                                      }}
                                    />
                                  </Tooltip>
                                )}
                                <Tooltip title="复制路径">
                                  <Button
                                    size="small"
                                    type="text"
                                    icon={<CopyOutlined />}
                                    onClick={event => {
                                      event.preventDefault()
                                      event.stopPropagation()
                                      handleCopyPath(copyPath)
                                    }}
                                  />
                                </Tooltip>
                              </div>
                            )
                          })}
                        </Space>
                      </div>
                    )}
                    {searchQueryTrimmed && (
                      <div style={{ marginTop: 8 }}>
                        <Button
                          size="small"
                          type="link"
                          style={{ padding: 0 }}
                          onClick={() => {
                            const params = new URLSearchParams()
                            params.set('q', searchQueryTrimmed)
                            setSearchValue('')
                            navigate(
                              `/page-index?${params.toString()}`
                            )
                          }}
                        >
                          查看更多结果
                        </Button>
                      </div>
                    )}
                  </>
                ) : (
                  <Text type="secondary">
                    建议使用关键词或 Ctrl+K / 打开全局搜索
                  </Text>
                )}
              </div>
            </Card>
          </Col>
          <Col xs={24} lg={10}>
            <Card
              title="系统健康"
              extra={
                <Tooltip title={healthTooltip}>
                  <Tag color={healthInfo.color}>{healthInfo.label}</Tag>
                </Tooltip>
              }
            >
              <Space direction="vertical" size={8} style={{ width: '100%' }}>
                <Text type="secondary">
                  {healthUpdatedAt
                    ? `最近更新 ${healthUpdatedAt}`
                    : '暂无更新记录'}
                </Text>
                {healthError && (
                  <NetworkErrorAlert
                    error={healthError}
                    onRetry={refreshHealth}
                  />
                )}
                <Space>
                  <Button
                    size="small"
                    icon={<ReloadOutlined />}
                    onClick={refreshHealth}
                    loading={healthLoading}
                  >
                    刷新
                  </Button>
                  <Button
                    size="small"
                    type="link"
                    onClick={() => navigate('/system-health')}
                  >
                    查看详情
                  </Button>
                </Space>
              </Space>
            </Card>
          </Col>
        </Row>

        <Card title="核心入口">
          {coreItems.length > 0 ? (
            <Row gutter={[16, 16]}>
              {coreItems.map(item => (
                <Col xs={24} sm={12} md={8} key={`core-${String(item.key)}`}>
                  <Card
                    size="small"
                    hoverable
                    onClick={() =>
                      navigate(resolveNavigationPath(String(item.key)))
                    }
                  >
                    <Space>
                      {getMenuItemIcon(item)}
                      <div>
                        <Text strong>{getMenuLabelText(item.label)}</Text>
                        <div>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {resolveNavigationPath(String(item.key))}
                          </Text>
                        </div>
                      </div>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          ) : (
            <Empty description="暂无核心入口配置" />
          )}
        </Card>

        <Card
          title="最近对话"
          extra={
            <Button type="link" size="small" onClick={() => navigate('/history')}>
              查看全部
            </Button>
          }
        >
          {historyError && (
            <div className="mb-3">
              <NetworkErrorAlert
                error={historyError}
                onRetry={refreshRecentConversations}
                onDismiss={() => setHistoryError(null)}
              />
            </div>
          )}
          {recentConversations.length > 0 ? (
            <List
              loading={historyLoading}
              dataSource={recentConversations}
              rowKey={conversation => conversation.id}
              renderItem={conversation => {
                const lastMessage = conversation.lastMessage
                const updatedAt = conversation.updatedAt || conversation.createdAt
                const updatedText = updatedAt
                  ? new Date(updatedAt).toLocaleString()
                  : ''
                const lastMessageLabel =
                  lastMessage?.role === 'user' ? '用户' : '助手'
                return (
                  <List.Item
                    actions={[
                      <Button
                        key="open"
                        size="small"
                        type="link"
                        onClick={() => handleOpenConversation(conversation)}
                      >
                        继续
                      </Button>,
                    ]}
                  >
                    <List.Item.Meta
                      title={conversation.title || '对话'}
                      description={
                        lastMessage?.content ? (
                          <Text type="secondary" className="text-xs line-clamp-2">
                            {lastMessageLabel}: {lastMessage.content}
                          </Text>
                        ) : (
                          '暂无消息'
                        )
                      }
                    />
                    <div className="text-xs text-gray-400">{updatedText}</div>
                  </List.Item>
                )
              }}
            />
          ) : (
            <Empty
              description={historyLoading ? '加载中...' : '暂无对话历史'}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            >
              <Button size="small" type="primary" onClick={() => navigate('/chat')}>
                开始对话
              </Button>
            </Empty>
          )}
        </Card>

        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Card
              title="收藏"
              extra={
                <Button
                  type="link"
                  size="small"
                  onClick={clearFavorites}
                  disabled={favoriteItems.length === 0}
                >
                  清空
                </Button>
              }
            >
              {favoriteItems.length > 0 ? (
                <Space size={[8, 8]} wrap>
                  {favoriteItems.map(item => (
                    <Button
                      key={`favorite-${String(item.key)}`}
                      size="small"
                      type="text"
                      icon={getMenuItemIcon(item)}
                      title={getMenuMetaText(String(item.key))}
                      onClick={() =>
                        navigate(resolveNavigationPath(String(item.key)))
                      }
                    >
                      {getMenuLabelText(item.label)}
                    </Button>
                  ))}
                </Space>
              ) : (
                <Empty description="暂无收藏" />
              )}
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card
              title="最近访问"
              extra={
                <Button
                  type="link"
                  size="small"
                  onClick={clearRecents}
                  disabled={recentItems.length === 0}
                >
                  清空
                </Button>
              }
            >
              {recentItems.length > 0 ? (
                <Space size={[8, 8]} wrap>
                  {recentItems.map(item => (
                    <Button
                      key={`recent-${String(item.key)}`}
                      size="small"
                      type="text"
                      icon={getMenuItemIcon(item)}
                      title={getMenuMetaText(String(item.key))}
                      onClick={() =>
                        navigate(resolveNavigationPath(String(item.key)))
                      }
                    >
                      {getMenuLabelText(item.label)}
                    </Button>
                  ))}
                </Space>
              ) : (
                <Empty description="暂无最近访问" />
              )}
            </Card>
          </Col>
        </Row>
      </Space>
    </div>
  )
}

export default WorkspacePage
