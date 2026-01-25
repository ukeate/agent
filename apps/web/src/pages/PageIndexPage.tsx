import React, { useEffect, useMemo, useState } from 'react'
import {
  Button,
  Input,
  List,
  Radio,
  Space,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  CopyOutlined,
  ArrowRightOutlined,
  SearchOutlined,
  StarFilled,
  StarOutlined,
} from '@ant-design/icons'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ROUTE_PATHS, ROUTE_PATH_SET } from '../routes/routeCatalog'
import {
  MENU_INDEX,
  MENU_KEY_SET,
  getMenuParentLabelPath,
} from '../routes/menuIndex'
import {
  getMenuLabelText,
  resolveMenuKey,
  resolveMenuPath,
} from '../routes/menuConfig'
import { renderHighlightedText } from '../utils/highlightText'
import {
  buildSearchIndexText,
  matchSearchTokens,
  splitSearchTokens,
} from '../utils/searchText'
import { copyToClipboard } from '../utils/clipboard'
import { useMenuShortcuts } from '../hooks/useMenuShortcuts'

const { Title, Text } = Typography

type FilterMode =
  | 'all'
  | 'menu'
  | 'missing'
  | 'unregistered'
  | 'favorites'
  | 'recents'

const FILTER_MODES: FilterMode[] = [
  'all',
  'menu',
  'missing',
  'unregistered',
  'favorites',
  'recents',
]

const resolveFilterMode = (value: string | null): FilterMode => {
  if (value && FILTER_MODES.includes(value as FilterMode)) {
    return value as FilterMode
  }
  return 'all'
}

type RouteItem = {
  path: string
  menuKey: string
  inMenu: boolean
  registered: boolean
  label: string
  meta: string
  searchText: string
}

const PageIndexPage: React.FC = () => {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const initialQuery = searchParams.get('q') ?? ''
  const [query, setQuery] = useState(initialQuery)
  const [filterMode, setFilterMode] = useState<FilterMode>(() =>
    resolveFilterMode(searchParams.get('mode'))
  )
  const queryTokens = useMemo(() => splitSearchTokens(query), [query])
  const { favoriteKeys, recentKeys, setFavoriteKeys, setRecentKeys } =
    useMenuShortcuts(MENU_KEY_SET)
  const favoriteKeySet = useMemo(
    () => new Set(favoriteKeys),
    [favoriteKeys]
  )
  const recentKeySet = useMemo(() => new Set(recentKeys), [recentKeys])

  useEffect(() => {
    const next = searchParams.get('q') ?? ''
    setQuery(prev => (prev === next ? prev : next))
    const nextMode = resolveFilterMode(searchParams.get('mode'))
    setFilterMode(prev => (prev === nextMode ? prev : nextMode))
  }, [searchParams])

  const updateQuery = (value: string) => {
    setQuery(value)
    const params = new URLSearchParams(searchParams)
    const trimmed = value.trim()
    if (trimmed) {
      params.set('q', trimmed)
    } else {
      params.delete('q')
    }
    setSearchParams(params, { replace: true })
  }

  const updateFilterMode = (next: FilterMode) => {
    setFilterMode(next)
    const params = new URLSearchParams(searchParams)
    if (next && next !== 'all') {
      params.set('mode', next)
    } else {
      params.delete('mode')
    }
    setSearchParams(params, { replace: true })
  }

  const routeItems = useMemo<RouteItem[]>(() => {
    const menuPaths = MENU_INDEX.menuKeys.map(key => resolveMenuPath(key))
    const uniquePaths = new Set([...ROUTE_PATHS, ...menuPaths])
    const items = Array.from(uniquePaths)
      .filter(path => path && path !== '/')
      .map(path => {
        const menuKey = resolveMenuKey(path)
        const inMenu = MENU_KEY_SET.has(menuKey)
        const registered = ROUTE_PATH_SET.has(path)
        const menuItem = inMenu ? MENU_INDEX.itemByKey.get(menuKey) : null
        const label = menuItem ? getMenuLabelText(menuItem.label) : ''
        const parentPath = inMenu ? getMenuParentLabelPath(menuKey) : ''
        const meta = parentPath ? `${parentPath} / ${label}` : label
        const searchText = buildSearchIndexText(
          `${path} ${menuKey} ${label} ${parentPath}`
        )
        return {
          path,
          menuKey,
          inMenu,
          registered,
          label,
          meta,
          searchText,
        }
      })
    return items.sort((a, b) => {
      if (a.inMenu !== b.inMenu) return a.inMenu ? -1 : 1
      return a.path.localeCompare(b.path)
    })
  }, [])

  const totalCount = routeItems.length
  const menuCount = useMemo(
    () => routeItems.filter(item => item.inMenu).length,
    [routeItems]
  )
  const missingCount = totalCount - menuCount
  const unregisteredCount = useMemo(
    () => routeItems.filter(item => item.inMenu && !item.registered).length,
    [routeItems]
  )
  const favoriteCount = favoriteKeys.length
  const recentCount = recentKeys.length

  const emptyText = useMemo(() => {
    if (query.trim()) return '未找到匹配页面'
    if (filterMode === 'favorites') return '暂无收藏页面'
    if (filterMode === 'recents') return '暂无最近访问'
    if (filterMode === 'missing') return '暂无未收录页面'
    if (filterMode === 'unregistered') return '暂无缺少路由'
    if (filterMode === 'menu') return '暂无已收录页面'
    return '暂无页面数据'
  }, [filterMode, query])

  const filteredItems = useMemo(() => {
    const items = routeItems.filter(item => {
      if (filterMode === 'menu' && !item.inMenu) return false
      if (filterMode === 'missing' && item.inMenu) return false
      if (
        filterMode === 'unregistered' &&
        (item.registered || !item.inMenu)
      )
        return false
      if (filterMode === 'favorites' && !favoriteKeySet.has(item.menuKey))
        return false
      if (filterMode === 'recents' && !recentKeySet.has(item.menuKey))
        return false
      if (queryTokens.length === 0) return true
      return matchSearchTokens(item.searchText, queryTokens)
    })
    if (filterMode === 'favorites') {
      const order = new Map(
        favoriteKeys.map((key, index) => [key, index])
      )
      return items.sort((a, b) => {
        const aIndex = order.get(a.menuKey)
        const bIndex = order.get(b.menuKey)
        if (aIndex !== undefined && bIndex !== undefined) {
          return aIndex - bIndex || a.path.localeCompare(b.path)
        }
        if (aIndex !== undefined) return -1
        if (bIndex !== undefined) return 1
        return a.path.localeCompare(b.path)
      })
    }
    if (filterMode === 'recents') {
      const order = new Map(recentKeys.map((key, index) => [key, index]))
      return items.sort((a, b) => {
        const aIndex = order.get(a.menuKey)
        const bIndex = order.get(b.menuKey)
        if (aIndex !== undefined && bIndex !== undefined) {
          return aIndex - bIndex || a.path.localeCompare(b.path)
        }
        if (aIndex !== undefined) return -1
        if (bIndex !== undefined) return 1
        return a.path.localeCompare(b.path)
      })
    }
    return items
  }, [
    favoriteKeySet,
    favoriteKeys,
    filterMode,
    queryTokens,
    recentKeySet,
    recentKeys,
    routeItems,
  ])
  const firstNavigable = useMemo(
    () => filteredItems.find(item => item.registered || item.inMenu) ?? null,
    [filteredItems]
  )

  const toggleFavorite = (menuKey: string) => {
    if (!MENU_KEY_SET.has(menuKey)) return
    setFavoriteKeys(prev => {
      if (prev.includes(menuKey)) {
        return prev.filter(key => key !== menuKey)
      }
      return [menuKey, ...prev]
    })
  }

  const handleCopyPath = async (path: string) => {
    try {
      await copyToClipboard(path)
      message.success('路径已复制')
    } catch {
      message.error('复制失败')
    }
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex flex-col gap-1">
        <Title level={4} className="!mb-0">
          页面索引
        </Title>
        <Text type="secondary">
          统一查看前端路由，快速跳转并定位未收录的页面入口。
        </Text>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <Input
          allowClear
          value={query}
          onChange={e => updateQuery(e.target.value)}
          onPressEnter={() => {
            if (!firstNavigable) return
            navigate(firstNavigable.path)
          }}
          prefix={<SearchOutlined className="text-gray-400" />}
          placeholder="搜索路由、菜单名称或路径"
          className="w-full sm:w-80"
        />
        <div className="flex flex-wrap items-center gap-2">
          <Radio.Group
            value={filterMode}
            onChange={e => updateFilterMode(e.target.value)}
            optionType="button"
            buttonStyle="solid"
          >
            <Radio.Button value="all">全部</Radio.Button>
            <Radio.Button value="menu">已收录</Radio.Button>
            <Radio.Button value="missing">未收录</Radio.Button>
            <Radio.Button value="unregistered">缺少路由</Radio.Button>
            <Radio.Button value="favorites">收藏</Radio.Button>
            <Radio.Button value="recents">最近</Radio.Button>
          </Radio.Group>
          {filterMode === 'favorites' && favoriteCount > 0 && (
            <Button
              size="small"
              type="link"
              onClick={() => setFavoriteKeys([])}
            >
              清空收藏
            </Button>
          )}
          {filterMode === 'recents' && recentCount > 0 && (
            <Button
              size="small"
              type="link"
              onClick={() => setRecentKeys([])}
            >
              清空最近
            </Button>
          )}
        </div>
      </div>

      <Space size="small" wrap>
        <Tag color="blue">总计 {totalCount}</Tag>
        <Tag color="green">已收录 {menuCount}</Tag>
        <Tag color="gold">未收录 {missingCount}</Tag>
        <Tag color="red">缺少路由 {unregisteredCount}</Tag>
        <Tag color="volcano">收藏 {favoriteCount}</Tag>
        <Tag color="geekblue">最近 {recentCount}</Tag>
        {query.trim() && (
          <Tag color="purple">匹配 {filteredItems.length}</Tag>
        )}
      </Space>

      <List
        dataSource={filteredItems}
        rowKey={item => item.path}
        pagination={{ pageSize: 24, showSizeChanger: true }}
        locale={{ emptyText }}
        renderItem={item => {
          const isFavorite = favoriteKeySet.has(item.menuKey)
          const isRecent = recentKeySet.has(item.menuKey)
          const canFavorite = item.inMenu
          return (
            <List.Item
              actions={[
                <Button
                  key="enter"
                  type="text"
                  icon={<ArrowRightOutlined />}
                  disabled={!item.registered && !item.inMenu}
                  onClick={() => navigate(item.path)}
                >
                  进入
                </Button>,
                canFavorite ? (
                  <Button
                    key="favorite"
                    type="text"
                    icon={
                      isFavorite ? (
                        <StarFilled style={{ color: '#fadb14' }} />
                      ) : (
                        <StarOutlined />
                      )
                    }
                    onClick={() => toggleFavorite(item.menuKey)}
                  >
                    {isFavorite ? '取消收藏' : '收藏'}
                  </Button>
                ) : null,
                <Button
                  key="copy"
                  type="text"
                  icon={<CopyOutlined />}
                  onClick={() => handleCopyPath(item.path)}
                >
                  复制路径
                </Button>,
              ]}
            >
              <div className="flex flex-col gap-1">
                <Space size="small" wrap>
                  <Text strong>
                    {renderHighlightedText(item.label || item.path, query)}
                  </Text>
                  {item.inMenu ? (
                    item.registered ? (
                      <Tag color="green">已收录</Tag>
                    ) : (
                      <Tag color="red">缺少路由</Tag>
                    )
                  ) : (
                    <Tag color="gold">未收录</Tag>
                  )}
                  {isFavorite && <Tag color="volcano">收藏</Tag>}
                  {isRecent && <Tag color="geekblue">最近</Tag>}
                </Space>
                <Text type="secondary" className="text-xs">
                  {renderHighlightedText(item.path, query)}
                </Text>
                {item.meta && (
                  <Text type="secondary" className="text-xs">
                    {renderHighlightedText(item.meta, query)}
                  </Text>
                )}
              </div>
            </List.Item>
          )
        }}
      />
    </div>
  )
}

export default PageIndexPage
