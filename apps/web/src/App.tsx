import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  Suspense,
  useDeferredValue,
  useCallback,
} from 'react'
import {
  Layout,
  Menu,
  Button,
  Breadcrumb,
  Drawer,
  Modal,
  Typography,
  Space,
  Avatar,
  Spin,
  Tag,
  Tooltip,
  Input,
  message,
  type InputRef,
} from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import AppRoutes from './routes/AppRoutes'
import {
  MENU_ITEMS,
  resolveMenuKey,
  resolveMenuPath,
  getMenuLabelText,
} from './routes/menuConfig'
import {
  MENU_INDEX,
  MENU_KEY_SET,
  MENU_SUBMENU_KEY_SET,
  getMenuParentLabelPath,
  type MenuItem,
} from './routes/menuIndex'
import {
  buildNavigationResults,
  collectMenuKeys,
  collectSubmenuKeys,
  filterMenuItems,
  countMenuMatches,
  countRouteMatches,
  getMenuMetaText,
  getNavigationMetaText,
  getMenuItemIcon,
  resolveDirectNavigationMeta,
  resolveNavigationPath,
} from './routes/menuSearch'
import { renderHighlightedText } from './utils/highlightText'
import { PALETTE_OPEN_EVENT } from './utils/palette'
import { copyToClipboard } from './utils/clipboard'
import {
  readStoredMenuCollapsed,
  readStoredMenuOpenKeys,
  writeStoredLastRoute,
  writeStoredMenuCollapsed,
  writeStoredMenuOpenKeys,
} from './routes/navigationStorage'
import {
  normalizeSearchText,
  resolveDeferredQuery,
} from './utils/searchText'
import { getHealthStatusInfo } from './utils/healthStatus'
import { clampIndex, wrapIndex } from './utils/number'
import { isSameStringArray } from './utils/array'
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  RobotOutlined,
  SearchOutlined,
  ReloadOutlined,
  QuestionCircleOutlined,
  StarFilled,
  StarOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import {
  healthService,
  type HealthAlert,
  type HealthCheckResult,
  type SystemMetrics,
} from './services/healthService'
import HealthDetailModal from './components/health/HealthDetailModal'
import ShortcutHelpModal from './components/ui/ShortcutHelpModal'
import useHealthMonitor from './hooks/useHealthMonitor'
import useMenuShortcuts from './hooks/useMenuShortcuts'

const { Header, Sider, Content } = Layout
const { Title, Text } = Typography

const MENU_RECENTS_LIMIT = 6
const MENU_SEARCH_RESULTS_LIMIT = 8

const buildPageIndexPath = (query: string) => {
  const trimmed = query.trim()
  const params = new URLSearchParams()
  if (trimmed) params.set('q', trimmed)
  const suffix = params.toString()
  return `/page-index${suffix ? `?${suffix}` : ''}`
}

const App: React.FC = () => {
  const [manualCollapsed, setManualCollapsed] = useState(() =>
    readStoredMenuCollapsed()
  )
  const [collapsed, setCollapsed] = useState(manualCollapsed)
  const [siderBroken, setSiderBroken] = useState(false)
  const [menuFilter, setMenuFilter] = useState('')
  const deferredMenuFilter = useDeferredValue(menuFilter)
  const [menuSearchActiveIndex, setMenuSearchActiveIndex] = useState(0)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [paletteQuery, setPaletteQuery] = useState('')
  const deferredPaletteQuery = useDeferredValue(paletteQuery)
  const [paletteActiveIndex, setPaletteActiveIndex] = useState(0)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [shortcutOpen, setShortcutOpen] = useState(false)
  const [healthDetailOpen, setHealthDetailOpen] = useState(false)
  const [healthDetailLoading, setHealthDetailLoading] = useState(false)
  const [healthDetailError, setHealthDetailError] = useState<string | null>(null)
  const [healthDetail, setHealthDetail] = useState<HealthCheckResult | null>(
    null
  )
  const [healthMetrics, setHealthMetrics] = useState<SystemMetrics | null>(null)
  const [healthAlerts, setHealthAlerts] = useState<HealthAlert[]>([])
  const [healthAlertsTotal, setHealthAlertsTotal] = useState(0)
  const [openKeys, setOpenKeys] = useState<string[]>(() =>
    readStoredMenuOpenKeys(MENU_SUBMENU_KEY_SET)
  )
  const { favoriteKeys, recentKeys, setFavoriteKeys, setRecentKeys } =
    useMenuShortcuts(MENU_KEY_SET)
  const searchInputRef = useRef<InputRef>(null)
  const paletteInputRef = useRef<InputRef>(null)
  const paletteListRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const location = useLocation()
  const hideSider = location.pathname === '/multi-step-reasoning'
  const menuFilterQuery = resolveDeferredQuery(menuFilter, deferredMenuFilter)
  const paletteQueryText = resolveDeferredQuery(
    paletteQuery,
    deferredPaletteQuery
  )
  const {
    path: directMenuPath,
    targetPath: directMenuTargetPath,
    label: directMenuLabel,
  } = resolveDirectNavigationMeta(menuFilter, MENU_KEY_SET)
  const {
    path: directPalettePath,
    menuKey: directPaletteKey,
    targetPath: directPaletteTargetPath,
    isRegistered: directPaletteRegistered,
    known: directPaletteKnown,
    label: directPaletteEntryLabel,
  } = resolveDirectNavigationMeta(paletteQuery, MENU_KEY_SET)
  const directPaletteEntryKey = directPalettePath
    ? directPaletteRegistered
      ? directPaletteKey
      : directPaletteTargetPath
    : ''
  const {
    status: healthStatus,
    timestamp: healthTimestamp,
    loading: healthLoading,
    error: healthError,
    refresh: refreshHealthSummary,
    setSnapshot: setHealthSnapshot,
  } = useHealthMonitor({ intervalMs: 30000 })
  const loadHealthDetail = useCallback(async () => {
    if (healthDetailLoading) return
    setHealthDetailLoading(true)
    setHealthDetailError(null)
    try {
      const [detailResult, metricsResult, alertsResult] = await Promise.allSettled([
        healthService.getDetailedHealth(),
        healthService.getSystemMetrics(),
        healthService.getHealthAlerts({ limit: 20 }),
      ])
      let nextError = ''
      if (detailResult.status === 'fulfilled') {
        const detail = detailResult.value
        setHealthDetail(detail)
        setHealthSnapshot(detail)
        const detailError =
          typeof detail?.error === 'string' ? detail.error.trim() : ''
        if (detailError) {
          nextError = nextError
            ? `${nextError}，详细健康信息异常: ${detailError}`
            : `详细健康信息异常: ${detailError}`
        }
      } else {
        setHealthDetail(null)
        nextError = '获取详细健康信息失败'
      }
      if (metricsResult.status === 'fulfilled') {
        const metrics = metricsResult.value
        setHealthMetrics(metrics)
        const metricsError =
          typeof metrics?.error === 'string' ? metrics.error.trim() : ''
        if (metricsError) {
          nextError = nextError
            ? `${nextError}，系统指标获取失败: ${metricsError}`
            : `系统指标获取失败: ${metricsError}`
        }
      } else {
        setHealthMetrics(null)
        nextError = nextError
          ? `${nextError}，系统指标获取失败`
          : '获取系统指标失败'
      }
      if (alertsResult.status === 'fulfilled') {
        const payload = alertsResult.value
        const alerts = Array.isArray(payload?.alerts) ? payload.alerts : []
        setHealthAlerts(alerts)
        setHealthAlertsTotal(
          typeof payload?.total_alerts === 'number'
            ? payload.total_alerts
            : alerts.length
        )
        const alertsError =
          typeof payload?.error === 'string' ? payload.error.trim() : ''
        if (alertsError) {
          nextError = nextError
            ? `${nextError}，告警获取失败: ${alertsError}`
            : `告警获取失败: ${alertsError}`
        }
      } else {
        setHealthAlerts([])
        setHealthAlertsTotal(0)
        nextError = nextError ? `${nextError}，告警获取失败` : '告警获取失败'
      }
      setHealthDetailError(nextError || null)
    } finally {
      setHealthDetailLoading(false)
    }
  }, [healthDetailLoading, setHealthSnapshot])
  const refreshHealth = useCallback(async () => {
    await refreshHealthSummary()
    if (healthDetailOpen) {
      loadHealthDetail()
    }
  }, [healthDetailOpen, loadHealthDetail, refreshHealthSummary])

  const openHealthDetail = useCallback(() => {
    setHealthDetailOpen(true)
  }, [])
  const closeHealthDetail = useCallback(() => {
    setHealthDetailOpen(false)
  }, [])

  const closeShortcut = useCallback(() => {
    setShortcutOpen(false)
  }, [])

  const closePalette = useCallback(() => {
    setPaletteOpen(false)
    setPaletteQuery('')
    setPaletteActiveIndex(0)
  }, [])

  const openPalette = useCallback(() => {
    closeShortcut()
    setPaletteOpen(true)
    setPaletteQuery('')
    setPaletteActiveIndex(0)
    setDrawerOpen(false)
  }, [closeShortcut])

  const openDrawer = useCallback(() => {
    setDrawerOpen(true)
    closePalette()
    closeShortcut()
  }, [closePalette, closeShortcut])

  const openShortcut = useCallback(() => {
    closePalette()
    setDrawerOpen(false)
    setShortcutOpen(true)
  }, [closePalette])

  useEffect(() => {
    writeStoredMenuCollapsed(manualCollapsed)
  }, [manualCollapsed])

  useEffect(() => {
    writeStoredMenuOpenKeys(openKeys)
  }, [openKeys])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const syncOpenKeys = () => {
      const nextOpenKeys = readStoredMenuOpenKeys(MENU_SUBMENU_KEY_SET)
      setOpenKeys(prev =>
        isSameStringArray(prev, nextOpenKeys) ? prev : nextOpenKeys
      )
    }
    syncOpenKeys()
    const handleStorage = () => syncOpenKeys()
    window.addEventListener('storage', handleStorage)
    return () => {
      window.removeEventListener('storage', handleStorage)
    }
  }, [])

  useEffect(() => {
    setMenuSearchActiveIndex(0)
  }, [menuFilter])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const content = document.getElementById('app-content')
    if (content) content.scrollTop = 0
  }, [location.pathname])

  useEffect(() => {
    if (!healthDetailOpen) return
    loadHealthDetail()
  }, [healthDetailOpen, loadHealthDetail])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const isEditableTarget = Boolean(
        target &&
          (target.isContentEditable ||
            target.tagName === 'INPUT' ||
            target.tagName === 'TEXTAREA' ||
            target.tagName === 'SELECT')
      )
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        if (isEditableTarget) return
        event.preventDefault()
        openPalette()
        return
      }
      if (!isEditableTarget && event.key === '/' && !event.shiftKey) {
        event.preventDefault()
        openPalette()
        return
      }
      if (
        !isEditableTarget &&
        (event.key === '?' || (event.key === '/' && event.shiftKey))
      ) {
        event.preventDefault()
        openShortcut()
        return
      }
      if (event.key === 'Escape') {
        if (shortcutOpen) {
          event.preventDefault()
          closeShortcut()
          return
        }
        if (paletteOpen) {
          event.preventDefault()
          closePalette()
          return
        }
        if (drawerOpen) {
          event.preventDefault()
          setDrawerOpen(false)
          return
        }
        if (!isEditableTarget && menuFilter) setMenuFilter('')
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [
    drawerOpen,
    menuFilter,
    paletteOpen,
    shortcutOpen,
    closePalette,
    closeShortcut,
    openPalette,
    openShortcut,
  ])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const handleOpenPalette = () => openPalette()
    window.addEventListener(PALETTE_OPEN_EVENT, handleOpenPalette)
    return () => {
      window.removeEventListener(PALETTE_OPEN_EVENT, handleOpenPalette)
    }
  }, [openPalette])

  useEffect(() => {
    if (!paletteOpen) return
    requestAnimationFrame(() => {
      paletteInputRef.current?.focus()
      paletteInputRef.current?.select()
    })
  }, [paletteOpen])

  useEffect(() => {
    if (!drawerOpen) return
    requestAnimationFrame(() => {
      searchInputRef.current?.focus()
      searchInputRef.current?.select()
    })
  }, [drawerOpen])

  useEffect(() => {
    if (!siderBroken && !hideSider) setDrawerOpen(false)
  }, [hideSider, siderBroken])

  const getSelectedKey = (menuKeys?: Set<string>) => {
    const resolvedKey = resolveMenuKey(location.pathname)
    if (menuKeys?.has(resolvedKey)) return resolvedKey
    return ''
  }
  const getMenuTitleText = (menuKey: string, label: string) => {
    const parentPath = getMenuParentLabelPath(menuKey)
    return parentPath ? `${parentPath} / ${label}` : label
  }

  const menuKeys = MENU_INDEX.menuKeys
  const menuKeySet = MENU_KEY_SET
  const favoriteKeySet = useMemo(() => new Set(favoriteKeys), [favoriteKeys])
  const favoriteMenuItems = favoriteKeys
    .map(key => MENU_INDEX.itemByKey.get(key))
    .filter(Boolean) as MenuItem[]
  const recentMenuItems = recentKeys
    .map(key => MENU_INDEX.itemByKey.get(key))
    .filter(Boolean) as MenuItem[]
  const defaultPaletteItems = useMemo(() => {
    const combined = [
      ...favoriteMenuItems,
      ...recentMenuItems.filter(
        item => item.key && !favoriteKeys.includes(String(item.key))
      ),
    ]
    if (combined.length > 0) {
      return combined.slice(0, MENU_SEARCH_RESULTS_LIMIT)
    }
    return MENU_INDEX.menuKeys
      .slice(0, MENU_SEARCH_RESULTS_LIMIT)
      .map(key => MENU_INDEX.itemByKey.get(key))
      .filter(Boolean) as MenuItem[]
  }, [favoriteKeys, favoriteMenuItems, recentMenuItems])
  const normalizedMenuFilter = normalizeSearchText(menuFilterQuery)
  const hasMenuFilter = normalizedMenuFilter.length > 0
  const normalizedPaletteQuery = normalizeSearchText(paletteQueryText)
  const hasPaletteQuery = normalizedPaletteQuery.length > 0

  useEffect(() => {
    const routePath = `${location.pathname}${location.search}${location.hash}`
    writeStoredLastRoute(routePath, menuKeySet)
    const resolvedKey = resolveMenuKey(location.pathname)
    if (!menuKeySet.has(resolvedKey)) return
    setRecentKeys(prev => {
      if (prev[0] === resolvedKey) return prev
      const next = [resolvedKey, ...prev.filter(key => key !== resolvedKey)]
      return next.slice(0, MENU_RECENTS_LIMIT)
    })
  }, [
    location.hash,
    location.pathname,
    location.search,
    menuKeySet,
    setRecentKeys,
  ])

  const filteredMenuItems = useMemo(
    () =>
      hasMenuFilter
        ? filterMenuItems(MENU_ITEMS, normalizedMenuFilter)
        : MENU_ITEMS,
    [hasMenuFilter, normalizedMenuFilter]
  )
  const filteredMenuKeys = useMemo(
    () => (hasMenuFilter ? collectMenuKeys(filteredMenuItems) : menuKeys),
    [hasMenuFilter, filteredMenuItems, menuKeys]
  )
  const selectedKey = getSelectedKey(menuKeySet)
  const isFavorite = selectedKey ? favoriteKeySet.has(selectedKey) : false
  const selectedParentKeys =
    MENU_INDEX.parentKeysByKey.get(selectedKey) ?? []
  const selectedMenuItem = selectedKey
    ? MENU_INDEX.itemByKey.get(selectedKey) ?? null
    : null
  const currentMenuLabel = selectedMenuItem
    ? getMenuLabelText(selectedMenuItem.label)
    : ''
  const currentMenuTitle =
    selectedKey && currentMenuLabel
      ? getMenuTitleText(selectedKey, currentMenuLabel)
      : ''
  const currentRoutePath = `${location.pathname}${location.search}${location.hash}`
  useEffect(() => {
    if (typeof document === 'undefined') return
    const baseTitle = 'AI Agent'
    document.title = currentMenuTitle
      ? `${currentMenuTitle} - ${baseTitle}`
      : baseTitle
  }, [currentMenuTitle])
  const breadcrumbItems = selectedParentKeys.reduce(
    (acc, key) => {
      const item = MENU_INDEX.itemByKey.get(key)
      if (!item) return acc
      const label = getMenuLabelText(item.label)
      if (!label) return acc
      if (menuKeySet.has(key)) {
        acc.push({
          title: (
            <a
              href={resolveMenuPath(key)}
              onClick={event => {
                event.preventDefault()
                handleMenuNavigate(key)
              }}
            >
              {label}
            </a>
          ),
        })
        return acc
      }
      acc.push({ title: label })
      return acc
    },
    [] as Array<{ title: React.ReactNode }>
  )
  const searchResults = useMemo(
    () =>
      hasMenuFilter
        ? buildNavigationResults(
            filteredMenuKeys,
            menuFilterQuery,
            MENU_SEARCH_RESULTS_LIMIT,
            { favorites: favoriteKeySet, recents: recentKeys }
          )
        : [],
    [favoriteKeySet, filteredMenuKeys, hasMenuFilter, menuFilterQuery, recentKeys]
  )
  const menuRouteMatchCount = useMemo(
    () => (hasMenuFilter ? countRouteMatches(menuFilterQuery) : 0),
    [hasMenuFilter, menuFilterQuery]
  )
  const menuMatchTotal = hasMenuFilter
    ? countMenuMatches(filteredMenuKeys, menuFilterQuery) + menuRouteMatchCount
    : 0
  const showMenuSearchMore =
    hasMenuFilter && menuMatchTotal > searchResults.length
  const paletteMenuKeys = useMemo(
    () =>
      hasPaletteQuery
        ? collectMenuKeys(filterMenuItems(MENU_ITEMS, normalizedPaletteQuery))
        : [],
    [hasPaletteQuery, normalizedPaletteQuery]
  )
  const paletteRouteMatchCount = useMemo(
    () => (hasPaletteQuery ? countRouteMatches(paletteQueryText) : 0),
    [hasPaletteQuery, paletteQueryText]
  )
  const paletteMatchTotal = hasPaletteQuery
    ? countMenuMatches(paletteMenuKeys, paletteQueryText) +
      paletteRouteMatchCount
    : 0
  const paletteResults = useMemo(
    () => {
      const results = hasPaletteQuery
        ? buildNavigationResults(
            paletteMenuKeys,
            paletteQueryText,
            MENU_SEARCH_RESULTS_LIMIT,
            { favorites: favoriteKeySet, recents: recentKeys }
          )
        : defaultPaletteItems
      if (!directPaletteEntryKey) return results
      if (results.some(item => String(item.key) === directPaletteEntryKey)) {
        return results
      }
      return [
        {
          key: directPaletteEntryKey,
          label: directPaletteEntryLabel,
        } as MenuItem,
        ...results,
      ]
    },
    [
      defaultPaletteItems,
      directPaletteEntryKey,
      directPaletteEntryLabel,
      favoriteKeySet,
      hasPaletteQuery,
      paletteQueryText,
      paletteMenuKeys,
      recentKeys,
    ]
  )
  const showPaletteSearchMore =
    hasPaletteQuery && paletteMatchTotal > paletteResults.length

  useEffect(() => {
    setMenuSearchActiveIndex(index => {
      const next = clampIndex(index, searchResults.length)
      return next === index ? index : next
    })
  }, [searchResults.length])

  useEffect(() => {
    setPaletteActiveIndex(index => {
      const next = clampIndex(index, paletteResults.length)
      return next === index ? index : next
    })
  }, [paletteResults.length])

  useEffect(() => {
    if (!paletteOpen) return
    const container = paletteListRef.current
    if (!container) return
    const frame = requestAnimationFrame(() => {
      const active = container.querySelector<HTMLElement>(
        `[data-palette-index="${paletteActiveIndex}"]`
      )
      if (active) active.scrollIntoView({ block: 'nearest' })
    })
    return () => cancelAnimationFrame(frame)
  }, [paletteActiveIndex, paletteOpen, paletteResults.length])
  const resolvedOpenKeys = hasMenuFilter
    ? collectSubmenuKeys(filteredMenuItems)
    : selectedParentKeys.length > 0
      ? Array.from(new Set([...openKeys, ...selectedParentKeys]))
      : openKeys

  const toggleFavorite = (menuKey: string) => {
    if (!menuKeySet.has(menuKey)) return
    setFavoriteKeys(prev => {
      if (prev.includes(menuKey)) {
        return prev.filter(key => key !== menuKey)
      }
      return [menuKey, ...prev]
    })
  }

  const handleMenuNavigate = (menuKey: string) => {
    const targetPath = resolveNavigationPath(menuKey)
    if (!targetPath) return
    if (!menuKey.startsWith('/') && !menuKeySet.has(menuKey)) return
    navigate(targetPath)
    setDrawerOpen(false)
  }

  const handlePaletteNavigate = (menuKey: string) => {
    closePalette()
    handleMenuNavigate(menuKey)
  }

  const handleCopyPath = async (path: string) => {
    if (!path) return
    try {
      await copyToClipboard(path)
      message.success('路径已复制')
    } catch {
      message.error('复制失败')
    }
  }

  const handleToggleCollapsed = () => {
    const next = !collapsed
    setCollapsed(next)
    if (!siderBroken) setManualCollapsed(next)
  }

  const healthStatusInfo =
    getHealthStatusInfo(healthStatus, healthError)
  const healthUpdatedAt = healthTimestamp
    ? new Date(healthTimestamp).toLocaleString()
    : ''
  const healthTooltip = healthError
    ? `${healthError}${healthUpdatedAt ? ` · 最近更新 ${healthUpdatedAt}` : ''}`
    : healthUpdatedAt
      ? `最近更新 ${healthUpdatedAt}，点击查看详情`
      : '点击查看详情'

  const renderNavContent = (isCollapsed: boolean) => {
    const showRecentSection =
      !isCollapsed && !hasMenuFilter && recentMenuItems.length > 0
    const showFavoriteSection =
      !isCollapsed && !hasMenuFilter && favoriteMenuItems.length > 0

    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          minHeight: 0,
          flex: 1,
        }}
      >
        <div
          style={{
            padding: '16px',
            borderBottom: '1px solid #e8e8e8',
            textAlign: isCollapsed ? 'center' : 'left',
            height: 'auto',
            minHeight: 'auto',
          }}
        >
          <Space
            align="center"
            style={{ justifyContent: isCollapsed ? 'center' : 'flex-start' }}
          >
            <Avatar
              size={isCollapsed ? 32 : 40}
              icon={<RobotOutlined />}
              style={{ backgroundColor: '#1890ff' }}
            />
            {!isCollapsed && (
              <div>
                <Title level={5} style={{ margin: 0 }}>
                  AI Agent
                </Title>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  完整技术架构映射
                </Text>
              </div>
            )}
          </Space>
        </div>

        {!isCollapsed && (
          <div
            style={{
              padding: '12px 16px',
              borderBottom: '1px solid #e8e8e8',
            }}
          >
            <Input
              ref={searchInputRef}
              allowClear
              size="small"
              placeholder='搜索功能或路由，支持 "短语"，或输入 /path (Ctrl+K)'
              prefix={<SearchOutlined style={{ color: '#999' }} />}
              value={menuFilter}
              onChange={e => setMenuFilter(e.target.value)}
              onKeyDown={event => {
                if (event.key === 'Escape' && menuFilter) {
                  event.preventDefault()
                  setMenuFilter('')
                  return
                }
                if (!hasMenuFilter || searchResults.length === 0) return
                if (event.key === 'ArrowDown') {
                  event.preventDefault()
                  setMenuSearchActiveIndex(index =>
                    wrapIndex(index + 1, searchResults.length)
                  )
                }
                if (event.key === 'ArrowUp') {
                  event.preventDefault()
                  setMenuSearchActiveIndex(index =>
                    wrapIndex(index - 1, searchResults.length)
                  )
                }
              }}
              onPressEnter={() => {
                if (directMenuPath) {
                  setMenuFilter('')
                  handleMenuNavigate(directMenuTargetPath)
                  return
                }
                if (searchResults.length === 0) {
                  if (menuFilter.trim()) {
                    setMenuFilter('')
                    handleMenuNavigate(buildPageIndexPath(menuFilter))
                  }
                  return
                }
                const targetIndex = clampIndex(
                  menuSearchActiveIndex,
                  searchResults.length
                )
                const target = searchResults[targetIndex] || searchResults[0]
                if (!target) return
                setMenuFilter('')
                handleMenuNavigate(String(target.key))
              }}
            />
            {hasMenuFilter && (
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {menuMatchTotal > 0
                    ? `匹配 ${menuMatchTotal} 项，↑↓ 选择后回车跳转`
                    : '未找到匹配项，回车打开页面索引'}
                </Text>
                {directMenuPath && (
                  <div style={{ marginTop: '6px' }}>
                    <Button
                      size="small"
                      type="dashed"
                      onClick={() => {
                        setMenuFilter('')
                        handleMenuNavigate(directMenuTargetPath)
                      }}
                    >
                      {directMenuLabel}
                    </Button>
                  </div>
                )}
                {searchResults.length > 0 && (
                  <div style={{ marginTop: '6px' }}>
                    <div
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 6,
                      }}
                    >
                      {searchResults.map((item, index) => {
                        const active = index === menuSearchActiveIndex
                        const menuKey = String(item.key)
                        const metaText = getNavigationMetaText(menuKey)
                        const canFavorite = menuKeySet.has(menuKey)
                        const isSearchFavorite =
                          canFavorite && favoriteKeySet.has(menuKey)
                        return (
                          <div
                            key={`search-${menuKey}`}
                            style={{
                              display: 'flex',
                              alignItems: 'stretch',
                              gap: 6,
                            }}
                            onMouseEnter={() =>
                              setMenuSearchActiveIndex(index)
                            }
                          >
                            <Button
                              size="small"
                              type={active ? 'primary' : 'text'}
                              icon={getMenuItemIcon(item)}
                              title={metaText}
                              onClick={() => {
                                setMenuFilter('')
                                handleMenuNavigate(menuKey)
                              }}
                              style={{
                                flex: 1,
                                justifyContent: 'flex-start',
                                textAlign: 'left',
                                height: 'auto',
                                padding: '6px 8px',
                                whiteSpace: 'normal',
                              }}
                            >
                              <div
                                style={{
                                  display: 'flex',
                                  flexDirection: 'column',
                                  alignItems: 'flex-start',
                                  width: '100%',
                                }}
                              >
                                <span>
                                  {renderHighlightedText(
                                    getMenuLabelText(item.label),
                                    menuFilterQuery
                                  )}
                                </span>
                                <Text
                                  type={active ? undefined : 'secondary'}
                                  style={{ fontSize: 11 }}
                                >
                                  {metaText}
                                </Text>
                              </div>
                            </Button>
                            {canFavorite && (
                              <Tooltip
                                title={
                                  isSearchFavorite ? '取消收藏' : '收藏'
                                }
                              >
                                <Button
                                  type="text"
                                  icon={
                                    isSearchFavorite ? (
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
                                type="text"
                                icon={<CopyOutlined />}
                                onClick={event => {
                                  event.preventDefault()
                                  event.stopPropagation()
                                  handleCopyPath(resolveNavigationPath(menuKey))
                                }}
                              />
                            </Tooltip>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
                {showMenuSearchMore && (
                  <div style={{ marginTop: '6px' }}>
                    <Button
                      size="small"
                      type="link"
                      style={{ padding: 0 }}
                      onClick={() => {
                        setMenuFilter('')
                        handleMenuNavigate(buildPageIndexPath(menuFilterQuery))
                      }}
                    >
                      查看更多结果
                    </Button>
                  </div>
                )}
                {menuMatchTotal === 0 && (
                  <div style={{ marginTop: '6px' }}>
                    <Button
                      size="small"
                      type="link"
                      style={{ padding: 0 }}
                      onClick={() => {
                        setMenuFilter('')
                        handleMenuNavigate(buildPageIndexPath(menuFilterQuery))
                      }}
                    >
                      打开页面索引
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {showFavoriteSection && (
          <div
            style={{
              padding: '8px 16px 12px',
              borderBottom: '1px solid #e8e8e8',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Text type="secondary" style={{ fontSize: '12px' }}>
                收藏
              </Text>
              <Button
                size="small"
                type="link"
                onClick={() => setFavoriteKeys([])}
                style={{ padding: 0 }}
              >
                清空
              </Button>
            </div>
            <div style={{ marginTop: '8px' }}>
              <Space size={[8, 8]} wrap>
                {favoriteMenuItems.map(item => (
                  <Button
                    key={`favorite-${String(item.key)}`}
                    size="small"
                    type="text"
                    icon={getMenuItemIcon(item)}
                    title={getMenuMetaText(String(item.key))}
                    onClick={() => handleMenuNavigate(String(item.key))}
                    style={{ paddingInline: 4 }}
                  >
                    {getMenuLabelText(item.label)}
                  </Button>
                ))}
              </Space>
            </div>
          </div>
        )}

        {showRecentSection && (
          <div
            style={{
              padding: '8px 16px 12px',
              borderBottom: '1px solid #e8e8e8',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <Text type="secondary" style={{ fontSize: '12px' }}>
                最近访问
              </Text>
              <Button
                size="small"
                type="link"
                onClick={() => setRecentKeys([])}
                style={{ padding: 0 }}
              >
                清空
              </Button>
            </div>
            <div style={{ marginTop: '8px' }}>
              <Space size={[8, 8]} wrap>
                {recentMenuItems.map(item => (
                  <Button
                    key={`recent-${String(item.key)}`}
                    size="small"
                    type="text"
                    icon={getMenuItemIcon(item)}
                    title={getMenuMetaText(String(item.key))}
                    onClick={() => handleMenuNavigate(String(item.key))}
                    style={{ paddingInline: 4 }}
                  >
                    {getMenuLabelText(item.label)}
                  </Button>
                ))}
              </Space>
            </div>
          </div>
        )}

        <Menu
          mode="inline"
          selectedKeys={selectedKey ? [selectedKey] : []}
          openKeys={resolvedOpenKeys}
          items={filteredMenuItems}
          style={{
            border: 'none',
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            background: '#fff',
          }}
          onOpenChange={keys => {
            if (hasMenuFilter) return
            setOpenKeys(
              (keys as string[]).filter(key => MENU_SUBMENU_KEY_SET.has(key))
            )
          }}
          onClick={({ key }) => {
            if (menuFilter) setMenuFilter('')
            handleMenuNavigate(String(key))
          }}
        />
      </div>
    )
  }

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {!hideSider && (
        <Sider
          data-testid="sidebar"
          trigger={null}
          collapsible
          collapsed={collapsed}
          breakpoint="lg"
          onBreakpoint={broken => {
            setSiderBroken(broken)
            setCollapsed(broken ? true : manualCollapsed)
          }}
          width={280}
          collapsedWidth={siderBroken ? 0 : 80}
          style={{
            background: '#fff',
            borderRight:
              siderBroken && collapsed ? 'none' : '1px solid #e8e8e8',
            position: 'sticky',
            top: 0,
            height: '100vh',
            overflow: 'auto',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {renderNavContent(collapsed)}
        </Sider>
      )}

      <Layout>
        <Header
          style={{
            background: '#fff',
            borderBottom: '1px solid #e8e8e8',
            padding: '0 20px',
            height: '60px',
            lineHeight: '60px',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Space align="center" size={16}>
              <Button
                type="text"
                icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                onClick={siderBroken || hideSider ? openDrawer : handleToggleCollapsed}
                style={{ fontSize: '16px' }}
              />
              {siderBroken && (
                <Title level={5} style={{ margin: 0 }}>
                  AI Agent
                </Title>
              )}
              {currentMenuLabel && (
                <div
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    lineHeight: 1.1,
                  }}
                >
                  {breadcrumbItems.length > 0 && (
                    <Breadcrumb
                      items={breadcrumbItems}
                      style={{ fontSize: '12px' }}
                    />
                  )}
                  <Title level={5} style={{ margin: 0 }}>
                    {currentMenuLabel}
                  </Title>
                </div>
              )}
              {selectedKey && (
                <Tooltip title={isFavorite ? '取消收藏' : '收藏当前页面'}>
                  <Button
                    type="text"
                    icon={
                      isFavorite ? (
                        <StarFilled style={{ color: '#fadb14' }} />
                      ) : (
                        <StarOutlined />
                      )
                    }
                    onClick={() => toggleFavorite(selectedKey)}
                  />
                </Tooltip>
              )}
              {currentRoutePath && (
                <Tooltip title="复制当前路径">
                  <Button
                    type="text"
                    icon={<CopyOutlined />}
                    onClick={() => handleCopyPath(currentRoutePath)}
                  />
                </Tooltip>
              )}
            </Space>
            <Space align="center" size={12}>
              {healthLoading ? (
                <Text type="secondary" style={{ fontSize: 12 }}>
                  健康检测中...
                </Text>
              ) : (
                <Tooltip
                  title={healthTooltip}
                >
                  <Tag
                    color={healthStatusInfo.color}
                    onClick={openHealthDetail}
                    style={{ cursor: 'pointer' }}
                  >
                    {healthStatusInfo.label}
                  </Tag>
                </Tooltip>
              )}
              <Tooltip title="刷新健康状态">
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={refreshHealth}
                  loading={healthLoading}
                />
              </Tooltip>
              <Tooltip title="快捷键速览 (?)">
                <Button
                  type="text"
                  icon={<QuestionCircleOutlined />}
                  onClick={openShortcut}
                >
                  快捷键
                </Button>
              </Tooltip>
              <Tooltip title="快速搜索 (Ctrl+K /)">
                <Button
                  type="text"
                  icon={<SearchOutlined />}
                  onClick={openPalette}
                >
                  快速搜索
                </Button>
              </Tooltip>
            </Space>
          </div>
        </Header>
        <Drawer
          title="功能导航"
          open={drawerOpen}
          destroyOnClose
          onClose={() => setDrawerOpen(false)}
          placement="left"
          width={280}
          bodyStyle={{
            padding: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {renderNavContent(false)}
        </Drawer>

        <Content
          id="app-content"
          style={{
            background: '#f0f2f5',
            overflow: 'auto',
          }}
        >
          <Suspense
            fallback={
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  height: 'calc(100vh - 120px)',
                  flexDirection: 'column',
                  gap: '16px',
                }}
              >
                <Spin size="large" />
                <div style={{ color: '#666', fontSize: '16px' }}>加载中...</div>
              </div>
            }
          >
            <AppRoutes />
          </Suspense>
        </Content>
        <Modal
          title="快速跳转"
          open={paletteOpen}
          onCancel={closePalette}
          footer={null}
          width={520}
          centered
        >
          <Input
            ref={paletteInputRef}
            allowClear
            placeholder='搜索功能或路由，支持 "短语"，或输入 /path'
            prefix={<SearchOutlined style={{ color: '#999' }} />}
            value={paletteQuery}
            onChange={e => {
              setPaletteQuery(e.target.value)
              setPaletteActiveIndex(0)
            }}
            onKeyDown={event => {
              if (event.key === 'Escape') {
                event.preventDefault()
                closePalette()
                return
              }
              if (!paletteResults.length) return
              if (event.key === 'ArrowDown') {
                event.preventDefault()
                setPaletteActiveIndex(index =>
                  wrapIndex(index + 1, paletteResults.length)
                )
              }
              if (event.key === 'ArrowUp') {
                event.preventDefault()
                setPaletteActiveIndex(index =>
                  wrapIndex(index - 1, paletteResults.length)
                )
              }
            }}
            onPressEnter={() => {
              if (paletteResults.length === 0) {
                if (paletteQuery.trim()) {
                  handlePaletteNavigate(buildPageIndexPath(paletteQuery))
                }
                return
              }
              const targetIndex = clampIndex(
                paletteActiveIndex,
                paletteResults.length
              )
              const target = paletteResults[targetIndex] || paletteResults[0]
              if (target) {
                const targetKey =
                  directPaletteRegistered &&
                  String(target.key) === directPaletteKey
                    ? directPaletteTargetPath
                    : String(target.key)
                handlePaletteNavigate(targetKey)
              }
            }}
          />
          <div style={{ marginTop: 12 }}>
            {paletteResults.length > 0 ? (
              <div
                style={{
                  maxHeight: 360,
                  overflowY: 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                }}
                ref={paletteListRef}
              >
                {paletteResults.map((item, index) => {
                  const menuKey = String(item.key)
                  const metaText = getNavigationMetaText(menuKey)
                  const active = index === paletteActiveIndex
                  const canFavorite = menuKeySet.has(menuKey)
                  const isPaletteFavorite =
                    canFavorite && favoriteKeySet.has(menuKey)
                  const copyPath =
                    directPaletteKey && menuKey === directPaletteKey
                      ? directPaletteTargetPath
                      : resolveNavigationPath(menuKey)
                  return (
                    <div
                      key={`palette-${menuKey}`}
                      data-palette-index={index}
                      onMouseEnter={() => setPaletteActiveIndex(index)}
                      style={{
                        display: 'flex',
                        gap: 6,
                        alignItems: 'stretch',
                      }}
                    >
                      <Button
                        type={active ? 'primary' : 'text'}
                        icon={getMenuItemIcon(item)}
                        onClick={() => {
                          const targetKey =
                            directPaletteKey && menuKey === directPaletteKey
                              ? directPaletteTargetPath
                              : menuKey
                          handlePaletteNavigate(targetKey)
                        }}
                        style={{
                          flex: 1,
                          justifyContent: 'flex-start',
                          textAlign: 'left',
                          height: 'auto',
                          padding: '6px 8px',
                          whiteSpace: 'normal',
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'flex-start',
                            width: '100%',
                          }}
                        >
                          <span>
                            {renderHighlightedText(
                              getMenuLabelText(item.label),
                              paletteQueryText
                            )}
                          </span>
                          <Text
                            type={active ? undefined : 'secondary'}
                            style={{ fontSize: 11 }}
                          >
                            {metaText}
                          </Text>
                        </div>
                      </Button>
                      {canFavorite && (
                        <Tooltip
                          title={isPaletteFavorite ? '取消收藏' : '收藏'}
                        >
                          <Button
                            type="text"
                            icon={
                              isPaletteFavorite ? (
                                <StarFilled style={{ color: '#fadb14' }} />
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
              </div>
            ) : (
              <div>
                <Text type="secondary">
                  {hasPaletteQuery
                    ? '未找到匹配项，回车打开页面索引'
                    : '暂无最近访问，可输入 /path 直达'}
                </Text>
                {hasPaletteQuery && (
                  <div style={{ marginTop: 8 }}>
                    <Button
                      size="small"
                      type="link"
                      style={{ padding: 0 }}
                      onClick={() =>
                        handlePaletteNavigate(buildPageIndexPath(paletteQueryText))
                      }
                    >
                      打开页面索引
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginTop: 12,
            }}
          >
            <Space size={8}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {directPalettePath
                  ? `回车直达 ${directPaletteTargetPath}${
                      directPaletteRegistered
                        ? ''
                        : directPaletteKnown
                          ? '（未收录）'
                          : '（未注册）'
                    } · 匹配 ${paletteMatchTotal} 项`
                  : hasPaletteQuery
                    ? `匹配 ${paletteMatchTotal} 项`
                    : '使用 ↑↓ 选择，回车跳转'}
              </Text>
              {showPaletteSearchMore && (
                <Button
                  size="small"
                  type="link"
                  style={{ padding: 0 }}
                  onClick={() =>
                    handlePaletteNavigate(buildPageIndexPath(paletteQueryText))
                  }
                >
                  查看更多
                </Button>
              )}
            </Space>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Esc 关闭
            </Text>
          </div>
        </Modal>
        <HealthDetailModal
          open={healthDetailOpen}
          loading={healthDetailLoading}
          error={healthDetailError}
          detail={healthDetail}
          metrics={healthMetrics}
          alerts={healthAlerts}
          alertsTotal={healthAlertsTotal}
          onRefresh={refreshHealth}
          onClose={closeHealthDetail}
        />
        <ShortcutHelpModal open={shortcutOpen} onClose={closeShortcut} />
      </Layout>
    </Layout>
  )
}

export default App
