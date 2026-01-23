import React, { useMemo } from 'react'
import { Button, Result, Space, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { getMenuLabelText, resolveMenuPath } from '../routes/menuConfig'
import { MENU_INDEX, MENU_KEY_SET } from '../routes/menuIndex'
import { readStoredLastRoute } from '../routes/navigationStorage'
import { splitSearchTokens } from '../utils/searchText'
import FallbackActions from '../components/ui/FallbackActions'

const { Text } = Typography

type MenuEntry = {
  key: string
  label: string
  path: string
  searchText: string
}

const MENU_ENTRIES: MenuEntry[] = MENU_INDEX.menuKeys
  .map(key => {
    const item = MENU_INDEX.itemByKey.get(key)
    const searchText = MENU_INDEX.searchTextByKey.get(key)
    if (!item || !searchText) return null
    return {
      key,
      label: getMenuLabelText(item.label) || key,
      path: resolveMenuPath(key),
      searchText,
    }
  })
  .filter(Boolean) as MenuEntry[]

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()
  const recentRoute = useMemo(() => {
    const lastRoute = readStoredLastRoute(MENU_KEY_SET)
    if (!lastRoute) return null
    if (lastRoute === location.pathname) return null
    if (lastRoute === '/workspace') return null
    return lastRoute
  }, [location.pathname])
  const suggestions = useMemo(() => {
    const tokens = splitSearchTokens(location.pathname)
    if (tokens.length === 0) return []
    return MENU_ENTRIES.map(entry => {
      const score = tokens.reduce((acc, token) => {
        return entry.searchText.includes(token) ? acc + 1 : acc
      }, 0)
      return { entry, score }
    })
      .filter(item => item.score > 0)
      .sort((a, b) => {
        if (a.score !== b.score) return b.score - a.score
        return a.entry.label.length - b.entry.label.length
      })
      .slice(0, 6)
      .map(item => item.entry)
  }, [location.pathname])

  return (
    <div style={{ padding: '48px 24px' }}>
      <Result
        status="404"
        title="页面不存在"
        subTitle="当前地址无法匹配任何功能模块"
        extra={
          <FallbackActions
            backLabel="返回上一页"
            recentLabel={recentRoute ? '返回最近访问' : undefined}
            recentPath={recentRoute || undefined}
            homeLabel="返回首页"
            homePath="/workspace"
            searchLabel="打开导航搜索"
            searchDescription="快速定位已上线页面"
          />
        }
      />
      <div style={{ textAlign: 'center' }}>
        <Text type="secondary">路径：{location.pathname}</Text>
        <div style={{ marginTop: 8 }}>
          <Text type="secondary">建议使用左侧菜单或 Ctrl+K 搜索功能</Text>
        </div>
        {suggestions.length > 0 && (
          <div style={{ marginTop: 24 }}>
            <Text type="secondary">你可能在找</Text>
            <div style={{ marginTop: 12 }}>
              <Space size={[8, 8]} wrap>
                {suggestions.map(item => (
                  <Button
                    key={`suggestion-${item.key}`}
                    size="small"
                    type="link"
                    title={item.path}
                    onClick={() => navigate(item.path)}
                  >
                    {item.label}
                  </Button>
                ))}
              </Space>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default NotFoundPage
