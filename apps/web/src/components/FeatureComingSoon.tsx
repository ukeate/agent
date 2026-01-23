import React from 'react'
import { Card, Typography } from 'antd'
import { ExperimentOutlined } from '@ant-design/icons'
import { useLocation } from 'react-router-dom'
import FallbackActions from './ui/FallbackActions'
import { readStoredLastRoute } from '../routes/navigationStorage'
import { MENU_KEY_SET } from '../routes/menuIndex'
import { resolveMenuKey } from '../routes/menuConfig'
import { MENU_INDEX } from '../routes/menuIndex'
import { getMenuLabelText } from '../routes/menuConfig'

interface FeatureComingSoonProps {
  title: string
  description?: string
}

export const FeatureComingSoon: React.FC<FeatureComingSoonProps> = ({
  title,
  description = '该功能正在开发中，敬请期待。',
}) => {
  const location = useLocation()
  const routeKey = resolveMenuKey(location.pathname)
  const menuItem = MENU_INDEX.itemByKey.get(routeKey)
  const label = menuItem ? getMenuLabelText(menuItem.label) : ''
  const isMenuRoute = MENU_KEY_SET.has(routeKey)
  const recentRoute = (() => {
    const lastRoute = readStoredLastRoute(MENU_KEY_SET)
    if (!lastRoute) return null
    if (lastRoute === location.pathname) return null
    if (lastRoute === '/workspace') return null
    return lastRoute
  })()

  return (
    <div
      style={{
        padding: '24px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '60vh',
      }}
    >
      <Card style={{ textAlign: 'center', maxWidth: '500px', width: '100%' }}>
        <ExperimentOutlined
          style={{ fontSize: '64px', color: '#1890ff', marginBottom: '16px' }}
        />
        <h2 style={{ marginBottom: '8px', fontSize: '24px' }}>{title}</h2>
        <p style={{ color: '#666', fontSize: '16px', marginBottom: '0' }}>
          {description}
        </p>
        {(label || isMenuRoute) && (
          <Typography.Text
            type="secondary"
            style={{ display: 'block', marginTop: 12, fontSize: 12 }}
          >
            {label ? `导航名称: ${label}` : '该页面已在导航中登记'}
            {` · 路径: ${location.pathname}`}
          </Typography.Text>
        )}
        <div
          style={{
            marginTop: '16px',
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <FallbackActions
            backLabel="返回上一页"
            recentLabel={recentRoute ? '返回最近访问' : undefined}
            recentPath={recentRoute || undefined}
            homeLabel="返回首页"
            homePath="/workspace"
            searchLabel="打开导航搜索"
            searchDescription="快速定位已上线页面"
          />
        </div>
      </Card>
    </div>
  )
}

export default FeatureComingSoon
