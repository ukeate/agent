import React from 'react'
import { Button, Space } from 'antd'
import { useNavigate } from 'react-router-dom'
import { dispatchPaletteOpen } from '../../utils/palette'

type FallbackActionsProps = {
  backLabel: string
  recentLabel?: string
  recentPath?: string
  homeLabel: string
  homePath: string
  searchLabel?: string
  searchDescription?: string
}

const FallbackActions: React.FC<FallbackActionsProps> = ({
  backLabel,
  recentLabel,
  recentPath,
  homeLabel,
  homePath,
  searchLabel,
  searchDescription,
}) => {
  const navigate = useNavigate()

  return (
    <Space>
      <Button onClick={() => navigate(-1)}>{backLabel}</Button>
      {recentLabel && recentPath && (
        <Button onClick={() => navigate(recentPath)}>{recentLabel}</Button>
      )}
      {searchLabel && (
        <Button onClick={dispatchPaletteOpen} title={searchDescription}>
          {searchLabel}
        </Button>
      )}
      <Button type="primary" onClick={() => navigate(homePath)}>
        {homeLabel}
      </Button>
    </Space>
  )
}

export default FallbackActions
