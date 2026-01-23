import React from 'react'
import { Alert, Button, Space } from 'antd'
import { ReloadOutlined, WifiOutlined } from '@ant-design/icons'

interface NetworkErrorAlertProps {
  error: string | null
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
}

const NetworkErrorAlert: React.FC<NetworkErrorAlertProps> = ({
  error,
  onRetry,
  onDismiss,
  className = '',
}) => {
  if (!error) return null

  const isNetworkError =
    error.toLowerCase().includes('network') ||
    error.toLowerCase().includes('fetch') ||
    error.toLowerCase().includes('连接')

  return (
    <Alert
      message={isNetworkError ? '网络连接异常' : '请求失败'}
      description={error}
      type="error"
      showIcon
      icon={isNetworkError ? <WifiOutlined /> : undefined}
      className={`animate-slide-up ${className}`}
      action={
        <Space>
          {onRetry && (
            <Button
              size="small"
              type="primary"
              ghost
              icon={<ReloadOutlined />}
              onClick={onRetry}
            >
              重试
            </Button>
          )}
          {onDismiss && (
            <Button size="small" type="text" onClick={onDismiss}>
              关闭
            </Button>
          )}
        </Space>
      }
      closable={!!onDismiss}
      onClose={onDismiss}
    />
  )
}

export default NetworkErrorAlert
