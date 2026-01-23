import React from 'react'
import {
  Modal,
  Descriptions,
  Tag,
  Typography,
  Space,
  Button,
  Alert,
  Divider,
  List,
  Empty,
} from 'antd'
import {
  type HealthCheckResult,
  type HealthAlert,
  type HealthComponent,
  type HealthStatus,
  type SystemMetrics,
} from '../../services/healthService'

const { Text } = Typography

type DisplayStatus = HealthStatus | 'unknown'

type HealthDetailModalProps = {
  open: boolean
  loading: boolean
  error?: string | null
  detail?: HealthCheckResult | null
  metrics?: SystemMetrics | null
  alerts?: HealthAlert[]
  alertsTotal?: number
  onRefresh: () => void
  onClose: () => void
}

const getStatusInfo = (status: DisplayStatus) => {
  if (status === 'healthy') return { color: 'success', label: '正常' }
  if (status === 'degraded') return { color: 'warning', label: '降级' }
  if (status === 'unhealthy') return { color: 'error', label: '异常' }
  return { color: 'default', label: '未知' }
}

const formatValue = (value: unknown) => {
  if (value === null || value === undefined) return '--'
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '--'
  }
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? '是' : '否'
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

const formatNumberValue = (
  value?: number | null,
  fractionDigits: number = 0
) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--'
  return value.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  })
}

const formatPercentValue = (
  value?: number | null,
  fractionDigits: number = 1
) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--'
  return `${value.toFixed(fractionDigits)}%`
}

const formatRateValue = (value?: number | null) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--'
  const percent = value <= 1 ? value * 100 : value
  return `${percent.toFixed(1)}%`
}

const formatDurationSeconds = (value?: number | null) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '--'
  const total = Math.max(0, Math.floor(value))
  const days = Math.floor(total / 86400)
  const hours = Math.floor((total % 86400) / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const seconds = total % 60
  const parts: string[] = []
  if (days) parts.push(`${days}天`)
  if (hours) parts.push(`${hours}小时`)
  if (minutes) parts.push(`${minutes}分钟`)
  if (!parts.length) parts.push(`${seconds}秒`)
  return parts.join(' ')
}

const buildDetailText = (component: HealthComponent) => {
  const entries = Object.entries(component).filter(([key]) => key !== 'status')
  if (entries.length === 0) return ''
  return entries.map(([key, value]) => `${key}: ${formatValue(value)}`).join(' · ')
}

const renderComponentTags = (items?: string[]) => {
  if (!items || items.length === 0) return <Text type="secondary">无</Text>
  return (
    <Space size={[4, 4]} wrap>
      {items.map(item => (
        <Tag key={item}>{item}</Tag>
      ))}
    </Space>
  )
}

const getSeverityInfo = (
  severity: HealthAlert['severity'] | undefined
) => {
  if (severity === 'critical') return { color: 'volcano', label: '严重' }
  if (severity === 'error') return { color: 'red', label: '错误' }
  if (severity === 'warning') return { color: 'orange', label: '警告' }
  if (severity === 'info') return { color: 'blue', label: '提示' }
  return { color: 'default', label: '未知' }
}

const formatTimestamp = (value?: string) => {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

const HealthDetailModal: React.FC<HealthDetailModalProps> = ({
  open,
  loading,
  error,
  detail,
  metrics,
  alerts,
  alertsTotal,
  onRefresh,
  onClose,
}) => {
  const status = (detail?.status ?? 'unknown') as DisplayStatus
  const statusInfo = getStatusInfo(status)
  const updatedAt = detail?.timestamp
    ? new Date(detail.timestamp).toLocaleString()
    : '--'
  const duration =
    detail?.check_duration_ms !== undefined
      ? `${detail.check_duration_ms} ms`
      : '--'
  const componentEntries = detail?.components
    ? Object.entries(detail.components)
    : []
  const metricsPayload = metrics ?? (detail?.metrics as SystemMetrics | undefined)
  const metricsText = metricsPayload
    ? JSON.stringify(metricsPayload, null, 2)
    : ''
  const systemMetrics = metricsPayload?.system
  const performanceMetrics = metricsPayload?.performance
  const alertItems = alerts ?? []
  const alertCount =
    typeof alertsTotal === 'number' ? alertsTotal : alertItems.length

  return (
    <Modal
      title="系统健康详情"
      open={open}
      onCancel={onClose}
      footer={
        <Space>
          <Button onClick={onRefresh} loading={loading}>
            刷新
          </Button>
          <Button type="primary" onClick={onClose}>
            关闭
          </Button>
        </Space>
      }
      width={720}
      destroyOnClose
    >
      {error && (
        <div style={{ marginBottom: 12 }}>
          <Alert type="error" message={error} showIcon />
        </div>
      )}
      <Descriptions column={1} size="small" bordered>
        <Descriptions.Item label="总体状态">
          <Tag color={statusInfo.color}>{statusInfo.label}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="最近更新时间">{updatedAt}</Descriptions.Item>
        <Descriptions.Item label="检查耗时">{duration}</Descriptions.Item>
        <Descriptions.Item label="异常组件">
          {renderComponentTags(detail?.failed_components)}
        </Descriptions.Item>
        <Descriptions.Item label="降级组件">
          {renderComponentTags(detail?.degraded_components)}
        </Descriptions.Item>
      </Descriptions>

      <Divider orientation="left" plain>
        组件状态
      </Divider>
      {componentEntries.length === 0 ? (
        <Empty description="暂无组件信息" />
      ) : (
        <List
          size="small"
          dataSource={componentEntries}
          renderItem={([name, component]) => {
            const componentStatus =
              typeof component === 'object' && component && 'status' in component
                ? ((component as HealthComponent).status as DisplayStatus)
                : 'unknown'
            const info = getStatusInfo(componentStatus)
            const detailText =
              typeof component === 'object' && component
                ? buildDetailText(component as HealthComponent)
                : ''
            return (
              <List.Item>
                <Space direction="vertical" size={2} style={{ width: '100%' }}>
                  <Space>
                    <Text strong>{name}</Text>
                    <Tag color={info.color}>{info.label}</Tag>
                  </Space>
                  {detailText && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {detailText}
                    </Text>
                  )}
                </Space>
              </List.Item>
            )
          }}
        />
      )}

      <Divider orientation="left" plain>
        活动告警
      </Divider>
      {alertItems.length > 0 ? (
        <>
          <Text
            type="secondary"
            style={{ fontSize: 12, display: 'block', marginBottom: 8 }}
          >
            {alertCount > alertItems.length
              ? `当前显示 ${alertItems.length} / ${alertCount}`
              : `共 ${alertCount} 条`}
          </Text>
          <List
            size="small"
            dataSource={alertItems}
            renderItem={alert => {
              const severityInfo = getSeverityInfo(alert.severity)
              return (
                <List.Item>
                  <Space
                    direction="vertical"
                    size={2}
                    style={{ width: '100%' }}
                  >
                    <Space wrap>
                      <Text strong>{alert.name || '告警'}</Text>
                      <Tag color={severityInfo.color}>
                        {severityInfo.label}
                      </Tag>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {formatTimestamp(alert.timestamp)}
                      </Text>
                    </Space>
                    {alert.message && (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {alert.message}
                      </Text>
                    )}
                  </Space>
                </List.Item>
              )
            }}
          />
        </>
      ) : (
        <Text type="secondary">暂无活动告警</Text>
      )}

      <Divider orientation="left" plain>
        指标摘要
      </Divider>
      {metricsPayload ? (
        <>
          {metricsPayload.error && (
            <div style={{ marginBottom: 12 }}>
              <Alert type="warning" message={metricsPayload.error} showIcon />
            </div>
          )}
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="CPU使用率">
              {formatPercentValue(systemMetrics?.cpu_percent)}
            </Descriptions.Item>
            <Descriptions.Item label="内存使用率">
              {formatPercentValue(systemMetrics?.memory_percent)}
            </Descriptions.Item>
            <Descriptions.Item label="磁盘使用率">
              {formatPercentValue(systemMetrics?.disk_percent)}
            </Descriptions.Item>
            <Descriptions.Item label="网络连接数">
              {systemMetrics?.network_connections_error
                ? '获取失败'
                : formatNumberValue(systemMetrics?.network_connections ?? null)}
            </Descriptions.Item>
            <Descriptions.Item label="运行时长">
              {formatDurationSeconds(performanceMetrics?.uptime_seconds)}
            </Descriptions.Item>
            <Descriptions.Item label="平均响应时间">
              {performanceMetrics?.average_response_time_ms !== undefined
                ? `${formatNumberValue(performanceMetrics.average_response_time_ms, 1)} ms`
                : '--'}
            </Descriptions.Item>
            <Descriptions.Item label="每分钟请求数">
              {formatNumberValue(performanceMetrics?.requests_per_minute, 1)}
            </Descriptions.Item>
            <Descriptions.Item label="错误率">
              {formatRateValue(performanceMetrics?.error_rate)}
            </Descriptions.Item>
            <Descriptions.Item label="活跃请求数">
              {formatNumberValue(performanceMetrics?.active_requests ?? null)}
            </Descriptions.Item>
            <Descriptions.Item label="累计请求数">
              {formatNumberValue(performanceMetrics?.total_requests ?? null)}
            </Descriptions.Item>
          </Descriptions>
        </>
      ) : (
        <Text type="secondary">暂无指标数据</Text>
      )}

      <Divider orientation="left" plain>
        指标原始数据
      </Divider>
      {metricsText ? (
        <pre
          style={{
            background: '#f6f8fa',
            padding: '12px',
            borderRadius: '8px',
            maxHeight: 240,
            overflow: 'auto',
            fontSize: '12px',
            margin: 0,
          }}
        >
          {metricsText}
        </pre>
      ) : (
        <Text type="secondary">暂无指标数据</Text>
      )}
    </Modal>
  )
}

export default HealthDetailModal
