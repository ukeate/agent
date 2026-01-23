import React from 'react'
import { Tag, Typography } from 'antd'
import { useAgentStore } from '@/stores/agentStore'

const { Text } = Typography

const STATUS_META: Record<
  'idle' | 'thinking' | 'acting' | 'error',
  { color: string; label: string }
> = {
  idle: { color: 'green', label: '就绪' },
  thinking: { color: 'blue', label: '思考中' },
  acting: { color: 'gold', label: '执行中' },
  error: { color: 'red', label: '异常' },
}

const formatAverage = (value: number) => {
  if (!Number.isFinite(value) || value <= 0) return '--'
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`
  return `${Math.round(value)}ms`
}

const AgentStatusBar: React.FC = () => {
  const { status, stats, error } = useAgentStore()
  const currentStatus = status?.status ?? 'idle'
  const statusMeta = STATUS_META[currentStatus] ?? STATUS_META.idle
  const currentTask = status?.currentTask
  const averageLabel = formatAverage(stats.averageResponseTime)

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500">
      <Tag color={statusMeta.color}>{statusMeta.label}</Tag>
      {currentTask && (
        <Text type="secondary" className="text-xs">
          当前：{currentTask}
        </Text>
      )}
      <Tag color="blue">累计消息 {stats.totalMessages}</Tag>
      <Tag color="geekblue">累计工具 {stats.totalTools}</Tag>
      <Tag color="cyan">平均响应 {averageLabel}</Tag>
      {error && (
        <Text type="danger" className="text-xs">
          错误：{error}
        </Text>
      )}
    </div>
  )
}

export default AgentStatusBar
