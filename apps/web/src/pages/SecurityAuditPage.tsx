import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Alert,
  Tag,
  Statistic,
  message,
} from 'antd'
import { ReloadOutlined, SecurityScanOutlined } from '@ant-design/icons'
import authService from '../services/authService'

type AuditLog = {
  id?: string
  timestamp?: string
  user_id?: string
  resource?: string
  event_type?: string
  result?: string
  details?: any
}

const SecurityAuditPage: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(false)

  const loadLogs = async () => {
    setLoading(true)
    try {
      const token = authService.getToken()
      if (!token) {
        setLogs([])
        return
      }
      const res = await apiFetch(
        buildApiUrl('/api/v1/security/mcp-tools/audit?limit=200'),
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      )
      const data = await res.json()
      setLogs(Array.isArray(data?.logs) ? data.logs : [])
    } catch (e: any) {
      message.error(e?.message || '加载审计日志失败')
      setLogs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadLogs()
  }, [])

  const columns = [
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp' },
    { title: '用户', dataIndex: 'user_id', key: 'user_id' },
    { title: '事件', dataIndex: 'event_type', key: 'event_type' },
    { title: '资源', dataIndex: 'resource', key: 'resource' },
    {
      title: '结果',
      dataIndex: 'result',
      key: 'result',
      render: (val: string) => {
        const color =
          val === 'success' ? 'green' : val === 'failure' ? 'red' : 'orange'
        return <Tag color={color}>{val || 'unknown'}</Tag>
      },
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <SecurityScanOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              安全审计
            </Typography.Title>
          </Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadLogs}
            loading={loading}
          >
            刷新
          </Button>
        </Space>

        <Card>
          <Space size="large">
            <Statistic title="日志条数" value={logs.length} />
            <Statistic
              title="失败次数"
              value={logs.filter(l => l.result === 'failure').length}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Space>
        </Card>

        <Card title="审计日志">
          {logs.length === 0 ? (
            <Alert
              type="info"
              showIcon
              message="暂无日志，请先触发真实的 MCP 工具调用以生成审计数据。"
            />
          ) : (
            <Table
              dataSource={logs.map((log, idx) => ({
                ...log,
                key: log.id || idx,
              }))}
              columns={columns}
              loading={loading}
              pagination={{ pageSize: 20 }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default SecurityAuditPage
