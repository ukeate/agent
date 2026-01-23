import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert } from 'antd'
import { ReloadOutlined, UserOutlined } from '@ant-design/icons'

type UserProfile = {
  user_id: string
  feedback_count?: number
  avg_score?: number
  preferences?: string[]
}

const UserFeedbackProfilesPage: React.FC = () => {
  const [profiles, setProfiles] = useState<UserProfile[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/feedback/profiles'))
      const data = await res.json()
      setProfiles(Array.isArray(data?.profiles) ? data.profiles : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setProfiles([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const columns = [
    { title: '用户', dataIndex: 'user_id' },
    { title: '反馈数', dataIndex: 'feedback_count' },
    { title: '平均得分', dataIndex: 'avg_score' },
    {
      title: '偏好',
      dataIndex: 'preferences',
      render: (prefs: string[]) =>
        (prefs || []).map(p => <Tag key={p}>{p}</Tag>),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <UserOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              用户反馈画像
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Card>
          <Table
            rowKey="user_id"
            dataSource={profiles}
            columns={columns}
            loading={loading}
          />
        </Card>
      </Space>
    </div>
  )
}

export default UserFeedbackProfilesPage
