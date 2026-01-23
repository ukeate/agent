import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Button, Card, Space, Table, Tag, Typography, message } from 'antd'
import { ReloadOutlined, TrophyOutlined } from '@ant-design/icons'

import { logger } from '../utils/logger'
const { Title, Text } = Typography

interface BenchmarkSuite {
  name: string
  description: string
  total_tasks: number
  estimated_runtime_hours: number
  benchmarks: {
    name: string
    display_name: string
    type: string
    difficulty: string
  }[]
}

const BenchmarkSuiteManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [suites, setSuites] = useState<BenchmarkSuite[]>([])

  const loadSuites = async () => {
    try {
      setLoading(true)
      const res = await apiFetch(
        buildApiUrl('/api/v1/model-evaluation/benchmark-suites')
      )
      const data = await res.json()
      setSuites(Array.isArray(data?.suites) ? data.suites : [])
    } catch (e) {
      logger.error('加载基准测试套件失败:', e)
      message.error('加载基准测试套件失败')
      setSuites([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSuites()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16,
        }}
      >
        <div>
          <Title level={2} style={{ margin: 0 }}>
            <TrophyOutlined style={{ marginRight: 12 }} />
            基准测试套件
          </Title>
          <Text type="secondary">
            数据来自 /api/v1/model-evaluation/benchmark-suites
          </Text>
        </div>
        <Space>
          <Button
            onClick={loadSuites}
            icon={<ReloadOutlined />}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
      </div>

      <Card>
        <Table
          rowKey="name"
          loading={loading}
          dataSource={suites}
          pagination={false}
          columns={[
            { title: '套件', dataIndex: 'name' },
            { title: '描述', dataIndex: 'description' },
            { title: '任务数', dataIndex: 'total_tasks' },
            {
              title: '预计时长(小时)',
              dataIndex: 'estimated_runtime_hours',
              render: (v: any) => Number(v).toFixed(1),
            },
            {
              title: '包含基准测试',
              dataIndex: 'benchmarks',
              render: (benchmarks: BenchmarkSuite['benchmarks']) => (
                <Space wrap>
                  {(benchmarks || []).slice(0, 6).map(b => (
                    <Tag key={b.name} title={`${b.type} / ${b.difficulty}`}>
                      {b.display_name || b.name}
                    </Tag>
                  ))}
                  {(benchmarks || []).length > 6 ? (
                    <Tag>+{benchmarks.length - 6}</Tag>
                  ) : null}
                </Space>
              ),
            },
          ]}
          expandable={{
            expandedRowRender: record => (
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(record, null, 2)}
              </pre>
            ),
          }}
        />
      </Card>
    </div>
  )
}

export default BenchmarkSuiteManagementPage
