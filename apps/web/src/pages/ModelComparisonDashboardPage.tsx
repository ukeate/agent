import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, message } from 'antd'
import { ReloadOutlined, CompareOutlined } from '@ant-design/icons'
import { modelEvaluationService } from '../services/modelEvaluationService'

interface ComparisonRow {
  model: string
  benchmark: string
  score: number
  rank?: number
}

const ModelComparisonDashboardPage: React.FC = () => {
  const [rows, setRows] = useState<ComparisonRow[]>([])
  const [models, setModels] = useState<any[]>([])
  const [benchmarks, setBenchmarks] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [m, b, comp] = await Promise.all([
        modelEvaluationService.listModels(),
        modelEvaluationService.listBenchmarks(),
        modelEvaluationService.getPerformanceComparison(),
      ])
      setModels(m || [])
      setBenchmarks(b || [])
      const list: ComparisonRow[] = (comp || []).flatMap(
        (c: any) => c.comparison_matrix || []
      )
      setRows(list)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setModels([])
      setBenchmarks([])
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <CompareOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              模型对比仪表
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Typography.Text type="danger">{error}</Typography.Text>}

        <Card title="模型列表">
          <Tag color="blue">数量 {models.length}</Tag>
        </Card>

        <Card title="基准测试">
          <Tag color="cyan">数量 {benchmarks.length}</Tag>
        </Card>

        <Card title="对比结果">
          <Table
            rowKey={r => `${r.model}-${r.benchmark}`}
            dataSource={rows}
            loading={loading}
            locale={{ emptyText: '暂无数据' }}
            columns={[
              { title: '模型', dataIndex: 'model' },
              { title: '基准', dataIndex: 'benchmark' },
              { title: '得分', dataIndex: 'score' },
              { title: '排名', dataIndex: 'rank' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default ModelComparisonDashboardPage
