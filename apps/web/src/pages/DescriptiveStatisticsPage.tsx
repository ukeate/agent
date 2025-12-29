import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Input, Button, Statistic, Typography, message } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

const { Title } = Typography

interface StatsResponse {
  mean: number
  median: number
  mode: number
  variance: number
  std_dev: number
  min: number
  max: number
  quartiles: [number, number, number]
  sample_size: number
}

const DescriptiveStatisticsPage: React.FC = () => {
  const [dataInput, setDataInput] = useState('1,2,3,4,5')
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const computeStats = async () => {
    setLoading(true)
    try {
      const numbers = dataInput
        .split(',')
        .map(n => n.trim())
        .filter(n => n.length)
        .map(Number)
      const res = await apiFetch(buildApiUrl('/api/v1/statistical-analysis/descriptive'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: numbers })
      })
      const resp = await res.json()
      setStats(resp)
    } catch (e: any) {
      message.error(e?.message || '统计计算失败')
      setStats(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    computeStats()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Title level={3}>描述性统计</Title>
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={8}>
          <Col flex="auto">
            <Input
              value={dataInput}
              onChange={e => setDataInput(e.target.value)}
              placeholder="以逗号分隔的数值，例如 1,2,3"
            />
          </Col>
          <Col>
            <Button type="primary" icon={<ReloadOutlined />} loading={loading} onClick={computeStats}>
              计算
            </Button>
          </Col>
        </Row>
      </Card>

      <Card>
        {stats ? (
          <Row gutter={16}>
            <Col span={6}><Statistic title="均值" value={stats.mean} /></Col>
            <Col span={6}><Statistic title="中位数" value={stats.median} /></Col>
            <Col span={6}><Statistic title="众数" value={stats.mode} /></Col>
            <Col span={6}><Statistic title="样本量" value={stats.sample_size} /></Col>
            <Col span={6}><Statistic title="方差" value={stats.variance} precision={4} /></Col>
            <Col span={6}><Statistic title="标准差" value={stats.std_dev} precision={4} /></Col>
            <Col span={6}><Statistic title="最小值" value={stats.min} /></Col>
            <Col span={6}><Statistic title="最大值" value={stats.max} /></Col>
          </Row>
        ) : (
          <div style={{ color: '#888' }}>暂无结果</div>
        )}
      </Card>
    </div>
  )
}

export default DescriptiveStatisticsPage
