import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Input,
  Button,
  Statistic,
  Typography,
  message,
} from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import statisticalAnalysisService, {
  DescriptiveStats,
} from '../services/statisticalAnalysisService'

const { Title } = Typography

const DescriptiveStatisticsPage: React.FC = () => {
  const [dataInput, setDataInput] = useState('1,2,3,4,5')
  const [stats, setStats] = useState<DescriptiveStats | null>(null)
  const [loading, setLoading] = useState(false)

  const computeStats = async () => {
    setLoading(true)
    try {
      const numbers = dataInput
        .split(',')
        .map(n => n.trim())
        .filter(n => n.length)
        .map(Number)
      if (!numbers.length || numbers.some(value => Number.isNaN(value))) {
        throw new Error('请输入有效的数值列表')
      }
      const resp = await statisticalAnalysisService.getDescriptiveStats(numbers)
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
            <Button
              type="primary"
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={computeStats}
            >
              计算
            </Button>
          </Col>
        </Row>
      </Card>

      <Card>
        {stats ? (
          <Row gutter={16}>
            <Col span={6}>
              <Statistic title="均值" value={stats.mean} />
            </Col>
            <Col span={6}>
              <Statistic title="中位数" value={stats.median} />
            </Col>
            <Col span={6}>
              <Statistic title="样本量" value={stats.count} />
            </Col>
            <Col span={6}>
              <Statistic title="方差" value={stats.variance} precision={4} />
            </Col>
            <Col span={6}>
              <Statistic title="标准差" value={stats.std_dev} precision={4} />
            </Col>
            <Col span={6}>
              <Statistic title="最小值" value={stats.min_value} />
            </Col>
            <Col span={6}>
              <Statistic title="最大值" value={stats.max_value} />
            </Col>
            <Col span={6}>
              <Statistic title="25分位" value={stats.q25} />
            </Col>
            <Col span={6}>
              <Statistic title="75分位" value={stats.q75} />
            </Col>
          </Row>
        ) : (
          <div style={{ color: '#888' }}>暂无结果</div>
        )}
      </Card>
    </div>
  )
}

export default DescriptiveStatisticsPage
