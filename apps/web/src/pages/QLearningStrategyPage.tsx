import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Alert,
  Typography,
  List,
  Tag,
  Timeline,
  message,
} from 'antd'
import {
  BulbOutlined,
  ReloadOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

const { Title, Text, Paragraph } = Typography

interface QModel {
  id: string
  name: string
  status: string
  performance?: number
  episodes?: number
}

const QLearningStrategyPage: React.FC = () => {
  const [models, setModels] = useState<QModel[]>([])
  const [trend, setTrend] = useState<
    Array<{ episode: number; reward: number }>
  >([])
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const resp = await apiClient.get('/tensorflow-qlearning/overview')
      const data: any = resp.data || {}
      const mapped: QModel[] = Array.isArray(data.models)
        ? data.models.map((m: any) => ({
            id: m.id || m.name,
            name: m.name || m.id,
            status: m.status || 'unknown',
            performance: m.performance,
            episodes: m.episodes,
          }))
        : []
      setModels(mapped)
      setTrend(Array.isArray(data.trend) ? data.trend : [])
      setUpdatedAt(data.updated_at || null)
    } catch (err) {
      message.error('加载策略推理数据失败')
      setModels([])
      setTrend([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BulbOutlined /> Q-Learning策略推理
      </Title>
      <Paragraph type="secondary">
        展示后端提供的Q-Learning模型与训练趋势，移除本地模拟数据，等待真实推理API对接。
      </Paragraph>
      <Space style={{ marginBottom: 16 }}>
        <Button icon={<ReloadOutlined />} onClick={fetchData} loading={loading}>
          刷新数据
        </Button>
        {updatedAt && <Text type="secondary">最近更新: {updatedAt}</Text>}
      </Space>

      <Alert
        message="策略推理提示"
        description="当前页面仅展示真实模型与训练趋势，推理/批量推理需接入后端推理接口。"
        type="info"
        showIcon
        style={{ marginBottom: 16 }}
      />

      <Card title="可用模型" extra={<ThunderboltOutlined />}>
        {models.length ? (
          <List
            dataSource={models}
            renderItem={item => (
              <List.Item>
                <Space direction="vertical" size={2}>
                  <Space>
                    <Text strong>{item.name}</Text>
                    <Tag
                      color={
                        item.status === 'running' || item.status === 'training'
                          ? 'blue'
                          : 'default'
                      }
                    >
                      {item.status}
                    </Tag>
                  </Space>
                  <Text type="secondary">
                    Episodes: {item.episodes || 0} | Performance:{' '}
                    {item.performance ?? 0}
                  </Text>
                </Space>
              </List.Item>
            )}
          />
        ) : (
          <Alert message="暂无模型数据" type="info" showIcon />
        )}
      </Card>

      <Card
        title="训练趋势"
        style={{ marginTop: 16 }}
        extra={<ExperimentOutlined />}
      >
        {trend.length ? (
          <Timeline
            items={trend.slice(-5).map(item => ({
              children: `Episode ${item.episode}: 奖励 ${item.reward}`,
              color: item.reward >= 0 ? 'green' : 'red',
            }))}
          />
        ) : (
          <Alert message="暂无训练趋势数据" type="info" showIcon />
        )}
      </Card>
    </div>
  )
}

export default QLearningStrategyPage
