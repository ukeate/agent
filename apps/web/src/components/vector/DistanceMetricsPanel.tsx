/**
 * 距离度量面板
 * 展示多种距离度量方法和自定义距离函数
 */

import React, { useState } from 'react'
import {
  Card,
  Table,
  Button,
  Select,
  Input,
  Space,
  Alert,
  Tag,
  Progress,
  Row,
  Col,
} from 'antd'
import { CalculatorOutlined, ThunderboltOutlined } from '@ant-design/icons'

const DistanceMetricsPanel: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState('cosine')

  const metrics = [
    {
      name: 'cosine',
      display: '余弦距离',
      type: 'pgvector',
      performance: 95,
      accuracy: 98,
    },
    {
      name: 'euclidean',
      display: '欧氏距离',
      type: 'pgvector',
      performance: 92,
      accuracy: 96,
    },
    {
      name: 'manhattan',
      display: '曼哈顿距离',
      type: 'custom',
      performance: 88,
      accuracy: 94,
    },
    {
      name: 'minkowski',
      display: '闵可夫斯基距离',
      type: 'custom',
      performance: 85,
      accuracy: 95,
    },
    {
      name: 'chebyshev',
      display: '切比雪夫距离',
      type: 'custom',
      performance: 90,
      accuracy: 93,
    },
  ]

  const columns = [
    {
      title: '距离度量',
      dataIndex: 'display',
      key: 'display',
      render: (text: string, record: any) => (
        <Space>
          <span>{text}</span>
          <Tag color={record.type === 'pgvector' ? 'blue' : 'green'}>
            {record.type === 'pgvector' ? '内置' : '扩展'}
          </Tag>
        </Space>
      ),
    },
    {
      title: '性能',
      dataIndex: 'performance',
      key: 'performance',
      render: (value: number) => (
        <Progress percent={value} size="small" strokeColor="#52c41a" />
      ),
    },
    {
      title: '精度',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (value: number) => (
        <Progress percent={value} size="small" strokeColor="#1890ff" />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: any) => (
        <Button size="small" onClick={() => setSelectedMetric(record.name)}>
          测试
        </Button>
      ),
    },
  ]

  return (
    <div>
      <Alert
        message="距离度量方法"
        description="支持pgvector内置的余弦、欧氏距离等，以及自定义的闵可夫斯基、切比雪夫等距离度量，可进行性能基准测试。"
        type="info"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        <Col span={12}>
          <Card title="距离度量列表" size="small">
            <Table
              columns={columns}
              dataSource={metrics}
              rowKey="name"
              size="small"
              pagination={false}
            />
          </Card>
        </Col>

        <Col span={12}>
          <Card title="自定义距离函数" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input placeholder="函数名称" />
              <Input.TextArea placeholder="输入自定义距离函数代码" rows={6} />
              <Button type="primary" block>
                创建自定义函数
              </Button>
            </Space>
          </Card>

          <Card title="性能基准测试" size="small" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Select
                value={selectedMetric}
                onChange={setSelectedMetric}
                style={{ width: '100%' }}
                placeholder="选择距离度量"
              >
                {metrics.map(m => (
                  <Select.Option key={m.name} value={m.name}>
                    {m.display}
                  </Select.Option>
                ))}
              </Select>
              <Button type="primary" block>
                运行基准测试
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default DistanceMetricsPanel
