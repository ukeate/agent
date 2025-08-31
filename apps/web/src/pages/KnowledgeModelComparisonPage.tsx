import React from 'react'
import { Card, Table, Progress, Typography, Tag } from 'antd'
import { CompressOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeModelComparisonPage: React.FC = () => {
  const modelData = [
    {
      key: '1',
      name: 'BERT-Large',
      type: '实体识别',
      accuracy: 96.2,
      speed: 450,
      size: '1.2GB'
    },
    {
      key: '2',
      name: 'spaCy-zh',
      type: '实体识别',
      accuracy: 94.5,
      speed: 1200,
      size: '50MB'
    }
  ]

  const columns = [
    { title: '模型名称', dataIndex: 'name', key: 'name' },
    { 
      title: '类型', 
      dataIndex: 'type', 
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    { 
      title: '准确率', 
      dataIndex: 'accuracy', 
      key: 'accuracy',
      render: (accuracy: number) => (
        <Progress percent={accuracy} size="small" format={percent => `${percent}%`} />
      )
    },
    { title: '速度 (tokens/s)', dataIndex: 'speed', key: 'speed' },
    { title: '模型大小', dataIndex: 'size', key: 'size' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <CompressOutlined style={{ marginRight: 8 }} />
          模型对比
        </Title>
        <Paragraph type="secondary">
          对比不同知识抽取模型的性能和准确率
        </Paragraph>
      </div>

      <Card title="模型对比表">
        <Table columns={columns} dataSource={modelData} />
      </Card>
    </div>
  )
}

export default KnowledgeModelComparisonPage