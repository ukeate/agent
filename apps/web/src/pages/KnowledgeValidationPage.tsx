import React from 'react'
import { Card, Table, Tag, Button, Space, Typography, Progress } from 'antd'
import { CheckCircleOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeValidationPage: React.FC = () => {
  const data = [
    {
      key: '1',
      entity: '苹果公司',
      type: '组织',
      confidence: 95,
      status: '已验证'
    }
  ]

  const columns = [
    { title: '实体', dataIndex: 'entity', key: 'entity' },
    { title: '类型', dataIndex: 'type', key: 'type' },
    { 
      title: '置信度', 
      dataIndex: 'confidence', 
      key: 'confidence',
      render: (confidence: number) => <Progress percent={confidence} size="small" />
    },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color="success">{status}</Tag>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button size="small" type="link">验证</Button>
          <Button size="small" type="link">编辑</Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <CheckCircleOutlined style={{ marginRight: 8 }} />
          知识验证
        </Title>
        <Paragraph type="secondary">
          验证和评估知识图谱中的实体和关系的准确性
        </Paragraph>
      </div>

      <Card title="知识验证列表">
        <Table columns={columns} dataSource={data} />
      </Card>
    </div>
  )
}

export default KnowledgeValidationPage