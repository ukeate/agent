import React from 'react'
import { Card, Table, Tag, Typography, Timeline } from 'antd'
import { ExclamationCircleOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeErrorAnalysisPage: React.FC = () => {
  const errorData = [
    {
      key: '1',
      type: '实体识别错误',
      description: '误将“苹果”识别为水果而非公司',
      severity: '中等',
      count: 15
    }
  ]

  const columns = [
    { title: '错误类型', dataIndex: 'type', key: 'type' },
    { title: '描述', dataIndex: 'description', key: 'description' },
    { 
      title: '严重性', 
      dataIndex: 'severity', 
      key: 'severity',
      render: (severity: string) => (
        <Tag color={severity === '高' ? 'red' : severity === '中等' ? 'orange' : 'green'}>
          {severity}
        </Tag>
      )
    },
    { title: '频次', dataIndex: 'count', key: 'count' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ExclamationCircleOutlined style={{ marginRight: 8 }} />
          错误分析
        </Title>
        <Paragraph type="secondary">
          分析和识别知识抽取过程中的错误和问题
        </Paragraph>
      </div>

      <Card title="错误统计" style={{ marginBottom: 16 }}>
        <Table columns={columns} dataSource={errorData} />
      </Card>

      <Card title="错误日志">
        <Timeline>
          <Timeline.Item color="red">
            2024-01-20 11:30 - 实体识别错误：误将苹果识别为水果
          </Timeline.Item>
          <Timeline.Item color="orange">
            2024-01-20 10:45 - 关系抽取错误：关系类型错误
          </Timeline.Item>
        </Timeline>
      </Card>
    </div>
  )
}

export default KnowledgeErrorAnalysisPage