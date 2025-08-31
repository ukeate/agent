import React from 'react'
import { Card, Table, Tag, Button, Space, Typography } from 'antd'
import { DatabaseOutlined, PlayCircleOutlined, PauseCircleOutlined } from '@ant-design/icons'

const { Title, Paragraph } = Typography

const KnowledgeBatchJobsPage: React.FC = () => {
  const data = [
    {
      key: '1',
      name: '新闻文档批处理',
      status: 'running',
      progress: 68,
      documents: 5000
    }
  ]

  const columns = [
    { title: '任务名称', dataIndex: 'name', key: 'name' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'processing' : 'success'}>
          {status === 'running' ? '运行中' : '已完成'}
        </Tag>
      )
    },
    { title: '进度', dataIndex: 'progress', key: 'progress', render: (progress: number) => `${progress}%` },
    { title: '文档数', dataIndex: 'documents', key: 'documents' },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button size="small" icon={<PauseCircleOutlined />}>暂停</Button>
          <Button size="small" type="link">详情</Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: 8 }} />
          批处理作业管理
        </Title>
        <Paragraph type="secondary">
          管理知识图谱的批量处理任务
        </Paragraph>
      </div>

      <Card title="批处理任务" extra={<Button type="primary" icon={<PlayCircleOutlined />}>新建任务</Button>}>
        <Table columns={columns} dataSource={data} />
      </Card>
    </div>
  )
}

export default KnowledgeBatchJobsPage