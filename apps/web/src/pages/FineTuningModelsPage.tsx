import React from 'react';
import { Card, Table, Tag, Button, Space, Typography, Row, Col, Statistic } from 'antd';
import { RobotOutlined, CheckCircleOutlined, ExclamationCircleOutlined, CloudDownloadOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const FineTuningModelsPage: React.FC = () => {
  const models = [
    { id: '1', name: 'LLaMA 2 7B', size: '7B', status: '支持', type: 'decoder-only' },
    { id: '2', name: 'Mistral 7B', size: '7B', status: '支持', type: 'decoder-only' },
    { id: '3', name: 'Qwen 14B', size: '14B', status: '支持', type: 'decoder-only' },
    { id: '4', name: 'ChatGLM3 6B', size: '6B', status: '支持', type: 'decoder-only' },
  ];

  const columns = [
    { title: '模型名称', dataIndex: 'name', key: 'name' },
    { title: '参数规模', dataIndex: 'size', key: 'size' },
    { title: '架构类型', dataIndex: 'type', key: 'type' },
    { 
      title: '支持状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color={status === '支持' ? 'green' : 'red'} icon={<CheckCircleOutlined />}>
          {status}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button type="primary" size="small" icon={<CloudDownloadOutlined />}>
            下载
          </Button>
          <Button size="small">详情</Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <RobotOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          支持的模型列表
        </Title>
        <Text type="secondary">查看和管理支持微调的预训练模型</Text>
      </div>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic title="支持模型" value={models.length} prefix={<RobotOutlined />} />
          </Card>
        </Col>
      </Row>

      <Card>
        <Table columns={columns} dataSource={models} rowKey="id" />
      </Card>
    </div>
  );
};

export default FineTuningModelsPage;