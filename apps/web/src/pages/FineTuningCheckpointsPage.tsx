import React from 'react';
import { Card, Table, Tag, Button, Space, Typography, Row, Col, Statistic } from 'antd';
import { DatabaseOutlined, DownloadOutlined, DeleteOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const FineTuningCheckpointsPage: React.FC = () => {
  const checkpoints = [
    { id: '1', name: 'llama2-lora-epoch-3', epoch: 3, loss: 0.8234, size: '45MB', time: '2025-08-23 16:30' },
    { id: '2', name: 'mistral-qlora-epoch-2', epoch: 2, loss: 0.9123, size: '32MB', time: '2025-08-23 15:45' },
  ];

  const columns = [
    { title: '检查点名称', dataIndex: 'name', key: 'name' },
    { title: '轮次', dataIndex: 'epoch', key: 'epoch' },
    { title: '损失', dataIndex: 'loss', key: 'loss' },
    { title: '大小', dataIndex: 'size', key: 'size' },
    { title: '创建时间', dataIndex: 'time', key: 'time' },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button type="primary" size="small" icon={<DownloadOutlined />}>下载</Button>
          <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <DatabaseOutlined style={{ marginRight: 8, color: '#722ed1' }} />
        模型检查点管理
      </Title>
      <Text type="secondary">管理训练过程中保存的模型检查点</Text>

      <Row gutter={16} style={{ marginTop: 24, marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="检查点总数" value={checkpoints.length} prefix={<DatabaseOutlined />} />
          </Card>
        </Col>
      </Row>

      <Card>
        <Table columns={columns} dataSource={checkpoints} rowKey="id" />
      </Card>
    </div>
  );
};

export default FineTuningCheckpointsPage;