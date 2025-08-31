import React from 'react';
import { Card, Upload, Button, Table, Tag, Typography, Row, Col, Statistic } from 'antd';
import { FileImageOutlined, UploadOutlined, CheckCircleOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const FineTuningDatasetsPage: React.FC = () => {
  const datasets = [
    { id: '1', name: '对话数据集', size: '1.2GB', samples: 50000, format: 'jsonl', status: '已验证' },
    { id: '2', name: '代码生成数据', size: '800MB', samples: 25000, format: 'json', status: '处理中' },
  ];

  const columns = [
    { title: '数据集名称', dataIndex: 'name', key: 'name' },
    { title: '大小', dataIndex: 'size', key: 'size' },
    { title: '样本数', dataIndex: 'samples', key: 'samples' },
    { title: '格式', dataIndex: 'format', key: 'format' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color={status === '已验证' ? 'green' : 'processing'}>
          {status}
        </Tag>
      )
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <FileImageOutlined style={{ marginRight: 8, color: '#52c41a' }} />
          数据集管理
        </Title>
        <Text type="secondary">上传、验证和管理训练数据集</Text>
      </div>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic title="数据集总数" value={datasets.length} prefix={<FileImageOutlined />} />
          </Card>
        </Col>
      </Row>

      <Card title="上传数据集" style={{ marginBottom: 16 }}>
        <Upload>
          <Button icon={<UploadOutlined />}>选择文件</Button>
        </Upload>
      </Card>

      <Card title="数据集列表">
        <Table columns={columns} dataSource={datasets} rowKey="id" />
      </Card>
    </div>
  );
};

export default FineTuningDatasetsPage;