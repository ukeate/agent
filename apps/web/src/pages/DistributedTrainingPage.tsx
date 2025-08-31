import React from 'react';
import { Card, Typography, Row, Col, Statistic, Progress, Descriptions } from 'antd';
import { ClusterOutlined, ThunderboltOutlined, DatabaseOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const DistributedTrainingPage: React.FC = () => {
  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ClusterOutlined style={{ marginRight: 8, color: '#13c2c2' }} />
        分布式训练管理
      </Title>
      <Text type="secondary">管理和监控多GPU分布式训练任务</Text>

      <Row gutter={16} style={{ marginTop: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic title="GPU节点" value={4} prefix={<ThunderboltOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="总显存" value="96GB" prefix={<DatabaseOutlined />} />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="同步效率" value={89} suffix="%" />
          </Card>
        </Col>
      </Row>

      <Card title="GPU状态" style={{ marginTop: 16 }}>
        <Row gutter={16}>
          {[1, 2, 3, 4].map(gpu => (
            <Col span={6} key={gpu}>
              <div style={{ border: '1px solid #d9d9d9', padding: '16px', borderRadius: '6px' }}>
                <div style={{ marginBottom: 8 }}>
                  <Text strong>GPU {gpu}</Text>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <Text type="secondary">使用率</Text>
                  <Progress percent={85 + gpu} size="small" style={{ marginTop: 4 }} />
                </div>
                <div>
                  <Text type="secondary">显存: 20.5/24GB</Text>
                </div>
              </div>
            </Col>
          ))}
        </Row>
      </Card>

      <Card title="DeepSpeed配置" style={{ marginTop: 16 }}>
        <Descriptions column={2}>
          <Descriptions.Item label="ZeRO Stage">2</Descriptions.Item>
          <Descriptions.Item label="Offload">CPU</Descriptions.Item>
          <Descriptions.Item label="通信后端">NCCL</Descriptions.Item>
          <Descriptions.Item label="梯度压缩">启用</Descriptions.Item>
        </Descriptions>
      </Card>
    </div>
  );
};

export default DistributedTrainingPage;