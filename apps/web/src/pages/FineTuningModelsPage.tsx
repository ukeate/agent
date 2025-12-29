import React, { useEffect, useMemo, useState } from 'react';
import { Card, Table, Tag, Button, Space, Typography, Row, Col, Statistic, message } from 'antd';
import { RobotOutlined, ReloadOutlined } from '@ant-design/icons';
import { fineTuningService, ModelInfo } from '../services/fineTuningService';

import { logger } from '../utils/logger'
const { Title, Text } = Typography;

type ModelRow = {
  key: string;
  name: string;
  architecture: string;
  max_seq_length: number;
};

const FineTuningModelsPage: React.FC = () => {
  const [supportedModels, setSupportedModels] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      setSupportedModels(await fineTuningService.getSupportedModels());
    } catch (e) {
      logger.error('加载模型列表失败:', e);
      message.error('加载模型列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const rows: ModelRow[] = useMemo(() => {
    if (!supportedModels?.models?.length) return [];
    return supportedModels.models.flatMap(group =>
      group.models.map(name => ({
        key: `${group.architecture}:${name}`,
        name,
        architecture: group.architecture,
        max_seq_length: group.max_seq_length,
      }))
    );
  }, [supportedModels]);

  const columns = [
    { title: '模型', dataIndex: 'name', key: 'name' },
    { title: '架构', dataIndex: 'architecture', key: 'architecture', render: (v: string) => <Tag>{v}</Tag> },
    { title: '最大序列长度', dataIndex: 'max_seq_length', key: 'max_seq_length' },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <RobotOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          支持的模型列表
        </Title>
        <Text type="secondary">展示本机已缓存且可识别架构的模型；也可在创建任务时直接输入HuggingFace模型名</Text>
      </div>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic title="模型数量" value={rows.length} prefix={<RobotOutlined />} />
          </Card>
        </Col>
        <Col span={18}>
          <Card>
            <Space wrap>
              <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
                刷新
              </Button>
              {supportedModels?.architectures?.map(a => (
                <Tag key={a}>{a}</Tag>
              )) || null}
            </Space>
          </Card>
        </Col>
      </Row>

      <Card>
        <Table columns={columns} dataSource={rows} loading={loading} />
      </Card>
    </div>
  );
};

export default FineTuningModelsPage;
