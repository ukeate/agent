/**
 * 索引管理面板
 * 
 * 展示多种索引类型的管理和优化功能：
 * - HNSW索引参数调优
 * - IVF索引配置
 * - LSH哈希索引设置
 * - 自适应索引选择
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Form,
  Select,
  Slider,
  Button,
  Table,
  Tag,
  Space,
  Progress,
  Alert,
  Statistic,
  message,
  Tooltip,
  Typography
} from 'antd';
import {
  ThunderboltOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Option } = Select;
const { Text, Title } = Typography;

interface IndexConfig {
  type: 'hnsw' | 'ivf' | 'lsh' | 'flat';
  name: string;
  table: string;
  column: string;
  parameters: Record<string, any>;
  status: 'active' | 'building' | 'error' | 'optimizing';
  performance: {
    latency_p50: number;
    latency_p95: number;
    recall: number;
    memory_usage: number;
  };
}

const IndexManagerPanel: React.FC = () => {
  const [indexes, setIndexes] = useState<IndexConfig[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  useEffect(() => {
    loadIndexes();
  }, []);

  const loadIndexes = async () => {
    setLoading(true);
    try {
      // Mock data - 在实际应用中调用相应的API
      const mockIndexes: IndexConfig[] = [
        {
          type: 'hnsw',
          name: 'embeddings_hnsw_idx',
          table: 'documents',
          column: 'embedding',
          parameters: { m: 16, ef_construction: 200, ef_search: 100 },
          status: 'active',
          performance: { latency_p50: 0.8, latency_p95: 2.1, recall: 0.95, memory_usage: 245 }
        },
        {
          type: 'ivf',
          name: 'embeddings_ivf_idx',
          table: 'documents',
          column: 'embedding',
          parameters: { lists: 100, probes: 10 },
          status: 'active',
          performance: { latency_p50: 1.2, latency_p95: 3.5, recall: 0.92, memory_usage: 180 }
        },
        {
          type: 'lsh',
          name: 'embeddings_lsh_idx',
          table: 'documents',
          column: 'embedding',
          parameters: { n_tables: 8, n_bits: 12 },
          status: 'building',
          performance: { latency_p50: 0.5, latency_p95: 1.8, recall: 0.88, memory_usage: 120 }
        }
      ];
      setIndexes(mockIndexes);
    } catch (error) {
      message.error('加载索引信息失败');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateIndex = async (values: any) => {
    setLoading(true);
    try {
      // Mock API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      message.success('索引创建任务已提交');
      loadIndexes();
    } catch (error) {
      message.error('创建索引失败');
    } finally {
      setLoading(false);
    }
  };

  const handleOptimizeIndex = async (indexName: string) => {
    setLoading(true);
    try {
      // Mock API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      message.success(`索引 ${indexName} 优化完成`);
      loadIndexes();
    } catch (error) {
      message.error('优化索引失败');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'building': return 'processing';
      case 'error': return 'error';
      case 'optimizing': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircleOutlined />;
      case 'building': return <PlayCircleOutlined />;
      case 'error': return <ExclamationCircleOutlined />;
      case 'optimizing': return <SettingOutlined spin />;
      default: return <InfoCircleOutlined />;
    }
  };

  const renderHNSWParameters = () => (
    <Card size="small" title="HNSW 参数配置">
      <Form.Item label="M (连接数)" name={['parameters', 'm']}>
        <Slider min={4} max={64} step={4} marks={{ 16: '16', 32: '32', 48: '48' }} />
      </Form.Item>
      <Form.Item label="ef_construction" name={['parameters', 'ef_construction']}>
        <Slider min={100} max={800} step={50} marks={{ 200: '200', 400: '400', 600: '600' }} />
      </Form.Item>
      <Form.Item label="ef_search" name={['parameters', 'ef_search']}>
        <Slider min={50} max={400} step={25} marks={{ 100: '100', 200: '200', 300: '300' }} />
      </Form.Item>
    </Card>
  );

  const renderIVFParameters = () => (
    <Card size="small" title="IVF 参数配置">
      <Form.Item label="Lists (聚类数)" name={['parameters', 'lists']}>
        <Slider min={50} max={500} step={25} marks={{ 100: '100', 200: '200', 300: '300' }} />
      </Form.Item>
      <Form.Item label="Probes (探测数)" name={['parameters', 'probes']}>
        <Slider min={5} max={50} step={5} marks={{ 10: '10', 20: '20', 30: '30' }} />
      </Form.Item>
    </Card>
  );

  const renderLSHParameters = () => (
    <Card size="small" title="LSH 参数配置">
      <Form.Item label="Hash Tables" name={['parameters', 'n_tables']}>
        <Slider min={4} max={16} step={2} marks={{ 8: '8', 12: '12', 16: '16' }} />
      </Form.Item>
      <Form.Item label="Hash Bits" name={['parameters', 'n_bits']}>
        <Slider min={8} max={20} step={2} marks={{ 12: '12', 16: '16', 20: '20' }} />
      </Form.Item>
    </Card>
  );

  const columns = [
    {
      title: '索引名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: IndexConfig) => (
        <Space>
          <Text strong>{name}</Text>
          <Tag color={getStatusColor(record.status)}>
            {getStatusIcon(record.status)}
            {record.status.toUpperCase()}
          </Tag>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'hnsw' ? 'blue' : type === 'ivf' ? 'green' : 'orange'}>
          {type.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '表/列',
      key: 'table_column',
      render: (record: IndexConfig) => (
        <Text>{record.table}.{record.column}</Text>
      ),
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (record: IndexConfig) => (
        <Space direction="vertical" size="small">
          <Text type="secondary">P50: {record.performance.latency_p50}ms</Text>
          <Text type="secondary">召回: {(record.performance.recall * 100).toFixed(1)}%</Text>
        </Space>
      ),
    },
    {
      title: '内存使用',
      key: 'memory',
      render: (record: IndexConfig) => (
        <Statistic
          value={record.performance.memory_usage}
          suffix="MB"
          valueStyle={{ fontSize: 14 }}
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: IndexConfig) => (
        <Space>
          <Button 
            size="small" 
            onClick={() => handleOptimizeIndex(record.name)}
            disabled={record.status !== 'active'}
            loading={loading}
          >
            优化
          </Button>
          <Button size="small" type="link">详情</Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      {/* 功能说明 */}
      <Alert
        message="索引管理功能"
        description="管理和优化多种向量索引类型，包括HNSW图索引、IVF倒排索引和LSH哈希索引。可以实时调整参数并监控性能指标。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 左侧：创建索引 */}
        <Col span={8}>
          <Card title="创建新索引" size="small">
            <Form
              form={form}
              layout="vertical"
              onFinish={handleCreateIndex}
              initialValues={{
                type: 'hnsw',
                parameters: { m: 16, ef_construction: 200, ef_search: 100 }
              }}
            >
              <Form.Item label="索引类型" name="type">
                <Select>
                  <Option value="hnsw">
                    <Space>
                      <ThunderboltOutlined />
                      HNSW (推荐)
                    </Space>
                  </Option>
                  <Option value="ivf">IVF (倒排)</Option>
                  <Option value="lsh">LSH (哈希)</Option>
                  <Option value="flat">FLAT (暴力)</Option>
                </Select>
              </Form.Item>

              <Form.Item label="表名" name="table">
                <Select placeholder="选择表">
                  <Option value="documents">documents</Option>
                  <Option value="images">images</Option>
                  <Option value="audio">audio</Option>
                </Select>
              </Form.Item>

              <Form.Item label="向量列" name="column">
                <Select placeholder="选择向量列">
                  <Option value="embedding">embedding</Option>
                  <Option value="vector">vector</Option>
                </Select>
              </Form.Item>

              {/* 动态参数配置 */}
              <Form.Item shouldUpdate={(prev, curr) => prev.type !== curr.type}>
                {({ getFieldValue }) => {
                  const indexType = getFieldValue('type');
                  switch (indexType) {
                    case 'hnsw': return renderHNSWParameters();
                    case 'ivf': return renderIVFParameters();
                    case 'lsh': return renderLSHParameters();
                    default: return null;
                  }
                }}
              </Form.Item>

              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} block>
                  创建索引
                </Button>
              </Form.Item>
            </Form>
          </Card>

          {/* 性能比较 */}
          <Card title="索引性能对比" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="HNSW"
                  value="0.8"
                  suffix="ms"
                  valueStyle={{ color: '#3f8600' }}
                />
                <Progress percent={95} size="small" strokeColor="#52c41a" />
              </Col>
              <Col span={8}>
                <Statistic
                  title="IVF"
                  value="1.2"
                  suffix="ms"
                  valueStyle={{ color: '#1890ff' }}
                />
                <Progress percent={92} size="small" strokeColor="#1890ff" />
              </Col>
              <Col span={8}>
                <Statistic
                  title="LSH"
                  value="0.5"
                  suffix="ms"
                  valueStyle={{ color: '#fa8c16' }}
                />
                <Progress percent={88} size="small" strokeColor="#fa8c16" />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 右侧：索引列表 */}
        <Col span={16}>
          <Card 
            title="现有索引" 
            size="small"
            extra={
              <Button onClick={loadIndexes} loading={loading}>
                刷新
              </Button>
            }
          >
            <Table
              columns={columns}
              dataSource={indexes}
              rowKey="name"
              size="small"
              pagination={false}
              loading={loading}
            />
          </Card>

          {/* 自适应索引选择建议 */}
          <Card title="智能索引建议" size="small" style={{ marginTop: 16 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f6ffed' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>高精度场景</Text>
                    <Text type="secondary">推荐 HNSW</Text>
                    <Text type="secondary">召回率 &gt; 95%</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#f0f9ff' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>大数据场景</Text>
                    <Text type="secondary">推荐 IVF</Text>
                    <Text type="secondary">平衡性能与内存</Text>
                  </Space>
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small" style={{ backgroundColor: '#fff7e6' }}>
                  <Space direction="vertical" size="small">
                    <Text strong>低延迟场景</Text>
                    <Text type="secondary">推荐 LSH</Text>
                    <Text type="secondary">亚毫秒响应</Text>
                  </Space>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default IndexManagerPanel;