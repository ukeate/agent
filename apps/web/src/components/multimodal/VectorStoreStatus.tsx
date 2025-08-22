/**
 * 向量存储状态展示组件
 * 展示Chroma向量数据库的技术状态和索引信息
 */

import React from 'react';
import {
  Card,
  Statistic,
  Row,
  Col,
  Progress,
  Tag,
  Space,
  Typography,
  Table,
  Alert
} from 'antd';
import {
  DatabaseOutlined,
  FileTextOutlined,
  FileImageOutlined,
  TableOutlined,
  ClusterOutlined,
  ThunderboltOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Text, Title } = Typography;

interface VectorStoreStats {
  totalDocuments: number;
  textChunks: number;
  images: number;
  tables: number;
  embeddingDimension: number;
  cacheHitRate: number;
}

interface CollectionInfo {
  name: string;
  count: number;
  dimension: number;
  indexType: string;
  memoryUsage: number;
}

interface VectorStoreStatusProps {
  stats: VectorStoreStats;
}

const VectorStoreStatus: React.FC<VectorStoreStatusProps> = ({ stats }) => {
  // 模拟的collection信息
  const collections: CollectionInfo[] = [
    {
      name: 'multimodal_text',
      count: stats.textChunks,
      dimension: 768,
      indexType: 'HNSW',
      memoryUsage: stats.textChunks * 0.003 // 每个向量约3KB
    },
    {
      name: 'multimodal_image',
      count: stats.images,
      dimension: 768,
      indexType: 'HNSW',
      memoryUsage: stats.images * 0.003
    },
    {
      name: 'multimodal_table',
      count: stats.tables,
      dimension: 768,
      indexType: 'HNSW',
      memoryUsage: stats.tables * 0.003
    }
  ];

  const totalMemory = collections.reduce((sum, col) => sum + col.memoryUsage, 0);
  const indexingSpeed = 1000; // 向量/秒
  const queryLatency = 5; // ms

  const columns = [
    {
      title: 'Collection',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => (
        <Space>
          <ClusterOutlined />
          <Text code>{text}</Text>
        </Space>
      )
    },
    {
      title: '向量数',
      dataIndex: 'count',
      key: 'count',
      render: (count: number) => (
        <Tag color={count > 0 ? 'green' : 'default'}>{count}</Tag>
      )
    },
    {
      title: '维度',
      dataIndex: 'dimension',
      key: 'dimension',
      render: (dim: number) => <Text>{dim}D</Text>
    },
    {
      title: '索引类型',
      dataIndex: 'indexType',
      key: 'indexType',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '内存占用',
      dataIndex: 'memoryUsage',
      key: 'memoryUsage',
      render: (usage: number) => <Text>{usage.toFixed(2)} MB</Text>
    }
  ];

  return (
    <Card 
      title={
        <span>
          <DatabaseOutlined className="mr-2" />
          Chroma向量存储状态
        </span>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        {/* 总体统计 */}
        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总向量数"
                value={stats.totalDocuments}
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="嵌入维度"
                value={stats.embeddingDimension}
                suffix="D"
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="内存使用"
                value={totalMemory}
                suffix="MB"
                precision={2}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="缓存命中率"
                value={stats.cacheHitRate}
                suffix="%"
                prefix={
                  <Progress
                    type="circle"
                    percent={stats.cacheHitRate}
                    width={30}
                    strokeColor="#52c41a"
                  />
                }
              />
            </Card>
          </Col>
        </Row>

        {/* Collection详情 */}
        <div>
          <Title level={5}>Collection详情</Title>
          <Table
            dataSource={collections}
            columns={columns}
            pagination={false}
            size="small"
            rowKey="name"
          />
        </div>

        {/* 性能指标 */}
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="索引速度"
                value={indexingSpeed}
                suffix="vectors/s"
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="查询延迟"
                value={queryLatency}
                suffix="ms"
                prefix={<ClusterOutlined />}
                valueStyle={{ color: queryLatency < 10 ? '#52c41a' : '#faad14' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <Statistic
                title="相似度算法"
                value="Cosine"
                prefix={<InfoCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>

        {/* 内容分布 */}
        <div>
          <Title level={5}>内容类型分布</Title>
          <Row gutter={16}>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <FileTextOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                <div>文本块</div>
                <Progress
                  percent={stats.totalDocuments > 0 ? Math.round((stats.textChunks / stats.totalDocuments) * 100) : 0}
                  strokeColor="#1890ff"
                />
                <Text type="secondary">{stats.textChunks} 个</Text>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <FileImageOutlined style={{ fontSize: 24, color: '#722ed1' }} />
                <div>图像</div>
                <Progress
                  percent={stats.totalDocuments > 0 ? Math.round((stats.images / stats.totalDocuments) * 100) : 0}
                  strokeColor="#722ed1"
                />
                <Text type="secondary">{stats.images} 个</Text>
              </div>
            </Col>
            <Col span={8}>
              <div style={{ textAlign: 'center' }}>
                <TableOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                <div>表格</div>
                <Progress
                  percent={stats.totalDocuments > 0 ? Math.round((stats.tables / stats.totalDocuments) * 100) : 0}
                  strokeColor="#52c41a"
                />
                <Text type="secondary">{stats.tables} 个</Text>
              </div>
            </Col>
          </Row>
        </div>

        {/* 技术栈信息 */}
        <Alert
          message="向量存储技术栈"
          description={
            <Space direction="vertical" size="small">
              <Text>• 数据库: Chroma DB (持久化存储)</Text>
              <Text>• 索引算法: HNSW (分层可导航小世界)</Text>
              <Text>• 嵌入模型: Nomic Embed v1.5</Text>
              <Text>• 相似度度量: 余弦相似度</Text>
              <Text>• 缓存层: Redis (查询结果缓存)</Text>
            </Space>
          }
          variant="default"
          showIcon
        />
      </Space>
    </Card>
  );
};

export default VectorStoreStatus;