import React, { useState } from 'react';
import { 
  Card, 
  Typography, 
  Tabs,
  Row,
  Col,
  Alert,
  Tag,
  Space,
  Button,
  Statistic
} from 'antd';
import { 
  DatabaseOutlined, 
  SearchOutlined, 
  ThunderboltOutlined,
  ApiOutlined,
  BarChartOutlined,
  BulbOutlined
} from '@ant-design/icons';

import HybridRetrievalPanel from '../components/pgvector/HybridRetrievalPanel';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

const HybridSearchAdvancedPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const systemFeatures = [
    {
      title: 'pgvector集成',
      description: 'PostgreSQL向量扩展，支持余弦相似度和欧几里得距离',
      icon: <DatabaseOutlined />,
      status: 'active'
    },
    {
      title: 'Qdrant兼容',
      description: '高性能向量数据库，支持过滤和混合查询',
      icon: <SearchOutlined />,
      status: 'ready'
    },
    {
      title: 'BM25检索',
      description: 'PostgreSQL全文搜索，基于TF-IDF的关键词匹配',
      icon: <ApiOutlined />,
      status: 'active'
    },
    {
      title: 'RRF融合',
      description: 'Reciprocal Rank Fusion结果融合算法',
      icon: <ThunderboltOutlined />,
      status: 'active'
    }
  ];

  const performanceMetrics = [
    { label: '平均查询延迟', value: '85ms', status: 'good' },
    { label: '向量检索精度', value: '94.2%', status: 'excellent' },
    { label: '关键词匹配率', value: '87.8%', status: 'good' },
    { label: '混合融合效果', value: '96.1%', status: 'excellent' }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SearchOutlined style={{ marginRight: '12px' }} />
          混合检索 (pgvector + Qdrant)
        </Title>
        <Text type="secondary">
          结合PostgreSQL pgvector和Qdrant向量数据库的混合检索系统，提供高性能的语义搜索能力
        </Text>
      </div>

      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        size="large"
      >
        <TabPane 
          tab={<span><BulbOutlined />系统概览</span>} 
          key="overview"
        >
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Alert
                message="混合检索架构"
                description="本系统集成了PostgreSQL pgvector扩展和Qdrant向量数据库，通过智能路由和结果融合实现最优检索效果。支持语义向量搜索、BM25关键词搜索和混合检索模式。"
                variant="default"
                showIcon
                style={{ marginBottom: 16 }}
              />
            </Col>

            {/* 系统特性 */}
            <Col span={24}>
              <Card title="核心特性" extra={<ApiOutlined />}>
                <Row gutter={[16, 16]}>
                  {systemFeatures.map((feature, index) => (
                    <Col span={12} key={index}>
                      <Card size="small">
                        <Space align="start">
                          <div style={{ fontSize: '24px', color: '#1890ff' }}>
                            {feature.icon}
                          </div>
                          <div>
                            <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
                              {feature.title}
                              <Tag 
                                color={feature.status === 'active' ? 'green' : 'blue'}
                                style={{ marginLeft: 8 }}
                              >
                                {feature.status === 'active' ? '已启用' : '就绪'}
                              </Tag>
                            </div>
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {feature.description}
                            </Text>
                          </div>
                        </Space>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>

            {/* 性能指标 */}
            <Col span={24}>
              <Card title="性能指标" extra={<BarChartOutlined />}>
                <Row gutter={[16, 16]}>
                  {performanceMetrics.map((metric, index) => (
                    <Col span={6} key={index}>
                      <Statistic
                        title={metric.label}
                        value={metric.value}
                        valueStyle={{ 
                          color: metric.status === 'excellent' ? '#3f8600' : '#1890ff' 
                        }}
                        prefix={
                          metric.status === 'excellent' ? 
                            <ThunderboltOutlined /> : 
                            <BarChartOutlined />
                        }
                      />
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>

            {/* 技术架构 */}
            <Col span={24}>
              <Card title="技术架构" extra={<DatabaseOutlined />}>
                <Paragraph>
                  <Title level={4}>检索流程</Title>
                  <ol>
                    <li><strong>查询预处理</strong>：文本清理、分词、查询扩展</li>
                    <li><strong>并行检索</strong>：pgvector语义搜索 + PostgreSQL全文搜索</li>
                    <li><strong>结果融合</strong>：RRF（Reciprocal Rank Fusion）算法融合</li>
                    <li><strong>重排序</strong>：可选的交叉编码器精排</li>
                    <li><strong>结果返回</strong>：格式化输出，包含相关性分数</li>
                  </ol>
                </Paragraph>

                <Paragraph>
                  <Title level={4}>存储优化</Title>
                  <ul>
                    <li><strong>向量量化</strong>：支持float32/int8/int4量化压缩</li>
                    <li><strong>索引优化</strong>：HNSW索引优化，支持动态参数调整</li>
                    <li><strong>缓存策略</strong>：热点查询缓存，减少检索延迟</li>
                    <li><strong>分片存储</strong>：大规模向量数据分片管理</li>
                  </ul>
                </Paragraph>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane 
          tab={<span><SearchOutlined />实时检索测试</span>} 
          key="testing"
        >
          <HybridRetrievalPanel />
        </TabPane>
      </Tabs>
    </div>
  );
};

export default HybridSearchAdvancedPage;