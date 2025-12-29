import React, { useEffect, useState } from 'react';
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
  BulbOutlined,
  ReloadOutlined
} from '@ant-design/icons';

import HybridRetrievalPanel from '../components/pgvector/HybridRetrievalPanel';
import { pgvectorApi } from '../services/pgvectorApi';
import apiClient from '../services/apiClient';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

const HybridSearchAdvancedPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [health, setHealth] = useState<any>(null);
  const [quantizationConfig, setQuantizationConfig] = useState<any>(null);
  const [performanceTargets, setPerformanceTargets] = useState<any[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadOverview();
  }, []);

  const normalizePerformanceMetrics = (metricsResp: any) => {
    if (!metricsResp) return [];
    if (Array.isArray(metricsResp)) return metricsResp;
    const qp = metricsResp.query_performance || {};
    const endTime = metricsResp.report_period?.end_time || new Date().toISOString();
    return [
      {
        timestamp: endTime,
        avg_latency_ms: qp.average_execution_time_ms || 0,
        p95_latency_ms: qp.max_execution_time_ms || 0,
        cache_hit_rate: qp.cache_hit_ratio || 0,
        quantization_ratio: qp.quantization_ratio || 0,
        search_count: qp.total_queries || 0,
      },
    ];
  };

  const loadOverview = async () => {
    setLoading(true);
    try {
      setError(null);
      const [targetsResp, metricsResp, quantConfig, healthResp] = await Promise.all([
        pgvectorApi.getPerformanceTargets().catch(() => []),
        pgvectorApi.getPerformanceMetrics('1h').catch(() => null),
        pgvectorApi.getQuantizationConfig().catch(() => null),
        apiClient.get('/pgvector/health').then(res => res.data).catch(() => null)
      ]);
      setPerformanceTargets(Array.isArray(targetsResp) ? targetsResp : []);
      setPerformanceMetrics(normalizePerformanceMetrics(metricsResp));
      setQuantizationConfig(quantConfig);
      setHealth(healthResp);
    } catch (err) {
      setError((err as Error).message || '加载混合检索状态失败');
    } finally {
      setLoading(false);
    }
  };

  const latestMetric = performanceMetrics.length ? performanceMetrics[performanceMetrics.length - 1] : null;

  const formatTargetValue = (item: any) => {
    if (typeof item.current !== 'number') return '-';
    if (item.metric.includes('延迟')) return `${(item.current * 1000).toFixed(1)} ms`;
    if (item.metric.includes('率') || item.metric.includes('使用率')) return `${(item.current * 100).toFixed(1)}%`;
    return item.current;
  };

  const performanceDisplay = latestMetric
    ? [
        { label: '平均查询延迟', value: `${latestMetric.avg_latency_ms.toFixed(1)} ms` },
        { label: 'P95查询延迟', value: `${latestMetric.p95_latency_ms.toFixed(1)} ms` },
        { label: '缓存命中率', value: `${(latestMetric.cache_hit_rate * 100).toFixed(1)}%` },
        { label: '量化使用率', value: `${(latestMetric.quantization_ratio * 100).toFixed(1)}%` }
      ]
    : performanceTargets.map((item) => ({
        label: item.metric,
        value: formatTargetValue(item)
      }));

  const systemFeatures = [
    {
      title: 'pgvector健康',
      description: health ? `版本: ${health.pgvector_version || '未知'}` : '等待健康检查',
      icon: <DatabaseOutlined />,
      status: health?.status === 'healthy' ? 'normal' : 'unknown'
    },
    {
      title: '性能监控',
      description: latestMetric ? `最近查询 ${latestMetric.search_count || 0} 次` : '暂无性能数据',
      icon: <BarChartOutlined />,
      status: latestMetric ? 'normal' : 'unknown'
    },
    {
      title: '量化配置',
      description: quantizationConfig ? `模式: ${quantizationConfig.mode}` : '未配置',
      icon: <ThunderboltOutlined />,
      status: quantizationConfig ? 'normal' : 'unknown'
    },
    {
      title: '缓存状态',
      description: latestMetric ? `命中率 ${(latestMetric.cache_hit_rate * 100).toFixed(1)}%` : '暂无数据',
      icon: <ApiOutlined />,
      status: latestMetric ? 'normal' : 'unknown'
    }
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
        <div style={{ marginTop: 12 }}>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadOverview} loading={loading}>
              刷新状态
            </Button>
          </Space>
        </div>
      </div>

      {error && (
        <Alert
          message="状态加载失败"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

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
                                color={feature.status === 'normal' ? 'green' : 'default'}
                                style={{ marginLeft: 8 }}
                              >
                                {feature.status === 'normal' ? '可用' : '未就绪'}
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
                {performanceDisplay.length ? (
                  <Row gutter={[16, 16]}>
                    {performanceDisplay.map((metric, index) => (
                      <Col span={6} key={index}>
                        <Statistic
                          title={metric.label}
                          value={metric.value}
                          valueStyle={{ color: '#1890ff' }}
                          prefix={<ThunderboltOutlined />}
                        />
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <Alert message="暂无性能数据" type="info" showIcon />
                )}
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
