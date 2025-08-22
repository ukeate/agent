/**
 * 向量索引高级功能页面 - Story 4.2
 * 
 * 展示学习型AI系统的高级向量技术，包括：
 * - 多种索引类型管理（HNSW、IVF、LSH）
 * - 混合搜索引擎（语义+关键词）
 * - 多模态向量搜索（图像、音频、文本）
 * - 时序向量索引和轨迹分析
 * - 向量聚类和异常检测
 * - 向量可视化（t-SNE、UMAP、PCA）
 * - 自定义距离度量
 * - 向量数据导入导出工具
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Tabs,
  Typography,
  Row,
  Col,
  Button,
  Alert,
  Space,
  Tag,
  Statistic,
  Progress,
  Divider,
  message,
  Spin
} from 'antd';
import {
  DatabaseOutlined,
  ClusterOutlined,
  SearchOutlined,
  PictureOutlined,
  LineChartOutlined,
  EyeOutlined,
  CalculatorOutlined,
  ImportOutlined,
  ExportOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  CloudOutlined
} from '@ant-design/icons';

import IndexManagerPanel from '../components/vector/IndexManagerPanel';
import HybridSearchPanel from '../components/vector/HybridSearchPanel';
import MultiModalSearchPanel from '../components/vector/MultiModalSearchPanel';
import TemporalVectorPanel from '../components/vector/TemporalVectorPanel';
import VectorClusteringPanel from '../components/vector/VectorClusteringPanel';
import VectorVisualizationPanel from '../components/vector/VectorVisualizationPanel';
import DistanceMetricsPanel from '../components/vector/DistanceMetricsPanel';
import DataToolsPanel from '../components/vector/DataToolsPanel';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface SystemStats {
  total_vectors: number;
  unique_entities: number;
  active_indexes: number;
  clusters_detected: number;
  patterns_detected: number;
  last_updated: string;
}

const VectorAdvancedPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('indexes');
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStats();
  }, []);

  const fetchSystemStats = async () => {
    try {
      setLoading(true);
      // Mock data - 在实际应用中应该调用对应的API
      const stats: SystemStats = {
        total_vectors: 1250000,
        unique_entities: 8500,
        active_indexes: 12,
        clusters_detected: 45,
        patterns_detected: 128,
        last_updated: new Date().toISOString()
      };
      setSystemStats(stats);
    } catch (error) {
      message.error('获取系统统计信息失败');
      console.error('Failed to fetch system stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderSystemOverview = () => (
    <Card 
      title="向量系统概览" 
      style={{ marginBottom: 24 }}
      extra={
        <Button onClick={fetchSystemStats} loading={loading}>
          刷新统计
        </Button>
      }
    >
      {systemStats && (
        <>
          <Row gutter={[24, 16]}>
            <Col span={6}>
              <Statistic
                title="总向量数"
                value={systemStats.total_vectors}
                valueStyle={{ color: '#3f8600' }}
                prefix={<DatabaseOutlined />}
                formatter={(value) => `${(Number(value) / 1000000).toFixed(1)}M`}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="实体数量"
                value={systemStats.unique_entities}
                valueStyle={{ color: '#1890ff' }}
                prefix={<CloudOutlined />}
                formatter={(value) => `${(Number(value) / 1000).toFixed(1)}K`}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="活跃索引"
                value={systemStats.active_indexes}
                valueStyle={{ color: '#722ed1' }}
                prefix={<ThunderboltOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="已检测聚类"
                value={systemStats.clusters_detected}
                valueStyle={{ color: '#fa8c16' }}
                prefix={<ClusterOutlined />}
              />
            </Col>
          </Row>
          
          <Divider />
          
          <Row gutter={[24, 16]}>
            <Col span={12}>
              <Card size="small" title="索引类型分布">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>HNSW索引</span>
                    <Tag color="blue">5个</Tag>
                  </div>
                  <Progress percent={42} size="small" strokeColor="#1890ff" />
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>IVF索引</span>
                    <Tag color="green">4个</Tag>
                  </div>
                  <Progress percent={33} size="small" strokeColor="#52c41a" />
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>LSH索引</span>
                    <Tag color="orange">3个</Tag>
                  </div>
                  <Progress percent={25} size="small" strokeColor="#fa8c16" />
                </Space>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card size="small" title="功能使用统计">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>混合搜索</span>
                    <Tag color="cyan">活跃</Tag>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>多模态搜索</span>
                    <Tag color="purple">活跃</Tag>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>时序分析</span>
                    <Tag color="gold">活跃</Tag>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>聚类分析</span>
                    <Tag color="lime">活跃</Tag>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </>
      )}
    </Card>
  );

  const renderFeatureDescription = () => (
    <Alert
      message="高级向量索引功能说明"
      description={
        <div>
          <Paragraph>
            这是基于 <strong>pgvector 0.8</strong> 构建的高级向量索引学习系统，展示了现代AI系统中的关键向量技术：
          </Paragraph>
          <ul>
            <li><strong>多索引架构</strong>：HNSW图索引、IVF倒排索引、LSH哈希索引的智能选择</li>
            <li><strong>混合检索</strong>：语义向量搜索与BM25关键词搜索的深度融合</li>
            <li><strong>多模态支持</strong>：文本、图像、音频向量的统一处理和跨模态搜索</li>
            <li><strong>时序分析</strong>：向量轨迹跟踪、模式检测和趋势预测</li>
            <li><strong>智能聚类</strong>：K-means、DBSCAN、异常检测算法</li>
            <li><strong>高维可视化</strong>：t-SNE、UMAP、PCA降维和交互式探索</li>
            <li><strong>距离度量</strong>：余弦、欧氏、曼哈顿等多种度量和自定义函数</li>
            <li><strong>数据管理</strong>：向量数据的导入、导出、迁移和备份工具</li>
          </ul>
        </div>
      }
      variant="default"
      showIcon
      style={{ marginBottom: 24 }}
    />
  );

  if (loading && !systemStats) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}>
          <Text>加载向量系统数据...</Text>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          高级向量索引系统
          <Tag color="blue" style={{ marginLeft: 12 }}>Story 4.2</Tag>
        </Title>
        <Text type="secondary" style={{ fontSize: 16 }}>
          基于pgvector 0.8的多维向量检索与智能分析平台
        </Text>
      </div>

      {/* 功能说明 */}
      {renderFeatureDescription()}

      {/* 系统概览 */}
      {renderSystemOverview()}

      {/* 功能标签页 */}
      <Card>
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          size="large"
          type="card"
        >
          {/* 索引管理 */}
          <TabPane
            tab={
              <span>
                <ThunderboltOutlined />
                索引管理
              </span>
            }
            key="indexes"
          >
            <IndexManagerPanel />
          </TabPane>

          {/* 混合搜索 */}
          <TabPane
            tab={
              <span>
                <SearchOutlined />
                混合搜索
              </span>
            }
            key="hybrid-search"
          >
            <HybridSearchPanel />
          </TabPane>

          {/* 多模态搜索 */}
          <TabPane
            tab={
              <span>
                <PictureOutlined />
                多模态搜索
              </span>
            }
            key="multimodal"
          >
            <MultiModalSearchPanel />
          </TabPane>

          {/* 时序向量 */}
          <TabPane
            tab={
              <span>
                <LineChartOutlined />
                时序分析
              </span>
            }
            key="temporal"
          >
            <TemporalVectorPanel />
          </TabPane>

          {/* 聚类分析 */}
          <TabPane
            tab={
              <span>
                <ClusterOutlined />
                聚类分析
              </span>
            }
            key="clustering"
          >
            <VectorClusteringPanel />
          </TabPane>

          {/* 可视化 */}
          <TabPane
            tab={
              <span>
                <EyeOutlined />
                向量可视化
              </span>
            }
            key="visualization"
          >
            <VectorVisualizationPanel />
          </TabPane>

          {/* 距离度量 */}
          <TabPane
            tab={
              <span>
                <CalculatorOutlined />
                距离度量
              </span>
            }
            key="distance"
          >
            <DistanceMetricsPanel />
          </TabPane>

          {/* 数据工具 */}
          <TabPane
            tab={
              <span>
                <ImportOutlined />
                数据工具
              </span>
            }
            key="data-tools"
          >
            <DataToolsPanel />
          </TabPane>
        </Tabs>
      </Card>

      {/* 页面样式 */}
      <style>{`
        .ant-tabs-card > .ant-tabs-content {
          margin-top: 16px;
        }
        
        .ant-tabs-card > .ant-tabs-content > .ant-tabs-tabpane {
          background: #fff;
          padding: 24px;
          border-radius: 8px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        
        .ant-statistic-title {
          font-weight: 500;
        }
        
        .ant-progress-line {
          margin: 8px 0;
        }
      `}</style>
    </div>
  );
};

export default VectorAdvancedPage;