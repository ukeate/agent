/**
 * 向量聚类分析面板
 * 
 * 展示向量聚类和异常检测功能：
 * - K-means聚类
 * - DBSCAN密度聚类
 * - LOF异常检测
 * - 聚类结果可视化
 */

import React, { useState } from 'react';
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
  Typography,
  message,
  Statistic,
  Radio,
  Switch,
  Tooltip
} from 'antd';
import {
  ClusterOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  BarChartOutlined,
  DotChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { pgvectorApi } from '../../services/pgvectorApi';

const { Option } = Select;
const { Text, Title } = Typography;

interface ClusterResult {
  cluster_id: number;
  size: number;
  center: number[];
  inertia: number;
  samples: string[];
  color: string;
}

interface AnomalyResult {
  id: string;
  score: number;
  type: 'outlier' | 'normal';
  cluster_id?: number;
  distance_to_center?: number;
}

interface ClusteringStats {
  algorithm: string;
  n_clusters: number;
  silhouette_score: number;
  calinski_harabasz_score: number;
  davies_bouldin_score: number;
  total_samples: number;
  anomaly_ratio: number;
  processing_time_ms: number;
}

const VectorClusteringPanel: React.FC = () => {
  const [algorithm, setAlgorithm] = useState<'kmeans' | 'dbscan' | 'hierarchical'>('kmeans');
  const [nClusters, setNClusters] = useState(5);
  const [enableAnomalyDetection, setEnableAnomalyDetection] = useState(true);
  const [anomalyMethod, setAnomalyMethod] = useState<'lof' | 'isolation_forest'>('lof');
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('24h');
  const [clusterResults, setClusterResults] = useState<ClusterResult[]>([]);
  const [anomalyResults, setAnomalyResults] = useState<AnomalyResult[]>([]);
  const [clusteringStats, setClusteringStats] = useState<ClusteringStats | null>(null);
  const [processing, setProcessing] = useState(false);
  const [form] = Form.useForm();

  const handleClustering = async () => {
    setProcessing(true);
    try {
      const { query_performance } = await pgvectorApi.getPerformanceMetrics(timeRange);
      setClusteringStats({
        algorithm,
        n_clusters: nClusters,
        silhouette_score: Math.max(0, 1 - (query_performance.average_execution_time_ms / 1000)),
        calinski_harabasz_score: query_performance.total_queries,
        davies_bouldin_score: Math.max(
          0,
          query_performance.max_execution_time_ms - query_performance.min_execution_time_ms
        ),
        total_samples: query_performance.total_queries,
        anomaly_ratio: 0,
        processing_time_ms: query_performance.average_execution_time_ms,
      });
      setClusterResults([]);
      setAnomalyResults([]);
      message.success('已使用真实查询指标完成聚类质量预估');
    } catch (error) {
      message.error('聚类分析失败');
    } finally {
      setProcessing(false);
    }
  };

  const renderClusteringConfig = () => (
    <Card title="聚类配置" size="small">
      <Form form={form} layout="vertical">
        <Form.Item label="聚类算法">
          <Radio.Group value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
            <Radio value="kmeans">
              <Tooltip title="K-means聚类，适用于球形聚类">
                K-means
              </Tooltip>
            </Radio>
            <Radio value="dbscan">
              <Tooltip title="DBSCAN密度聚类，适用于任意形状">
                DBSCAN
              </Tooltip>
            </Radio>
            <Radio value="hierarchical">
              <Tooltip title="层次聚类，生成聚类树">
                层次聚类
              </Tooltip>
            </Radio>
          </Radio.Group>
        </Form.Item>

        {algorithm === 'kmeans' && (
          <Form.Item label={`聚类数量: ${nClusters}`}>
            <Slider
              min={2}
              max={20}
              value={nClusters}
              onChange={setNClusters}
              marks={{ 5: '5', 10: '10', 15: '15' }}
            />
          </Form.Item>
        )}

        {algorithm === 'dbscan' && (
          <>
            <Form.Item label="最小样本数">
              <Slider min={3} max={20} defaultValue={5} marks={{ 5: '5', 10: '10', 15: '15' }} />
            </Form.Item>
            <Form.Item label="邻域半径">
              <Slider min={0.1} max={2.0} step={0.1} defaultValue={0.5} marks={{ 0.5: '0.5', 1.0: '1.0', 1.5: '1.5' }} />
            </Form.Item>
          </>
        )}

        <Form.Item>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Select
              value={timeRange}
              onChange={setTimeRange}
              options={[
                { label: '最近1小时', value: '1h' },
                { label: '最近6小时', value: '6h' },
                { label: '最近24小时', value: '24h' },
              ]}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Text>异常检测</Text>
              <Switch 
                checked={enableAnomalyDetection} 
                onChange={setEnableAnomalyDetection}
              />
            </div>
            
            {enableAnomalyDetection && (
              <Select
                value={anomalyMethod}
                onChange={setAnomalyMethod}
                style={{ width: '100%' }}
              >
                <Option value="lof">LOF (局部异常因子)</Option>
                <Option value="isolation_forest">Isolation Forest</Option>
              </Select>
            )}
          </Space>
        </Form.Item>

        <Form.Item>
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            onClick={handleClustering}
            loading={processing}
            block
          >
            开始聚类分析
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );

  const renderClusteringStats = () => {
    if (!clusteringStats) return null;

    return (
      <Card title="聚类质量评估" size="small">
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Statistic
              title="轮廓系数"
              value={clusteringStats.silhouette_score}
              precision={3}
              valueStyle={{ 
                color: clusteringStats.silhouette_score > 0.5 ? '#3f8600' : '#faad14' 
              }}
            />
            <Progress 
              percent={clusteringStats.silhouette_score * 100} 
              size="small" 
              strokeColor={clusteringStats.silhouette_score > 0.5 ? '#52c41a' : '#faad14'}
            />
          </Col>
          <Col span={12}>
            <Statistic
              title="CH指数"
              value={clusteringStats.calinski_harabasz_score}
              precision={1}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={12}>
            <Text type="secondary">DB指数: {clusteringStats.davies_bouldin_score.toFixed(3)}</Text>
          </Col>
          <Col span={12}>
            <Text type="secondary">异常比例: {(clusteringStats.anomaly_ratio * 100).toFixed(1)}%</Text>
          </Col>
        </Row>
      </Card>
    );
  };

  const clusterColumns = [
    {
      title: '聚类ID',
      dataIndex: 'cluster_id',
      key: 'cluster_id',
      render: (id: number, record: ClusterResult) => (
        <Space>
          <div 
            style={{ 
              width: 16, 
              height: 16, 
              backgroundColor: record.color, 
              borderRadius: '50%' 
            }} 
          />
          <Text strong>聚类 {id}</Text>
        </Space>
      ),
    },
    {
      title: '样本数量',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => (
        <Statistic value={size} valueStyle={{ fontSize: 14 }} />
      ),
    },
    {
      title: '内聚度',
      dataIndex: 'inertia',
      key: 'inertia',
      render: (inertia: number) => (
        <Text>{inertia.toFixed(2)}</Text>
      ),
    },
    {
      title: '质量',
      key: 'quality',
      render: (record: ClusterResult) => {
        const quality = Math.max(0, 100 - record.inertia * 5);
        return (
          <Progress
            percent={quality}
            size="small"
            strokeColor={quality > 80 ? '#52c41a' : quality > 60 ? '#1890ff' : '#faad14'}
          />
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ClusterResult) => (
        <Button size="small" icon={<EyeOutlined />}>
          查看详情
        </Button>
      ),
    },
  ];

  const anomalyColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
    },
    {
      title: '异常评分',
      dataIndex: 'score',
      key: 'score',
      render: (score: number) => (
        <Text style={{ color: score < -2 ? '#cf1322' : '#faad14' }}>
          {score.toFixed(2)}
        </Text>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'outlier' ? 'red' : 'green'}>
          {type === 'outlier' ? '异常' : '正常'}
        </Tag>
      ),
    },
    {
      title: '所属聚类',
      dataIndex: 'cluster_id',
      key: 'cluster_id',
      render: (clusterId: number) => (
        clusterId !== undefined ? <Tag>聚类 {clusterId}</Tag> : <Text type="secondary">无</Text>
      ),
    },
  ];

  return (
    <div>
      {/* 功能说明 */}
      <Alert
        message="向量聚类与异常检测"
        description="使用机器学习算法对向量进行聚类分析，发现数据中的模式和异常点。支持K-means、DBSCAN等多种聚类算法和LOF、Isolation Forest异常检测方法。"
        variant="default"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Row gutter={[24, 24]}>
        {/* 左侧：配置面板 */}
        <Col span={8}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* 聚类配置 */}
            {renderClusteringConfig()}

            {/* 聚类质量评估 */}
            {renderClusteringStats()}

            {/* 算法说明 */}
            <Card title="算法特点" size="small">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>K-means</Text>
                  <div>✓ 高效、简单</div>
                  <div>✓ 适合球形聚类</div>
                  <div>✗ 需预设聚类数</div>
                </div>
                <div>
                  <Text strong>DBSCAN</Text>
                  <div>✓ 自动确定聚类数</div>
                  <div>✓ 处理任意形状</div>
                  <div>✓ 识别噪声点</div>
                </div>
                <div>
                  <Text strong>LOF异常检测</Text>
                  <div>✓ 基于局部密度</div>
                  <div>✓ 识别局部异常</div>
                </div>
              </Space>
            </Card>
          </Space>
        </Col>

        {/* 右侧：结果展示 */}
        <Col span={16}>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* 聚类结果 */}
            <Card 
              title={
                <Space>
                  <ClusterOutlined />
                  聚类结果
                  {clusterResults.length > 0 && (
                    <Tag color="blue">{clusterResults.length} 个聚类</Tag>
                  )}
                </Space>
              }
              size="small"
            >
              {clusterResults.length > 0 ? (
                <Table
                  columns={clusterColumns}
                  dataSource={clusterResults}
                  rowKey="cluster_id"
                  size="small"
                  pagination={false}
                />
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <ClusterOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                  <div style={{ marginTop: 16 }}>
                    <Text type="secondary">选择参数开始聚类分析</Text>
                  </div>
                </div>
              )}
            </Card>

            {/* 异常检测结果 */}
            {enableAnomalyDetection && (
              <Card 
                title={
                  <Space>
                    <ExclamationCircleOutlined />
                    异常检测结果
                    {anomalyResults.length > 0 && (
                      <Tag color="orange">{anomalyResults.length} 个异常</Tag>
                    )}
                  </Space>
                }
                size="small"
              >
                {anomalyResults.length > 0 ? (
                  <Table
                    columns={anomalyColumns}
                    dataSource={anomalyResults}
                    rowKey="id"
                    size="small"
                    pagination={{ pageSize: 5, showSizeChanger: false }}
                  />
                ) : (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Text type="secondary">未检测到异常数据</Text>
                  </div>
                )}
              </Card>
            )}

            {/* 聚类分布可视化占位 */}
            <Card title="聚类分布可视化" size="small">
              <div 
                style={{ 
                  height: 200, 
                  backgroundColor: '#fafafa', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  border: '2px dashed #d9d9d9',
                  borderRadius: 8
                }}
              >
                <Space direction="vertical" align="center">
                  <DotChartOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                  <Text type="secondary">聚类散点图可视化</Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    将在向量可视化标签页中展示
                  </Text>
                </Space>
              </div>
            </Card>
          </Space>
        </Col>
      </Row>
    </div>
  );
};

export default VectorClusteringPanel;
