import React, { useState, useEffect } from 'react';
import { 
import { logger } from '../utils/logger'
  Card, 
  Tabs, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Alert, 
  Progress,
  Tag,
  Statistic,
  Space,
  Divider
} from 'antd';
import { 
  DatabaseOutlined, 
  ThunderboltOutlined, 
  BarChartOutlined, 
  SearchOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';

import VectorQuantizationPanel from '../components/pgvector/VectorQuantizationPanel';
import PerformanceMonitorPanel from '../components/pgvector/PerformanceMonitorPanel';
import HybridRetrievalPanel from '../components/pgvector/HybridRetrievalPanel';
import DataIntegrityPanel from '../components/pgvector/DataIntegrityPanel';
import { pgvectorApi } from '../services/pgvectorApi';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface SystemStatus {
  pgvector_version: string;
  upgrade_available: boolean;
  quantization_enabled: boolean;
  cache_status: 'healthy' | 'warning' | 'error';
  index_health: 'optimal' | 'needs_optimization' | 'error';
  last_updated: string;
}

const PgVectorPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [upgradeInProgress, setUpgradeInProgress] = useState(false);

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  const fetchSystemStatus = async () => {
    try {
      setLoading(true);
      const status = await pgvectorApi.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      logger.error('获取系统状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpgradePgVector = async () => {
    try {
      setUpgradeInProgress(true);
      const result = await pgvectorApi.upgradeToPgVector08();
      if (result.success) {
        await fetchSystemStatus();
      }
    } catch (error) {
      logger.error('升级失败:', error);
    } finally {
      setUpgradeInProgress(false);
    }
  };

  const getVersionStatusColor = (version: string) => {
    if (version >= '0.8.0') return 'success';
    if (version >= '0.5.0') return 'warning';
    return 'error';
  };

  const getCacheStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>Loading pgvector system status...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: '12px' }} />
          pgvector 0.8 向量优化系统
        </Title>
        <Text type="secondary">
          PostgreSQL向量扩展升级、量化压缩和性能优化管理
        </Text>
      </div>

      {/* 系统状态概览 */}
      <Card 
        title="系统状态概览" 
        style={{ marginBottom: '24px' }}
        extra={
          <Button onClick={fetchSystemStatus} loading={loading}>
            刷新状态
          </Button>
        }
      >
        {systemStatus && (
          <Row gutter={24}>
            <Col span={6}>
              <Statistic
                title="pgvector版本"
                value={systemStatus.pgvector_version}
                valueStyle={{ 
                  color: getVersionStatusColor(systemStatus.pgvector_version) === 'success' ? '#3f8600' : '#cf1322' 
                }}
                suffix={
                  <Tag color={getVersionStatusColor(systemStatus.pgvector_version)}>
                    {systemStatus.pgvector_version >= '0.8.0' ? '已升级' : '需升级'}
                  </Tag>
                }
              />
            </Col>
            
            <Col span={6}>
              <Statistic
                title="量化状态"
                value={systemStatus.quantization_enabled ? '已启用' : '未启用'}
                valueStyle={{ 
                  color: systemStatus.quantization_enabled ? '#3f8600' : '#cf1322' 
                }}
                prefix={systemStatus.quantization_enabled ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
              />
            </Col>

            <Col span={6}>
              <Statistic
                title="缓存状态"
                value={systemStatus.cache_status.toUpperCase()}
                valueStyle={{ 
                  color: getCacheStatusColor(systemStatus.cache_status) === 'success' ? '#3f8600' : '#faad14' 
                }}
              />
            </Col>

            <Col span={6}>
              <Statistic
                title="索引健康度"
                value={systemStatus.index_health === 'optimal' ? '优化' : '需优化'}
                valueStyle={{ 
                  color: systemStatus.index_health === 'optimal' ? '#3f8600' : '#faad14' 
                }}
              />
            </Col>
          </Row>
        )}

        {systemStatus?.upgrade_available && (
          <div style={{ marginTop: '16px' }}>
            <Alert
              message="pgvector升级可用"
              description="检测到pgvector 0.8版本可用，建议升级以获得更好的性能和量化支持。"
              variant="default"
              showIcon
              action={
                <Button 
                  type="primary" 
                  onClick={handleUpgradePgVector}
                  loading={upgradeInProgress}
                >
                  升级到0.8版本
                </Button>
              }
            />
          </div>
        )}
      </Card>

      {/* 功能标签页 */}
      <Card>
        <Tabs defaultActiveKey="quantization" size="large">
          <TabPane
            tab={
              <span>
                <ThunderboltOutlined />
                向量量化配置
              </span>
            }
            key="quantization"
          >
            <VectorQuantizationPanel />
          </TabPane>

          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                性能监控
              </span>
            }
            key="performance"
          >
            <PerformanceMonitorPanel />
          </TabPane>

          <TabPane
            tab={
              <span>
                <SearchOutlined />
                混合检索测试
              </span>
            }
            key="retrieval"
          >
            <HybridRetrievalPanel />
          </TabPane>

          <TabPane
            tab={
              <span>
                <DatabaseOutlined />
                数据完整性
              </span>
            }
            key="integrity"
          >
            <DataIntegrityPanel />
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default PgVectorPage;
