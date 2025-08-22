import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Alert, 
  Tag,
  Statistic,
  Space,
  Divider,
  List
} from 'antd';
import { 
  WarningOutlined, 
  CheckCircleOutlined, 
  BranchesOutlined, 
  ClockCircleOutlined, 
  DatabaseOutlined,
  ArrowRightOutlined, 
  ArrowLeftOutlined, 
  ReloadOutlined, 
  EyeOutlined, 
  FileTextOutlined,
  ThunderboltOutlined, 
  RiseOutlined, 
  BarChartOutlined, 
  SettingOutlined, 
  UserOutlined,
  LoadingOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface ConflictRecord {
  id: string;
  table_name: string;
  object_id: string;
  conflict_type: 'update_update' | 'update_delete' | 'delete_update' | 'create_create' | 'schema_mismatch' | 'permission_denied';
  local_data: Record<string, any>;
  remote_data: Record<string, any>;
  created_at: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'data_conflict' | 'schema_conflict' | 'permission_conflict' | 'temporal_conflict' | 'semantic_conflict';
  auto_resolvable: boolean;
  confidence_score: number;
  suggested_resolution?: string;
}

interface ResolutionStrategy {
  strategy: 'last_writer_wins' | 'first_writer_wins' | 'client_wins' | 'server_wins' | 'merge' | 'manual';
  description: string;
  preview?: Record<string, any>;
  confidence: number;
}

interface ConflictStats {
  total_conflicts: number;
  auto_resolvable: number;
  manual_resolution_required: number;
  severity_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  type_distribution: Record<string, number>;
  average_confidence: number;
  resolution_rate: number;
}

const ConflictResolutionPage: React.FC = () => {
  const [conflicts, setConflicts] = useState<ConflictRecord[]>([]);
  const [conflictStats, setConflictStats] = useState<ConflictStats | null>(null);
  const [selectedConflict, setSelectedConflict] = useState<ConflictRecord | null>(null);
  const [resolutionStrategies, setResolutionStrategies] = useState<ResolutionStrategy[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchConflicts = async () => {
    try {
      // 模拟API调用
      const mockConflicts: ConflictRecord[] = [
        {
          id: 'conflict-001',
          table_name: 'users',
          object_id: 'user-123',
          conflict_type: 'update_update',
          local_data: {
            name: 'Alice Johnson',
            age: 31,
            email: 'alice.johnson@example.com',
            updated_at: '2023-12-01T10:30:00Z'
          },
          remote_data: {
            name: 'Alice Smith',
            age: 30,
            email: 'alice.smith@example.com',
            phone: '+1-555-0123',
            updated_at: '2023-12-01T10:25:00Z'
          },
          created_at: new Date(Date.now() - 300000).toISOString(),
          severity: 'medium',
          category: 'data_conflict',
          auto_resolvable: true,
          confidence_score: 0.75,
          suggested_resolution: 'merge'
        }
      ];
      setConflicts(mockConflicts);
    } catch (error) {
      console.error('获取冲突列表失败:', error);
    }
  };

  const fetchConflictStats = async () => {
    try {
      const mockStats: ConflictStats = {
        total_conflicts: 12,
        auto_resolvable: 8,
        manual_resolution_required: 4,
        severity_distribution: {
          'low': 3,
          'medium': 6,
          'high': 2,
          'critical': 1
        },
        category_distribution: {
          'data_conflict': 8,
          'schema_conflict': 2,
          'permission_conflict': 1,
          'temporal_conflict': 1
        },
        type_distribution: {
          'update_update': 7,
          'update_delete': 3,
          'create_create': 2
        },
        average_confidence: 0.72,
        resolution_rate: 0.85
      };
      setConflictStats(mockStats);
    } catch (error) {
      console.error('获取冲突统计失败:', error);
    }
  };

  const generateResolutionStrategies = (conflict: ConflictRecord): ResolutionStrategy[] => {
    const strategies: ResolutionStrategy[] = [];

    if (conflict.conflict_type === 'update_update') {
      strategies.push({
        strategy: 'last_writer_wins',
        description: '使用最新的更改',
        preview: conflict.remote_data,
        confidence: 0.8
      });

      strategies.push({
        strategy: 'merge',
        description: '智能合并两个版本的更改',
        preview: { ...conflict.local_data, ...conflict.remote_data },
        confidence: 0.7
      });

      strategies.push({
        strategy: 'client_wins',
        description: '保留本地更改',
        preview: conflict.local_data,
        confidence: 0.6
      });

      strategies.push({
        strategy: 'server_wins',
        description: '使用服务器版本',
        preview: conflict.remote_data,
        confidence: 0.6
      });
    }

    strategies.push({
      strategy: 'manual',
      description: '手动解决冲突',
      confidence: 1.0
    });

    return strategies;
  };

  const resolveConflict = async (conflictId: string, strategy: string, customData?: Record<string, any>) => {
    try {
      console.log(`解决冲突 ${conflictId} 使用策略 ${strategy}`, customData);
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 从列表中移除已解决的冲突
      setConflicts(prev => prev.filter(c => c.id !== conflictId));
      setSelectedConflict(null);
      
      // 刷新统计
      await fetchConflictStats();
    } catch (error) {
      console.error('解决冲突失败:', error);
    }
  };

  const handleConflictSelect = (conflict: ConflictRecord) => {
    setSelectedConflict(conflict);
    setResolutionStrategies(generateResolutionStrategies(conflict));
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchConflicts(),
        fetchConflictStats()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 10000); // 10秒刷新

    return () => clearInterval(interval);
  }, []);

  const getConflictTypeIcon = (type: string) => {
    switch (type) {
      case 'update_update': return <BranchesOutlined />;
      case 'update_delete': return <WarningOutlined />;
      case 'delete_update': return <WarningOutlined />;
      case 'create_create': return <DatabaseOutlined />;
      case 'schema_mismatch': return <FileTextOutlined />;
      case 'permission_denied': return <UserOutlined />;
      default: return <WarningOutlined />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'green';
      case 'medium': return 'orange';
      case 'high': return 'red';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'data_conflict': return 'blue';
      case 'schema_conflict': return 'purple';
      case 'permission_conflict': return 'red';
      case 'temporal_conflict': return 'orange';
      case 'semantic_conflict': return 'green';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>加载冲突数据中...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <WarningOutlined style={{ marginRight: '12px' }} />
          冲突解决中心
        </Title>
        <Space>
          <Button onClick={() => window.location.reload()} icon={<ReloadOutlined />}>
            刷新
          </Button>
        </Space>
      </div>

      {/* 冲突统计概览 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总冲突数"
              value={conflictStats?.total_conflicts || 0}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="可自动解决"
              value={conflictStats?.auto_resolvable || 0}
              prefix={<CheckCircleOutlined />}
              suffix={
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  / {conflictStats?.total_conflicts || 0}
                </Text>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="解决率"
              value={((conflictStats?.resolution_rate || 0) * 100).toFixed(1)}
              prefix={<RiseOutlined />}
              suffix="%"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均置信度"
              value={((conflictStats?.average_confidence || 0) * 100).toFixed(1)}
              prefix={<BarChartOutlined />}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        {/* 冲突列表 */}
        <Col span={8}>
          <Card title="冲突列表" style={{ height: '600px', overflow: 'auto' }}>
            <List
              dataSource={conflicts}
              renderItem={(conflict) => (
                <List.Item
                  onClick={() => handleConflictSelect(conflict)}
                  style={{
                    cursor: 'pointer',
                    backgroundColor: selectedConflict?.id === conflict.id ? '#f0f9ff' : 'white',
                    border: selectedConflict?.id === conflict.id ? '2px solid #1890ff' : '1px solid #f0f0f0',
                    borderRadius: '8px',
                    marginBottom: '8px',
                    padding: '12px'
                  }}
                >
                  <div style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {getConflictTypeIcon(conflict.conflict_type)}
                        <Text strong>{conflict.table_name}</Text>
                      </div>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <Tag color={getSeverityColor(conflict.severity)}>
                          {conflict.severity}
                        </Tag>
                        {conflict.auto_resolvable && (
                          <Tag color="success">
                            <ThunderboltOutlined />
                            自动
                          </Tag>
                        )}
                      </div>
                    </div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {conflict.object_id}
                    </Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      创建时间: {new Date(conflict.created_at).toLocaleString()}
                    </Text>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 冲突详情和解决方案 */}
        <Col span={16}>
          {selectedConflict ? (
            <Card>
              <Tabs defaultActiveKey="details">
                <TabPane
                  tab={
                    <span>
                      <FileTextOutlined />
                      冲突详情
                    </span>
                  }
                  key="details"
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Text strong>冲突ID:</Text>
                      <div style={{ 
                        fontFamily: 'monospace', 
                        backgroundColor: '#f5f5f5', 
                        padding: '4px', 
                        borderRadius: '4px',
                        marginBottom: '12px'
                      }}>
                        {selectedConflict.id}
                      </div>
                    </Col>
                    <Col span={12}>
                      <Text strong>对象ID:</Text>
                      <div style={{ marginBottom: '12px' }}>
                        {selectedConflict.object_id}
                      </div>
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Text strong>冲突类型:</Text>
                      <div style={{ marginBottom: '12px' }}>
                        <Tag>{selectedConflict.conflict_type}</Tag>
                      </div>
                    </Col>
                    <Col span={12}>
                      <Text strong>严重程度:</Text>
                      <div style={{ marginBottom: '12px' }}>
                        <Tag color={getSeverityColor(selectedConflict.severity)}>
                          {selectedConflict.severity}
                        </Tag>
                      </div>
                    </Col>
                  </Row>

                  <div style={{ marginTop: '16px' }}>
                    <Text strong>自动解决能力:</Text>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                      {selectedConflict.auto_resolvable ? (
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                      ) : (
                        <WarningOutlined style={{ color: '#fa8c16' }} />
                      )}
                      <Text>
                        {selectedConflict.auto_resolvable ? '可以自动解决' : '需要手动解决'}
                        （置信度: {(selectedConflict.confidence_score * 100).toFixed(1)}%）
                      </Text>
                    </div>
                  </div>
                </TabPane>

                <TabPane
                  tab={
                    <span>
                      <SettingOutlined />
                      解决方案
                    </span>
                  }
                  key="resolution"
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    {resolutionStrategies.map((strategy, index) => (
                      <Card key={index} size="small">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <Text strong>{strategy.description}</Text>
                          <Tag color={strategy.confidence > 0.8 ? 'green' : strategy.confidence > 0.6 ? 'orange' : 'red'}>
                            {(strategy.confidence * 100).toFixed(0)}% 置信度
                          </Tag>
                        </div>
                        <Text type="secondary" style={{ fontSize: '12px', marginBottom: '12px' }}>
                          策略: {strategy.strategy}
                        </Text>
                        {strategy.preview && (
                          <div style={{ 
                            backgroundColor: '#f5f5f5', 
                            padding: '8px', 
                            borderRadius: '4px',
                            marginBottom: '12px'
                          }}>
                            <Text strong style={{ fontSize: '12px' }}>预览结果:</Text>
                            <pre style={{ fontSize: '12px', margin: 0, whiteSpace: 'pre-wrap' }}>
                              {JSON.stringify(strategy.preview, null, 2)}
                            </pre>
                          </div>
                        )}
                        <Button
                          type="primary"
                          size="small"
                          onClick={() => resolveConflict(selectedConflict.id, strategy.strategy, strategy.preview)}
                        >
                          应用此解决方案
                        </Button>
                      </Card>
                    ))}
                  </div>
                </TabPane>

                <TabPane
                  tab={
                    <span>
                      <EyeOutlined />
                      数据预览
                    </span>
                  }
                  key="preview"
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Card 
                        size="small"
                        title={
                          <div style={{ display: 'flex', alignItems: 'center', color: '#1890ff' }}>
                            <ArrowLeftOutlined style={{ marginRight: '8px' }} />
                            本地数据
                          </div>
                        }
                      >
                        <pre style={{ 
                          fontSize: '12px', 
                          backgroundColor: '#f0f9ff', 
                          padding: '12px', 
                          borderRadius: '4px',
                          margin: 0,
                          whiteSpace: 'pre-wrap'
                        }}>
                          {JSON.stringify(selectedConflict.local_data, null, 2)}
                        </pre>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card 
                        size="small"
                        title={
                          <div style={{ display: 'flex', alignItems: 'center', color: '#52c41a' }}>
                            <ArrowRightOutlined style={{ marginRight: '8px' }} />
                            远程数据
                          </div>
                        }
                      >
                        <pre style={{ 
                          fontSize: '12px', 
                          backgroundColor: '#f6ffed', 
                          padding: '12px', 
                          borderRadius: '4px',
                          margin: 0,
                          whiteSpace: 'pre-wrap'
                        }}>
                          {JSON.stringify(selectedConflict.remote_data, null, 2)}
                        </pre>
                      </Card>
                    </Col>
                  </Row>
                </TabPane>
              </Tabs>
            </Card>
          ) : (
            <Card style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ textAlign: 'center', color: '#8c8c8c' }}>
                <WarningOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                <div>请从左侧列表选择一个冲突来查看详情</div>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default ConflictResolutionPage;