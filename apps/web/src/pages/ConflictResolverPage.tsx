import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tag,
  Alert,
  Space,
  Typography,
  Statistic,
  Form,
  Input,
  Select,
  Modal,
  message,
  Tabs,
  Progress,
  Badge,
  Tooltip,
  Timeline,
  Descriptions,
  Tree,
  Radio
} from 'antd';
import {
  ExclamationTriangleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  WarningOutlined,
  BugOutlined,
  SettingOutlined,
  ReloadOutlined,
  EyeOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

// 冲突类型枚举
enum ConflictType {
  RESOURCE_CONFLICT = 'resource_conflict',
  STATE_CONFLICT = 'state_conflict', 
  ASSIGNMENT_CONFLICT = 'assignment_conflict',
  DEPENDENCY_CONFLICT = 'dependency_conflict'
}

interface Conflict {
  conflict_id: string;
  conflict_type: ConflictType;
  description: string;
  involved_tasks: string[];
  involved_agents: string[];
  involved_resources: string[];
  severity: 'low' | 'medium' | 'high' | 'critical';
  detected_at: string;
  resolved: boolean;
  resolution_strategy?: string;
  resolution_result?: any;
  resolution_time?: string;
  auto_resolvable: boolean;
  impact_score: number;
}

interface ResolutionStrategy {
  key: string;
  name: string;
  description: string;
  applicable_conflicts: ConflictType[];
  success_rate: number;
  avg_resolution_time: number;
  cost: 'low' | 'medium' | 'high';
  side_effects: string[];
}

interface ConflictMetrics {
  total_conflicts: number;
  resolved_conflicts: number;
  auto_resolved: number;
  manual_resolved: number;
  avg_resolution_time: number;
  success_rate: number;
  conflicts_by_type: Record<ConflictType, number>;
}

const ConflictResolverPage: React.FC = () => {
  const [conflicts, setConflicts] = useState<Conflict[]>([]);
  const [metrics, setMetrics] = useState<ConflictMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [autoDetection, setAutoDetection] = useState(true);
  const [selectedConflict, setSelectedConflict] = useState<Conflict | null>(null);
  const [resolutionModalVisible, setResolutionModalVisible] = useState(false);
  const [form] = Form.useForm();

  // 解决策略定义
  const resolutionStrategies: ResolutionStrategy[] = [
    {
      key: 'priority_based',
      name: '基于优先级解决',
      description: '根据任务优先级选择保留高优先级任务',
      applicable_conflicts: [ConflictType.RESOURCE_CONFLICT, ConflictType.ASSIGNMENT_CONFLICT],
      success_rate: 85,
      avg_resolution_time: 2.5,
      cost: 'low',
      side_effects: ['可能延迟低优先级任务', '资源利用率下降']
    },
    {
      key: 'resource_optimization',
      name: '资源优化解决',
      description: '重新分配资源以消除冲突',
      applicable_conflicts: [ConflictType.RESOURCE_CONFLICT, ConflictType.STATE_CONFLICT],
      success_rate: 78,
      avg_resolution_time: 4.2,
      cost: 'medium',
      side_effects: ['需要重新计算分配', '可能影响系统性能']
    },
    {
      key: 'load_balancing',
      name: '负载均衡解决',
      description: '通过负载均衡策略分散冲突',
      applicable_conflicts: [ConflictType.RESOURCE_CONFLICT, ConflictType.ASSIGNMENT_CONFLICT],
      success_rate: 82,
      avg_resolution_time: 3.8,
      cost: 'medium',
      side_effects: ['增加网络通信', '可能降低局部性']
    },
    {
      key: 'dependency_reorder',
      name: '依赖重排解决',
      description: '重新排序任务依赖关系',
      applicable_conflicts: [ConflictType.DEPENDENCY_CONFLICT],
      success_rate: 90,
      avg_resolution_time: 1.8,
      cost: 'low',
      side_effects: ['执行顺序变更', '可能影响性能']
    },
    {
      key: 'rollback_and_retry',
      name: '回滚重试解决',
      description: '回滚到冲突前状态并重新执行',
      applicable_conflicts: [ConflictType.STATE_CONFLICT, ConflictType.DEPENDENCY_CONFLICT],
      success_rate: 75,
      avg_resolution_time: 6.5,
      cost: 'high',
      side_effects: ['数据回滚', '执行时间增加', '可能丢失部分进度']
    },
    {
      key: 'fairness_based',
      name: '公平性解决',
      description: '基于公平性原则分配资源',
      applicable_conflicts: [ConflictType.RESOURCE_CONFLICT, ConflictType.ASSIGNMENT_CONFLICT],
      success_rate: 88,
      avg_resolution_time: 3.2,
      cost: 'medium',
      side_effects: ['可能不是最优解', '整体效率略降']
    }
  ];

  // 生成模拟冲突数据
  const generateMockConflicts = (): Conflict[] => {
    const conflictTypes = Object.values(ConflictType);
    const severities: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];
    
    return Array.from({ length: 8 }, (_, i) => {
      const conflictType = conflictTypes[Math.floor(Math.random() * conflictTypes.length)];
      const severity = severities[Math.floor(Math.random() * severities.length)];
      const resolved = Math.random() > 0.4; // 60% 已解决
      
      return {
        conflict_id: `conflict_${String(i + 1).padStart(3, '0')}`,
        conflict_type: conflictType,
        description: generateConflictDescription(conflictType),
        involved_tasks: Array.from({ length: Math.floor(Math.random() * 3) + 1 }, (_, j) => `task_${i}_${j}`),
        involved_agents: Array.from({ length: Math.floor(Math.random() * 2) + 1 }, (_, j) => `agent_${i + j + 1}`),
        involved_resources: Array.from({ length: Math.floor(Math.random() * 2) + 1 }, (_, j) => `resource_${j + 1}`),
        severity,
        detected_at: new Date(Date.now() - Math.random() * 3600000).toISOString(),
        resolved,
        resolution_strategy: resolved ? resolutionStrategies[Math.floor(Math.random() * resolutionStrategies.length)].key : undefined,
        resolution_time: resolved ? new Date(Date.now() - Math.random() * 1800000).toISOString() : undefined,
        auto_resolvable: Math.random() > 0.3,
        impact_score: Math.floor(Math.random() * 80) + 20
      };
    });
  };

  const generateConflictDescription = (type: ConflictType): string => {
    const descriptions = {
      [ConflictType.RESOURCE_CONFLICT]: [
        'CPU资源超限：任务请求超过可用容量',
        '内存资源冲突：多个任务竞争相同内存区域',
        '磁盘I/O冲突：并发访问导致性能下降',
        '网络带宽不足：任务网络需求超过可用带宽'
      ],
      [ConflictType.STATE_CONFLICT]: [
        '状态不一致：节点间状态同步失败',
        '数据版本冲突：并发修改导致版本冲突',
        '锁状态异常：分布式锁状态不一致',
        '缓存一致性问题：缓存与存储数据不匹配'
      ],
      [ConflictType.ASSIGNMENT_CONFLICT]: [
        '重复分配：任务被分配给多个智能体',
        '分配不平衡：负载分配严重倾斜',
        '能力不匹配：任务分配给不具备相应能力的智能体',
        '地域冲突：任务分配违反地域约束'
      ],
      [ConflictType.DEPENDENCY_CONFLICT]: [
        '循环依赖：任务间存在循环依赖关系',
        '依赖缺失：任务依赖的前置任务不存在',
        '依赖超时：前置任务执行时间超过预期',
        '依赖版本不匹配：依赖的资源版本不兼容'
      ]
    };

    const typeDescriptions = descriptions[type];
    return typeDescriptions[Math.floor(Math.random() * typeDescriptions.length)];
  };

  // 计算指标
  const calculateMetrics = (conflictList: Conflict[]): ConflictMetrics => {
    const resolvedConflicts = conflictList.filter(c => c.resolved);
    const conflictsByType = conflictList.reduce((acc, conflict) => {
      acc[conflict.conflict_type] = (acc[conflict.conflict_type] || 0) + 1;
      return acc;
    }, {} as Record<ConflictType, number>);

    return {
      total_conflicts: conflictList.length,
      resolved_conflicts: resolvedConflicts.length,
      auto_resolved: resolvedConflicts.filter(c => c.auto_resolvable).length,
      manual_resolved: resolvedConflicts.filter(c => !c.auto_resolvable).length,
      avg_resolution_time: resolvedConflicts.length > 0 
        ? resolvedConflicts.reduce((sum, c) => {
            if (c.resolution_time && c.detected_at) {
              return sum + (new Date(c.resolution_time).getTime() - new Date(c.detected_at).getTime());
            }
            return sum;
          }, 0) / (resolvedConflicts.length * 1000 * 60) // 转换为分钟
        : 0,
      success_rate: conflictList.length > 0 ? (resolvedConflicts.length / conflictList.length) * 100 : 0,
      conflicts_by_type: conflictsByType
    };
  };

  // 自动检测冲突
  const detectConflicts = async () => {
    setLoading(true);
    try {
      // 模拟检测延迟
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // 生成新的冲突
      const newConflicts = Array.from({ length: Math.floor(Math.random() * 3) + 1 }, (_, i) => {
        const conflictTypes = Object.values(ConflictType);
        const conflictType = conflictTypes[Math.floor(Math.random() * conflictTypes.length)];
        const severity: ('low' | 'medium' | 'high' | 'critical') = 
          ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any;
        
        return {
          conflict_id: `conflict_${Date.now()}_${i}`,
          conflict_type: conflictType,
          description: generateConflictDescription(conflictType),
          involved_tasks: [`task_${Date.now()}_${i}`],
          involved_agents: [`agent_${Math.floor(Math.random() * 5) + 1}`],
          involved_resources: [`resource_${Math.floor(Math.random() * 3) + 1}`],
          severity,
          detected_at: new Date().toISOString(),
          resolved: false,
          auto_resolvable: Math.random() > 0.4,
          impact_score: Math.floor(Math.random() * 60) + 40
        };
      });

      setConflicts(prev => [...newConflicts, ...prev]);
      message.success(`检测到 ${newConflicts.length} 个新冲突`);
    } catch (error) {
      message.error('冲突检测失败');
    } finally {
      setLoading(false);
    }
  };

  // 解决冲突
  const resolveConflict = async (conflictId: string, strategy: string) => {
    setLoading(true);
    try {
      const strategyInfo = resolutionStrategies.find(s => s.key === strategy);
      if (!strategyInfo) {
        throw new Error('未知的解决策略');
      }

      // 模拟解决过程
      await new Promise(resolve => setTimeout(resolve, strategyInfo.avg_resolution_time * 1000));

      // 模拟成功率
      const success = Math.random() * 100 < strategyInfo.success_rate;
      
      if (success) {
        setConflicts(prev =>
          prev.map(conflict =>
            conflict.conflict_id === conflictId
              ? {
                  ...conflict,
                  resolved: true,
                  resolution_strategy: strategy,
                  resolution_time: new Date().toISOString(),
                  resolution_result: { strategy: strategyInfo.name, success: true }
                }
              : conflict
          )
        );
        message.success(`冲突 ${conflictId} 解决成功`);
      } else {
        message.error(`冲突 ${conflictId} 解决失败`);
      }
    } catch (error) {
      message.error(`解决冲突失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 批量自动解决
  const autoResolveConflicts = async () => {
    const autoResolvableConflicts = conflicts.filter(c => !c.resolved && c.auto_resolvable);
    
    if (autoResolvableConflicts.length === 0) {
      message.info('没有可自动解决的冲突');
      return;
    }

    setLoading(true);
    let resolvedCount = 0;

    for (const conflict of autoResolvableConflicts) {
      const applicableStrategies = resolutionStrategies.filter(s => 
        s.applicable_conflicts.includes(conflict.conflict_type)
      );
      
      if (applicableStrategies.length > 0) {
        // 选择成功率最高的策略
        const bestStrategy = applicableStrategies.reduce((best, current) => 
          current.success_rate > best.success_rate ? current : best
        );

        try {
          await new Promise(resolve => setTimeout(resolve, 500)); // 模拟处理时间
          
          const success = Math.random() * 100 < bestStrategy.success_rate;
          if (success) {
            setConflicts(prev =>
              prev.map(c =>
                c.conflict_id === conflict.conflict_id
                  ? {
                      ...c,
                      resolved: true,
                      resolution_strategy: bestStrategy.key,
                      resolution_time: new Date().toISOString(),
                      resolution_result: { strategy: bestStrategy.name, auto: true }
                    }
                  : c
              )
            );
            resolvedCount++;
          }
        } catch (error) {
          console.error(`Failed to resolve conflict ${conflict.conflict_id}:`, error);
        }
      }
    }

    setLoading(false);
    message.success(`自动解决了 ${resolvedCount}/${autoResolvableConflicts.length} 个冲突`);
  };

  // 初始化数据
  useEffect(() => {
    const initialConflicts = generateMockConflicts();
    setConflicts(initialConflicts);
    setMetrics(calculateMetrics(initialConflicts));
  }, []);

  // 更新指标
  useEffect(() => {
    setMetrics(calculateMetrics(conflicts));
  }, [conflicts]);

  // 自动检测循环
  useEffect(() => {
    if (!autoDetection) return;

    const interval = setInterval(() => {
      // 随机生成新冲突
      if (Math.random() > 0.7) {
        detectConflicts();
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [autoDetection]);

  // 冲突表格列
  const conflictColumns: ColumnsType<Conflict> = [
    {
      title: '冲突ID',
      dataIndex: 'conflict_id',
      key: 'conflict_id',
      render: (id: string) => <Text code copyable={{ text: id }}>{id.substring(9)}</Text>
    },
    {
      title: '类型',
      dataIndex: 'conflict_type',
      key: 'conflict_type',
      render: (type: ConflictType) => {
        const colors = {
          [ConflictType.RESOURCE_CONFLICT]: 'orange',
          [ConflictType.STATE_CONFLICT]: 'red',
          [ConflictType.ASSIGNMENT_CONFLICT]: 'blue',
          [ConflictType.DEPENDENCY_CONFLICT]: 'purple'
        };
        return <Tag color={colors[type]}>{type.replace('_', ' ').toUpperCase()}</Tag>;
      }
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => {
        const colors = { low: 'green', medium: 'orange', high: 'red', critical: 'purple' };
        return <Tag color={colors[severity as keyof typeof colors]}>{severity.toUpperCase()}</Tag>;
      }
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    },
    {
      title: '影响分数',
      dataIndex: 'impact_score',
      key: 'impact_score',
      render: (score: number) => (
        <Progress
          percent={score}
          size="small"
          status={score > 70 ? 'exception' : score > 40 ? 'normal' : 'success'}
          format={() => score}
        />
      )
    },
    {
      title: '涉及资源',
      key: 'resources',
      render: (_, record: Conflict) => (
        <Space wrap>
          <Badge count={record.involved_tasks.length} color="blue" title="任务数" />
          <Badge count={record.involved_agents.length} color="green" title="智能体数" />
          <Badge count={record.involved_resources.length} color="orange" title="资源数" />
        </Space>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record: Conflict) => (
        <Space direction="vertical" size="small">
          <Tag color={record.resolved ? 'success' : 'error'}>
            {record.resolved ? '已解决' : '待解决'}
          </Tag>
          {record.auto_resolvable && <Tag color="blue" size="small">可自动解决</Tag>}
        </Space>
      )
    },
    {
      title: '检测时间',
      dataIndex: 'detected_at',
      key: 'detected_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: Conflict) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedConflict(record);
              Modal.info({
                title: `冲突详情: ${record.conflict_id}`,
                width: 700,
                content: (
                  <div style={{ marginTop: 16 }}>
                    <Descriptions column={2} bordered size="small">
                      <Descriptions.Item label="类型">{record.conflict_type}</Descriptions.Item>
                      <Descriptions.Item label="严重程度">{record.severity}</Descriptions.Item>
                      <Descriptions.Item label="影响分数">{record.impact_score}</Descriptions.Item>
                      <Descriptions.Item label="自动解决">{record.auto_resolvable ? '是' : '否'}</Descriptions.Item>
                      <Descriptions.Item label="涉及任务" span={2}>
                        {record.involved_tasks.join(', ')}
                      </Descriptions.Item>
                      <Descriptions.Item label="涉及智能体" span={2}>
                        {record.involved_agents.join(', ')}
                      </Descriptions.Item>
                      <Descriptions.Item label="涉及资源" span={2}>
                        {record.involved_resources.join(', ')}
                      </Descriptions.Item>
                      <Descriptions.Item label="描述" span={2}>
                        {record.description}
                      </Descriptions.Item>
                      {record.resolved && (
                        <>
                          <Descriptions.Item label="解决策略">{record.resolution_strategy}</Descriptions.Item>
                          <Descriptions.Item label="解决时间">
                            {record.resolution_time ? new Date(record.resolution_time).toLocaleString() : '-'}
                          </Descriptions.Item>
                        </>
                      )}
                    </Descriptions>
                  </div>
                )
              });
            }}
          >
            详情
          </Button>
          {!record.resolved && (
            <Button
              size="small"
              type="primary"
              icon={<CheckCircleOutlined />}
              onClick={() => {
                setSelectedConflict(record);
                form.setFieldsValue({
                  strategy: resolutionStrategies.find(s => 
                    s.applicable_conflicts.includes(record.conflict_type)
                  )?.key
                });
                setResolutionModalVisible(true);
              }}
            >
              解决
            </Button>
          )}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>冲突解决器</Title>
      <Paragraph>
        分布式系统冲突检测与解决，支持资源冲突、状态冲突、分配冲突和依赖冲突的自动化处理。
      </Paragraph>

      {/* 系统概览 */}
      <Card title="冲突处理概览" style={{ marginBottom: 24 }}>
        {metrics && (
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="总冲突数"
                value={metrics.total_conflicts}
                prefix={<BugOutlined />}
                suffix="个"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="已解决"
                value={metrics.resolved_conflicts}
                prefix={<CheckCircleOutlined />}
                suffix="个"
                valueStyle={{ color: '#3f8600' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="解决成功率"
                value={metrics.success_rate}
                precision={1}
                suffix="%"
                prefix={<ThunderboltOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="平均解决时间"
                value={metrics.avg_resolution_time}
                precision={1}
                suffix="分钟"
                prefix={<ClockCircleOutlined />}
              />
            </Col>
          </Row>
        )}

        <div style={{ marginTop: 16 }}>
          <Space>
            <Button
              type="primary"
              icon={<SyncOutlined />}
              onClick={detectConflicts}
              loading={loading}
            >
              检测冲突
            </Button>
            <Button
              icon={<PlayCircleOutlined />}
              onClick={autoResolveConflicts}
              loading={loading}
            >
              自动解决
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => {
                const newConflicts = generateMockConflicts();
                setConflicts(newConflicts);
                message.success('数据已刷新');
              }}
            >
              刷新数据
            </Button>
            <span style={{ marginLeft: 16 }}>
              <Text>自动检测:</Text>
              <Button
                type="link"
                size="small"
                onClick={() => setAutoDetection(!autoDetection)}
              >
                {autoDetection ? '已开启' : '已关闭'}
              </Button>
            </span>
          </Space>
        </div>
      </Card>

      <Tabs defaultActiveKey="conflicts">
        <TabPane tab="活跃冲突" key="conflicts">
          <Card title="冲突列表">
            <Table
              columns={conflictColumns}
              dataSource={conflicts}
              rowKey="conflict_id"
              pagination={{ pageSize: 8 }}
              loading={loading}
              size="small"
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ padding: 16, background: '#fafafa' }}>
                    <Row gutter={16}>
                      <Col span={8}>
                        <Text strong>涉及任务:</Text>
                        <ul>
                          {record.involved_tasks.map(task => (
                            <li key={task}><Text code>{task}</Text></li>
                          ))}
                        </ul>
                      </Col>
                      <Col span={8}>
                        <Text strong>涉及智能体:</Text>
                        <ul>
                          {record.involved_agents.map(agent => (
                            <li key={agent}><Text code>{agent}</Text></li>
                          ))}
                        </ul>
                      </Col>
                      <Col span={8}>
                        <Text strong>涉及资源:</Text>
                        <ul>
                          {record.involved_resources.map(resource => (
                            <li key={resource}><Text code>{resource}</Text></li>
                          ))}
                        </ul>
                      </Col>
                    </Row>
                  </div>
                )
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="解决策略" key="strategies">
          <Card title="可用策略">
            <Row gutter={16}>
              {resolutionStrategies.map(strategy => (
                <Col span={12} key={strategy.key} style={{ marginBottom: 16 }}>
                  <Card size="small" title={strategy.name}>
                    <Descriptions size="small" column={1}>
                      <Descriptions.Item label="描述">
                        {strategy.description}
                      </Descriptions.Item>
                      <Descriptions.Item label="成功率">
                        <Progress 
                          percent={strategy.success_rate} 
                          size="small" 
                          format={percent => `${percent}%`}
                        />
                      </Descriptions.Item>
                      <Descriptions.Item label="平均时间">
                        {strategy.avg_resolution_time}秒
                      </Descriptions.Item>
                      <Descriptions.Item label="成本">
                        <Tag color={strategy.cost === 'low' ? 'green' : strategy.cost === 'medium' ? 'orange' : 'red'}>
                          {strategy.cost}
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="适用冲突">
                        <Space wrap>
                          {strategy.applicable_conflicts.map(type => (
                            <Tag key={type} size="small">{type.replace('_', ' ')}</Tag>
                          ))}
                        </Space>
                      </Descriptions.Item>
                      <Descriptions.Item label="副作用">
                        <ul style={{ margin: 0, paddingLeft: 16 }}>
                          {strategy.side_effects.map((effect, index) => (
                            <li key={index}><Text type="secondary" style={{ fontSize: '12px' }}>{effect}</Text></li>
                          ))}
                        </ul>
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </TabPane>

        <TabPane tab="统计分析" key="analytics">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="冲突类型分布" size="small">
                {metrics && Object.entries(metrics.conflicts_by_type).map(([type, count]) => (
                  <div key={type} style={{ marginBottom: 8 }}>
                    <Text>{type.replace('_', ' ').toUpperCase()}: {count}个</Text>
                    <Progress
                      percent={metrics.total_conflicts > 0 ? (count / metrics.total_conflicts) * 100 : 0}
                      size="small"
                      showInfo={false}
                    />
                  </div>
                ))}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="解决方式分布" size="small">
                {metrics && (
                  <>
                    <div style={{ marginBottom: 8 }}>
                      <Text>自动解决: {metrics.auto_resolved}个</Text>
                      <Progress
                        percent={metrics.resolved_conflicts > 0 ? (metrics.auto_resolved / metrics.resolved_conflicts) * 100 : 0}
                        size="small"
                        showInfo={false}
                        strokeColor="#52c41a"
                      />
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <Text>手动解决: {metrics.manual_resolved}个</Text>
                      <Progress
                        percent={metrics.resolved_conflicts > 0 ? (metrics.manual_resolved / metrics.resolved_conflicts) * 100 : 0}
                        size="small"
                        showInfo={false}
                        strokeColor="#1890ff"
                      />
                    </div>
                  </>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 冲突解决模态框 */}
      <Modal
        title={`解决冲突: ${selectedConflict?.conflict_id}`}
        visible={resolutionModalVisible}
        onCancel={() => {
          setResolutionModalVisible(false);
          form.resetFields();
          setSelectedConflict(null);
        }}
        onOk={async () => {
          try {
            const values = await form.validateFields();
            if (selectedConflict) {
              await resolveConflict(selectedConflict.conflict_id, values.strategy);
              setResolutionModalVisible(false);
              form.resetFields();
              setSelectedConflict(null);
            }
          } catch (error) {
            // 表单验证失败
          }
        }}
        confirmLoading={loading}
      >
        {selectedConflict && (
          <div>
            <Alert
              message={`冲突类型: ${selectedConflict.conflict_type.replace('_', ' ').toUpperCase()}`}
              description={selectedConflict.description}
              type="warning"
              style={{ marginBottom: 16 }}
            />
            
            <Form form={form} layout="vertical">
              <Form.Item
                label="解决策略"
                name="strategy"
                rules={[{ required: true, message: '请选择解决策略' }]}
              >
                <Radio.Group>
                  <Space direction="vertical">
                    {resolutionStrategies
                      .filter(strategy => strategy.applicable_conflicts.includes(selectedConflict.conflict_type))
                      .map(strategy => (
                        <Radio key={strategy.key} value={strategy.key}>
                          <div>
                            <Text strong>{strategy.name}</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              成功率: {strategy.success_rate}% | 
                              时间: {strategy.avg_resolution_time}s | 
                              成本: {strategy.cost}
                            </Text>
                          </div>
                        </Radio>
                      ))}
                  </Space>
                </Radio.Group>
              </Form.Item>
            </Form>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default ConflictResolverPage;