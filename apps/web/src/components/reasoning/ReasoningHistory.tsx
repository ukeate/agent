import React, { useState, useEffect } from 'react';
import {
  Table,
  Card,
  Button,
  Space,
  Tag,
  Tooltip,
  Drawer,
  Descriptions,
  Timeline,
  Popconfirm,
  Select,
  DatePicker,
  Input,
  Row,
  Col,
  Statistic,
  Typography
} from 'antd';
import {
  EyeOutlined,
  DeleteOutlined,
  FilterOutlined,
  ReloadOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';
import { useReasoningStore } from '../../stores/reasoningStore';

const { Text, Paragraph } = Typography;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { Search } = Input;

interface ReasoningChain {
  id: string;
  problem: string;
  strategy: string;
  conclusion?: string;
  confidence_score?: number;
  created_at: string;
  completed_at?: string;
  steps: Array<{
    step_number: number;
    step_type: string;
    content: string;
    confidence: number;
    duration_ms?: number;
  }>;
  branches?: Array<{
    id: string;
    reason: string;
    created_at: string;
  }>;
}

export const ReasoningHistory: React.FC = () => {
  const {
    reasoningHistory,
    isLoading,
    getReasoningHistory,
    deleteReasoningChain,
    setCurrentChain
  } = useReasoningStore();

  const [selectedChain, setSelectedChain] = useState<ReasoningChain | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [filteredData, setFilteredData] = useState<ReasoningChain[]>([]);
  const [filters, setFilters] = useState({
    strategy: '',
    dateRange: null as any,
    searchText: '',
    confidence: ''
  });

  useEffect(() => {
    getReasoningHistory();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [reasoningHistory, filters]);

  const applyFilters = () => {
    let filtered = [...reasoningHistory];

    if (filters.strategy) {
      filtered = filtered.filter(chain => chain.strategy === filters.strategy);
    }

    if (filters.searchText) {
      const searchLower = filters.searchText.toLowerCase();
      filtered = filtered.filter(chain => 
        chain.problem.toLowerCase().includes(searchLower) ||
        (chain.conclusion && chain.conclusion.toLowerCase().includes(searchLower))
      );
    }

    if (filters.confidence) {
      const [min, max] = filters.confidence.split('-').map(Number);
      filtered = filtered.filter(chain => {
        if (!chain.confidence_score) return false;
        return chain.confidence_score >= min/100 && chain.confidence_score <= max/100;
      });
    }

    if (filters.dateRange && filters.dateRange.length === 2) {
      const [start, end] = filters.dateRange;
      filtered = filtered.filter(chain => {
        const chainDate = new Date(chain.created_at);
        return chainDate >= start.toDate() && chainDate <= end.toDate();
      });
    }

    setFilteredData(filtered);
  };

  const handleViewChain = (chain: ReasoningChain) => {
    setSelectedChain(chain);
    setDrawerVisible(true);
  };

  const handleDeleteChain = async (chainId: string) => {
    await deleteReasoningChain(chainId);
    getReasoningHistory(); // 刷新列表
  };

  const handleLoadChain = (chain: ReasoningChain) => {
    setCurrentChain(chain);
    setDrawerVisible(false);
  };

  const columns = [
    {
      title: '问题',
      dataIndex: 'problem',
      key: 'problem',
      width: 300,
      render: (text: string) => (
        <div>
          <Text strong>{text.length > 50 ? `${text.substring(0, 50)}...` : text}</Text>
          <br />
          <Text type="secondary" className="text-xs">
            点击查看完整内容
          </Text>
        </div>
      )
    },
    {
      title: '推理策略',
      dataIndex: 'strategy',
      key: 'strategy',
      width: 120,
      render: (strategy: string) => {
        const strategyColors = {
          'ZERO_SHOT': 'blue',
          'FEW_SHOT': 'green',
          'AUTO_COT': 'purple'
        };
        const strategyNames = {
          'ZERO_SHOT': 'Zero-shot',
          'FEW_SHOT': 'Few-shot',
          'AUTO_COT': 'Auto-CoT'
        };
        return (
          <Tag color={strategyColors[strategy] || 'default'}>
            {strategyNames[strategy] || strategy}
          </Tag>
        );
      }
    },
    {
      title: '步骤数',
      key: 'steps_count',
      width: 80,
      render: (_, record: ReasoningChain) => (
        <div className="text-center">
          <div className="font-bold">{record.steps?.length || 0}</div>
          <Text type="secondary" className="text-xs">步骤</Text>
        </div>
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      width: 100,
      render: (confidence: number) => {
        if (!confidence) return <Text type="secondary">N/A</Text>;
        
        const percentage = Math.round(confidence * 100);
        const color = percentage >= 80 ? 'green' : percentage >= 60 ? 'orange' : 'red';
        
        return (
          <div className="text-center">
            <div className={`font-bold text-${color}-600`}>{percentage}%</div>
            <div className="text-xs text-gray-500">
              {percentage >= 80 ? '高' : percentage >= 60 ? '中' : '低'}
            </div>
          </div>
        );
      }
    },
    {
      title: '状态',
      key: 'status',
      width: 100,
      render: (_, record: ReasoningChain) => {
        const isCompleted = !!record.conclusion;
        const hasBranches = record.branches && record.branches.length > 0;
        
        return (
          <div className="text-center">
            {isCompleted ? (
              <Tag color="green" icon={<CheckCircleOutlined />}>
                已完成
              </Tag>
            ) : (
              <Tag color="orange" icon={<ExclamationCircleOutlined />}>
                未完成
              </Tag>
            )}
            {hasBranches && (
              <Tag color="blue" className="mt-1">
                {record.branches.length} 分支
              </Tag>
            )}
          </div>
        );
      }
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date: string) => (
        <div>
          <div>{new Date(date).toLocaleDateString()}</div>
          <Text type="secondary" className="text-xs">
            {new Date(date).toLocaleTimeString()}
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record: ReasoningChain) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="primary"
              icon={<EyeOutlined />}
              size="small"
              onClick={() => handleViewChain(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Popconfirm
              title="确定删除这个推理链吗？"
              onConfirm={() => handleDeleteChain(record.id)}
              okText="确定"
              cancelText="取消"
            >
              <Button
                danger
                icon={<DeleteOutlined />}
                size="small"
              />
            </Popconfirm>
          </Tooltip>
        </Space>
      )
    }
  ];

  const getStatistics = () => {
    const total = filteredData.length;
    const completed = filteredData.filter(c => c.conclusion).length;
    const avgConfidence = filteredData.reduce((sum, c) => sum + (c.confidence_score || 0), 0) / total || 0;
    const totalSteps = filteredData.reduce((sum, c) => sum + (c.steps?.length || 0), 0);
    
    return { total, completed, avgConfidence, totalSteps };
  };

  const stats = getStatistics();

  return (
    <div className="reasoning-history">
      {/* 统计面板 */}
      <Row gutter={16} className="mb-4">
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="推理链总数"
              value={stats.total}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="完成数量"
              value={stats.completed}
              suffix={`/ ${stats.total}`}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="平均置信度"
              value={stats.avgConfidence * 100}
              precision={1}
              suffix="%"
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card size="small">
            <Statistic
              title="总推理步骤"
              value={stats.totalSteps}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 过滤器 */}
      <Card className="mb-4" size="small">
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <Search
              placeholder="搜索问题或结论"
              value={filters.searchText}
              onChange={(e) => setFilters({...filters, searchText: e.target.value})}
              allowClear
            />
          </Col>
          <Col span={4}>
            <Select
              placeholder="推理策略"
              value={filters.strategy}
              onChange={(value) => setFilters({...filters, strategy: value})}
              allowClear
              style={{ width: '100%' }}
            >
              <Option value="ZERO_SHOT">Zero-shot</Option>
              <Option value="FEW_SHOT">Few-shot</Option>
              <Option value="AUTO_COT">Auto-CoT</Option>
            </Select>
          </Col>
          <Col span={4}>
            <Select
              placeholder="置信度范围"
              value={filters.confidence}
              onChange={(value) => setFilters({...filters, confidence: value})}
              allowClear
              style={{ width: '100%' }}
            >
              <Option value="80-100">高 (80-100%)</Option>
              <Option value="60-79">中 (60-79%)</Option>
              <Option value="0-59">低 (0-59%)</Option>
            </Select>
          </Col>
          <Col span={6}>
            <RangePicker
              value={filters.dateRange}
              onChange={(dates) => setFilters({...filters, dateRange: dates})}
              placeholder={['开始日期', '结束日期']}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4}>
            <Space>
              <Button
                icon={<FilterOutlined />}
                onClick={() => setFilters({
                  strategy: '',
                  dateRange: null,
                  searchText: '',
                  confidence: ''
                })}
              >
                清除过滤
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => getReasoningHistory()}
              >
                刷新
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 推理链列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredData}
          rowKey="id"
          loading={isLoading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `显示 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
            pageSizeOptions: ['10', '20', '50']
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 详情抽屉 */}
      <Drawer
        title="推理链详情"
        width={800}
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        extra={
          <Space>
            <Button
              type="primary"
              onClick={() => handleLoadChain(selectedChain!)}
            >
              加载到工作区
            </Button>
          </Space>
        }
      >
        {selectedChain && (
          <div>
            {/* 基本信息 */}
            <Descriptions title="基本信息" bordered size="small" className="mb-4">
              <Descriptions.Item label="问题" span={3}>
                {selectedChain.problem}
              </Descriptions.Item>
              <Descriptions.Item label="推理策略">
                <Tag color="blue">{selectedChain.strategy}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="置信度">
                {selectedChain.confidence_score ? 
                  `${(selectedChain.confidence_score * 100).toFixed(1)}%` : 
                  'N/A'
                }
              </Descriptions.Item>
              <Descriptions.Item label="步骤数">
                {selectedChain.steps?.length || 0}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>
                {new Date(selectedChain.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedChain.completed_at ? 
                  new Date(selectedChain.completed_at).toLocaleString() : 
                  '未完成'
                }
              </Descriptions.Item>
            </Descriptions>

            {/* 结论 */}
            {selectedChain.conclusion && (
              <Card title="结论" size="small" className="mb-4">
                <Paragraph>{selectedChain.conclusion}</Paragraph>
              </Card>
            )}

            {/* 推理步骤 */}
            <Card title="推理步骤" size="small" className="mb-4">
              <Timeline>
                {selectedChain.steps?.map((step, index) => (
                  <Timeline.Item
                    key={index}
                    color={step.confidence >= 0.8 ? 'green' : step.confidence >= 0.6 ? 'blue' : 'red'}
                  >
                    <div>
                      <Text strong>步骤 {step.step_number} - {step.step_type}</Text>
                      <Tag color="blue" className="ml-2">
                        置信度: {(step.confidence * 100).toFixed(1)}%
                      </Tag>
                      {step.duration_ms && (
                        <Tag color="green" className="ml-1">
                          {step.duration_ms}ms
                        </Tag>
                      )}
                    </div>
                    <div className="mt-2">
                      <Paragraph className="mb-1">{step.content}</Paragraph>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>

            {/* 分支信息 */}
            {selectedChain.branches && selectedChain.branches.length > 0 && (
              <Card title="推理分支" size="small">
                {selectedChain.branches.map((branch, index) => (
                  <div key={branch.id} className="border-l-4 border-blue-500 pl-4 mb-3">
                    <Text strong>分支 {index + 1}</Text>
                    <div className="text-gray-600">原因: {branch.reason}</div>
                    <div className="text-gray-400 text-sm">
                      创建时间: {new Date(branch.created_at).toLocaleString()}
                    </div>
                  </div>
                ))}
              </Card>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
};