import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Button, Table, Tag, Space, Typography, Alert, Timeline, Tabs } from 'antd';
import { 
  TrophyOutlined, 
  ExperimentOutlined, 
  LineChartOutlined, 
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface EvaluationOverview {
  totalModels: number;
  runningEvaluations: number;
  completedToday: number;
  averageAccuracy: number;
  systemUtilization: number;
  queuedTasks: number;
}

interface RecentEvaluation {
  id: string;
  modelName: string;
  benchmark: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  accuracy: number;
  startTime: string;
  duration: number;
}

const ModelEvaluationOverviewPage: React.FC = () => {
  const [overview, setOverview] = useState<EvaluationOverview>({
    totalModels: 0,
    runningEvaluations: 0,
    completedToday: 0,
    averageAccuracy: 0,
    systemUtilization: 0,
    queuedTasks: 0
  });

  const [recentEvaluations, setRecentEvaluations] = useState<RecentEvaluation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadOverviewData();
    const interval = setInterval(loadOverviewData, 30000); // 30秒更新一次
    return () => clearInterval(interval);
  }, []);

  const loadOverviewData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用 - 在实际项目中替换为真实的API调用
      const overviewData: EvaluationOverview = {
        totalModels: 24,
        runningEvaluations: 3,
        completedToday: 15,
        averageAccuracy: 0.847,
        systemUtilization: 67,
        queuedTasks: 8
      };

      const evaluationsData: RecentEvaluation[] = [
        {
          id: 'eval_001',
          modelName: 'BERT-Large-Uncased',
          benchmark: 'GLUE',
          status: 'running',
          accuracy: 0.0,
          startTime: '2024-01-15 14:30:00',
          duration: 1800
        },
        {
          id: 'eval_002',
          modelName: 'GPT-3.5-Turbo',
          benchmark: 'MMLU',
          status: 'completed',
          accuracy: 0.892,
          startTime: '2024-01-15 13:45:00',
          duration: 3600
        },
        {
          id: 'eval_003',
          modelName: 'Claude-3-Sonnet',
          benchmark: 'HumanEval',
          status: 'completed',
          accuracy: 0.734,
          startTime: '2024-01-15 12:15:00',
          duration: 2700
        },
        {
          id: 'eval_004',
          modelName: 'Llama-2-70B',
          benchmark: 'SuperGLUE',
          status: 'failed',
          accuracy: 0.0,
          startTime: '2024-01-15 11:20:00',
          duration: 450
        },
        {
          id: 'eval_005',
          modelName: 'T5-3B',
          benchmark: 'GLUE',
          status: 'queued',
          accuracy: 0.0,
          startTime: '2024-01-15 15:00:00',
          duration: 0
        }
      ];

      setOverview(overviewData);
      setRecentEvaluations(evaluationsData);
    } catch (error) {
      console.error('加载概览数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusTag = (status: string) => {
    const statusConfig = {
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      queued: { color: 'default', text: '队列中' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const columns = [
    {
      title: '模型名称',
      dataIndex: 'modelName',
      key: 'modelName',
      width: 180,
    },
    {
      title: '基准测试',
      dataIndex: 'benchmark',
      key: 'benchmark',
      width: 120,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
      width: 100,
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => accuracy > 0 ? `${(accuracy * 100).toFixed(1)}%` : '-',
      width: 100,
    },
    {
      title: '开始时间',
      dataIndex: 'startTime',
      key: 'startTime',
      width: 150,
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => duration > 0 ? `${Math.floor(duration / 60)}分钟` : '-',
      width: 80,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: RecentEvaluation) => (
        <Space size="small">
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => console.log('查看详情', record.id)}
          >
            详情
          </Button>
          {record.status === 'completed' && (
            <Button 
              type="link" 
              size="small" 
              icon={<DownloadOutlined />}
              onClick={() => console.log('下载报告', record.id)}
            >
              报告
            </Button>
          )}
        </Space>
      ),
      width: 120,
    },
  ];

  const systemStatus = [
    {
      title: '系统状态',
      items: [
        { label: 'GPU利用率', value: `${overview.systemUtilization}%`, color: overview.systemUtilization > 80 ? 'red' : 'green' },
        { label: '内存使用', value: '14.2GB / 32GB', color: 'blue' },
        { label: '磁盘空间', value: '2.1TB / 5TB', color: 'green' },
        { label: '网络状态', value: '正常', color: 'green' }
      ]
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>模型评估系统总览</Title>
        <Text type="secondary">
          全面管理和监控AI模型的性能评估，支持多种基准测试和自动化评估流程
        </Text>
      </div>

      <Alert
        message="系统运行正常"
        description="所有评估服务正常运行，当前有3个评估任务正在执行中"
        type="success"
        showIcon
        closable
        style={{ marginBottom: '24px' }}
      />

      {/* 关键指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="注册模型数量"
              value={overview.totalModels}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="运行中评估"
              value={overview.runningEvaluations}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="今日完成"
              value={overview.completedToday}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card>
            <Statistic
              title="平均准确率"
              value={overview.averageAccuracy}
              precision={3}
              suffix="%"
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 系统资源使用 */}
        <Col xs={24} lg={8}>
          <Card 
            title="系统资源使用" 
            extra={
              <Button 
                type="link" 
                icon={<ReloadOutlined />} 
                onClick={loadOverviewData}
                loading={loading}
              >
                刷新
              </Button>
            }
          >
            <div style={{ marginBottom: '16px' }}>
              <Text>GPU利用率</Text>
              <Progress 
                percent={overview.systemUtilization} 
                status={overview.systemUtilization > 85 ? 'exception' : 'active'}
                strokeColor={{
                  from: '#108ee9',
                  to: '#87d068',
                }}
              />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <Text>队列任务数</Text>
              <div style={{ display: 'flex', alignItems: 'center', marginTop: '8px' }}>
                <Progress 
                  percent={(overview.queuedTasks / 20) * 100} 
                  showInfo={false}
                  strokeColor="#faad14"
                  style={{ flex: 1, marginRight: '8px' }}
                />
                <Text strong>{overview.queuedTasks}</Text>
              </div>
            </div>
            
            <Timeline size="small" style={{ marginTop: '16px' }}>
              <Timeline.Item color="green">系统启动完成</Timeline.Item>
              <Timeline.Item color="blue">加载预训练模型</Timeline.Item>
              <Timeline.Item color="orange">开始批量评估任务</Timeline.Item>
              <Timeline.Item>等待新的评估请求...</Timeline.Item>
            </Timeline>
          </Card>
        </Col>

        {/* 近期评估任务 */}
        <Col xs={24} lg={16}>
          <Card 
            title="近期评估任务" 
            extra={
              <Space>
                <Button 
                  type="primary" 
                  icon={<PlayCircleOutlined />}
                  onClick={() => console.log('启动新评估')}
                >
                  新建评估
                </Button>
                <Button 
                  icon={<PauseCircleOutlined />}
                  onClick={() => console.log('暂停所有评估')}
                >
                  暂停所有
                </Button>
              </Space>
            }
          >
            <Table
              dataSource={recentEvaluations}
              columns={columns}
              rowKey="id"
              pagination={{ 
                pageSize: 5,
                showSizeChanger: false,
                showQuickJumper: true,
                showTotal: (total, range) => `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
              }}
              loading={loading}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* 快速操作面板 */}
      <Card title="快速操作" style={{ marginTop: '16px' }}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<ExperimentOutlined />}
              onClick={() => console.log('创建基准测试')}
            >
              创建基准测试
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<TrophyOutlined />}
              onClick={() => console.log('模型对比分析')}
            >
              模型对比分析
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<DownloadOutlined />}
              onClick={() => console.log('导出评估报告')}
            >
              导出评估报告
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<LineChartOutlined />}
              onClick={() => console.log('性能趋势分析')}
            >
              性能趋势分析
            </Button>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ModelEvaluationOverviewPage;