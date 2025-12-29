import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Button, Table, Tag, Space, Typography, Alert, Timeline, Tabs, message } from 'antd';
import { 
import { logger } from '../utils/logger'
  TrophyOutlined, 
  ExperimentOutlined, 
  LineChartOutlined, 
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';
import { modelEvaluationService } from '../services/modelEvaluationService';
import { healthService, SystemMetrics } from '../services/healthService';

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
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    loadOverviewData();
    const interval = setInterval(loadOverviewData, 30000); // 30秒更新一次
    return () => clearInterval(interval);
  }, []);

  const loadOverviewData = async () => {
    try {
      setLoading(true);
      const [history, metrics] = await Promise.all([
        modelEvaluationService.getEvaluationHistory({ limit: 10 }),
        healthService.getSystemMetrics().catch(() => null)
      ]);
      setSystemMetrics(metrics);

      const runningEvals = history.evaluations.filter(e => e.status === 'in_progress' || e.status === 'running').length;
      const completedToday = history.evaluations.filter(e => {
        const evalDate = new Date(e.timestamp).toDateString();
        const today = new Date().toDateString();
        return evalDate === today && e.status === 'completed';
      }).length;

      const validScores = history.evaluations.filter(e => e.score > 0);
      const avgAccuracy = validScores.length === 0
        ? 0
        : validScores.reduce((sum, e) => sum + e.score, 0) / validScores.length;

      const cpuUsage = metrics?.cpu_usage ?? 0;
      const utilization = Math.round(cpuUsage * 10) / 10;

      setOverview({
        totalModels: history.total,
        runningEvaluations: runningEvals,
        completedToday,
        averageAccuracy: avgAccuracy,
        systemUtilization: utilization,
        queuedTasks: history.evaluations.filter(e => e.status === 'queued').length
      });

      const statusMap: Record<string, RecentEvaluation['status']> = {
        in_progress: 'running',
        running: 'running',
        completed: 'completed',
        failed: 'failed',
        queued: 'queued'
      };
      const evaluationsData: RecentEvaluation[] = history.evaluations.slice(0, 10).map(evaluation => ({
        id: evaluation.id,
        modelName: evaluation.model_name,
        benchmark: evaluation.benchmark || 'unknown',
        status: statusMap[evaluation.status] || 'queued',
        accuracy: evaluation.score,
        startTime: evaluation.timestamp,
        duration: evaluation.duration || 0
      }));

      setRecentEvaluations(evaluationsData);
    } catch (error) {
      logger.error('加载评估概览数据失败:', error);
      message.error('无法加载评估数据');
      setOverview({
        totalModels: 0,
        runningEvaluations: 0,
        completedToday: 0,
        averageAccuracy: 0,
        systemUtilization: 0,
        queuedTasks: 0
      });
      setRecentEvaluations([]);
      setSystemMetrics(null);
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
            onClick={() => logger.log('查看详情', record.id)}
          >
            详情
          </Button>
          {record.status === 'completed' && (
            <Button 
              type="link" 
              size="small" 
              icon={<DownloadOutlined />}
              onClick={() => logger.log('下载报告', record.id)}
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
        { label: 'CPU使用率', value: systemMetrics?.cpu_usage != null ? `${systemMetrics.cpu_usage.toFixed(1)}%` : '-', color: systemMetrics?.cpu_usage && systemMetrics.cpu_usage > 80 ? 'red' : 'green' },
        { label: '内存使用率', value: systemMetrics?.memory_usage != null ? `${systemMetrics.memory_usage.toFixed(1)}%` : '-', color: systemMetrics?.memory_usage && systemMetrics.memory_usage > 80 ? 'red' : 'green' },
        { label: '磁盘使用率', value: systemMetrics?.disk_usage != null ? `${systemMetrics.disk_usage.toFixed(1)}%` : '-', color: systemMetrics?.disk_usage && systemMetrics.disk_usage > 85 ? 'red' : 'green' },
        { label: '请求错误率', value: systemMetrics?.error_rate != null ? `${systemMetrics.error_rate.toFixed(2)}%` : '-', color: systemMetrics?.error_rate && systemMetrics.error_rate > 5 ? 'red' : 'green' }
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
        message={overview.runningEvaluations > 0 ? '评估任务运行中' : '当前无运行中的评估任务'}
        description={`今日已完成 ${overview.completedToday} 个评估，队列中 ${overview.queuedTasks} 个任务`}
        type={systemMetrics?.error_rate && systemMetrics.error_rate > 5 ? 'error' : systemMetrics?.error_rate && systemMetrics.error_rate > 1 ? 'warning' : 'success'}
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
              value={overview.averageAccuracy * 100}
              precision={1}
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
              <Text>CPU使用率</Text>
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
              {recentEvaluations.length > 0 ? recentEvaluations.slice(0, 4).map((evaluation) => (
                <Timeline.Item
                  key={evaluation.id}
                  color={evaluation.status === 'completed' ? 'green' : evaluation.status === 'failed' ? 'red' : 'blue'}
                >
                  {evaluation.modelName} - {evaluation.status === 'completed' ? '已完成' : evaluation.status === 'failed' ? '失败' : evaluation.status === 'running' ? '运行中' : '排队中'}
                </Timeline.Item>
              )) : (
                <Timeline.Item>暂无评估记录</Timeline.Item>
              )}
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
                  onClick={() => logger.log('启动新评估')}
                >
                  新建评估
                </Button>
                <Button 
                  icon={<PauseCircleOutlined />}
                  onClick={() => logger.log('暂停所有评估')}
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
              onClick={() => message.info('请在模型评估管理页创建基准测试')}
            >
              创建基准测试
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<TrophyOutlined />}
              onClick={() => message.info('请在模型评估管理页进行模型对比')}
            >
              模型对比分析
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<DownloadOutlined />}
              onClick={() => message.info('请在评估结果详情中导出报告')}
            >
              导出评估报告
            </Button>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Button 
              type="dashed" 
              block 
              icon={<LineChartOutlined />}
              onClick={() => message.info('请在评估管理页查看性能趋势')}
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
