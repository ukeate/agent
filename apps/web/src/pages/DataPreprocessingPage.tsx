import React, { useState, useEffect } from 'react';
import {
  Card,
  Typography,
  Button,
  Row,
  Col,
  Progress,
  Tag,
  Alert,
  Modal,
  Form,
  Select,
  Checkbox,
  Input,
  Switch,
  Slider,
  Collapse,
  Space,
  Statistic,
  message,
  Empty
} from 'antd';
import {
  PlayCircleOutlined,
  SettingOutlined,
  ReloadOutlined,
  FilterOutlined,
  ClearOutlined,
  CopyOutlined,
  SwapOutlined,
  StarOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  TrendingUpOutlined,
  CheckCircleOutlined,
  WarningOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;
const { Panel } = Collapse;
const { TextArea } = Input;

interface ProcessingRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  config: any;
  success_rate: number;
  avg_processing_time: number;
}

interface ProcessingJob {
  job_id: string;
  rule_ids: string[];
  total_records: number;
  processed_records: number;
  success_records: number;
  failed_records: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  start_time: string;
  estimated_completion?: string;
  error_message?: string;
}

interface ProcessingStats {
  total_processed_today: number;
  avg_quality_improvement: number;
  processing_speed: number;
  success_rate: number;
  active_rules: number;
  queue_size: number;
}

export default function DataPreprocessingPage() {
  const [rules, setRules] = useState<ProcessingRule[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [stats, setStats] = useState<ProcessingStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [configDialog, setConfigDialog] = useState(false);
  const [selectedRule, setSelectedRule] = useState<ProcessingRule | null>(null);
  const [startJobDialog, setStartJobDialog] = useState(false);
  const [newJob, setNewJob] = useState({
    rule_ids: [] as string[],
    record_filter: {
      source_id: '',
      status: '',
      min_quality_score: 0,
      limit: 1000
    }
  });

  const fetchRules = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/preprocessing/rules');
      if (response.ok) {
        const data = await response.json();
        setRules(data);
      }
    } catch (err) {
      setError('获取预处理规则失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchJobs = async () => {
    try {
      const response = await fetch('/api/v1/training-data/preprocessing/jobs');
      if (response.ok) {
        const data = await response.json();
        setJobs(data);
      }
    } catch (err) {
      console.error('获取处理任务失败:', err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/training-data/preprocessing/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('获取统计信息失败:', err);
    }
  };

  const handleRuleToggle = async (ruleId: string, enabled: boolean) => {
    try {
      const response = await fetch(`/api/v1/training-data/preprocessing/rules/${ruleId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      
      if (response.ok) {
        await fetchRules();
      } else {
        throw new Error('更新规则状态失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新规则状态失败');
    }
  };

  const handleStartJob = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/v1/training-data/preprocessing/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newJob)
      });
      
      if (response.ok) {
        setStartJobDialog(false);
        setNewJob({
          rule_ids: [],
          record_filter: {
            source_id: '',
            status: '',
            min_quality_score: 0,
            limit: 1000
          }
        });
        await fetchJobs();
        message.success('处理任务启动成功');
      } else {
        throw new Error('启动处理任务失败');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动处理任务失败');
    } finally {
      setLoading(false);
    }
  };

  const getRuleIcon = (ruleName: string) => {
    switch (ruleName.toLowerCase()) {
      case 'text_cleaning': return <ClearOutlined />;
      case 'deduplication': return <CopyOutlined />;
      case 'format_standardization': return <SwapOutlined />;
      case 'quality_filtering': return <StarOutlined />;
      case 'data_enrichment': return <ThunderboltOutlined />;
      default: return <FilterOutlined />;
    }
  };

  const getProgressStatus = (successRate: number) => {
    if (successRate >= 0.9) return 'success';
    if (successRate >= 0.7) return 'normal';
    return 'exception';
  };

  const getJobStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'running': return 'processing';
      default: return 'default';
    }
  };

  const getJobStatusText = (status: string) => {
    switch (status) {
      case 'completed': return '已完成';
      case 'failed': return '失败';
      case 'running': return '运行中';
      case 'pending': return '等待中';
      default: return status;
    }
  };

  const defaultRules: ProcessingRule[] = [
    {
      id: 'text_cleaning',
      name: '文本清理',
      description: '清理HTML标签、特殊字符、多余空格等',
      enabled: true,
      config: { remove_html: true, normalize_whitespace: true, remove_special_chars: false },
      success_rate: 0.95,
      avg_processing_time: 0.2
    },
    {
      id: 'deduplication',
      name: '去重处理',
      description: '基于内容相似度去除重复记录',
      enabled: true,
      config: { similarity_threshold: 0.95, hash_algorithm: 'sha256' },
      success_rate: 0.88,
      avg_processing_time: 0.5
    },
    {
      id: 'format_standardization',
      name: '格式标准化',
      description: '统一数据格式，标准化结构',
      enabled: true,
      config: { normalize_encoding: 'utf-8', standardize_dates: true },
      success_rate: 0.92,
      avg_processing_time: 0.3
    },
    {
      id: 'quality_filtering',
      name: '质量过滤',
      description: '过滤低质量和无效数据',
      enabled: true,
      config: { min_length: 10, max_length: 10000, language_filter: ['zh', 'en'] },
      success_rate: 0.85,
      avg_processing_time: 0.4
    },
    {
      id: 'data_enrichment',
      name: '数据丰富化',
      description: '添加元数据、语言检测、情感分析等',
      enabled: false,
      config: { language_detection: true, sentiment_analysis: false, entity_extraction: false },
      success_rate: 0.78,
      avg_processing_time: 1.2
    }
  ];

  useEffect(() => {
    setRules(defaultRules);
    setStats({
      total_processed_today: 1245,
      avg_quality_improvement: 0.23,
      processing_speed: 120,
      success_rate: 0.89,
      active_rules: 4,
      queue_size: 89
    });
    
    const mockJobs: ProcessingJob[] = [
      {
        job_id: 'job_001',
        rule_ids: ['text_cleaning', 'deduplication'],
        total_records: 1000,
        processed_records: 750,
        success_records: 720,
        failed_records: 30,
        status: 'running',
        progress: 75,
        start_time: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
        estimated_completion: new Date(Date.now() + 10 * 60 * 1000).toISOString()
      }
    ];
    setJobs(mockJobs);
    
    fetchRules();
    fetchJobs();
    fetchStats();
    
    const interval = setInterval(() => {
      fetchJobs();
      fetchStats();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>数据预处理</Title>
      <Text type="secondary" style={{ marginBottom: '24px', display: 'block' }}>
        配置和管理数据预处理规则，提升数据质量和标准化程度
      </Text>

      {error && (
        <Alert
          message={error}
          type="error"
          closable
          onClose={() => setError(null)}
          style={{ marginBottom: '16px' }}
        />
      )}

      {/* 统计概览 */}
      {stats && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="今日处理量"
                value={stats.total_processed_today}
                prefix={<DashboardOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="质量提升"
                value={stats.avg_quality_improvement * 100}
                precision={1}
                suffix="%"
                prefix={<TrendingUpOutlined />}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="条/分钟"
                value={stats.processing_speed}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="成功率"
                value={stats.success_rate * 100}
                precision={1}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="启用规则"
                value={stats.active_rules}
                prefix={<FilterOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={12} sm={8} md={4}>
            <Card>
              <Statistic
                title="待处理队列"
                value={stats.queue_size}
                prefix={<WarningOutlined />}
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={[24, 24]}>
        {/* 预处理规则配置 */}
        <Col xs={24} md={12}>
          <Card
            title="预处理规则"
            extra={
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => setStartJobDialog(true)}
                disabled={rules.filter(r => r.enabled).length === 0}
              >
                启动处理
              </Button>
            }
          >
            <Collapse>
              {rules.map((rule) => (
                <Panel
                  key={rule.id}
                  header={
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                      <Space>
                        {getRuleIcon(rule.id)}
                        <div>
                          <Text strong>{rule.name}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            成功率: {(rule.success_rate * 100).toFixed(1)}% | 
                            平均耗时: {rule.avg_processing_time}s
                          </Text>
                        </div>
                      </Space>
                      <Space>
                        <Tag color={rule.enabled ? 'green' : 'default'}>
                          {rule.enabled ? '启用' : '禁用'}
                        </Tag>
                        <Switch
                          checked={rule.enabled}
                          onChange={(checked) => handleRuleToggle(rule.id, checked)}
                          size="small"
                        />
                      </Space>
                    </div>
                  }
                >
                  <Text type="secondary" style={{ marginBottom: '16px', display: 'block' }}>
                    {rule.description}
                  </Text>
                  
                  <Title level={5}>性能指标</Title>
                  <div style={{ marginBottom: '16px' }}>
                    <Text>成功率: {(rule.success_rate * 100).toFixed(1)}%</Text>
                    <Progress
                      percent={rule.success_rate * 100}
                      status={getProgressStatus(rule.success_rate)}
                      style={{ margin: '8px 0' }}
                    />
                  </div>
                  
                  <Button
                    size="small"
                    icon={<SettingOutlined />}
                    onClick={() => {
                      setSelectedRule(rule);
                      setConfigDialog(true);
                    }}
                  >
                    配置参数
                  </Button>
                </Panel>
              ))}
            </Collapse>
          </Card>
        </Col>

        {/* 处理任务监控 */}
        <Col xs={24} md={12}>
          <Card
            title="处理任务"
            extra={
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  fetchJobs();
                  fetchStats();
                }}
                size="small"
              >
                刷新
              </Button>
            }
          >
            {jobs.length === 0 ? (
              <Empty description="暂无处理任务" />
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {jobs.map((job) => (
                  <Card key={job.job_id} size="small" style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <div>
                        <Text strong>任务 #{job.job_id.substring(0, 8)}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(job.start_time).toLocaleString()}
                        </Text>
                      </div>
                      <Tag color={getJobStatusColor(job.status)}>
                        {getJobStatusText(job.status)}
                      </Tag>
                    </div>
                    
                    <div style={{ marginBottom: '16px' }}>
                      <Text>进度: {job.processed_records}/{job.total_records} ({job.progress.toFixed(1)}%)</Text>
                      <Progress
                        percent={job.progress}
                        style={{ marginBottom: '8px' }}
                      />
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text style={{ color: '#52c41a' }}>
                          成功: {job.success_records}
                        </Text>
                        {job.failed_records > 0 && (
                          <Text style={{ color: '#ff4d4f' }}>
                            失败: {job.failed_records}
                          </Text>
                        )}
                      </div>
                    </div>
                    
                    <Space wrap>
                      {job.rule_ids.map(ruleId => (
                        <Tag key={ruleId} color="blue">
                          {rules.find(r => r.id === ruleId)?.name || ruleId}
                        </Tag>
                      ))}
                    </Space>
                    
                    {job.estimated_completion && (
                      <Text type="secondary" style={{ display: 'block', marginTop: '8px', fontSize: '12px' }}>
                        预计完成: {new Date(job.estimated_completion).toLocaleString()}
                      </Text>
                    )}
                  </Card>
                ))}
              </Space>
            )}
          </Card>
        </Col>
      </Row>

      {/* 启动处理任务对话框 */}
      <Modal
        title="启动数据处理任务"
        open={startJobDialog}
        onCancel={() => setStartJobDialog(false)}
        onOk={handleStartJob}
        okText="启动处理"
        cancelText="取消"
        okButtonProps={{ disabled: newJob.rule_ids.length === 0 || loading }}
        confirmLoading={loading}
      >
        <div style={{ marginBottom: '16px' }}>
          <Title level={5}>选择处理规则</Title>
          <Card size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              {rules.filter(r => r.enabled).map((rule) => (
                <Checkbox
                  key={rule.id}
                  checked={newJob.rule_ids.includes(rule.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setNewJob(prev => ({
                        ...prev,
                        rule_ids: [...prev.rule_ids, rule.id]
                      }));
                    } else {
                      setNewJob(prev => ({
                        ...prev,
                        rule_ids: prev.rule_ids.filter(id => id !== rule.id)
                      }));
                    }
                  }}
                >
                  <Space>
                    {getRuleIcon(rule.id)}
                    <div>
                      <Text>{rule.name}</Text>
                      <br />
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        成功率: {(rule.success_rate * 100).toFixed(1)}%
                      </Text>
                    </div>
                  </Space>
                </Checkbox>
              ))}
            </Space>
          </Card>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <Title level={5}>数据过滤条件</Title>
          <Input
            type="number"
            addonBefore="处理记录数限制"
            value={newJob.record_filter.limit}
            onChange={(e) => setNewJob(prev => ({
              ...prev,
              record_filter: { ...prev.record_filter, limit: parseInt(e.target.value) || 1000 }
            }))}
            placeholder="限制本次处理的记录数量"
          />
        </div>
        
        <div>
          <Title level={5}>最低质量分数过滤</Title>
          <Slider
            value={newJob.record_filter.min_quality_score}
            onChange={(value) => setNewJob(prev => ({
              ...prev,
              record_filter: { ...prev.record_filter, min_quality_score: value }
            }))}
            min={0}
            max={1}
            step={0.1}
            marks={{
              0: '0',
              0.5: '0.5',
              1: '1.0'
            }}
          />
        </div>
      </Modal>

      {/* 规则配置对话框 */}
      <Modal
        title={`配置规则: ${selectedRule?.name}`}
        open={configDialog}
        onCancel={() => setConfigDialog(false)}
        onOk={() => {
          setConfigDialog(false);
          message.success('配置保存成功');
        }}
        okText="保存配置"
        cancelText="取消"
      >
        {selectedRule && (
          <div>
            <Text type="secondary" style={{ marginBottom: '16px', display: 'block' }}>
              {selectedRule.description}
            </Text>
            
            <TextArea
              rows={8}
              defaultValue={JSON.stringify(selectedRule.config, null, 2)}
              placeholder="修改规则的配置参数"
            />
          </div>
        )}
      </Modal>
    </div>
  );
}