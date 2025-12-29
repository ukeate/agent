import React, { useState, useEffect } from 'react';
import {
  Card,
  Button,
  Table,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Row,
  Col,
  Tabs,
  Alert,
  Badge,
  Tag,
  Statistic,
  Progress,
  notification,
  Space,
  Tooltip,
  DatePicker,
  Modal
} from 'antd';
import {
  AlertOutlined,
  BarChartOutlined,
  SettingOutlined,
  MonitorOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  SearchOutlined,
  SafetyCertificateOutlined
} from '@ant-design/icons';
import anomalyDetectionService, {
  DetectAnomaliesRequest,
  SRMCheckRequest,
  DataQualityCheckRequest,
  ConfigureDetectionRequest,
  RealTimeMonitorRequest,
  Anomaly
} from '../services/anomalyDetectionService';

const { Option } = Select;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

const AnomalyDetectionPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [anomalyTypes, setAnomalyTypes] = useState<any[]>([]);
  const [detectionMethods, setDetectionMethods] = useState<any[]>([]);
  const [detectForm] = Form.useForm();
  const [srmForm] = Form.useForm();
  const [qualityForm] = Form.useForm();
  const [configForm] = Form.useForm();
  const [activeTab, setActiveTab] = useState('detect');
  const [statistics, setStatistics] = useState({
    totalAnomalies: 0,
    highSeverity: 0,
    mediumSeverity: 0,
    lowSeverity: 0,
    qualityScore: null as number | null
  });
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [monitoringStatus, setMonitoringStatus] = useState<any>(null);
  const [summaryData, setSummaryData] = useState<any>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [typesResponse, methodsResponse, healthResponse] = await Promise.all([
        anomalyDetectionService.getAnomalyTypes(),
        anomalyDetectionService.getDetectionMethods(),
        anomalyDetectionService.healthCheck()
      ]);
      
      setAnomalyTypes(typesResponse.types || []);
      setDetectionMethods(methodsResponse.methods || []);
      setHealthStatus(healthResponse);
    } catch (error) {
      notification.error({
        message: '加载初始数据失败',
        description: '无法获取异常类型和检测方法'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDetectAnomalies = async (values: any) => {
    try {
      setLoading(true);
      
      const request: DetectAnomaliesRequest = {
        experiment_id: values.experiment_id,
        metric_name: values.metric_name,
        values: values.values.split(',').map((v: string) => parseFloat(v.trim())),
        timestamps: values.timestamps ? values.timestamps.split(',') : undefined,
        variant: values.variant,
        methods: values.methods
      };

      const response = await anomalyDetectionService.detectAnomalies(request);
      
      if (response.success) {
        setAnomalies(response.anomalies);
        setStatistics(prev => ({
          ...prev,
          totalAnomalies: response.total_count,
          highSeverity: response.anomalies.filter(a => a.severity === 'high').length,
          mediumSeverity: response.anomalies.filter(a => a.severity === 'medium').length,
          lowSeverity: response.anomalies.filter(a => a.severity === 'low').length
        }));
        
        notification.success({
          message: '异常检测完成',
          description: `检测到 ${response.total_count} 个异常，使用方法: ${response.methods_used.join(', ')}`
        });
      }
    } catch (error) {
      notification.error({
        message: '异常检测失败',
        description: '请检查输入参数和网络连接'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSRMCheck = async (values: any) => {
    try {
      setLoading(true);
      
      const request: SRMCheckRequest = {
        experiment_id: values.experiment_id,
        control_count: values.control_count,
        treatment_count: values.treatment_count,
        expected_ratio: values.expected_ratio || 0.5
      };

      const response = await anomalyDetectionService.checkSampleRatioMismatch(request);
      
      if (response.success) {
        if (response.has_srm && response.anomaly) {
          notification.warning({
            message: '检测到样本比例不匹配',
            description: `观察比例: ${response.anomaly.observed_ratio.toFixed(3)}, 期望比例: ${response.anomaly.expected_ratio.toFixed(3)}`
          });
        } else {
          notification.success({
            message: '样本比例正常',
            description: response.message
          });
        }
      }
    } catch (error) {
      notification.error({
        message: 'SRM检查失败',
        description: '请检查输入参数'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDataQualityCheck = async (values: any) => {
    try {
      setLoading(true);
      
      const request: DataQualityCheckRequest = {
        experiment_id: values.experiment_id,
        missing_rate: values.missing_rate / 100,
        duplicate_rate: values.duplicate_rate / 100,
        null_count: values.null_count,
        total_count: values.total_count
      };

      const response = await anomalyDetectionService.checkDataQuality(request);
      
      if (response.success) {
        setStatistics(prev => ({
          ...prev,
          qualityScore: Math.round(response.quality_score * 100)
        }));
        
        if (response.has_issues) {
          notification.warning({
            message: '检测到数据质量问题',
            description: `质量得分: ${Math.round(response.quality_score * 100)}%, 发现 ${response.quality_issues.length} 个问题`
          });
        } else {
          notification.success({
            message: '数据质量良好',
            description: `质量得分: ${Math.round(response.quality_score * 100)}%`
          });
        }
      }
    } catch (error) {
      notification.error({
        message: '数据质量检查失败',
        description: '请检查输入参数'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleConfigureDetection = async (values: any) => {
    try {
      setLoading(true);
      
      const request: ConfigureDetectionRequest = {
        methods: values.methods,
        sensitivity: values.sensitivity,
        window_size: values.window_size,
        min_samples: values.min_samples,
        z_threshold: values.z_threshold,
        iqr_multiplier: values.iqr_multiplier,
        enable_seasonal: values.enable_seasonal,
        enable_trend: values.enable_trend
      };

      const response = await anomalyDetectionService.configureDetection(request);
      
      if (response.success) {
        notification.success({
          message: '检测配置已更新',
          description: response.message
        });
      }
    } catch (error) {
      notification.error({
        message: '配置更新失败',
        description: '请检查输入参数'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleGetAnomalySummary = async (values: any) => {
    try {
      setLoading(true);
      
      const response = await anomalyDetectionService.getAnomalySummary(
        values.experiment_id,
        values.start_time,
        values.end_time
      );
      
      if (response.success) {
        setSummaryData(response.summary);
        notification.success({
          message: '异常摘要获取成功',
          description: `实验 ${values.experiment_id} 的异常摘要已加载`
        });
      }
    } catch (error) {
      notification.error({
        message: '获取异常摘要失败',
        description: '请检查实验ID和时间范围'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSetupRealTimeMonitoring = async (values: any) => {
    try {
      setLoading(true);
      
      const request: RealTimeMonitorRequest = {
        experiment_id: values.experiment_id,
        metrics: values.metrics,
        check_interval: values.check_interval || 60,
        alert_threshold: values.alert_threshold || 'medium'
      };

      const response = await anomalyDetectionService.setupRealTimeMonitoring(request);
      
      if (response.success) {
        setMonitoringStatus({ 
          active: true, 
          experiment_id: values.experiment_id,
          started_at: new Date().toISOString()
        });
        notification.success({
          message: '实时监控已启动',
          description: response.message || `实验 ${values.experiment_id} 的实时监控已开始`
        });
      }
    } catch (error) {
      notification.error({
        message: '启动实时监控失败',
        description: '请检查实验配置参数'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleBatchDetectAnomalies = async (values: any) => {
    try {
      setLoading(true);
      
      const experiments = values.experiments.split(',').map((id: string) => id.trim());
      const metrics = values.metrics.split(',').map((metric: string) => metric.trim());

      const response = await anomalyDetectionService.batchDetectAnomalies(experiments, metrics);
      
      if (response.success) {
        setBatchResults([response.results]);
        
        // 计算总异常数量
        const totalAnomalies = Object.values(response.results).reduce((total, expResults) => {
          return total + Object.values(expResults as any).reduce((expTotal, metricResult) => {
            return expTotal + (metricResult as any).anomaly_count;
          }, 0);
        }, 0);
        
        notification.success({
          message: '批量异常检测完成',
          description: `检测了 ${experiments.length} 个实验的 ${metrics.length} 个指标，发现 ${totalAnomalies} 个异常`
        });
      }
    } catch (error) {
      notification.error({
        message: '批量检测失败',
        description: '请检查实验ID列表和指标名称'
      });
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  const anomaliesColumns = [
    {
      title: '时间戳',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleString('zh-CN')
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => 
        <Badge color={getSeverityColor(severity)} text={severity.toUpperCase()} />
    },
    {
      title: '指标',
      dataIndex: 'metric',
      key: 'metric'
    },
    {
      title: '观察值',
      dataIndex: 'observed',
      key: 'observed',
      render: (value: number) => value.toFixed(3)
    },
    {
      title: '期望值',
      dataIndex: 'expected',
      key: 'expected',
      render: (value: number) => value.toFixed(3)
    },
    {
      title: '偏差',
      dataIndex: 'deviation',
      key: 'deviation',
      render: (value: number) => `${(value * 100).toFixed(1)}%`
    },
    {
      title: '检测方法',
      dataIndex: 'method',
      key: 'method',
      render: (method: string) => <Tag color="green">{method}</Tag>
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1>
          <AlertOutlined style={{ marginRight: '8px', color: '#ff4d4f' }} />
          异常检测系统
        </h1>
        <p>使用多种算法检测实验数据中的异常模式和质量问题</p>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总异常数"
              value={statistics.totalAnomalies}
              prefix={<AlertOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="高严重程度"
              value={statistics.highSeverity}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="中等严重程度"
              value={statistics.mediumSeverity}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
              <div>
                <div style={{ marginBottom: '8px' }}>数据质量得分</div>
                <Progress
                  type="circle"
                  percent={statistics.qualityScore ?? 0}
                  width={60}
                  format={(percent) => (statistics.qualityScore === null ? '暂无数据' : `${percent}%`)}
                  strokeColor={{
                    '0%': '#ff4d4f',
                    '50%': '#faad14',
                    '100%': '#52c41a',
                  }}
                />
              </div>
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane 
            tab={
              <span>
                <SearchOutlined />
                异常检测
              </span>
            } 
            key="detect"
          >
            <Form
              form={detectForm}
              layout="vertical"
              onFinish={handleDetectAnomalies}
            >
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="experiment_id"
                    label="实验ID"
                    rules={[{ required: true, message: '请输入实验ID' }]}
                  >
                    <Input placeholder="输入实验ID" />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="metric_name"
                    label="指标名称"
                    rules={[{ required: true, message: '请输入指标名称' }]}
                  >
                    <Input placeholder="例如: conversion_rate" />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="variant"
                    label="变体名称"
                  >
                    <Input placeholder="例如: treatment" />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item
                name="values"
                label="指标值列表"
                rules={[{ required: true, message: '请输入指标值' }]}
              >
                <Input.TextArea
                  placeholder="输入逗号分隔的数值，例如: 0.12, 0.15, 0.11, 0.18, 0.13"
                  rows={3}
                />
              </Form.Item>
              
              <Form.Item
                name="timestamps"
                label="时间戳列表（可选）"
              >
                <Input.TextArea
                  placeholder="输入逗号分隔的时间戳"
                  rows={2}
                />
              </Form.Item>
              
              <Form.Item
                name="methods"
                label="检测方法"
                initialValue={['z_score', 'iqr']}
              >
                <Select
                  mode="multiple"
                  placeholder="选择检测方法"
                  style={{ width: '100%' }}
                >
                  {detectionMethods.map(method => (
                    <Option key={method.value} value={method.value}>
                      {method.name} - {method.description}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<SearchOutlined />}>
                  开始检测
                </Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane 
            tab={
              <span>
                <ExperimentOutlined />
                SRM检查
              </span>
            } 
            key="srm"
          >
            <Form
              form={srmForm}
              layout="vertical"
              onFinish={handleSRMCheck}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="experiment_id"
                    label="实验ID"
                    rules={[{ required: true, message: '请输入实验ID' }]}
                  >
                    <Input placeholder="输入实验ID" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="expected_ratio"
                    label="预期比例"
                    initialValue={0.5}
                  >
                    <InputNumber
                      min={0}
                      max={1}
                      step={0.01}
                      style={{ width: '100%' }}
                      placeholder="0.5"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="control_count"
                    label="对照组样本数"
                    rules={[{ required: true, message: '请输入对照组样本数' }]}
                  >
                    <InputNumber
                      min={0}
                      style={{ width: '100%' }}
                      placeholder="输入对照组样本数"
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="treatment_count"
                    label="实验组样本数"
                    rules={[{ required: true, message: '请输入实验组样本数' }]}
                  >
                    <InputNumber
                      min={0}
                      style={{ width: '100%' }}
                      placeholder="输入实验组样本数"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<ExperimentOutlined />}>
                  检查SRM
                </Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane 
            tab={
              <span>
                <SafetyCertificateOutlined />
                数据质量
              </span>
            } 
            key="quality"
          >
            <Form
              form={qualityForm}
              layout="vertical"
              onFinish={handleDataQualityCheck}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="experiment_id"
                    label="实验ID"
                    rules={[{ required: true, message: '请输入实验ID' }]}
                  >
                    <Input placeholder="输入实验ID" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="total_count"
                    label="总数量"
                    rules={[{ required: true, message: '请输入总数量' }]}
                  >
                    <InputNumber
                      min={0}
                      style={{ width: '100%' }}
                      placeholder="输入总数量"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="missing_rate"
                    label="缺失率 (%)"
                    initialValue={0}
                  >
                    <InputNumber
                      min={0}
                      max={100}
                      style={{ width: '100%' }}
                      placeholder="缺失率百分比"
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="duplicate_rate"
                    label="重复率 (%)"
                    initialValue={0}
                  >
                    <InputNumber
                      min={0}
                      max={100}
                      style={{ width: '100%' }}
                      placeholder="重复率百分比"
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="null_count"
                    label="空值数量"
                    initialValue={0}
                  >
                    <InputNumber
                      min={0}
                      style={{ width: '100%' }}
                      placeholder="空值数量"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<SafetyCertificateOutlined />}>
                  检查数据质量
                </Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane 
            tab={
              <span>
                <SettingOutlined />
                检测配置
              </span>
            } 
            key="config"
          >
            <Form
              form={configForm}
              layout="vertical"
              onFinish={handleConfigureDetection}
              initialValues={{
                methods: ['z_score', 'iqr'],
                sensitivity: 0.95,
                window_size: 100,
                min_samples: 30,
                z_threshold: 3.0,
                iqr_multiplier: 1.5,
                enable_seasonal: true,
                enable_trend: true
              }}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="methods"
                    label="检测方法"
                  >
                    <Select
                      mode="multiple"
                      placeholder="选择检测方法"
                      style={{ width: '100%' }}
                    >
                      {detectionMethods.map(method => (
                        <Option key={method.value} value={method.value}>
                          {method.name}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="sensitivity"
                    label="灵敏度"
                  >
                    <InputNumber
                      min={0.5}
                      max={1.0}
                      step={0.01}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="window_size"
                    label="窗口大小"
                  >
                    <InputNumber
                      min={10}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="min_samples"
                    label="最小样本数"
                  >
                    <InputNumber
                      min={10}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="z_threshold"
                    label="Z-score阈值"
                  >
                    <InputNumber
                      min={2.0}
                      step={0.1}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="iqr_multiplier"
                    label="IQR乘数"
                  >
                    <InputNumber
                      min={1.0}
                      step={0.1}
                      style={{ width: '100%' }}
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="enable_seasonal"
                    label="启用季节性检测"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="enable_trend"
                    label="启用趋势检测"
                    valuePropName="checked"
                  >
                    <Switch />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<SettingOutlined />}>
                  保存配置
                </Button>
              </Form.Item>
            </Form>
          </TabPane>

          <TabPane 
            tab={
              <span>
                <LineChartOutlined />
                异常摘要
              </span>
            } 
            key="summary"
          >
            <Form
              layout="vertical"
              onFinish={handleGetAnomalySummary}
            >
              <Row gutter={16}>
                <Col span={8}>
                  <Form.Item
                    name="experiment_id"
                    label="实验ID"
                    rules={[{ required: true, message: '请输入实验ID' }]}
                  >
                    <Input placeholder="输入实验ID" />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="start_time"
                    label="开始时间"
                  >
                    <DatePicker
                      showTime
                      style={{ width: '100%' }}
                      placeholder="选择开始时间"
                    />
                  </Form.Item>
                </Col>
                <Col span={8}>
                  <Form.Item
                    name="end_time"
                    label="结束时间"
                  >
                    <DatePicker
                      showTime
                      style={{ width: '100%' }}
                      placeholder="选择结束时间"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<LineChartOutlined />}>
                  获取异常摘要
                </Button>
              </Form.Item>
            </Form>
            
            {summaryData && (
              <Card title="异常摘要数据" style={{ marginTop: 16 }}>
                <pre>{JSON.stringify(summaryData, null, 2)}</pre>
              </Card>
            )}
          </TabPane>

          <TabPane 
            tab={
              <span>
                <MonitorOutlined />
                实时监控
              </span>
            } 
            key="realtime"
          >
            <Form
              layout="vertical"
              onFinish={handleSetupRealTimeMonitoring}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    name="experiment_id"
                    label="实验ID"
                    rules={[{ required: true, message: '请输入实验ID' }]}
                  >
                    <Input placeholder="输入实验ID" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="check_interval"
                    label="检查间隔(秒)"
                    initialValue={60}
                  >
                    <InputNumber
                      min={10}
                      style={{ width: '100%' }}
                      placeholder="检查间隔秒数"
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Form.Item
                name="metrics"
                label="监控指标"
                rules={[{ required: true, message: '请输入监控指标' }]}
              >
                <Select
                  mode="multiple"
                  placeholder="选择或输入监控指标"
                  style={{ width: '100%' }}
                >
                  <Option value="conversion_rate">转化率</Option>
                  <Option value="click_through_rate">点击率</Option>
                  <Option value="bounce_rate">跳出率</Option>
                  <Option value="response_time">响应时间</Option>
                  <Option value="error_rate">错误率</Option>
                </Select>
              </Form.Item>
              
              <Form.Item
                name="alert_threshold"
                label="报警阈值"
                initialValue="medium"
              >
                <Select placeholder="选择报警阈值">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Form.Item>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<MonitorOutlined />}>
                  启动实时监控
                </Button>
              </Form.Item>
            </Form>
            
            {monitoringStatus && (
              <Card title="监控状态" style={{ marginTop: 16 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="监控状态"
                      value={monitoringStatus.active ? "运行中" : "已停止"}
                      valueStyle={{ color: monitoringStatus.active ? '#52c41a' : '#ff4d4f' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="监控实验"
                      value={monitoringStatus.experiment_id || "未设置"}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="开始时间"
                      value={monitoringStatus.started_at ? new Date(monitoringStatus.started_at).toLocaleString() : ""}
                    />
                  </Col>
                </Row>
              </Card>
            )}
          </TabPane>

          <TabPane 
            tab={
              <span>
                <BarChartOutlined />
                批量检测
              </span>
            } 
            key="batch"
          >
            <Form
              layout="vertical"
              onFinish={handleBatchDetectAnomalies}
            >
              <Form.Item
                name="experiments"
                label="实验ID列表"
                rules={[{ required: true, message: '请输入实验ID列表' }]}
              >
                <Input.TextArea
                  placeholder="输入逗号分隔的实验ID，例如: exp_001, exp_002, exp_003"
                  rows={3}
                />
              </Form.Item>
              
              <Form.Item
                name="metrics"
                label="检测指标"
                rules={[{ required: true, message: '请输入检测指标' }]}
              >
                <Input.TextArea
                  placeholder="输入逗号分隔的指标名称，例如: conversion_rate, click_through_rate"
                  rows={2}
                />
              </Form.Item>
              
              <Form.Item>
                <Button type="primary" htmlType="submit" loading={loading} icon={<BarChartOutlined />}>
                  开始批量检测
                </Button>
              </Form.Item>
            </Form>
            
            {batchResults.length > 0 && (
              <Card title="批量检测结果" style={{ marginTop: 16 }}>
                {batchResults.map((result, index) => (
                  <Card key={index} type="inner" title={`批次 ${index + 1}`} style={{ marginBottom: 16 }}>
                    <pre>{JSON.stringify(result, null, 2)}</pre>
                  </Card>
                ))}
              </Card>
            )}
          </TabPane>
        </Tabs>
      </Card>

      {anomalies.length > 0 && (
        <Card title="检测结果" style={{ marginTop: '24px' }}>
          <Table
            columns={anomaliesColumns}
            dataSource={anomalies}
            rowKey={(record, index) => `${record.timestamp}_${index}`}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 条记录`
            }}
            scroll={{ x: 1200 }}
          />
        </Card>
      )}
    </div>
  );
};

export default AnomalyDetectionPage;
