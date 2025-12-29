import React, { useState, useEffect } from 'react';
import {
import { logger } from '../../utils/logger'
  Card,
  Form,
  Select,
  Slider,
  InputNumber,
  Button,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Table,
  Tag,
  Typography,
  Space,
  Divider,
  Switch,
  message
} from 'antd';
import {
  ThunderboltOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  CompressOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import { pgvectorApi } from '../../services/pgvectorApi';

const { Title, Text } = Typography;
const { Option } = Select;

interface QuantizationConfig {
  mode: 'float32' | 'int8' | 'int4' | 'adaptive';
  precision_threshold: number;
  compression_ratio: number;
  enable_dynamic: boolean;
}

interface QuantizationResult {
  original_size: number;
  quantized_size: number;
  compression_ratio: number;
  precision_loss: number;
  processing_time_ms: number;
  mode_used: string;
}

const VectorQuantizationPanel: React.FC = () => {
  const [config, setConfig] = useState<QuantizationConfig>({
    mode: 'adaptive',
    precision_threshold: 0.95,
    compression_ratio: 4.0,
    enable_dynamic: true
  });
  
  const [testResults, setTestResults] = useState<QuantizationResult[]>([]);
  const [testing, setTesting] = useState(false);
  const [applying, setApplying] = useState(false);

  const quantizationModes = [
    { value: 'float32', label: 'Float32 (原始精度)', description: '32位浮点数，最高精度' },
    { value: 'int8', label: 'INT8 (8位量化)', description: '8位整数，4倍压缩' },
    { value: 'int4', label: 'INT4 (4位量化)', description: '4位整数，8倍压缩' },
    { value: 'adaptive', label: 'Adaptive (自适应)', description: '根据精度阈值自动选择' }
  ];

  useEffect(() => {
    fetchCurrentConfig();
  }, []);

  const fetchCurrentConfig = async () => {
    try {
      const currentConfig = await pgvectorApi.getQuantizationConfig();
      setConfig(currentConfig);
    } catch (error) {
      logger.error('获取量化配置失败:', error);
    }
  };

  const handleTestQuantization = async () => {
    try {
      setTesting(true);
      const results = await pgvectorApi.testQuantization(config);
      setTestResults(results);
      if (results.length === 0) {
        message.warning('暂无可量化向量数据');
      } else {
        message.success('量化测试完成！');
      }
    } catch (error) {
      logger.error('量化测试失败:', error);
      message.error('量化测试失败');
    } finally {
      setTesting(false);
    }
  };

  const handleApplyConfig = async () => {
    try {
      setApplying(true);
      await pgvectorApi.applyQuantizationConfig(config);
      message.success('量化配置已应用！');
    } catch (error) {
      logger.error('应用配置失败:', error);
      message.error('配置应用失败');
    } finally {
      setApplying(false);
    }
  };

  const getCompressionColor = (ratio: number) => {
    if (ratio >= 6) return '#f50';
    if (ratio >= 3) return '#fa8c16';
    return '#52c41a';
  };

  const getPrecisionColor = (loss: number) => {
    if (loss <= 0.05) return '#52c41a';
    if (loss <= 0.15) return '#fa8c16';
    return '#f50';
  };

  const resultColumns = [
    {
      title: '量化模式',
      dataIndex: 'mode_used',
      key: 'mode',
      render: (mode: string) => (
        <Tag color={mode === 'int4' ? 'red' : mode === 'int8' ? 'orange' : 'green'}>
          {mode.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '原始大小 (KB)',
      dataIndex: 'original_size',
      key: 'original_size',
      render: (size: number) => (size / 1024).toFixed(1)
    },
    {
      title: '量化后大小 (KB)',
      dataIndex: 'quantized_size',
      key: 'quantized_size',
      render: (size: number) => (size / 1024).toFixed(1)
    },
    {
      title: '压缩比',
      dataIndex: 'compression_ratio',
      key: 'compression_ratio',
      render: (ratio: number) => (
        <span style={{ color: getCompressionColor(ratio) }}>
          {ratio.toFixed(1)}x
        </span>
      )
    },
    {
      title: '精度损失',
      dataIndex: 'precision_loss',
      key: 'precision_loss',
      render: (loss: number) => (
        <span style={{ color: getPrecisionColor(loss) }}>
          {(loss * 100).toFixed(2)}%
        </span>
      )
    },
    {
      title: '处理时间 (ms)',
      dataIndex: 'processing_time_ms',
      key: 'processing_time',
      render: (time: number) => time.toFixed(1)
    }
  ];

  return (
    <div>
      <Row gutter={24}>
        {/* 配置面板 */}
        <Col span={12}>
          <Card title="量化配置" extra={<ThunderboltOutlined />}>
            <Form layout="vertical">
              <Form.Item label="量化模式">
                <Select
                  value={config.mode}
                  onChange={(value) => setConfig({ ...config, mode: value })}
                  style={{ width: '100%' }}
                >
                  {quantizationModes.map(mode => (
                    <Option key={mode.value} value={mode.value}>
                      <div>
                        <div>{mode.label}</div>
                        <small style={{ color: '#666' }}>{mode.description}</small>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item label={`精度阈值: ${(config.precision_threshold * 100).toFixed(0)}%`}>
                <Slider
                  min={0.8}
                  max={0.99}
                  step={0.01}
                  value={config.precision_threshold}
                  onChange={(value) => setConfig({ ...config, precision_threshold: value })}
                  marks={{
                    0.8: '80%',
                    0.9: '90%',
                    0.95: '95%',
                    0.99: '99%'
                  }}
                />
                <Text type="secondary">
                  自适应模式下的最小精度要求
                </Text>
              </Form.Item>

              <Form.Item label="启用动态量化">
                <Switch
                  checked={config.enable_dynamic}
                  onChange={(checked) => setConfig({ ...config, enable_dynamic: checked })}
                />
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary">
                    根据向量特征动态选择量化策略
                  </Text>
                </div>
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<ExperimentOutlined />}
                    onClick={handleTestQuantization}
                    loading={testing}
                  >
                    测试量化效果
                  </Button>
                  <Button 
                    type="default"
                    onClick={handleApplyConfig}
                    loading={applying}
                    disabled={testResults.length === 0}
                  >
                    应用配置
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Col>

        {/* 实时统计 */}
        <Col span={12}>
          <Card title="量化效果预览" extra={<BarChartOutlined />}>
            {testResults.length > 0 ? (
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="平均压缩比"
                    value={testResults.reduce((sum, r) => sum + r.compression_ratio, 0) / testResults.length}
                    precision={1}
                    suffix="x"
                    valueStyle={{ color: '#fa8c16' }}
                    prefix={<CompressOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="平均精度保持"
                    value={(1 - testResults.reduce((sum, r) => sum + r.precision_loss, 0) / testResults.length) * 100}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: '#52c41a' }}
                    prefix={<LineChartOutlined />}
                  />
                </Col>
              </Row>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Text type="secondary">点击"测试量化效果"查看结果</Text>
              </div>
            )}

            {/* 内存节省可视化 */}
            {testResults.length > 0 && (
              <div style={{ marginTop: 24 }}>
                <Title level={5}>内存节省效果</Title>
                {testResults.map((result, index) => {
                  const savingPercent = ((result.original_size - result.quantized_size) / result.original_size) * 100;
                  return (
                    <div key={index} style={{ marginBottom: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <Text>{result.mode_used.toUpperCase()}</Text>
                        <Text strong>{savingPercent.toFixed(1)}% 节省</Text>
                      </div>
                      <Progress 
                        percent={savingPercent} 
                        strokeColor={getCompressionColor(result.compression_ratio)}
                        size="small"
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 测试结果详情 */}
      {testResults.length > 0 && (
        <Card 
          title="量化测试结果" 
          style={{ marginTop: 24 }}
          extra={<ExperimentOutlined />}
        >
          <Table
            dataSource={testResults.map((result, index) => ({ ...result, key: index }))}
            columns={resultColumns}
            size="small"
            pagination={false}
          />
          
          <Alert
            style={{ marginTop: 16 }}
            message="量化建议"
            description={
              <ul>
                <li>INT8量化适合大多数场景，提供良好的压缩比和精度平衡</li>
                <li>INT4量化适合对存储空间要求极高的场景，但可能有精度损失</li>
                <li>自适应模式会根据精度阈值自动选择最优策略</li>
                <li>建议在生产环境使用前充分测试量化效果</li>
              </ul>
            }
            variant="default"
            showIcon
          />
        </Card>
      )}
    </div>
  );
};

export default VectorQuantizationPanel;
