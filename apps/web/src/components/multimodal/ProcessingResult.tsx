import React, { useEffect, useState } from 'react';
import { 
  Card, 
  Descriptions, 
  Tag, 
  Progress, 
  Tabs, 
  Typography,
  Space,
  Button,
  Collapse,
  List,
  Badge,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Empty
} from 'antd';
import { multimodalService } from '../../services/multimodalService';
import {
  FileImageOutlined,
  FileTextOutlined,
  VideoCameraOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  CodeOutlined,
  DatabaseOutlined,
  EyeOutlined,
  DownloadOutlined
} from '@ant-design/icons';
import ReactJson from 'react-json-view';

const { TabPane } = Tabs;
const { Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface ProcessingResultProps {
  result: any;
  showDetails?: boolean;
}

const ProcessingResult: React.FC<ProcessingResultProps> = ({ result, showDetails = true }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // 获取状态图标和颜色
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'completed':
        return { icon: <CheckCircleOutlined />, color: 'success', text: '已完成' };
      case 'failed':
        return { icon: <CloseCircleOutlined />, color: 'error', text: '失败' };
      case 'processing':
        return { icon: <ClockCircleOutlined />, color: 'processing', text: '处理中' };
      case 'cached':
        return { icon: <DatabaseOutlined />, color: 'default', text: '缓存' };
      default:
        return { icon: <ClockCircleOutlined />, color: 'default', text: status };
    }
  };

  // 获取文件类型图标
  const getFileIcon = (type: string) => {
    if (type?.includes('image')) return <FileImageOutlined style={{ fontSize: 24, color: '#1890ff' }} />;
    if (type?.includes('video')) return <VideoCameraOutlined style={{ fontSize: 24, color: '#722ed1' }} />;
    return <FileTextOutlined style={{ fontSize: 24, color: '#52c41a' }} />;
  };

  const statusConfig = getStatusConfig(result.status);
  const [pricingReady, setPricingReady] = useState(false);

  useEffect(() => {
    multimodalService.ensureModelPricing()
      .then(() => setPricingReady(true))
      .catch(() => setPricingReady(false));
  }, []);

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (!bytes) return 'N/A';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  // 计算成本
  const calculateCost = () => {
    if (!pricingReady || !result.tokens_used || !result.model_used) return 0;
    const totalTokens = result.tokens_used.total_tokens || 0;
    const inputTokens = typeof result.tokens_used.prompt_tokens === 'number'
      ? result.tokens_used.prompt_tokens
      : Math.floor(totalTokens / 3);
    const outputTokens = typeof result.tokens_used.completion_tokens === 'number'
      ? result.tokens_used.completion_tokens
      : Math.max(0, totalTokens - inputTokens);
    return multimodalService.calculateCost(result.model_used, inputTokens, outputTokens);
  };

  return (
    <Card 
      className="mb-4"
      title={
        <div className="flex items-center justify-between">
          <Space>
            {getFileIcon(result.fileType)}
            <Text strong>{result.fileName || `内容 ${result.content_id?.slice(0, 8)}...`}</Text>
            <Badge status={statusConfig.color as any} text={statusConfig.text} />
          </Space>
          {result.model_used && (
            <Tag color="blue">{result.model_used}</Tag>
          )}
        </div>
      }
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* 概览标签 */}
        <TabPane tab="概览" key="overview">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="置信度"
                value={result.confidence_score ? (result.confidence_score * 100).toFixed(1) : 0}
                suffix="%"
                valueStyle={{ color: result.confidence_score > 0.8 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="处理时间"
                value={result.processing_time?.toFixed(2) || 0}
                suffix="秒"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="Token使用"
                value={result.tokens_used?.total_tokens || 0}
                suffix="tokens"
              />
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Statistic
                title="估算成本"
                value={result.cost || calculateCost()}
                prefix="$"
                precision={4}
              />
            </Col>
          </Row>

          <div className="mt-4">
            <Descriptions column={{ xs: 1, sm: 2 }} bordered size="small">
              <Descriptions.Item label="文件类型">
                {result.fileType || result.content_type}
              </Descriptions.Item>
              <Descriptions.Item label="文件大小">
                {formatFileSize(result.fileSize)}
              </Descriptions.Item>
              {result.isQuickAnalysis && (
                <Descriptions.Item label="分析模式" span={2}>
                  <Tag color="orange">快速分析（未保存）</Tag>
                </Descriptions.Item>
              )}
            </Descriptions>
          </div>

          {result.error_message && (
            <Alert
              className="mt-4"
              message="处理错误"
              description={result.error_message}
              type="error"
              showIcon
            />
          )}
        </TabPane>

        {/* 提取的数据 */}
        <TabPane tab="提取的数据" key="extracted">
          {result.extracted_data ? (
            <div>
              {/* 主要内容 */}
              {result.extracted_data.description && (
                <Card size="small" className="mb-3">
                  <Text strong>描述：</Text>
                  <Paragraph className="mt-2">
                    {result.extracted_data.description}
                  </Paragraph>
                </Card>
              )}

              {result.extracted_data.summary && (
                <Card size="small" className="mb-3">
                  <Text strong>摘要：</Text>
                  <Paragraph className="mt-2">
                    {result.extracted_data.summary}
                  </Paragraph>
                </Card>
              )}

              {/* 识别的对象 */}
              {result.extracted_data.objects && result.extracted_data.objects.length > 0 && (
                <Card size="small" className="mb-3" title="识别的对象">
                  <Space wrap>
                    {result.extracted_data.objects.map((obj: string, idx: number) => (
                      <Tag key={idx} color="blue">{obj}</Tag>
                    ))}
                  </Space>
                </Card>
              )}

              {/* 提取的文本 */}
              {result.extracted_data.text_content && (
                <Card size="small" className="mb-3" title="提取的文本">
                  <Paragraph copyable>
                    {result.extracted_data.text_content}
                  </Paragraph>
                </Card>
              )}

              {/* 关键点 */}
              {result.extracted_data.key_points && result.extracted_data.key_points.length > 0 && (
                <Card size="small" className="mb-3" title="关键点">
                  <List
                    size="small"
                    dataSource={result.extracted_data.key_points}
                    renderItem={(item: string) => (
                      <List.Item>• {item}</List.Item>
                    )}
                  />
                </Card>
              )}

              {/* 情感分析 */}
              {result.extracted_data.sentiment && (
                <Card size="small" className="mb-3">
                  <Text strong>情感分析：</Text>
                  <Tag 
                    className="ml-2"
                    color={
                      result.extracted_data.sentiment === 'positive' ? 'success' :
                      result.extracted_data.sentiment === 'negative' ? 'error' : 'default'
                    }
                  >
                    {result.extracted_data.sentiment}
                  </Tag>
                </Card>
              )}
            </div>
          ) : (
            <Empty description="暂无提取的数据" />
          )}
        </TabPane>

        {/* 结构化数据 */}
        {result.structured_data && (
          <TabPane tab="结构化数据" key="structured">
            <ReactJson
              src={result.structured_data}
              theme="monokai"
              collapsed={1}
              displayDataTypes={false}
              displayObjectSize={true}
              enableClipboard={true}
            />
          </TabPane>
        )}

        {/* 原始JSON */}
        {showDetails && (
          <TabPane 
            tab={
              <span>
                <CodeOutlined /> 原始数据
              </span>
            } 
            key="raw"
          >
            <ReactJson
              src={result}
              theme="monokai"
              collapsed={2}
              displayDataTypes={false}
              displayObjectSize={true}
              enableClipboard={true}
            />
          </TabPane>
        )}

        {/* Token详情 */}
        {result.tokens_used && (
          <TabPane tab="Token详情" key="tokens">
            <Descriptions bordered column={1} size="small">
              <Descriptions.Item label="输入Tokens">
                {result.tokens_used.prompt_tokens || 0}
              </Descriptions.Item>
              <Descriptions.Item label="输出Tokens">
                {result.tokens_used.completion_tokens || 0}
              </Descriptions.Item>
              <Descriptions.Item label="总计Tokens">
                <Text strong>{result.tokens_used.total_tokens || 0}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="模型">
                {result.model_used || 'N/A'}
              </Descriptions.Item>
              <Descriptions.Item label="成本估算">
                <Text type="danger">${result.cost || calculateCost()}</Text>
              </Descriptions.Item>
            </Descriptions>

            <div className="mt-4">
              <Progress
                percent={Math.min((result.tokens_used.total_tokens / 4096) * 100, 100)}
                status="active"
                format={percent => `${result.tokens_used.total_tokens} / 4096`}
              />
              <Text type="secondary" className="text-xs">
                Token使用率（基于4096限制）
              </Text>
            </div>
          </TabPane>
        )}
      </Tabs>

      {/* 操作按钮 */}
      <div className="mt-4 flex justify-end space-x-2">
        <Button icon={<EyeOutlined />} size="small">
          查看详情
        </Button>
        <Button icon={<DownloadOutlined />} size="small">
          导出结果
        </Button>
      </div>
    </Card>
  );
};

export default ProcessingResult;
