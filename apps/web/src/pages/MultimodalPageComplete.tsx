import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Upload, 
  Button, 
  Select, 
  Radio, 
  Space, 
  message, 
  Alert,
  Progress,
  Divider,
  Typography,
  Row,
  Col,
  Tag,
  Statistic,
  Table,
  Spin,
  Badge,
  Descriptions,
  Empty,
  Input
} from 'antd';
import { 
  UploadOutlined, 
  FileImageOutlined, 
  FileTextOutlined, 
  VideoCameraOutlined,
  AudioOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  ApiOutlined,
  DollarOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface QueueItem {
  id: string;
  fileName: string;
  fileType: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  model: string;
  cost?: number;
  processingTime?: number;
}

interface ProcessingResult {
  id: string;
  fileName: string;
  status: string;
  extractedData: {
    description?: string;
    objects?: string[];
    text?: string;
    keyPoints?: string[];
  };
  confidence: number;
  cost: number;
  processingTime: number;
  model: string;
}

const MultimodalPageComplete: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [processingMode, setProcessingMode] = useState('standard');
  const [uploadFiles, setUploadFiles] = useState<any[]>([]);
  const [processingQueue, setProcessingQueue] = useState<QueueItem[]>([]);
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [queueStats, setQueueStats] = useState({
    total: 0,
    completed: 0,
    failed: 0,
    processing: 0
  });

  // 模拟API响应
  useEffect(() => {
    // 模拟队列状态更新
    const mockQueueData: QueueItem[] = [
      {
        id: '1',
        fileName: 'sample-image.jpg',
        fileType: 'image/jpeg',
        status: 'completed',
        progress: 100,
        model: 'gpt-4o-mini',
        cost: 0.003,
        processingTime: 2.4
      },
      {
        id: '2',
        fileName: 'document.pdf',
        fileType: 'application/pdf',
        status: 'processing',
        progress: 65,
        model: 'gpt-4o',
        cost: 0.012
      }
    ];
    
    setProcessingQueue(mockQueueData);
    setQueueStats({
      total: 2,
      completed: 1,
      failed: 0,
      processing: 1
    });

    // 模拟处理结果
    const mockResults: ProcessingResult[] = [
      {
        id: '1',
        fileName: 'sample-image.jpg',
        status: 'completed',
        extractedData: {
          description: '这是一张包含建筑物和天空的城市风景照片',
          objects: ['建筑物', '天空', '云朵', '窗户'],
          keyPoints: ['现代建筑风格', '清晰的天空背景', '良好的光照条件']
        },
        confidence: 0.92,
        cost: 0.003,
        processingTime: 2.4,
        model: 'gpt-4o-mini'
      }
    ];
    
    setResults(mockResults);
  }, []);

  const handleUpload = async (file: any) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', selectedModel);
    formData.append('mode', processingMode);

    try {
      setLoading(true);
      // 这里会调用实际的API
      // const response = await fetch('/api/v1/multimodal/upload', {
      //   method: 'POST',
      //   body: formData
      // });
      
      message.success('文件上传成功，已加入处理队列');
      
      // 模拟添加到队列
      const newItem: QueueItem = {
        id: Date.now().toString(),
        fileName: file.name,
        fileType: file.type,
        status: 'pending',
        progress: 0,
        model: selectedModel
      };
      
      setProcessingQueue(prev => [...prev, newItem]);
    } catch (error) {
      message.error('文件上传失败');
    } finally {
      setLoading(false);
    }
    
    return false; // 阻止默认上传行为
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'processing':
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const queueColumns = [
    {
      title: '文件名',
      dataIndex: 'fileName',
      key: 'fileName',
      render: (text: string, record: QueueItem) => (
        <Space>
          {record.fileType.includes('image') && <FileImageOutlined />}
          {record.fileType.includes('pdf') && <FileTextOutlined />}
          {record.fileType.includes('video') && <VideoCameraOutlined />}
          {text}
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <span>{status === 'completed' ? '已完成' : status === 'processing' ? '处理中' : status === 'failed' ? '失败' : '等待中'}</span>
        </Space>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      )
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      render: (model: string) => <Tag color="blue">{model}</Tag>
    },
    {
      title: '成本',
      dataIndex: 'cost',
      key: 'cost',
      render: (cost: number) => cost ? `$${cost.toFixed(4)}` : '-'
    }
  ];

  return (
    <div className="p-6">
      <Title level={2}>GPT-4o 多模态API集成</Title>
      <Paragraph>
        支持图像、文档、视频等多种内容的智能分析处理，集成OpenAI最新的多模态模型。
      </Paragraph>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* 文件上传标签页 */}
        <TabPane tab="文件上传" key="upload">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Card title="上传设置" className="mb-4">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>选择模型:</Text>
                    <Select
                      value={selectedModel}
                      onChange={setSelectedModel}
                      style={{ width: '100%', marginTop: 8 }}
                    >
                      <Option value="gpt-4o">GPT-4o (高质量，$5/$15 per 1K tokens)</Option>
                      <Option value="gpt-4o-mini">GPT-4o-mini (性价比，$0.15/$0.6 per 1K tokens)</Option>
                      <Option value="gpt-5">GPT-5 (最新，$12.5/$25 per 1K tokens)</Option>
                      <Option value="gpt-5-nano">GPT-5-nano (经济，$0.05/$0.4 per 1K tokens)</Option>
                    </Select>
                  </div>
                  
                  <div>
                    <Text strong>处理模式:</Text>
                    <Radio.Group
                      value={processingMode}
                      onChange={(e) => setProcessingMode(e.target.value)}
                      style={{ marginTop: 8 }}
                    >
                      <Radio value="quick">快速分析（不保存）</Radio>
                      <Radio value="standard">标准处理</Radio>
                      <Radio value="detailed">详细分析</Radio>
                    </Radio.Group>
                  </div>
                </Space>
              </Card>

              <Card title="文件上传">
                <Upload.Dragger
                  name="file"
                  multiple
                  beforeUpload={handleUpload}
                  showUploadList={false}
                  disabled={loading}
                >
                  <p className="ant-upload-drag-icon">
                    <UploadOutlined />
                  </p>
                  <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                  <p className="ant-upload-hint">
                    支持单个或批量上传。支持图像、PDF文档、视频文件。
                  </p>
                </Upload.Dragger>

                <div className="mt-4">
                  <Text type="secondary" className="text-xs">
                    支持的格式：JPG, PNG, GIF, PDF, MP4, MOV, AVI 等
                    <br />
                    最大文件大小：20MB
                  </Text>
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="快速分析" className="mb-4">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <TextArea
                    placeholder="粘贴图像URL或输入分析提示..."
                    rows={4}
                  />
                  <Button type="primary" block>
                    快速分析
                  </Button>
                </Space>
              </Card>

              <Card title="队列统计">
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic title="总计" value={queueStats.total} />
                  </Col>
                  <Col span={6}>
                    <Statistic 
                      title="已完成" 
                      value={queueStats.completed} 
                      valueStyle={{ color: '#3f8600' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic 
                      title="处理中" 
                      value={queueStats.processing}
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic 
                      title="失败" 
                      value={queueStats.failed}
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* 处理队列标签页 */}
        <TabPane tab="处理队列" key="queue">
          <Card title="处理队列" extra={
            <Button type="primary">
              刷新状态
            </Button>
          }>
            <Table
              columns={queueColumns}
              dataSource={processingQueue}
              rowKey="id"
              pagination={false}
            />
          </Card>
        </TabPane>

        {/* 处理结果标签页 */}
        <TabPane tab="处理结果" key="results">
          {results.length > 0 ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {results.map((result) => (
                <Card 
                  key={result.id}
                  title={
                    <Space>
                      <FileImageOutlined />
                      {result.fileName}
                      <Badge status="success" text="已完成" />
                    </Space>
                  }
                  extra={<Tag color="blue">{result.model}</Tag>}
                >
                  <Row gutter={[16, 16]}>
                    <Col span={16}>
                      <Descriptions column={1} size="small">
                        <Descriptions.Item label="描述">
                          {result.extractedData.description}
                        </Descriptions.Item>
                        <Descriptions.Item label="识别对象">
                          <Space wrap>
                            {result.extractedData.objects?.map((obj, idx) => (
                              <Tag key={idx} color="geekblue">{obj}</Tag>
                            ))}
                          </Space>
                        </Descriptions.Item>
                        <Descriptions.Item label="关键点">
                          <ul className="mb-0">
                            {result.extractedData.keyPoints?.map((point, idx) => (
                              <li key={idx}>{point}</li>
                            ))}
                          </ul>
                        </Descriptions.Item>
                      </Descriptions>
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="置信度"
                        value={result.confidence * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: result.confidence > 0.8 ? '#3f8600' : '#cf1322' }}
                      />
                      <div className="mt-4">
                        <Text type="secondary">
                          处理时间: {result.processingTime}秒
                          <br />
                          成本: ${result.cost.toFixed(4)}
                        </Text>
                      </div>
                      <div className="mt-4">
                        <Space>
                          <Button icon={<EyeOutlined />} size="small">
                            查看详情
                          </Button>
                          <Button icon={<DownloadOutlined />} size="small">
                            导出结果
                          </Button>
                        </Space>
                      </div>
                    </Col>
                  </Row>
                </Card>
              ))}
            </Space>
          ) : (
            <Empty description="暂无处理结果" />
          )}
        </TabPane>

        {/* 成本监控标签页 */}
        <TabPane tab="成本监控" key="cost">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Card title="成本统计">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="今日总成本"
                      value={0.025}
                      precision={4}
                      prefix="$"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="总Token使用"
                      value={15420}
                      suffix="tokens"
                    />
                  </Col>
                </Row>
                
                <Divider />
                
                <div>
                  <Text strong>成本预警</Text>
                  <Progress
                    percent={25}
                    status="normal"
                    format={() => `$0.025 / $0.10`}
                  />
                  <Text type="secondary" className="text-xs">
                    每日预算限制
                  </Text>
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="模型使用分布">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div className="flex justify-between">
                    <Text>GPT-4o-mini</Text>
                    <Text>65%</Text>
                  </div>
                  <Progress percent={65} size="small" />
                  
                  <div className="flex justify-between">
                    <Text>GPT-4o</Text>
                    <Text>25%</Text>
                  </div>
                  <Progress percent={25} size="small" />
                  
                  <div className="flex justify-between">
                    <Text>GPT-5-nano</Text>
                    <Text>10%</Text>
                  </div>
                  <Progress percent={10} size="small" />
                </Space>
              </Card>
            </Col>
          </Row>

          <Card title="成本计算说明" className="mt-6">
            <Row gutter={16}>
              <Col span={12}>
                <ul className="text-sm">
                  <li>GPT-4o: $5/1K输入, $15/1K输出</li>
                  <li>GPT-4o-mini: $0.15/1K输入, $0.6/1K输出</li>
                </ul>
              </Col>
              <Col span={12}>
                <ul className="text-sm">
                  <li>GPT-5: $12.5/1K输入, $25/1K输出</li>
                  <li>GPT-5-nano: $0.05/1K输入, $0.4/1K输出</li>
                </ul>
              </Col>
            </Row>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default MultimodalPageComplete;