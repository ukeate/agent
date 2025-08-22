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
  Empty
} from 'antd';
import { 
  UploadOutlined, 
  FileImageOutlined, 
  FileTextOutlined, 
  VideoCameraOutlined,
  AudioOutlined,
  CloudUploadOutlined,
  RocketOutlined,
  DollarOutlined,
  ThunderboltOutlined,
  BalanceOutlined,
  ApiOutlined
} from '@ant-design/icons';
import { multimodalService } from '../services/multimodalService';
import ModelSelector from '../components/multimodal/ModelSelector';
import ProcessingResult from '../components/multimodal/ProcessingResult';
import ProcessingQueue from '../components/multimodal/ProcessingQueue';
import BatchProcessor from '../components/multimodal/BatchProcessor';
import CostMonitor from '../components/multimodal/CostMonitor';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Dragger } = Upload;

const MultimodalPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<any[]>([]);
  const [processingResults, setProcessingResults] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState('gpt-4o');
  const [priority, setPriority] = useState('balanced');
  const [complexity, setComplexity] = useState('medium');
  const [extractOptions, setExtractOptions] = useState({
    extractText: true,
    extractObjects: true,
    extractSentiment: false
  });
  const [queueStatus, setQueueStatus] = useState<any>(null);
  const [totalCost, setTotalCost] = useState(0);
  const [totalTokens, setTotalTokens] = useState(0);

  // 获取队列状态
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const status = await multimodalService.getQueueStatus();
        setQueueStatus(status);
      } catch (error) {
        console.error('获取队列状态失败:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // 文件上传配置
  const uploadProps = {
    multiple: true,
    beforeUpload: (file: any) => {
      // 验证文件类型
      const supportedTypes = [
        'image/jpeg', 'image/png', 'image/webp', 'image/gif',
        'application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'video/mp4', 'video/avi', 'video/quicktime'
      ];
      
      if (!supportedTypes.includes(file.type)) {
        message.error(`${file.name} 不是支持的文件类型`);
        return false;
      }

      // 验证文件大小（100MB）
      const isLt100M = file.size / 1024 / 1024 < 100;
      if (!isLt100M) {
        message.error('文件大小必须小于100MB');
        return false;
      }

      setSelectedFiles(prev => [...prev, file]);
      return false; // 阻止自动上传
    },
    onRemove: (file: any) => {
      setSelectedFiles(prev => prev.filter(f => f.uid !== file.uid));
    },
    fileList: selectedFiles
  };

  // 处理单个文件
  const handleProcessSingle = async () => {
    if (selectedFiles.length === 0) {
      message.warning('请先选择文件');
      return;
    }

    setLoading(true);
    try {
      for (const file of selectedFiles) {
        // 上传文件
        const uploadResult = await multimodalService.uploadFile(file);
        
        // 处理文件
        const processResult = await multimodalService.processContent({
          contentId: uploadResult.content_id,
          contentType: uploadResult.content_type,
          priority,
          complexity,
          maxTokens: 1000,
          ...extractOptions
        });

        // 添加到结果列表
        setProcessingResults(prev => [...prev, {
          ...processResult,
          fileName: file.name,
          fileType: file.type,
          fileSize: file.size
        }]);

        // 更新成本和Token统计
        if (processResult.tokens_used) {
          setTotalTokens(prev => prev + processResult.tokens_used.total_tokens);
        }
        if (processResult.cost) {
          setTotalCost(prev => prev + processResult.cost);
        }
      }
      
      message.success('处理完成');
      setSelectedFiles([]);
    } catch (error) {
      message.error('处理失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 批量处理
  const handleBatchProcess = async () => {
    if (selectedFiles.length < 2) {
      message.warning('批量处理至少需要2个文件');
      return;
    }

    setLoading(true);
    try {
      // 先上传所有文件
      const uploadPromises = selectedFiles.map(file => 
        multimodalService.uploadFile(file)
      );
      const uploadResults = await Promise.all(uploadPromises);
      
      // 批量处理
      const contentIds = uploadResults.map(r => r.content_id);
      const batchResult = await multimodalService.processBatch({
        contentIds,
        priority,
        complexity,
        maxTokens: 1000
      });

      message.success(`批量处理已提交，批次ID: ${batchResult.batch_id}`);
      setSelectedFiles([]);
      
      // 轮询获取结果
      pollBatchResults(contentIds);
    } catch (error) {
      message.error('批量处理失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // 轮询批处理结果
  const pollBatchResults = async (contentIds: string[]) => {
    const checkResults = async () => {
      for (const contentId of contentIds) {
        try {
          const status = await multimodalService.getProcessingStatus(contentId);
          if (status.status === 'completed') {
            setProcessingResults(prev => {
              const exists = prev.find(r => r.content_id === contentId);
              if (!exists) {
                return [...prev, status];
              }
              return prev;
            });
          }
        } catch (error) {
          console.error(`获取状态失败: ${contentId}`, error);
        }
      }
    };

    // 每2秒检查一次，持续1分钟
    const interval = setInterval(checkResults, 2000);
    setTimeout(() => clearInterval(interval), 60000);
  };

  // 直接分析图像（不保存）
  const handleQuickAnalyze = async (file: any) => {
    setLoading(true);
    try {
      const result = await multimodalService.analyzeImageDirect(
        file,
        '详细分析这张图像的内容',
        {
          extractText: true,
          extractObjects: true,
          priority
        }
      );

      setProcessingResults(prev => [...prev, {
        ...result,
        fileName: file.name,
        fileType: 'image',
        isQuickAnalysis: true
      }]);

      message.success('快速分析完成');
    } catch (error) {
      message.error('分析失败');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <Title level={2}>
        <ApiOutlined /> GPT-4o 多模态处理中心
      </Title>
      <Paragraph type="secondary">
        学习和测试OpenAI GPT-4o多模态API的各种功能，支持图像、文档、视频等多种内容类型的智能处理
      </Paragraph>

      <Row gutter={[16, 16]}>
        {/* 左侧：上传和配置 */}
        <Col xs={24} lg={10}>
          <Card title="内容上传与配置" className="mb-4">
            <Tabs defaultActiveKey="upload">
              <TabPane tab="文件上传" key="upload">
                <Dragger {...uploadProps}>
                  <p className="ant-upload-drag-icon">
                    <CloudUploadOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                  </p>
                  <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                  <p className="ant-upload-hint">
                    支持图像(JPG/PNG/WebP)、文档(PDF/Word/TXT)、视频(MP4/AVI)
                  </p>
                </Dragger>

                {selectedFiles.length > 0 && (
                  <div className="mt-4">
                    <Alert
                      message={`已选择 ${selectedFiles.length} 个文件`}
                      variant="default"
                      showIcon
                    />
                  </div>
                )}
              </TabPane>

              <TabPane tab="快速分析" key="quick">
                <Upload
                  accept="image/*"
                  showUploadList={false}
                  beforeUpload={(file) => {
                    handleQuickAnalyze(file);
                    return false;
                  }}
                >
                  <Button icon={<ThunderboltOutlined />} type="primary">
                    选择图像快速分析（不保存文件）
                  </Button>
                </Upload>
                <Alert
                  className="mt-2"
                  message="快速分析模式直接处理图像，不会保存到服务器"
                  variant="default"
                />
              </TabPane>
            </Tabs>
          </Card>

          <Card title="模型与参数配置" className="mb-4">
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={setSelectedModel}
              priority={priority}
              onPriorityChange={setPriority}
              complexity={complexity}
              onComplexityChange={setComplexity}
            />
            
            <Divider />
            
            <div>
              <Text strong>提取选项：</Text>
              <div className="mt-2">
                <Space direction="vertical">
                  <label>
                    <input
                      type="checkbox"
                      checked={extractOptions.extractText}
                      onChange={(e) => setExtractOptions(prev => ({
                        ...prev,
                        extractText: e.target.checked
                      }))}
                    />
                    <span className="ml-2">提取文本内容</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={extractOptions.extractObjects}
                      onChange={(e) => setExtractOptions(prev => ({
                        ...prev,
                        extractObjects: e.target.checked
                      }))}
                    />
                    <span className="ml-2">识别对象和场景</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={extractOptions.extractSentiment}
                      onChange={(e) => setExtractOptions(prev => ({
                        ...prev,
                        extractSentiment: e.target.checked
                      }))}
                    />
                    <span className="ml-2">分析情感倾向</span>
                  </label>
                </Space>
              </div>
            </div>

            <Divider />

            <Space>
              <Button
                type="primary"
                icon={<RocketOutlined />}
                onClick={handleProcessSingle}
                loading={loading}
                disabled={selectedFiles.length === 0}
              >
                处理文件
              </Button>
              
              <Button
                icon={<CloudUploadOutlined />}
                onClick={handleBatchProcess}
                loading={loading}
                disabled={selectedFiles.length < 2}
              >
                批量处理
              </Button>
            </Space>
          </Card>

          <CostMonitor
            totalCost={totalCost}
            totalTokens={totalTokens}
            modelUsage={processingResults.reduce((acc, r) => {
              const model = r.model_used || 'unknown';
              acc[model] = (acc[model] || 0) + 1;
              return acc;
            }, {} as Record<string, number>)}
          />
        </Col>

        {/* 右侧：结果展示 */}
        <Col xs={24} lg={14}>
          <ProcessingQueue status={queueStatus} />
          
          <Card 
            title="处理结果" 
            className="mt-4"
            extra={
              processingResults.length > 0 && (
                <Button 
                  size="small" 
                  onClick={() => setProcessingResults([])}
                >
                  清空结果
                </Button>
              )
            }
          >
            {processingResults.length === 0 ? (
              <Empty description="暂无处理结果" />
            ) : (
              <div className="space-y-4">
                {processingResults.map((result, index) => (
                  <ProcessingResult
                    key={result.content_id || index}
                    result={result}
                    showDetails
                  />
                ))}
              </div>
            )}
          </Card>
        </Col>
      </Row>

      {/* 批量处理监控 */}
      <BatchProcessor 
        visible={false}
        onClose={() => {}}
      />
    </div>
  );
};

export default MultimodalPage;