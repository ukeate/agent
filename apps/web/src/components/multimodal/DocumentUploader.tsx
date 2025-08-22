/**
 * 文档上传和处理可视化组件
 * 展示文档解析管道的技术细节
 */

import React, { useState } from 'react';
import {
  Card,
  Upload,
  Button,
  Progress,
  List,
  Tag,
  Space,
  Typography,
  Timeline,
  Alert,
  Divider,
  Row,
  Col,
  Statistic
} from 'antd';
import {
  UploadOutlined,
  FileTextOutlined,
  FileImageOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  ClockCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';
import type { UploadFile } from 'antd/es/upload/interface';

const { Text, Title } = Typography;

interface ProcessingStep {
  step: string;
  status: 'waiting' | 'processing' | 'completed' | 'error';
  duration?: number;
  details?: string;
}

interface DocumentMetadata {
  chunks: number;
  images: number;
  tables: number;
  embeddingTime: number;
  vectorDimension: number;
}

interface DocumentUploaderProps {
  onUploadSuccess?: () => void;
}

const DocumentUploader: React.FC<DocumentUploaderProps> = ({ onUploadSuccess }) => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [processing, setProcessing] = useState(false);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [documentMetadata, setDocumentMetadata] = useState<DocumentMetadata | null>(null);

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return <FilePdfOutlined style={{ color: '#ff4d4f' }} />;
    if (['jpg', 'jpeg', 'png', 'gif'].includes(ext || '')) 
      return <FileImageOutlined style={{ color: '#722ed1' }} />;
    if (['xlsx', 'xls', 'csv'].includes(ext || '')) 
      return <FileExcelOutlined style={{ color: '#52c41a' }} />;
    return <FileTextOutlined style={{ color: '#1890ff' }} />;
  };

  const simulateProcessing = async (file: File) => {
    setProcessing(true);
    setCurrentFile(file.name);
    
    // 初始化处理步骤
    const steps: ProcessingStep[] = [
      { step: '文件验证', status: 'waiting' },
      { step: 'Unstructured库解析', status: 'waiting' },
      { step: '内容分类 (文本/图像/表格)', status: 'waiting' },
      { step: '文本分块处理', status: 'waiting' },
      { step: '生成Nomic嵌入向量', status: 'waiting' },
      { step: '存储到Chroma向量库', status: 'waiting' },
      { step: '更新元数据索引', status: 'waiting' }
    ];
    setProcessingSteps(steps);

    // 模拟处理每个步骤
    for (let i = 0; i < steps.length; i++) {
      steps[i].status = 'processing';
      setProcessingSteps([...steps]);
      
      await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));
      
      steps[i].status = 'completed';
      steps[i].duration = Math.round(300 + Math.random() * 700);
      
      // 添加技术细节
      switch (i) {
        case 1:
          steps[i].details = `识别格式: ${file.type}, 大小: ${(file.size / 1024).toFixed(2)}KB`;
          break;
        case 2:
          steps[i].details = '提取: 15个文本块, 3张图像, 2个表格';
          break;
        case 3:
          steps[i].details = 'chunk_size=512, overlap=50';
          break;
        case 4:
          steps[i].details = 'nomic-embed-text-v1.5 (维度: 768)';
          break;
        case 5:
          steps[i].details = 'collection: multimodal_docs';
          break;
      }
      
      setProcessingSteps([...steps]);
    }

    // 设置文档元数据
    setDocumentMetadata({
      chunks: 15,
      images: 3,
      tables: 2,
      embeddingTime: 1250,
      vectorDimension: 768
    });

    setProcessing(false);
    setCurrentFile(null);
    
    if (onUploadSuccess) {
      onUploadSuccess();
    }
  };

  const handleUpload = async () => {
    for (const file of fileList) {
      if (file.originFileObj) {
        await simulateProcessing(file.originFileObj);
      }
    }
    setFileList([]);
  };

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'processing':
        return <LoadingOutlined style={{ color: '#1890ff' }} />;
      case 'error':
        return <ClockCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  return (
    <Card 
      title={
        <span>
          <SyncOutlined className="mr-2" />
          文档处理管道 - Unstructured + Nomic + Chroma
        </span>
      }
    >
      <Row gutter={16}>
        <Col span={12}>
          <Title level={5}>文档上传</Title>
          <Upload
            fileList={fileList}
            onChange={({ fileList }) => setFileList(fileList)}
            beforeUpload={() => false}
            multiple
            disabled={processing}
          >
            <Button icon={<UploadOutlined />} disabled={processing}>
              选择文档
            </Button>
          </Upload>
          
          <div className="mt-3">
            {fileList.map(file => (
              <Tag
                key={file.uid}
                icon={getFileIcon(file.name)}
                closable
                onClose={() => {
                  setFileList(fileList.filter(f => f.uid !== file.uid));
                }}
              >
                {file.name}
              </Tag>
            ))}
          </div>

          <Button
            type="primary"
            onClick={handleUpload}
            disabled={fileList.length === 0 || processing}
            loading={processing}
            className="mt-3"
            block
          >
            开始处理
          </Button>

          <Alert
            message="支持的格式"
            description="PDF, Word, Excel, Text, Markdown, HTML, 图片 (PNG/JPG)"
            variant="default"
            className="mt-3"
          />
        </Col>

        <Col span={12}>
          <Title level={5}>处理流程可视化</Title>
          
          {processing && currentFile && (
            <Alert
              message={`正在处理: ${currentFile}`}
              variant="default"
              showIcon
              className="mb-3"
            />
          )}

          <Timeline>
            {processingSteps.map((step, index) => (
              <Timeline.Item
                key={index}
                dot={getStepIcon(step.status)}
                color={step.status === 'completed' ? 'green' : 
                       step.status === 'processing' ? 'blue' : 'gray'}
              >
                <Space direction="vertical" size="small">
                  <Text strong>{step.step}</Text>
                  {step.duration && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      耗时: {step.duration}ms
                    </Text>
                  )}
                  {step.details && (
                    <Text code style={{ fontSize: 11 }}>
                      {step.details}
                    </Text>
                  )}
                </Space>
              </Timeline.Item>
            ))}
          </Timeline>
        </Col>
      </Row>

      {documentMetadata && (
        <>
          <Divider />
          <Title level={5}>处理结果统计</Title>
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="文本块"
                value={documentMetadata.chunks}
                prefix={<FileTextOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="图像"
                value={documentMetadata.images}
                prefix={<FileImageOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="表格"
                value={documentMetadata.tables}
                prefix={<FileExcelOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="嵌入耗时"
                value={documentMetadata.embeddingTime}
                suffix="ms"
              />
            </Col>
          </Row>
          
          <Alert
            message="技术细节"
            description={
              <Space direction="vertical">
                <Text>• 使用Unstructured库进行文档解析</Text>
                <Text>• Nomic嵌入模型: nomic-embed-text-v1.5 (768维)</Text>
                <Text>• 向量存储: Chroma数据库</Text>
                <Text>• 分块策略: size=512, overlap=50</Text>
              </Space>
            }
            type="success"
            className="mt-3"
          />
        </>
      )}
    </Card>
  );
};

export default DocumentUploader;