import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Typography,
  Tabs,
  Upload,
  Button,
  Table,
  Space,
  message,
  Progress,
  Tag,
  Statistic,
  Row,
  Col,
  Modal,
  Input,
  Select,
  Popconfirm,
  Alert,
  Empty,
  InputNumber,
  Descriptions,
  Spin
} from 'antd';
import {
  UploadOutlined,
  FileOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  SearchOutlined,
  FilterOutlined,
  CloudUploadOutlined,
  FileImageOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  VideoCameraOutlined,
  AudioOutlined,
  InboxOutlined
} from '@ant-design/icons';
import type { UploadProps, UploadFile } from 'antd';
import { filesService } from '../services/filesService';
import type { BatchUploadResponse } from '../services/filesService';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Dragger } = Upload;
const { Search } = Input;
const { Option } = Select;

interface FileInfo {
  file_id: string;
  filename: string;
  file_size: number;
  content_type?: string;
  created_at: number;
  modified_at: number;
  file_path: string;
}

interface FileStats {
  total_files: number;
  total_size: number;
  total_size_mb: number;
  file_types: Record<string, number>;
  upload_path: string;
}

const FileManagementPageComplete: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stats, setStats] = useState<FileStats | null>(null);
  const [searchText, setSearchText] = useState('');
  const [selectedFileType, setSelectedFileType] = useState<string>('all');
  const [infoModalVisible, setInfoModalVisible] = useState(false);
  const [infoLoading, setInfoLoading] = useState(false);
  const [selectedFileInfo, setSelectedFileInfo] = useState<Record<string, any> | null>(null);
  const [batchFileList, setBatchFileList] = useState<UploadFile[]>([]);
  const [batchUploading, setBatchUploading] = useState(false);
  const [batchUploadResult, setBatchUploadResult] = useState<BatchUploadResponse | null>(null);
  const [cleanupDays, setCleanupDays] = useState<number>(7);
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const [cleanupMessage, setCleanupMessage] = useState('');

  // 获取文件列表
  const fetchFiles = async () => {
    setLoading(true);
    try {
      const result = await filesService.listFiles({ limit: 100 });
      if (result.success) {
        setFiles(result.data.files || []);
      } else {
        setFiles([]);
        message.error('获取文件列表失败');
      }
    } catch (error) {
      logger.error('获取文件列表失败:', error);
      message.error('获取文件列表失败');
      // 使用空数组作为后备
      setFiles([]);
    } finally {
      setLoading(false);
    }
  };

  // 获取文件统计
  const fetchStats = async () => {
    try {
      const result = await filesService.getFileStats();
      if (result.success) {
        setStats(result.data);
      }
    } catch (error) {
      logger.error('获取文件统计失败:', error);
    }
  };

  useEffect(() => {
    fetchFiles();
    fetchStats();
  }, []);

  const handleViewFileInfo = async (fileId: string) => {
    setInfoLoading(true);
    setSelectedFileInfo(null);
    setInfoModalVisible(true);
    try {
      const result = await filesService.getFileInfo(fileId);
      if (result.success) {
        setSelectedFileInfo(result.data);
      } else {
        message.error('获取文件信息失败');
      }
    } catch (error) {
      message.error('获取文件信息失败');
      logger.error('获取文件信息失败:', error);
    } finally {
      setInfoLoading(false);
    }
  };

  const handleCloseInfoModal = () => {
    setInfoModalVisible(false);
    setSelectedFileInfo(null);
  };

  // 文件上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true,
    customRequest: async ({ file, onProgress, onSuccess, onError }) => {
      try {
        setUploading(true);
        setUploadProgress(0);
        const response = await filesService.uploadFile(file as File);
        if (response.success) {
          onSuccess?.(response.data);
          setUploadProgress(100);
          onProgress?.({ percent: 100 } as any);
          message.success(`${file.name} 上传成功`);
          fetchFiles();
          fetchStats();
        } else {
          onError?.(new Error(response.message));
          message.error(`${file.name} 上传失败: ${response.message}`);
        }
      } catch (error) {
        onError?.(error as Error);
        message.error(`${file.name} 上传失败`);
      } finally {
        setUploading(false);
        setUploadProgress(0);
      }
    },
    beforeUpload: (file) => {
      const isLt100M = file.size / 1024 / 1024 < 100;
      if (!isLt100M) {
        message.error('文件大小不能超过 100MB!');
      }
      return isLt100M;
    },
  };

  const batchUploadProps: UploadProps<UploadFile> = {
    multiple: true,
    fileList: batchFileList,
    beforeUpload: (file) => {
      setBatchFileList(prev => {
        if (prev.some(item => item.uid === file.uid)) {
          return prev;
        }
        const uploadFile: UploadFile = {
          uid: file.uid,
          name: file.name,
          size: file.size,
          type: file.type,
          originFileObj: file,
          status: 'done'
        };
        return [...prev, uploadFile];
      });
      return false;
    },
    onRemove: (file) => {
      setBatchFileList(prev => prev.filter(item => item.uid !== file.uid));
    }
  };

  const handleBatchUpload = async () => {
    if (batchFileList.length === 0) {
      message.warning('请先选择需要上传的文件');
      return;
    }

    const filesToUpload = batchFileList
      .map(item => item.originFileObj)
      .filter((item): item is File => !!item);

    if (filesToUpload.length === 0) {
      message.error('文件选择无效，请重新选择');
      return;
    }

    setBatchUploading(true);
    setBatchUploadResult(null);
    try {
      const result = await filesService.uploadMultipleFiles(filesToUpload);
      setBatchUploadResult(result);
      if (result.success) {
        message.success(`批量上传完成，成功 ${result.uploaded_count} 个文件`);
      } else {
        message.warning('批量上传部分文件失败，请查看详情');
      }
      setBatchFileList([]);
      fetchFiles();
      fetchStats();
    } catch (error) {
      message.error('批量上传失败');
      logger.error('批量上传失败:', error);
    } finally {
      setBatchUploading(false);
    }
  };

  const handleCleanup = async () => {
    setCleanupLoading(true);
    setCleanupMessage('');
    try {
      const result = await filesService.cleanupOldFiles(cleanupDays);
      setCleanupMessage(result.message);
      if (result.success) {
        message.success(result.message);
      } else {
        message.warning('文件清理完成，请检查结果');
      }
      fetchFiles();
      fetchStats();
    } catch (error) {
      message.error('文件清理失败');
      logger.error('文件清理失败:', error);
    } finally {
      setCleanupLoading(false);
    }
  };

  // 删除文件
  const handleDeleteFile = async (fileId: string) => {
    try {
      const result = await filesService.deleteFile(fileId);
      if (result.success) {
        message.success('文件删除成功');
        fetchFiles();
        fetchStats();
      } else {
        message.error('文件删除失败');
      }
    } catch (error) {
      message.error('文件删除失败');
    }
  };

  // 下载文件
  const handleDownloadFile = async (fileId: string, filename: string) => {
    try {
      const blob = await filesService.downloadFile(fileId);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      message.success('文件下载成功');
    } catch (error) {
      message.error('文件下载失败');
      logger.error('下载文件失败:', error);
    }
  };

  // 获取文件图标
  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf':
        return <FilePdfOutlined style={{ color: '#ff4d4f' }} />;
      case 'doc':
      case 'docx':
        return <FileWordOutlined style={{ color: '#1890ff' }} />;
      case 'xls':
      case 'xlsx':
        return <FileExcelOutlined style={{ color: '#52c41a' }} />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'bmp':
      case 'webp':
        return <FileImageOutlined style={{ color: '#722ed1' }} />;
      case 'mp4':
      case 'avi':
      case 'mov':
      case 'mkv':
        return <VideoCameraOutlined style={{ color: '#eb2f96' }} />;
      case 'mp3':
      case 'wav':
      case 'aac':
        return <AudioOutlined style={{ color: '#fa8c16' }} />;
      default:
        return <FileOutlined />;
    }
  };

  // 格式化文件大小
  const formatFileSize = (size: number) => {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(2)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(2)} MB`;
    return `${(size / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  // 格式化时间
  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  // 过滤文件
  const filteredFiles = files.filter(file => {
    const matchesSearch = file.filename.toLowerCase().includes(searchText.toLowerCase());
    const matchesType = selectedFileType === 'all' || 
      file.filename.toLowerCase().endsWith(selectedFileType);
    return matchesSearch && matchesType;
  });

  // 文件表格列配置
  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
      render: (filename: string) => (
        <Space>
          {getFileIcon(filename)}
          <Text>{filename}</Text>
        </Space>
      ),
    },
    {
      title: '大小',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => formatFileSize(size),
      sorter: (a: FileInfo, b: FileInfo) => a.file_size - b.file_size,
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (timestamp: number) => formatTime(timestamp),
      sorter: (a: FileInfo, b: FileInfo) => a.created_at - b.created_at,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: FileInfo) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => handleViewFileInfo(record.file_id)}
          >
            详情
          </Button>
          <Button
            type="text"
            icon={<DownloadOutlined />}
            onClick={() => handleDownloadFile(record.file_id, record.filename)}
          >
            下载
          </Button>
          <Popconfirm
            title="确定要删除这个文件吗？"
            onConfirm={() => handleDeleteFile(record.file_id)}
            okText="确定"
            cancelText="取消"
          >
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <CloudUploadOutlined style={{ marginRight: '12px' }} />
          文件管理系统
        </Title>
        <Text type="secondary">
          企业级文件管理系统，支持多格式文件上传、内容解析和智能分类
        </Text>
      </div>

      {/* 统计信息 */}
      {stats && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总文件数"
                value={stats.total_files}
                prefix={<FileOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总大小"
                value={stats.total_size_mb}
                suffix="MB"
                precision={2}
                prefix={<CloudUploadOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="文件类型"
                value={Object.keys(stats.file_types).length}
                prefix={<FilterOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均大小"
                value={stats.total_files > 0 ? (stats.total_size_mb / stats.total_files) : 0}
                suffix="MB"
                precision={2}
                prefix={<FileOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Tabs activeKey={activeTab} onChange={setActiveTab} size="large">
        <TabPane
          tab={<span><UploadOutlined />文件上传</span>}
          key="upload"
        >
          <Card>
            <Alert
              message="支持的文件格式"
              description="图片 (JPG, PNG, GIF, BMP, WEBP)、文档 (PDF, DOC, DOCX, TXT)、表格 (XLS, XLSX, CSV)、演示 (PPT, PPTX)、音频 (MP3, WAV, AAC)、视频 (MP4, AVI, MOV)"
              variant="default"
              showIcon
              style={{ marginBottom: 16 }}
            />
            
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域进行上传</p>
              <p className="ant-upload-hint">
                支持单个或批量上传，文件大小限制 100MB
              </p>
            </Dragger>

            {uploading && (
              <div style={{ marginTop: 16 }}>
                <Text>上传进度:</Text>
                <Progress percent={uploadProgress} />
              </div>
            )}
          </Card>
        </TabPane>

        <TabPane
          tab={<span><CloudUploadOutlined />批量上传</span>}
          key="batch"
        >
          <Card>
            <Alert
              message="批量上传"
              description="选择多个文件后点击开始上传，可一次性上传最大10个文件。"
              variant="default"
              showIcon
              style={{ marginBottom: 16 }}
            />

            <Dragger {...batchUploadProps} multiple>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">拖拽或点击选择多个文件</p>
              <p className="ant-upload-hint">支持与单文件上传相同的文件类型</p>
            </Dragger>

            <Space style={{ marginTop: 16 }}>
              <Button
                type="primary"
                icon={<CloudUploadOutlined />}
                loading={batchUploading}
                onClick={handleBatchUpload}
              >
                开始上传
              </Button>
              <Button
                onClick={() => {
                  setBatchFileList([]);
                  setBatchUploadResult(null);
                }}
              >
                清空列表
              </Button>
            </Space>

            {batchUploadResult && (
              <Card style={{ marginTop: 24 }} title="上传结果">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>成功上传: {batchUploadResult.uploaded_count} 个</Text>
                  <Text>失败: {batchUploadResult.error_count} 个</Text>
                  {batchUploadResult.errors.length > 0 && (
                    <Alert
                      type="warning"
                      message="失败详情"
                      description={
                        <Space direction="vertical">
                          {batchUploadResult.errors.map((item) => (
                            <Text key={item.filename}>{`${item.filename}: ${item.error}`}</Text>
                          ))}
                        </Space>
                      }
                      showIcon
                    />
                  )}
                </Space>
              </Card>
            )}
          </Card>
        </TabPane>

        <TabPane
          tab={<span><FileOutlined />文件管理</span>}
          key="management"
        >
          <Card>
            <div style={{ marginBottom: 16 }}>
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Search
                    placeholder="搜索文件名"
                    allowClear
                    value={searchText}
                    onChange={(e) => setSearchText(e.target.value)}
                    style={{ width: '100%' }}
                  />
                </Col>
                <Col span={8}>
                  <Select
                    placeholder="文件类型"
                    style={{ width: '100%' }}
                    value={selectedFileType}
                    onChange={setSelectedFileType}
                  >
                    <Option value="all">全部类型</Option>
                    <Option value="pdf">PDF</Option>
                    <Option value="jpg">图片</Option>
                    <Option value="mp4">视频</Option>
                    <Option value="mp3">音频</Option>
                    <Option value="docx">文档</Option>
                    <Option value="xlsx">表格</Option>
                  </Select>
                </Col>
                <Col span={4}>
                  <Button
                    type="primary"
                    icon={<SearchOutlined />}
                    onClick={fetchFiles}
                  >
                    刷新
                  </Button>
                </Col>
              </Row>
            </div>

            {filteredFiles.length > 0 ? (
              <Table
                columns={columns}
                dataSource={filteredFiles}
                rowKey="file_id"
                loading={loading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 个文件`,
                }}
              />
            ) : (
              <Empty description="暂无文件" />
            )}
          </Card>
        </TabPane>

        <TabPane
          tab={<span><EyeOutlined />文件统计</span>}
          key="statistics"
        >
          <Card title="存储统计" style={{ marginBottom: 16 }}>
            {stats ? (
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <div>
                    <Title level={4}>文件类型分布</Title>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      {Object.entries(stats.file_types).map(([ext, count]) => (
                        <div key={ext} style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>{ext || '无扩展名'}</span>
                          <Tag color="blue">{count} 个</Tag>
                        </div>
                      ))}
                    </Space>
                  </div>
                </Col>
                <Col span={12}>
                  <div>
                    <Title level={4}>存储信息</Title>
                    <Paragraph>
                      <Text strong>存储路径:</Text> {stats.upload_path}
                    </Paragraph>
                    <Paragraph>
                      <Text strong>总文件数:</Text> {stats.total_files} 个
                    </Paragraph>
                    <Paragraph>
                      <Text strong>总存储大小:</Text> {stats.total_size_mb} MB
                    </Paragraph>
                    <Paragraph>
                      <Text strong>平均文件大小:</Text>{' '}
                      {stats.total_files > 0 ? (stats.total_size_mb / stats.total_files).toFixed(2) : 0} MB
                    </Paragraph>
                  </div>
                </Col>
              </Row>
            ) : (
              <Empty description="暂无统计数据" />
            )}
          </Card>

          <Card title="文件清理">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Text>清理早于</Text>
                <InputNumber
                  min={1}
                  max={365}
                  value={cleanupDays}
                  onChange={(value) => {
                    if (typeof value === 'number') {
                      setCleanupDays(value);
                    }
                  }}
                />
                <Text>天的文件</Text>
                <Button
                  type="primary"
                  loading={cleanupLoading}
                  onClick={handleCleanup}
                >
                  执行清理
                </Button>
              </Space>
              {cleanupMessage && (
                <Alert
                  type="info"
                  message="清理结果"
                  description={cleanupMessage}
                  showIcon
                />
              )}
            </Space>
          </Card>
        </TabPane>
      </Tabs>

      <Modal
        title="文件详情"
        open={infoModalVisible}
        onCancel={handleCloseInfoModal}
        footer={null}
        destroyOnClose
      >
        {infoLoading ? (
          <div style={{ textAlign: 'center', padding: '24px 0' }}>
            <Spin />
          </div>
        ) : selectedFileInfo ? (
          <Descriptions column={1} bordered>
            <Descriptions.Item label="文件ID">{selectedFileInfo.file_id || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="文件大小">{formatFileSize(selectedFileInfo.file_size || 0)}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{selectedFileInfo.created_at || 'N/A'}</Descriptions.Item>
            <Descriptions.Item label="存储路径">{selectedFileInfo.file_path || 'N/A'}</Descriptions.Item>
          </Descriptions>
        ) : (
          <Alert type="error" message="未找到文件信息" showIcon />
        )}
      </Modal>
    </div>
  );
};

export default FileManagementPageComplete;
