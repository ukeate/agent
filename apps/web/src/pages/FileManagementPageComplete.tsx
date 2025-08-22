import React, { useState, useEffect } from 'react';
import {
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
  Divider,
  Empty
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
import type { UploadProps } from 'antd';

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

  // 获取文件列表
  const fetchFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/files/list?limit=100');
      const result = await response.json();
      if (result.success) {
        setFiles(result.data.files);
      } else {
        message.error('获取文件列表失败');
      }
    } catch (error) {
      message.error('获取文件列表失败');
    } finally {
      setLoading(false);
    }
  };

  // 获取文件统计
  const fetchStats = async () => {
    try {
      const response = await fetch('/api/v1/files/stats/summary');
      const result = await response.json();
      if (result.success) {
        setStats(result.data);
      }
    } catch (error) {
      console.error('获取文件统计失败:', error);
    }
  };

  useEffect(() => {
    fetchFiles();
    fetchStats();
  }, []);

  // 文件上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    multiple: true,
    action: '/api/v1/files/upload',
    onChange(info) {
      const { status } = info.file;
      if (status === 'uploading') {
        setUploading(true);
        setUploadProgress(info.file.percent || 0);
      } else if (status === 'done') {
        message.success(`${info.file.name} 上传成功`);
        setUploading(false);
        setUploadProgress(0);
        fetchFiles();
        fetchStats();
      } else if (status === 'error') {
        message.error(`${info.file.name} 上传失败`);
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

  // 删除文件
  const handleDeleteFile = async (fileId: string) => {
    try {
      const response = await fetch(`/api/v1/files/${fileId}`, {
        method: 'DELETE',
      });
      const result = await response.json();
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
  const handleDownloadFile = (fileId: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `/api/v1/files/${fileId}/download`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
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
        </TabPane>
      </Tabs>
    </div>
  );
};

export default FileManagementPageComplete;