import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Table,
  Upload,
  Progress,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Form,
  Input,
  Checkbox,
  Select,
  Tooltip,
  Modal,
  message,
  Tabs,
  List,
  Avatar,
} from 'antd'
import {
  UploadOutlined,
  DownloadOutlined,
  DeleteOutlined,
  FileOutlined,
  FolderOutlined,
  EyeOutlined,
  CloudUploadOutlined,
  CloudDownloadOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  ClearOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import filesService from '../services/filesService'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs
const { Dragger } = Upload

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const getMimeType = (type: string): string => {
  const mimeTypes = {
    PDF: 'application/pdf',
    DOC: 'application/msword',
    TXT: 'text/plain',
    IMAGE: 'image/jpeg',
    VIDEO: 'video/mp4',
    AUDIO: 'audio/mpeg',
    JSON: 'application/json',
    CSV: 'text/csv',
  }
  return mimeTypes[type as keyof typeof mimeTypes] || 'application/octet-stream'
}

const FileManagementAdvancedPage: React.FC = () => {
  const [fileList, setFileList] = useState<any[]>([])
  const [uploading, setUploading] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [filterType, setFilterType] = useState('all')
  const [searchKeyword, setSearchKeyword] = useState('')
  const [previewVisible, setPreviewVisible] = useState(false)
  const [previewFile, setPreviewFile] = useState<any>(null)

  const loadFiles = async () => {
    try {
      const res = await filesService.listFiles({ limit: 100, offset: 0 })
      const files = (res.data?.files || []).map(f => {
        const type = getFileTypeFromName(f.filename)
        return {
          id: f.file_id || f.filename,
          name: f.filename,
          type,
          size: f.file_size,
          sizeFormatted: formatFileSize(f.file_size),
          uploadTime: new Date(f.created_at || Date.now()),
          status: 'completed',
          downloadCount: 0,
          metadata: {
            checksum: f.file_id,
            encoding: 'UTF-8',
            mimeType: getMimeType(type),
          },
        }
      })
      setFileList(files)
    } catch (e: any) {
      message.error(e?.message || '文件列表加载失败')
    }
  }

  useEffect(() => {
    loadFiles()
  }, [])

  // 过滤文件列表
  const filteredFiles = fileList.filter(file => {
    const matchesType = filterType === 'all' || file.type === filterType
    const matchesSearch = file.name
      .toLowerCase()
      .includes(searchKeyword.toLowerCase())
    return matchesType && matchesSearch
  })

  // 统计数据
  const stats = {
    totalFiles: fileList.length,
    totalSize: fileList.reduce((sum, file) => sum + file.size, 0),
    typeDistribution: fileList.reduce(
      (acc, file) => {
        acc[file.type] = (acc[file.type] || 0) + 1
        return acc
      },
      {} as Record<string, number>
    ),
    recentUploads: fileList.filter(
      file => Date.now() - file.uploadTime.getTime() < 24 * 3600 * 1000
    ).length,
  }

  // 文件上传处理
  const handleUpload = async (options: any) => {
    const { file, onSuccess, onError } = options
    setUploading(true)
    try {
      await filesService.uploadFile(file as File)
      await loadFiles()
      onSuccess?.({})
      message.success(`${file.name} 文件上传成功`)
    } catch (e: any) {
      onError?.(e)
      message.error(e?.message || `${file.name} 上传失败`)
    } finally {
      setUploading(false)
    }
  }

  const getFileTypeFromName = (fileName: string): string => {
    const ext = fileName.split('.').pop()?.toUpperCase() || 'UNKNOWN'
    const typeMap: Record<string, string> = {
      PDF: 'PDF',
      DOC: 'DOC',
      DOCX: 'DOC',
      TXT: 'TXT',
      MD: 'TXT',
      JPG: 'IMAGE',
      PNG: 'IMAGE',
      GIF: 'IMAGE',
      MP4: 'VIDEO',
      AVI: 'VIDEO',
      MP3: 'AUDIO',
      WAV: 'AUDIO',
      JSON: 'JSON',
      CSV: 'CSV',
    }
    return typeMap[ext] || 'OTHER'
  }

  // 删除文件
  const handleDelete = (fileId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个文件吗？此操作不可恢复。',
      onOk: async () => {
        await filesService.deleteFile(fileId)
        await loadFiles()
        message.success('文件删除成功')
      },
    })
  }

  // 批量删除
  const handleBatchDelete = () => {
    if (selectedFiles.length === 0) {
      message.warning('请选择要删除的文件')
      return
    }

    Modal.confirm({
      title: '批量删除确认',
      content: `确定要删除选中的 ${selectedFiles.length} 个文件吗？`,
      onOk: () => {
        setFileList(prev =>
          prev.filter(file => !selectedFiles.includes(file.id))
        )
        setSelectedFiles([])
        message.success(`成功删除 ${selectedFiles.length} 个文件`)
      },
    })
  }

  // 预览文件
  const handlePreview = (file: any) => {
    setPreviewFile(file)
    setPreviewVisible(true)
  }

  // 文件表格列定义
  const columns = [
    {
      title: '文件名',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: any) => (
        <Space>
          <FileOutlined style={{ color: getFileColor(record.type) }} />
          <span>{name}</span>
          {record.status === 'processing' && <Tag color="orange">处理中</Tag>}
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color={getFileColor(type)}>{type}</Tag>,
    },
    {
      title: '大小',
      dataIndex: 'sizeFormatted',
      key: 'size',
    },
    {
      title: '上传时间',
      dataIndex: 'uploadTime',
      key: 'uploadTime',
      render: (time: Date) => time.toLocaleString(),
    },
    {
      title: '下载次数',
      dataIndex: 'downloadCount',
      key: 'downloadCount',
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: any) => (
        <Space>
          <Tooltip title="预览">
            <Button
              icon={<EyeOutlined />}
              size="small"
              onClick={() => handlePreview(record)}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button icon={<DownloadOutlined />} size="small" type="primary" />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              icon={<DeleteOutlined />}
              size="small"
              danger
              onClick={() => handleDelete(record.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const getFileColor = (type: string): string => {
    const colors: Record<string, string> = {
      PDF: '#ff4d4f',
      DOC: '#1890ff',
      TXT: '#52c41a',
      IMAGE: '#fa8c16',
      VIDEO: '#722ed1',
      AUDIO: '#eb2f96',
      JSON: '#13c2c2',
      CSV: '#a0d911',
    }
    return colors[type] || '#666666'
  }

  // 统计卡片
  const StatsCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <Statistic
            title="文件总数"
            value={stats.totalFiles}
            prefix={<FileOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="总存储空间"
            value={formatFileSize(stats.totalSize)}
            prefix={<DatabaseOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="今日上传"
            value={stats.recentUploads}
            prefix={<CloudUploadOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="文件类型"
            value={Object.keys(stats.typeDistribution).length}
            prefix={<BarChartOutlined />}
          />
        </Card>
      </Col>
    </Row>
  )

  // 文件类型分布
  const FileTypeDistribution = () => {
    const data = Object.entries(stats.typeDistribution).map(
      ([type, count]) => ({
        type,
        count,
        percentage: ((count / stats.totalFiles) * 100).toFixed(1),
      })
    )

    return (
      <Card title="文件类型分布" size="small">
        <List
          dataSource={data}
          renderItem={item => (
            <List.Item>
              <List.Item.Meta
                avatar={
                  <Avatar style={{ backgroundColor: getFileColor(item.type) }}>
                    {item.type}
                  </Avatar>
                }
                title={item.type}
                description={`${item.count} 个文件 (${item.percentage}%)`}
              />
              <Progress
                percent={parseFloat(item.percentage)}
                size="small"
                showInfo={false}
              />
            </List.Item>
          )}
        />
      </Card>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <CloudUploadOutlined /> 高级文件管理系统
      </Title>
      <Paragraph type="secondary">
        全面的文件上传、下载、管理和分析功能，支持批量操作和多格式文件处理
      </Paragraph>

      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="文件管理" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <StatsCards />

            {/* 操作面板 */}
            <Card title="操作面板" size="small">
              <Row gutter={16} align="middle">
                <Col span={8}>
                  <Dragger
                    name="file"
                    multiple
                    customRequest={handleUpload}
                    showUploadList={false}
                    style={{ height: 100 }}
                  >
                    <p className="ant-upload-drag-icon">
                      <UploadOutlined />
                    </p>
                    <p className="ant-upload-text">
                      点击或拖拽文件到此区域上传
                    </p>
                    <p className="ant-upload-hint">支持单个或批量上传</p>
                  </Dragger>
                </Col>
                <Col span={16}>
                  <Space wrap>
                    <Input.Search
                      name="fileSearch"
                      placeholder="搜索文件名"
                      allowClear
                      value={searchKeyword}
                      onChange={e => setSearchKeyword(e.target.value)}
                      style={{ width: 200 }}
                    />
                    <Select
                      name="fileTypeFilter"
                      value={filterType}
                      onChange={setFilterType}
                      style={{ width: 120 }}
                    >
                      <Option value="all">所有类型</Option>
                      <Option value="PDF">PDF</Option>
                      <Option value="DOC">DOC</Option>
                      <Option value="TXT">TXT</Option>
                      <Option value="IMAGE">图片</Option>
                      <Option value="VIDEO">视频</Option>
                      <Option value="AUDIO">音频</Option>
                    </Select>
                    <Button
                      danger
                      onClick={handleBatchDelete}
                      disabled={selectedFiles.length === 0}
                    >
                      批量删除 ({selectedFiles.length})
                    </Button>
                    <Button
                      icon={<ClearOutlined />}
                      onClick={() => message.info('请使用后端清理接口')}
                    >
                      清理临时文件
                    </Button>
                    <Button icon={<ReloadOutlined />} onClick={loadFiles}>
                      刷新列表
                    </Button>
                  </Space>
                </Col>
              </Row>
            </Card>

            {/* 文件列表 */}
            <Card title={`文件列表 (${filteredFiles.length})`} size="small">
              <Table
                columns={columns}
                dataSource={filteredFiles}
                rowKey="id"
                rowSelection={{
                  selectedRowKeys: selectedFiles,
                  onChange: (keys: React.Key[]) =>
                    setSelectedFiles(keys as string[]),
                  getCheckboxProps: (record: any) => ({
                    name: `fileSelect-${record.id}`,
                  }),
                  columnTitle: (
                    <Checkbox
                      name="selectAllFiles"
                      indeterminate={
                        selectedFiles.length > 0 &&
                        selectedFiles.length < filteredFiles.length
                      }
                      checked={
                        filteredFiles.length > 0 &&
                        selectedFiles.length === filteredFiles.length
                      }
                      onChange={e =>
                        setSelectedFiles(
                          e.target.checked
                            ? filteredFiles.map(item => item.id)
                            : []
                        )
                      }
                    />
                  ),
                }}
                pagination={{
                  pageSize: 15,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total, range) =>
                    `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
                }}
                scroll={{ x: 1000 }}
              />
            </Card>
          </Space>
        </TabPane>

        <TabPane tab="统计分析" key="2">
          <Row gutter={16}>
            <Col span={12}>
              <FileTypeDistribution />
            </Col>
            <Col span={12}>
              <Card title="存储使用情况" size="small">
                <div style={{ marginBottom: 16 }}>
                  <Text>已使用存储空间</Text>
                  <Progress
                    percent={75}
                    format={percent =>
                      `${formatFileSize(stats.totalSize)} / 10GB`
                    }
                  />
                </div>

                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="平均文件大小"
                      value={formatFileSize(stats.totalSize / stats.totalFiles)}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic title="最大文件" value="45.2MB" />
                  </Col>
                </Row>

                <Divider />

                <div>
                  <Text strong>最近活动</Text>
                  <List
                    size="small"
                    dataSource={fileList.slice(0, 5)}
                    renderItem={file => (
                      <List.Item>
                        <Text>{file.name}</Text>
                        <Text type="secondary">
                          {file.uploadTime.toLocaleString()}
                        </Text>
                      </List.Item>
                    )}
                  />
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="批量操作" key="3">
          <Card title="批量文件操作" size="small">
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Alert
                message="批量操作功能"
                description="可以对多个文件执行批量上传、下载、压缩、格式转换等操作"
                type="info"
                showIcon
              />

              <Row gutter={16}>
                <Col span={8}>
                  <Card title="批量上传" size="small">
                    <Dragger
                      name="files"
                      multiple
                      directory
                      customRequest={handleUpload}
                    >
                      <p className="ant-upload-drag-icon">
                        <FolderOutlined />
                      </p>
                      <p className="ant-upload-text">上传整个文件夹</p>
                    </Dragger>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card title="批量下载" size="small">
                    <Button
                      type="primary"
                      icon={<CloudDownloadOutlined />}
                      disabled={selectedFiles.length === 0}
                      style={{ width: '100%', height: 60 }}
                    >
                      下载选中文件 ({selectedFiles.length})
                    </Button>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card title="批量清理" size="small">
                    <Button
                      danger
                      icon={<ClearOutlined />}
                      onClick={async () => {
                        await filesService.cleanupOldFiles(7)
                        await loadFiles()
                        message.success('已清理7天前文件')
                      }}
                      style={{ width: '100%', height: 60 }}
                    >
                      清理7天前文件
                    </Button>
                  </Card>
                </Col>
              </Row>
            </Space>
          </Card>
        </TabPane>
      </Tabs>

      {/* 文件预览模态框 */}
      <Modal
        title="文件预览"
        visible={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        footer={[
          <Button key="download" type="primary" icon={<DownloadOutlined />}>
            下载
          </Button>,
          <Button key="close" onClick={() => setPreviewVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        {previewFile && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>文件名: </Text>
                <Text>{previewFile.name}</Text>
              </Col>
              <Col span={12}>
                <Text strong>文件大小: </Text>
                <Text>{previewFile.sizeFormatted}</Text>
              </Col>
            </Row>
            <Row gutter={16} style={{ marginTop: 8 }}>
              <Col span={12}>
                <Text strong>上传时间: </Text>
                <Text>{previewFile.uploadTime?.toLocaleString()}</Text>
              </Col>
              <Col span={12}>
                <Text strong>校验和: </Text>
                <Text>{previewFile.metadata?.checksum}</Text>
              </Col>
            </Row>
            <Divider />
            <div
              style={{
                height: 300,
                border: '1px solid #d9d9d9',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#fafafa',
              }}
            >
              <Text type="secondary">文件预览区域 ({previewFile.type})</Text>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default FileManagementAdvancedPage
