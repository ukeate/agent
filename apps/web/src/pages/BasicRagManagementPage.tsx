/**
 * 基础RAG管理页面
 * 专门用于测试和使用未被利用的RAG API
 */

import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Row,
  Col,
  Button,
  Space,
  message,
  Spin,
  Typography,
  Input,
  Switch,
  Tag,
  Modal,
  Statistic,
  Progress,
  Divider,
  Alert,
  Badge,
  Tabs,
  List
} from 'antd';
import {
  SearchOutlined,
  ReloadOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  ApiOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';
import { ragService, QueryRequest, QueryResponse, AddDocumentRequest } from '../services/ragService';

const { Title, Text, Paragraph } = Typography;
const { Search, TextArea } = Input;
const { TabPane } = Tabs;

interface ApiCall {
  endpoint: string;
  method: string;
  status: 'success' | 'error' | 'pending';
  response?: any;
  error?: string;
  timestamp: string;
}

const BasicRagManagementPage: React.FC = () => {
  // 状态管理
  const [searchResults, setSearchResults] = useState<QueryResponse | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [stats, setStats] = useState({
    total_documents: 0,
    total_vectors: 0,
    index_size: 0,
    last_updated: ''
  });
  const [apiCalls, setApiCalls] = useState<ApiCall[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [docText, setDocText] = useState('');
  const [docMetadata, setDocMetadata] = useState('');
  const [indexFilePath, setIndexFilePath] = useState('');
  const [indexDirectoryPath, setIndexDirectoryPath] = useState('');
  const [indexExtensions, setIndexExtensions] = useState('');
  const [indexRecursive, setIndexRecursive] = useState(true);
  const [indexForce, setIndexForce] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  // 记录API调用
  const recordApiCall = (endpoint: string, method: string, status: 'success' | 'error' | 'pending', response?: any, error?: string) => {
    const call: ApiCall = {
      endpoint,
      method,
      status,
      response,
      error,
      timestamp: new Date().toLocaleString()
    };
    setApiCalls(prev => [call, ...prev.slice(0, 9)]); // 保持最新10条记录
  };

  // 加载统计信息
  const loadStats = async () => {
    recordApiCall('/api/v1/rag/index/stats', 'GET', 'pending');
    
    try {
      const statsData = await ragService.getIndexStats();
      setStats(statsData.stats);
      recordApiCall('/api/v1/rag/index/stats', 'GET', 'success', statsData);
    } catch (error) {
      logger.error('加载统计失败:', error);
      recordApiCall('/api/v1/rag/index/stats', 'GET', 'error', null, (error as Error).message);
    }
  };

  // 执行搜索
  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      message.warning('请输入搜索关键词');
      return;
    }

    setSearchLoading(true);
    setSearchQuery(query);
    recordApiCall('/api/v1/rag/query', 'POST', 'pending');
    
    try {
      const request: QueryRequest = {
        query: query.trim(),
        search_type: 'hybrid',
        limit: 10,
        score_threshold: 0.5
      };
      
      const response = await ragService.query(request);
      setSearchResults(response);
      recordApiCall('/api/v1/rag/query', 'POST', 'success', response);
      
      if (response.success) {
        message.success(`找到 ${response.results.length} 个相关结果`);
      }
    } catch (error) {
      logger.error('搜索失败:', error);
      recordApiCall('/api/v1/rag/query', 'POST', 'error', null, (error as Error).message);
      message.error('搜索失败');
    } finally {
      setSearchLoading(false);
    }
  };

  // 添加文档
  const handleAddDocument = async () => {
    if (!docText.trim()) {
      message.warning('请输入文档内容');
      return;
    }

    recordApiCall('/api/v1/rag/documents', 'POST', 'pending');
    setActionLoading(true);
    
    try {
      const metadataValue = docMetadata.trim() ? JSON.parse(docMetadata) : undefined;
      const request: AddDocumentRequest = {
        text: docText.trim(),
        metadata: metadataValue
      };
      const response = await ragService.addDocument(request);
      recordApiCall('/api/v1/rag/documents', 'POST', 'success', response);
      message.success(response.message || '文档添加成功');
      setDocText('');
      setDocMetadata('');
      loadStats();
    } catch (error) {
      const err = error as Error;
      recordApiCall('/api/v1/rag/documents', 'POST', 'error', null, err.message);
      message.error('添加文档失败: ' + err.message);
    } finally {
      setActionLoading(false);
    }
  };

  // 索引文件
  const handleIndexFile = async () => {
    if (!indexFilePath.trim()) {
      message.warning('请输入文件路径');
      return;
    }

    recordApiCall('/api/v1/rag/index/file', 'POST', 'pending');
    setActionLoading(true);
    
    try {
      const response = await ragService.indexFile(indexFilePath.trim(), indexForce);
      recordApiCall('/api/v1/rag/index/file', 'POST', 'success', response);
      message.success('文件索引完成');
      loadStats();
    } catch (error) {
      const err = error as Error;
      recordApiCall('/api/v1/rag/index/file', 'POST', 'error', null, err.message);
      message.error('文件索引失败: ' + err.message);
    } finally {
      setActionLoading(false);
    }
  };

  // 索引目录
  const handleIndexDirectory = async () => {
    if (!indexDirectoryPath.trim()) {
      message.warning('请输入目录路径');
      return;
    }

    recordApiCall('/api/v1/rag/index/directory', 'POST', 'pending');
    setActionLoading(true);
    
    try {
      const extensions = indexExtensions
        .split(',')
        .map(ext => ext.trim())
        .filter(Boolean);
      const response = await ragService.indexDirectory(
        indexDirectoryPath.trim(),
        indexRecursive,
        indexForce,
        extensions.length > 0 ? extensions : undefined
      );
      recordApiCall('/api/v1/rag/index/directory', 'POST', 'success', response);
      message.success('目录索引完成');
      loadStats();
    } catch (error) {
      const err = error as Error;
      recordApiCall('/api/v1/rag/index/directory', 'POST', 'error', null, err.message);
      message.error('目录索引失败: ' + err.message);
    } finally {
      setActionLoading(false);
    }
  };

  // 重置索引
  const handleResetIndex = async () => {
    Modal.confirm({
      title: '重置索引',
      icon: <ExclamationCircleOutlined />,
      content: '重置索引将清空当前集合数据，确定继续吗？',
      onOk: async () => {
        recordApiCall('/api/v1/rag/index/reset', 'DELETE', 'pending');
        setActionLoading(true);
        try {
          const response = await ragService.resetIndex();
          recordApiCall('/api/v1/rag/index/reset', 'DELETE', 'success', response);
          message.success(response.message || '索引已重置');
          loadStats();
        } catch (error) {
          const err = error as Error;
          recordApiCall('/api/v1/rag/index/reset', 'DELETE', 'error', null, err.message);
          message.error('重置索引失败: ' + err.message);
        } finally {
          setActionLoading(false);
        }
      }
    });
  };

  // 健康检查
  const handleHealthCheck = async () => {
    recordApiCall('/api/v1/rag/health', 'GET', 'pending');
    setActionLoading(true);
    try {
      const response = await ragService.healthCheck();
      recordApiCall('/api/v1/rag/health', 'GET', 'success', response);
      const status = response?.status || 'unknown';
      if (status === 'healthy') {
        message.success('RAG服务健康');
      } else {
        message.warning(`RAG服务状态: ${status}`);
      }
    } catch (error) {
      const err = error as Error;
      recordApiCall('/api/v1/rag/health', 'GET', 'error', null, err.message);
      message.error('健康检查失败: ' + err.message);
    } finally {
      setActionLoading(false);
    }
  };

  // 初始化加载
  useEffect(() => {
    loadStats();
  }, []);

  // API调用状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'pending':
        return <Spin size="small" />;
      default:
        return null;
    }
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <Title level={2}>基础RAG管理</Title>
        <Paragraph>
          这个页面用于测试基础RAG能力，包括文档搜索、文本入库、文件/目录索引、索引统计和健康检查。
        </Paragraph>
        
        <Alert
          message="API状态说明"
          description="所有操作直接调用后端RAG服务，调用结果会记录在下方的调用历史中。"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      </div>

      {/* 统计面板 */}
      <Row gutter={16} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="总文档数"
              value={stats.total_documents}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="向量数量"
              value={stats.total_vectors}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="索引大小"
              value={Number((stats.index_size / 1024 / 1024).toFixed(2))}
              suffix="MB"
              precision={2}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="最后更新"
              value={stats.last_updated ? new Date(stats.last_updated).toLocaleString() : '未知'}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="search">
        {/* 文档搜索 */}
        <TabPane tab="文档搜索" key="search">
          <Card title="RAG文档搜索" className="mb-4">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Search
                placeholder="输入搜索关键词..."
                allowClear
                enterButton={<SearchOutlined />}
                size="large"
                loading={searchLoading}
                name="rag-search"
                onSearch={handleSearch}
              />
              
              {searchResults && (
                <div>
                  <Divider orientation="left">
                    搜索结果 ({searchResults.results.length}个)
                    {searchQuery && <Text type="secondary"> - "{searchQuery}"</Text>}
                  </Divider>
                  
                  <List
                    dataSource={searchResults.results}
                    renderItem={(item) => (
                      <List.Item key={item.id}>
                        <List.Item.Meta
                          title={
                            <Space>
                              <Text strong>{item.file_path || item.id}</Text>
                              <Tag color="blue">得分: {item.score.toFixed(3)}</Tag>
                            </Space>
                          }
                          description={
                            <div>
                              <Paragraph ellipsis={{ rows: 2 }}>
                                {item.content}
                              </Paragraph>
                              {item.file_path && (
                                <Text type="secondary" code>{item.file_path}</Text>
                              )}
                            </div>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </div>
              )}
            </Space>
          </Card>
        </TabPane>

        {/* 文档添加 */}
        <TabPane tab="文档添加" key="documents">
          <Card title="添加文本到索引">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <TextArea
                rows={6}
                value={docText}
                onChange={(e) => setDocText(e.target.value)}
                name="rag-doc-text"
                placeholder="输入文档内容"
              />
              <Input
                value={docMetadata}
                onChange={(e) => setDocMetadata(e.target.value)}
                name="rag-doc-metadata"
                placeholder={'可选：metadata JSON，例如 {"source":"manual"}'}
              />
              <Button
                type="primary"
                icon={<FileTextOutlined />}
                onClick={handleAddDocument}
                loading={actionLoading}
              >
                添加文档
              </Button>
            </Space>
          </Card>
        </TabPane>

        {/* 索引管理 */}
        <TabPane tab="索引管理" key="index">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="索引操作" className="mb-4">
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  <Input
                    value={indexFilePath}
                    onChange={(e) => setIndexFilePath(e.target.value)}
                    name="rag-index-file"
                    placeholder="文件路径（支持 .py/.ts/.md/.txt/.json/.yaml 等）"
                  />
                  <Button
                    type="primary"
                    icon={<FileTextOutlined />}
                    onClick={handleIndexFile}
                    loading={actionLoading}
                    block
                  >
                    索引文件
                  </Button>

                  <Input
                    value={indexDirectoryPath}
                    onChange={(e) => setIndexDirectoryPath(e.target.value)}
                    name="rag-index-directory"
                    placeholder="目录路径"
                  />
                  <Input
                    value={indexExtensions}
                    onChange={(e) => setIndexExtensions(e.target.value)}
                    name="rag-index-extensions"
                    placeholder="可选：扩展名过滤（逗号分隔，如 .md,.txt）"
                  />
                  <Space>
                    <Switch checked={indexRecursive} onChange={setIndexRecursive} />
                    <Text>递归</Text>
                    <Switch checked={indexForce} onChange={setIndexForce} />
                    <Text>强制</Text>
                  </Space>
                  <Button
                    icon={<DatabaseOutlined />}
                    onClick={handleIndexDirectory}
                    loading={actionLoading}
                    block
                  >
                    索引目录
                  </Button>

                  <Button
                    danger
                    icon={<ReloadOutlined />}
                    onClick={handleResetIndex}
                    loading={actionLoading}
                    block
                  >
                    重置索引
                  </Button>

                  <Button
                    icon={<ApiOutlined />}
                    onClick={handleHealthCheck}
                    loading={actionLoading}
                    block
                  >
                    健康检查
                  </Button>

                  <Button
                    icon={<DatabaseOutlined />}
                    onClick={loadStats}
                    loading={actionLoading}
                    block
                  >
                    刷新统计
                  </Button>
                </Space>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="索引状态">
                <div className="space-y-4">
                  <div>
                    <Text>最后更新时间：</Text>
                    <Text type="secondary">
                      {stats.last_updated ? new Date(stats.last_updated).toLocaleString() : '未知'}
                    </Text>
                  </div>
                  
                  <div>
                    <Text>存储使用率：</Text>
                    <Progress
                      percent={Math.min((stats.index_size / 1024 / 1024 / 1024) * 100, 100)}
                      status="active"
                      format={() => `${Number((stats.index_size / 1024 / 1024).toFixed(2))}MB`}
                    />
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* API调用历史 */}
        <TabPane tab={`API调用历史 (${apiCalls.length})`} key="api-calls">
          <Card title="API调用记录">
            <List
              dataSource={apiCalls}
              renderItem={(call, index) => (
                <List.Item key={index}>
                  <List.Item.Meta
                    avatar={getStatusIcon(call.status)}
                    title={
                      <Space>
                        <Tag color="blue">{call.method}</Tag>
                        <Text code>{call.endpoint}</Text>
                        <Badge
                          status={call.status === 'success' ? 'success' : call.status === 'error' ? 'error' : 'processing'}
                          text={call.status.toUpperCase()}
                        />
                      </Space>
                    }
                    description={
                      <div>
                        <Text type="secondary">{call.timestamp}</Text>
                        {call.error && (
                          <div>
                            <Text type="danger">错误: {call.error}</Text>
                          </div>
                        )}
                        {call.response && call.status === 'success' && (
                          <div>
                            <Text type="success">
                              响应: {JSON.stringify(call.response).substring(0, 100)}...
                            </Text>
                          </div>
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default BasicRagManagementPage;
