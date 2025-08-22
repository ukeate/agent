/**
 * RAG索引状态监控组件
 * 
 * 功能包括：
 * - 显示向量数据库连接状态和实时性能指标
 * - 实现索引统计信息展示（文档数量、向量维度、存储大小）
 * - 添加索引更新进度展示和手动更新功能
 * - 实现索引健康状态检查和错误诊断提示
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Button,
  Badge,
  Space,
  Alert,
  Tooltip,
  Modal,
  Typography,
  Upload,
  Input,
  message,
} from 'antd';
import {
  DatabaseOutlined,
  CloudServerOutlined,
  ReloadOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  FileAddOutlined,
  FolderAddOutlined,
  DeleteOutlined,
  UploadOutlined,
} from '@ant-design/icons';
import { useRagStore } from '../../stores/ragStore';
import { ragService } from '../../services/ragService';

const { Text, Title } = Typography;
const { Dragger } = Upload;

// ==================== 组件props类型 ====================

interface RagIndexStatusProps {
  autoRefresh?: boolean;
  refreshInterval?: number;
  className?: string;
}

// ==================== 辅助类型 ====================

interface IndexingProgress {
  isIndexing: boolean;
  progress: number;
  currentFile: string;
  processed: number;
  total: number;
  errors: string[];
}

// ==================== 主组件 ====================

const RagIndexStatus: React.FC<RagIndexStatusProps> = ({
  autoRefresh = true,
  refreshInterval = 30000, // 30秒自动刷新
  className = '',
}) => {
  // ==================== 状态管理 ====================
  
  const {
    indexStatus,
    updateIndexStatus,
    setIndexLoading,
    setError,
    clearErrors,
  } = useRagStore();

  // ==================== 本地状态 ====================
  
  const [healthCheck, setHealthCheck] = useState<{
    status: 'healthy' | 'unhealthy' | 'checking';
    lastCheck: Date | null;
    details: any;
  }>({
    status: 'checking',
    lastCheck: null,
    details: null,
  });

  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress>({
    isIndexing: false,
    progress: 0,
    currentFile: '',
    processed: 0,
    total: 0,
    errors: [],
  });

  const [showIndexModal, setShowIndexModal] = useState(false);
  const [indexPath, setIndexPath] = useState('');
  const [indexRecursive, setIndexRecursive] = useState(true);
  const [showResetModal, setShowResetModal] = useState(false);

  // ==================== 自动刷新逻辑 ====================
  
  const fetchIndexStats = useCallback(async () => {
    try {
      setIndexLoading(true);
      clearErrors();
      
      const stats = await ragService.getIndexStats();
      updateIndexStatus(stats);
      
      return stats.success;
    } catch (error: any) {
      setError(error.message || '获取索引状态失败');
      return false;
    } finally {
      setIndexLoading(false);
    }
  }, [updateIndexStatus, setIndexLoading, setError, clearErrors]);

  const performHealthCheck = useCallback(async () => {
    setHealthCheck(prev => ({ ...prev, status: 'checking' }));
    
    try {
      const result = await ragService.healthCheck();
      
      setHealthCheck({
        status: result.status === 'healthy' ? 'healthy' : 'unhealthy',
        lastCheck: new Date(),
        details: result,
      });
      
      return result.status === 'healthy';
    } catch (error) {
      setHealthCheck({
        status: 'unhealthy',
        lastCheck: new Date(),
        details: { error: error instanceof Error ? error.message : '健康检查失败' },
      });
      
      return false;
    }
  }, []);

  const refreshStatus = useCallback(async () => {
    const statsSuccess = await fetchIndexStats();
    const healthSuccess = await performHealthCheck();
    
    if (statsSuccess && healthSuccess) {
      message.success('状态更新成功');
    }
  }, [fetchIndexStats, performHealthCheck]);

  // ==================== 索引操作 ====================
  
  const handleIndexDirectory = useCallback(async () => {
    if (!indexPath.trim()) {
      message.error('请输入目录路径');
      return;
    }

    try {
      setIndexingProgress({
        isIndexing: true,
        progress: 0,
        currentFile: '正在准备...',
        processed: 0,
        total: 0,
        errors: [],
      });

      const result = await ragService.indexDirectory(
        indexPath.trim(),
        indexRecursive,
        false // force
      );

      if (result.success) {
        message.success(`成功索引 ${result.indexed_files} 个文件`);
        await refreshStatus();
      } else {
        throw new Error(result.error || '索引失败');
      }

    } catch (error: any) {
      message.error(error.message || '索引目录失败');
      setIndexingProgress(prev => ({
        ...prev,
        errors: [...prev.errors, error.message || '索引失败'],
      }));
    } finally {
      setIndexingProgress(prev => ({
        ...prev,
        isIndexing: false,
        progress: 100,
        currentFile: '索引完成',
      }));
      
      setShowIndexModal(false);
      setIndexPath('');
    }
  }, [indexPath, indexRecursive, refreshStatus]);

  const handleResetIndex = useCallback(async () => {
    try {
      setIndexLoading(true);
      const result = await ragService.resetIndex();
      
      if (result.success) {
        message.success('索引重置成功');
        await refreshStatus();
      } else {
        throw new Error(result.error || '重置失败');
      }
      
    } catch (error: any) {
      message.error(error.message || '重置索引失败');
    } finally {
      setIndexLoading(false);
      setShowResetModal(false);
    }
  }, [setIndexLoading, refreshStatus]);

  const handleFileUpload = useCallback(async (_file: File) => {
    // 这里应该实现文件上传到服务器并索引的逻辑
    message.info('文件上传功能需要后端支持');
    return false; // 阻止默认上传行为
  }, []);

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    // 组件加载时获取初始状态
    refreshStatus();
  }, [refreshStatus]);

  useEffect(() => {
    // 自动刷新逻辑
    if (!autoRefresh) return;

    const interval = setInterval(refreshStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, refreshStatus]);

  // ==================== 辅助函数 ====================
  
  const formatBytes = useCallback((bytes: number) => {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }, []);

  const getStatusBadge = useCallback(() => {
    if (indexStatus.is_loading || healthCheck.status === 'checking') {
      return <Badge status="processing" text="检查中..." />;
    }
    
    if (indexStatus.status === 'healthy' && healthCheck.status === 'healthy') {
      return <Badge status="success" text="正常" />;
    }
    
    if (indexStatus.status === 'unhealthy' || healthCheck.status === 'unhealthy') {
      return <Badge status="error" text="异常" />;
    }
    
    return <Badge status="default" text="未知" />;
  }, [indexStatus, healthCheck]);

  // ==================== 渲染组件 ====================

  return (
    <div className={`rag-index-status ${className}`}>
      
      {/* 主状态卡片 */}
      <Card 
        title={
          <Space>
            <DatabaseOutlined />
            <Title level={4} style={{ margin: 0 }}>索引状态</Title>
            {getStatusBadge()}
          </Space>
        }
        extra={
          <Space>
            <Tooltip title="刷新状态">
              <Button
                icon={<ReloadOutlined />}
                onClick={refreshStatus}
                loading={indexStatus.is_loading}
                size="small"
              />
            </Tooltip>
            <Tooltip title="添加索引">
              <Button
                icon={<FileAddOutlined />}
                onClick={() => setShowIndexModal(true)}
                type="primary"
                size="small"
              />
            </Tooltip>
          </Space>
        }
      >
        {/* 统计信息 */}
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic
              title="文档数量"
              value={indexStatus.total_documents}
              prefix={<FileAddOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="向量数量"
              value={indexStatus.total_vectors}
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="存储大小"
              value={formatBytes(indexStatus.index_size)}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="最后更新"
              value={indexStatus.last_updated ? 
                new Date(indexStatus.last_updated).toLocaleDateString() : 
                '未知'
              }
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Col>
        </Row>

        {/* 健康检查信息 */}
        {healthCheck.lastCheck && (
          <Alert
            message={
              <Space>
                <Text>
                  健康检查: {healthCheck.status === 'healthy' ? '正常' : '异常'}
                </Text>
                <Text type="secondary">
                  上次检查: {healthCheck.lastCheck.toLocaleTimeString()}
                </Text>
              </Space>
            }
            type={healthCheck.status === 'healthy' ? 'success' : 'warning'}
            icon={healthCheck.status === 'healthy' ? 
              <CheckCircleOutlined /> : 
              <WarningOutlined />
            }
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        {/* 索引进度 */}
        {indexingProgress.isIndexing && (
          <Card size="small" title="索引进度" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Progress
                percent={indexingProgress.progress}
                status={indexingProgress.errors.length > 0 ? 'exception' : 'active'}
              />
              <Text type="secondary">
                当前文件: {indexingProgress.currentFile}
              </Text>
              {indexingProgress.total > 0 && (
                <Text type="secondary">
                  进度: {indexingProgress.processed} / {indexingProgress.total}
                </Text>
              )}
              {indexingProgress.errors.length > 0 && (
                <Alert
                  message={`发现 ${indexingProgress.errors.length} 个错误`}
                  variant="destructive"
                />
              )}
            </Space>
          </Card>
        )}

        {/* 操作按钮 */}
        <Row gutter={16}>
          <Col span={12}>
            <Button
              block
              icon={<FolderAddOutlined />}
              onClick={() => setShowIndexModal(true)}
              disabled={indexingProgress.isIndexing}
            >
              索引目录
            </Button>
          </Col>
          <Col span={12}>
            <Button
              block
              danger
              icon={<DeleteOutlined />}
              onClick={() => setShowResetModal(true)}
              disabled={indexingProgress.isIndexing}
            >
              重置索引
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 文件上传区域 */}
      <Card title="文件上传" size="small" style={{ marginTop: 16 }}>
        <Dragger
          multiple
          beforeUpload={handleFileUpload}
          showUploadList={false}
          disabled={indexingProgress.isIndexing}
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持单个或批量上传，上传后将自动建立索引
          </p>
        </Dragger>
      </Card>

      {/* 索引目录模态框 */}
      <Modal
        title="索引目录"
        open={showIndexModal}
        onOk={handleIndexDirectory}
        onCancel={() => {
          setShowIndexModal(false);
          setIndexPath('');
        }}
        confirmLoading={indexingProgress.isIndexing}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>目录路径:</Text>
            <Input
              value={indexPath}
              onChange={(e) => setIndexPath(e.target.value)}
              placeholder="请输入要索引的目录路径，例如: /path/to/documents"
              style={{ marginTop: 8 }}
            />
          </div>
          
          <div>
            <Space>
              <input
                type="checkbox"
                checked={indexRecursive}
                onChange={(e) => setIndexRecursive(e.target.checked)}
              />
              <Text>递归索引子目录</Text>
            </Space>
          </div>

          <Alert
            message="提示"
            description="索引过程可能需要一些时间，请耐心等待。大量文件的索引会消耗较多系统资源。"
            variant="default"
            showIcon
          />
        </Space>
      </Modal>

      {/* 重置索引确认模态框 */}
      <Modal
        title="重置索引"
        open={showResetModal}
        onOk={handleResetIndex}
        onCancel={() => setShowResetModal(false)}
        okText="确认重置"
        cancelText="取消"
        okType="danger"
        confirmLoading={indexStatus.is_loading}
      >
        <Space direction="vertical">
          <Alert
            message="警告"
            description="此操作将删除所有已建立的索引数据，包括向量数据和元数据。此操作不可撤销！"
            variant="warning"
            showIcon
          />
          <Text>请确认您要重置所有索引数据。</Text>
        </Space>
      </Modal>

    </div>
  );
};

export default RagIndexStatus;