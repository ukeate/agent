import React from 'react';
import { Badge, Progress, Space, Typography } from 'antd';
import { LoadingOutlined, CheckCircleOutlined, CloseCircleOutlined, PauseCircleOutlined, ClockCircleOutlined } from '@ant-design/icons';

interface NodeStatusProps {
  nodeId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused' | 'skipped';
  progress?: number;
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export const NodeStatus: React.FC<NodeStatusProps> = ({
  nodeId,
  status,
  progress = 0,
  startedAt,
  completedAt,
  error
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <LoadingOutlined spin style={{ color: '#1890ff' }} />;
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'paused':
        return <PauseCircleOutlined style={{ color: '#faad14' }} />;
      case 'pending':
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running': return 'processing';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const calculateDuration = () => {
    if (!startedAt) return null;
    const start = new Date(startedAt).getTime();
    const end = completedAt ? new Date(completedAt).getTime() : Date.now();
    const duration = Math.floor((end - start) / 1000);
    return `${duration}s`;
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Space>
          {getStatusIcon()}
          <Typography.Text strong>{nodeId}</Typography.Text>
        </Space>
        <Badge status={getStatusColor()} text={status} />
      </div>

      {status === 'running' && (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Progress percent={progress} size="small" />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            进度: {progress}%
            {startedAt && ` • 耗时: ${calculateDuration()}`}
          </Typography.Text>
        </Space>
      )}

      {status === 'completed' && startedAt && completedAt && (
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          完成耗时: {calculateDuration()}
        </Typography.Text>
      )}

      {status === 'failed' && error && (
        <div style={{ fontSize: 12, color: '#ff4d4f', backgroundColor: '#fff2f0', padding: 8, borderRadius: 6 }}>
          错误: {error}
        </div>
      )}
    </Space>
  );
};