import React, { useState } from 'react';
import { Modal, List, Progress, Tag, Typography, Space, Button, Alert } from 'antd';
import { 
  CheckCircleOutlined, 
  ClockCircleOutlined, 
  CloseCircleOutlined,
  SyncOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface BatchProcessorProps {
  visible: boolean;
  onClose: () => void;
  batchId?: string;
  items?: any[];
}

const BatchProcessor: React.FC<BatchProcessorProps> = ({ 
  visible, 
  onClose, 
  batchId,
  items = [] 
}) => {
  const [processing, setProcessing] = useState(false);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'processing':
        return <SyncOutlined spin style={{ color: '#1890ff' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const completedCount = items.filter(item => item.status === 'completed').length;
  const failedCount = items.filter(item => item.status === 'failed').length;
  const progress = items.length > 0 ? (completedCount / items.length) * 100 : 0;

  return (
    <Modal
      title={`批量处理监控 ${batchId ? `- ${batchId}` : ''}`}
      visible={visible}
      onCancel={onClose}
      width={700}
      footer={[
        <Button key="close" onClick={onClose}>
          关闭
        </Button>,
        <Button key="refresh" icon={<SyncOutlined />} onClick={() => {}}>
          刷新状态
        </Button>
      ]}
    >
      {items.length > 0 ? (
        <>
          <Progress
            percent={Number(progress.toFixed(1))}
            status={failedCount > 0 ? 'exception' : 'active'}
            format={percent => `${completedCount} / ${items.length}`}
          />
          
          <div className="mt-2 mb-4">
            <Space>
              <Tag color="success">完成: {completedCount}</Tag>
              <Tag color="processing">处理中: {items.filter(i => i.status === 'processing').length}</Tag>
              <Tag color="default">等待: {items.filter(i => i.status === 'pending').length}</Tag>
              {failedCount > 0 && <Tag color="error">失败: {failedCount}</Tag>}
            </Space>
          </div>

          <List
            className="max-h-96 overflow-auto"
            dataSource={items}
            renderItem={(item: any) => (
              <List.Item>
                <List.Item.Meta
                  avatar={getStatusIcon(item.status)}
                  title={
                    <Space>
                      <Text>{item.fileName || item.content_id}</Text>
                      <Tag>{item.fileType}</Tag>
                    </Space>
                  }
                  description={
                    <div>
                      {item.status === 'completed' && (
                        <Text type="success">处理成功 - 耗时: {item.processingTime}s</Text>
                      )}
                      {item.status === 'failed' && (
                        <Text type="danger">处理失败: {item.error}</Text>
                      )}
                      {item.status === 'processing' && (
                        <Text type="secondary">正在处理...</Text>
                      )}
                      {item.status === 'pending' && (
                        <Text type="secondary">等待处理</Text>
                      )}
                    </div>
                  }
                />
              </List.Item>
            )}
          />

          {failedCount > 0 && (
            <Alert
              className="mt-4"
              message={`有 ${failedCount} 个文件处理失败`}
              description="失败的文件可以选择重试或查看错误详情"
              variant="destructive"
              showIcon
            />
          )}
        </>
      ) : (
        <div className="text-center py-8">
          <Text type="secondary">暂无批处理任务</Text>
        </div>
      )}
    </Modal>
  );
};

export default BatchProcessor;