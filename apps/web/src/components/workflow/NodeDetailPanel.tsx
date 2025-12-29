import React from 'react';
import { Drawer, Descriptions, Badge, Timeline, Button, Space } from 'antd';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';

interface WorkflowState {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  type: 'start' | 'process' | 'decision' | 'end';
}

interface ExecutionLog {
  timestamp: string;
  message: string;
  level: 'info' | 'warning' | 'error';
}

interface NodeDetailPanelProps {
  visible: boolean;
  onClose: () => void;
  nodeData: WorkflowState | null;
  executionLogs?: ExecutionLog[];
  onNodeAction?: (action: string, nodeId: string) => void;
}

export const NodeDetailPanel: React.FC<NodeDetailPanelProps> = ({
  visible,
  onClose,
  nodeData,
  executionLogs = [],
  onNodeAction = () => {},
}) => {
  if (!nodeData) return null;

  const getStatusBadge = (status: string) => {
    const statusMap = {
      pending: { status: 'default', text: '待执行' },
      running: { status: 'processing', text: '执行中' },
      completed: { status: 'success', text: '已完成' },
      failed: { status: 'error', text: '失败' },
      paused: { status: 'warning', text: '暂停' },
    };
    const config = statusMap[status as keyof typeof statusMap] || statusMap.pending;
    return <Badge status={config.status as any} text={config.text} />;
  };

  const getTypeDescription = (type: string) => {
    const typeMap = {
      start: '开始节点',
      process: '处理节点',
      decision: '决策节点',
      end: '结束节点',
    };
    return typeMap[type as keyof typeof typeMap] || '未知类型';
  };

  const handleAction = (action: string) => {
    onNodeAction(action, nodeData.id);
  };

  const renderActionButtons = () => {
    const { status } = nodeData;
    
    return (
      <Space>
        {status === 'paused' && (
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            onClick={() => handleAction('resume')}
          >
            恢复
          </Button>
        )}
        {status === 'running' && (
          <Button 
            icon={<PauseCircleOutlined />}
            onClick={() => handleAction('pause')}
          >
            暂停
          </Button>
        )}
        {(status === 'running' || status === 'paused') && (
          <Button 
            danger 
            icon={<StopOutlined />}
            onClick={() => handleAction('cancel')}
          >
            取消
          </Button>
        )}
        {status === 'failed' && (
          <Button 
            type="primary" 
            onClick={() => handleAction('restart')}
          >
            重新运行
          </Button>
        )}
      </Space>
    );
  };

  const renderExecutionTimeline = () => {
    const timelineItems = executionLogs.map((log) => ({
      color: log.level === 'error' ? 'red' : log.level === 'warning' ? 'orange' : 'blue',
      children: (
        <div>
          <div style={{ fontWeight: 'bold' }}>{log.timestamp}</div>
          <div>{log.message}</div>
        </div>
      ),
    }));

    return timelineItems.length > 0 ? (
      <Timeline items={timelineItems} />
    ) : (
      <div className="text-gray-500">暂无执行日志</div>
    );
  };

  return (
    <Drawer
      title={`节点详情 - ${nodeData.name}`}
      placement="right"
      onClose={onClose}
      open={visible}
      width={400}
    >
      <Descriptions column={1} bordered>
        <Descriptions.Item label="节点ID">{nodeData.id}</Descriptions.Item>
        <Descriptions.Item label="节点名称">{nodeData.name}</Descriptions.Item>
        <Descriptions.Item label="节点类型">{getTypeDescription(nodeData.type)}</Descriptions.Item>
        <Descriptions.Item label="执行状态">{getStatusBadge(nodeData.status)}</Descriptions.Item>
      </Descriptions>

      <div style={{ marginTop: 16 }}>
        <h4>操作控制</h4>
        {renderActionButtons()}
      </div>

      <div style={{ marginTop: 24 }}>
        <h4>执行日志</h4>
        {renderExecutionTimeline()}
      </div>
    </Drawer>
  );
};

export default NodeDetailPanel;
