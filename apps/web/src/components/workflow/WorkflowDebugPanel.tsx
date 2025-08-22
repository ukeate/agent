import React, { useState } from 'react';
import { 
  Drawer, 
  Tabs, 
  Timeline, 
  Table, 
  Tag, 
  Button, 
  Space, 
  Collapse,
  Typography,
  Divider
} from 'antd';
import { 
  BugOutlined, 
  HistoryOutlined, 
  InfoCircleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

const { Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface StateHistoryItem {
  timestamp: string;
  nodeId: string;
  nodeName: string;
  previousStatus: string;
  newStatus: string;
  metadata?: any;
}

interface ExecutionStep {
  stepId: string;
  timestamp: string;
  nodeId: string;
  nodeName: string;
  action: string;
  duration: number;
  status: 'success' | 'error' | 'warning';
  details?: any;
  error?: string;
}

interface WorkflowDebugPanelProps {
  visible: boolean;
  onClose: () => void;
  workflowId: string;
  stateHistory?: StateHistoryItem[];
  executionSteps?: ExecutionStep[];
  currentState?: any;
}

export const WorkflowDebugPanel: React.FC<WorkflowDebugPanelProps> = ({
  visible,
  onClose,
  workflowId,
  stateHistory = [],
  executionSteps = [],
  currentState
}) => {
  const [activeTab, setActiveTab] = useState('history');

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'running':
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
      case 'failed':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'paused':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      default:
        return <InfoCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getStatusTag = (status: string) => {
    const colorMap = {
      'success': 'green',
      'error': 'red',
      'warning': 'orange',
      'completed': 'green',
      'running': 'blue',
      'failed': 'red',
      'paused': 'orange',
      'pending': 'default'
    };
    return <Tag color={colorMap[status as keyof typeof colorMap]}>{status}</Tag>;
  };

  // 状态历史表格列
  const historyColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
    },
    {
      title: '节点',
      dataIndex: 'nodeName',
      key: 'nodeName',
      width: 120,
    },
    {
      title: '状态变化',
      key: 'statusChange',
      render: (record: StateHistoryItem) => (
        <Space>
          {getStatusTag(record.previousStatus)}
          <span>→</span>
          {getStatusTag(record.newStatus)}
        </Space>
      ),
    },
  ];

  // 执行步骤表格列
  const stepsColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
    },
    {
      title: '节点',
      dataIndex: 'nodeName',
      key: 'nodeName',
      width: 120,
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      width: 100,
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      width: 80,
      render: (duration: number) => `${duration}ms`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => getStatusTag(status),
    },
    {
      title: '详情',
      key: 'details',
      render: (record: ExecutionStep) => (
        <Button 
          size="small" 
          type="link"
          onClick={() => console.log('查看详情:', record)}
        >
          查看
        </Button>
      ),
    },
  ];

  // 渲染状态历史时间轴
  const renderStateTimeline = () => {
    const timelineItems = stateHistory.map((item) => ({
      color: item.newStatus === 'failed' ? 'red' : 
             item.newStatus === 'completed' ? 'green' : 'blue',
      dot: getStatusIcon(item.newStatus),
      children: (
        <div>
          <div style={{ fontWeight: 'bold' }}>
            {item.nodeName} - {item.timestamp}
          </div>
          <div>
            状态从 {getStatusTag(item.previousStatus)} 变更为 {getStatusTag(item.newStatus)}
          </div>
          {item.metadata && (
            <Collapse size="small" style={{ marginTop: 8 }}>
              <Panel header="元数据" key="1">
                <pre style={{ fontSize: '12px' }}>
                  {JSON.stringify(item.metadata, null, 2)}
                </pre>
              </Panel>
            </Collapse>
          )}
        </div>
      ),
    }));

    return <Timeline items={timelineItems} />;
  };

  // 渲染当前状态
  const renderCurrentState = () => {
    if (!currentState) {
      return <Text type="secondary">暂无状态信息</Text>;
    }

    return (
      <div>
        <Paragraph>
          <Text strong>工作流状态:</Text> {getStatusTag(currentState.status)}
        </Paragraph>
        
        <Paragraph>
          <Text strong>创建时间:</Text> {currentState.created_at || 'N/A'}
        </Paragraph>
        
        <Paragraph>
          <Text strong>开始时间:</Text> {currentState.started_at || 'N/A'}
        </Paragraph>
        
        {currentState.current_state && (
          <>
            <Divider />
            <Text strong>当前状态详情:</Text>
            <Collapse style={{ marginTop: 8 }}>
              <Panel header="状态数据" key="1">
                <pre style={{ fontSize: '12px', maxHeight: '300px', overflow: 'auto' }}>
                  {JSON.stringify(currentState.current_state, null, 2)}
                </pre>
              </Panel>
            </Collapse>
          </>
        )}
      </div>
    );
  };

  const tabItems = [
    {
      key: 'history',
      label: (
        <span>
          <HistoryOutlined />
          状态历史
        </span>
      ),
      children: (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Text type="secondary">显示工作流节点状态变化历史</Text>
          </div>
          <Tabs
            size="small"
            defaultActiveKey="timeline"
            items={[
              {
                key: 'timeline',
                label: '时间轴视图',
                children: renderStateTimeline(),
              },
              {
                key: 'table',
                label: '表格视图',
                children: (
                  <Table
                    dataSource={stateHistory}
                    columns={historyColumns}
                    size="small"
                    pagination={{ pageSize: 10 }}
                    rowKey={(record) => `${record.timestamp}-${record.nodeId}`}
                  />
                ),
              },
            ]}
          />
        </div>
      ),
    },
    {
      key: 'execution',
      label: (
        <span>
          <BugOutlined />
          执行日志
        </span>
      ),
      children: (
        <div>
          <div style={{ marginBottom: 16 }}>
            <Text type="secondary">显示详细的执行步骤和性能信息</Text>
          </div>
          <Table
            dataSource={executionSteps}
            columns={stepsColumns}
            size="small"
            pagination={{ pageSize: 10 }}
            rowKey="stepId"
            expandable={{
              expandedRowRender: (record) => (
                <div style={{ padding: 16, background: '#fafafa' }}>
                  {record.details && (
                    <>
                      <Text strong>详细信息:</Text>
                      <pre style={{ marginTop: 8, fontSize: '12px' }}>
                        {JSON.stringify(record.details, null, 2)}
                      </pre>
                    </>
                  )}
                  {record.error && (
                    <>
                      <Text strong style={{ color: '#ff4d4f' }}>错误信息:</Text>
                      <div style={{ marginTop: 8, color: '#ff4d4f' }}>
                        {record.error}
                      </div>
                    </>
                  )}
                </div>
              ),
            }}
          />
        </div>
      ),
    },
    {
      key: 'state',
      label: (
        <span>
          <InfoCircleOutlined />
          当前状态
        </span>
      ),
      children: renderCurrentState(),
    },
  ];

  return (
    <Drawer
      title={`工作流调试 - ${workflowId}`}
      placement="bottom"
      onClose={onClose}
      open={visible}
      height="60vh"
      extra={
        <Space>
          <Button size="small" onClick={() => window.location.reload()}>
            刷新页面
          </Button>
          <Button size="small" type="primary" onClick={onClose}>
            关闭
          </Button>
        </Space>
      }
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
      />
    </Drawer>
  );
};

export default WorkflowDebugPanel;