import React from 'react';
import { Card, Row, Col, Statistic, Progress, Tag, Alert, Badge, Space } from 'antd';
import {
  ClockCircleOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  ExclamationCircleOutlined,
  TeamOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

interface ProcessingQueueProps {
  status: any;
}

const ProcessingQueue: React.FC<ProcessingQueueProps> = ({ status }) => {
  if (!status) {
    return (
      <Card title="处理队列状态" loading>
        加载中...
      </Card>
    );
  }

  const totalTasks = (status.queued_tasks || 0) + (status.completed_tasks || 0) + (status.failed_tasks || 0);
  const completionRate = totalTasks > 0 ? ((status.completed_tasks || 0) / totalTasks * 100) : 0;

  return (
    <Card 
      title={
        <div className="flex items-center justify-between">
          <span>处理队列监控</span>
          <Badge 
            status={status.is_running ? "processing" : "default"} 
            text={status.is_running ? "运行中" : "已停止"}
          />
        </div>
      }
    >
      <Row gutter={[16, 16]}>
        <Col xs={12} sm={6}>
          <Statistic
            title="活跃任务"
            value={status.active_tasks || 0}
            prefix={<SyncOutlined spin={status.active_tasks > 0} />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col xs={12} sm={6}>
          <Statistic
            title="排队中"
            value={status.queued_tasks || 0}
            prefix={<ClockCircleOutlined />}
            valueStyle={{ color: '#faad14' }}
          />
        </Col>
        <Col xs={12} sm={6}>
          <Statistic
            title="已完成"
            value={status.completed_tasks || 0}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col xs={12} sm={6}>
          <Statistic
            title="失败"
            value={status.failed_tasks || 0}
            prefix={<ExclamationCircleOutlined />}
            valueStyle={{ color: status.failed_tasks > 0 ? '#ff4d4f' : undefined }}
          />
        </Col>
      </Row>

      {totalTasks > 0 && (
        <div className="mt-4">
          <Progress
            percent={Number(completionRate.toFixed(1))}
            status={status.failed_tasks > 0 ? "exception" : "active"}
            strokeColor={{
              '0%': '#108ee9',
              '100%': '#87d068',
            }}
          />
          <div className="mt-2 flex justify-between text-sm text-gray-500">
            <span>完成率</span>
            <span>{status.completed_tasks} / {totalTasks} 任务</span>
          </div>
        </div>
      )}

      {status.active_tasks > 0 && (
        <Alert
          className="mt-4"
          message={`正在处理 ${status.active_tasks} 个任务`}
          variant="default"
          icon={<SyncOutlined spin />}
          showIcon
        />
      )}

      {status.failed_tasks > 0 && (
        <Alert
          className="mt-4"
          message={`有 ${status.failed_tasks} 个任务处理失败`}
          variant="destructive"
          showIcon
        />
      )}

      <div className="mt-4">
        <Space>
          <Tag icon={<TeamOutlined />} color="blue">
            并发数: 5
          </Tag>
          <Tag icon={<ThunderboltOutlined />} color="green">
            自动重试: 启用
          </Tag>
        </Space>
      </div>
    </Card>
  );
};

export default ProcessingQueue;