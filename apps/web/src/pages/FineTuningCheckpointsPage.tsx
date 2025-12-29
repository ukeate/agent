import React, { useEffect, useMemo, useState } from 'react';
import { Card, Table, Tag, Button, Space, message } from 'antd';
import { DownloadOutlined, ReloadOutlined } from '@ant-design/icons';
import { fineTuningService, TrainingJob } from '../services/fineTuningService';

import { logger } from '../utils/logger'
type CheckpointRow = {
  key: string;
  job_id: string;
  job_name: string;
  status: string;
  completed_at?: string;
  output_dir?: string;
};

const statusColor: Record<string, string> = {
  running: 'processing',
  completed: 'success',
  failed: 'error',
  pending: 'default',
  paused: 'warning',
  cancelled: 'default',
};

const FineTuningCheckpointsPage: React.FC = () => {
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    try {
      setLoading(true);
      const list = await fineTuningService.getTrainingJobs();
      setJobs(list);
    } catch (e) {
      logger.error('加载检查点列表失败:', e);
      message.error('加载检查点列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const rows = useMemo<CheckpointRow[]>(
    () =>
      jobs.map(job => ({
        key: job.job_id,
        job_id: job.job_id,
        job_name: job.job_name,
        status: job.status,
        completed_at: job.completed_at,
        output_dir: job.config?.output_dir,
      })),
    [jobs]
  );

  const columns = [
    { title: '任务名称', dataIndex: 'job_name', key: 'job_name' },
    { title: '任务ID', dataIndex: 'job_id', key: 'job_id' },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (value: string) => <Tag color={statusColor[value] || 'default'}>{value}</Tag>,
    },
    { title: '输出目录', dataIndex: 'output_dir', key: 'output_dir', render: (v: string) => v || '-' },
    { title: '完成时间', dataIndex: 'completed_at', key: 'completed_at', render: (v: string) => v || '-' },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, record: CheckpointRow) => (
        <Button
          icon={<DownloadOutlined />}
          size="small"
          disabled={record.status !== 'completed'}
          onClick={() => {
            const a = document.createElement('a');
            a.href = `/api/v1/fine-tuning/jobs/${record.job_id}/download`;
            a.rel = 'noopener';
            document.body.appendChild(a);
            a.click();
            a.remove();
          }}
        >
          下载产物
        </Button>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title="模型检查点与产物下载"
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
          </Space>
        }
      >
        <Table columns={columns} dataSource={rows} loading={loading} />
      </Card>
    </div>
  );
};

export default FineTuningCheckpointsPage;
