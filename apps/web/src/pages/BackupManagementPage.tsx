import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Space, Typography, Button, Alert, Spin, Statistic, Tag } from 'antd';
import { DatabaseOutlined, ReloadOutlined, CloudDownloadOutlined } from '@ant-design/icons';

type BackupStats = {
  total_backups: number;
  total_size: number;
  success_rate: number;
  last_backup_time: string;
  components?: Record<string, { backup_count: number; last_backup: string; total_size: number }>;
};

type BackupRecord = {
  backup_id: string;
  component_id: string;
  backup_type: string;
  created_at: string;
  size: number;
  checksum?: string;
  storage_path?: string;
  valid?: boolean;
};

const BackupManagementPage: React.FC = () => {
  const [stats, setStats] = useState<BackupStats | null>(null);
  const [records, setRecords] = useState<BackupRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const resStats = await apiFetch(buildApiUrl('/api/v1/fault-tolerance/backup/statistics'));
      const dataStats = await resStats.json();
      setStats(dataStats);

    } catch (e: any) {
      setError(e?.message || '加载失败');
      setStats(null);
      setRecords([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const totalSizeGB = stats ? (stats.total_size || 0) / (1024 * 1024 * 1024) : 0;

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <DatabaseOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              备份管理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="统计">
          {loading ? (
            <Spin />
          ) : stats ? (
            <Space size="large">
              <Statistic title="备份总数" value={stats.total_backups} />
              <Statistic title="总大小(GB)" value={totalSizeGB.toFixed(2)} />
              <Statistic title="成功率" value={(stats.success_rate || 0) * 100} suffix="%" />
              <Statistic title="最近备份时间" value={stats.last_backup_time} />
            </Space>
          ) : (
            <Alert type="info" message="暂无备份统计，请先触发 /api/v1/fault-tolerance/backup/manual。" />
          )}
        </Card>

        <Card title="备份列表">
          {loading ? (
            <Spin />
          ) : records.length === 0 ? (
            <Alert type="info" message="暂无备份记录，先执行备份后查看。" />
          ) : (
            <Table
              rowKey="backup_id"
              dataSource={records}
              columns={[
                { title: 'ID', dataIndex: 'backup_id' },
                { title: '组件', dataIndex: 'component_id' },
                { title: '类型', dataIndex: 'backup_type' },
                { title: '时间', dataIndex: 'created_at' },
                { title: '大小(bytes)', dataIndex: 'size' },
                {
                  title: '有效',
                  dataIndex: 'valid',
                  render: (v) => <Tag color={v ? 'green' : 'red'}>{v ? '是' : '否'}</Tag>
                },
              ]}
            />
          )}
        </Card>

        <Alert
          type="info"
          icon={<CloudDownloadOutlined />}
          message="操作指引"
          description="手动/恢复/校验请直接调用后端接口：/backup/manual、/backup/restore、/backup/validate，页面不再提供假按钮。"
        />
      </Space>
    </div>
  );
};

export default BackupManagementPage;
