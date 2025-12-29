import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Button, Alert, Space } from 'antd';

type ConsistencyStat = {
  total_checks: number;
  consistency_rate: number;
  last_check_time?: string;
  avg_check_duration?: number;
};

type ConsistencyCheck = {
  check_id: string;
  checked_at?: string;
  consistent?: boolean;
  components?: string[];
};

type Inconsistency = {
  inconsistency_id: string;
  data_key?: string;
  severity?: string;
  detected_at?: string;
};

const ConsistencyManagementPage: React.FC = () => {
  const [stats, setStats] = useState<ConsistencyStat | null>(null);
  const [history, setHistory] = useState<ConsistencyCheck[]>([]);
  const [inconsistencies, setInconsistencies] = useState<Inconsistency[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sRes, hRes, iRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/consistency/statistics'),
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/consistency/checks'),
        apiFetch(buildApiUrl('/api/v1/fault-tolerance/consistency/inconsistencies'))
      ]);
      setStats(await sRes.json());
      setHistory((await hRes.json())?.checks || []);
      setInconsistencies((await iRes.json())?.items || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setStats(null);
      setHistory([]);
      setInconsistencies([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Button onClick={load} loading={loading}>刷新</Button>
        {error && <Alert type="error" message={error} />}

        <Card title="一致性统计">
          {stats ? (
            <div>
              <div>检查总数: {stats.total_checks}</div>
              <div>一致率: {(stats.consistency_rate * 100).toFixed(2)}%</div>
              <div>最近检查: {stats.last_check_time || '-'}</div>
              <div>平均用时(s): {stats.avg_check_duration ?? '-'}</div>
            </div>
          ) : (
            <div>暂无数据</div>
          )}
        </Card>

        <Card title="检查历史">
          <Table
            rowKey="check_id"
            loading={loading}
            dataSource={history}
            locale={{ emptyText: '暂无记录。' }}
            columns={[
              { title: 'ID', dataIndex: 'check_id' },
              { title: '时间', dataIndex: 'checked_at' },
              { title: '一致', dataIndex: 'consistent', render: (v) => (v ? '是' : '否') },
              { title: '组件', dataIndex: 'components', render: (v) => (v || []).join(', ') }
            ]}
          />
        </Card>

        <Card title="不一致项">
          <Table
            rowKey="inconsistency_id"
            loading={loading}
            dataSource={inconsistencies}
            locale={{ emptyText: '暂无不一致项。' }}
            columns={[
              { title: 'ID', dataIndex: 'inconsistency_id' },
              { title: '数据键', dataIndex: 'data_key' },
              { title: '严重度', dataIndex: 'severity' },
              { title: '发现时间', dataIndex: 'detected_at' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default ConsistencyManagementPage;
