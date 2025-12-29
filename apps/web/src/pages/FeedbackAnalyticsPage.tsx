import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Alert, Button, Space } from 'antd';

type TrendPoint = { date: string; score?: number; count?: number };
type Distribution = { label: string; value: number };

const FeedbackAnalyticsPage: React.FC = () => {
  const [trends, setTrends] = useState<TrendPoint[]>([]);
  const [distribution, setDistribution] = useState<Distribution[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/feedback/overview'));
      const data = await res.json();
      setTrends(data?.data?.trends || []);
      setDistribution(data?.data?.distribution || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setTrends([]);
      setDistribution([]);
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
        <Card title="反馈趋势">
          <Table
            rowKey="date"
            loading={loading}
            dataSource={trends}
            locale={{ emptyText: '暂无数据，请先产生反馈记录。' }}
            columns={[
              { title: '日期', dataIndex: 'date' },
              { title: '反馈量', dataIndex: 'count' },
              { title: '平均得分', dataIndex: 'score' }
            ]}
          />
        </Card>
        <Card title="反馈分布">
          <Table
            rowKey="label"
            loading={loading}
            dataSource={distribution}
            locale={{ emptyText: '暂无数据。' }}
            columns={[
              { title: '类别', dataIndex: 'label' },
              { title: '数量', dataIndex: 'value' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default FeedbackAnalyticsPage;
