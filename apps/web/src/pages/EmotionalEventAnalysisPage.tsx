import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Space, Button, Alert, Input, Tag } from 'antd';

type EmotionalEvent = {
  event_id?: string;
  id?: string;
  user_id?: string;
  event_type?: string;
  timestamp?: string;
  impact_score?: number;
  duration_seconds?: number;
  causal_strength?: number;
};

const EmotionalEventAnalysisPage: React.FC = () => {
  const [userId, setUserId] = useState('demo-user');
  const [events, setEvents] = useState<EmotionalEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl(`/api/v1/emotional-memory/events/${userId}`));
      const data = await res.json();
      setEvents(data || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setEvents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space align="center" style={{ marginBottom: 16 }}>
        <Input placeholder="用户ID" value={userId} onChange={(e) => setUserId(e.target.value)} style={{ width: 220 }} />
        <Button type="primary" onClick={load} loading={loading}>查询事件</Button>
      </Space>

      {error && <Alert type="error" message={error} style={{ marginBottom: 16 }} />}

      <Card title="情感事件列表">
        <Table
          rowKey={(r, idx) => r.event_id || r.id || `event-${idx}`}
          loading={loading}
          dataSource={events}
          pagination={{ pageSize: 10 }}
          columns={[
            { title: '事件ID', dataIndex: 'event_id', render: (_, r) => r.event_id || r.id || '-' },
            { title: '类型', dataIndex: 'event_type', render: (v) => v || '-' },
            { title: '时间', dataIndex: 'timestamp', render: (v) => v ? new Date(v).toLocaleString('zh-CN') : '-' },
            { title: '影响', dataIndex: 'impact_score', render: (v) => v ?? '-' },
            { title: '持续(秒)', dataIndex: 'duration_seconds', render: (v) => v ?? '-' },
            { title: '因果强度', dataIndex: 'causal_strength', render: (v) => v ?? '-' },
          ]}
          locale={{ emptyText: '暂无事件，请先调用 /emotional-memory/memories 创建情感数据' }}
        />
      </Card>
    </div>
  );
};

export default EmotionalEventAnalysisPage;
