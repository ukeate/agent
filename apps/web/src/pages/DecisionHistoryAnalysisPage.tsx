import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Space, Typography, Button, Alert, Spin, Statistic } from 'antd';
import { ReloadOutlined, HistoryOutlined } from '@ant-design/icons';

type DecisionRow = {
  decision_id: string;
  user_id: string;
  timestamp: string;
  chosen_strategy: string;
  confidence_score?: number;
  execution_status?: string;
};

const DecisionHistoryAnalysisPage: React.FC = () => {
  const [rows, setRows] = useState<DecisionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/emotional-intelligence/decisions/history'));
      const data = await res.json();
      setRows(Array.isArray(data?.decisions) ? data.decisions : []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setRows([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const total = rows.length;
  const success = rows.filter(r => r.execution_status === 'completed').length;
  const pending = rows.filter(r => r.execution_status === 'pending').length;
  const avgConfidence =
    rows.reduce((a, b) => a + (b.confidence_score || 0), 0) / (rows.length || 1);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <HistoryOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              决策历史分析
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card>
          <Space size="large">
            <Statistic title="总决策数" value={total} />
            <Statistic title="完成" value={success} />
            <Statistic title="待处理" value={pending} />
            <Statistic
              title="平均置信度"
              value={avgConfidence.toFixed(2)}
              suffix=""
            />
          </Space>
        </Card>

        <Card title="决策列表">
          {loading ? (
            <Spin />
          ) : rows.length === 0 ? (
            <Alert type="info" message="暂无决策记录，先调用 /api/v1/emotional-intelligence/decide 生成真实数据。" />
          ) : (
            <Table
              rowKey="decision_id"
              dataSource={rows}
              pagination={{ pageSize: 10 }}
              columns={[
                { title: 'ID', dataIndex: 'decision_id' },
                { title: '用户', dataIndex: 'user_id' },
                { title: '时间', dataIndex: 'timestamp' },
                { title: '策略', dataIndex: 'chosen_strategy' },
                {
                  title: '置信度',
                  dataIndex: 'confidence_score',
                  render: (v: number) => (v !== undefined ? v.toFixed(2) : '-')
                },
                { title: '状态', dataIndex: 'execution_status' }
              ]}
            />
          )}
        </Card>
      </Space>
    </div>
  );
};

export default DecisionHistoryAnalysisPage;
