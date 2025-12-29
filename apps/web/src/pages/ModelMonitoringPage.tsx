import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Alert, Button, Space } from 'antd';

type Overview = {
  total_requests?: number;
  avg_latency_ms?: number;
  error_rate?: number;
  active_models?: number;
  active_deployments?: number;
  timestamp?: string;
};

type AlertItem = { id: string; title?: string; message?: string; level?: string; timestamp?: string; status?: string };
type MetricItem = { metric: string; value: number; unit?: string };

const ModelMonitoringPage: React.FC = () => {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [metrics, setMetrics] = useState<MetricItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [oRes, aRes, mRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/model-service/monitoring/overview'),
        apiFetch(buildApiUrl('/api/v1/model-service/monitoring/alerts?active_only=false'),
        apiFetch(buildApiUrl('/api/v1/model-service/monitoring/dashboard'))
      ]);
      setOverview(await oRes.json());
      const alertPayload = await aRes.json();
      setAlerts(Array.isArray(alertPayload?.alerts) ? alertPayload.alerts : []);
      const dash = await mRes.json();
      const metricsArray: MetricItem[] = Object.keys(dash?.metrics || {}).map(key => {
        const arr = dash.metrics[key];
        const latest = Array.isArray(arr) && arr.length > 0 ? arr[arr.length - 1] : null;
        return { metric: key, value: latest?.value ?? 0, unit: latest?.labels?.unit };
      });
      setMetrics(metricsArray);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setOverview(null);
      setAlerts([]);
      setMetrics([]);
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

        <Card title="系统概览">
          {overview ? (
            <div>
              <div>总请求: {overview.total_requests ?? '-'}</div>
              <div>平均延迟(ms): {overview.avg_latency_ms ?? '-'}</div>
              <div>错误率: {overview.error_rate ?? '-'}%</div>
              <div>活跃模型: {overview.active_models ?? '-'}</div>
              <div>活跃部署: {overview.active_deployments ?? '-'}</div>
              <div>时间: {overview.timestamp || '-'}</div>
            </div>
          ) : (
            <div>暂无数据</div>
          )}
        </Card>

        <Card title="告警">
          <Table
            rowKey="id"
            loading={loading}
            dataSource={alerts}
            locale={{ emptyText: '暂无告警' }}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '标题', dataIndex: 'title' },
              { title: '级别', dataIndex: 'level' },
              { title: '状态', dataIndex: 'status' },
              { title: '时间', dataIndex: 'timestamp' }
            ]}
          />
        </Card>

        <Card title="核心指标">
          <Table
            rowKey="metric"
            loading={loading}
            dataSource={metrics}
            locale={{ emptyText: '暂无指标' }}
            columns={[
              { title: '指标', dataIndex: 'metric' },
              { title: '数值', dataIndex: 'value' },
              { title: '单位', dataIndex: 'unit' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default ModelMonitoringPage;
