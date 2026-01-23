import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Alert, Button, Space } from 'antd';

type Metric = { name: string; value: number; unit?: string; timestamp?: string };
type ServiceMetric = { service: string; latency_ms?: number; throughput?: number; error_rate?: number };

const ServicePerformanceDashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [services, setServices] = useState<ServiceMetric[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [overallRes, svcRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/realtime-metrics/overview')),
        apiFetch(buildApiUrl('/api/v1/realtime-metrics/services'))
      ]);
      setMetrics((await overallRes.json())?.metrics || []);
      setServices((await svcRes.json())?.services || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setMetrics([]);
      setServices([]);
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

        <Card title="整体指标">
          <Table
            rowKey="name"
            loading={loading}
            dataSource={metrics}
            locale={{ emptyText: '暂无指标' }}
            columns={[
              { title: '指标', dataIndex: 'name' },
              { title: '数值', dataIndex: 'value' },
              { title: '单位', dataIndex: 'unit' },
              { title: '时间', dataIndex: 'timestamp' }
            ]}
          />
        </Card>

        <Card title="服务级指标">
          <Table
            rowKey="service"
            loading={loading}
            dataSource={services}
            locale={{ emptyText: '暂无服务数据' }}
            columns={[
              { title: '服务', dataIndex: 'service' },
              { title: '延迟(ms)', dataIndex: 'latency_ms' },
              { title: '吞吐', dataIndex: 'throughput' },
              { title: '错误率', dataIndex: 'error_rate' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default ServicePerformanceDashboardPage;
