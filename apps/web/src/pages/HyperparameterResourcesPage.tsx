import React, { useEffect, useState } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert } from '../components/ui/alert';
import { Table } from 'antd';
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type ResourceStats = { resource_usage?: Record<string, number>; current_trials?: number; max_concurrent?: number };
type Allocation = { id: string; status?: string; algorithm?: string; objective?: string };

const API = buildApiUrl('/api/v1/hyperparameter-optimization');

const HyperparameterResourcesPage: React.FC = () => {
  const [resource, setResource] = useState<ResourceStats | null>(null);
  const [allocations, setAllocations] = useState<Allocation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [rRes, aRes] = await Promise.all([
        apiFetch(`${API}/resource-status`),
        apiFetch(`${API}/active-experiments`)
      ]);
      setResource(await rRes.json());
      setAllocations(await aRes.json());
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setResource(null);
      setAllocations([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">资源与分配</h2>
        <Button onClick={load} disabled={loading}>刷新</Button>
      </div>
      {error && <Alert variant="destructive">{error}</Alert>}

      <Card className="p-4">
        <h3 className="font-medium mb-2">资源使用</h3>
        {resource ? (
          <div className="text-sm text-gray-700 space-y-1">
            <div>当前试验: {resource.current_trials ?? '-'} / {resource.max_concurrent ?? '-'}</div>
            <div>资源占用:</div>
            <ul className="list-disc ml-5">
              {Object.entries(resource.resource_usage || {}).map(([k, v]) => (
                <li key={k}>{k}: {v}%</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="text-sm text-gray-500">暂无数据</div>
        )}
      </Card>

      <Card className="p-4">
        <h3 className="font-medium mb-2">活跃实验</h3>
        <Table
          rowKey={(r) => r.id || `${r.algorithm}-${r.objective}`}
          loading={loading}
          dataSource={allocations}
          locale={{ emptyText: '暂无活跃实验' }}
          columns={[
            { title: 'ID', dataIndex: 'id' },
            { title: '状态', dataIndex: 'status' },
            { title: '算法', dataIndex: 'algorithm' },
            { title: '目标', dataIndex: 'objective' }
          ]}
        />
      </Card>
    </div>
  );
};

export default HyperparameterResourcesPage;
