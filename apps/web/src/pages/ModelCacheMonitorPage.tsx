import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Alert, Button, Space } from 'antd';

type ModelInfo = {
  model_id: string;
  name?: string;
  version?: string;
  format?: string;
  framework?: string;
  model_size_mb?: number;
  parameter_count?: number;
  created_at?: string;
  updated_at?: string;
};

const ModelCacheMonitorPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/model-service/models'));
      const data = await res.json();
      setModels(data?.models || data || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const totalSize = models.reduce((acc, m) => acc + (m.model_size_mb || 0), 0);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Button onClick={load} loading={loading}>刷新</Button>
        {error && <Alert type="error" message={error} />}
        <Card title="模型缓存 / 列表">
          <div style={{ marginBottom: 8 }}>模型数量: {models.length}，总大小: {totalSize.toFixed(2)} MB</div>
          <Table
            rowKey="model_id"
            loading={loading}
            dataSource={models}
            locale={{ emptyText: '暂无模型，请先通过后端注册模型。' }}
            columns={[
              { title: 'ID', dataIndex: 'model_id' },
              { title: '名称', dataIndex: 'name' },
              { title: '版本', dataIndex: 'version' },
              { title: '格式', dataIndex: 'format' },
              { title: '框架', dataIndex: 'framework' },
              { title: '大小(MB)', dataIndex: 'model_size_mb' },
              { title: '参数量', dataIndex: 'parameter_count' },
              { title: '创建时间', dataIndex: 'created_at' },
              { title: '更新时间', dataIndex: 'updated_at' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default ModelCacheMonitorPage;
