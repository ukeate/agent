import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react';
import { Card, Input, Button, Table, Alert, Space } from 'antd';

type Entity = { id: string; label?: string; type?: string; score?: number };

const KnowledgeGraphEntityManagement: React.FC = () => {
  const [keyword, setKeyword] = useState('');
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-graph/entities/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: keyword })
      });
      const data = await res.json();
      setEntities(data?.entities || []);
    } catch (e: any) {
      setError(e?.message || '查询失败');
      setEntities([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card title="实体检索">
          <Space>
            <Input
              placeholder="输入关键词"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              style={{ width: 320 }}
            />
            <Button type="primary" onClick={search} loading={loading} disabled={!keyword.trim()}>
              搜索
            </Button>
          </Space>
        </Card>

        {error && <Alert type="error" message={error} />}

        <Card title="结果">
          <Table
            rowKey="id"
            loading={loading}
            dataSource={entities}
            locale={{ emptyText: '暂无数据，请先在后端接入实体检索数据源' }}
            columns={[
              { title: 'ID', dataIndex: 'id' },
              { title: '标签', dataIndex: 'label' },
              { title: '类型', dataIndex: 'type' },
              { title: '得分', dataIndex: 'score' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default KnowledgeGraphEntityManagement;
