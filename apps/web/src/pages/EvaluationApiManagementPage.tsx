import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Button, Space, Typography, Alert, Spin, Tag, Divider } from 'antd';
import { ApiOutlined, ReloadOutlined, KeyOutlined, FileSearchOutlined } from '@ant-design/icons';

type BenchmarkRow = { name: string; tasks?: string[]; description?: string };
type ApiKeyRow = { id: string; name: string; key: string; permissions?: string[]; expires_at?: string; created_at?: string };
type AuditLogRow = { id?: string; timestamp?: string; resource?: string; user_id?: string; action?: string };

const EvaluationApiManagementPage: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<BenchmarkRow[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeyRow[]>([]);
  const [logs, setLogs] = useState<AuditLogRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [bmRes, keyRes, logRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/model-evaluation/benchmarks'),
        apiFetch(buildApiUrl('/api/v1/security/api-keys'),
        apiFetch(buildApiUrl('/api/v1/security/mcp-tools/audit?limit=100'))
      ]);
      const bmData = await bmRes.json();
      const keyData = await keyRes.json();
      const logData = await logRes.json();
      setBenchmarks(Array.isArray(bmData) ? bmData : bmData?.benchmarks || []);
      setApiKeys(Array.isArray(keyData?.api_keys) ? keyData.api_keys : []);
      setLogs(Array.isArray(logData?.logs) ? logData.logs : []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setBenchmarks([]);
      setApiKeys([]);
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <ApiOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              评估 API 管理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="可用基准测试">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey={(r) => r.name}
              dataSource={benchmarks}
              pagination={{ pageSize: 8 }}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: '任务', render: (_, r) => (r.tasks || []).join(', ') || '-' },
                { title: '描述', dataIndex: 'description' }
              ]}
              locale={{ emptyText: '暂无基准测试，请先在后端注册 BenchmarkManager 数据。' }}
            />
          )}
        </Card>

        <Card title="API Keys">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey={(r) => r.id}
              dataSource={apiKeys}
              pagination={{ pageSize: 5 }}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: 'Key', dataIndex: 'key', render: (v) => (v ? v.slice(0, 8) + '...' : '-') },
                { title: '权限', render: (_, r) => (r.permissions || []).map(p => <Tag key={p}>{p}</Tag>) },
                { title: '到期', dataIndex: 'expires_at' },
                { title: '创建时间', dataIndex: 'created_at' },
              ]}
              locale={{ emptyText: '暂无API Key，使用 /api/v1/security/api-keys 创建后查看。' }}
            />
          )}
        </Card>

        <Card title="审计日志">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey={(r, idx) => r.id || String(idx)}
              dataSource={logs}
              pagination={{ pageSize: 10 }}
              columns={[
                { title: '时间', dataIndex: 'timestamp' },
                { title: '资源', dataIndex: 'resource' },
                { title: '用户', dataIndex: 'user_id' },
                { title: '动作', dataIndex: 'action' },
              ]}
              locale={{ emptyText: '暂无日志，如需数据请触发工具调用。' }}
            />
          )}
        </Card>

        <Divider />
        <Alert
          type="info"
          icon={<FileSearchOutlined />}
          message="提示"
          description="该页面仅展示真实后端返回的数据，不再使用任何静态模拟。"
        />
      </Space>
    </div>
  );
};

export default EvaluationApiManagementPage;
