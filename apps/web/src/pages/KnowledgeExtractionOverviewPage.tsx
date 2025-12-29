import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react';
import { Card, Table, Alert, Button, Space } from 'antd';

type Job = {
  job_id: string;
  status?: string;
  created_at?: string;
  progress?: number;
  source?: string;
};

const KnowledgeExtractionOverviewPage: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-extraction/jobs'));
      const data = await res.json();
      setJobs(data?.jobs || []);
    } catch (e: any) {
      setError(e?.message || '加载失败');
      setJobs([]);
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
        <Card title="抽取任务">
          <Table
            rowKey="job_id"
            loading={loading}
            dataSource={jobs}
            locale={{ emptyText: '暂无任务，请先通过后端接口创建知识抽取任务。' }}
            columns={[
              { title: '任务ID', dataIndex: 'job_id' },
              { title: '状态', dataIndex: 'status' },
              { title: '来源', dataIndex: 'source' },
              { title: '进度', dataIndex: 'progress' },
              { title: '创建时间', dataIndex: 'created_at' }
            ]}
          />
        </Card>
      </Space>
    </div>
  );
};

export default KnowledgeExtractionOverviewPage;
